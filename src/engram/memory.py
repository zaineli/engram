"""EngramMemory: the assembled system.

Write path
----------
``remember`` encodes an experience once and writes it to three places at once:
a dense semantic vector, a k-sparse dentate code stored as a CA3 attractor, and
a theta/position code binding it to its slot in the stream. CA1 scores its
novelty against what is already held, and that score becomes the trace's replay
priority.

Read path
---------
``recall`` runs four pathways in parallel and fuses them. They are kept separate
rather than collapsed into one index because they fail in different, useful
ways, and because keeping them separable is what makes the ablation in
``bench/`` meaningful:

``semantic``
    Cosine over dense embeddings. Strong on paraphrase, and the pathway that
    flat vector stores consist entirely of. Fails on confusable near-duplicates.

``conjunctive``
    Weighted conjunction-code intersection (:mod:`engram.binding`). Resolves
    the case where the semantic pathway collapses: episodes sharing a frame and
    differing only in which value filled which slot. The dentate/CA3 pair is
    *not* used for ranking — it was measured at 0.048 top-1 on the
    incident-report set, no better than cosine, because expansion cannot
    separate what the input never distinguished. Its job is storage-side
    separation and fragment completion, which is what it is good at.

``temporal``
    Theta-context similarity. Answers "what happened around then" and recovers
    order, which neither of the content pathways can do. Opt-in: it needs the
    caller to say which session and position it is asking about.

``schema``
    Neocortical gist. The only pathway that can answer a question whose answer
    was never in any single episode.

On top of the four, a query that names an episode and asks for its *adjacent*
one ("what did she do right after...") triggers **temporal context
reinstatement**: the anchor episode is retrieved on content, its encoding
context is reinstated, and that context cues the adjacent slots while the
anchor itself is suppressed. A looser temporal word ("how many days before the
move did I...") does not trigger it. Those questions relate events days apart
and usually need the anchor as evidence too; see ``_ADJACENT`` for the measured
reason.

Sleep
-----
``sleep`` runs ripple bursts. This is not an optimisation — it is where
generalisation comes from. Nothing in the read path creates schemas.

Eviction is separate from consolidation and happens only under a configured
capacity bound; see :class:`~engram.replay.ReplayScheduler` for why coupling the
two destroyed exactly the information the architecture exists to protect.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace

import numpy as np

from .binding import ConjunctionIndex, ConjunctiveBinder
from .ca1 import CA1
from .ca3 import CA3
from .dentate import DentateGyrus
from .encoding import Encoder, HashingEncoder, ThetaContext
from .neocortex import Neocortex
from .replay import ReplayScheduler
from .types import Episode, Recall, Trace

__all__ = ["EngramMemory", "EngramConfig", "FUSIONS"]

#: Adjacency cues: the query names one episode and asks for its neighbour.
#: "What did Priya do right after she signed off on the migration plan" names
#: an anchor and asks for a different episode, so content matching alone
#: retrieves the anchor, which is the one answer guaranteed to be wrong.
#:
#: v0.1 fired on any of ``after|following|next|then|before|prior to|...``. On
#: LongMemEval-S those words appear in 28 of the 470 answerable questions, and
#: nearly all of them are of the form "how many days before X did I do Y",
#: where X is part of the evidence. Suppressing the anchor there removes a gold
#: item. The cue list is now restricted to explicit adjacency.
_ADJ_AFTER = re.compile(
    r"\b(?:right|just|immediately|straight)\s+(?:after|following)\b"
    r"|\bwhat\s+(?:came|happened)\s+next\b",
    re.I,
)
_ADJ_BEFORE = re.compile(
    r"\b(?:right|just|immediately|straight)\s+before\b"
    r"|\bwhat\s+(?:came|happened)\s+(?:just\s+)?before\b",
    re.I,
)

#: Generalisation cues. These route the query to the slow store. "Which project
#: does Priya *usually* work on" is not a question about an episode — no episode
#: contains the answer — so ranking schemas against episodes by score is the
#: wrong operation. The two stores answer different questions, and the query
#: says which one it is asking. This is the complementary-learning-systems
#: division of labour made explicit at the read path.
_GENERIC = re.compile(
    r"\b(usually|typically|generally|normally|habitually|most often|most of the time|"
    r"tend(?:s)? to|in general|on average|for the most part|mostly)\b",
    re.I,
)

#: Per-pathway score normalisations. See :func:`_normalise`.
FUSIONS = ("minmax", "zscore", "rrf")


def _normalise(v: np.ndarray, how: str, rrf_k: float = 60.0) -> np.ndarray:
    """Put one pathway's scores on a common scale before the weighted sum.

    ``minmax`` maps the candidate range to [0, 1], so a single outlier sets the
    scale for everything else. ``zscore`` centres and divides by the standard
    deviation over the candidates, the combiner that lexical-dense fusion work
    on conversational memory has found more robust than min-max. ``rrf`` throws
    the scores away and keeps the rank, ``1 / (rrf_k + rank)``: parameter-free
    and immune to scale, at the price of ignoring how far apart two candidates
    were. A pathway that is constant over the candidates carries no
    information and contributes zero under all three.
    """
    if v.size == 0:
        return v
    if how == "rrf":
        if float(v.max() - v.min()) < 1e-9:
            return np.zeros_like(v)
        ranks = np.empty(v.size, dtype=np.float64)
        ranks[np.argsort(-v, kind="stable")] = np.arange(1, v.size + 1)
        return 1.0 / (rrf_k + ranks)
    if how == "zscore":
        sd = float(v.std())
        return np.zeros_like(v) if sd < 1e-9 else (v - float(v.mean())) / sd
    if how == "minmax":
        lo, hi = float(v.min()), float(v.max())
        return np.zeros_like(v) if hi - lo < 1e-6 else (v - lo) / (hi - lo)
    raise ValueError(f"unknown fusion {how!r}; expected one of {FUSIONS}")


@dataclass
class EngramConfig:
    """Everything tunable, in one place so the benchmark can sweep it.

    The four settings marked *frozen* were chosen on the LongMemEval-S
    dev split (``bench/longmemeval.py tune``: 294 configurations, objective the
    mean of the four official headline metrics) and then fixed before the test
    split or the synthetic benchmark was scored with them. v0.1's values were
    ``conj_window=None`` (all pairs), ``w_conjunctive=1.2`` and a 2**16 code.
    """

    expansion: int = 8
    sparsity: float = 0.02
    neurogenesis: float = 0.7
    beta: float = 22.0
    context_dim: int = 32

    conj_width: int = 1 << 32
    conj_order: int = 2
    #: Pair window in content terms (SDM's ``#uwN``). None = every pair.
    conj_window: int | None = 4  # frozen
    #: "idf" weights each conjunction unit by its own document frequency in a
    #: cosine; "binary" counts shared units, as v0.1 did; "bm25" scores units
    #: with BM25's saturation and length normalisation. See ConjunctionIndex.
    conj_weighting: str = "idf"  # frozen

    schema_merge_threshold: float = 0.72
    replay_batch: int = 32
    sequence_span: int = 3
    interleave: float = 0.3
    promote_after: int = 3
    #: Resident hippocampal traces. None = unbounded (better recall, unbounded
    #: memory). Set it to bound the fast store; eviction prefers consolidated,
    #: weak, never-retrieved traces and will never drop an unconsolidated one.
    hippocampal_capacity: int | None = None

    decay_rate: float = 0.9995
    decay_floor: float = 0.05

    #: How pathway scores are normalised before fusion: one of ``FUSIONS``.
    fusion: str = "minmax"  # frozen
    rrf_k: float = 60.0

    # Pathway fusion weights. Set any to 0 to ablate that pathway.
    w_semantic: float = 1.0
    w_conjunctive: float = 0.9  # frozen
    w_temporal: float = 0.25
    w_schema: float = 0.6

    #: Temporal context reinstatement (Howard & Kahana's temporal context
    #: model): recalling an item reinstates the context it was encoded in,
    #: which then cues its neighbours. Set to 0 to ablate.
    w_reinstate: float = 2.0
    #: How many slots forward/back a reinstated context reaches.
    reinstate_span: int = 3
    #: Multiplier applied to ``w_schema`` when the query carries a
    #: generalisation cue. Above 1/``w_schema`` so the slow store can actually
    #: win those queries rather than merely place well.
    schema_generic_boost: float = 2.4

    seed: int = 0


class EngramMemory:
    """Hippocampal-neocortical memory for long-horizon agents."""

    def __init__(self, encoder: Encoder | None = None, config: EngramConfig | None = None) -> None:
        self.cfg = config or EngramConfig()
        if self.cfg.fusion not in FUSIONS:
            raise ValueError(f"unknown fusion {self.cfg.fusion!r}; expected one of {FUSIONS}")
        self.encoder = encoder or HashingEncoder(dim=384)
        d = self.encoder.dim

        self.dg = DentateGyrus(
            d,
            expansion=self.cfg.expansion,
            sparsity=self.cfg.sparsity,
            neurogenesis=self.cfg.neurogenesis,
            inhibition=0.0,
            seed=self.cfg.seed,
        )
        self.ca3 = CA3(self.dg.dim_out, beta=self.cfg.beta)
        self.ca1 = CA1()
        self.neocortex = Neocortex(d, merge_threshold=self.cfg.schema_merge_threshold)
        self.binder = ConjunctiveBinder(
            width=self.cfg.conj_width, order=self.cfg.conj_order, window=self.cfg.conj_window
        )
        self.conj_index = ConjunctionIndex(weighting=self.cfg.conj_weighting)
        self.theta = ThetaContext(dim=self.cfg.context_dim)
        self.replay = ReplayScheduler(
            batch=self.cfg.replay_batch,
            sequence_span=self.cfg.sequence_span,
            interleave=self.cfg.interleave,
            promote_after=self.cfg.promote_after,
            capacity=self.cfg.hippocampal_capacity,
            seed=self.cfg.seed,
        )
        self._n_written = 0
        self._session_counts: dict[int, int] = {}
        self._matrix: tuple[list[int], np.ndarray, np.ndarray] | None = None

    # ---------------------------------------------------------------- write

    def remember(
        self,
        text: str,
        session: int = 0,
        salience: float = 0.0,
        timestamp: float | None = None,
        meta: dict | None = None,
    ) -> Trace:
        """Encode and store one experience."""
        dense = self.encoder.encode([text])[0]
        return self._write(text, dense, session, salience, timestamp, meta)

    def remember_many(self, texts: list[str], session: int = 0, **kw) -> list[Trace]:
        """Store several experiences in order, embedding them in one batch."""
        if not texts:
            return []
        dense = self.encoder.encode(list(texts))
        return [self._write(t, v, session, **kw) for t, v in zip(texts, dense)]

    def _write(
        self,
        text: str,
        dense: np.ndarray,
        session: int,
        salience: float = 0.0,
        timestamp: float | None = None,
        meta: dict | None = None,
    ) -> Trace:
        pos = self._session_counts.get(session, 0)
        self._session_counts[session] = pos + 1
        ep = Episode(
            text=text,
            session=session,
            index_in_session=pos,
            timestamp=timestamp if timestamp is not None else time.time(),
            meta=meta or {},
            salience=salience,
        )
        surprise = self.ca1.novelty(dense)  # computed before the write
        code = self.dg.encode(dense, learn=True)
        conj, conj_tf, conj_len = self.binder.encode_counts(text)
        ctx = self.theta.encode(session, pos)

        tr = Trace(
            episode=ep,
            dense=dense,
            sparse_idx=code,
            conj_idx=conj,
            context=ctx,
            surprise=surprise,
        )
        slot = self.ca3.store(code)
        self.ca1.bind(slot, tr)
        self.conj_index.add(slot, conj, conj_tf, conj_len)
        self._matrix = None
        self._n_written += 1

        for dead in self.ca1.decay(self.cfg.decay_rate, self.cfg.decay_floor):
            self.ca1.release(dead)
        self._reconcile()
        return tr

    def _reconcile(self) -> None:
        """Drop every per-slot structure whose trace CA1 no longer holds.

        CA1 is the source of truth for residency. Decay and capacity eviction
        both release from CA1 directly; v0.1 removed the CA3 pattern after
        decay but not after eviction, so an evicted trace's attractor stayed
        in CA3 and could still capture a completion.
        """
        live = set(self.ca1.slots())
        stale = [s for s in self.conj_index.keys() if s not in live]
        for s in stale:
            self.conj_index.remove(s)
            self.ca3.remove(s)
        if stale:
            self._matrix = None

    #: Config fields that only the read path consults. Changing them on a
    #: populated memory is exactly equivalent to rebuilding it with them.
    READ_TIME = frozenset({
        "fusion", "rrf_k", "w_semantic", "w_conjunctive", "w_temporal", "w_schema",
        "w_reinstate", "reinstate_span", "schema_generic_boost", "conj_weighting",
    })

    def reconfigure(self, **read_time) -> None:
        """Change read-time settings without re-ingesting (used by parameter sweeps)."""
        bad = set(read_time) - self.READ_TIME
        if bad:
            raise ValueError(f"not read-time settings, would need a rebuild: {sorted(bad)}")
        if read_time.get("fusion", self.cfg.fusion) not in FUSIONS:
            raise ValueError(f"unknown fusion {read_time['fusion']!r}")
        self.cfg = replace(self.cfg, **read_time)
        self.conj_index.set_weighting(self.cfg.conj_weighting)

    # ----------------------------------------------------------------- read

    def _stacked(self) -> tuple[list[int], np.ndarray, np.ndarray]:
        if self._matrix is None:
            slots = self.ca1.slots()
            traces = [self.ca1.get(s) for s in slots]
            d = self.encoder.dim
            D = np.stack([t.dense for t in traces]) if traces else np.zeros((0, d), np.float32)
            C = (
                np.stack([t.context for t in traces])
                if traces
                else np.zeros((0, self.cfg.context_dim), np.float32)
            )
            self._matrix = (slots, D, C)
        return self._matrix

    def _encode_query(self, query: str) -> np.ndarray:
        enc = getattr(self.encoder, "encode_query", None) or self.encoder.encode
        return enc([query])[0]

    def recall(
        self,
        query: str,
        k: int | None = 5,
        session: int | None = None,
        position: int | None = None,
        include_schemas: bool = True,
        rehearse: bool = True,
        dedup: bool = True,
    ) -> list[Recall]:
        """Retrieve up to ``k`` items, fusing all four pathways.

        ``k=None`` returns the full ranking. ``rehearse=False`` makes the call
        read-only: by default a recalled trace is strengthened, which is the
        intended behaviour for an agent and the wrong one for an evaluation,
        where it makes each query's score depend on the queries before it.
        ``dedup`` collapses items with identical text into their best copy.
        """
        cfg = self.cfg
        q = self._encode_query(query)
        slots, D, C = self._stacked()
        n = len(slots)
        out: list[Recall] = []

        if n:
            raw: dict[str, np.ndarray] = {"semantic": D @ q}
            if cfg.w_conjunctive > 0:
                keys, s = self.conj_index.score(self.binder.encode(query))
                pos_of = {key: i for i, key in enumerate(keys)}
                raw["conjunctive"] = np.array([s[pos_of[sl]] for sl in slots], dtype=np.float64)
            if cfg.w_temporal > 0 and session is not None:
                ctx = self.theta.encode(session, position if position is not None else 0)
                raw["temporal"] = C @ ctx

            # ---- normalise each pathway before fusing ----------------------
            # The pathways live on incompatible scales: dense cosine spans
            # roughly 0.45-0.55 across candidates while conjunction overlap
            # spans 0.0-0.3. Summing them raw lets whichever happens to have
            # the larger absolute range dominate regardless of how
            # discriminative it is, which is why an earlier revision could
            # delete the entire schema pathway and move no number at all.
            weights = {"semantic": cfg.w_semantic, "conjunctive": cfg.w_conjunctive,
                       "temporal": cfg.w_temporal}
            normed = {p: _normalise(v.astype(np.float64), cfg.fusion, cfg.rrf_k)
                      for p, v in raw.items()}
            total = np.zeros(n, dtype=np.float64)
            for p, v in normed.items():
                total += weights[p] * v
            if cfg.fusion == "zscore":
                # z-scores are signed; shift so the worst candidate sits at 0.
                # Ranking is unchanged, and the multiplicative terms below
                # (decay, anchor suppression) keep their meaning.
                total -= float(total.min())

            for i, s in enumerate(slots):
                tr = self.ca1.get(s)
                ev = {p: float(raw[p][i]) for p in raw}
                ev.update({f"{p}_n": float(normed[p][i]) for p in normed})
                # A trace that has decayed is less retrievable, as in the biology.
                score = float(total[i]) * (0.85 + 0.15 * tr.strength)
                out.append(Recall(episode=tr.episode, score=score, evidence=ev))

        # The schema pathway is scaled to the best episodic content score, taken
        # before any reinstatement bonus, as in v0.1.
        hmax = max((r.score for r in out), default=1.0) or 1.0

        # ---- temporal context reinstatement -------------------------------
        # An adjacency cue means the query names an *anchor* and asks for its
        # neighbour. Retrieve the anchor on content, reinstate its encoding
        # context, and let that context cue the slots adjacent to it. The
        # anchor is chosen among episodes only: v0.1 could pick a schema hit,
        # whose placeholder key (0, 0) then "reinstated" the first episode of
        # session 0.
        direction = 1 if _ADJ_AFTER.search(query) else (-1 if _ADJ_BEFORE.search(query) else 0)
        if direction and cfg.w_reinstate > 0 and out:
            anchor = max(out, key=lambda r: r.score)
            a_sess, a_pos = anchor.episode.key()
            a_score = anchor.score
            for r in out:
                sess, pos = r.episode.key()
                if sess != a_sess:
                    continue
                offset = (pos - a_pos) * direction
                if 1 <= offset <= cfg.reinstate_span:
                    # Nearest neighbour gets the full bonus, decaying with distance.
                    r.score += cfg.w_reinstate * a_score / offset
                    r.evidence["reinstated"] = float(offset)
                elif offset == 0:
                    # Suppress the anchor: it is what was asked *about*, not for.
                    r.score *= 0.15
                    r.evidence["anchor"] = 1.0

        if include_schemas and cfg.w_schema > 0 and len(self.neocortex):
            # Dedup happens *after* ranking, not here. Filtering schema
            # exemplars against every hippocampal candidate deleted the entire
            # pathway: a schema's exemplar is by construction one of the stored
            # episodes, so the filter matched everything and the slow store
            # silently contributed nothing.
            sch = list(self.neocortex.query(q, top_k=max(4, k or 4)))
            if sch:
                # Express schema scores in hippocampal units so the weight is a
                # real routing decision rather than an arbitrary constant that
                # happens to sit below every episodic score.
                generic = bool(_GENERIC.search(query))
                w = cfg.w_schema * (cfg.schema_generic_boost if generic else 1.0)
                vals = np.array([r.score for r in sch], dtype=np.float32)
                lo, hi = float(vals.min()), float(vals.max())
                rng = hi - lo
                for r in sch:
                    norm = (r.score - lo) / rng if rng > 1e-6 else 1.0
                    r.score = w * hmax * float(norm)
                    r.evidence["routed_generic"] = float(generic)
                    out.append(r)

        out.sort(key=lambda r: r.score, reverse=True)

        if dedup:
            # A memory that surfaced through both stores is one memory.
            deduped: list[Recall] = []
            seen_text: set[str] = set()
            for r in out:
                if r.episode.text in seen_text:
                    continue
                seen_text.add(r.episode.text)
                deduped.append(r)
            out = deduped
        top = out if k is None else out[:k]

        if rehearse:
            # Retrieval is itself a rehearsal event: recalling strengthens. A
            # schema hit carries a placeholder key and rehearses nothing.
            for r in top:
                if r.source != "hippocampus":
                    continue
                slot = self.ca1.slot_of(r.episode.key())
                if slot is not None and (tr := self.ca1.get(slot)):
                    tr.strength = min(1.0, tr.strength + 0.05)
        return top

    def complete_partial(self, key: tuple[int, int], keep: float = 0.15) -> Recall | None:
        """Recover a full episode from a fragment of its *code*.

        This is CA3's actual competence, and the API is shaped to say so. The
        attractor network completes degraded **codes** — it is not a text
        retriever. Handing it a paraphrase does not work and the reason is the
        same one that rules the dentate pathway out of ranking: a paraphrase
        produces a different embedding, hence a different sparse code, and a
        different code is a different address, not a corrupted version of the
        same one. An earlier revision exposed ``complete(text)`` and it settled
        confidently onto unrelated attractors, which is worse than failing.

        What CA3 does do, measured in ``bench/ca3_capacity.py``: recover the
        right episode from 90% deletion of its active units (six of 61 kept)
        at 0.993 accuracy with 400 stored patterns and 0.907 with 4,000. It
        degrades slowly with store size and sharply once fewer than about
        three units survive. v0.1 described this as flat at 100%, which the
        committed script never showed. For text queries use :meth:`recall`.

        ``keep`` is the fraction of the original code's active units retained.
        """
        slot = self.ca1.slot_of(key)
        if slot is None:
            return None
        tr = self.ca1.get(slot)
        if tr is None or tr.sparse_idx.size == 0:
            return None

        n_keep = max(1, int(round(tr.sparse_idx.size * keep)))
        rng = np.random.default_rng(abs(hash(key)) % (2**32))
        partial = np.sort(rng.choice(tr.sparse_idx, size=n_keep, replace=False)).astype(np.int32)

        res = self.ca3.complete(partial, k=tr.sparse_idx.size)
        if res.index < 0:
            return None
        got = self.ca1.get(res.index)
        if got is None:
            return None
        return Recall(
            episode=got.episode,
            score=res.confidence,
            evidence={
                "confidence": res.confidence,
                "converged": float(res.converged),
                "units_kept": float(n_keep),
                "units_total": float(tr.sparse_idx.size),
                "correct": float(got.key == key),
            },
            completion_steps=res.steps,
        )

    # ---------------------------------------------------------------- sleep

    def sleep(self, cycles: int = 10, now: float | None = None) -> list:
        """Run ``cycles`` ripple bursts. Where generalisation happens.

        Replay priority has a recency term, measured from ``now`` (default: the
        wall clock). Pass ``now`` explicitly, with explicit episode timestamps,
        for a run that must be reproducible: v0.1 always read the clock, so two
        identical benchmark runs could consolidate different schemas and
        report different abstraction scores.
        """
        now = time.time() if now is None else now
        events = []
        for _ in range(cycles):
            events.append(self.replay.ripple(self.ca1, self.neocortex, now))
            self._reconcile()
        return events

    # ---------------------------------------------------------------- misc

    def stats(self) -> dict[str, float]:
        s = {
            "written": float(self._n_written),
            "hippocampal": float(len(self.ca1)),
            "ca3_patterns": float(self.ca3.resident),
            "ripples": float(self.replay.step),
        }
        s.update({f"dg_{k}": v for k, v in self.dg.stats().items()})
        s.update({f"nc_{k}": v for k, v in self.neocortex.stats().items()})
        # How much of the written stream the fast store is still holding.
        s["retention"] = s["hippocampal"] / max(s["written"], 1.0)
        return s
