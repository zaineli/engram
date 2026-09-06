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
    Conjunction-code intersection (:mod:`engram.binding`). Resolves the case
    where the semantic pathway collapses: episodes sharing a frame and
    differing only in which value filled which slot. On the incident-report set
    this pathway takes top-1 from 0.040 to 0.200, the ceiling for that query
    set. The dentate/CA3 pair is *not* used for ranking — it was measured at
    0.056 there, no better than cosine, because expansion cannot separate what
    the input never distinguished. Its job is storage-side separation and
    fragment completion, which is what it is good at.

``temporal``
    Theta-context similarity. Answers "what happened around then" and recovers
    order, which neither of the content pathways can do.

``schema``
    Neocortical gist. The only pathway that can answer a question whose answer
    was never in any single episode.

On top of the four, a directional query ("what happened right after...") also
triggers **temporal context reinstatement**: the anchor episode is retrieved on
content, its encoding context is reinstated, and that context cues the adjacent
slots while the anchor itself is suppressed. This is the one operation that
makes sequence questions answerable at all — every pure-retrieval baseline
returns the anchor and scores zero.

Sleep
-----
``sleep`` runs ripple bursts. This is not an optimisation — it is where
generalisation comes from. Nothing in the read path creates schemas.

Eviction is separate from consolidation and happens only under a configured
capacity bound; see :class:`~engram.replay.ReplayScheduler` for why coupling the
two destroyed exactly the information the architecture exists to protect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .ca1 import CA1
from .ca3 import CA3
from .dentate import DentateGyrus
import re

from .binding import ConjunctiveBinder
from .encoding import Encoder, HashingEncoder, ThetaContext
from .neocortex import Neocortex
from .replay import ReplayScheduler
from .types import Episode, Recall, Trace

__all__ = ["EngramMemory", "EngramConfig"]

#: Relational cues that make a query a question about *sequence* rather than
#: content. "What happened right after the migration review" names one episode
#: and asks for a different one, so content matching alone retrieves the anchor
#: and scores it as a miss — which is exactly what every baseline does on the
#: temporal task, all of them landing on 0.242.
_AFTER = re.compile(r"\b(right after|just after|after|following|next|then|subsequently|later that)\b", re.I)
_BEFORE = re.compile(r"\b(right before|just before|before|preceding|prior to|leading up to)\b", re.I)

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


@dataclass
class EngramConfig:
    """Everything tunable, in one place so the benchmark can sweep it."""

    expansion: int = 8
    sparsity: float = 0.02
    neurogenesis: float = 0.7
    beta: float = 22.0
    context_dim: int = 32

    conj_width: int = 1 << 16
    conj_order: int = 2

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

    # Pathway fusion weights. Set any to 0 to ablate that pathway.
    w_semantic: float = 1.0
    w_conjunctive: float = 1.2
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
        self.binder = ConjunctiveBinder(width=self.cfg.conj_width, order=self.cfg.conj_order)
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
        dense = self.encoder.encode([text])[0]
        surprise = self.ca1.novelty(dense)  # computed before the write
        code = self.dg.encode(dense, learn=True)
        conj = self.binder.encode(text)
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
        self._n_written += 1

        for dead in self.ca1.decay(self.cfg.decay_rate, self.cfg.decay_floor):
            self.ca1.release(dead)
            self.ca3.remove(dead)
        return tr

    def remember_many(self, texts: list[str], session: int = 0, **kw) -> list[Trace]:
        return [self.remember(t, session=session, **kw) for t in texts]

    # ----------------------------------------------------------------- read

    def recall(
        self,
        query: str,
        k: int = 5,
        session: int | None = None,
        position: int | None = None,
        include_schemas: bool = True,
    ) -> list[Recall]:
        """Retrieve up to ``k`` items, fusing all four pathways."""
        cfg = self.cfg
        q = self.encoder.encode([query])[0]
        slots = self.ca1.slots()

        scored: dict[int, dict[str, float]] = {}

        if slots:
            traces = [self.ca1.get(s) for s in slots]
            D = np.stack([t.dense for t in traces])
            sem = D @ q
            for s, v in zip(slots, sem):
                scored.setdefault(s, {})["semantic"] = float(v)

            if cfg.w_conjunctive > 0:
                qc = self.binder.encode(query)
                for s, t in zip(slots, traces):
                    scored.setdefault(s, {})["conjunctive"] = self.binder.match(qc, t.conj_idx)

            if cfg.w_temporal > 0 and session is not None:
                ctx = self.theta.encode(session, position if position is not None else 0)
                C = np.stack([t.context for t in traces])
                tsim = C @ ctx
                for s, v in zip(slots, tsim):
                    scored.setdefault(s, {})["temporal"] = float(v)

        # ---- normalise each pathway before fusing --------------------------
        # The pathways live on incompatible scales: dense cosine spans roughly
        # 0.45-0.55 across candidates while conjunction overlap spans 0.0-0.3.
        # Summing them raw lets whichever happens to have the larger absolute
        # range dominate regardless of how discriminative it is, which is why an
        # earlier revision could delete the entire schema pathway and move no
        # number at all — it was scoring 0.30 against a hippocampal 0.57 and
        # could never rank first. Min-max per pathway makes the weights mean
        # what they say: relative contribution to the ranking.
        slot_ids = list(scored.keys())
        for pathway in ("semantic", "conjunctive", "temporal"):
            vals = np.array([scored[s].get(pathway, 0.0) for s in slot_ids], dtype=np.float32)
            if vals.size == 0:
                continue
            lo, hi = float(vals.min()), float(vals.max())
            rng = hi - lo
            if rng < 1e-6:
                # Uniform pathway carries no information; contribute nothing
                # rather than a constant that shifts every candidate equally.
                for s in slot_ids:
                    scored[s][f"{pathway}_n"] = 0.0
            else:
                for s, v in zip(slot_ids, vals):
                    scored[s][f"{pathway}_n"] = float((v - lo) / rng)

        out: list[Recall] = []
        for s, ev in scored.items():
            tr = self.ca1.get(s)
            if tr is None:
                continue
            total = (
                cfg.w_semantic * ev.get("semantic_n", 0.0)
                + cfg.w_conjunctive * ev.get("conjunctive_n", 0.0)
                + cfg.w_temporal * ev.get("temporal_n", 0.0)
            )
            # A trace that has decayed is less retrievable, as in the biology.
            total *= 0.85 + 0.15 * tr.strength
            out.append(Recall(episode=tr.episode, score=float(total), evidence=dict(ev)))

        if include_schemas and cfg.w_schema > 0 and len(self.neocortex):
            # A schema whose exemplar is already on the hippocampal list is the
            # same memory arriving twice; keep the episodic copy, which carries
            # the real key and provenance.
            # Dedup happens *after* ranking, not here. Filtering schema
            # exemplars against every hippocampal candidate deleted the entire
            # pathway: a schema's exemplar is by construction one of the stored
            # episodes, and `out` holds all of them, so the filter matched
            # everything and the slow store silently contributed nothing.
            sch = list(self.neocortex.query(q, top_k=max(4, k)))
            if sch:
                # Express schema scores in hippocampal units so the weight is a
                # real routing decision rather than an arbitrary constant that
                # happens to sit below every episodic score.
                hmax = max((r.score for r in out), default=1.0) or 1.0
                w = cfg.w_schema * (cfg.schema_generic_boost if _GENERIC.search(query) else 1.0)
                vals = np.array([r.score for r in sch], dtype=np.float32)
                lo, hi = float(vals.min()), float(vals.max())
                rng = hi - lo
                for r in sch:
                    norm = (r.score - lo) / rng if rng > 1e-6 else 1.0
                    r.score = w * hmax * float(norm)
                    r.evidence["routed_generic"] = float(bool(_GENERIC.search(query)))
                    out.append(r)

        # ---- temporal context reinstatement -------------------------------
        # A directional cue means the query names an *anchor* and asks for its
        # neighbour. Retrieve the anchor on content, reinstate its encoding
        # context, and let that context cue the slots adjacent to it. Without
        # this the system confidently returns the anchor itself, which is the
        # one episode guaranteed to be wrong.
        direction = 1 if _AFTER.search(query) else (-1 if _BEFORE.search(query) else 0)
        if direction and cfg.w_reinstate > 0 and out:
            anchor = max(out, key=lambda r: r.score)
            a_sess, a_pos = anchor.episode.key()
            for r in out:
                sess, pos = r.episode.key()
                if sess != a_sess:
                    continue
                offset = (pos - a_pos) * direction
                if 1 <= offset <= cfg.reinstate_span:
                    # Nearest neighbour gets the full bonus, decaying with distance.
                    r.score += cfg.w_reinstate * anchor.score / offset
                    r.evidence["reinstated"] = float(offset)
                elif offset == 0:
                    # Suppress the anchor: it is what was asked *about*, not for.
                    r.score *= 0.15
                    r.evidence["anchor"] = 1.0

        out.sort(key=lambda r: r.score, reverse=True)

        # Collapse duplicates by text, keeping the highest-scoring copy. A
        # memory that surfaced through both stores is one memory.
        deduped: list[Recall] = []
        seen_text: set[str] = set()
        for r in out:
            if r.episode.text in seen_text:
                continue
            seen_text.add(r.episode.text)
            deduped.append(r)
        top = deduped[:k]

        # Retrieval is itself a rehearsal event: recalling strengthens.
        for r in top:
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
        right episode from 90% deletion of its active units at 100% accuracy,
        flat from 400 to 4,000 stored patterns, degrading only when fewer than
        about three units survive. For text queries use :meth:`recall`.

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

    def sleep(self, cycles: int = 10) -> list:
        """Run ``cycles`` ripple bursts. Where generalisation happens."""
        now = time.time()
        return [self.replay.ripple(self.ca1, self.neocortex, now) for _ in range(cycles)]

    # ---------------------------------------------------------------- misc

    def stats(self) -> dict[str, float]:
        s = {
            "written": float(self._n_written),
            "hippocampal": float(len(self.ca1)),
            "ca3_patterns": float(len(self.ca3)),
            "ripples": float(self.replay.step),
        }
        s.update({f"dg_{k}": v for k, v in self.dg.stats().items()})
        s.update({f"nc_{k}": v for k, v in self.neocortex.stats().items()})
        # How much of the written stream the fast store is still holding.
        s["retention"] = s["hippocampal"] / max(s["written"], 1.0)
        return s
