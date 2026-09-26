"""Conjunctive coding: binding *what* to *where* to *when*.

This module exists because of a measured failure, and the failure is worth
stating precisely because it is the one that decides whether an agent memory is
usable.

Take 125 incident reports of the form *"On {day} the {service} service suffered
{fault} during the rollout"*, and ask *"What went wrong with the auth service on
Monday?"*. Sentence-embedding cosine retrieves the right episode 4% of the time.
Mean pairwise cosine across the set is 0.529 and the maximum is 0.998: the
episodes are, to the encoder, nearly the same sentence. The decisive tokens —
``Monday``, ``auth`` — are two words inside a shared frame, and a sentence
encoder's pooled vector does not keep *which* values filled *which* slots. It
represents the frame. (A lexical vector does: with ``HashingEncoder`` dense
cosine solves this set outright. The failure belongs to semantic encoders,
which is why the fix is a lexical code.)

Expanding that vector does not help, because expansion is injective: the dentate
gyrus can only separate what the input already distinguishes. Measured on the
same set, the DG pathway reaches 4.8%, no better than cosine
(``bench/incidents.py``, MiniLM).

The hippocampus does not solve this with a pooled vector. CA3's recurrent
network implements *conjunctive* coding: cells fire for combinations — this
object, in this place, at this time — and not for the elements separately. A
conjunction is a distinct addressable unit, so *auth-on-Monday* is a different
memory address from *auth-on-Tuesday*, however similar the surface forms.

The implementation is a sparse binding code. Each content term, and each
unordered pair of content terms that occur within ``window`` terms of each
other, hashes to one unit. A query mentioning ``auth`` and ``Monday`` activates
the unit for that pair, and that unit is active for exactly the episodes
containing both close together. Retrieval becomes weighted set intersection in
conjunction space rather than angle comparison in embedding space.

Information retrieval has a name for this feature. Metzler & Croft's sequential
dependence model (SIGIR 2005) scores unordered-window co-occurrences
``#uwN(q_i, q_j)`` next to single terms, and found a window of 8 terms a good
model of sentence-level proximity. A pair unit here is that feature, with two
differences: the window counts content terms (stopwords removed), and the unit
is weighted by its *own* document frequency in the store, so a rare
conjunction of two common words counts as rare. :class:`ConjunctionIndex`
holds those counts.

Measured on the 125 reports: top-1 goes 0.040 -> 0.200 and MRR 0.165 -> 0.457.
Both numbers are the ceiling for that query set — the queries name only two of
three attributes, so five episodes are genuinely tied, and 0.457 is exactly the
expected MRR of a uniform draw within a tied group of five.

It is deliberately not a learned component. It is a hash and a counter: it costs
no training, and two episodes sharing a conjunction share a unit, with
collisions bounded by the code width.

Correction (v0.2)
-----------------
Before v0.2 the terms were deduplicated into a *sorted* list and truncated to
the first 24, and every pair among those 24 was coded. The docstring claimed
the truncation kept "the earliest, which in practice are the topical ones"; it
kept the alphabetically first. On the templated benchmark no episode has 24
content terms, so the truncation never fired there, but on real conversational
turns (LongMemEval user turns: median 31 words, p90 66) most of the content was
silently discarded. Single-character tokens were also dropped, which removed
the digit in ``week 7`` while keeping the one in ``week 14``. Both are fixed,
and both have tests that fail on the old code.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter
from collections.abc import Iterable

import numpy as np
from scipy import sparse

__all__ = ["ConjunctiveBinder", "ConjunctionIndex", "STOPWORDS", "WEIGHTINGS", "content_terms"]

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'\-]*")

#: Function words carry frame, not content. Binding on them manufactures
#: spurious conjunctions between unrelated episodes that share syntax.
STOPWORDS = frozenset(
    """
    a an the this that these those it its
    is are was were be been being am
    do does did done doing have has had having
    of in on at by to for from with without into onto over under
    and or but nor so yet if then than as
    what which who whom whose when where why how
    i you he she they we me him her them us
    my your his their our not no yes
    during about after before between while
    """.split()
)


def content_terms(text: str, stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    """Content tokens of ``text`` in reading order, repeats kept.

    Order matters because pair units are windowed by position. A one-character
    token survives if it is a digit (``week 7``) and is dropped otherwise (the
    ``s`` left by a split possessive).
    """
    out: list[str] = []
    for w in _TOKEN.findall(text.lower()):
        if w in stopwords:
            continue
        if len(w) == 1 and not w.isdigit():
            continue
        out.append(w)
    return out


class ConjunctiveBinder:
    """Sparse conjunction codes over content terms.

    Parameters
    ----------
    width:
        Size of the code space. Units are 64-bit hashes reduced modulo
        ``width``. The default, 2**32, keeps collisions negligible for a store
        of a few million distinct conjunctions. The old default of 2**16 suited
        the templated benchmark and collides heavily on real text, where one
        60-word turn already produces several hundred units.
    order:
        Maximum conjunction arity. 1 is a bag of terms. 2 (pairs) captures the
        *which value filled which slot* structure that defeats pooled
        embeddings. 3 adds triples inside the same window.
    window:
        Two terms bind only if they are fewer than ``window`` content terms
        apart: SDM's ``#uwN`` with N = ``window``. ``None`` binds every pair in
        the text, which is quadratic in its length and is what v0.1 did. The
        default, 4 content terms (about 6-8 words once stopwords are counted
        back in), was chosen on the LongMemEval-S dev split over 2, 3, 6, 8,
        16 and unbounded.
    """

    def __init__(
        self,
        width: int = 1 << 32,
        order: int = 2,
        window: int | None = 4,
        stopwords: frozenset[str] = STOPWORDS,
    ) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        if window is not None and window < 2:
            raise ValueError("window must be >= 2 (or None for unbounded)")
        self.width = width
        self.order = order
        self.window = window
        self.stopwords = stopwords

    @staticmethod
    def _h(s: str) -> int:
        return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")

    def _unit(self, terms: tuple[str, ...]) -> int:
        return self._h("\x1f".join(terms)) % self.width

    def conjunction_counts(self, text: str) -> Counter[tuple[str, ...]]:
        """How often each sorted term tuple occurs in ``text``, before hashing.

        A term counts once per occurrence. A pair counts once per pair of
        positions inside the window, so two terms that recur together recur as
        a conjunction.
        """
        toks = content_terms(text, self.stopwords)
        out: Counter[tuple[str, ...]] = Counter((w,) for w in toks)
        if self.order < 2 or len(toks) < 2:
            return out
        span = len(toks) if self.window is None else self.window
        for i in range(len(toks)):
            # Every combination of later terms inside the window that opens at i.
            ahead = toks[i + 1 : i + span]
            for k in range(1, self.order):
                for rest in itertools.combinations(ahead, k):
                    combo = (toks[i], *rest)
                    if len(set(combo)) == len(combo):
                        out[tuple(sorted(combo))] += 1
        return out

    def conjunctions(self, text: str) -> set[tuple[str, ...]]:
        """The sorted term tuples ``text`` activates, before hashing."""
        return set(self.conjunction_counts(text))

    def encode_counts(self, text: str) -> tuple[np.ndarray, np.ndarray, int]:
        """Sorted unique units, their occurrence counts, and the text's length in content terms."""
        counts: dict[int, int] = {}
        for c, n in self.conjunction_counts(text).items():
            u = self._unit(c)
            counts[u] = counts.get(u, 0) + n
        units = np.array(sorted(counts), dtype=np.int64)
        tf = np.array([counts[u] for u in units.tolist()], dtype=np.float64)
        return units, tf, len(content_terms(text, self.stopwords))

    def encode(self, text: str) -> np.ndarray:
        """Sorted unique unit indices active for ``text``."""
        return self.encode_counts(text)[0]

    def encode_many(self, texts: Iterable[str]) -> list[np.ndarray]:
        return [self.encode(t) for t in texts]

    # ------------------------------------------------------------------ #

    @staticmethod
    def match(query_code: np.ndarray, episode_code: np.ndarray) -> float:
        """Unweighted, cosine-normalised conjunction overlap in [0, 1].

        Normalising by ``sqrt(|q| * |e|)`` rather than by the union keeps a
        short query from being penalised for the episode carrying detail it did
        not ask about — which is the normal case for a question against a full
        episode. :meth:`ConjunctionIndex.score` is the weighted, vectorised
        form the memory uses.
        """
        if query_code.size == 0 or episode_code.size == 0:
            return 0.0
        inter = np.intersect1d(query_code, episode_code, assume_unique=True).size
        return float(inter / np.sqrt(query_code.size * episode_code.size))

    def match_many(self, query_code: np.ndarray, codes: list[np.ndarray]) -> np.ndarray:
        return np.array([self.match(query_code, c) for c in codes], dtype=np.float32)


WEIGHTINGS = ("idf", "binary", "bm25")


class ConjunctionIndex:
    """Inverted index over conjunction codes, weighted by unit rarity.

    Each stored code is a row of a sparse matrix (rows = episodes, columns =
    units seen so far). Every weighting uses the unit's own document frequency
    ``df(u)`` among the ``N`` stored episodes, through the non-negative BM25 IDF

        w(u) = ln(1 + (N - df(u) + 0.5) / (df(u) + 0.5)),

    so a pair unit is weighted by how rare the *conjunction* is in this store,
    not by assuming its terms are independent.

    ``weighting="idf"``
        Weighted cosine over unit presence,
        ``s(q, e) = sum_{u in q & e} w(u)^2 / (||q||_w * ||e||_w)``.
    ``weighting="binary"``
        The same with every weight 1. It reproduces
        :meth:`ConjunctiveBinder.match` up to the query norm, which is constant
        within a query and so never changes a ranking.
    ``weighting="bm25"``
        Okapi BM25 with conjunction units as the terms:
        ``s(q, e) = sum_{u in q & e} w(u) * tf(k1 + 1) / (tf + k1 (1 - b + b |e| / avg|e|))``,
        with ``tf`` the unit's count in the episode and ``|e|`` its length in
        content terms. Term frequency saturates and long episodes are
        discounted, which the cosine forms lack. With ``order=1`` it is BM25
        over content terms.

    Scoring is one sparse matrix-vector product, O(nnz), in place of the
    per-episode Python loop v0.1 ran.
    """

    def __init__(self, weighting: str = "idf", k1: float = 1.5, b: float = 0.75) -> None:
        self._check(weighting)
        self.weighting = weighting
        self.k1, self.b = k1, b
        self._col: dict[int, int] = {}
        self._df: list[int] = []
        self._rows: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        self._cache: tuple[list[int], sparse.csr_matrix, np.ndarray, np.ndarray] | None = None

    @staticmethod
    def _check(weighting: str) -> None:
        if weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}")

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, key: int) -> bool:
        return key in self._rows

    def keys(self) -> list[int]:
        return list(self._rows)

    def set_weighting(self, weighting: str) -> None:
        """Switch weighting in place. Only the scores change; the counts do not."""
        self._check(weighting)
        self.weighting = weighting
        self._cache = None

    def add(self, key: int, code: np.ndarray, tf: np.ndarray | None = None, length: float | None = None) -> None:
        """Store one episode's units, with their counts and its length if known."""
        if key in self._rows:
            self.remove(key)
        cols = np.empty(code.size, dtype=np.int64)
        for i, u in enumerate(code.tolist()):
            c = self._col.get(u)
            if c is None:
                c = len(self._df)
                self._col[u] = c
                self._df.append(0)
            self._df[c] += 1
            cols[i] = c
        tf = np.ones(code.size) if tf is None else np.asarray(tf, dtype=np.float64)
        self._rows[key] = (cols, tf, float(code.size if length is None else length))
        self._cache = None

    def remove(self, key: int) -> None:
        row = self._rows.pop(key, None)
        if row is None:
            return
        for c in row[0].tolist():
            self._df[c] -= 1
        self._cache = None

    def df(self, unit: int) -> int:
        c = self._col.get(unit)
        return 0 if c is None else self._df[c]

    def weights(self) -> np.ndarray:
        df = np.asarray(self._df, dtype=np.float64)
        if self.weighting == "binary":
            return (df > 0).astype(np.float64)
        n = float(len(self._rows))
        w = np.log1p((n - df + 0.5) / (df + 0.5))
        w[df <= 0] = 0.0
        return w

    def _build(self) -> tuple[list[int], sparse.csr_matrix, np.ndarray, np.ndarray]:
        """(keys, matrix, query weights per column, row norms), cached until the next write."""
        if self._cache is None:
            keys = list(self._rows)
            n_cols = max(len(self._df), 1)
            if not keys:
                self._cache = (keys, sparse.csr_matrix((0, n_cols)), np.zeros(n_cols), np.zeros(0))
                return self._cache
            indptr = np.zeros(len(keys) + 1, dtype=np.int64)
            for i, k in enumerate(keys):
                indptr[i + 1] = indptr[i] + self._rows[k][0].size
            indices = np.concatenate([self._rows[k][0] for k in keys])
            w = self.weights()
            if w.size == 0:
                w = np.zeros(1)
            if self.weighting == "bm25":
                tf = np.concatenate([self._rows[k][1] for k in keys])
                length = np.array([self._rows[k][2] for k in keys])
                avg = float(length.mean()) or 1.0
                norm = self.k1 * (1 - self.b + self.b * length / avg)
                per_entry = np.repeat(norm, np.diff(indptr))
                data = tf * (self.k1 + 1) / (tf + per_entry)
                A = sparse.csr_matrix((data, indices, indptr), shape=(len(keys), n_cols))
                self._cache = (keys, A, w, np.ones(len(keys)))
            else:
                A = sparse.csr_matrix((np.ones(indices.size), indices, indptr), shape=(len(keys), n_cols))
                w2 = w**2
                self._cache = (keys, A, w2, np.sqrt(A @ w2))
        return self._cache

    def score(self, code: np.ndarray) -> tuple[list[int], np.ndarray]:
        """Scores for every stored key, in the order of the returned key list."""
        keys, A, qw, norms = self._build()
        if not keys:
            return keys, np.zeros(0)
        cols = sorted({self._col[u] for u in code.tolist() if u in self._col})
        if not cols:
            return keys, np.zeros(len(keys))
        q = np.zeros(A.shape[1], dtype=np.float64)
        q[cols] = qw[cols]
        dots = A @ q
        if self.weighting == "bm25":
            return keys, dots
        qn = math.sqrt(float(q.sum()))
        if qn <= 0:
            return keys, np.zeros(len(keys))
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(norms > 0, dots / (norms * qn), 0.0)
        return keys, s
