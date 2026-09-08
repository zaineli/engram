"""Conjunctive coding: binding *what* to *where* to *when*.

This module exists because of a measured failure, and the failure is worth
stating precisely because it is the one that decides whether an agent memory is
usable.

Take 125 incident reports of the form *"On {day} the {service} service suffered
{fault} during the rollout"*, and ask *"What went wrong with the auth service on
Monday?"*. Sentence-embedding cosine retrieves the right episode 4% of the time.
Mean pairwise cosine across the set is 0.465 and the maximum is 0.998: the
episodes are, to the encoder, nearly the same sentence. The decisive tokens —
``Monday``, ``auth`` — are two words inside a shared frame, and a single pooled
vector cannot represent *which* values filled *which* slots. It represents the
frame.

Expanding that vector does not help, because expansion is injective: the dentate
gyrus can only separate what the input already distinguishes. Measured on the
same set, the DG pathway reaches 5.6%, no better than cosine.

The hippocampus does not solve this with a pooled vector. CA3's recurrent
network implements *conjunctive* coding: cells fire for combinations — this
object, in this place, at this time — and not for the elements separately. A
conjunction is a distinct addressable unit, so *auth-on-Monday* is a different
memory address from *auth-on-Tuesday*, however similar the surface forms.

The implementation is a sparse binding code. Every unordered subset of content
terms up to ``order`` hashes to one unit in a large binary space. A query
mentioning ``auth`` and ``Monday`` activates the unit for that pair, and that
unit is active for exactly the episodes containing both. Retrieval becomes set
intersection in conjunction space rather than angle comparison in embedding
space.

Measured on the same 125 episodes: top-1 goes 0.040 -> 0.200 and MRR 0.165 ->
0.457. Both numbers are the ceiling for that query set — the queries name only
two of three attributes, so five episodes are genuinely tied, and 0.457 is
exactly the expected MRR of a uniform draw within a tied group of five. The
pathway is not approximately right; it narrows to the correct equivalence class
and then cannot do better because the question does not contain the answer.

This is deliberately not a learned component. It is a hash, it costs no
training, and it is exact: two episodes sharing a conjunction share a unit, with
collisions bounded by the code width.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Iterable

import numpy as np

__all__ = ["ConjunctiveBinder", "STOPWORDS"]

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


def _content_terms(text: str, stopwords: frozenset[str]) -> list[str]:
    seen: dict[str, None] = {}
    for w in _TOKEN.findall(text.lower()):
        if len(w) > 1 and w not in stopwords:
            seen[w] = None
    return sorted(seen)


class ConjunctiveBinder:
    """Sparse conjunction codes over content terms.

    Parameters
    ----------
    width:
        Size of the binary code space. Must be large relative to the number of
        distinct conjunctions in the corpus or collisions start to matter;
        2**16 comfortably handles corpora in the 10^4-10^5 episode range.
    order:
        Maximum conjunction arity. 2 (pairs) captures the *which value filled
        which slot* structure that defeats pooled embeddings. 3 adds triples,
        which sharpens further at a cost of ``C(n,3)`` units per episode — the
        benchmark sweeps this and finds the elbow at 2 for most corpora.
    max_terms:
        Guard against a long episode producing a combinatorial blow-up. Terms
        beyond this are truncated, keeping the earliest, which in practice are
        the topical ones.
    """

    def __init__(
        self,
        width: int = 1 << 16,
        order: int = 2,
        max_terms: int = 24,
        stopwords: frozenset[str] = STOPWORDS,
    ) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.width = width
        self.order = order
        self.max_terms = max_terms
        self.stopwords = stopwords

    @staticmethod
    def _h(s: str) -> int:
        return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")

    def encode(self, text: str) -> np.ndarray:
        """Sorted unique unit indices active for ``text``."""
        terms = _content_terms(text, self.stopwords)[: self.max_terms]
        units = {self._h(w) % self.width for w in terms}
        for k in range(2, self.order + 1):
            if len(terms) < k:
                break
            for combo in itertools.combinations(terms, k):
                units.add(self._h("\x1f".join(combo)) % self.width)
        return np.array(sorted(units), dtype=np.int64)

    def encode_many(self, texts: Iterable[str]) -> list[np.ndarray]:
        return [self.encode(t) for t in texts]

    # ------------------------------------------------------------------ #

    @staticmethod
    def match(query_code: np.ndarray, episode_code: np.ndarray) -> float:
        """Cosine-normalised conjunction overlap in [0, 1].

        Normalising by ``sqrt(|q| * |e|)`` rather than by the union keeps a
        short query from being penalised for the episode carrying detail it did
        not ask about — which is the normal case for a question against a full
        episode.
        """
        if query_code.size == 0 or episode_code.size == 0:
            return 0.0
        inter = np.intersect1d(query_code, episode_code, assume_unique=True).size
        return float(inter / np.sqrt(query_code.size * episode_code.size))

    def match_many(self, query_code: np.ndarray, codes: list[np.ndarray]) -> np.ndarray:
        return np.array([self.match(query_code, c) for c in codes], dtype=np.float32)
