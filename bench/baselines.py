"""Baselines. The comparison only means something if these are honest.

``FlatVectorStore`` is the one that matters — dense embeddings, cosine, top-k is
what essentially every production agent-memory layer is, and it is the thing
this architecture claims to beat. ``HybridRAG`` is included because a serious
engineer's first response to a pattern-separation failure is "add BM25", and
that response deserves to be measured rather than dismissed: lexical matching
genuinely does recover some conjunction information, since ``week 7`` and
``Monday`` are literal tokens.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

__all__ = ["FlatVectorStore", "BM25", "HybridRAG", "RecencyWindow"]

_TOK = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> list[str]:
    return _TOK.findall(s.lower())


class FlatVectorStore:
    """Dense embedding + cosine top-k. The standard RAG memory."""

    name = "flat-vector"

    def __init__(self, encoder) -> None:
        self.encoder = encoder
        self._texts: list[str] = []
        self._keys: list[tuple[int, int]] = []
        self._M: np.ndarray | None = None

    def add_many(self, texts: list[str], keys: list[tuple[int, int]]) -> None:
        self._texts.extend(texts)
        self._keys.extend(keys)
        V = self.encoder.encode(texts)
        self._M = V if self._M is None else np.vstack([self._M, V])

    def search(self, query: str, k: int = 5):
        if self._M is None:
            return []
        q = self.encoder.encode([query])[0]
        s = self._M @ q
        order = np.argsort(s)[::-1][:k]
        return [(self._texts[i], float(s[i]), self._keys[i]) for i in order]


class BM25:
    """Okapi BM25. Lexical, and surprisingly hard to beat on token-literal keys."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._docs: list[list[str]] = []
        self._texts: list[str] = []
        self._keys: list[tuple[int, int]] = []
        self._df: Counter = Counter()
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._avgdl = 0.0

    def add_many(self, texts: list[str], keys: list[tuple[int, int]]) -> None:
        for t, key in zip(texts, keys):
            toks = _tok(t)
            i = len(self._docs)
            self._docs.append(toks)
            self._texts.append(t)
            self._keys.append(key)
            tf = Counter(toks)
            for w, c in tf.items():
                self._postings[w].append((i, c))
            self._df.update(tf.keys())
        self._avgdl = sum(len(d) for d in self._docs) / max(len(self._docs), 1)

    def search(self, query: str, k: int = 5):
        N = len(self._docs)
        if N == 0:
            return []
        scores = np.zeros(N, dtype=np.float32)
        for w in set(_tok(query)):
            post = self._postings.get(w)
            if not post:
                continue
            idf = math.log(1 + (N - self._df[w] + 0.5) / (self._df[w] + 0.5))
            for i, c in post:
                dl = len(self._docs[i])
                denom = c + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1e-9))
                scores[i] += idf * (c * (self.k1 + 1)) / denom
        order = np.argsort(scores)[::-1][:k]
        return [(self._texts[i], float(scores[i]), self._keys[i]) for i in order]


class HybridRAG:
    """Dense + BM25 with reciprocal rank fusion. The strong, realistic baseline."""

    name = "hybrid-rag"

    def __init__(self, encoder, rrf_k: int = 60) -> None:
        self.dense = FlatVectorStore(encoder)
        self.sparse = BM25()
        self.rrf_k = rrf_k

    def add_many(self, texts: list[str], keys: list[tuple[int, int]]) -> None:
        self.dense.add_many(texts, keys)
        self.sparse.add_many(texts, keys)

    def search(self, query: str, k: int = 5):
        pool = max(k * 6, 30)
        fused: dict[tuple[int, int], list] = {}
        for lst in (self.dense.search(query, pool), self.sparse.search(query, pool)):
            for rank, (text, _, key) in enumerate(lst):
                slot = fused.setdefault(key, [text, 0.0])
                slot[1] += 1.0 / (self.rrf_k + rank + 1)
        out = sorted(fused.items(), key=lambda kv: kv[1][1], reverse=True)[:k]
        return [(v[0], v[1], key) for key, v in out]


class RecencyWindow:
    """Last-N episodes. The floor: what a chat history buffer gives you."""

    name = "recency"

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self._texts: list[str] = []
        self._keys: list[tuple[int, int]] = []

    def add_many(self, texts: list[str], keys: list[tuple[int, int]]) -> None:
        self._texts.extend(texts)
        self._keys.extend(keys)

    def search(self, query: str, k: int = 5):
        n = len(self._texts)
        picked = list(range(max(0, n - self.window), n))[::-1][:k]
        return [(self._texts[i], 1.0 / (r + 1), self._keys[i]) for r, i in enumerate(picked)]
