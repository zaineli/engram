"""Dense encoders for the benchmarks, with a disk cache.

LongMemEval-S has about 97k distinct user turns across its 500 haystacks, and
every configuration of every system re-reads the same ones. Embedding them once
per model and keeping the vectors on disk turns a sweep from hours into
minutes. The cache is keyed by a hash of the exact string the model saw,
prefix included, so a changed prefix can never be served a stale vector.

Some retrieval models were trained with instruction prefixes and are measurably
worse without them. Each is used the way its model card specifies.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["CachedEncoder", "MODELS", "load_encoder"]

#: name -> (hub id, query prefix, document prefix)
MODELS: dict[str, tuple[str, str, str]] = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "bge-small": (
        "BAAI/bge-small-en-v1.5",
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    "e5-small": ("intfloat/e5-small-v2", "query: ", "passage: "),
    "gte-small": ("thenlper/gte-small", "", ""),
}

CACHE = Path(os.environ.get("ENGRAM_CACHE", Path(__file__).resolve().parent / ".cache"))


def _key(s: str) -> str:
    return hashlib.blake2b(s.encode(), digest_size=16).hexdigest()


class CachedEncoder:
    """A sentence-transformers model behind a persistent embedding cache.

    ``encode`` embeds documents and ``encode_query`` embeds queries; engram's
    memory calls ``encode_query`` when an encoder has one.
    """

    def __init__(
        self,
        name: str,
        cache_dir: Path = CACHE,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        if name not in MODELS:
            raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")
        self.name = name
        self.hub_id, self.query_prefix, self.doc_prefix = MODELS[name]
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.cache_dir / f"{name}.npz"
        self._index: dict[str, int] = {}
        self._vecs = np.zeros((0, 0), dtype=np.float32)
        self._pending: list[tuple[str, np.ndarray]] = []
        if self._path.exists():
            z = np.load(self._path)
            self._vecs = z["vecs"]
            self._index = {k: i for i, k in enumerate(z["keys"].tolist())}
        self.dim = int(self._vecs.shape[1]) if self._vecs.size else self._load().get_sentence_embedding_dimension()

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.hub_id, device=self.device)
        return self._model

    def _lookup(self, key: str) -> np.ndarray | None:
        i = self._index.get(key)
        if i is None:
            return None
        if i < self._vecs.shape[0]:
            return self._vecs[i]
        return self._pending[i - self._vecs.shape[0]][1]

    def _embed(self, strings: Sequence[str]) -> np.ndarray:
        keys = [_key(s) for s in strings]
        missing = sorted({(k, s) for k, s in zip(keys, strings) if k not in self._index})
        if missing:
            model = self._load()
            vecs = model.encode(
                [s for _, s in missing],
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(missing) > 5000,
            ).astype(np.float32)
            base = self._vecs.shape[0] + len(self._pending)
            for j, ((k, _), v) in enumerate(zip(missing, vecs)):
                self._index[k] = base + j
                self._pending.append((k, v))
        return np.stack([self._lookup(k) for k in keys]) if keys else np.zeros((0, self.dim), np.float32)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([self.doc_prefix + t for t in texts])

    def encode_query(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([self.query_prefix + t for t in texts])

    def save(self) -> None:
        """Flush newly computed vectors to disk."""
        if not self._pending:
            return
        new = np.stack([v for _, v in self._pending])
        vecs = new if self._vecs.size == 0 else np.vstack([self._vecs, new])
        keys = [None] * vecs.shape[0]
        for k, i in self._index.items():
            keys[i] = k
        tmp = self._path.with_suffix(".tmp.npz")
        np.savez(tmp, vecs=vecs, keys=np.array(keys))
        tmp.replace(self._path)
        self._vecs, self._pending = vecs, []


def load_encoder(name: str, **kw):
    """``hashing`` for the hermetic encoder, else a cached model from ``MODELS``."""
    if name == "hashing":
        from engram import HashingEncoder

        return HashingEncoder(dim=384)
    return CachedEncoder(name, **kw)
