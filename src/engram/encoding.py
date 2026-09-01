"""Entorhinal cortex: the encoding front door.

Everything entering the hippocampal formation passes through here. Two codes
come out of a single experience:

``dense``
    A semantic embedding. This is the "what" — it should place similar
    meanings near each other, because that is what makes semantic retrieval and
    neocortical schema formation possible.

``context``
    A position/phase code. This is the "when and where in the stream" — grid
    and time cells in medial entorhinal cortex supply a slowly drifting signal
    that lets the hippocampus bind an item to its place in a sequence.

Keeping these separate matters. If sequence position is mixed into the semantic
vector, two unrelated events that happened at the same moment become similar,
and every downstream stage inherits that error.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

import numpy as np

__all__ = ["Encoder", "HashingEncoder", "SentenceTransformerEncoder", "ThetaContext"]

_TOKEN = re.compile(r"[a-z0-9']+")


class Encoder(Protocol):
    """Anything that turns text into L2-normalised dense vectors."""

    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class HashingEncoder:
    """Deterministic bag-of-ngrams encoder with no model download.

    This exists so the test suite and CI have a fast, hermetic encoder, and so
    the architecture can be exercised on a machine with no network. It is a
    genuinely weak semantic encoder — it captures lexical overlap and nothing
    else — which makes it a useful stress case: mechanisms that only work with
    a strong embedding model are not mechanisms, they are the embedding model.
    """

    def __init__(self, dim: int = 384, ngram: tuple[int, int] = (1, 2), seed: int = 0) -> None:
        self.dim = dim
        self.ngram = ngram
        self.seed = seed

    def _hash(self, token: str) -> int:
        h = hashlib.blake2b(f"{self.seed}:{token}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "little")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        lo, hi = self.ngram
        for row, text in enumerate(texts):
            toks = _TOKEN.findall(text.lower())
            for n in range(lo, hi + 1):
                for i in range(len(toks) - n + 1):
                    gram = " ".join(toks[i : i + n])
                    h = self._hash(gram)
                    idx = h % self.dim
                    sign = 1.0 if (h >> 63) & 1 else -1.0
                    # Sublinear weighting; long episodes should not dominate.
                    out[row, idx] += sign / math.sqrt(n)
        return _l2(out)


class SentenceTransformerEncoder:
    """Real semantic embeddings, loaded lazily.

    Default is ``all-MiniLM-L6-v2``: 384-d, ~22M params, fast enough on an M-series
    CPU that the full benchmark sweep stays in the tens of seconds.
    """

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # local import: heavy

        self._model = SentenceTransformer(model, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        v = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128,
        )
        return v.astype(np.float32, copy=False)


class ThetaContext:
    """Theta-phase position code.

    Place cells fire at a progressively earlier phase of the theta cycle as an
    animal moves through a place field — phase precession. Compressed into one
    theta cycle, a whole upcoming sequence is represented in order. That is the
    mechanism that lets the hippocampus store *order*, not just contents.

    The code here is the engineering analogue: a sequence position is mapped to
    a bank of sinusoids at geometrically spaced frequencies, which gives a
    representation where dot product falls off smoothly with distance in the
    sequence and stays near-orthogonal across sessions.

    ``sim(i, j)`` therefore behaves like a soft "how close in the stream were
    these" kernel, which is exactly what a sequence-order query needs.
    """

    def __init__(self, dim: int = 32, bands: int = 8, session_scale: float = 97.0) -> None:
        if dim % 2:
            raise ValueError("ThetaContext dim must be even (sin/cos pairs)")
        self.dim = dim
        self.bands = bands
        self.session_scale = session_scale
        pairs = dim // 2
        # Geometric frequency ladder, the positional-encoding trick, chosen so the
        # slowest band spans a long session and the fastest resolves adjacency.
        self._freqs = np.array(
            [1.0 / (self.bands ** (2 * i / max(pairs - 1, 1))) for i in range(pairs)],
            dtype=np.float32,
        )

    def encode(self, session: int, position: int) -> np.ndarray:
        # Sessions are pushed far apart so that slot 3 of session 1 and slot 3
        # of session 2 do not collide.
        t = float(session) * self.session_scale + float(position)
        ang = t * self._freqs
        v = np.empty(self.dim, dtype=np.float32)
        v[0::2] = np.sin(ang)
        v[1::2] = np.cos(ang)
        return v / math.sqrt(self.dim // 2)

    def encode_many(self, sessions: Sequence[int], positions: Sequence[int]) -> np.ndarray:
        return np.stack([self.encode(s, p) for s, p in zip(sessions, positions)])
