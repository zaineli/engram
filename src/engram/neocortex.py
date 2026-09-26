"""Neocortex: the slow, semantic, schema-forming store.

The complementary-learning-systems argument (McClelland, McNaughton & O'Reilly,
1995) is that one store cannot do both jobs. Learning fast enough to capture a
single experience in one shot requires large weight changes, and large weight
changes overwrite what is already there — catastrophic interference. Learning
slowly enough to extract stable statistical structure means you cannot capture
anything in one shot.

The resolution is two stores with different learning rates and a transfer
process between them. The hippocampus takes the one-shot write. The neocortex
learns slowly, interleaved, from replayed hippocampal traces, and what it ends
up representing is not the episodes but the regularities across them.

Here the neocortical store is a set of :class:`~engram.types.Schema` objects:
online-clustered centroids over replayed traces. It answers a different kind of
query than the hippocampus does. Ask it for a specific Tuesday and it will fail.
Ask it what usually happens and it is the only part of the system that can
answer at all, because that fact was never in any single episode.
"""

from __future__ import annotations

import numpy as np

from .types import Recall, Schema

__all__ = ["Neocortex"]


class Neocortex:
    """Online schema formation over consolidated traces.

    Parameters
    ----------
    dim:
        Dense embedding dimension.
    merge_threshold:
        Cosine above which a replayed trace joins an existing schema rather
        than founding a new one. This is the abstraction/precision dial: low
        values produce few, broad schemas, high values produce many narrow ones
        that are barely more than the episodes they came from.
    learning_rate:
        Weight a single replay contributes relative to the schema's accumulated
        mass. Deliberately small — this is the slow system.
    """

    def __init__(self, dim: int, merge_threshold: float = 0.72, learning_rate: float = 1.0) -> None:
        self.dim = dim
        self.merge_threshold = merge_threshold
        self.learning_rate = learning_rate
        self.schemas: list[Schema] = []
        self._centroids: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.schemas)

    def _invalidate(self) -> None:
        self._centroids = None

    def _stack(self) -> np.ndarray:
        if self._centroids is None:
            self._centroids = (
                np.stack([s.centroid for s in self.schemas])
                if self.schemas
                else np.zeros((0, self.dim), dtype=np.float32)
            )
        return self._centroids

    # ------------------------------------------------------------------ #

    def absorb(
        self, dense: np.ndarray, key: tuple[int, int], text: str, weight: float = 1.0
    ) -> tuple[int, bool]:
        """Fold one replayed trace into the schema set.

        Returns ``(schema_index, created_new)``.
        """
        w = weight * self.learning_rate
        C = self._stack()
        if C.shape[0]:
            sims = C @ dense
            j = int(np.argmax(sims))
            if float(sims[j]) >= self.merge_threshold:
                self.schemas[j].merge(dense, key, text, w)
                self._invalidate()
                return j, False

        s = Schema(centroid=dense.astype(np.float32).copy(), mass=0.0, label=_label(text))
        s.merge(dense, key, text, w)
        self.schemas.append(s)
        self._invalidate()
        return len(self.schemas) - 1, True

    def query(self, dense: np.ndarray, top_k: int = 3, mass_weight: float = 0.6) -> list[Recall]:
        """Gist retrieval. Returns schema exemplars, not episodes.

        Ranking combines centroid similarity with accumulated ``mass``, and the
        mass term is not a tie-breaker. Asked *"which project does Priya usually
        work on"* (``bench/schemas.py``, seed 0), Priya's 12 schemas span
        similarity 0.311-0.562, and the most similar one is an *ingest* schema
        of mass 4.1; similarity alone answers wrongly. Mass spans 2.4-53.2, and
        with it the top schema is an *atlas* one, her modal project (30 of 50
        events). Similarity finds the topic; mass answers "usually". An earlier
        version ranked on similarity alone and the abstraction task did not
        move when this pathway was ablated.
        """
        C = self._stack()
        if C.shape[0] == 0:
            return []
        sims = C @ dense
        mass = np.array([s.mass for s in self.schemas], dtype=np.float32)
        # Log-compressed and normalised: evidence should matter with strongly
        # diminishing returns, or one runaway schema swamps the ranking.
        m = np.log1p(mass)
        m = m / max(float(m.max()), 1e-6)
        scores = sims * (1.0 - mass_weight) + sims * mass_weight * m
        order = np.argsort(scores)[::-1][:top_k]
        out: list[Recall] = []
        for j in order:
            s = self.schemas[int(j)]
            if not s.exemplars:
                continue
            from .types import Episode

            ep = Episode(text=s.exemplars[0], meta={"schema": int(j), "support": len(s.support)})
            out.append(
                Recall(
                    episode=ep,
                    score=float(scores[j]),
                    evidence={"schema_similarity": float(sims[j]), "schema_mass": s.mass},
                    source="neocortex",
                )
            )
        return out

    def stats(self) -> dict[str, float]:
        if not self.schemas:
            return {"schemas": 0.0, "mean_support": 0.0, "max_support": 0.0, "compression": 0.0}
        # Unique supporting episodes, not absorb events: a trace replayed five
        # times must not read as five episodes of evidence.
        sup = np.array([len(set(s.support)) for s in self.schemas], dtype=np.float64)
        return {
            "schemas": float(len(self.schemas)),
            "mean_support": float(sup.mean()),
            "max_support": float(sup.max()),
            # Episodes absorbed per schema retained: the abstraction ratio.
            "compression": float(sup.sum() / len(self.schemas)),
        }


def _label(text: str) -> str:
    words = text.split()
    return " ".join(words[:6]) + ("..." if len(words) > 6 else "")
