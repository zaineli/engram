"""Core data types for the engram memory system.

The vocabulary here is deliberately anatomical. Each structure in the
hippocampal formation solves a specific computational problem, and keeping the
names attached to the mechanisms makes the trade-offs legible: when retrieval
degrades you want to know whether separation failed (DG), completion failed
(CA3), or consolidation ran too aggressively (neocortex).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "Episode",
    "Trace",
    "Recall",
    "Schema",
    "ConsolidationEvent",
]


@dataclass(slots=True)
class Episode:
    """A single experience, before it has been encoded.

    ``session`` groups episodes that were experienced contiguously. Sequence
    order inside a session is what theta-phase binding encodes, so it has to
    survive into the store rather than being flattened away.
    """

    text: str
    session: int = 0
    index_in_session: int = 0
    timestamp: float = field(default_factory=time.time)
    #: Free-form ground truth / provenance. Never consulted during retrieval.
    meta: dict[str, Any] = field(default_factory=dict)
    #: Salience in [0, 1]. Drives replay priority when no prediction error is
    #: available (an agent that knows something mattered can say so).
    salience: float = 0.0

    def key(self) -> tuple[int, int]:
        return (self.session, self.index_in_session)


@dataclass(slots=True)
class Trace:
    """An encoded episode resident in the hippocampal (fast) store.

    Holds three parallel representations of the same experience:

    * ``dense``   — the entorhinal code, an L2-normalised semantic embedding.
    * ``sparse``  — the dentate code, a k-sparse binary expansion of ``dense``.
    * ``conj``    — the CA3 conjunction code, binding which value filled which
      slot. This is the representation that survives confusable near-duplicates.
    * ``context`` — the theta/position code binding it to its session and slot.

    Retrieval uses all four, and they fail in different ways: semantic
    similarity comes from ``dense``, interference resistance from ``conj``,
    storage-side separation from ``sparse``, and sequence queries from
    ``context``.
    """

    episode: Episode
    dense: np.ndarray
    sparse_idx: np.ndarray  # int32 indices of active DG units
    conj_idx: np.ndarray  # int64 indices of active conjunction units
    context: np.ndarray
    #: Number of times a sharp-wave ripple has replayed this trace.
    replays: int = 0
    #: Running prediction error at encode time; seeds replay priority.
    surprise: float = 0.0
    #: Decays with time, refreshed by replay and by successful retrieval.
    strength: float = 1.0
    #: Set once the trace has been absorbed into a neocortical schema.
    consolidated: bool = False

    @property
    def key(self) -> tuple[int, int]:
        return self.episode.key()


@dataclass(slots=True)
class Recall:
    """One retrieved item plus the evidence that produced it."""

    episode: Episode
    score: float
    #: Per-pathway contributions, for ablation and for debugging a bad recall.
    evidence: dict[str, float] = field(default_factory=dict)
    #: Which store answered: "hippocampus" or "neocortex".
    source: str = "hippocampus"
    #: CA3 settling steps used. 0 means the cue was already an attractor.
    completion_steps: int = 0


@dataclass(slots=True)
class Schema:
    """A neocortical abstraction distilled from repeatedly replayed traces.

    A schema is what survives when the episode-specific detail is stripped and
    only the statistical regularity across many episodes remains. It carries
    the centroid of its constituent traces plus the surface forms that
    generated it, so a schema hit can still cite its evidence.
    """

    centroid: np.ndarray
    support: list[tuple[int, int]] = field(default_factory=list)
    exemplars: list[str] = field(default_factory=list)
    #: Sum of replay counts absorbed. Proxy for how entrenched the schema is.
    mass: float = 0.0
    label: str = ""

    def merge(self, dense: np.ndarray, key: tuple[int, int], text: str, weight: float) -> None:
        """Fold one more trace into this schema, centroid weighted by mass."""
        total = self.mass + weight
        if total <= 0:
            return
        self.centroid = (self.centroid * self.mass + dense * weight) / total
        n = np.linalg.norm(self.centroid)
        if n > 0:
            self.centroid = self.centroid / n
        self.mass = total
        self.support.append(key)
        if len(self.exemplars) < 8:
            self.exemplars.append(text)


@dataclass(slots=True)
class ConsolidationEvent:
    """Telemetry for one sharp-wave-ripple burst.

    Consolidation is the part of the system most likely to silently destroy
    information, so every burst is recorded rather than run blind.
    """

    step: int
    replayed: list[tuple[int, int]]
    promoted: list[tuple[int, int]]
    schemas_created: int
    schemas_updated: int
    hippocampal_size: int
    neocortical_size: int
