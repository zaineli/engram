"""CA1: the output stage and the mismatch detector.

CA3 settles on an attractor; CA1 is where that settled state is compared
against what actually arrived from entorhinal cortex, and where the resulting
*prediction error* is computed. This comparison is not incidental plumbing. It
is the signal that decides what gets replayed: an experience that matched the
existing model teaches nothing, an experience that violated it is the one worth
consolidating.

So CA1 does two jobs here:

1. Map a CA3 pattern index back to the episode that produced it.
2. Emit a novelty score = 1 - (best overlap with anything already stored),
   which seeds replay priority in :mod:`engram.replay`.
"""

from __future__ import annotations

import numpy as np

from .types import Trace

__all__ = ["CA1"]


class CA1:
    """Index from CA3 pattern slots to traces, plus novelty read-out."""

    def __init__(self) -> None:
        self._traces: dict[int, Trace] = {}
        self._by_key: dict[tuple[int, int], int] = {}

    def __len__(self) -> int:
        return len(self._traces)

    def bind(self, slot: int, trace: Trace) -> None:
        self._traces[slot] = trace
        self._by_key[trace.key] = slot

    def get(self, slot: int) -> Trace | None:
        return self._traces.get(slot)

    def slot_of(self, key: tuple[int, int]) -> int | None:
        return self._by_key.get(key)

    def release(self, slot: int) -> Trace | None:
        """Drop a trace from the fast store once it is safely consolidated."""
        tr = self._traces.pop(slot, None)
        if tr is not None:
            self._by_key.pop(tr.key, None)
        return tr

    def traces(self) -> list[Trace]:
        return list(self._traces.values())

    def slots(self) -> list[int]:
        return list(self._traces.keys())

    # ------------------------------------------------------------------ #

    def novelty(self, dense: np.ndarray) -> float:
        """How unlike anything already held this input is, in [0, 1].

        Cheap dense scan. The fast store is bounded by consolidation, so this
        stays small; it is deliberately computed against the *hippocampal*
        store only, because novelty relative to consolidated knowledge is a
        different quantity and is handled by the neocortex.
        """
        if not self._traces:
            return 1.0
        M = np.stack([t.dense for t in self._traces.values()])
        return float(np.clip(1.0 - (M @ dense).max(), 0.0, 1.0))

    def decay(self, rate: float = 0.995, floor: float = 0.05) -> list[int]:
        """Apply passive forgetting; return slots that fell below ``floor``.

        Trace strength decays every step and is refreshed by replay and by
        successful retrieval. Anything that is never retrieved and never
        replayed eventually drops out, which is the intended behaviour: an
        episodic store that only grows is not a memory system, it is a log.
        """
        dead: list[int] = []
        for slot, tr in self._traces.items():
            tr.strength *= rate
            if tr.strength < floor and not tr.consolidated:
                dead.append(slot)
        return dead
