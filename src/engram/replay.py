"""Sharp-wave ripples: prioritised, sequential, interleaved replay.

During quiet waking and slow-wave sleep the hippocampus emits sharp-wave
ripple complexes, and inside them it replays recent trajectories at roughly
10-20x compressed speed. Three properties of biological replay are load-bearing
and all three are implemented here, because dropping any of them changes the
behaviour of the slow store:

**It is prioritised.** Replay is biased toward experiences that were surprising
or rewarded (Michon et al., 2019). Uniform replay wastes the budget on episodes
the model already predicts. Priority here combines prediction error from CA1,
declared salience, recency, and an inverse-count term so that nothing monopolises
the budget forever.

**It is sequential.** Replay follows trajectories, not independent samples. Once
a trace is selected, its temporal neighbours in the same session are pulled in
with it. This is what preserves order information during transfer; sampling
i.i.d. would hand the neocortex a bag of disconnected facts.

**It is interleaved.** This is the part that is easy to omit and expensive to
omit. The entire reason for a slow second store is to avoid catastrophic
interference, and a slow store fed only new material interferes with itself
just as badly as a fast one. Each ripple therefore mixes fresh traces with a
sample of already-consolidated ones. ``bench/ablate_replay.py`` measures what
turning this off costs.
"""

from __future__ import annotations

import numpy as np

from .ca1 import CA1
from .neocortex import Neocortex
from .types import ConsolidationEvent, Trace

__all__ = ["ReplayScheduler"]


class ReplayScheduler:
    """Selects what to replay and drives hippocampal to neocortical transfer.

    Parameters
    ----------
    batch:
        Traces replayed per ripple burst.
    sequence_span:
        How many temporal neighbours to pull in around each selected trace.
        1 means the trace alone.
    interleave:
        Fraction of each batch reserved for already-consolidated material.
    promote_after:
        Replay count at which a trace is considered consolidated.
    capacity:
        Maximum resident hippocampal traces. ``None`` means unbounded. When the
        store exceeds this, the lowest-utility *consolidated* traces are evicted
        until it fits. Eviction is deliberately decoupled from promotion: an
        earlier version released a trace as soon as it was promoted, and that
        destroyed episodic detail the instant a schema absorbed it — the exact
        catastrophic forgetting the two-store design exists to prevent, since a
        schema answers *what usually happens* and can never answer *what
        happened on Tuesday*. Promotion now only marks a trace as safely
        represented in the slow store; whether it is still worth its space is a
        separate question, asked only under real pressure.
    alpha:
        Priority exponent, as in prioritised experience replay. 0 is uniform
        sampling, 1 is fully proportional.
    """

    def __init__(
        self,
        batch: int = 32,
        sequence_span: int = 3,
        interleave: float = 0.3,
        promote_after: int = 3,
        capacity: int | None = None,
        alpha: float = 0.7,
        seed: int = 0,
    ) -> None:
        self.batch = batch
        self.sequence_span = max(1, sequence_span)
        self.interleave = float(np.clip(interleave, 0.0, 0.9))
        self.promote_after = promote_after
        self.capacity = capacity
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.history: list[ConsolidationEvent] = []

    # ------------------------------------------------------------------ #

    def priority(self, tr: Trace, now: float) -> float:
        """Replay priority for one trace.

        Surprise and salience push up; age pushes down slowly; replay count
        pushes down so that consolidation finishes and moves on.
        """
        recency = 1.0 / (1.0 + max(0.0, now - tr.episode.timestamp) / 3600.0)
        novelty = 0.5 * tr.surprise + 0.5 * tr.episode.salience
        fatigue = 1.0 / (1.0 + tr.replays)
        return float(max(1e-6, (0.15 + novelty) * (0.35 + 0.65 * recency) * fatigue))

    def _select(self, ca1: CA1, now: float) -> list[int]:
        slots = [s for s in ca1.slots() if (t := ca1.get(s)) and not t.consolidated]
        if not slots:
            return []
        p = np.array([self.priority(ca1.get(s), now) for s in slots], dtype=np.float64)
        p = p**self.alpha
        p /= p.sum()
        n_fresh = max(1, int(round(self.batch * (1.0 - self.interleave))))
        n_fresh = min(n_fresh, len(slots))
        picked = self.rng.choice(len(slots), size=n_fresh, replace=False, p=p)
        chosen = [slots[int(i)] for i in picked]

        # Sequential extension: pull temporal neighbours of each seed so that
        # order information survives the transfer.
        if self.sequence_span > 1:
            extra: list[int] = []
            for s in chosen:
                tr = ca1.get(s)
                if tr is None:
                    continue
                sess, pos = tr.key
                for d in range(1, self.sequence_span):
                    nb = ca1.slot_of((sess, pos + d))
                    if nb is not None and nb not in chosen:
                        extra.append(nb)
            chosen.extend(extra)

        # Interleaving: mix in already-consolidated traces so the slow store is
        # never fed a pure stream of novel material.
        if self.interleave > 0:
            old = [s for s in ca1.slots() if (t := ca1.get(s)) and t.consolidated]
            if old:
                n_old = min(len(old), max(1, int(round(self.batch * self.interleave))))
                chosen.extend(
                    old[int(i)] for i in self.rng.choice(len(old), size=n_old, replace=False)
                )
        return chosen

    # ------------------------------------------------------------------ #

    def utility(self, tr: Trace) -> float:
        """How much keeping this trace resident is worth.

        Eviction order, lowest first. An unconsolidated trace is effectively
        un-evictable: nothing else holds it, so dropping it is true data loss
        rather than a cache decision.
        """
        if not tr.consolidated:
            return float("inf")
        return float(tr.strength * (1.0 + tr.episode.salience) * (1.0 + 0.25 * tr.replays))

    def evict(self, ca1: CA1) -> list[tuple[int, int]]:
        """Trim the fast store to capacity, cheapest consolidated trace first."""
        if self.capacity is None or len(ca1) <= self.capacity:
            return []
        slots = ca1.slots()
        ranked = sorted(slots, key=lambda s: self.utility(ca1.get(s)))
        dropped: list[tuple[int, int]] = []
        for slot in ranked:
            if len(ca1) <= self.capacity:
                break
            tr = ca1.get(slot)
            if tr is None or not tr.consolidated:
                break  # ranked ascending: everything after this is also resident-only
            ca1.release(slot)
            dropped.append(tr.key)
        return dropped

    def ripple(self, ca1: CA1, neocortex: Neocortex, now: float) -> ConsolidationEvent:
        """Run one sharp-wave-ripple burst."""
        self.step += 1
        chosen = self._select(ca1, now)
        replayed: list[tuple[int, int]] = []
        promoted: list[tuple[int, int]] = []
        created = updated = 0

        for slot in chosen:
            tr = ca1.get(slot)
            if tr is None:
                continue
            tr.replays += 1
            # Replay is itself a rehearsal: it refreshes the trace against decay.
            tr.strength = min(1.0, tr.strength + 0.15)
            replayed.append(tr.key)

            weight = 1.0 + tr.surprise
            _, is_new = neocortex.absorb(tr.dense, tr.key, tr.episode.text, weight=weight)
            created += int(is_new)
            updated += int(not is_new)

            if not tr.consolidated and tr.replays >= self.promote_after:
                tr.consolidated = True
                promoted.append(tr.key)

        evicted = self.evict(ca1)

        _ = evicted
        ev = ConsolidationEvent(
            step=self.step,
            replayed=replayed,
            promoted=promoted,
            schemas_created=created,
            schemas_updated=updated,
            hippocampal_size=len(ca1),
            neocortical_size=len(neocortex),
        )
        self.history.append(ev)
        return ev
