"""Dentate gyrus: pattern separation.

The problem this solves is the one that breaks flat vector stores. Two episodes
that differ in a single decisive detail — *"the deploy on Tuesday failed on
auth"* vs *"the deploy on Thursday failed on billing"* — sit at cosine ~0.95 in
any sentence embedding. Top-k retrieval over that geometry cannot reliably
return the right one, and no amount of k fixes it: the correct answer and the
confusable distractor are separated by less than the noise floor.

Biology's answer is to spend representational space. The dentate gyrus receives
input from entorhinal cortex and expands it into a population roughly five
times larger, while holding activity extremely sparse — on the order of 1-2% of
granule cells active for a given input. Expansion plus sparsity is a
decorrelating transform: it pushes two highly similar inputs onto nearly
disjoint sets of active units.

The engineering version is a fixed random expansion followed by
k-winners-take-all. What makes it work is not the randomness but the
combination: expansion raises the dimension so that random codes are
near-orthogonal, and k-WTA turns small differences in the dense input into
discrete, all-or-nothing differences in *which* units win.

Two refinements were implemented. Only one survived measurement.

``neurogenesis`` (kept, on by default)
    Adult-born granule cells are transiently hyper-excitable and are recruited
    preferentially by novel input. Modelled as a per-unit excitability gain
    that decays with use, biasing novel episodes toward under-used units. This
    is strongly load-bearing: over a 600-episode stream of confusable text it
    drops mean pairwise code overlap from 0.156 to 0.016 and raises population
    coverage from 33% to 86% (``bench/ablate_dg.py``). Without it the code
    collapses onto a hot minority of units — usage Gini 0.91 — and the
    expansion is mostly wasted.

``lateral inhibition`` (implemented, measured, defaulted off)
    Winners were made to suppress units correlated with them. It does not do
    what it was added to do, and the reason is worth stating rather than hiding
    behind a knob: for a random sparse projection the columns are already
    near-orthogonal. The measured off-diagonal Gram mean is 0.020 against an
    activation standard deviation of 1.00, so the suppression term at
    inhibition 0.25 is roughly 200x below the activation scale. It is not
    literally inert — it perturbs about 12% of the selected units (mean Jaccard
    0.88 against the uninhibited code) — but it buys no separation: mean
    pairwise code overlap is 0.2229 without it and 0.2255 with it, marginally
    worse. Retained behind a default-zero parameter because it becomes relevant
    if ``W`` is ever learned rather than sampled, at which point column
    correlation is real. Measured in ``bench/ablate_dg.py``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DentateGyrus", "separation_gain"]


class DentateGyrus:
    """Sparse expansion coder.

    Parameters
    ----------
    dim_in:
        Entorhinal input dimension.
    expansion:
        Output population is ``dim_in * expansion``. Biological DG/EC ratio is
        roughly 5:1 in rodent; 5-10 works well here and the benchmark sweeps it.
    sparsity:
        Fraction of units allowed to win. 0.02 means a 2% active code.
    neurogenesis:
        Strength of the novelty bias toward under-used units. 0 disables it.
        Defaults to 0.7; see the module docstring for the measured effect.
    inhibition:
        Lateral inhibition between correlated winners. Defaults to 0 because it
        is a measured no-op for random projections; see module docstring.
    """

    def __init__(
        self,
        dim_in: int,
        expansion: int = 8,
        sparsity: float = 0.02,
        neurogenesis: float = 0.7,
        inhibition: float = 0.0,
        seed: int = 0,
    ) -> None:
        if not 0 < sparsity < 1:
            raise ValueError("sparsity must lie in (0, 1)")
        self.dim_in = dim_in
        self.dim_out = dim_in * expansion
        self.k = max(1, int(round(self.dim_out * sparsity)))
        self.neurogenesis = neurogenesis
        self.inhibition = inhibition

        rng = np.random.default_rng(seed)
        # Sparse random projection (Achlioptas): each weight is +1/-1 with
        # probability 1/(2s) and 0 otherwise. Cheaper than dense Gaussian and
        # preserves distances just as well at this scale.
        s = 3.0
        density = 1.0 / s
        w = np.zeros((dim_in, self.dim_out), dtype=np.float32)
        mask = rng.random((dim_in, self.dim_out)) < density
        signs = rng.integers(0, 2, size=(dim_in, self.dim_out)).astype(np.float32) * 2 - 1
        w[mask] = signs[mask] * np.float32(np.sqrt(s))
        self.W = w

        # Column norms, needed by lateral inhibition; cached once.
        self._wnorm = np.maximum(np.linalg.norm(self.W, axis=0), 1e-6).astype(np.float32)
        # Gram matrix of the projection, built lazily and only when inhibition
        # is on. dim_out^2 floats (~38 MB at dim_out=3072) bought once, versus a
        # dim_in x dim_out matmul per winner per episode.
        self._gram: np.ndarray | None = None

        # Usage counter drives the neurogenesis bias: rarely-used units stay
        # excitable, heavily-used units habituate.
        self._usage = np.zeros(self.dim_out, dtype=np.float32)
        self._n_encoded = 0

    def _gram_matrix(self) -> np.ndarray:
        if self._gram is None:
            wn = self.W / self._wnorm
            g = (wn.T @ wn).astype(np.float32)
            np.maximum(g, 0.0, out=g)  # only positive correlation inhibits
            np.fill_diagonal(g, 0.0)
            self._gram = g
        return self._gram

    # ------------------------------------------------------------------ #

    def _excitability(self) -> np.ndarray:
        """Per-unit gain in [1-g, 1+g], high for under-used units."""
        if self.neurogenesis <= 0 or self._n_encoded == 0:
            return np.ones(self.dim_out, dtype=np.float32)
        mean_use = self._usage.mean()
        if mean_use <= 0:
            return np.ones(self.dim_out, dtype=np.float32)
        # Relative under-use, squashed to keep the bias bounded.
        rel = (mean_use - self._usage) / (mean_use + 1e-6)
        return (1.0 + self.neurogenesis * np.tanh(rel)).astype(np.float32)

    def _kwta(self, act: np.ndarray) -> np.ndarray:
        """Top-k selection with optional lateral inhibition between winners.

        Plain top-k can select a block of units that all read from the same
        correlated slice of the projection, which wastes the expansion. With
        inhibition on, each selected winner discounts its correlated neighbours
        before the next is chosen, so the code spreads across the population.
        """
        if self.inhibition <= 0:
            idx = np.argpartition(act, -self.k)[-self.k :]
            return np.sort(idx).astype(np.int32)

        # Greedy selection with suppression. k is small (tens of units), so the
        # loop is a few thousand flops against a cached Gram row.
        gram = self._gram_matrix()
        work = act.copy()
        chosen: list[int] = []
        for _ in range(self.k):
            j = int(np.argmax(work))
            chosen.append(j)
            work[j] = -np.inf
            work -= self.inhibition * gram[j]
        return np.sort(np.asarray(chosen, dtype=np.int32))

    def encode(self, dense: np.ndarray, learn: bool = True) -> np.ndarray:
        """Return the indices of the active DG units for one dense input."""
        act = dense.astype(np.float32) @ self.W
        act *= self._excitability()
        idx = self._kwta(act)
        if learn:
            self._usage[idx] += 1.0
            self._n_encoded += 1
        return idx

    def encode_batch(self, dense: np.ndarray, learn: bool = False) -> list[np.ndarray]:
        return [self.encode(row, learn=learn) for row in np.atleast_2d(dense)]

    # ------------------------------------------------------------------ #

    @staticmethod
    def overlap(a: np.ndarray, b: np.ndarray) -> float:
        """Jaccard overlap of two sparse codes — the separation read-out."""
        if a.size == 0 or b.size == 0:
            return 0.0
        inter = np.intersect1d(a, b, assume_unique=True).size
        union = a.size + b.size - inter
        return inter / union if union else 0.0

    def stats(self) -> dict[str, float]:
        used = float((self._usage > 0).sum())
        return {
            "dim_out": float(self.dim_out),
            "k": float(self.k),
            "sparsity": self.k / self.dim_out,
            "units_used": used,
            "unit_coverage": used / self.dim_out,
            # Gini of usage: 0 = perfectly even recruitment, 1 = total collapse
            # onto a few units. This is the number that tells you whether
            # neurogenesis is actually doing anything.
            "usage_gini": _gini(self._usage),
        }


def _gini(x: np.ndarray) -> float:
    if x.size == 0 or x.sum() == 0:
        return 0.0
    v = np.sort(x.astype(np.float64))
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def separation_gain(dg: DentateGyrus, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Measure how much separation the DG buys on one pair of inputs.

    Returns the input cosine, the output overlap, and the ratio. A value well
    below 1 is the whole point: highly similar inputs should emerge as codes
    that barely intersect.
    """
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    ca, cb = dg.encode(a, learn=False), dg.encode(b, learn=False)
    ov = DentateGyrus.overlap(ca, cb)
    return {"input_cosine": cos, "output_overlap": ov, "ratio": ov / cos if cos > 1e-6 else 0.0}
