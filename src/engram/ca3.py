"""CA3: autoassociative pattern completion.

CA3 is the one region in the hippocampal circuit with massive recurrent
collaterals — pyramidal cells synapse densely back onto each other. That
recurrence makes it an attractor network: present a fragment of a stored
pattern and the dynamics pull the state to the nearest stored fixed point.
Behaviourally this is what lets a partial cue recover a whole episode.

The implementation uses the *modern* Hopfield update (Ramsauer et al., 2020)
rather than the classical binary one, because the classical network stores only
~0.14N patterns before spurious minima take over, and because the modern update
is one line:

    xi <- X^T softmax(beta * X xi)

with ``X`` the stored patterns. That is exactly scaled dot-product attention
with the stored set as both keys and values, which is a useful thing to notice:
CA3 completion and a single attention head are the same operation, differing
only in whether the pattern set is learned or written. The practical
consequence is capacity — exponential in dimension rather than linear — which
is what makes it usable as an episodic store.

The inverse temperature ``beta`` is the interesting control. Low beta averages
over many stored patterns and returns a blend, which is a *metastable* state
and behaves like a prototype or gist. High beta drives the update toward the
single nearest pattern and returns a verbatim episode. That single knob is the
gist/verbatim axis, and the benchmark sweeps it.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

__all__ = ["CA3", "CompletionResult"]


class CompletionResult:
    """Outcome of settling the attractor network from one cue."""

    __slots__ = ("index", "steps", "confidence", "converged", "weights", "energy")

    def __init__(
        self,
        index: int,
        steps: int,
        confidence: float,
        converged: bool,
        weights: np.ndarray,
        energy: list[float],
    ) -> None:
        self.index = index
        self.steps = steps
        #: softmax mass on the winning pattern. Near 1 => clean verbatim recall;
        #: spread out => the network settled into a blend of several episodes.
        self.confidence = confidence
        self.converged = converged
        self.weights = weights
        self.energy = energy

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"CompletionResult(index={self.index}, steps={self.steps}, "
            f"confidence={self.confidence:.3f}, converged={self.converged})"
        )


class CA3:
    """Sparse modern-Hopfield attractor store.

    Patterns are the k-sparse DG codes. They are held in a CSR matrix because
    the population is large (thousands of units) and each pattern touches only
    ~2% of it, so the dense form would be ~50x wasteful and the sparse matvec is
    what makes retrieval cheap.

    Parameters
    ----------
    dim:
        Size of the DG population.
    beta:
        Inverse temperature. See the module docstring: this is the gist vs
        verbatim control.
    max_steps:
        Cap on settling iterations. Convergence is usually 1-2 steps; the cap
        exists to bound the pathological case rather than because it is hit.
    """

    def __init__(self, dim: int, beta: float = 22.0, max_steps: int = 8) -> None:
        self.dim = dim
        self.beta = beta
        self.max_steps = max_steps
        self._rows: list[np.ndarray] = []
        self._matrix: sparse.csr_matrix | None = None
        self._dirty = True

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._rows)

    def store(self, code: np.ndarray) -> int:
        """Write one sparse code; returns its pattern index."""
        self._rows.append(np.asarray(code, dtype=np.int32))
        self._dirty = True
        return len(self._rows) - 1

    def remove(self, index: int) -> None:
        """Forget a pattern in place, keeping indices stable.

        Used by consolidation: once an episode has been absorbed into a
        neocortical schema the hippocampal copy can be released. Indices must
        stay stable because CA1 holds references to them.
        """
        self._rows[index] = np.zeros(0, dtype=np.int32)
        self._dirty = True

    def _build(self) -> sparse.csr_matrix:
        if self._matrix is None or self._dirty:
            n = len(self._rows)
            indptr = np.zeros(n + 1, dtype=np.int64)
            for i, r in enumerate(self._rows):
                indptr[i + 1] = indptr[i] + r.size
            indices = (
                np.concatenate(self._rows) if n and indptr[-1] else np.zeros(0, dtype=np.int32)
            )
            # Unit-normalise each pattern so overlaps are cosines and beta means
            # the same thing regardless of how many units a code happens to use.
            data = np.ones(int(indptr[-1]), dtype=np.float32)
            for i, r in enumerate(self._rows):
                if r.size:
                    data[indptr[i] : indptr[i + 1]] = 1.0 / np.sqrt(r.size)
            self._matrix = sparse.csr_matrix(
                (data, indices.astype(np.int32), indptr), shape=(n, self.dim)
            )
            self._dirty = False
        return self._matrix

    # ------------------------------------------------------------------ #

    def _densify(self, code: np.ndarray) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        if code.size:
            v[code] = 1.0 / np.sqrt(code.size)
        return v

    def complete(
        self,
        cue: np.ndarray,
        k: int | None = None,
        beta: float | None = None,
        prior: np.ndarray | None = None,
    ) -> CompletionResult:
        """Settle the network from ``cue`` and report the attractor reached.

        ``cue`` is a sparse index array, typically a degraded DG code — a
        partial episode, or a query re-encoded through the same DG.

        ``prior`` optionally biases the softmax in log-space. The memory system
        uses it to inject recency and trace strength, so a stale episode does
        not win a tie against a fresh one purely on surface overlap.
        """
        beta = self.beta if beta is None else beta
        X = self._build()
        if X.shape[0] == 0:
            return CompletionResult(-1, 0, 0.0, False, np.zeros(0, np.float32), [])

        xi = self._densify(np.asarray(cue, dtype=np.int32))
        k = k or max(int(cue.size), 1)
        energy: list[float] = []
        prev: np.ndarray | None = None
        steps = 0
        w = np.zeros(X.shape[0], dtype=np.float32)

        for step in range(1, self.max_steps + 1):
            steps = step
            overlap = X @ xi  # (n_patterns,) cosine-like similarity
            logits = beta * overlap
            if prior is not None:
                logits = logits + prior
            logits -= logits.max()
            w = np.exp(logits, dtype=np.float32)
            s = w.sum()
            if s <= 0:
                break
            w /= s
            # Modern Hopfield energy, the quantity the update descends.
            energy.append(float(-np.log(s) / beta - 0.5 * float(xi @ xi)))

            recon = X.T @ w  # (dim,) weighted blend of stored patterns
            top = np.argpartition(recon, -k)[-k:]
            top = np.sort(top[recon[top] > 0]).astype(np.int32)
            if prev is not None and top.size == prev.size and np.array_equal(top, prev):
                return CompletionResult(
                    int(np.argmax(w)), steps, float(w.max()), True, w, energy
                )
            prev = top
            xi = self._densify(top)

        return CompletionResult(int(np.argmax(w)), steps, float(w.max()), False, w, energy)

    def similarity(self, cue: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
        """One-shot overlap of a cue against every stored pattern.

        Skips the settling loop. Used on the ranking path, where the full
        posterior over episodes is wanted rather than the single attractor.
        """
        X = self._build()
        if X.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        ov = X @ self._densify(np.asarray(cue, dtype=np.int32))
        return ov + prior if prior is not None else ov

    # ------------------------------------------------------------------ #

    def capacity_probe(self, corruption: float = 0.5, trials: int = 64, seed: int = 0) -> float:
        """Fraction of stored patterns recoverable from a degraded cue.

        Deletes ``corruption`` of each cue's active units and asks whether the
        network still settles on the right attractor. This is the number that
        decides how aggressively the dentate gyrus can separate: every increase
        in sparsity buys interference resistance and costs completion here.
        """
        X = self._build()
        n = X.shape[0]
        if n == 0:
            return 0.0
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=min(trials, n), replace=False)
        ok = 0
        for i in idx:
            full = self._rows[int(i)]
            if full.size == 0:
                continue
            keep = max(1, int(round(full.size * (1.0 - corruption))))
            partial = np.sort(rng.choice(full, size=keep, replace=False)).astype(np.int32)
            if self.complete(partial, k=full.size).index == int(i):
                ok += 1
        return ok / len(idx)
