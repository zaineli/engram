"""Paired statistics for comparing two systems on the same queries.

Every comparison in this repository is paired: both systems answer the same
questions, so the unit of analysis is the per-question difference. Two tools:

``paired_bootstrap``
    Percentile interval for the mean difference, resampling questions with
    replacement. Answers "how big is the gap, and how sure are we".

``paired_permutation``
    Two-sided sign-flip randomisation test of "no difference". Under the null
    the two systems' labels are exchangeable within a question, so each
    difference is equally likely to have either sign. Smucker, Allan &
    Carterette (CIKM 2007) compared the tests IR papers use and recommend this
    one, with the bootstrap and the t-test close behind; the sign and Wilcoxon
    tests they found unreliable.

Both are seeded, so a rerun reproduces the same interval and p-value exactly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["paired_bootstrap", "paired_permutation", "compare", "mean_ci", "cluster_bootstrap"]


def _diff(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired samples must be 1-d and the same length")
    return a - b


def paired_bootstrap(a, b, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean of ``a - b`` and its (1 - alpha) percentile interval."""
    d = _diff(a, b)
    if d.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def paired_permutation(a, b, n_perm: int = 10_000, seed: int = 0) -> float:
    """Two-sided p-value for mean(a - b) = 0 by random sign flips.

    The observed statistic is counted as one of the permutations, so the
    smallest reportable p is ``1 / (n_perm + 1)`` rather than zero.
    """
    d = _diff(a, b)
    if d.size == 0 or not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.count_nonzero(null >= obs - 1e-12)) / (n_perm + 1))


def mean_ci(a, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean of one sample and its percentile bootstrap interval."""
    a = np.asarray(a, dtype=np.float64)
    return paired_bootstrap(a, np.zeros_like(a), n_boot=n_boot, alpha=alpha, seed=seed)


def compare(a, b, **kw) -> dict[str, float]:
    """Everything a results table needs for one paired comparison."""
    mean, lo, hi = paired_bootstrap(a, b, seed=kw.get("seed", 0))
    return {
        "a": float(np.mean(a)),
        "b": float(np.mean(b)),
        "diff": mean,
        "lo": lo,
        "hi": hi,
        "p": paired_permutation(a, b, seed=kw.get("seed", 0)),
        "n": int(len(a)),
    }


def cluster_bootstrap(a, b, clusters, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Paired interval for mean(a - b) resampling whole clusters, not items.

    When questions come in groups that share a corpus (LoCoMo has ten
    conversations), resampling questions treats correlated items as
    independent and understates the uncertainty. Resampling the groups keeps
    whatever a conversation does to every system inside one draw.
    """
    d = _diff(a, b)
    clusters = np.asarray(clusters)
    ids = np.unique(clusters)
    sums = np.array([d[clusters == c].sum() for c in ids])
    counts = np.array([np.count_nonzero(clusters == c) for c in ids])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, ids.size, size=(n_boot, ids.size))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)
