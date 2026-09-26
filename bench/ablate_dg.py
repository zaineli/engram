"""Dentate gyrus: does sparse expansion coding actually separate?

Two claims are tested, both storage-side. Neither shows up in retrieval
accuracy on ``bench/run.py``, which is why they are measured here instead of
being asserted in a docstring: that benchmark never fills the store to the point
where code collisions matter, and a mechanism that does not move the number you
happen to be reporting is not thereby useless — it is unmeasured.

1. Expansion + k-winners-take-all decorrelates similar inputs.
2. Neurogenesis (novelty-biased excitability) keeps the population from
   collapsing onto a hot minority of units as the store fills.

    uv run --extra bench python bench/ablate_dg.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import build_encoder  # noqa: E402

from engram.dentate import DentateGyrus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()
    name = "hashing" if args.fast else "minilm"
    enc = build_encoder(name)

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    svcs = ["auth", "billing", "search", "cache", "ingest"]
    faults = ["a timeout cascade", "a 5xx spike", "a memory leak", "a cert expiry", "a replica lag"]
    texts = [
        f"On {d} the {s} service suffered {f} during the rollout."
        for d, s, f in itertools.islice(itertools.product(days, svcs, faults), args.n)
    ]
    while len(texts) < args.n:
        texts.extend(texts[: args.n - len(texts)])
    V = enc.encode(texts)
    print(f"encoder={name}  n={len(texts)}")

    cs = V @ V.T
    np.fill_diagonal(cs, 0.0)
    print(f"input space: mean pairwise cosine={cs.mean():.3f}  max={cs.max():.3f}\n")

    print("separation: input cosine -> DG code overlap")
    dg = DentateGyrus(enc.dim, expansion=8, sparsity=0.02, neurogenesis=0.0)
    for i, j in [(0, 1), (0, 5), (0, 25), (0, len(texts) - 1)]:
        a, b = V[i], V[j]
        ca, cb = dg.encode(a, learn=False), dg.encode(b, learn=False)
        print(f"  cos={float(a @ b):.3f} -> overlap={DentateGyrus.overlap(ca, cb):.3f}")

    print("\nneurogenesis sweep (whole stream encoded, then sampled pairs)")
    print(f"  {'ng':>5} {'mean ovl':>9} {'p95 ovl':>9} {'coverage':>9} {'gini':>7}")
    for ng in (0.0, 0.35, 0.7, 1.0):
        d = DentateGyrus(enc.dim, expansion=8, sparsity=0.02, neurogenesis=ng, seed=1)
        codes = [d.encode(v) for v in V]
        ov = [
            DentateGyrus.overlap(codes[i], codes[j])
            for i in range(0, len(codes), 7)
            for j in range(i + 1, len(codes), 11)
        ]
        st = d.stats()
        print(f"  {ng:>5.2f} {np.mean(ov):>9.4f} {np.percentile(ov, 95):>9.4f} "
              f"{st['unit_coverage']:>9.3f} {st['usage_gini']:>7.3f}")

    print("\nlateral inhibition (documented no-op for random projections)")
    d = DentateGyrus(enc.dim, expansion=8, inhibition=0.25)
    g = d._gram_matrix()
    off = g[np.triu_indices_from(g, k=1)]
    act = V[0] @ d.W
    print(f"  Gram off-diagonal mean={off.mean():.5f}   activation sd={act.std():.4f}")
    print(f"  suppression at inhibition=0.25 is {0.25 * off.mean():.5f}, "
          f"~{act.std() / max(0.25 * off.mean(), 1e-9):.0f}x below the activation scale")
    off_dg = DentateGyrus(enc.dim, expansion=8, inhibition=0.0, seed=3)
    on_dg = DentateGyrus(enc.dim, expansion=8, inhibition=0.25, seed=3)
    agree = [
        DentateGyrus.overlap(off_dg.encode(v, learn=False), on_dg.encode(v, learn=False))
        for v in V[:100]
    ]
    print(f"  code agreement on/off: mean Jaccard={np.mean(agree):.4f} "
          f"(min={np.min(agree):.4f}) over 100 inputs")

    # What matters is not whether a few units differ but whether separation
    # changes. It does not.
    def mean_sep(d: DentateGyrus) -> float:
        codes = [d.encode(v, learn=False) for v in V[:200]]
        return float(np.mean([
            DentateGyrus.overlap(codes[i], codes[j])
            for i in range(0, 200, 5) for j in range(i + 1, 200, 7)
        ]))

    print(f"  mean pairwise code overlap: inhibition off={mean_sep(off_dg):.4f} "
          f"on={mean_sep(on_dg):.4f}")


if __name__ == "__main__":
    main()
