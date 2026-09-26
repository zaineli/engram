"""CA3 attractor capacity: how much of a cue can be destroyed and still recover.

The modern Hopfield update has capacity exponential in dimension rather than the
classical ~0.14N, and that is the property which makes it usable as an episodic
store rather than a toy. This measures it directly: delete a fraction of a
stored code's active units and ask whether the network settles back on the right
attractor.

    uv run --extra bench python bench/ca3_capacity.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import build_encoder  # noqa: E402

from engram.ca3 import CA3  # noqa: E402
from engram.dentate import DentateGyrus  # noqa: E402


def corpus(n: int) -> list[str]:
    return [
        f"On day {i // 4} the {['auth', 'billing', 'search', 'cache'][i % 4]} service "
        f"reported {['a timeout', 'a 5xx spike', 'latency', 'an oom'][(i // 4) % 4]} "
        f"lasting {i % 97} minutes in region {i % 13}."
        for i in range(n)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max-n", type=int, default=4000)
    args = ap.parse_args()
    name = "hashing" if args.fast else "minilm"
    enc = build_encoder(name)
    V = enc.encode(corpus(args.max_n))
    print(f"encoder={name}\n")

    print("capacity vs store size, 90% of cue units deleted")
    print(f"  {'n':>6} {'accuracy':>9} {'ms/query':>9}")
    for n in (400, 1000, 2000, args.max_n):
        dg = DentateGyrus(enc.dim, expansion=8, seed=2)
        ca3 = CA3(dg.dim_out, beta=22.0)
        for v in V[:n]:
            ca3.store(dg.encode(v))
        t = time.perf_counter()
        acc = ca3.capacity_probe(0.90, trials=150)
        dt = (time.perf_counter() - t) / 150 * 1000
        print(f"  {n:>6} {acc:>9.3f} {dt:>9.2f}")

    print("\ndegradation curve at n=2000 (sparsity 0.02, k=61)")
    dg = DentateGyrus(enc.dim, expansion=8, seed=2)
    ca3 = CA3(dg.dim_out, beta=22.0)
    for v in V[:2000]:
        ca3.store(dg.encode(v))
    print(f"  {'deleted':>8} {'units kept':>11} {'accuracy':>9}")
    for c in (0.50, 0.80, 0.90, 0.95, 0.97, 0.98):
        kept = max(1, int(round(dg.k * (1 - c))))
        print(f"  {c:>7.0%} {kept:>11} {ca3.capacity_probe(c, trials=150):>9.3f}")

    print("\nsparsity vs completion (n=2000, 90% deleted)")
    print(f"  {'sparsity':>9} {'k':>5} {'accuracy':>9}")
    for sp in (0.005, 0.01, 0.02, 0.05, 0.10):
        d = DentateGyrus(enc.dim, expansion=8, sparsity=sp, seed=2)
        c3 = CA3(d.dim_out, beta=22.0)
        for v in V[:2000]:
            c3.store(d.encode(v))
        print(f"  {sp:>9.3f} {d.k:>5} {c3.capacity_probe(0.90, trials=100):>9.3f}")


if __name__ == "__main__":
    main()
