"""Slot confusion, measured: the experiment that motivated conjunctive binding.

125 incident reports, every combination of 5 days x 5 services x 5 faults, in
one frame: *"On {day} the {service} service suffered {fault} during the
rollout."* Each query names two of the three slots, *"What went wrong with the
{service} service on {day}?"*, for one report chosen as its target.

Two scores, because the query set is ambiguous by design:

``class@1``
    The top result has the right service *and* day. This is what the query
    can specify. Ceiling 1.0.
``key@1``
    The top result is the target report itself. Five reports share the named
    pair and differ only in the fault the question does not mention, so the
    ceiling is 0.2, and the best possible MRR is that of a uniform draw among
    five tied items, (1 + 1/2 + 1/3 + 1/4 + 1/5) / 5 = 0.457.

Pathways compared: dense cosine, the dentate expansion of that same vector
(DG code overlap), and the conjunction code, each ranking alone.

v0.1's README quoted numbers for this experiment (0.040 / 0.056 / 0.200 key@1)
without the script that produced them. This is that experiment written down;
its numbers replace those.

    python bench/incidents.py [--encoder minilm] [--out bench/results/incidents.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoders import load_encoder  # noqa: E402

from engram import ConjunctionIndex, ConjunctiveBinder, DentateGyrus  # noqa: E402

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SERVICES = ["auth", "billing", "search", "payments", "gateway"]
FAULTS = ["a timeout cascade", "a memory leak", "a certificate expiry", "a disk-full outage", "a DNS failure"]


def corpus():
    rows = list(itertools.product(DAYS, SERVICES, FAULTS))
    texts = [f"On {d} the {s} service suffered {f} during the rollout." for d, s, f in rows]
    queries = [(f"What went wrong with the {s} service on {d}?", i) for i, (d, s, _) in enumerate(rows)]
    return rows, texts, queries


def evaluate(scores: np.ndarray, rows, target: int) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    d, s, _ = rows[target]
    top = rows[int(order[0])]
    rank = int(np.where(order == target)[0][0])
    return {"key@1": float(order[0] == target), "key_rr": 1.0 / (rank + 1),
            "class@1": float(top[0] == d and top[1] == s)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="minilm")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows, texts, queries = corpus()
    enc = load_encoder(args.encoder)
    E = enc.encode(texts)
    qenc = getattr(enc, "encode_query", enc.encode)
    Q = qenc([q for q, _ in queries])
    if hasattr(enc, "save"):
        enc.save()

    G = E @ E.T
    off = G[~np.eye(len(texts), dtype=bool)]
    print(f"{len(texts)} reports; pairwise cosine mean {off.mean():.3f}, max {off.max():.3f}")

    dg = DentateGyrus(E.shape[1], expansion=8, sparsity=0.02, neurogenesis=0.7, seed=0)
    codes = [dg.encode(v, learn=True) for v in E]
    binder = ConjunctiveBinder()
    idx = ConjunctionIndex()
    for i, t in enumerate(texts):
        idx.add(i, binder.encode(t))

    out: dict[str, dict[str, float]] = {}
    for name in ("dense cosine", "dentate expansion", "conjunctive binding"):
        acc = {"key@1": 0.0, "key_rr": 0.0, "class@1": 0.0}
        for (q, target), qv in zip(queries, Q):
            if name == "dense cosine":
                s = E @ qv
            elif name == "dentate expansion":
                qc = dg.encode(qv, learn=False)
                s = np.array([DentateGyrus.overlap(qc, c) for c in codes])
            else:
                keys, sc = idx.score(binder.encode(q))
                s = np.empty(len(texts))
                s[keys] = sc
            for m, v in evaluate(s, rows, target).items():
                acc[m] += v
        out[name] = {m: v / len(queries) for m, v in acc.items()}

    print(f"\n{'pathway':<22}{'class@1':>9}{'key@1':>8}{'key MRR':>9}   (ceilings 1.000 / 0.200 / 0.457)")
    for name, m in out.items():
        print(f"{name:<22}{m['class@1']:>9.3f}{m['key@1']:>8.3f}{m['key_rr']:>9.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "encoder": args.encoder, "n": len(texts),
            "cosine_mean": float(off.mean()), "cosine_max": float(off.max()), "results": out,
        }, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
