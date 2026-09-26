"""What the slow store holds for one person, and why ranking uses mass.

Builds the synthetic corpus, consolidates it exactly as ``bench/run.py`` does,
and prints each schema whose first exemplar names the person, with its centroid
similarity to *"Which project does {person} usually work on?"* and its mass
(the replay-weighted evidence it absorbed).

    python bench/schemas.py [--person Priya --seed 0 --encoder minilm]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate import build_corpus  # noqa: E402
from run import build_encoder, build_engram  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="Priya")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder", default="minilm")
    ap.add_argument("--sleep-cycles", type=int, default=40)
    args = ap.parse_args()

    enc = build_encoder(args.encoder)
    corpus = build_corpus(seed=args.seed)
    mem = build_engram(corpus, enc, args.seed, args.sleep_cycles)
    qenc = getattr(enc, "encode_query", enc.encode)
    q = qenc([f"Which project does {args.person} usually work on?"])[0]

    nc = mem.neocortex
    sims = nc._stack() @ q
    mine = [i for i, s in enumerate(nc.schemas) if args.person in s.exemplars[0]]
    projects = Counter(f["project"] for f in corpus.facts if f["person"] == args.person)
    print(f"{len(nc)} schemas in the store; {len(mine)} whose first exemplar names {args.person}")
    print(f"{args.person}'s projects: {dict(projects.most_common())}\n")
    print(f"{'similarity':>10} {'mass':>7}  first exemplar")
    for i in sorted(mine, key=lambda i: -float(sims[i])):
        s = nc.schemas[i]
        print(f"{float(sims[i]):>10.3f} {s.mass:>7.1f}  {s.exemplars[0]}")
    lo, hi = min(float(sims[i]) for i in mine), max(float(sims[i]) for i in mine)
    masses = sorted(nc.schemas[i].mass for i in mine)
    print(f"\nsimilarity {lo:.3f}-{hi:.3f}; mass {masses[0]:.1f}-{masses[-1]:.1f}")
    top = nc.query(q, top_k=1)
    if top:
        print(f"slow-store answer: {top[0].episode.text}")


if __name__ == "__main__":
    main()
