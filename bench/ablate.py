"""Pathway ablation: which mechanism is actually carrying which task.

Every pathway is a weight in :class:`~engram.memory.EngramConfig`, so ablation
is exact rather than approximate — zeroing ``w_conjunctive`` removes the
conjunction pathway and changes nothing else. A pathway that does not move a
number when removed is not a mechanism; it is a decoration, and the honest
thing is to find that out and say so (see ``lateral inhibition`` in
:mod:`engram.dentate`, which is exactly that and is documented as such).

    uv run --extra bench python bench/ablate.py --fast
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate import build_corpus  # noqa: E402
from paraphrase import paraphrase_corpus  # noqa: E402
from run import TASKS, build_encoder, evaluate  # noqa: E402

from engram import EngramConfig, EngramMemory  # noqa: E402

ABLATIONS: dict[str, dict] = {
    "full": {},
    "-semantic": {"w_semantic": 0.0},
    "-conjunctive": {"w_conjunctive": 0.0},
    "-temporal-ctx": {"w_temporal": 0.0},
    "-reinstatement": {"w_reinstate": 0.0},
    "-schema": {"w_schema": 0.0},
    "-neurogenesis": {"neurogenesis": 0.0},
    "-sleep": {"__no_sleep__": True},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--people", type=int, default=16)
    ap.add_argument("--events", type=int, default=50)
    ap.add_argument("--paraphrase", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep-cycles", type=int, default=40)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    corpus = build_corpus(n_people=args.people, events_per_person=args.events, seed=args.seed)
    if args.paraphrase:
        corpus = paraphrase_corpus(corpus, seed=args.seed)
    encoder, enc_name = build_encoder(args.fast)
    mode = "paraphrased" if args.paraphrase else "token-literal"
    print(f"{len(corpus.episodes)} episodes  {len(corpus.queries)} queries  "
          f"encoder={enc_name}  queries={mode}\n")

    results: dict[str, dict] = {}
    for name, overrides in ABLATIONS.items():
        no_sleep = overrides.pop("__no_sleep__", False)
        cfg = replace(EngramConfig(seed=args.seed), **overrides)
        mem = EngramMemory(encoder=encoder, config=cfg)
        for text, key in zip(corpus.episodes, corpus.keys):
            mem.remember(text, session=key[0])
        if not no_sleep:
            mem.sleep(cycles=args.sleep_cycles)
        results[name] = evaluate(
            lambda q, k: [(r.episode.text, r.score, r.episode.key()) for r in mem.recall(q, k=k)],
            corpus, 5,
        )
        print(f"  {name:16s} overall top1={results[name]['overall']['top1']:.3f}")

    base = results["full"]
    hdr = f"{'ablation':<16}" + "".join(f"{t:>14}" for t in TASKS) + f"{'overall':>10}{'delta':>9}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        row = f"{name:<16}" + "".join(f"{r[t]['top1']:>14.3f}" for t in TASKS)
        d = r["overall"]["top1"] - base["overall"]["top1"]
        row += f"{r['overall']['top1']:>10.3f}" + (f"{d:>+9.3f}" if name != "full" else f"{'--':>9}")
        print(row)
    print("=" * len(hdr))

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"encoder": enc_name, "query_mode": mode, "results": results}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
