"""Pathway ablation: which mechanism is actually carrying which task.

Every pathway is a weight in :class:`~engram.memory.EngramConfig`, so ablation
is exact rather than approximate — zeroing ``w_conjunctive`` removes the
conjunction pathway and changes nothing else. A pathway that does not move a
number when removed is not a mechanism; it is a decoration, and the honest
thing is to find that out and say so (see ``lateral inhibition`` in
:mod:`engram.dentate`, which is exactly that and is documented as such).

Scores are the primary scores of ``bench/run.py`` (key-level where a query has
a key), pooled over seeds, with each ablation compared to the full system on
the same queries by a paired bootstrap interval and randomisation test.

    python bench/ablate.py --fast
    python bench/ablate.py --encoder minilm --seeds 5 [--paraphrase]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import TASKS, build_encoder, build_engram, engram_search, evaluate, make_corpus, primary  # noqa: E402
from stats import compare  # noqa: E402

ABLATIONS: dict[str, dict] = {
    "full": {},
    "-semantic": {"w_semantic": 0.0},
    "-conjunctive": {"w_conjunctive": 0.0},
    "conj: all pairs": {"conj_window": None},
    "conj: unigrams": {"conj_order": 1},
    "conj: binary": {"conj_weighting": "binary"},
    "fusion: minmax": {"fusion": "minmax"},
    "fusion: rrf": {"fusion": "rrf"},
    "-temporal-ctx": {"w_temporal": 0.0},
    "-reinstatement": {"w_reinstate": 0.0},
    "-schema": {"w_schema": 0.0},
    "-neurogenesis": {"neurogenesis": 0.0},
    "-sleep": {"__no_sleep__": True},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--encoder", default="minilm")
    ap.add_argument("--people", type=int, default=16)
    ap.add_argument("--events", type=int, default=50)
    ap.add_argument("--paraphrase", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sleep-cycles", type=int, default=40)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    enc_name = "hashing" if args.fast else args.encoder
    encoder = build_encoder(enc_name)
    mode = "paraphrased" if args.paraphrase else "token-literal"
    print(f"seeds 0..{args.seeds - 1}  encoder={enc_name}  queries={mode}\n")

    rows: dict[str, list[dict]] = {name: [] for name in ABLATIONS}
    for seed in range(args.seeds):
        corpus = make_corpus(seed, args)
        for name, overrides in ABLATIONS.items():
            ov = dict(overrides)
            no_sleep = ov.pop("__no_sleep__", False)
            mem = build_engram(corpus, encoder, seed, 0 if no_sleep else args.sleep_cycles, ov)
            rows[name].extend(evaluate(engram_search(mem), corpus, args.k))
    if hasattr(encoder, "save"):
        encoder.save()

    def prim(name: str, task: str | None = None) -> list[float]:
        return [r[primary(r["task"])] for r in rows[name] if task is None or r["task"] == task]

    results: dict[str, dict] = {}
    for name in ABLATIONS:
        results[name] = {t: float(np.mean(prim(name, t))) for t in TASKS}
        results[name]["overall"] = float(np.mean(prim(name)))
        if name != "full":
            results[name]["vs_full"] = compare(prim(name), prim("full"))
            results[name]["vs_full_by_task"] = {t: compare(prim(name, t), prim("full", t)) for t in TASKS}

    hdr = f"{'ablation':<18}" + "".join(f"{t:>14}" for t in TASKS) + f"{'overall':>10}{'delta':>9}  95% CI"
    print("=" * (len(hdr) + 16))
    print(hdr)
    print("-" * (len(hdr) + 16))
    for name, r in results.items():
        row = f"{name:<18}" + "".join(f"{r[t]:>14.3f}" for t in TASKS) + f"{r['overall']:>10.3f}"
        if name != "full":
            c = r["vs_full"]
            row += f"{c['diff']:>+9.3f}  [{c['lo']:+.3f}, {c['hi']:+.3f}]"
        print(row)
    print("=" * (len(hdr) + 16))

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"encoder": enc_name, "query_mode": mode, "seeds": args.seeds, "results": results}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
