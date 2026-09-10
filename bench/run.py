"""Benchmark harness: engram against flat-vector, BM25, hybrid RAG, recency.

Scoring is answer-level, not key-level. A system that returns a *different*
episode which nonetheless contains the right project has answered the question,
and penalising it would flatter engram on the abstraction task, where the right
answer is a fact that lives in no single episode.

Usage
-----
    uv run --extra bench python bench/run.py                  # full sweep
    uv run --extra bench python bench/run.py --fast           # hashing encoder
    uv run --extra bench python bench/run.py --out results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import BM25, FlatVectorStore, HybridRAG, RecencyWindow  # noqa: E402
from generate import Corpus, build_corpus  # noqa: E402
from paraphrase import paraphrase_corpus  # noqa: E402

from engram import EngramConfig, EngramMemory, HashingEncoder  # noqa: E402

TASKS = ("episodic", "interference", "abstraction", "temporal")


def _hit(text: str, needle: str) -> bool:
    return bool(needle) and re.search(rf"\b{re.escape(needle)}\b", text, re.I) is not None


def evaluate(search, corpus: Corpus, k: int = 5) -> dict:
    per_task = {t: {"n": 0, "top1": 0, "hit@k": 0, "mrr": 0.0} for t in TASKS}
    t0 = time.perf_counter()
    for q in corpus.queries:
        res = search(q.text, k)
        acc = per_task[q.task]
        acc["n"] += 1
        if res and _hit(res[0][0], q.answer_contains):
            acc["top1"] += 1
        rank = next((i for i, (txt, _, _) in enumerate(res) if _hit(txt, q.answer_contains)), None)
        if rank is not None:
            acc["hit@k"] += 1
            acc["mrr"] += 1.0 / (rank + 1)
    elapsed = time.perf_counter() - t0

    out: dict = {}
    tot = {"n": 0, "top1": 0, "hit@k": 0, "mrr": 0.0}
    for t, a in per_task.items():
        n = max(a["n"], 1)
        out[t] = {"n": a["n"], "top1": a["top1"] / n, "hit@k": a["hit@k"] / n, "mrr": a["mrr"] / n}
        for key in tot:
            tot[key] += a[key]
    n = max(tot["n"], 1)
    out["overall"] = {
        "n": tot["n"], "top1": tot["top1"] / n,
        "hit@k": tot["hit@k"] / n, "mrr": tot["mrr"] / n,
    }
    out["query_ms"] = 1000 * elapsed / n
    return out


def build_encoder(fast: bool):
    if fast:
        return HashingEncoder(dim=384), "hashing"
    from engram import SentenceTransformerEncoder

    return SentenceTransformerEncoder(), "all-MiniLM-L6-v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="hashing encoder, no model download")
    ap.add_argument("--people", type=int, default=16)
    ap.add_argument("--events", type=int, default=50, help="events per person timeline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sleep-cycles", type=int, default=40)
    ap.add_argument("--paraphrase", action="store_true",
                    help="rewrite queries to break literal token overlap (realistic setting)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    corpus = build_corpus(n_people=args.people, events_per_person=args.events, seed=args.seed)
    if args.paraphrase:
        corpus = paraphrase_corpus(corpus, seed=args.seed)
    encoder, enc_name = build_encoder(args.fast)
    mode = "paraphrased" if args.paraphrase else "token-literal"
    print(f"corpus: {len(corpus.episodes)} episodes  {len(corpus.queries)} queries  "
          f"encoder={enc_name}  queries={mode}\n")

    results: dict[str, dict] = {}
    timings: dict[str, float] = {}

    for name, ctor in [
        ("recency", lambda: RecencyWindow(window=50)),
        ("bm25", lambda: BM25()),
        ("flat-vector", lambda: FlatVectorStore(encoder)),
        ("hybrid-rag", lambda: HybridRAG(encoder)),
    ]:
        sysobj = ctor()
        t0 = time.perf_counter()
        sysobj.add_many(corpus.episodes, corpus.keys)
        timings[name] = time.perf_counter() - t0
        results[name] = evaluate(sysobj.search, corpus, args.k)
        print(f"  {name:14s} ingest {timings[name]:6.2f}s  top1={results[name]['overall']['top1']:.3f}")

    mem = EngramMemory(encoder=encoder, config=EngramConfig(seed=args.seed))
    t0 = time.perf_counter()
    for text, key in zip(corpus.episodes, corpus.keys):
        mem.remember(text, session=key[0])
    mem.sleep(cycles=args.sleep_cycles)
    timings["engram"] = time.perf_counter() - t0

    def engram_search(q: str, k: int):
        return [(r.episode.text, r.score, r.episode.key()) for r in mem.recall(q, k=k)]

    results["engram"] = evaluate(engram_search, corpus, args.k)
    print(f"  {'engram':14s} ingest {timings['engram']:6.2f}s  top1={results['engram']['overall']['top1']:.3f}")

    names = ["recency", "bm25", "flat-vector", "hybrid-rag", "engram"]
    print("\n" + "=" * 78)
    print(f"{'top-1 accuracy':<16}" + "".join(f"{n:>13}" for n in names))
    print("-" * 78)
    for t in TASKS + ("overall",):
        print(f"{t:<16}" + "".join(f"{results[n][t]['top1']:>13.3f}" for n in names))
    print("-" * 78)
    print(f"{'MRR (overall)':<16}" + "".join(f"{results[n]['overall']['mrr']:>13.3f}" for n in names))
    print(f"{'query ms':<16}" + "".join(f"{results[n]['query_ms']:>13.2f}" for n in names))
    print(f"{'ingest s':<16}" + "".join(f"{timings[n]:>13.2f}" for n in names))
    print("=" * 78)

    st = mem.stats()
    print(f"\nengram store: hippocampal={st['hippocampal']:.0f} schemas={st['nc_schemas']:.0f} "
          f"compression={st['nc_compression']:.1f} dg_coverage={st['dg_unit_coverage']:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "config": vars(args), "encoder": enc_name, "query_mode": mode,
            "n_episodes": len(corpus.episodes), "n_queries": len(corpus.queries),
            "results": results, "timings": timings, "engram_stats": st,
        }, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
