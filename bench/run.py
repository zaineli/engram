"""Benchmark harness: engram against flat-vector, BM25, hybrid RAG, recency.

Two scores per query, and the difference between them is a finding.

``key``
    The top result is the episode the query is about (its ``answer_key``). This
    is what "episodic recall" means, and it is the primary score for the
    episodic, interference and temporal tasks.

``answer``
    The top result *mentions* the right project. v0.1 reported only this, and
    it is inflated by the generator's own prior: each person works on a home
    project about 70% of the time, so a wrong episode by the right person
    carries the right project about 40% of the time, and on the temporal task
    the anchor itself — the one episode the question rules out — carries the
    successor's project in 33-40% of queries (0.333 on seed 0, which is exactly
    what BM25, flat-vector and hybrid RAG scored there). It remains the only
    possible score for the abstraction task, whose answer is a statistic over
    episodes and has no key.

Evaluation is read-only (``rehearse=False``): v0.1 let every query strengthen
what it retrieved, so each score depended on the queries before it.

Several seeds are pooled, since the abstraction task has one query per person
(16 per seed). Intervals are percentile bootstraps over queries and paired
comparisons use a sign-flip randomisation test (``bench/stats.py``).

Usage
-----
    python bench/run.py --fast                      # hashing encoder, no download
    python bench/run.py --encoder minilm --seeds 5  # the README tables
    python bench/run.py --paraphrase ...            # rewritten queries
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import BM25, FlatVectorStore, HybridRAG, RecencyWindow  # noqa: E402
from generate import Corpus, build_corpus  # noqa: E402
from paraphrase import paraphrase_corpus  # noqa: E402
from stats import compare, mean_ci  # noqa: E402

TASKS = ("episodic", "interference", "abstraction", "temporal")
KEYED = ("episodic", "interference", "temporal")
SYSTEMS = ("recency", "bm25", "flat-vector", "hybrid-rag", "engram")


def _hit(text: str, needle: str) -> bool:
    return bool(needle) and re.search(rf"\b{re.escape(needle)}\b", text, re.I) is not None


def score_query(res, q) -> dict[str, float]:
    """Key- and answer-level top-1 and reciprocal rank for one query."""
    key_rank = next((i for i, (_, _, key) in enumerate(res) if tuple(key) == tuple(q.answer_key)), None)
    ans_rank = next((i for i, (txt, _, _) in enumerate(res) if _hit(txt, q.answer_contains)), None)
    return {
        "key@1": float(key_rank == 0),
        "key_rr": 0.0 if key_rank is None else 1.0 / (key_rank + 1),
        "answer@1": float(ans_rank == 0),
        "answer_rr": 0.0 if ans_rank is None else 1.0 / (ans_rank + 1),
    }


def primary(task: str) -> str:
    return "key@1" if task in KEYED else "answer@1"


def evaluate(search, corpus: Corpus, k: int = 5) -> list[dict]:
    """One row per query: task plus both scores."""
    rows = []
    for q in corpus.queries:
        row = score_query(search(q.text, k), q)
        row["task"] = q.task
        rows.append(row)
    return rows


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for t in TASKS:
        sub = [r for r in rows if r["task"] == t]
        if not sub:
            continue
        out[t] = {m: float(np.mean([r[m] for r in sub])) for m in ("key@1", "key_rr", "answer@1", "answer_rr")}
        out[t]["n"] = len(sub)
        out[t]["primary"] = out[t][primary(t)]
    prim = [r[primary(r["task"])] for r in rows]
    m, lo, hi = mean_ci(prim)
    out["overall"] = {"primary": m, "lo": lo, "hi": hi, "n": len(rows),
                      "answer@1": float(np.mean([r["answer@1"] for r in rows]))}
    return out


def build_encoder(name: str):
    from encoders import load_encoder

    return load_encoder(name)


def engram_search(mem):
    params = inspect.signature(mem.recall).parameters
    kw = {"rehearse": False} if "rehearse" in params else {}

    def search(q: str, k: int):
        return [(r.episode.text, r.score, r.episode.key()) for r in mem.recall(q, k=k, **kw)]

    return search


def build_engram(corpus: Corpus, encoder, seed: int, sleep_cycles: int, overrides: dict | None = None):
    from engram import EngramConfig, EngramMemory

    cfg = replace(EngramConfig(seed=seed), **(overrides or {}))
    mem = EngramMemory(encoder=encoder, config=cfg)
    # Episodes one minute apart on a fixed clock, so consolidation is a
    # function of the seed alone (see EngramMemory.sleep).
    for i, (text, key) in enumerate(zip(corpus.episodes, corpus.keys)):
        mem.remember(text, session=key[0], timestamp=60.0 * i)
    if sleep_cycles:
        if "now" in inspect.signature(mem.sleep).parameters:
            mem.sleep(cycles=sleep_cycles, now=60.0 * len(corpus.episodes))
        else:  # v0.1 reads the wall clock
            mem.sleep(cycles=sleep_cycles)
    return mem


def make_corpus(seed: int, args) -> Corpus:
    corpus = build_corpus(n_people=args.people, events_per_person=args.events, seed=seed)
    return paraphrase_corpus(corpus, seed=seed) if args.paraphrase else corpus


def run_seed(seed: int, args, encoder) -> dict[str, list[dict]]:
    corpus = make_corpus(seed, args)
    rows: dict[str, list[dict]] = {}
    for name, ctor in [
        ("recency", lambda: RecencyWindow(window=50)),
        ("bm25", lambda: BM25()),
        ("flat-vector", lambda: FlatVectorStore(encoder)),
        ("hybrid-rag", lambda: HybridRAG(encoder)),
    ]:
        sysobj = ctor()
        sysobj.add_many(corpus.episodes, corpus.keys)
        rows[name] = evaluate(sysobj.search, corpus, args.k)
    mem = build_engram(corpus, encoder, seed, args.sleep_cycles)
    rows["engram"] = evaluate(engram_search(mem), corpus, args.k)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="hashing encoder, no model download")
    ap.add_argument("--encoder", default="minilm")
    ap.add_argument("--people", type=int, default=16)
    ap.add_argument("--events", type=int, default=50, help="events per person timeline")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sleep-cycles", type=int, default=40)
    ap.add_argument("--paraphrase", action="store_true",
                    help="rewrite queries to break literal token overlap (realistic setting)")
    ap.add_argument("--engram-src", default=None, help="score another engram source tree (e.g. v0.1)")
    ap.add_argument("--engram-label", default="", help="how to name that tree in the results, e.g. 'v0.1 (cc2577a)'")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    if args.engram_src:
        sys.path.insert(0, args.engram_src)
    enc_name = "hashing" if args.fast else args.encoder
    encoder = build_encoder(enc_name)
    mode = "paraphrased" if args.paraphrase else "token-literal"
    print(f"seeds 0..{args.seeds - 1}  encoder={enc_name}  queries={mode}")

    t0 = time.perf_counter()
    pooled: dict[str, list[dict]] = {s: [] for s in SYSTEMS}
    for seed in range(args.seeds):
        for name, rows in run_seed(seed, args, encoder).items():
            pooled[name].extend(rows)
        print(f"  seed {seed} done  {time.perf_counter() - t0:.0f}s", flush=True)
    if hasattr(encoder, "save"):
        encoder.save()

    results = {s: summarise(pooled[s]) for s in SYSTEMS}
    print("\n" + "=" * 86)
    print(f"{'primary top-1':<16}" + "".join(f"{n:>14}" for n in SYSTEMS))
    print("-" * 86)
    for t in TASKS:
        print(f"{t:<16}" + "".join(f"{results[n][t]['primary']:>14.3f}" for n in SYSTEMS)
              + f"   ({primary(t)}, n={results['engram'][t]['n']})")
    print(f"{'overall':<16}" + "".join(f"{results[n]['overall']['primary']:>14.3f}" for n in SYSTEMS))
    print(f"{'answer@1 (v0.1)':<16}" + "".join(f"{results[n]['overall']['answer@1']:>14.3f}" for n in SYSTEMS))
    print("=" * 86)

    prim = {s: [r[primary(r["task"])] for r in pooled[s]] for s in SYSTEMS}
    strongest = max((s for s in SYSTEMS if s != "engram"), key=lambda s: np.mean(prim[s]))
    comps = {f"engram vs {b}": compare(prim["engram"], prim[b]) for b in SYSTEMS if b != "engram"}
    per_task = {
        t: {f"engram vs {b}": compare([r[primary(t)] for r in pooled["engram"] if r["task"] == t],
                                      [r[primary(t)] for r in pooled[b] if r["task"] == t])
            for b in SYSTEMS if b != "engram"}
        for t in TASKS
    }
    c = comps[f"engram vs {strongest}"]
    print(f"engram vs strongest baseline ({strongest}): {c['diff']:+.3f} "
          f"[{c['lo']:+.3f}, {c['hi']:+.3f}]  p={c['p']:.4f}  n={c['n']}")

    if args.out:
        from engram import EngramConfig

        config = dict(vars(args))
        if args.engram_src:  # a label, not a local path
            config["engram_src"] = args.engram_label or "external source tree"
        Path(args.out).write_text(json.dumps({
            "config": config, "encoder": enc_name, "query_mode": mode,
            "engram_config": "source tree defaults" if args.engram_src else asdict(EngramConfig()),
            "results": results, "strongest_baseline": strongest,
            "comparisons": comps, "per_task_comparisons": per_task,
            "seconds": time.perf_counter() - t0,
        }, indent=1, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
