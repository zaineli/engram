"""LoCoMo retrieval: the held-out check.

Nothing in engram's configuration, or in the baselines', was chosen on LoCoMo.
Everything is frozen from the LongMemEval-S dev split (``bench/longmemeval.py
tune``) and this script is run once with it. That is the point of it: the
LongMemEval test split was scored more than once while v0.2 was developed (see
the tuning notes there), and a second corpus that no decision ever saw is the
only clean answer to "does this generalise".

LoCoMo (Maharana et al., ACL 2024) is ten long two-person conversations, 19 to
32 sessions each, with question-answer pairs whose ``evidence`` lists the turns
that support the answer. Protocol here:

* **Corpus.** One per conversation: every turn, rendered as
  ``(session date) Speaker said, "text"``, plus ``[shares caption]`` when the
  turn carries an image. That is the rendering the authors' retrieval code
  uses. engram stores one episode per turn, with the session index and the
  turn's position in it.
* **Questions.** Categories 1-4. Category 5 is adversarial and has no
  evidence. Evidence ids are parsed as ``D<session>:<turn>``; ids that do not
  resolve to a turn are dropped, and a question with none left is excluded.
  Counts are printed and stored.
* **Metrics.** ``recall_all@k`` (every evidence turn in the top k), the
  official fractional ``recall@k`` (share of evidence turns in the top k), and
  ``ndcg_any@10`` with the same DCG as the LongMemEval harness.
* **Intervals.** Questions within a conversation share a corpus, so each
  comparison reports a conversation-cluster bootstrap next to the question-
  level one. With ten clusters it is the honest, and wide, one.

The data is CC BY-NC 4.0 and is downloaded at run time, never committed.

    python bench/locomo.py download
    python bench/locomo.py run --encoder minilm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from encoders import CachedEncoder  # noqa: E402
from longmemeval import Okapi, _at_k, _rrf, _z, complete, rank_scores, tok_normalised  # noqa: E402
from stats import cluster_bootstrap, compare  # noqa: E402

DATA = Path(os.environ.get("LOCOMO", HERE / "data" / "locomo10.json"))
URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
RESULTS = HERE / "results"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
_EVID = re.compile(r"D(\d+):(\d+)")


def _date(s: str) -> float:
    # "1:56 pm on 8 May, 2023"
    try:
        return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y").timestamp()
    except ValueError:
        return 0.0


def conversation(entry: dict) -> dict:
    conv = entry["conversation"]
    texts, ids, sess, pos, ts = [], [], [], [], []
    n = 1
    while f"session_{n}" in conv:
        date = conv.get(f"session_{n}_date_time", "")
        for j, turn in enumerate(conv[f"session_{n}"]):
            t = f'({date}) {turn["speaker"]} said, "{turn["text"]}"'
            if turn.get("blip_caption"):
                t += f' [shares {turn["blip_caption"]}]'
            texts.append(t)
            ids.append(turn["dia_id"])
            sess.append(n - 1)
            pos.append(j)
            ts.append(_date(date))
        n += 1
    known = set(ids)
    questions, dropped_ids, excluded = [], 0, 0
    for qa in entry["qa"]:
        if qa.get("category") not in CATEGORIES:
            continue
        ev = {f"D{a}:{b}" for e in qa.get("evidence", []) for a, b in _EVID.findall(e)}
        good = ev & known
        dropped_ids += len(ev - known)
        if not good:
            excluded += 1
            continue
        questions.append({"question": qa["question"], "category": qa["category"], "gold": sorted(good)})
    return {"id": entry["sample_id"], "texts": texts, "ids": ids, "session": sess, "position": pos,
            "timestamp": ts, "questions": questions, "dropped_ids": dropped_ids, "excluded": excluded}


def score(order: list[int], conv: dict, gold: list[str]) -> dict[str, float]:
    ranked = [conv["ids"][i] for i in order]
    g = set(gold)
    out = {}
    for k in (5, 10):
        _, r_all, _ = _at_k(ranked, conv["ids"], g, k)
        out[f"recall_all@{k}"] = r_all
        out[f"recall@{k}"] = len(g & set(ranked[:k])) / len(g)
    out["ndcg_any@10"] = _at_k(ranked, conv["ids"], g, 10)[2]
    return out


def run(args) -> None:
    from engram import EngramConfig, EngramMemory

    tuning = json.loads((RESULTS / f"lme_tuning_{args.tuned_on}.json").read_text())
    frozen, alpha, lexical = tuning["engram"], tuning["hybrid_z"]["alpha"], tuning["hybrid_z"]["lexical"]
    if lexical != "bm25":
        raise SystemExit("the frozen hybrid uses the official whitespace BM25, which is LongMemEval-specific")
    data = json.loads(DATA.read_text())
    convs = [conversation(e) for e in data]
    enc = CachedEncoder(args.encoder)
    t0 = time.perf_counter()

    systems = ("bm25", "dense", "hybrid_rrf", "hybrid_z", "engram")
    rows: dict[str, list[dict]] = {s: [] for s in systems}
    meta: list[dict] = []
    for conv in convs:
        D = enc.encode(conv["texts"])
        Q = enc.encode_query([q["question"] for q in conv["questions"]])
        bm = Okapi([tok_normalised(t) for t in conv["texts"]])
        mem = EngramMemory(encoder=enc, config=EngramConfig(**frozen))
        for i, text in enumerate(conv["texts"]):
            mem.remember(text, session=conv["session"][i], timestamp=conv["timestamp"][i], meta={"i": i})
        for q, qv in zip(conv["questions"], Q):
            s_bm = bm.scores(tok_normalised(q["question"]))
            s_de = D @ qv
            orders = {
                "bm25": rank_scores(s_bm),
                "dense": rank_scores(s_de),
                "hybrid_rrf": rank_scores(_rrf(s_bm, s_de)),
                "hybrid_z": rank_scores(alpha * _z(s_bm) + (1 - alpha) * _z(s_de)),
            }
            got = mem.recall(q["question"], k=None, include_schemas=False, rehearse=False, dedup=False)
            orders["engram"] = complete([r.episode.meta["i"] for r in got], len(conv["texts"]))
            for s in systems:
                rows[s].append(score(orders[s], conv, q["gold"]))
            meta.append({"conversation": conv["id"], "category": q["category"]})
    enc.save()

    metrics = list(rows["engram"][0])
    table = {s: {m: [r[m] for r in rows[s]] for m in metrics} for s in systems}
    summary = {s: {m: float(np.mean(v)) for m, v in table[s].items()} for s in systems}
    by_cat = {s: {CATEGORIES[c]: float(np.mean([r["recall@10"] for r, mm in zip(rows[s], meta) if mm["category"] == c]))
                  for c in CATEGORIES} for s in systems}
    clusters = [m["conversation"] for m in meta]
    comps = {}
    for b in systems[:-1]:
        comps[f"engram vs {b}"] = {}
        for m in metrics:
            c = compare(table["engram"][m], table[b][m])
            _, clo, chi = cluster_bootstrap(table["engram"][m], table[b][m], clusters)
            c.update({"cluster_lo": clo, "cluster_hi": chi})
            comps[f"engram vs {b}"][m] = c

    n_q = len(meta)
    print(f"LoCoMo: {len(convs)} conversations, {sum(len(c['texts']) for c in convs)} turns, {n_q} questions "
          f"(categories 1-4; {sum(c['excluded'] for c in convs)} excluded for unresolvable evidence, "
          f"{sum(c['dropped_ids'] for c in convs)} evidence ids dropped)  encoder={args.encoder}")
    print(f"engram config frozen on {args.tuned_on} LongMemEval dev: {frozen}; hybrid_z alpha={alpha}\n")
    print(f"{'system':<12}" + "".join(f"{m:>14}" for m in metrics))
    for s in systems:
        print(f"{s:<12}" + "".join(f"{summary[s][m]:>14.3f}" for m in metrics))
    for name, comp in comps.items():
        c = comp["recall@10"]
        print(f"  {name:<22} recall@10 {c['diff']:+.3f}  questions [{c['lo']:+.3f},{c['hi']:+.3f}] p={c['p']:.3f}"
              f"  conversations [{c['cluster_lo']:+.3f},{c['cluster_hi']:+.3f}]")

    out = {
        "benchmark": "LoCoMo, turn-level retrieval, categories 1-4",
        "encoder": args.encoder, "engram_config": frozen, "tuned_on": f"{args.tuned_on} LongMemEval-S dev",
        "hybrid_z": {"lexical": lexical, "alpha": alpha}, "n_questions": n_q,
        "excluded": sum(c["excluded"] for c in convs), "dropped_evidence_ids": sum(c["dropped_ids"] for c in convs),
        "summary": summary, "recall@10_by_category": by_cat, "comparisons": comps,
        "per_question": table, "meta": meta, "seconds": time.perf_counter() - t0,
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"locomo_{args.encoder}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")


def download(_args) -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if DATA.exists():
        print(f"{DATA} already present")
        return
    urllib.request.urlretrieve(URL, DATA)
    print(f"wrote {DATA}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    p = sub.add_parser("run")
    p.add_argument("--encoder", default="minilm")
    p.add_argument("--tuned-on", default="minilm")
    args = ap.parse_args()
    {"download": download, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
