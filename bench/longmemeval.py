"""LongMemEval-S retrieval, scored with the official protocol.

The synthetic benchmark in ``bench/generate.py`` has exact ground truth and
isolates one mechanism per task, and it is templated: every episode is one
sentence from one frame. This harness measures engram on real conversational
memory, LongMemEval (Wu et al., ICLR 2025), in the retrieval-only setting the
paper defines, so the numbers sit next to the paper's own retrieval tables.

Protocol (reimplemented here from the paper and the reference harness; no code
was copied):

* **Data.** ``longmemeval_s_cleaned.json`` (MIT), the September 2025 cleaned
  release: 500 questions, each with its own haystack of about 48 sessions.
* **Keys.** One key per *user* turn, id ``f"{session_id}_{turn_index + 1}"``
  where the index counts all turns. A haystack is its own corpus; every system
  starts empty for every question.
* **Gold.** Inside an evidence session (id contains ``answer``), a user turn
  without ``has_answer`` is relabelled ``noans`` and is not gold. Gold is every
  key still containing ``answer``.
* **Exclusions.** Abstention questions (``_abs``) and questions with no
  user-side evidence turn are not scored. That leaves n = 419, and it removes
  most single-session-assistant questions, whose evidence is an assistant turn.
* **Metrics.** ``recall_all@k`` (every gold item in the top k; the paper's
  "Recall@k") and ``ndcg_any@k`` with binary relevance and the reference
  harness's DCG, ``rel_1 + sum_{p>=2} rel_p / log2(p)``. Session-level numbers
  come from the turn ranking the official way: strip the turn suffix and grow k
  until the top-k turns cover k distinct sessions.
* **Split.** Questions whose ``sha1(question_id)`` is 0 mod 5 form a dev split
  (n = 76 after exclusions). Every tuned choice, engram's and the baselines',
  is made there and frozen before the test split (n = 343) is scored.

Usage::

    python bench/longmemeval.py download
    python bench/longmemeval.py embed --encoder minilm
    python bench/longmemeval.py tune  --encoder minilm
    python bench/longmemeval.py run   --encoder minilm --split test
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stats import compare  # noqa: E402

DATA = Path(os.environ.get("LONGMEMEVAL_S", HERE / "data" / "longmemeval_s_cleaned.json"))
URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_s_cleaned.json"
)
RESULTS = HERE / "results"

SESSION_KS = (5, 10)
TURN_KS = (5, 10, 50)
#: The four numbers the reference harness prints, all session level.
HEADLINE = ("session_recall_all@5", "session_ndcg_any@5", "session_recall_all@10", "session_ndcg_any@10")


# ---------------------------------------------------------------- data


@dataclass
class Instance:
    qid: str
    qtype: str
    question: str
    texts: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    #: Haystack index of each turn's session, and its position among that
    #: session's user turns: engram's (session, index_in_session).
    session: list[int] = field(default_factory=list)
    position: list[int] = field(default_factory=list)
    timestamp: list[float] = field(default_factory=list)

    @property
    def gold(self) -> set[str]:
        return {i for i in self.ids if "answer" in i}


def _parse_date(s: str) -> float:
    # "2023/05/20 (Sat) 02:21"
    try:
        return datetime.strptime(s, "%Y/%m/%d (%a) %H:%M").timestamp()
    except ValueError:
        return 0.0


def build_instance(entry: dict) -> Instance:
    inst = Instance(entry["question_id"], entry["question_type"], entry["question"])
    for s_idx, (sid, sess, date) in enumerate(
        zip(entry["haystack_session_ids"], entry["haystack_sessions"], entry["haystack_dates"])
    ):
        ts = _parse_date(date)
        pos = 0
        for t_idx, turn in enumerate(sess):
            if turn["role"] != "user":
                continue
            cid = f"{sid}_{t_idx + 1}"
            if "answer" in sid and not turn.get("has_answer", False):
                cid = cid.replace("answer", "noans")
            inst.texts.append(turn["content"])
            inst.ids.append(cid)
            inst.session.append(s_idx)
            inst.position.append(pos)
            inst.timestamp.append(ts)
            pos += 1
    return inst


def scored(entry: dict) -> bool:
    """The reference harness's exclusion rule."""
    if "_abs" in entry["question_id"]:
        return False
    return any(
        t.get("has_answer", False) for sess in entry["haystack_sessions"] for t in sess if t["role"] == "user"
    )


def split_of(qid: str) -> str:
    return "dev" if int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 5 == 0 else "test"


def load(path: Path = DATA, split: str = "all") -> list[Instance]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python bench/longmemeval.py download` first.")
    with open(path) as f:
        data = json.load(f)
    out = [build_instance(e) for e in data if scored(e)]
    if split != "all":
        out = [i for i in out if split_of(i.qid) == split]
    return out


# ---------------------------------------------------------------- scoring


def _dcg(rels: list[int]) -> float:
    if not rels:
        return 0.0
    return rels[0] + sum(r / math.log2(p) for p, r in enumerate(rels[1:], start=2))


def _at_k(ranked: list[str], corpus: list[str], gold: set[str], k: int) -> tuple[float, float, float]:
    """(recall_any, recall_all, ndcg_any) of the first ``k`` of ``ranked``."""
    top = ranked[:k]
    got = set(top)
    r_any = float(any(g in got for g in gold))
    r_all = float(all(g in got for g in gold))
    ideal = _dcg(sorted((int(c in gold) for c in corpus), reverse=True)[:k])
    ndcg = _dcg([int(c in gold) for c in top]) / ideal if ideal > 0 else 0.0
    return r_any, r_all, ndcg


def _strip(cid: str) -> str:
    return cid.rsplit("_", 1)[0]


def score_ranking(inst: Instance, order: list[int]) -> dict[str, float]:
    """Official turn- and session-level metrics for one ranking of ``inst.ids``."""
    ranked = [inst.ids[i] for i in order]
    gold = inst.gold
    out: dict[str, float] = {}
    for k in TURN_KS:
        _, r_all, ndcg = _at_k(ranked, inst.ids, gold, k)
        out[f"turn_recall_all@{k}"] = r_all
        out[f"turn_ndcg_any@{k}"] = ndcg

    s_ranked = [_strip(c) for c in ranked]
    s_corpus = [_strip(c) for c in inst.ids]
    s_gold = {_strip(g) for g in gold}
    for k in SESSION_KS:
        # Grow the cut until it spans k distinct sessions (or the corpus ends).
        eff, seen = 0, set()
        while eff < len(s_ranked) and len(seen) < k:
            seen.add(s_ranked[eff])
            eff += 1
        _, r_all, ndcg = _at_k(s_ranked, s_corpus, s_gold, eff)
        out[f"session_recall_all@{k}"] = r_all
        out[f"session_ndcg_any@{k}"] = ndcg
    return out


def complete(order: list[int], n: int) -> list[int]:
    """Append any missing indices so every ranking covers the whole corpus."""
    seen = set(order)
    return list(order) + [i for i in range(n) if i not in seen]


# ---------------------------------------------------------------- lexical


_WORD = re.compile(r"[a-z0-9]+")


def tok_official(s: str) -> list[str]:
    """The reference harness's tokeniser: case-sensitive split on single spaces."""
    return s.split(" ")


def tok_normalised(s: str) -> list[str]:
    return _WORD.findall(s.lower())


class Okapi:
    """BM25 with the Okapi IDF and its negative-IDF floor, as ``rank_bm25`` defines it.

    ``idf(t) = ln(N - df + 0.5) - ln(df + 0.5)``; a term in more than half the
    documents gets a negative value, which is replaced by ``eps`` times the
    mean IDF over the vocabulary. A query term repeated in the query counts
    once per occurrence.
    """

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75, eps: float = 0.25):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.dl = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if self.n else 0.0
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                self.post[t].append((i, c))
        idf = {t: math.log(self.n - len(p) + 0.5) - math.log(len(p) + 0.5) for t, p in self.post.items()}
        floor = eps * (sum(idf.values()) / len(idf)) if idf else 0.0
        self.idf = {t: (v if v >= 0 else floor) for t, v in idf.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float64)
        norm = self.k1 * (1 - self.b + self.b * self.dl / max(self.avgdl, 1e-12))
        for t in query:
            p = self.post.get(t)
            if not p:
                continue
            idx = np.fromiter((i for i, _ in p), dtype=np.int64, count=len(p))
            tf = np.fromiter((c for _, c in p), dtype=np.float64, count=len(p))
            s[idx] += self.idf[t] * tf * (self.k1 + 1) / (tf + norm[idx])
        return s


def rank_scores(s: np.ndarray) -> list[int]:
    """Descending order with the reference harness's tie handling."""
    return np.argsort(s)[::-1].tolist()


def _z(v: np.ndarray) -> np.ndarray:
    sd = float(v.std())
    return np.zeros_like(v) if sd < 1e-12 else (v - float(v.mean())) / sd


def _rrf(*scores: np.ndarray, k: float = 60.0) -> np.ndarray:
    out = np.zeros_like(scores[0], dtype=np.float64)
    for s in scores:
        ranks = np.empty(s.size)
        ranks[np.argsort(s)[::-1]] = np.arange(1, s.size + 1)
        out += 1.0 / (k + ranks)
    return out


# ---------------------------------------------------------------- systems


def text_key(s: str) -> str:
    return hashlib.blake2b(s.encode(), digest_size=16).hexdigest()


class ArrayEncoder:
    """Serves cached vectors to engram inside a worker; never loads a model.

    Vectors are keyed by a hash of the raw text. ``encode`` falls back to the
    query table so that a memory that embeds its queries with ``encode`` (v0.1
    had no ``encode_query``) still finds them; for the one model with no
    prefixes, MiniLM, the two tables agree.
    """

    def __init__(self, dim: int, docs: dict[str, np.ndarray], queries: dict[str, np.ndarray]):
        self.dim = dim
        self._docs, self._queries = docs, queries

    def encode(self, texts):
        out = []
        for t in texts:
            k = text_key(t)
            out.append(self._docs[k] if k in self._docs else self._queries[k])
        return np.stack(out)

    def encode_query(self, texts):
        return np.stack([self._queries[text_key(t)] for t in texts])


def lexical_and_dense(inst: Instance, enc: ArrayEncoder) -> dict[str, np.ndarray]:
    D = enc.encode(inst.texts)
    q = enc.encode_query([inst.question])[0]
    return {
        "bm25_official": Okapi([tok_official(t) for t in inst.texts]).scores(tok_official(inst.question)),
        "bm25": Okapi([tok_normalised(t) for t in inst.texts]).scores(tok_normalised(inst.question)),
        "dense": D @ q,
    }


def baseline_rankings(inst: Instance, enc: ArrayEncoder, alpha: float, lexical: str) -> dict[str, list[int]]:
    s = lexical_and_dense(inst, enc)
    return {
        "bm25_official": rank_scores(s["bm25_official"]),
        "bm25": rank_scores(s["bm25"]),
        "dense": rank_scores(s["dense"]),
        "hybrid_rrf": rank_scores(_rrf(s["bm25"], s["dense"])),
        "hybrid_z": rank_scores(alpha * _z(s[lexical]) + (1 - alpha) * _z(s["dense"])),
    }


def ingest(inst: Instance, enc: ArrayEncoder, cfg, sleep: int = 0):
    from engram import EngramMemory

    mem = EngramMemory(encoder=enc, config=cfg)
    for i, text in enumerate(inst.texts):
        mem.remember(text, session=inst.session[i], timestamp=inst.timestamp[i], meta={"i": i})
    if sleep:
        mem.sleep(cycles=sleep, now=max(inst.timestamp))
    return mem


def rank_with(mem, inst: Instance) -> list[int]:
    params = inspect.signature(mem.recall).parameters
    if "rehearse" in params:
        got = mem.recall(inst.question, k=None, include_schemas=False, rehearse=False, dedup=False)
    else:  # v0.1: no read-only mode, and duplicates collapse by text
        got = mem.recall(inst.question, k=len(inst.texts), include_schemas=False)
    return complete([r.episode.meta["i"] for r in got], len(inst.texts))


def engram_ranking(inst: Instance, enc: ArrayEncoder, cfg, sleep: int = 0) -> list[int]:
    return rank_with(ingest(inst, enc, cfg, sleep), inst)


# ---------------------------------------------------------------- workers

_W: dict = {}


def _init_worker(enc_name: str, engram_src: str | None) -> None:
    if engram_src:
        sys.path.insert(0, engram_src)
    z = np.load(RESULTS.parent / ".cache" / f"lme_{enc_name}.npz", allow_pickle=False)
    _W["docs"] = dict(zip(z["doc_keys"].tolist(), z["doc_vecs"]))
    _W["queries"] = dict(zip(z["q_keys"].tolist(), z["q_vecs"]))
    _W["dim"] = int(z["doc_vecs"].shape[1])


def _enc() -> ArrayEncoder:
    return ArrayEncoder(_W["dim"], _W["docs"], _W["queries"])


def _job_run(args) -> tuple[str, dict[str, dict[str, float]]]:
    inst, systems, cfg_dict, alpha, lexical = args
    from engram import EngramConfig

    enc = _enc()
    out: dict[str, dict[str, float]] = {}
    base = [s for s in systems if not s.startswith("engram")]
    if base:
        for name, order in baseline_rankings(inst, enc, alpha, lexical).items():
            if name in base:
                out[name] = score_ranking(inst, order)
    for s in systems:
        if s.startswith("engram"):
            overrides = cfg_dict.get(s, {})
            fields = inspect.signature(EngramConfig).parameters
            cfg = EngramConfig(**{k: v for k, v in overrides.items() if k in fields and k != "sleep"})
            out[s] = score_ranking(inst, engram_ranking(inst, enc, cfg, sleep=overrides.get("sleep", 0)))
    return inst.qid, out


def _job_tune(args) -> tuple[str, dict[str, dict[str, float]]]:
    inst, grid, alphas = args
    from engram import EngramConfig, EngramMemory

    READ_TIME = EngramMemory.READ_TIME

    enc = _enc()
    out: dict[str, dict[str, float]] = {}
    s = lexical_and_dense(inst, enc)
    for lex in ("bm25", "bm25_official"):
        for a in alphas:
            out[f"hybrid_z|{lex}|{a}"] = score_ranking(inst, rank_scores(a * _z(s[lex]) + (1 - a) * _z(s["dense"])))
    # Ingest once per write-time setting, then sweep the read-time ones on the
    # same memory (EngramMemory.reconfigure; tested equal to a fresh build).
    by_write: dict[tuple, list[str]] = defaultdict(list)
    for key, ov in grid.items():
        by_write[tuple(sorted((k, v) for k, v in ov.items() if k not in READ_TIME))].append(key)
    for write_ov, keys in by_write.items():
        mem = ingest(inst, enc, EngramConfig(**dict(write_ov)))
        for key in keys:
            mem.reconfigure(**{k: v for k, v in grid[key].items() if k in READ_TIME})
            out[key] = score_ranking(inst, rank_with(mem, inst))
    return inst.qid, out


# ---------------------------------------------------------------- commands


def cmd_download(_args) -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if DATA.exists():
        print(f"{DATA} already present")
        return
    print(f"fetching {URL}")
    urllib.request.urlretrieve(URL, DATA)
    print(f"wrote {DATA} ({DATA.stat().st_size / 1e6:.0f} MB)")


def cmd_embed(args) -> None:
    from encoders import CachedEncoder

    insts = load(split="all")
    docs = sorted({t for i in insts for t in i.texts})
    qs = sorted({i.question for i in insts})
    enc = CachedEncoder(args.encoder)
    t0 = time.perf_counter()
    for j in range(0, len(docs), 4096):
        enc.encode(docs[j : j + 4096])
        enc.save()
        print(f"  {min(j + 4096, len(docs))}/{len(docs)} docs  {time.perf_counter() - t0:.0f}s", flush=True)
    dv = enc.encode(docs)
    qv = enc.encode_query(qs)
    enc.save()
    out = RESULTS.parent / ".cache" / f"lme_{args.encoder}.npz"
    np.savez(out, doc_keys=np.array([text_key(t) for t in docs]), doc_vecs=dv,
             q_keys=np.array([text_key(q) for q in qs]), q_vecs=qv)
    print(f"embedded {len(docs)} turns and {len(qs)} questions with {args.encoder} "
          f"in {time.perf_counter() - t0:.0f}s -> {out}")


def _pool(args, engram_src: str | None = None):
    ctx = mp.get_context("spawn")
    return ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.encoder, engram_src))


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}


def _objective(m: dict[str, float]) -> float:
    return float(np.mean([m[h] for h in HEADLINE]))


#: Three rounds, all on dev; each round's file is kept in bench/results.
#: r1 (w_conjunctive 0.6/1.2/2.4, window 4/8/16/None) put its optimum on two
#: edges, so r2 extended both axes. r2's frozen config was scored once on the
#: test split, where engram tied the z-score hybrid and its conjunctive
#: pathway alone trailed BM25; r3 added BM25 weighting of conjunction units.
#: Dev preferred IDF-cosine to BM25 weighting (0.845 against 0.841), so r2's
#: config stayed frozen and its test score stands. The test split was still
#: looked at before r3, which is why LoCoMo, untouched by every round, is
#: reported as the held-out check (bench/locomo.py).
TUNE_GRID = {
    "fusion": ("minmax", "zscore", "rrf"),
    "w_conjunctive": (0.2, 0.3, 0.45, 0.6, 0.9, 1.2, 2.4),
    "conj_window": (2, 3, 4, 6, 8, 16, None),
    "conj_weighting": ("idf", "binary", "bm25"),
}
#: 0 and 1 are dense alone and BM25 alone, for reference.
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def cmd_tune(args) -> None:
    insts = load(split="dev")
    keys = list(TUNE_GRID)
    grid: dict[str, dict] = {}
    for combo in np.array(np.meshgrid(*[range(len(TUNE_GRID[k])) for k in keys])).T.reshape(-1, len(keys)):
        ov = {k: TUNE_GRID[k][int(c)] for k, c in zip(keys, combo)}
        grid["engram|" + "|".join(f"{k}={v}" for k, v in ov.items())] = ov
    print(f"tuning on dev: n={len(insts)}, {len(grid)} engram configs, "
          f"{2 * len(ALPHAS)} hybrid configs, encoder={args.encoder}")
    t0 = time.perf_counter()
    rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    with _pool(args) as pool:
        for j, (_, res) in enumerate(pool.imap_unordered(_job_tune, [(i, grid, ALPHAS) for i in insts]), 1):
            for name, m in res.items():
                rows[name].append(m)
            if j % 10 == 0:
                print(f"  {j}/{len(insts)}  {time.perf_counter() - t0:.0f}s", flush=True)
    means = {name: _mean(r) for name, r in rows.items()}
    ranked = sorted(means, key=lambda n: _objective(means[n]), reverse=True)
    best_engram = next(n for n in ranked if n.startswith("engram"))
    best_hybrid = next(n for n in ranked if n.startswith("hybrid_z"))
    _, lex, alpha = best_hybrid.split("|")
    print("\ntop engram configs (objective = mean of the four headline metrics):")
    for n in [n for n in ranked if n.startswith("engram")][:8]:
        print(f"  {_objective(means[n]):.4f}  {n}")
    print(f"best hybrid_z: {lex} alpha={alpha}  {_objective(means[best_hybrid]):.4f}")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"lme_tuning_{args.encoder}.json"
    out.write_text(json.dumps({
        "encoder": args.encoder, "split": "dev", "n": len(insts), "objective": list(HEADLINE),
        "grid": {k: list(v) for k, v in TUNE_GRID.items()}, "alphas": list(ALPHAS),
        "engram": grid[best_engram], "hybrid_z": {"lexical": lex, "alpha": float(alpha)},
        "means": means,
    }, indent=1, default=str))
    print(f"wrote {out}")


#: Ablations of the frozen config, each one change away from it.
ABLATIONS = {
    "engram": {},
    "engram-conjunctive": {"w_conjunctive": 0.0},
    "engram-semantic": {"w_semantic": 0.0},
    "engram_window_all": {"conj_window": None},
    "engram_unigram": {"conj_order": 1},
    "engram_binary": {"conj_weighting": "binary"},
    "engram_bm25": {"conj_weighting": "bm25"},
    "engram_zscore": {"fusion": "zscore"},
    "engram_rrf": {"fusion": "rrf"},
    "engram-reinstate": {"w_reinstate": 0.0},
    "engram_slept": {"sleep": 40},
}


def cmd_run(args) -> None:
    tuning_path = RESULTS / f"lme_tuning_{args.tuned_on}.json"
    if not tuning_path.exists():
        raise SystemExit(f"{tuning_path} missing: run `tune --encoder {args.tuned_on}` first")
    tuning = json.loads(tuning_path.read_text())
    frozen = tuning["engram"]
    alpha, lexical = tuning["hybrid_z"]["alpha"], tuning["hybrid_z"]["lexical"]

    insts = load(split=args.split)
    baselines = ["bm25_official", "bm25", "dense", "hybrid_rrf", "hybrid_z"]
    if args.engram_src:
        systems = ["engram"]
        cfgs = {"engram": {}}  # the source tree's own defaults
    else:
        names = list(ABLATIONS) if args.ablations else ["engram"]
        systems = baselines + names
        cfgs = {n: {**frozen, **ABLATIONS[n]} for n in names}
    print(f"LongMemEval-S {args.split}: n={len(insts)}  encoder={args.encoder}  "
          f"config frozen on {args.tuned_on} dev: {frozen}  hybrid_z: {lexical} alpha={alpha}")

    t0 = time.perf_counter()
    per_q: dict[str, dict[str, dict[str, float]]] = {}
    with _pool(args, args.engram_src) as pool:
        jobs = [(i, systems, cfgs, alpha, lexical) for i in insts]
        for j, (qid, res) in enumerate(pool.imap_unordered(_job_run, jobs), 1):
            per_q[qid] = res
            if j % 50 == 0:
                print(f"  {j}/{len(insts)}  {time.perf_counter() - t0:.0f}s", flush=True)

    order = [i.qid for i in insts]
    metrics = list(next(iter(per_q.values()))[systems[0]])
    table = {s: {m: [per_q[q][s][m] for q in order] for m in metrics} for s in systems}
    summary = {s: {m: float(np.mean(v)) for m, v in table[s].items()} for s in systems}

    out = {
        "benchmark": "LongMemEval-S cleaned, retrieval, official protocol",
        "split": args.split, "n": len(insts), "encoder": args.encoder,
        "tuned_on": f"{args.tuned_on} dev", "engram_config": frozen,
        "engram_src": (args.engram_label or "external source tree") if args.engram_src else "this tree",
        "hybrid_z": {"lexical": lexical, "alpha": alpha},
        "qids": order, "qtypes": [i.qtype for i in insts],
        "summary": summary, "per_question": table,
        "seconds": time.perf_counter() - t0,
    }
    if not args.engram_src:
        strongest = max(baselines, key=lambda b: _objective(summary[b]))
        out["strongest_baseline"] = strongest
        out["comparisons"] = {
            f"engram vs {b}": {m: compare(table["engram"][m], table[b][m]) for m in metrics}
            for b in baselines
        }
    RESULTS.mkdir(exist_ok=True)
    tag = args.tag or (f"lme_{args.split}_{args.encoder}" + ("_ablations" if args.ablations else ""))
    path = RESULTS / f"{tag}.json"
    path.write_text(json.dumps(out, indent=1))
    report(out)
    print(f"\nwrote {path}  ({out['seconds']:.0f}s)")


def report(out: dict) -> None:
    cols = list(HEADLINE) + ["turn_recall_all@10", "turn_ndcg_any@10", "turn_recall_all@50"]
    short = ["S R@5", "S N@5", "S R@10", "S N@10", "T R@10", "T N@10", "T R@50"]
    print(f"\n{'system':<22}" + "".join(f"{c:>8}" for c in short))
    for s, m in out["summary"].items():
        print(f"{s:<22}" + "".join(f"{m[c]:>8.3f}" for c in cols))
    for name, comp in out.get("comparisons", {}).items():
        c = comp["session_recall_all@5"], comp["turn_ndcg_any@10"]
        print(f"  {name:<28} S R@5 {c[0]['diff']:+.3f} [{c[0]['lo']:+.3f},{c[0]['hi']:+.3f}] p={c[0]['p']:.3f}"
              f"   T N@10 {c[1]['diff']:+.3f} [{c[1]['lo']:+.3f},{c[1]['hi']:+.3f}] p={c[1]['p']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    for name in ("embed", "tune", "run"):
        p = sub.add_parser(name)
        p.add_argument("--encoder", default="minilm")
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
        if name == "run":
            p.add_argument("--split", default="test", choices=["dev", "test", "all"])
            p.add_argument("--tuned-on", default="minilm")
            p.add_argument("--ablations", action="store_true")
            p.add_argument("--engram-src", default=None,
                           help="score another engram source tree (e.g. v0.1) with its own defaults")
            p.add_argument("--engram-label", default="", help="how to name that tree in the results")
            p.add_argument("--tag", default="")
    args = ap.parse_args()
    {"download": cmd_download, "embed": cmd_embed, "tune": cmd_tune, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
