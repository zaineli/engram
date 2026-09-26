"""Check this repository's LongMemEval scorer and BM25 against the originals.

``bench/longmemeval.py`` reimplements the reference harness's metrics and
``rank_bm25``'s BM25Okapi rather than vendoring them. This script downloads
both originals at run time into a temporary directory (nothing is written into
the repository), scores random rankings of real haystacks with both
implementations, and fails on any disagreement.

    python bench/check_reference.py [--questions 60 --rankings 5]

Last run: 300 rankings, every turn- and session-level metric identical; BM25
scores identical to 1e-9 on 60 haystacks, with the rank order differing only
where two scores tie to within floating-point summation order.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import longmemeval as L  # noqa: E402

SOURCES = {
    "ref_eval": "https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/src/retrieval/eval_utils.py",
    "ref_bm25": "https://raw.githubusercontent.com/dorianbrown/rank_bm25/master/rank_bm25.py",
}


def _fetch(name: str, url: str, where: Path):
    path = where / f"{name}.py"
    urllib.request.urlretrieve(url, path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=60)
    ap.add_argument("--rankings", type=int, default=5)
    args = ap.parse_args()
    if not hasattr(np, "asfarray"):  # removed in numpy 2; the reference scorer uses it
        np.asfarray = lambda x: np.asarray(x, dtype=float)

    with tempfile.TemporaryDirectory() as tmp:
        ref = _fetch("ref_eval", SOURCES["ref_eval"], Path(tmp))
        bm = _fetch("ref_bm25", SOURCES["ref_bm25"], Path(tmp))

        insts = L.load(split="all")
        rng = random.Random(0)
        bad = checked = ties = 0
        for inst in rng.sample(insts, min(args.questions, len(insts))):
            gold = sorted(inst.gold)
            for _ in range(args.rankings):
                order = list(range(len(inst.texts)))
                rng.shuffle(order)
                mine = L.score_ranking(inst, order)
                for k in L.TURN_KS:
                    _, r_all, ndcg = ref.evaluate_retrieval(order, gold, inst.ids, k=k)
                    bad += abs(mine[f"turn_recall_all@{k}"] - r_all) > 1e-12
                    bad += abs(mine[f"turn_ndcg_any@{k}"] - ndcg) > 1e-9
                for k in L.SESSION_KS:
                    _, r_all, ndcg = ref.evaluate_retrieval_turn2session(order, gold, inst.ids, k=k)
                    bad += abs(mine[f"session_recall_all@{k}"] - r_all) > 1e-12
                    bad += abs(mine[f"session_ndcg_any@{k}"] - ndcg) > 1e-9
                checked += 1
            theirs = bm.BM25Okapi([L.tok_official(t) for t in inst.texts]).get_scores(L.tok_official(inst.question))
            ours = L.Okapi([L.tok_official(t) for t in inst.texts]).scores(L.tok_official(inst.question))
            if not np.allclose(theirs, ours, rtol=0, atol=1e-9):
                bad += 1
                print(f"BM25 disagrees on {inst.qid}: max |diff| {np.abs(theirs - ours).max():.2e}")
            o_ours, o_theirs = L.rank_scores(ours), np.argsort(theirs)[::-1].tolist()
            if o_ours != o_theirs:
                # Different orders are acceptable only if every rank holds the same score.
                if np.allclose(ours[o_ours], theirs[o_theirs], rtol=0, atol=1e-9):
                    ties += 1
                else:
                    bad += 1
                    print(f"BM25 rank order disagrees beyond ties on {inst.qid}")

    print(f"{checked} rankings over {min(args.questions, len(insts))} haystacks: {bad} disagreements; "
          f"{ties} BM25 rankings differ only in tie order")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
