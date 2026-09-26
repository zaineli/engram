"""Tests for the benchmark harness itself.

A scorer that is wrong produces numbers that look exactly like results, so the
LongMemEval scorer, the BM25 baseline and the statistics are pinned against
small cases worked out by hand.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import longmemeval as lme  # noqa: E402
from generate import Query, build_corpus  # noqa: E402
from run import score_query  # noqa: E402
from stats import compare, paired_bootstrap, paired_permutation  # noqa: E402


def _entry(qid: str = "q1") -> dict:
    """Three sessions. s_b and s_c are evidence sessions; s_c's evidence is assistant-side."""
    return {
        "question_id": qid,
        "question_type": "multi-session",
        "question": "where did I park",
        "haystack_session_ids": ["s_a", "answer_b", "answer_c"],
        "haystack_dates": ["2023/05/20 (Sat) 02:21"] * 3,
        "haystack_sessions": [
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
             {"role": "user", "content": "the weather"}],
            [{"role": "user", "content": "I parked on level 3", "has_answer": True},
             {"role": "assistant", "content": "noted"},
             {"role": "user", "content": "thanks", "has_answer": False}],
            [{"role": "user", "content": "remind me", "has_answer": False},
             {"role": "assistant", "content": "level 3", "has_answer": True}],
        ],
    }


def test_instance_keys_follow_the_protocol():
    inst = lme.build_instance(_entry())
    # user turns only; the index counts every turn, from 1
    assert inst.ids == ["s_a_1", "s_a_3", "answer_b_1", "noans_b_3", "noans_c_1"]
    assert inst.gold == {"answer_b_1"}
    assert inst.session == [0, 0, 1, 1, 2]
    assert inst.position == [0, 1, 0, 1, 0]
    assert inst.texts[2] == "I parked on level 3"


def test_exclusion_rule():
    assert lme.scored(_entry())
    assert not lme.scored(_entry("q1_abs"))
    e = _entry()
    e["haystack_sessions"][1][0]["has_answer"] = False
    assert not lme.scored(e), "no user-side evidence turn left"


def test_turn_metrics_by_hand():
    inst = lme.build_instance(_entry())
    e = lme.build_instance(_entry())
    # two gold turns, to exercise recall_all and the ideal DCG
    e.ids[4] = "answer_c_1"
    assert e.gold == {"answer_b_1", "answer_c_1"}
    order = [0, 2, 1, 4, 3]  # gold at ranks 2 and 4
    m = lme.score_ranking(e, order)
    assert m["turn_recall_all@5"] == 1.0
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(2)
    assert m["turn_ndcg_any@5"] == pytest.approx(dcg / idcg)
    m = lme.score_ranking(inst, [0, 1, 2, 3, 4])
    assert m["turn_recall_all@5"] == 1.0
    assert m["turn_ndcg_any@5"] == pytest.approx(1 / math.log2(3))


def test_session_cut_grows_until_k_distinct_sessions():
    inst = lme.build_instance(_entry())
    # turns of session s_a first: the top-2 turns cover one session, so the
    # official conversion widens the cut to three turns to reach two sessions.
    order = [0, 1, 2, 3, 4]
    s_ranked = [lme._strip(inst.ids[i]) for i in order]
    assert s_ranked[:3] == ["s_a", "s_a", "answer_b"]
    got = lme.score_ranking(inst, order)
    # with k=5 the cut spans the whole corpus: every session is in it
    assert got["session_recall_all@5"] == 1.0
    # gold session answer_b sits at the 3rd turn; ideal puts it first
    assert got["session_ndcg_any@5"] == pytest.approx(1 / math.log2(3))


def test_relabelled_turn_does_not_credit_its_session():
    """A 'noans' turn strips to a different id from its session's gold id.

    Ranking the evidence session's *non*-evidence turn first must earn nothing:
    the gold session is only reached at the fifth turn.
    """
    inst = lme.build_instance(_entry())
    assert lme._strip("noans_b_3") != lme._strip("answer_b_1")
    order = [3, 0, 1, 4, 2]  # noans_b_3 first, answer_b_1 last
    got = lme.score_ranking(inst, order)
    assert got["session_ndcg_any@5"] == pytest.approx(1 / math.log2(5))
    assert got["turn_ndcg_any@5"] == pytest.approx(1 / math.log2(5))


def test_complete_appends_missing_indices_in_order():
    assert lme.complete([3, 1], 5) == [3, 1, 0, 2, 4]


def test_okapi_by_hand():
    docs = [["apple", "pie"], ["apple", "tart", "tart"], ["pear"]]
    bm = lme.Okapi(docs, k1=1.5, b=0.75, eps=0.25)
    n, avgdl = 3, 2.0
    idf_apple = math.log(n - 2 + 0.5) - math.log(2 + 0.5)  # negative: in 2 of 3 docs
    idf_tart = math.log(n - 1 + 0.5) - math.log(1 + 0.5)
    idf_pie, idf_pear = idf_tart, idf_tart
    mean_idf = (idf_apple + idf_tart + idf_pie + idf_pear) / 4
    assert idf_apple < 0 and bm.idf["apple"] == pytest.approx(0.25 * mean_idf)

    def term(idf, tf, dl):
        return idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * dl / avgdl))

    s = bm.scores(["apple", "tart"])
    assert s[0] == pytest.approx(term(0.25 * mean_idf, 1, 2))
    assert s[1] == pytest.approx(term(0.25 * mean_idf, 1, 3) + term(idf_tart, 2, 3))
    assert s[2] == 0.0
    # a repeated query term counts once per occurrence
    assert bm.scores(["tart", "tart"])[1] == pytest.approx(2 * term(idf_tart, 2, 3))


def test_official_tokeniser_is_case_sensitive_whitespace():
    assert lme.tok_official("Where did I  park?") == ["Where", "did", "I", "", "park?"]
    assert lme.tok_normalised("Where did I  park?") == ["where", "did", "i", "park"]


def test_split_is_deterministic_and_about_one_fifth():
    ids = [f"q{i:05d}" for i in range(5000)]
    dev = sum(lme.split_of(q) == "dev" for q in ids)
    assert 900 < dev < 1100
    assert [lme.split_of(q) for q in ids[:50]] == [lme.split_of(q) for q in ids[:50]]


# ----------------------------------------------------------------- stats


def test_bootstrap_interval_brackets_a_constant_difference():
    a = np.full(200, 0.7)
    b = np.full(200, 0.5)
    mean, lo, hi = paired_bootstrap(a, b)
    assert mean == pytest.approx(0.2) and lo == pytest.approx(0.2) and hi == pytest.approx(0.2)


def test_bootstrap_interval_covers_zero_for_exchangeable_systems():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 400).astype(float)
    b = a[rng.permutation(400)]
    _, lo, hi = paired_bootstrap(a, b)
    assert lo < 0 < hi


def test_permutation_test():
    assert paired_permutation([1, 0, 1], [1, 0, 1]) == 1.0
    a = np.ones(60)
    b = np.zeros(60)
    p = paired_permutation(a, b, n_perm=2000)
    assert p == pytest.approx(1 / 2001), "never below 1/(n_perm + 1)"
    c = compare(a, b)
    assert c["diff"] == 1.0 and c["n"] == 60


# ------------------------------------------------------- synthetic scoring


def test_returning_the_anchor_earns_answer_credit_but_no_key_credit():
    """The v0.1 inflation, pinned: the anchor often names the successor's project."""
    c = build_corpus(n_people=4, events_per_person=20, seed=0)
    idx = {k: i for i, k in enumerate(c.keys)}
    for q in c.by_task("temporal"):
        anchor = idx[(q.answer_key[0], q.answer_key[1] - 1)]
        if c.facts[anchor]["project"] == q.answer_contains:
            res = [(c.episodes[anchor], 1.0, c.keys[anchor])]
            s = score_query(res, q)
            assert s["answer@1"] == 1.0 and s["key@1"] == 0.0
            return
    pytest.skip("no temporal query whose anchor shares the successor's project")


def test_key_rank_is_reciprocal():
    q = Query(text="", answer_key=(0, 3), task="episodic", answer_contains="atlas")
    res = [("beacon", 1.0, (0, 1)), ("atlas", 0.9, (0, 2)), ("atlas", 0.8, (0, 3))]
    s = score_query(res, q)
    assert s["key@1"] == 0.0 and s["key_rr"] == pytest.approx(1 / 3)
    assert s["answer@1"] == 0.0 and s["answer_rr"] == pytest.approx(1 / 2)
