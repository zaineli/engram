"""Tests for the conjunction code and its index.

Two of these fail on v0.1 by construction: the alphabetical truncation and the
dropped single digits. The rest pin the index against a direct computation of
its definition, so a faster implementation can never quietly compute something
else.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from engram import ConjunctionIndex, ConjunctiveBinder, content_terms

# Thirty filler words that all sort before the topic words.
FILLER = [f"a{c}{d}word" for c in "bcdefg" for d in "hijkl"]


def test_long_episode_keeps_topic_terms_that_sort_late():
    """v0.1 kept the alphabetically first 24 terms; here they all sort first."""
    b = ConjunctiveBinder()
    text = " ".join(FILLER) + " the zeppelin landed near the yacht"
    got = b.conjunctions(text)
    assert ("zeppelin",) in got and ("yacht",) in got
    assert ("yacht", "zeppelin") in got

    idx = ConjunctionIndex()
    idx.add(0, b.encode(text))
    idx.add(1, b.encode("a quiet morning at the harbour office"))
    keys, s = idx.score(b.encode("where did the zeppelin land near the yacht"))
    assert keys[int(np.argmax(s))] == 0 and s.max() > 0


def test_single_digits_are_content_terms():
    """v0.1 dropped one-character tokens, so 'week 7' lost its 7 but 'week 14' did not."""
    assert content_terms("In week 7 on Monday") == ["week", "7", "monday"]
    assert content_terms("Priya's notes") == ["priya's", "notes"]
    assert "s" not in content_terms("Priya s notes")

    b = ConjunctiveBinder()
    idx = ConjunctionIndex()
    for i, w in enumerate(range(1, 10)):
        idx.add(i, b.encode(f"In week {w} on Monday, Priya reviewed the atlas runbook."))
    keys, s = idx.score(b.encode("What did Priya review in week 7 on Monday?"))
    assert keys[int(np.argmax(s))] == 6
    assert np.sum(s == s.max()) == 1, "week 7 must be separable from weeks 1-9"


def test_window_bounds_which_terms_bind():
    text = "alpha " + " ".join(f"pad{i}" for i in range(9)) + " omega"
    assert ("alpha", "omega") not in ConjunctiveBinder(window=8).conjunctions(text)
    assert ("alpha", "omega") in ConjunctiveBinder(window=None).conjunctions(text)
    # Adjacent terms bind at any window.
    assert ("alpha", "pad0") in ConjunctiveBinder(window=2).conjunctions(text)
    assert ("alpha", "pad1") not in ConjunctiveBinder(window=2).conjunctions(text)


def test_pairs_are_unordered_and_never_self_pairs():
    b = ConjunctiveBinder()
    assert np.array_equal(b.encode("alpha beta gamma"), b.encode("gamma alpha beta"))
    assert all(len(set(c)) == len(c) for c in b.conjunctions("echo echo echo delta"))


def test_order_one_is_a_bag_of_terms():
    got = ConjunctiveBinder(order=1).conjunctions("alpha beta gamma")
    assert got == {("alpha",), ("beta",), ("gamma",)}


def test_order_three_adds_triples_inside_the_window():
    got = ConjunctiveBinder(order=3, window=3).conjunctions("alpha beta gamma delta")
    assert ("alpha", "beta", "gamma") in got
    assert ("alpha", "beta", "delta") not in got  # delta is 3 terms from alpha


def _brute_force(index_codes: dict[int, set[int]], query: set[int], weighting: str) -> dict[int, float]:
    """The definition in ConjunctionIndex's docstring, computed the slow way."""
    n = len(index_codes)
    df: dict[int, int] = {}
    for code in index_codes.values():
        for u in code:
            df[u] = df.get(u, 0) + 1

    def w(u: int) -> float:
        if weighting == "binary":
            return 1.0
        return math.log(1 + (n - df[u] + 0.5) / (df[u] + 0.5))

    q = {u for u in query if u in df}
    qn = math.sqrt(sum(w(u) ** 2 for u in q))
    out = {}
    for key, code in index_codes.items():
        en = math.sqrt(sum(w(u) ** 2 for u in code))
        dot = sum(w(u) ** 2 for u in q & code)
        out[key] = dot / (qn * en) if qn > 0 and en > 0 else 0.0
    return out


@pytest.mark.parametrize("weighting", ["idf", "binary"])
def test_index_equals_its_definition(weighting):
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(40)]
    b = ConjunctiveBinder()
    idx = ConjunctionIndex(weighting=weighting)
    codes: dict[int, set[int]] = {}
    for key in range(60):
        text = " ".join(rng.choice(vocab) for _ in range(rng.randint(3, 20)))
        c = b.encode(text)
        idx.add(key, c)
        codes[key] = set(c.tolist())
    for key in (3, 17, 41):  # removals must update the counts too
        idx.remove(key)
        del codes[key]
    for _ in range(10):
        qc = b.encode(" ".join(rng.choice(vocab) for _ in range(rng.randint(2, 6))))
        keys, s = idx.score(qc)
        want = _brute_force(codes, set(qc.tolist()), weighting)
        assert sorted(keys) == sorted(want)
        for key, v in zip(keys, s):
            assert v == pytest.approx(want[key], abs=1e-9)


def test_binary_index_ranks_like_the_unweighted_match():
    rng = random.Random(3)
    vocab = [f"t{i}" for i in range(25)]
    b = ConjunctiveBinder()
    idx = ConjunctionIndex(weighting="binary")
    codes = []
    for key in range(30):
        c = b.encode(" ".join(rng.choice(vocab) for _ in range(8)))
        idx.add(key, c)
        codes.append(c)
    qc = b.encode("t1 t2 t3 t4")
    keys, s = idx.score(qc)
    direct = np.array([ConjunctiveBinder.match(qc, codes[k]) for k in keys])
    # Same ranking: the index drops query units the store never saw, a
    # constant factor within one query.
    assert np.array_equal(np.argsort(-s, kind="stable"), np.argsort(-direct, kind="stable"))


def test_idf_prefers_the_rare_shared_conjunction():
    """Two episodes each share one query term. Binary weighting ties them; IDF must not."""
    b = ConjunctiveBinder()
    texts = [f"ledger filler{i}" for i in range(20)] + ["ledger review", "atlas review"]
    q = b.encode("ledger atlas")
    scores = {}
    for weighting in ("binary", "idf"):
        idx = ConjunctionIndex(weighting=weighting)
        for i, t in enumerate(texts):
            idx.add(i, b.encode(t))
        keys, s = idx.score(q)
        scores[weighting] = dict(zip(keys, s))
    assert scores["binary"][20] == pytest.approx(scores["binary"][21])
    assert scores["idf"][21] > 10 * scores["idf"][20]


def test_df_tracks_adds_and_removes():
    b = ConjunctiveBinder()
    idx = ConjunctionIndex()
    u = b.encode("alpha")[0]
    idx.add(0, b.encode("alpha beta"))
    idx.add(1, b.encode("alpha gamma"))
    assert idx.df(u) == 2 and len(idx) == 2
    idx.remove(0)
    assert idx.df(u) == 1 and 0 not in idx
    idx.add(1, b.encode("delta"))  # re-adding a key replaces it
    assert idx.df(u) == 0 and len(idx) == 1
    keys, s = idx.score(b.encode("alpha"))
    assert keys == [1] and s[0] == 0.0


def _brute_bm25(texts: dict[int, str], query: str, b: ConjunctiveBinder, k1=1.5, bb=0.75) -> dict[int, float]:
    """BM25 over conjunction tuples, from the definition, with no hashing."""
    counts = {k: b.conjunction_counts(t) for k, t in texts.items()}
    lengths = {k: len(content_terms(t)) for k, t in texts.items()}
    n = len(texts)
    avg = sum(lengths.values()) / n
    df: dict[tuple, int] = {}
    for c in counts.values():
        for u in c:
            df[u] = df.get(u, 0) + 1
    q = set(b.conjunction_counts(query))
    out = {}
    for k, c in counts.items():
        s = 0.0
        for u in q & set(c):
            idf = math.log(1 + (n - df[u] + 0.5) / (df[u] + 0.5))
            tf = c[u]
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - bb + bb * lengths[k] / avg))
        out[k] = s
    return out


@pytest.mark.parametrize("order,window", [(1, 4), (2, 4), (2, None), (3, 3)])
def test_bm25_weighting_equals_its_definition(order, window):
    rng = random.Random(11)
    vocab = [f"v{i}" for i in range(30)]
    b = ConjunctiveBinder(order=order, window=window)
    idx = ConjunctionIndex(weighting="bm25")
    texts = {}
    for key in range(40):
        t = " ".join(rng.choice(vocab) for _ in range(rng.randint(2, 25)))
        texts[key] = t
        idx.add(key, *b.encode_counts(t))
    idx.remove(5)
    del texts[5]
    for _ in range(8):
        q = " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 5)))
        keys, s = idx.score(b.encode(q))
        want = _brute_bm25(texts, q, b)
        for key, v in zip(keys, s):
            assert v == pytest.approx(want[key], abs=1e-9)


def test_counts_follow_occurrences():
    b = ConjunctiveBinder(window=3)
    c = b.conjunction_counts("rain rain snow rain")
    assert c[("rain",)] == 3 and c[("snow",)] == 1
    # positions (0,2) (1,2) (2,3) pair rain with snow; (0,1) (1,3) are rain-rain, not pairs
    assert c[("rain", "snow")] == 3
    units, tf, length = b.encode_counts("rain rain snow rain")
    assert length == 4 and tf.sum() == 3 + 1 + 3


def test_switching_weighting_keeps_the_counts():
    b = ConjunctiveBinder()
    idx = ConjunctionIndex(weighting="bm25")
    for i, t in enumerate(["alpha beta", "beta gamma gamma", "alpha alpha delta"]):
        idx.add(i, *b.encode_counts(t))
    q = b.encode("alpha gamma")
    before = idx.score(q)[1].copy()
    idx.set_weighting("idf")
    idx.set_weighting("bm25")
    assert np.array_equal(idx.score(q)[1], before)
    with pytest.raises(ValueError):
        idx.set_weighting("tfidf")
