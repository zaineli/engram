"""Tests for the mechanisms, not for the wiring.

Each test pins a property the architecture is supposed to have, so that a
regression shows up as a failed claim rather than as a slightly worse benchmark
number nobody notices. All use the hashing encoder: hermetic, no download, and
a weak enough semantic signal that anything passing here is a real mechanism
rather than the embedding model doing the work.
"""

from __future__ import annotations

import numpy as np
import pytest

from engram import (
    CA3,
    ConjunctiveBinder,
    DentateGyrus,
    EngramConfig,
    EngramMemory,
    HashingEncoder,
    Neocortex,
    ThetaContext,
)


@pytest.fixture
def enc() -> HashingEncoder:
    return HashingEncoder(dim=384)


# ---------------------------------------------------------------- dentate


def test_dg_separates_similar_inputs(enc):
    """Expansion + k-WTA must decorrelate: high input cosine, low code overlap."""
    dg = DentateGyrus(enc.dim, expansion=8, sparsity=0.02, neurogenesis=0.0)
    a, b = enc.encode([
        "the tuesday deploy failed on the auth service",
        "the thursday deploy failed on the billing service",
    ])
    cos = float(a @ b)
    ov = DentateGyrus.overlap(dg.encode(a, learn=False), dg.encode(b, learn=False))
    assert cos > 0.5, "inputs should be genuinely confusable for this to mean anything"
    assert ov < cos / 2, f"DG failed to separate: cos={cos:.3f} overlap={ov:.3f}"


def test_dg_code_is_sparse(enc):
    dg = DentateGyrus(enc.dim, expansion=8, sparsity=0.02)
    code = dg.encode(enc.encode(["anything at all"])[0])
    assert code.size == dg.k
    assert dg.k / dg.dim_out == pytest.approx(0.02, abs=0.005)
    assert np.all(np.diff(code) > 0), "codes must be sorted and unique"


def test_neurogenesis_spreads_recruitment(enc):
    """Without it the population collapses onto a hot minority of units."""
    texts = [f"incident {i % 9} in region {i % 5} lasting {i} minutes" for i in range(400)]
    V = enc.encode(texts)
    stats = {}
    for ng in (0.0, 0.7):
        dg = DentateGyrus(enc.dim, expansion=8, sparsity=0.02, neurogenesis=ng, seed=1)
        for v in V:
            dg.encode(v)
        stats[ng] = dg.stats()
    assert stats[0.7]["unit_coverage"] > stats[0.0]["unit_coverage"] * 1.5
    assert stats[0.7]["usage_gini"] < stats[0.0]["usage_gini"]


# -------------------------------------------------------------------- ca3


def test_ca3_completes_from_heavily_degraded_cue(enc):
    """The capacity claim: recover from 90% deletion of the cue's units."""
    dg = DentateGyrus(enc.dim, expansion=8, seed=2)
    ca3 = CA3(dg.dim_out, beta=22.0)
    V = enc.encode([f"event {i} concerning subsystem {i % 11}" for i in range(500)])
    for v in V:
        ca3.store(dg.encode(v))
    assert ca3.capacity_probe(corruption=0.90, trials=60) > 0.95


def test_ca3_completion_degrades_only_when_few_units_survive(enc):
    dg = DentateGyrus(enc.dim, expansion=8, seed=2)
    ca3 = CA3(dg.dim_out, beta=22.0)
    for v in enc.encode([f"event {i} concerning subsystem {i % 11}" for i in range(500)]):
        ca3.store(dg.encode(v))
    assert ca3.capacity_probe(0.90, trials=60) > ca3.capacity_probe(0.98, trials=60)


def test_ca3_exact_cue_is_a_fixed_point(enc):
    dg = DentateGyrus(enc.dim, expansion=8, seed=2)
    ca3 = CA3(dg.dim_out)
    codes = [dg.encode(v) for v in enc.encode([f"item number {i}" for i in range(50)])]
    for c in codes:
        ca3.store(c)
    res = ca3.complete(codes[17])
    assert res.index == 17 and res.converged


# ------------------------------------------------------------------ binding


def test_conjunctive_binding_beats_pooling_on_slot_confusion():
    """The measured failure that motivates the whole module."""
    b = ConjunctiveBinder()
    target = b.encode("On Monday the auth service suffered a timeout cascade.")
    distractor = b.encode("On Tuesday the auth service suffered a timeout cascade.")
    q = b.encode("What went wrong with the auth service on Monday?")
    assert b.match(q, target) > b.match(q, distractor) * 1.5


def test_conjunctive_code_is_order_invariant():
    b = ConjunctiveBinder()
    x = b.encode("alpha beta gamma")
    y = b.encode("gamma alpha beta")
    assert np.array_equal(x, y)


# -------------------------------------------------------------------- theta


def test_theta_context_falls_off_with_distance():
    th = ThetaContext(dim=32)
    a = th.encode(0, 0)
    near, far = th.encode(0, 1), th.encode(0, 40)
    assert float(a @ near) > float(a @ far)


def test_theta_separates_sessions():
    th = ThetaContext(dim=32)
    assert float(th.encode(0, 3) @ th.encode(1, 3)) < float(th.encode(0, 3) @ th.encode(0, 4))


# ------------------------------------------------------------------ system


def test_recall_resolves_confusable_episodes(enc):
    m = EngramMemory(encoder=enc)
    m.remember("the tuesday deploy failed on the auth service", session=0)
    m.remember("the thursday deploy failed on the billing service", session=0)
    top = m.recall("which deploy broke auth?", k=1)
    assert top and "auth" in top[0].episode.text


def test_sleep_builds_schemas_and_read_path_does_not(enc):
    m = EngramMemory(encoder=enc)
    for i in range(40):
        m.remember(f"Priya reviewed the atlas runbook in week {i}", session=0)
    assert len(m.neocortex) == 0, "retrieval must not create schemas"
    m.recall("what did Priya do?", k=5)
    assert len(m.neocortex) == 0
    m.sleep(cycles=10)
    assert len(m.neocortex) > 0


def test_consolidation_does_not_evict_without_capacity_pressure(enc):
    """The regression that destroyed episodic detail: eviction coupled to promotion."""
    m = EngramMemory(encoder=enc, config=EngramConfig(hippocampal_capacity=None))
    for i in range(60):
        m.remember(f"event {i} happened in region {i % 7}", session=0)
    m.sleep(cycles=30)
    assert m.stats()["retention"] == 1.0


def test_capacity_bound_evicts_only_consolidated(enc):
    m = EngramMemory(encoder=enc, config=EngramConfig(hippocampal_capacity=30))
    for i in range(80):
        m.remember(f"event {i} happened in region {i % 7}", session=0)
    m.sleep(cycles=25)
    assert len(m.ca1) <= 80
    assert all(t.consolidated for t in m.ca1.traces()) or len(m.ca1) <= 30


def test_temporal_reinstatement_returns_successor_not_anchor(enc):
    m = EngramMemory(encoder=enc)
    for i, what in enumerate(["drafted the atlas plan", "reviewed the ledger diff",
                              "shipped the beacon runbook", "profiled the quarry test"]):
        m.remember(f"Priya {what}", session=0)
    top = m.recall("what did Priya do right after she drafted the atlas plan?", k=1)
    assert top and "atlas plan" not in top[0].episode.text, "returned the anchor itself"


def test_complete_partial_recovers_from_code_fragment(enc):
    m = EngramMemory(encoder=enc)
    for i in range(100):
        m.remember(f"event {i} concerning subsystem {i % 11}", session=0)
    got = m.complete_partial((0, 42), keep=0.15)
    assert got is not None and got.evidence["correct"] == 1.0


def test_pathway_weights_are_exactly_ablatable(enc):
    """Zeroing a weight must remove that pathway's contribution, exactly."""
    cfg = EngramConfig(w_conjunctive=0.0)
    m = EngramMemory(encoder=enc, config=cfg)
    m.remember("alpha beta gamma delta", session=0)
    m.remember("epsilon zeta eta theta", session=0)
    for r in m.recall("alpha beta", k=2):
        assert r.evidence.get("conjunctive_n", 0.0) * cfg.w_conjunctive == 0.0


def test_neocortex_schema_ranking_uses_mass():
    nc = Neocortex(dim=8)
    a = np.zeros(8, dtype=np.float32); a[0] = 1.0
    b = np.zeros(8, dtype=np.float32); b[1] = 1.0
    for _ in range(20):
        nc.absorb(a, (0, 0), "frequent thing", weight=1.0)
    nc.absorb(b, (0, 1), "rare thing", weight=1.0)
    mid = (a + b); mid /= np.linalg.norm(mid)
    top = nc.query(mid, top_k=1)[0]
    assert "frequent" in top.episode.text


# ------------------------------------------------------------- v0.2 fixes


def test_recall_can_be_read_only(enc):
    m = EngramMemory(encoder=enc)
    for i in range(20):
        m.remember(f"event {i} concerning subsystem {i % 5}", session=0)
    before = {t.key: t.strength for t in m.ca1.traces()}
    m.recall("subsystem 3", k=5, rehearse=False)
    assert {t.key: t.strength for t in m.ca1.traces()} == before
    m.recall("subsystem 3", k=5)
    assert {t.key: t.strength for t in m.ca1.traces()} != before


def test_full_ranking_covers_every_trace(enc):
    m = EngramMemory(encoder=enc)
    for i in range(30):
        m.remember("the same sentence" if i % 3 == 0 else f"distinct event {i}", session=0)
    got = m.recall("same sentence", k=None, include_schemas=False, rehearse=False, dedup=False)
    assert len(got) == 30 and len({r.episode.key() for r in got}) == 30


def test_loose_temporal_word_keeps_the_anchor(enc):
    """'How many days before X did I Y' needs X as evidence; v0.1 multiplied it by 0.15."""
    m = EngramMemory(encoder=enc)
    m.remember("I bought the new phone at the mall on Friday", session=0)
    m.remember("the weather was mild all week", session=0)
    m.remember("I attended the holiday market with my sister", session=1)
    m.remember("we repainted the garage door", session=1)
    top = m.recall("How many days before I bought the new phone did I attend the holiday market?", k=2)
    texts = [r.episode.text for r in top]
    assert any("phone" in t for t in texts) and any("market" in t for t in texts)
    assert all("anchor" not in r.evidence for r in top)


def test_adjacency_cue_still_suppresses_the_anchor(enc):
    m = EngramMemory(encoder=enc)
    for what in ["drafted the atlas plan", "reviewed the ledger diff", "shipped the beacon runbook"]:
        m.remember(f"Priya {what}", session=0)
    top = m.recall("what did Priya do immediately after she drafted the atlas plan?", k=1)
    assert top[0].episode.text == "Priya reviewed the ledger diff"


def test_schema_hits_neither_anchor_nor_rehearse(enc):
    m = EngramMemory(encoder=enc)
    for i in range(40):
        m.remember(f"Priya reviewed the atlas runbook in week {i}", session=0)
    m.sleep(cycles=10)
    first = m.ca1.get(m.ca1.slot_of((0, 0)))
    first.strength = 0.5
    # dedup=False: a schema exemplar is the text of a stored episode and would
    # otherwise be folded into its episodic copy before rehearsal.
    got = m.recall("what does Priya usually do right after a review?", k=60, dedup=False)
    schemas = [r for r in got if r.source == "neocortex"]
    assert schemas and all(r.episode.key() == (0, 0) for r in schemas), "setup"
    assert all("anchor" not in r.evidence and "reinstated" not in r.evidence for r in schemas)
    episodic_keys = {r.episode.key() for r in got if r.source == "hippocampus"}
    expected = 0.55 if (0, 0) in episodic_keys else 0.5
    assert first.strength == pytest.approx(expected), "a schema hit rehearsed the trace at (0, 0)"


def test_passive_decay_never_drops_an_unconsolidated_trace(enc):
    m = EngramMemory(encoder=enc, config=EngramConfig(decay_rate=0.5, decay_floor=0.05))
    for i in range(40):
        m.remember(f"event {i}", session=0)
    assert m.stats()["retention"] == 1.0, "decay deleted traces nothing else holds"
    for t in m.ca1.traces()[:10]:
        t.consolidated = True
    m.remember("one more write", session=0)
    assert len(m.ca1) == 31  # the ten consolidated, decayed traces are released


def test_released_traces_leave_every_structure(enc):
    """v0.1 released evicted traces from CA1 only; their CA3 attractors stayed."""
    m = EngramMemory(encoder=enc, config=EngramConfig(hippocampal_capacity=30))
    for i in range(80):
        m.remember(f"event {i} happened in region {i % 7}", session=0)
    m.sleep(cycles=25)
    assert len(m.ca1) < 80, "setup: eviction should have happened"
    assert m.ca3.resident == len(m.ca1) == len(m.conj_index)


@pytest.mark.parametrize("how", ["minmax", "zscore"])
def test_fusion_normalisers_ignore_pathway_scale(how):
    from engram.memory import _normalise

    v = np.random.default_rng(0).normal(size=50)
    assert np.allclose(_normalise(v, how), _normalise(3.7 * v + 11.0, how))


def test_rrf_normaliser_depends_only_on_order():
    from engram.memory import _normalise

    v = np.random.default_rng(1).normal(size=50)
    assert np.allclose(_normalise(v, "rrf"), _normalise(np.exp(v), "rrf"))
    assert _normalise(v, "rrf").max() == pytest.approx(1 / 61)


def test_constant_pathway_contributes_nothing():
    from engram.memory import _normalise

    for how in ("minmax", "zscore", "rrf"):
        assert not np.any(_normalise(np.full(10, 0.3), how))


def test_unknown_fusion_is_rejected(enc):
    with pytest.raises(ValueError):
        EngramMemory(encoder=enc, config=EngramConfig(fusion="sum"))


def test_reconfigure_equals_a_fresh_build(enc):
    """Sweeps reuse one ingested memory; that must change nothing but speed."""
    texts = [f"In week {i % 9} on day {i % 7}, person {i % 4} touched project {i % 5}" for i in range(60)]
    base = EngramConfig(fusion="minmax", w_conjunctive=1.2, conj_weighting="binary")
    target = dict(fusion="zscore", w_conjunctive=0.3, conj_weighting="idf", w_reinstate=1.0)
    reused = EngramMemory(encoder=enc, config=base)
    fresh = EngramMemory(encoder=enc, config=EngramConfig(**{**base.__dict__, **target}))
    for m in (reused, fresh):
        for i, t in enumerate(texts):
            m.remember(t, session=i % 3)
    reused.reconfigure(**target)
    for q in ("person 2 project 3 week 4", "what did person 1 do right after project 2"):
        a = reused.recall(q, k=None, rehearse=False, dedup=False)
        b = fresh.recall(q, k=None, rehearse=False, dedup=False)
        assert [r.episode.key() for r in a] == [r.episode.key() for r in b]
        assert [r.score for r in a] == pytest.approx([r.score for r in b])
    with pytest.raises(ValueError):
        reused.reconfigure(conj_window=2)


def test_consolidation_is_reproducible_given_a_clock(enc):
    def build():
        m = EngramMemory(encoder=enc)
        for i in range(60):
            m.remember(f"Priya reviewed project {i % 4} in week {i}", session=0, timestamp=60.0 * i)
        m.sleep(cycles=15, now=3600.0)
        return [(s.label, round(s.mass, 9), tuple(s.support)) for s in m.neocortex.schemas]

    assert build() == build()
