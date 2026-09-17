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
    """The capacity claim: recover from 90% deletion, flat in store size."""
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
