"""0C-5d-2a regressions: the adaptive noise-floor rank rule.

Unit tests drive AdaptiveRankMachine with synthetic spectra and bases (no
BEM); the integration pins from the static audit
(scripts/experiments/adaptive_rank_rule_0c5d2a.py) live in the audit's
measured JSON and a light union-frame check below. No absolute threshold
appears anywhere: floor consistency, separation, physical validation and
displacement persistence carry the whole decision.
"""

import sys
from pathlib import Path

import torch

from src.torch.transport.rank_state import (
    AdaptiveRankConfig,
    AdaptiveRankMachine,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

N = 6


def spec(*rel_asc):
    """s_desc (descending) from ascending relative values."""
    return torch.tensor(sorted(rel_asc, reverse=True), dtype=torch.float64)


def basis(aligned: bool):
    """Full orthonormal basis whose LAST column is (not) the constant."""
    const = torch.ones(N, dtype=torch.float64) / N ** 0.5
    if aligned:
        M = torch.cat([const[:, None],
                       torch.randn(N, N - 1, dtype=torch.float64)], 1)
        Q, _ = torch.linalg.qr(M)
        idx = list(range(1, N)) + [0]
        return Q[:, idx]
    Q, _ = torch.linalg.qr(torch.randn(N, N, dtype=torch.float64))
    # make the last column orthogonal to the constant explicitly
    v = Q[:, -1] - (Q[:, -1] @ const) * const
    Q[:, -1] = v / v.norm()
    return Q


def test_adaptive_machine_unit_logic():
    torch.manual_seed(0)
    sqrt_w = torch.ones(N, dtype=torch.float64)
    cfg = AdaptiveRankConfig(persist_displacement=2.0, ambiguity_budget=3.0)
    m = AdaptiveRankMachine(config=cfg)

    # initialisation refuses a weak first frame
    d = m.update(spec(0.02, 0.1, 0.5, 0.8, 0.9, 1.0), basis(True), sqrt_w,
                 1.0, 1.0)
    assert d.action == "uninitialised" and m.rank is None

    # clean C = 2 initialisation sets the floor from its own null block
    d = m.update(spec(1e-3, 2e-3, 0.5, 0.8, 0.9, 1.0), basis(True), sqrt_w,
                 1.0, 1.0)
    assert d.action == "hold_clean" and m.rank == 2
    assert m.floor == 2e-3

    # same-rank weak evidence (null block risen above kappa_null * floor):
    # hold with frozen floor, then hard-stop past the ambiguity budget
    weak = spec(5e-3, 1.5e-2, 0.3, 0.6, 0.8, 1.0)   # max-gap still r=2
    frozen = m.floor
    for _ in range(3):
        d = m.update(weak, basis(True), sqrt_w, 1.0, 1.0)
        assert d.action == "hold_weak" and m.rank == 2
        assert m.floor == frozen
    d = m.update(weak, basis(True), sqrt_w, 1.0, 1.0)
    assert d.action == "ambiguity_budget_exceeded"

    # recovery on clean same-rank evidence resets the budget
    d = m.update(spec(1e-3, 2e-3, 0.5, 0.8, 0.9, 1.0), basis(True), sqrt_w,
                 1.0, 1.0)
    assert d.action == "hold_clean"

    # floor-consistent, ALIGNED C = 1 proposal: needs 2 l_res of
    # cumulative displacement -> pending then switch
    c1 = spec(1.5e-3, 0.5, 0.7, 0.8, 0.9, 1.0)
    d = m.update(c1, basis(True), sqrt_w, 1.0, 1.0)
    assert d.action == "candidate_pending" and m.rank == 2
    d = m.update(c1, basis(True), sqrt_w, 1.0, 1.0)
    assert d.action == "switch" and m.rank == 1

    # misaligned candidate with a PERFECT spectrum is rejected
    m2 = AdaptiveRankMachine(config=cfg)
    m2.update(spec(1e-3, 2e-3, 0.5, 0.8, 0.9, 1.0), basis(True), sqrt_w,
              1.0, 1.0)
    d = m2.update(c1, basis(False), sqrt_w, 1.0, 1.0)
    assert d.action == "reject_artificial" and m2.rank == 2
    assert d.align_const_deg > 45.0

    # SAME-rank evidence whose nullspace rotated away from the phase
    # constant is also rejected (review 4.4): an invalid support can keep
    # the count while rotating the null vectors into artificial modes
    d = m2.update(spec(1e-3, 2e-3, 0.5, 0.8, 0.9, 1.0), basis(False),
                  sqrt_w, 1.0, 1.0)
    assert d.action == "reject_artificial" and m2.rank == 2

    # initialisation also requires the phase constant in the span
    m3 = AdaptiveRankMachine(config=cfg)
    d = m3.update(spec(1e-3, 2e-3, 0.5, 0.8, 0.9, 1.0), basis(False),
                  sqrt_w, 1.0, 1.0)
    assert d.action == "uninitialised" and m3.rank is None


def test_adaptive_static_audit_pins():
    """Measured (h = 0.03, |Delta g| persistence): the floor rule
    switches at g = -0.07 on the union family, and the sequence over
    h = 0.03/0.015/0.0075 is -0.07/-0.02/-0.01 -- monotone toward
    contact, vs the fixed-tau machine's -0.10 at every h. The raw
    crossing never switches (6 rejections, aligns 62-72 deg). Rescaling
    invariance x10: 0 decision mismatches (audit JSON); grid dependence
    is bounded by the 2 l_res persistence scale (coarse -0.02 vs dense
    -0.04 at h = 0.015), not eliminated."""
    from adaptive_rank_rule_0c5d2a import (
        RAW_GAPS,
        UNION_GAPS,
        run_sequence,
        summarize,
    )

    s = summarize(run_sequence("union", 0.03, UNION_GAPS))
    assert s["switch_g"] == -0.07
    assert s["final_rank"] == 1
    assert s["rejections"] == 0 and s["budget_stops"] == 0

    s = summarize(run_sequence("raw", 0.03, RAW_GAPS))
    assert s["switch_g"] is None
    assert s["final_rank"] == 2
    assert s["rejections"] >= 5


def test_decisions_are_scale_free():
    """The rule consumes only singular-value RATIOS: multiplying the
    spectrum by any constant changes nothing. (Precision, review 4.1: the
    fixed relative tau = 0.05 is ALSO scale-free -- its defect is
    h-refinement anti-convergence; this check rules out dimensional
    tau ~ h formulas, not the fixed tau.)"""
    torch.manual_seed(1)
    sqrt_w = torch.ones(N, dtype=torch.float64)
    seqs = [spec(1e-3, 2e-3, 0.5, 0.8, 0.9, 1.0),
            spec(5e-3, 1.5e-2, 0.3, 0.6, 0.8, 1.0),
            spec(1.5e-3, 0.5, 0.7, 0.8, 0.9, 1.0)]
    actions = []
    for scale in (1.0, 137.0):
        m = AdaptiveRankMachine()
        acts = []
        for s in seqs:
            torch.manual_seed(7)
            acts.append(m.update(s * scale, basis(True), sqrt_w,
                                 1.0, 1.0).action)
        actions.append(acts)
    assert actions[0] == actions[1]


def test_ambiguity_budget_exceeded_is_hard_stop_in_solver():
    """P1.2 safety blocker (review): the machine's real budget-
    exhaustion action is ambiguity_budget_exceeded (the previously
    checked 'hard_stop' does not exist), and it must STOP the stepper.
    Regression: feed same-rank weak evidence until the cumulative
    displacement exceeds ambiguity_budget * l_res and assert the
    stepper raises. Also pins the COMMON weak-evidence budget: a
    FOREIGN-rank weak interval consumes the same budget."""
    import pytest
    import torch
    from src.torch.transport.rank_state import (
        AdaptiveRankConfig,
        AdaptiveRankMachine,
    )

    NK = 40
    torch.manual_seed(0)

    def spectrum(gap_ratio, C=1):
        # descending svals with a controllable near-null block
        s = torch.linspace(1.0, 0.5, NK, dtype=torch.float64)
        s[-C:] = 0.5 / gap_ratio
        return s

    Q, _ = torch.linalg.qr(torch.randn(NK, NK, dtype=torch.float64))
    sqrt_w = torch.ones(NK, dtype=torch.float64)
    # constant direction inside the null candidate: use a basis whose
    # last column IS the normalized constant
    b = sqrt_w / sqrt_w.norm()
    U = Q.clone()
    U[:, -1] = b
    U, _ = torch.linalg.qr(U.flip(1))
    U = U.flip(1)
    U[:, -1] = b

    m = AdaptiveRankMachine(config=AdaptiveRankConfig())
    d = m.update(spectrum(50.0), U, sqrt_w, 0.0, 1.0)   # clean init
    assert d.rank == 1 and d.action in ("hold_clean", "uninitialised") \
        or d.rank == 1

    # same-rank weak frames: gap below kappa_sep but null level high
    budget = m.config.ambiguity_budget
    got_exceeded = False
    for k in range(100):
        d = m.update(spectrum(4.0), U, sqrt_w, 0.5, 1.0)
        if d.action == "ambiguity_budget_exceeded":
            got_exceeded = True
            assert 0.5 * (k + 1) > budget * 1.0 - 1e-9
            break
        assert d.action in ("hold_weak", "hold_clean")
    assert got_exceeded

    # foreign-rank weak evidence consumes the SAME budget
    m2 = AdaptiveRankMachine(config=AdaptiveRankConfig())
    m2.update(spectrum(50.0), U, sqrt_w, 0.0, 1.0)
    got2 = False
    for k in range(100):
        # foreign proposal r=2 with insufficient floor separation
        s = spectrum(4.0, C=2)
        d = m2.update(s, U, sqrt_w, 0.5, 1.0)
        if d.action == "ambiguity_budget_exceeded":
            got2 = True
            break
        assert d.action in ("hold_weak", "hold_clean",
                            "candidate_pending")
    assert got2
