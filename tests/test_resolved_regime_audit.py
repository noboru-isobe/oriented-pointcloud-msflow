"""Closure patch 2: resolved-regime audit v2 invariants.

Permutation invariance of the audit geometry, the coupling-independent
weight control, the two-tier validity rule, and the linearized
q-channel additive identity -- small-N so the whole file stays fast.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent
                       / "scripts" / "experiments"))
from resolved_regime_audit import (  # noqa: E402
    d_super_norms,
    facing_stats,
    pair_frames,
    run_case,
)


def test_facing_stats_permutation_invariant():
    v, _, labels = pair_frames(48, 0.1)
    ref = facing_stats(v.positions, v.normals, labels)

    perm = torch.randperm(v.n_points)
    got = facing_stats(v.positions[perm], v.normals[perm], labels[perm])
    for k in ("n_facing", "h_same_q10", "h_same_median",
              "h_face_min", "h_face_median", "gap_facing_min"):
        assert abs(ref[k] - got[k]) < 1e-12, k


def test_d_super_norms_weight_semantics():
    torch.manual_seed(0)
    n = 40
    s_iso = torch.randn(n, dtype=torch.float64)
    s_pair = s_iso + 0.01 * torch.randn(n, dtype=torch.float64)
    m_ref = torch.rand(n, dtype=torch.float64) + 0.5
    # pair weights that CRUSH half the domain (overlap analogue): the
    # reference-weighted norm must not change, the pair-weighted one may
    m_pair = m_ref.clone()
    m_pair[: n // 2] *= 1e-6
    r = d_super_norms(s_pair, s_iso, m_ref, m_pair)
    r2 = d_super_norms(s_pair, s_iso, m_ref, m_ref)
    assert abs(r["d_super_mref"] - r2["d_super_mref"]) < 1e-15
    assert r["d_super_pairweight"] != r["d_super_mref"]
    assert r["num_mref"] > 0 and r["den_mref"] > 0


def test_far_control_case():
    """Far-apart control: m_pair ~ m_ref (coupling-independent weight
    control), energy side exactly zero under the compact kernel, and
    the additive identity holds."""
    row = run_case(48, 0.8, 0.1, 0.1)
    assert row["valid"]
    assert row["m_pair_vs_mref_reldev"] < 1e-12
    lin = row["linearized"]
    assert lin["linearized_energy_side"] < 1e-12
    assert lin["additivity_residual"] < 1e-8


def test_two_tier_validity_rejects_bad_solve():
    """A crippled optimizer budget must fail the QUANTITATIVE rule
    while the row is still recorded with its telemetry."""
    from resolved_regime_audit import one_step, pair_frames
    from src.torch.oriented_varifold.mass import (
        compute_recommended_params,
    )
    import resolved_regime_audit as rra

    v, _, labels = pair_frames(48, 0.1)
    delta, tau = compute_recommended_params(v.positions)
    geo = facing_stats(v.positions, v.normals, labels)

    orig = rra._cfg

    def crippled(rank, d, t, sp, sa):
        cfg = orig(rank, d, t, sp, sa)
        cfg.optimizer_max_iter = 1
        cfg.optimizer_tol = 1e-16
        return cfg

    rra._cfg = crippled
    try:
        _, _, tele = one_step(v, 2, float(delta), float(tau), 0.1, 0.1,
                              geo["h_same_q10"])
    finally:
        rra._cfg = orig
    assert not tele["solve_valid"]
    assert tele["relative_gradient_norm"] is not None
    assert tele["optimizer_message"] is not None


def test_linearized_additivity_coupled_case():
    row = run_case(48, 0.1, 0.1, 0.1)
    lin = row["linearized"]
    assert lin["additivity_residual"] < 1e-8
    # coupled regime at coarse N: full defect is O(1) and energy side
    # dominates the metric side (carrier channel)
    assert lin["linearized_full_defect"] > 0.1
    assert lin["linearized_energy_side"] > lin["linearized_metric_side"]
