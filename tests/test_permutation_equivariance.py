"""Permutation equivariance/invariance regressions (review, D2b-1
correction).

The state is an empirical oriented varifold sum_i m_i delta_(x_i, N_i):
ARRAY ORDER IS A PARTICLE LABEL, NOT CONNECTIVITY. Every generic
point-cloud routine must therefore be permutation-equivariant
(per-particle outputs) or -invariant (scalars):

    F(Pi X, Pi N) = Pi F(X, N),        E(Pi X, Pi N) = E(X, N).

Measured: the estimator layer holds to ~1e-16 and one full MM step to
~6e-12 (the spectral projection is basis-independent up to rounding).
Order-consuming diagnostics (shoelace, ordered-polygon metrics,
tangent_source="ordered_loop_oracle") are QUARANTINED: valid only on
generator-ordered fixtures, never part of the point-cloud API -- the
oracle tangent source refuses to run without an explicit cyclic order.
"""

import sys
from pathlib import Path

import pytest
import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import (
    compute_masses,
    compute_recommended_params,
)
from src.torch.solver.dead_point_rules import evaluate_dead_points
from src.torch.solver.redistribution_rules import redistribute_with_rule
from src.torch.transport import compute_coherence

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"
                       / "experiments"))

SIGMA = 0.1


def _fixture():
    from d2b0_timelevel_shadow import snapshot
    torch.manual_seed(7)
    v, n_out, n_in = snapshot(0.35, warp=0.35)
    delta, tau = compute_recommended_params(snapshot(0.5)[0].positions)
    N = v.n_points
    perm = torch.randperm(N)
    inv = torch.argsort(perm)
    v2 = OrientedPointCloudVarifold(positions=v.positions[perm],
                                    angles=v.angles[perm])
    return v, v2, perm, inv, delta, tau


def test_estimator_layer_is_permutation_equivariant():
    v, v2, perm, inv, delta, tau = _fixture()
    m1 = compute_masses(v.positions, delta, tau)
    m2 = compute_masses(v2.positions, delta, tau)
    assert (m2[inv] - m1).abs().max() < 1e-14
    q1 = compute_coherence(v, m1, SIGMA, "wendland_c2")
    q2 = compute_coherence(v2, m2, SIGMA, "wendland_c2")
    assert (q2[inv] - q1).abs().max() < 1e-13
    # coherent perimeter: scalar invariant
    assert abs((m1 * q1).sum() - (m2 * q2).sum()) < 1e-13


def test_dead_point_scores_are_permutation_equivariant():
    v, v2, perm, inv, delta, tau = _fixture()
    m1 = compute_masses(v.positions, delta, tau)
    q1 = compute_coherence(v, m1, SIGMA, "wendland_c2")
    d1 = evaluate_dead_points(v, m1, q1, "refreshed", 0.3,
                              delta_for_kde=delta, tau_for_kde=tau,
                              sigma=SIGMA)
    d2 = evaluate_dead_points(v2, m1[perm], q1[perm], "refreshed", 0.3,
                              delta_for_kde=delta, tau_for_kde=tau,
                              sigma=SIGMA)
    assert (d2.score[inv] - d1.score).abs().max() < 1e-13
    assert torch.equal(d2.keep_mask[inv], d1.keep_mask)
    assert abs(d2.threshold - d1.threshold) < 1e-14


def test_redistribution_rules_are_permutation_equivariant():
    v, v2, perm, inv, delta, tau = _fixture()
    kw = dict(delta=delta, n_iters=5, step_size=0.01, tol=1e-4,
              max_disp_ratio=0.05, mass_tau=tau,
              delta_redist=0.5 * delta)
    for rule, extra in (("legacy_hybrid", dict(q_override="unit")),
                        ("post_mm_frozen", dict(sigma=SIGMA)),
                        ("substep_refreshed", dict(sigma=SIGMA)),
                        ("legacy_hybrid",
                         dict(q_override="unit",
                              tangent_source="local_pca"))):
        p1, a1, _ = redistribute_with_rule(v.positions, v.angles, rule,
                                           **kw, **extra)
        p2, a2, _ = redistribute_with_rule(v2.positions, v2.angles,
                                           rule, **kw, **extra)
        assert (p2[inv] - p1).abs().max() < 1e-11, rule
        assert (a2[inv] - a1).abs().max() < 1e-11, rule


def test_mm_one_step_is_permutation_equivariant():
    """One full MM step (BEM assembly, spectral projection, angle map,
    trust-region optimization) is equivariant to ~1e-11: the projected
    solve does not depend on the (non-unique) SVD basis beyond
    rounding."""
    from d1a_mm_shadow import make_config
    from src.torch.solver.mm_step import MMStepper

    v, v2, perm, inv, delta, tau = _fixture()
    r1 = MMStepper(make_config(2e-5, delta, tau)).step(v)
    r2 = MMStepper(make_config(2e-5, delta, tau)).step(v2)
    assert (r2.varifold.positions[inv]
            - r1.varifold.positions).abs().max() < 1e-9
    dang = (r2.varifold.angles[inv] - r1.varifold.angles + torch.pi) \
        % (2 * torch.pi) - torch.pi
    assert dang.abs().max() < 1e-9
    assert (r2.displacements[inv]
            - r1.displacements).abs().max() < 1e-9
    assert abs(r1.perimeter - r2.perimeter) < 1e-12
    assert abs(r1.wasserstein - r2.wasserstein) < 1e-12
    # rank payload (oracle C=1 config: fields must MATCH, whatever
    # their value -- scalars are permutation-invariant)
    for f in ("detected_rank", "rank_gap_ratio", "rank_abs_level"):
        a, b = getattr(r1, f, None), getattr(r2, f, None)
        if isinstance(a, float) and isinstance(b, float):
            assert abs(a - b) < 1e-9, f
        else:
            assert a == b, f


def test_order_consuming_diagnostics_are_quarantined():
    """tangent_source='ordered_loop_oracle' consumes a cyclic order the
    point cloud does not carry -- it must refuse to run without an
    explicit `loops` order (and is documented DIAGNOSTIC ONLY)."""
    v, v2, perm, inv, delta, tau = _fixture()
    with pytest.raises(ValueError):
        redistribute_with_rule(
            v.positions, v.angles, "legacy_hybrid", q_override="unit",
            tangent_source="ordered_loop_oracle",
            delta=delta, mass_tau=tau, delta_redist=0.5 * delta)
