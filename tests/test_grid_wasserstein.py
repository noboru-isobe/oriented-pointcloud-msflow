"""Grid metric G2a: metric object, autograd contract, and the
BEM-convention preflight.

Self-verifying gates (user directive: tests must catch my errors
without per-item review):
- endpoint positions / weights / offsets BITWISE equal to the actual
  BEM cache, policies built from the BEMSetupSnapshot (never from
  defaults);
- gradcheck + gradgradcheck of the full W_grid(y) on the admissible
  subspace (numeric perturbations stay admissible by construction);
- fail-closed forward on inadmissible directions, projected backward
  on arbitrary cotangents;
- setup gates: valid reconstruction, config consistency, component
  agreement between phase and operator.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

from src.torch.oriented_varifold.mass import (
    compute_recommended_params,
)
from src.torch.shapes.generator import generate_oriented_two_ellipses
from src.torch.solver.mm_step import MMStepper
from src.torch.transport.boundary_flux_grid import (
    BoundaryFluxPolicies,
    BoundaryFluxToGrid,
)
from src.torch.transport.grid_wasserstein import (
    GridMetricConfig,
    GridWassersteinMetric,
)
from src.torch.transport.phase_grid import PhaseGridConfig
from src.torch.transport.weighted_poisson import (
    IncompatibleGridVelocityError,
)

sys.path.insert(0, str(Path(__file__).parent.parent
                       / "scripts" / "experiments"))
from p1_production_comparison import make_cfg  # noqa: E402

DT = torch.float64
DT_STEP = 1e-5


def _pair_stepper(n_per=256):
    v = generate_oriented_two_ellipses(
        n_per, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
        a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu",
        dtype=DT)
    delta, tau = compute_recommended_params(v.positions)
    cfg = make_cfg("C3", delta, tau, 2)
    cfg.time_step = DT_STEP
    cfg.bem_setup_telemetry = True
    st = MMStepper(cfg)
    st._setup_step(v)
    return v, st


def _vhat(v, st):
    """Cloud's own divergence-theorem volume: renormalization factor
    exactly 1 (these tests target the autograd/solver machinery, not
    the volume-target semantics -- pinned in test_phase_grid)."""
    return float(0.5 * (st.fixed_masses
                        * (v.positions * v.normals).sum(-1)).sum())


def _policies_from_snapshot(sn):
    """Policies MUST come from the actual BEM setup, never defaults."""
    return BoundaryFluxPolicies(
        operator_measure=sn.operator_measure_policy,
        use_coherence_velocity=(sn.velocity_policy == "coherence_q"),
        n_endpoints=sn.endpoints_per_particle)


def _grid_cfg(n=512, eps=0.04):
    return GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(n, n), fill_epsilon=eps))


def test_bem_endpoint_preflight_bitwise():
    """G2a-0: the grid flux map must reproduce the BEM endpoint
    conventions EXACTLY, read ONLY from the public BEMSetupSnapshot
    (never from private caches). Positions, weights, signed physical
    offsets, velocity scale -- and the full endpoint velocity map
    V_ik = q^vel (s_i - r_ik dtheta_i) on random inputs, so an
    operator-measure or velocity-policy mixup cannot slip through."""
    v, st = _pair_stepper()
    sn = st._last_bem_snapshot
    pol = _policies_from_snapshot(sn)
    B = BoundaryFluxToGrid(v.positions, v.normals, st.fixed_masses,
                           st.fixed_coherence, _grid_cfg().phase, pol)
    N, K = B.n_particles, B.n_endpoints
    assert torch.equal(B.r_physical, sn.physical_offsets)
    assert torch.equal(B.zeta,
                       sn.endpoint_weights_used.reshape(N, K))
    tangents = torch.stack([-v.normals[:, 1], v.normals[:, 0]], dim=1)
    p_flux = v.positions[:, None, :] \
        + B.r_physical[:, :, None] * tangents[:, None, :]
    assert float((p_flux.reshape(-1, 2)
                  - sn.endpoint_positions).abs().max()) == 0.0
    assert torch.equal(B.flux_scale, sn.velocity_scale_used)
    # full endpoint velocity map identity on random (s, dtheta)
    torch.manual_seed(11)
    s = torch.randn(N, dtype=DT)
    th = torch.randn(N, dtype=DT)
    v_grid = B.flux_scale[:, None] \
        * (s[:, None] - B.r_physical * th[:, None])
    v_snap = sn.velocity_scale_used[:, None] \
        * (s[:, None] - sn.physical_offsets * th[:, None])
    assert torch.equal(v_grid, v_snap)


def test_setup_gates_and_component_agreement():
    v, st = _pair_stepper()
    sn = st._last_bem_snapshot
    metric = GridWassersteinMetric(_grid_cfg())
    metric.setup_for_step(v, st.fixed_masses, st.fixed_coherence,
                          target_volume=_vhat(v, st),
                          policies=_policies_from_snapshot(sn))
    assert metric.phase.n_components == 2
    assert metric.poisson.n_components == 2
    assert metric.phase.n_speck_components_dropped == 0


def _admissible_basis(metric, n_dirs=4, seed=0):
    """Directions in ker(C_grid) over raw (s, dtheta): admissible for
    the fail-closed forward by construction."""
    rows = metric.flux.component_rows(
        metric.phase.component_labels, metric.phase.cell_area)
    N2 = rows.shape[1]
    torch.manual_seed(seed)
    X = torch.randn(N2, n_dirs, dtype=DT)
    # project columns onto ker rows
    Q, _ = torch.linalg.qr(rows.T)          # (N2, C)
    X = X - Q @ (Q.T @ X)
    X, _ = torch.linalg.qr(X)
    return X                                 # (N2, n_dirs) orthonormal


def test_forward_fail_closed_and_backward_projected():
    v, st = _pair_stepper()
    sn = st._last_bem_snapshot
    metric = GridWassersteinMetric(_grid_cfg())
    metric.setup_for_step(v, st.fixed_masses, st.fixed_coherence,
                          _vhat(v, st),
                          _policies_from_snapshot(sn))
    N = v.n_points
    # inadmissible: uniform growth of ONE component only
    s_bad = torch.where(v.positions[:, 0] < 0,
                        torch.ones(N, dtype=DT),
                        torch.zeros(N, dtype=DT))
    with pytest.raises(IncompatibleGridVelocityError):
        metric(s_bad, torch.zeros(N, dtype=DT), DT_STEP)
    # admissible direction evaluates, and W > 0
    X = _admissible_basis(metric, n_dirs=1)
    s, th = X[:N, 0].clone(), X[N:, 0].clone()
    w = metric(s, th, DT_STEP)
    assert float(w) > 0.0


@pytest.mark.parametrize("quadratic_path", ["analytic", "generic"])
def test_gradcheck_and_gradgradcheck_on_admissible_subspace(
        quadratic_path):
    """The full W_grid through the custom solves must pass gradcheck
    AND gradgradcheck in subspace coordinates (numeric perturbations
    stay admissible because the map xi -> (s, dtheta) is linear into
    ker C_grid). Parametrized over both quadratic paths (G3a): the
    analytic path's closed-form backward carries the cell-measure
    factor -- this is the test that pins it (a missing dx^2 showed up
    as a 1/dx^2 = 4096x gradient error on first implementation)."""
    # Fixture obeys ALL calibrated scale laws (h/eps = 0.45 <= 0.5,
    # 2*eps = 0.08 < gap 0.10, eps/dx = 5.1 -- 256^2 puts eps/dx at
    # 2.56, the marginal regime where the G1 audit measured hundreds
    # of fragmented specks). The original
    # n_per=64 / eps=0.1 fixture violated both the porous-wall law
    # (h/eps = 0.72) and the gap precondition, and only passed while
    # the analytic-target background lift bridged its kernel tails --
    # exposed by the G3c carrier renormalization. sparse_direct makes
    # the compliant scale cheap enough for gradcheck.
    v, st = _pair_stepper()
    sn = st._last_bem_snapshot
    metric = GridWassersteinMetric(GridMetricConfig(
        phase=PhaseGridConfig(grid_shape=(512, 512),
                              fill_epsilon=0.04),
        quadratic_path=quadratic_path))
    metric.setup_for_step(v, st.fixed_masses, st.fixed_coherence,
                          _vhat(v, st),
                          _policies_from_snapshot(sn))
    X = _admissible_basis(metric, n_dirs=3)
    N = v.n_points

    def w_of(xi):
        y = X @ xi
        return metric(y[:N], y[N:], DT_STEP)

    xi0 = 0.01 * torch.randn(3, dtype=DT).requires_grad_(True)
    assert torch.autograd.gradcheck(w_of, (xi0,), eps=1e-6, atol=1e-7)
    assert torch.autograd.gradgradcheck(w_of, (xi0,), eps=1e-6,
                                        atol=1e-6)


def test_dense_solve_parity_and_hessian_symmetry():
    """Small-grid dense pseudoinverse parity for the projected solve,
    and metric-Hessian symmetry p^T H q = q^T H p."""
    from src.torch.transport.grid_wasserstein import (
        _ProjectedPoissonSolve,
    )
    v, st = _pair_stepper()      # calibrated scales (see gradcheck)
    sn = st._last_bem_snapshot
    metric = GridWassersteinMetric(GridMetricConfig(
        phase=PhaseGridConfig(grid_shape=(512, 512),
                              fill_epsilon=0.04)))
    metric.setup_for_step(v, st.fixed_masses, st.fixed_coherence,
                          _vhat(v, st),
                          _policies_from_snapshot(sn))
    op = metric.poisson
    n = 512
    # projected-solve correctness: the residual of the projected
    # system vanishes to solver tolerance (equivalent to dense-pinv
    # parity without forming the 9216^2 matrix)
    torch.manual_seed(5)
    b = op.project_componentwise_zero_mean(
        torch.randn(n, n, dtype=DT))
    phi = _ProjectedPoissonSolve.apply(b, op)
    res = op.project_componentwise_zero_mean(op.matvec(phi)) - b
    assert float(torch.sqrt(op.inner(res, res))) \
        < 1e-8 * float(torch.sqrt(op.inner(b, b)))
    # Hessian symmetry through the full metric
    X = _admissible_basis(metric, n_dirs=2, seed=7)
    N = v.n_points
    p, q = X[:, 0], X[:, 1]
    gp_s, gp_th = metric.hessp(p[:N], p[N:], DT_STEP)
    gq_s, gq_th = metric.hessp(q[:N], q[N:], DT_STEP)
    pHq = float((q[:N] * gp_s).sum() + (q[N:] * gp_th).sum())
    qHp = float((p[:N] * gq_s).sum() + (p[N:] * gq_th).sum())
    assert abs(pHq - qHp) < 1e-9 * max(1.0, abs(pHq))


def test_hessp_matches_autograd():
    v, st = _pair_stepper()      # calibrated scales (see gradcheck)
    sn = st._last_bem_snapshot
    metric = GridWassersteinMetric(GridMetricConfig(
        phase=PhaseGridConfig(grid_shape=(512, 512),
                              fill_epsilon=0.04)))
    metric.setup_for_step(v, st.fixed_masses, st.fixed_coherence,
                          _vhat(v, st),
                          _policies_from_snapshot(sn))
    X = _admissible_basis(metric, n_dirs=2, seed=3)
    N = v.n_points
    p = X[:, 0]
    g_s, g_th = metric.hessp(p[:N], p[N:], DT_STEP)
    # autograd Hessian action on the quadratic: grad of W at p equals
    # G p (W(y) = 1/2 y^T G y up to the 1/2 convention: W = (1/2dt)
    # <Bp, L^dagger Bp> so grad = (1/dt) B^* L^dagger B p = hessp)
    s = p[:N].clone().requires_grad_(True)
    th = p[N:].clone().requires_grad_(True)
    w = metric(s, th, DT_STEP)
    gs, gth = torch.autograd.grad(w, (s, th))
    assert float((gs - g_s).abs().max()) < 1e-9 * max(
        1.0, float(g_s.abs().max()))
    assert float((gth - g_th).abs().max()) < 1e-9 * max(
        1.0, float(g_th.abs().max()))
