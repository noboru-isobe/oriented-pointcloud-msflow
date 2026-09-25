"""Grid metric G1: boundary-flux tangent map B.

Pins: exact total-flux identity (kernel deposits unit mass by
construction), agreement of component rows with the scattered field's
component integrals, differentiability (double backward included --
the map is linear), BEM endpoint-convention parity fields, and the
inactive-RHS fail-closed guard on an unresolved support.
"""

import math

import pytest
import torch

from src.torch.shapes.generator import (
    generate_oriented_circle,
    generate_oriented_two_ellipses,
)
from src.torch.transport.boundary_flux_grid import (
    BoundaryFluxPolicies,
    BoundaryFluxToGrid,
)
from src.torch.transport.phase_grid import (
    CurrentToPhase,
    PhaseGridConfig,
)
from src.torch.transport.weighted_poisson import (
    IncompatibleGridVelocityError,
    WeightedPoissonConfig,
    WeightedPoissonOperator,
)

DT = torch.float64


def _masses(v):
    from src.torch.oriented_varifold.mass import (
        compute_masses, compute_recommended_params,
    )
    delta, tau = compute_recommended_params(v.positions)
    return compute_masses(v.positions, delta, tau, "wendland_c2")


def _flux(v, m, q=None, cfg=None, **pol):
    q = q if q is not None else torch.ones_like(m)
    cfg = cfg or PhaseGridConfig(grid_shape=(256, 256),
                                 fill_epsilon=0.08)
    return BoundaryFluxToGrid(v.positions, v.normals, m, q, cfg,
                              BoundaryFluxPolicies(**pol))


def test_total_flux_identity_and_volume_change():
    """integral of delta_rho == total endpoint flux == perimeter * s
    for uniform normal speed on a circle (dV/dt identity)."""
    v = generate_oriented_circle(192, 1.0, (0.0, 0.0), "cpu", DT)
    m = _masses(v)
    B = _flux(v, m)
    s = torch.full((192,), 0.01, dtype=DT)
    dth = torch.zeros(192, dtype=DT)
    drho = B(s, dth)
    total_grid = float(drho.sum()) * B.stencil.cell_area
    total_analytic = float(B.total_flux(s, dth))
    assert abs(total_grid - total_analytic) < 1e-12 * abs(
        total_analytic)                       # unit-mass deposition
    # dV = perimeter * s with the KDE-mass perimeter estimate
    assert abs(total_analytic - float(m.sum()) * 0.01) < 1e-14


def test_component_rows_match_scattered_integrals():
    v2 = generate_oriented_two_ellipses(
        256, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
        a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu", dtype=DT)
    m2 = _masses(v2)
    cfg = PhaseGridConfig(grid_shape=(512, 512), fill_epsilon=0.04)
    pg = CurrentToPhase(cfg).reconstruct(v2, m2, 2 * math.pi * 0.4)
    assert pg.n_components == 2
    B = _flux(v2, m2, cfg=cfg)
    torch.manual_seed(0)
    N = v2.n_points
    s = 0.01 * torch.randn(N, dtype=DT)
    dth = 0.05 * torch.randn(N, dtype=DT)
    drho = B(s, dth)
    rows = B.component_rows(pg.component_labels, pg.cell_area)
    y = torch.cat([s, dth])
    for c in range(2):
        direct = float(
            drho[pg.component_labels == c].sum()) * pg.cell_area
        via_row = float(rows[c] @ y)
        assert abs(direct - via_row) < 1e-10 * max(1.0, abs(direct))


def test_linearity_and_double_backward():
    v = generate_oriented_circle(96, 1.0, (0.0, 0.0), "cpu", DT)
    m = _masses(v)
    B = _flux(v, m)
    s = torch.randn(96, dtype=DT, requires_grad=True)
    dth = torch.randn(96, dtype=DT, requires_grad=True)
    out = (B(s, dth) ** 2).sum()
    g_s, g_th = torch.autograd.grad(out, (s, dth), create_graph=True)
    gg = torch.autograd.grad(g_s.sum() + g_th.sum(), (s, dth),
                             allow_unused=False)
    assert all(torch.isfinite(x).all() for x in gg)


def test_policy_fields_recorded():
    v = generate_oriented_circle(64, 1.0, (0.0, 0.0), "cpu", DT)
    m = _masses(v)
    q = 0.5 * torch.ones(64, dtype=DT)
    Bc = _flux(v, m, q=q, operator_measure="carrier",
               use_coherence_velocity=False)
    Bv = _flux(v, m, q=q, operator_measure="visible",
               use_coherence_velocity=True)
    assert Bc.policies.operator_measure == "carrier"
    # visible measure halves zeta (q=0.5); coherence velocity halves
    # the flux again
    s = torch.ones(64, dtype=DT)
    z = torch.zeros(64, dtype=DT)
    assert abs(float(Bv.total_flux(s, z))
               - 0.25 * float(Bc.total_flux(s, z))) < 1e-12


def test_inactive_rhs_guard_fail_closed():
    """Scatter from a boundary the operator's support does not cover:
    the solve must REJECT (never silently zero the source)."""
    v = generate_oriented_circle(128, 1.0, (0.0, 0.0), "cpu", DT)
    m = _masses(v)
    cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
    pg = CurrentToPhase(cfg).reconstruct(v, m, math.pi)
    op = WeightedPoissonOperator(pg.rho_metric, WeightedPoissonConfig(
        grid_shape=(256, 256)))
    # a second, DISTANT circle scatters entirely into vacuum
    far = generate_oriented_circle(64, 0.3, (1.7, 1.7), "cpu", DT)
    far_m = torch.full((64,), 2 * math.pi * 0.3 / 64, dtype=DT)
    Bfar = _flux(far, far_m, cfg=cfg)
    s = torch.randn(64, dtype=DT)
    s -= s.mean()
    drho = Bfar(s, torch.zeros(64, dtype=DT))
    with pytest.raises(IncompatibleGridVelocityError):
        op.solve(drho)
    assert op.last_inactive_rhs_fraction > 0.5
