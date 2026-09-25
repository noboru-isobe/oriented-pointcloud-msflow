"""0A regressions: the scale API and the optimizer final-point diagnostics.

What is pinned:

  * eps_BEM modes -- the legacy "sigma" branch stays bit-identical, the
    carrier mode is invariant under q, the visible mode is not, and
    "explicit" is used verbatim;
  * angle_sigma decouples the weak-form angle matrices from perimeter_sigma
    (with None keeping the legacy coupling), and the radial mode of a
    concentric annulus is annihilated to machine precision
    (||dalpha||/||s|| ~ 1e-13 measured -- an exact structural zero, pinned
    at 1e-10);
  * MMStepResult diagnostics are evaluated at the RETURNED optimizer point:
    objective == perimeter + wasserstein there, W(0) == 0, the objective
    decreased, and the relative gradient norm is consistent.
"""

import math

import pytest
import torch

from src.torch.perimeter.angle_constraint import compute_angle_constraint_matrices
from src.torch.shapes.generator import (
    annulus_exact_masses,
    annulus_point_counts,
    generate_oriented_annulus,
    generate_oriented_circle,
    generate_oriented_flower,
)
from src.torch.solver.mm_step import MMConfig, MMStepper
from src.torch.transport.bem_wasserstein import BEMWasserstein

DTYPE = torch.float64


def _circle_setup(n=64):
    v = generate_oriented_circle(n, 1.0, (0.0, 0.0), "cpu", DTYPE)
    m = torch.full((n,), 2 * math.pi / n, dtype=DTYPE)
    q = torch.ones(n, dtype=DTYPE)
    # The dip must cover a MAJORITY of the points: the visible eps uses a
    # median, which is insensitive to any minority dip (a 25% dip left it
    # bit-identical -- itself worth knowing about the visible mode).
    q[: (2 * n) // 3] = 0.1
    return v, m, q


def _eps(mode, v, eff, carrier=None, sigma=None, explicit=None):
    bem = BEMWasserstein(method="point", epsilon_scale=0.1,
                         epsilon_mode=mode, epsilon_explicit=explicit)
    bem.setup_for_step(v.positions, v.normals, eff, sigma=sigma,
                       carrier_masses=carrier)
    return bem.endpoint_operator().epsilon


def test_legacy_sigma_mode_formula_and_fallback():
    v, m, q = _circle_setup()
    # with sigma: eps = scale * sigma / c_sigma
    assert _eps("sigma", v, q * m, sigma=0.3) == pytest.approx(0.1 * 0.3 / 3.0)
    # without sigma: legacy fallback to median(effective)
    assert _eps("sigma", v, q * m) == pytest.approx(
        0.1 * (q * m).median().item())


def test_carrier_epsilon_is_q_independent_visible_is_not():
    v, m, q = _circle_setup()
    ones = torch.ones_like(m)
    eps_c1 = _eps("carrier_segment_length", v, ones * m, carrier=m)
    eps_cq = _eps("carrier_segment_length", v, q * m, carrier=m)
    assert eps_c1 == pytest.approx(eps_cq, rel=1e-14)

    eps_v1 = _eps("visible_segment_length", v, ones * m)
    eps_vq = _eps("visible_segment_length", v, q * m)
    assert eps_v1 != pytest.approx(eps_vq, rel=1e-3)


def test_explicit_epsilon_and_validation():
    v, m, q = _circle_setup()
    assert _eps("explicit", v, m, explicit=0.0123) == pytest.approx(0.0123)
    with pytest.raises(ValueError, match="explicit"):
        _eps("explicit", v, m)
    with pytest.raises(ValueError, match="carrier_masses"):
        _eps("carrier_segment_length", v, m)
    with pytest.raises(ValueError, match="epsilon_mode"):
        _eps("bogus", v, m)


def test_mm_config_wires_epsilon_mode_and_carrier_masses():
    v = generate_oriented_flower(48, 5, 0.5, 1.0, (0, 0), "cpu", DTYPE)
    cfg = MMConfig(time_step=1e-5, bem_epsilon_mode="carrier_segment_length",
                   optimizer_method="bfgs", optimizer_max_iter=3)
    stepper = MMStepper(cfg)
    assert stepper.bem_wasserstein.epsilon_mode == "carrier_segment_length"
    stepper.step(v)                        # would raise without carrier_masses
    assert stepper.bem_wasserstein.endpoint_operator().epsilon is not None


def test_angle_sigma_decouples_angle_kernel_from_perimeter_sigma():
    """What 0A decouples is the KERNEL bandwidth of the weak-form angle
    constraint. The matrices also depend on perimeter_sigma through the
    coherence (m_eff = m q_sigma), which is genuine production behaviour and
    must NOT be frozen -- so the kernel decoupling is asserted with
    use_unit_coherence=True to shut that second path off, and the coherence
    path is checked separately below.
    """
    v = generate_oriented_flower(48, 5, 0.5, 1.0, (0, 0), "cpu", DTYPE)

    def ab_solve(perim_sigma, angle_sigma, unit_q=True):
        cfg = MMConfig(time_step=1e-5, perimeter_sigma=perim_sigma,
                       angle_sigma=angle_sigma, use_unit_coherence=unit_q)
        stepper = MMStepper(cfg)
        stepper._setup_step(v)             # the seam where A, B are built
        return stepper._sigma_angle, stepper.param.AB_solve.clone()

    sa1, m1 = ab_solve(0.2, 0.25)
    sa2, m2 = ab_solve(0.4, 0.25)
    assert sa1 == sa2 == 0.25
    assert torch.allclose(m1, m2, atol=0, rtol=0)      # kernel decoupled

    # legacy fallback: with angle_sigma=None the bandwidth follows sigma,
    # and the matrices then DO change with perimeter_sigma
    sa3, m3 = ab_solve(0.4, None)
    assert sa3 == 0.4
    assert not torch.allclose(m2, m3)

    # the coherence path stays live when q is not frozen: same angle_sigma,
    # different perimeter_sigma, real q -> different matrices (by design)
    _, m4 = ab_solve(0.2, 0.25, unit_q=False)
    _, m5 = ab_solve(0.4, 0.25, unit_q=False)
    assert not torch.allclose(m4, m5)


def test_radial_mode_of_concentric_annulus_is_annihilated():
    """Pure radial displacement of concentric rings changes no normal angle;
    the weak-form matrices annihilate it to machine precision (measured
    ||dalpha||_inf / ||s||_inf ~ 1.6e-13 .. 8.2e-13)."""
    no, ni = annulus_point_counts(128, 1.0, 0.5)
    v = generate_oriented_annulus(128, 1.0, 0.5, device="cpu", dtype=DTYPE)
    m = annulus_exact_masses(no, ni, 1.0, 0.5, device="cpu", dtype=DTYPE)
    nor = v.normals
    tang = torch.stack([-nor[:, 1], nor[:, 0]], 1)
    q = torch.ones(no + ni, dtype=DTYPE)

    a = (1.0 + 2.0) / math.log(2.0)
    s = 1e-5 * torch.cat([torch.full((no,), -a, dtype=DTYPE),
                          torch.full((ni,), +2 * a, dtype=DTYPE)])
    for sigma in (0.1, 0.2):
        A, B = compute_angle_constraint_matrices(v.positions, tang, m, q, sigma)
        dalpha = torch.linalg.lstsq(A, B).solution @ s
        assert dalpha.abs().max().item() < 1e-10 * s.abs().max().item()


def test_final_point_diagnostics_are_consistent():
    """The recorded perimeter / wasserstein / objective are re-evaluations at
    result.x, not the optimizer's last-touched point."""
    v = generate_oriented_flower(48, 5, 0.5, 1.0, (0, 0), "cpu", DTYPE)
    cfg = MMConfig(time_step=1e-5, optimizer_method="bfgs",
                   optimizer_max_iter=25)
    r = MMStepper(cfg).step(v)

    # consistency at the returned point and at zero
    assert r.objective == pytest.approx(r.perimeter + r.wasserstein, rel=1e-12)
    assert r.objective_initial == pytest.approx(
        r.frozen_perimeter_initial + r.wasserstein_initial, rel=1e-12)
    # zero displacement transports nothing
    assert abs(r.wasserstein_initial) < 1e-12

    # the step went downhill and the bookkeeping agrees
    assert r.objective_decreased
    assert r.objective <= r.objective_initial
    assert r.step_norm > 0
    assert r.gradient_norm_final >= 0
    assert r.relative_gradient_norm == pytest.approx(
        r.gradient_norm_final / max(1.0, r.gradient_norm_initial), rel=1e-12)
