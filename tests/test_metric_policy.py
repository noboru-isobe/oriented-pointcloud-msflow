"""0C-2c regressions: topology sensor vs flux measure.

Pins for scripts/experiments/bem_metric_policy_audit.py (full tables in
results/spectral_0c2c). Thresholds from measured values with margins; the
structural assertions (which mode keeps rank, which ideal each constraint
tracks) are the point.

Headline corrections/pins:

  * rank damage under q-degeneracy is a QUADRATURE effect, not panel
    collapse: M1 (healthy panels, q m quadrature) collapses exactly like
    M2 (collapsed panels, q m quadrature), while M0 (m quadrature) survives.
    This corrects the 0C-2b commit's attribution to "q-shrunk panels".
  * the extra velocity q converts the enforced measure q m -> q^2 m
    (M1 vs M3 crossover at moderate dip depth);
  * the production angle pullback Delta alpha = q A^+ B s changes the
    constraint row space by < 0.1 deg -- the symmetric-offset argument holds
    numerically, so the Delta alpha = 0 shortcut used across 0C-2b/c is
    sound;
  * on a real near-sheet coherence field the carrier operator still resolves
    C = 2 with a clean gap while every q m-quadrature metric's gap degrades
    by an order of magnitude -- the regime 0C-4 needs a reliable sensor in.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from bem_metric_policy_audit import (  # noqa: E402
    EPS_SCALE,
    angle_between,
    full_pullback_constraint,
    ideal_constraint,
    measured_constraint,
    mode_decomposition,
)
from bem_spectral_audit import GEOMETRIES  # noqa: E402
from bem_visible_weight_audit import (  # noqa: E402
    build_geometry_full,
    q_real,
    q_synthetic_dip,
)


@pytest.fixture(scope="module")
def geo():
    varifold, m, phases, bcomp, M, C = build_geometry_full(
        "two_unequal_ellipses", 0.05)
    eps = EPS_SCALE * m.median().item()
    return varifold, m, phases, bcomp, C, eps


def _decomp(geo, q0, mode, spacing=0.05):
    varifold, m, phases, bcomp, C, eps = geo
    q = q_synthetic_dip(bcomp, 0, q0)
    d = mode_decomposition(varifold, m, q, phases, C, mode, eps)
    return d, q, m, phases, C


def test_all_modes_coincide_at_q_equal_one(geo):
    sigmas = []
    for mode in ("M0", "M1", "M2", "M3", "M4"):
        d, *_ = _decomp(geo, 1.0, mode)
        C = geo[4]
        assert d["detected_C"] == C
        sigmas.append(d["sigma"][C - 1].item())
    for s in sigmas[1:]:
        assert s == pytest.approx(sigmas[0], rel=1e-10)


def test_rank_damage_is_quadrature_not_panel(geo):
    """M1 keeps carrier panels (min panel / eps ~ 10) yet collapses exactly
    like M2 whose panels have shrunk to the regularisation scale; M0 with
    carrier quadrature survives. Measured at h=0.03, q0=0.1: M0 gap 1.1e-2
    detected 2; M1 and M2 both gap 2.2e-1 detected 1 with sigma_C equal to
    three digits.
    """
    C = geo[4]
    d0, *_ = _decomp(geo, 0.1, "M0")
    d1, *_ = _decomp(geo, 0.1, "M1")
    d2, *_ = _decomp(geo, 0.1, "M2")

    assert d0["detected_C"] == C == 2
    assert (d0["sigma"][C - 1] / d0["sigma"][C]).item() < 0.05

    for d in (d1, d2):
        assert d["detected_C"] == 1
        assert (d["sigma"][C - 1] / d["sigma"][C]).item() > 0.1

    # same spectral damage with and without panel collapse
    assert d1["sigma"][C - 1].item() == pytest.approx(
        d2["sigma"][C - 1].item(), rel=0.05)
    # ... while the panel healths differ by an order of magnitude
    eps = geo[5]
    assert (d1["panel"].min() / eps).item() > 5.0
    assert (d2["panel"].min() / eps).item() < 2.0


def test_velocity_q_selects_the_measure(geo):
    """At moderate dip depth (q0=0.3) the qm and q^2m ideals are still
    distinguishable (they nearly coincide for deep dips, where q^2 ~ q ~ 0 on
    the arc: measured pairwise leakage 0.07 at q0=0.1). Measured at h=0.03:
    M1 constraint 5.80 deg from qm vs 7.22 from q^2m; M3 5.67 from q^2m vs
    9.09 from qm; M0 0.26 deg from m.
    """
    varifold, m, phases, bcomp, M, C = build_geometry_full(
        "two_unequal_ellipses", 0.03)
    eps = EPS_SCALE * m.median().item()
    q = q_synthetic_dip(bcomp, 0, 0.3)
    N = len(m)
    ideals = {u: ideal_constraint(w, phases, C, N)
              for u, w in [("m", m), ("qm", q * m), ("q2m", q * q * m)]}

    d0 = mode_decomposition(varifold, m, q, phases, C, "M0", eps)
    c0 = measured_constraint(d0, phases, C, N)
    assert angle_between(c0, ideals["m"]) < 1.0

    d1 = mode_decomposition(varifold, m, q, phases, C, "M1", eps)
    c1 = measured_constraint(d1, phases, C, N)
    assert angle_between(c1, ideals["qm"]) < angle_between(c1, ideals["q2m"])
    assert angle_between(c1, ideals["qm"]) < angle_between(c1, ideals["m"])

    d3 = mode_decomposition(varifold, m, q, phases, C, "M3", eps)
    c3 = measured_constraint(d3, phases, C, N)
    assert angle_between(c3, ideals["q2m"]) < angle_between(c3, ideals["qm"])
    assert angle_between(c3, ideals["q2m"]) < angle_between(c3, ideals["m"])


def test_angle_pullback_is_negligible(geo):
    """Production map Delta alpha = q A^+ B s: the constraint row space moves
    by 0.039 deg (measured), so the Delta alpha = 0 shortcut is sound and
    'approximately q^2 m' survives the full pullback.
    """
    varifold, m, phases, bcomp, C, eps = geo
    q = q_synthetic_dip(bcomp, 0, 0.1)
    d = mode_decomposition(varifold, m, q, phases, C, "M4", eps)
    C_full, C_simple = full_pullback_constraint(
        d, varifold, m, q, phases, C, len(m))
    assert angle_between(C_full, C_simple) < 0.5


def test_carrier_sensor_survives_real_near_sheet_field():
    """Two facing ellipses, gap 0.15, real coherence at sigma_coh=0.3
    (q_min ~ 0.7): the carrier operator keeps C=2 with a clean gap while the
    production metric's gap degrades by an order of magnitude -- measured
    1.4e-2 vs 2.5e-1 at h=0.02. This is the fusion-precursor regime where
    0C-4's rank transition must not misfire.
    """
    name = "near_sheet_test"
    GEOMETRIES[name] = dict(tier="primary", components=[
        ("ellipse", dict(a=0.4, b=1.0), (-0.475, 0.0), False, 0),
        ("ellipse", dict(a=0.4, b=1.0), (0.475, 0.0), False, 1)])
    try:
        varifold, m, phases, bcomp, M, C = build_geometry_full(name, 0.03)
        eps = EPS_SCALE * m.median().item()
        q = q_real(varifold, m, 0.30)
        assert q.min() < 0.85                       # the field genuinely dips

        d0 = mode_decomposition(varifold, m, q, phases, C, "M0", eps)
        d4 = mode_decomposition(varifold, m, q, phases, C, "M4", eps)
    finally:
        del GEOMETRIES[name]

    assert d0["detected_C"] == 2
    gap0 = (d0["sigma"][C - 1] / d0["sigma"][C]).item()
    gap4 = (d4["sigma"][C - 1] / d4["sigma"][C]).item()
    assert gap0 < 0.06
    assert gap4 > 3 * gap0
