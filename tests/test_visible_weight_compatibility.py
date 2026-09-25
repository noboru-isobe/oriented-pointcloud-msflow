"""0C-2b regressions: what compatibility the visible metric actually enforces.

Fast pins on the findings of scripts/experiments/bem_visible_weight_audit.py
(full sweeps and raw tables in results/spectral_0c2b). Thresholds from the
measured tables with ~2x margins.

Pinned facts:

  * with q == 1 the carrier / visible / reweight operators coincide and the
    enforced constraint is the physical componentwise m-flux;
  * once q dips WITHIN a component, the production (visible) operator
    enforces the q^2 m - flux instead: the measured constraint angle matches
    the closed-form idealised prediction, and physical-kernel leakage is
    large;
  * the carrier operator is exactly invariant under q, preserving physical
    compatibility;
  * the visible operator also loses near-null rank structure at moderate
    degeneracy, while the reweighting-only diagnostic keeps the rank but
    rotates the subspace coordinates -- operator-rebuild damage and
    weight-coordinate distortion are distinct effects.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from bem_visible_weight_audit import (  # noqa: E402
    EPS_SCALE,
    build_geometry_full,
    q_synthetic_dip,
    run_case,
)

SPACING = 0.05


@pytest.fixture(scope="module")
def two_ellipses():
    varifold, m, phases, bcomp, M, C = build_geometry_full(
        "two_unequal_ellipses", SPACING)
    eps = EPS_SCALE * m.median().item()
    return varifold, m, phases, bcomp, C, eps


def _run(two_ellipses, q0, mode):
    varifold, m, phases, bcomp, C, eps = two_ellipses
    q = q_synthetic_dip(bcomp, 0, q0)
    return run_case("two_unequal_ellipses", SPACING, q, phases, C,
                    varifold, m, mode, eps)


def test_modes_coincide_at_q_equal_one(two_ellipses):
    rows = [_run(two_ellipses, 1.0, mode)
            for mode in ("carrier", "visible", "reweight")]
    for r in rows:
        assert r["detected_C"] == 2
        assert r["theta_compat_meas_deg"] < 1.5      # measured 0.47 deg
        assert r["theta_compat_pred_deg"] < 1e-4     # q == 1: prediction is 0
        # (arccos roundoff near 1 produces ~1e-6 deg, not exact zero)
        assert r["leak_phys_into_vis"] < 0.02
    # identical spectra across modes
    assert rows[0]["sigma_C"] == pytest.approx(rows[1]["sigma_C"], rel=1e-10)
    assert rows[0]["sigma_C"] == pytest.approx(rows[2]["sigma_C"], rel=1e-10)


def test_visible_metric_enforces_q2m_flux(two_ellipses):
    """Quarter-arc dip to q0=0.1 on one ellipse: the enforced constraint
    rotates ~27 deg away from the physical m-flux, matching the idealised
    q^2 m prediction (measured 27.4 vs predicted 26.2 at this resolution),
    with about half the physical kernel violating it.
    """
    r = _run(two_ellipses, 0.1, "visible")
    assert r["theta_compat_meas_deg"] > 20.0
    assert abs(r["theta_compat_meas_deg"] - r["theta_compat_pred_deg"]) < 3.0
    assert r["leak_phys_into_vis"] > 0.3


def test_carrier_metric_is_invariant_under_q(two_ellipses):
    for q0 in (1.0, 0.1, 0.003):
        r = _run(two_ellipses, q0, "carrier")
        assert r["detected_C"] == 2
        assert r["theta_compat_meas_deg"] < 1.5
        assert r["leak_phys_into_vis"] < 0.02


def test_rank_damage_is_an_operator_rebuild_effect(two_ellipses):
    """At q0=0.1 the production operator no longer resolves C=2 (the max-gap
    detector reports 1 and the gap collapses by an order of magnitude), while
    the reweighting-only diagnostic -- same spectral weights, carrier
    operator -- keeps both. So the rank loss comes from rebuilding the
    operator with q-shrunk panels, not from the weight dynamic range; the
    dynamic range instead shows up as subspace-coordinate rotation.
    """
    vis = _run(two_ellipses, 0.1, "visible")
    rew = _run(two_ellipses, 0.1, "reweight")

    assert vis["detected_C"] == 1                    # measured: collapses
    assert vis["gap"] > 0.15                         # measured 0.22

    assert rew["detected_C"] == 2                    # rank survives
    assert rew["gap"] < 0.05                         # measured 0.018
    assert rew["theta_indicator_deg"] > 20.0         # measured 45.9 deg


def test_real_coherence_annulus_pin():
    """The regime the annulus closure experiment will actually probe: a
    shrinking inner ring whose REAL coherence collapses by resolution
    (R_in = 0.05, sigma_coh = 0.15 -> q_inner = 0.597 at h = 0.02). The
    production visible constraint drifts several degrees from the carrier
    m-flux while the carrier mode stays put. Measured: carrier 0.265 deg,
    visible 6.69 deg (idealised q^2 m prediction 7.93 deg), leakage 0.117 vs
    0.005.
    """
    from bem_spectral_audit import GEOMETRIES
    from bem_visible_weight_audit import q_real

    name = "annulus_pin_test"
    GEOMETRIES[name] = dict(tier="primary", components=[
        ("circle", dict(R=1.0), (0.0, 0.0), False, 0),
        ("circle", dict(R=0.05), (0.0, 0.0), True, 0)])
    try:
        varifold, m, phases, bcomp, M, C = build_geometry_full(name, 0.02)
        eps = EPS_SCALE * m.median().item()
        q = q_real(varifold, m, sigma_coh=0.15)
        assert q[bcomp == 1].max() < 0.75            # inner ring genuinely low

        carrier = run_case(name, 0.02, q, phases, C, varifold, m,
                           "carrier", eps)
        visible = run_case(name, 0.02, q, phases, C, varifold, m,
                           "visible", eps)
    finally:
        del GEOMETRIES[name]

    assert carrier["theta_compat_meas_deg"] < 1.0
    assert visible["theta_compat_meas_deg"] > 4.0
    assert (visible["theta_compat_meas_deg"]
            > 5 * carrier["theta_compat_meas_deg"])
    # and the drift is the predicted q^2 m rotation, not something else
    assert visible["theta_compat_meas_deg"] == pytest.approx(
        visible["theta_compat_pred_deg"], abs=3.0)
