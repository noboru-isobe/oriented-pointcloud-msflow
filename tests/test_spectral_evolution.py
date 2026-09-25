"""0C-5a regressions: the spectral componentwise metric inside MMStepper.

Pins for scripts/experiments/spectral_evolution_0c5a.py (tables in
results/spectral_0c5a). Scope: q = 1, oracle C, u = m, separated unequal
disks (R1 = 0.5, R2 = 1.0), no redistribution or deletion. Thresholds carry
2-10x margins over the measured values quoted inline.

The one-table story pinned here (h = 0.05, gap = 2, fluxes as
sum_i m_i s_i / dt):

    spectral/interior   fluxes (-8.6e-5, -2.2e-5): no exchange, r_comp 1e-16
    legacy/exterior     fluxes (-2.04, +2.04): the production solver runs
                        Ostwald-ripening exchange at O(1) velocity,
                        N-independent -- Gate A measured dynamically
    legacy/interior     fluxes (-1.1e-2, +1.1e-2): suppressed but nonzero,
                        halving with N -- regularisation-dependent drift

plus: an ISOLATED disk under the spectral machinery is exactly stationary.
The two-disk residual motion is componentwise compatible and decreases
under refinement; its detailed source (spectral-subspace approximation,
BEM cross-block discretization, or perimeter-estimator error) is tracked
by the script's `attribution` block -- it is NOT cross-component KDE
coupling, whose compactly supported kernel contributes exactly zero at the
tested gaps.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from spectral_evolution_0c5a import (  # noqa: E402
    DT,
    attribute_residual,
    kstar_block_diag,
    one_step,
    oracle_single_disk,
)


@pytest.fixture(scope="module")
def rows():
    return {mode: one_step(mode, 0.05, 2.0)
            for mode in ("spectral/interior", "legacy/exterior",
                         "legacy/interior")}


def test_spectral_mode_has_no_componentwise_exchange(rows):
    r = rows["spectral/interior"]
    assert abs(r["flux1_over_dt"]) < 1e-3          # measured 8.6e-5
    assert abs(r["flux2_over_dt"]) < 1e-3
    assert r["r_comp"] < 1e-10                     # measured ~1e-16
    assert r["objective_decreased"]
    assert r["max_s_over_dt"] < 0.05               # measured 1.3e-2


def test_production_mode_runs_ostwald_exchange(rows):
    """The dynamic realisation of Gate A: under the shipped configuration
    the small disk shrinks and the large one grows at O(1) rate."""
    r = rows["legacy/exterior"]
    assert r["flux1_over_dt"] < -0.5               # measured -2.04
    assert r["flux2_over_dt"] > +0.5               # measured +2.04


def test_interior_legacy_is_suppressed_but_not_clean(rows):
    r = rows["legacy/interior"]
    assert 1e-3 < abs(r["flux1_over_dt"]) < 0.5    # measured 1.1e-2
    assert 1e-3 < abs(r["flux2_over_dt"]) < 0.5


def test_spectral_residuals_decrease_under_refinement():
    coarse = one_step("spectral/interior", 0.05, 2.0)
    fine = one_step("spectral/interior", 0.03, 2.0)
    assert fine["max_s_over_dt"] < coarse["max_s_over_dt"]
    assert abs(fine["flux1_over_dt"]) < abs(coarse["flux1_over_dt"])


def test_residual_attribution_scenario():
    """0C-5a.1 source separation (measured, gap = 2, h = 0.05): the frozen
    perimeter gradient at s = 0 is componentwise compatible to machine
    precision (oracle projection 1.8e-14), but has a small component
    (2.9e-3) in the spectral compatible space -- the residual motion comes
    from the finite-N tilt of the BEM left nullspace away from the component
    indicators, not from estimator cross-coupling (the mass kernel's compact
    support delta = 0.249 is far below the gap)."""
    a = attribute_residual(0.05, 2.0)
    assert a["mass_delta"] < a["gap"]
    assert a["proj_oracle"] < 1e-10            # measured 1.8e-14
    assert 1e-4 < a["proj_spectral"] < 1e-1    # measured 2.9e-3
    assert 1e-4 < a["subspace_dist"] < 1e-1    # measured 3.7e-3


def test_kstar_cross_block_causes_gap_dependence():
    """Causal split of the U0 tilt (measured): zeroing the oracle
    cross-component K* blocks collapses the U0-vs-indicator angle to a
    gap-INDEPENDENT 1.0688 deg baseline (per-component quadrature error),
    while the full operator adds a gap-dependent increment that grows as
    the disks approach (+0.021 deg at gap 2, +0.041 deg at gap 1)."""
    b1 = kstar_block_diag(0.05, 1.0)
    b4 = kstar_block_diag(0.05, 4.0)
    # baseline is gap-independent
    assert abs(b1["angle_blockdiag_deg"] - b4["angle_blockdiag_deg"]) < 1e-3
    # cross-block increment exists and grows as the gap shrinks
    inc1 = b1["angle_full_deg"] - b1["angle_blockdiag_deg"]
    inc4 = b4["angle_full_deg"] - b4["angle_blockdiag_deg"]
    assert inc1 > inc4 > 0


def test_isolated_disk_is_exactly_stationary():
    """C = 1 spectral machinery on one disk: the optimizer takes the zero
    step (measured max|s|/dt = 0.0 exactly). The two-disk residual motion is
    therefore an interaction effect; see the script's attribution block for
    the source separation."""
    assert oracle_single_disk(0.5, 0.05) == 0.0
    assert oracle_single_disk(1.0, 0.05) == 0.0
