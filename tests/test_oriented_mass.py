"""IIIc-0R permanent pins for the antipodal-sheet-aware mass estimator.

Reviewer-mandated pins (Phase IIIc-0 acceptance, 2026-08-13):
1. clean circle: co-oriented kNN calibration reproduces the position
   rule exactly (identical delta, tau);
2. unequal-count antiparallel sheets: per-sheet totals match the true
   arclength;
3. zero cutoff firings under the estimator's own (delta, tau);
4. imperfect antiparallel: the coherence residual equals the PHYSICAL
   two-sheet residual |n1 + n2| / 2 (no estimator artifact);
5. default estimator ("kde") is bitwise-unchanged in the solver
   dispatch.

Scope of the estimator (stated in mass.py): antipodal-sheet
separation only — NOT a general multiplicity estimator.
"""

import math

import numpy as np
import pytest
import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import (
    chi_tau,
    compute_kde_density_oriented,
    compute_masses,
    compute_masses_oriented,
    compute_recommended_params,
    compute_recommended_params_oriented,
)
from src.torch.shapes.generator import generate_oriented_circle
from src.torch.transport.bem_wasserstein import compute_coherence

DT = torch.float64
SIGMA = 0.1


def wall_fixture(n1, n2, rbar=0.394, normal_tilt=0.0):
    """Two coincident circular sheets, outward vs (tilted) inward."""
    th1 = 2 * math.pi * np.arange(n1) / n1
    th2 = 2 * math.pi * np.arange(n2) / n2
    pos = np.concatenate([
        rbar * np.stack([np.cos(th1), np.sin(th1)], 1),
        rbar * np.stack([np.cos(th2), np.sin(th2)], 1)])
    ang = np.concatenate([th1, th2 + math.pi - normal_tilt])
    return OrientedPointCloudVarifold(
        positions=torch.tensor(pos, dtype=DT),
        angles=torch.tensor(ang, dtype=DT))


def oriented_estimate(v):
    d, t = compute_recommended_params_oriented(v.positions, v.normals)
    m = compute_masses_oriented(v.positions, v.normals, d, t,
                                "wendland_c2")
    th = compute_kde_density_oriented(v.positions, v.normals, d,
                                      "wendland_c2")
    return m, d, t, th


class TestOrientedEstimatorPins:
    def test_clean_circle_params_match_position_rule(self):
        """Pin 1: on a clean boundary the co-oriented kNN rule is the
        position rule (same neighbor set -> identical delta, tau)."""
        v = generate_oriented_circle(128, device="cpu", dtype=DT)
        dp, tp = compute_recommended_params(v.positions)
        do, to = compute_recommended_params_oriented(v.positions,
                                                     v.normals)
        # same neighbor rule; only float summation order may differ
        assert float(do) == pytest.approx(float(dp), rel=1e-12)
        assert float(to) == pytest.approx(float(tp), rel=1e-12)

    def test_sheet_mass_balance_unequal_counts(self):
        """Pin 2: 90:141 antiparallel sheets each recover the true
        arclength (the 51/231 count-ratio floor must stay gone)."""
        n1, n2, rbar = 90, 141, 0.394
        L = 2 * math.pi * rbar
        v = wall_fixture(n1, n2, rbar=rbar)
        m, *_ = oriented_estimate(v)
        e1 = abs(float(m[:n1].sum()) - L) / L
        e2 = abs(float(m[n1:].sum()) - L) / L
        assert max(e1, e2) < 0.02  # measured 0.005-0.006

    def test_zero_cutoff_firings(self):
        """Pin 3: under the estimator's own (delta, tau) no point is
        killed by chi_tau on the wall fixture."""
        v = wall_fixture(90, 141)
        _, _, t, th = oriented_estimate(v)
        assert int((chi_tau(th, t) < 1).sum()) == 0
        assert float((th / t).min()) > 1.0

    @pytest.mark.parametrize("ndot", [-1.0, -0.95, -0.9, -0.8])
    def test_imperfect_antiparallel_physical_residual(self, ndot):
        """Pin 4: e_q equals the physical residual |n1+n2|/2 =
        sqrt((1+ndot)/2) — zero estimator artifact."""
        tilt = math.acos(max(-1.0, min(1.0, ndot)))
        # tilt from exact antiparallel is pi - angle(n1, n2)
        v = wall_fixture(90, 141, normal_tilt=math.pi - tilt)
        m, *_ = oriented_estimate(v)
        q = compute_coherence(v, m, SIGMA, "wendland_c2")
        e_q = float((m * q).sum() / m.sum())
        physical = math.sqrt(max(0.0, (1.0 + ndot) / 2.0))
        assert abs(e_q - physical) < 0.01

    def test_default_dispatch_bitwise(self):
        """Pin 5: mass_estimator defaults to "kde" and the solver
        dispatch is bitwise-identical to compute_masses."""
        from src.torch.solver.mm_step import MMConfig, MMStepper

        v = generate_oriented_circle(64, device="cpu", dtype=DT)
        d, t = compute_recommended_params(v.positions)
        cfg = MMConfig(mass_delta=float(d), mass_tau=float(t))
        assert cfg.mass_estimator == "kde"
        stepper = MMStepper(cfg)
        got = stepper._masses_for(v.positions, v.normals)
        want = compute_masses(v.positions, float(d), float(t),
                              cfg.mass_kernel)
        assert torch.equal(got, want)

    def test_oriented_dispatch_matches_direct_call(self):
        """Dispatch parity for the oriented path as well."""
        from src.torch.solver.mm_step import MMConfig, MMStepper

        v = wall_fixture(90, 141)
        d, t = compute_recommended_params_oriented(v.positions,
                                                   v.normals)
        cfg = MMConfig(mass_delta=float(d), mass_tau=float(t),
                       mass_estimator="oriented_kde")
        stepper = MMStepper(cfg)
        got = stepper._masses_for(v.positions, v.normals)
        want = compute_masses_oriented(v.positions, v.normals,
                                       float(d), float(t),
                                       cfg.mass_kernel)
        assert torch.equal(got, want)
