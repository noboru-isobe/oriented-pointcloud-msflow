"""Annulus Run A pilot pin: one MM step reproduces the exact radial ODE.

The cheap gate in front of the trajectory benchmark
(scripts/experiments/annulus_run_a.py; trajectories and endpoint errors in
results/annulus_run_a). Configuration is the 0A-gated one: interior trace,
carrier-segment epsilon, fixed angle_sigma, unit coherence, no
redistribution, no deletion.

Measured one-step slope ratios against dR/dt = -a/R:
    n_outer =  64: 1.0141 (outer), 1.0208 (inner)
    n_outer = 128: 1.0019, 1.0035
    n_outer = 256: 0.9990, 0.9994
with machine-precision radial symmetry (noncircularity ~ 1e-16) and a
converged optimizer (n_iter = 1, relative gradient ~ 1e-9). Pinned at n = 64
with ~2x margins; the refinement trend lives in the experiment results.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from annulus_run_a import R_IN, R_OUT, ode_rhs, run  # noqa: E402


class _Args:
    n_outer = 64
    time_step = 1e-5
    gtol = 1e-8
    angle_sigma = 0.1
    max_iter = 300
    one_step = True
    n_steps = 1
    report_every = 1
    out = None


def test_one_step_matches_radial_ode():
    summary = run(_Args())
    f = summary["frames"][-1]

    ode_out, ode_in = ode_rhs(R_OUT, R_IN)
    assert f["slope_out"] / ode_out == pytest.approx(1.0, abs=0.04)
    assert f["slope_in"] / ode_in == pytest.approx(1.0, abs=0.05)

    # radial symmetry preserved to machine precision, no centroid drift
    assert f["noncirc_out"] < 1e-12
    assert f["noncirc_in"] < 1e-12
    assert f["centroid_out"] < 1e-12
    assert f["centroid_in"] < 1e-12

    # geometric area conserved and the step resolved against R-
    assert f["area_drift"] < 1e-5
    assert f["max_step_over_Rin"] < 1e-2

    # optimizer genuinely converged (the 0A diagnostics, not just a flag)
    assert f["objective_decreased"]
    assert f["relative_gradient_norm"] < 1e-6
