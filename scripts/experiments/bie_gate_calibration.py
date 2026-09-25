"""Quotient-switch gates for the BIE backend, fixed on the pre-contact
part of earlier trajectories (the paper's Appendix rule), never on the
switch step itself.

On the BIE backend the intermediate state "metric merged, rows
separate" lasts one step by construction, so the clean State-II window
of the grid calibration (quotient_switch_calibration.py) does not
exist. The positive envelope of the keep/drop switch difference is
therefore taken from the natural step-to-step motion of the two
production trajectories over the 25 steps that precede the merge:

    eps_switch_x     = max_n |X^{n+1} - X^n|_inf / h_wall,
    eps_switch_theta = max_n |theta^{n+1} - theta^n|_inf,

recomputed by replaying those steps from the stored states with the
production configuration of each driver (the replay is bitwise the
production step). eps_split and eps_volume are inherited from the grid
calibration (they measure a residual and a conservation error, not a
difference between two discretizations).

Inputs: results/two_ellipses/two_ellipses_bie_states.pt (stopped run,
states 2000 and 2025; activation at 1431 so the CC visibility applies),
results/exact_merger/exact_merger_bie_states.pt (states 2000, 2025;
merge at 2047).
Writes src/torch/solver/quotient_gates_bie.py and
results/reports/bie_gate_calibration.json.

Usage: uv run python scripts/experiments/bie_gate_calibration.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import two_ellipses_benchmark as teb  # noqa: E402
from exact_merger_benchmark import build_config  # noqa: E402
from quotient_switch_stability import production_args  # noqa: E402
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.solver.mm_solver import MMSolver  # noqa: E402
from src.torch.solver.mm_step import MMStepper  # noqa: E402
from src.torch.solver.quotient_gates import QUOTIENT_GATES  # noqa: E402

DT = torch.float64


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def replay(cfg, pos, ang, n_steps, target0):
    """n_steps committed steps (MM + redistribution, the production
    post-step under the quotient guards); per-step natural motion."""
    sv = MMSolver(cfg)
    st = MMStepper(cfg)
    st._grid_target_volume_initial = target0
    v = OrientedPointCloudVarifold(positions=pos.clone(),
                                   angles=ang.clone())
    out = []
    for _ in range(n_steps):
        res, com = sv._advance_committed(st, v)
        nor = torch.stack([v.angles.cos(), v.angles.sin()], 1)
        h_w = float(st.fixed_masses.median())
        out.append(dict(
            dx_over_h=float((com.positions - v.positions).abs().max())
            / h_w,
            dtheta=float(wrap(com.angles - v.angles).abs().max()),
            n_iter=int(res.n_iter)))
        v = OrientedPointCloudVarifold(positions=com.positions.clone(),
                                       angles=com.angles.clone())
    return out


def ellipses_window(step0=2000, n_steps=25):
    ck = torch.load(ROOT / "results/two_ellipses/two_ellipses_bie_states.pt",
                    weights_only=True)
    s0 = ck[0]
    delta, tau = map(float, compute_recommended_params(s0["positions"]))
    nor0 = torch.stack([s0["angles"].cos(), s0["angles"].sin()], 1)
    m0 = teb.resolve_m(s0["positions"], nor0, delta, tau)
    target0 = float(0.5 * (m0 * (s0["positions"] * nor0).sum(-1)).sum())
    cfg = build_config(production_args(
        grid=768, redist_monotone=True, quotient_mode="off",
        q_mode="contact_complex_renormalized", metric="bie",
        bridge_gap=1.0), delta, tau)
    cfg.time_step = 1e-5
    cfg.grid_bulk_first_moment_rows = True
    cfg.grid_bulk_first_moment_rows_form = "current_centered"
    st = ck[step0]
    return replay(cfg, st["positions"], st["angles"], n_steps, target0)


def annulus_window(step0=2000, n_steps=25):
    import argparse
    from exact_merger_benchmark import DT_STEP  # noqa: F401
    ck = torch.load(ROOT / "results/exact_merger/exact_merger_bie_states.pt",
                    weights_only=True)
    s0 = ck[0]
    delta, tau = map(float, compute_recommended_params(s0["positions"]))
    args = argparse.Namespace(
        grid=1024, fill_epsilon=0.04, metric="bie", bridge_gap=1.0,
        no_redistribute=False, advect=False, freeze_dead=False,
        mass_estimator="loopwise_oriented_kde", bulk_rows=True,
        bulk_functional="current", volume_target_functional="current",
        contact_rows_mode="aligned_prequotient", gauge_hold=False,
        quotient_mode="off", angle_scope="loopwise",
        angle_measure="raw_loopwise", angle_consistency="none",
        redist_scope="loopwise", redist_tangent="angles",
        redist_curvature="loopwise", redist_q_policy="r_loop",
        redist_retraction=False, redist_monotone=True,
        redist_monotone_parts="abc", q_mode="self_renormalized",
        vertical=False, lam=1.0, optimizer_max_iter=None,
        optimizer_tol=None)
    cfg = build_config(args, delta, tau)
    st = ck[step0]
    nor0 = torch.stack([s0["angles"].cos(), s0["angles"].sin()], 1)
    from exact_merger_benchmark import masses_for
    m0 = masses_for("loopwise_oriented_kde", s0["positions"], nor0,
                    delta, tau)
    target0 = float(0.5 * (m0 * (s0["positions"] * nor0).sum(-1)).sum())
    return replay(cfg, st["positions"], st["angles"], n_steps, target0)


def main():
    rec = {}
    for name, fn in (("two_ellipses", ellipses_window),
                     ("annulus", annulus_window)):
        w = fn()
        rec[name] = dict(
            steps=len(w),
            dx_over_h_max=max(r["dx_over_h"] for r in w),
            dtheta_max=max(r["dtheta"] for r in w),
            per_step=w)
        print(name, {k: v for k, v in rec[name].items()
                     if k != "per_step"}, flush=True)
    eps_x = max(rec[n]["dx_over_h_max"] for n in rec)
    eps_th = max(rec[n]["dtheta_max"] for n in rec)
    gates = dict(eps_split=QUOTIENT_GATES["eps_split"],
                 eps_switch_x=eps_x, eps_switch_theta=eps_th,
                 eps_volume=QUOTIENT_GATES["eps_volume"])
    rec["gates"] = gates
    rec["rule"] = ("switch gates = max natural per-step motion over the "
                   "25 pre-merge steps of the two BIE trajectories "
                   "(two_ellipses 2000-2025, annulus 2000-2025); "
                   "eps_split / eps_volume inherited from the grid "
                   "calibration")
    (ROOT / "results/reports/bie_gate_calibration.json").write_text(
        json.dumps(rec, indent=1))
    src = f'''"""Quotient-switch gates of the BIE backend -- GENERATED by
scripts/experiments/bie_gate_calibration.py from the natural
step-to-step motion of the 25 pre-merge steps of the two BIE
production trajectories (two ellipses 2000-2025, concentric annulus
2000-2025), replayed from the stored states. Never re-fit on a switch
run.

    eps_split        (R_drop)+ floor      : {gates["eps_split"]:.6e}  (grid calibration)
    eps_switch_x     |dX|_inf / h_wall    : {gates["eps_switch_x"]:.6e}
    eps_switch_theta |dtheta|_inf         : {gates["eps_switch_theta"]:.6e}
    eps_volume       |dV_merged|/V        : {gates["eps_volume"]:.6e}  (grid calibration)
"""

QUOTIENT_GATES_BIE = {{
    "eps_split": {gates["eps_split"]!r},
    "eps_switch_x": {gates["eps_switch_x"]!r},
    "eps_switch_theta": {gates["eps_switch_theta"]!r},
    "eps_volume": {gates["eps_volume"]!r},
}}
'''
    (ROOT / "src/torch/solver/quotient_gates_bie.py").write_text(src)
    print("gates", gates)


if __name__ == "__main__":
    main()
