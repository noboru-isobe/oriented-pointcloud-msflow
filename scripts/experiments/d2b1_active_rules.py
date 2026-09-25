"""D2b-1R: the redistribution time-level rules ACTIVE in the solver
(deletion OFF -- the deletion interaction belongs to D3).

Corrected annulus MM evolution with redistribution enabled every step
(the current production schedule), comparing

    none / legacy_hybrid / post_mm_frozen / substep_refreshed

x  {uniform, warped (spacing ratio ~2)} initial sampling
x  dt in {2e-5, 1e-5} at FIXED physical time T (invocation count
   doubles when dt halves -- the F4 axis, now measured in-solver).

Two report tracks:
  resolved   the whole T window here stays far from the closure layer
             (R_- ~ 0.5 -> 0.46): ODE radius error, per-loop geometric
             drift of the redistribution stage (post_mm_candidate ->
             post_redistribution), spacing CVs, q quantiles,
             uniform-reference q bias.
  stress     separate short runs from the warped R_- = 0.06 snapshot
             (deep closure layer): algorithmic behavior only.

Usage
-----
    uv run python scripts/experiments/d2b1_active_rules.py \
        --out results/d2b_redist
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from src.torch.diagnostics import SupportTimeline
from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import (
    compute_masses,
    compute_recommended_params,
)
from src.torch.solver.mm_solver import MMSolver
from src.torch.transport import compute_coherence

sys.path.insert(0, str(Path(__file__).parent))
from d1a_mm_shadow import make_config  # noqa: E402
from d1a_ode_calibration import ode_radii  # noqa: E402
from d1b_active_rules import t_at_inner_radius  # noqa: E402


def ode_radii_from_state(t: float, Rm0: float):
    """Exact annulus radii a time t AFTER the state with R_- = Rm0 on
    the SAME conserved-C trajectory (D2b-2.1 critical fix: runs started
    from a later snapshot were being compared against the canonical
    R_-(0) = 0.5 trajectory at raw t -- a 1e-1-scale reference error)."""
    if abs(Rm0 - 0.5) < 1e-14:
        return ode_radii(t)
    t0 = t_at_inner_radius(Rm0)
    return ode_radii(t0 + t)
from d2a_redistribution_calibration import (  # noqa: E402
    curve_metrics,
    sampled_polyline_hausdorff,
)
from d2b0_timelevel_shadow import snapshot  # noqa: E402

SIGMA = 0.1
RULES = ("none", "legacy_hybrid", "post_mm_frozen", "substep_refreshed")
QUANTS = (0.0, 0.1, 0.5)


def _loop_metrics(P0, P1, n_out):
    out = {}
    for name, sl in (("outer", slice(0, n_out)),
                     ("inner", slice(n_out, None))):
        A0, A1 = P0[sl], P1[sl]
        b, a = curve_metrics(A0), curve_metrics(A1)
        segs = (A0.roll(-1, 0) - A0).norm(dim=1)
        dx = (A1 - A0).norm(dim=1).max().item()
        out[name] = dict(
            dP_rel=abs(a["perimeter"] - b["perimeter"]) / b["perimeter"],
            dA_rel=abs(a["area"] - b["area"]) / abs(b["area"]),
            dH=sampled_polyline_hausdorff(A0, A1),
            cv_l_pre=b["cv_l"], cv_l_post=a["cv_l"],
            max_dx_over_h_min=dx / segs.min().item())
    return out


def _q_quants(q):
    return [q.quantile(x).item() for x in QUANTS]


def fit_circle(P):
    """Kasa algebraic least-squares circle fit -> (center, R_fit,
    radial_cv). Review correction 2: origin-centered mean radius mixes
    center drift into the radius error."""
    X, Y = P[:, 0], P[:, 1]
    A = torch.stack([2 * X, 2 * Y, torch.ones_like(X)], dim=1)
    b = X * X + Y * Y
    sol = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.flatten()
    cx, cy = sol[0].item(), sol[1].item()
    R = math.sqrt(max(sol[2].item() + cx * cx + cy * cy, 0.0))
    r = (P - torch.tensor([cx, cy], dtype=P.dtype)).norm(dim=1)
    return [cx, cy], R, (r.std() / r.mean()).item()


def run_one(rule, warp, dt, n_steps, delta, tau, Rm0=0.5,
            check_every=25, schedule="every_step",
            phys_interval=0.0, trigger_cv=0.15,
            return_final_positions=False):
    import time as _time
    _t0 = _time.perf_counter()
    v, n_out, n_in = snapshot(Rm0, warp=warp)
    cfg = make_config(dt, delta, tau)
    if rule != "none":
        cfg.redistribute = True
        cfg.redistribution_rule = rule
        cfg.redistribution_schedule = schedule
        cfg.redistribute_physical_interval = phys_interval
        cfg.redistribute_trigger_cv = trigger_cv
    solver = MMSolver(cfg)
    solver.support_timeline = SupportTimeline(
        initial_loops=[torch.arange(n_out),
                       torch.arange(n_out, n_out + n_in)])
    hist = solver.solve(v, n_steps)
    tl = solver.support_timeline
    errs = [(r.step, r.level, r.error) for r in tl.records if r.error]

    rows = []
    cum_dH_in = 0.0
    for step in range(hist.n_completed):
        cand = next(r for r in tl.records
                    if r.level == "post_mm_candidate" and r.step == step)
        red = next(r for r in tl.records
                   if r.level == "post_redistribution" and r.step == step)
        Rm = cand.positions[n_out:].norm(dim=1).mean().item()
        row = dict(step=step, R_in=Rm)
        if red.stage_executed:
            m = _loop_metrics(cand.positions, red.positions, n_out)
            cum_dH_in += m["inner"]["dH"]
            row.update(
                redist_inner=m["inner"], redist_outer=m["outer"],
                delta_redist_over_R_in=(
                    delta * cfg.redistribute_delta_ratio) / Rm)
        if cand.q_current_diagnostic is not None:
            row["q_inner_quants"] = _q_quants(
                cand.q_current_diagnostic[n_out:])
        # uniform-reference q bias at checkpoints
        if step % check_every == 0 and cand.q_current_diagnostic \
                is not None:
            Rp = cand.positions[:n_out].norm(dim=1).mean().item()
            # uniform reference at the fitted radii
            from src.torch.shapes.generator import (
                generate_oriented_annulus)
            vref = generate_oriented_annulus(
                n_out, Rp, Rm, (0.0, 0.0), "cpu", torch.float64,
                n_inner=n_in)
            mref = compute_masses(vref.positions, delta, tau)
            qref = compute_coherence(vref, mref, SIGMA, "wendland_c2")
            qa = cand.q_current_diagnostic[n_out:]
            row["q_bias_vs_uniform_ref"] = dict(
                actual=_q_quants(qa), uniform=_q_quants(qref[n_out:]),
                dq_inf=(qa - qref[n_out:]).abs().max().item()
                if qa.shape == qref[n_out:].shape else None)
        rows.append(row)

    # final committed state vs ODE
    fin = [r for r in tl.records if r.level == "post_deletion_final"][-1]
    Rm_f = fin.positions[n_out:].norm(dim=1).mean().item()
    Rp_f = fin.positions[:n_out].norm(dim=1).mean().item()
    c_in, Rfit_in, rcv_in = fit_circle(fin.positions[n_out:])
    c_out, Rfit_out, rcv_out = fit_circle(fin.positions[:n_out])
    t_end = hist.n_completed * dt
    Rp_ode, Rm_ode = ode_radii_from_state(t_end, Rm0)
    seg_in = fin.positions[n_out:]
    seg_in = (seg_in.roll(-1, 0) - seg_in).norm(dim=1)

    # early/tail decomposition of the per-step stage-dH sum (review
    # correction 1: this is a cumulative STAGE-DISPLACEMENT PROXY, not
    # net geometric drift -- it includes the WANTED tangential
    # resampling; the decomposition shows the one-time warp-relaxation
    # structure)
    relax_step = next(
        (r["step"] for r in rows
         if r.get("redist_inner")
         and r["redist_inner"]["cv_l_post"] < 0.02), None)
    S_before = sum(r["redist_inner"]["dH"] for r in rows
                   if r.get("redist_inner")
                   and (relax_step is None or r["step"] <= relax_step))
    S_after = cum_dH_in - S_before

    return dict(
        rule=rule, warp=warp, dt=dt, schedule=schedule,
        n_outer=n_out, n_inner=n_in, N_total=n_out + n_in,
        Rm0=Rm0, ode_reference=("canonical" if abs(Rm0 - 0.5) < 1e-14
                                else f"shifted to R_-(t0)={Rm0}"),
        wall_seconds=_time.perf_counter() - _t0,
        **(dict(final_positions=fin.positions)
           if return_final_positions else {}),
        n_steps_requested=n_steps, n_steps_completed=hist.n_completed,
        stop_reason=(dict(step=hist.stop_step, stage=hist.stop_stage,
                          exception=hist.stop_exception_type,
                          message=hist.stop_message)
                     if hist.stop_step is not None else None),
        timeline_errors=errs,
        invocations=sum(1 for r in tl.records
                        if r.level == "post_redistribution"
                        and r.stage_executed),
        R_in_final=Rm_f, R_in_ode=Rm_ode,
        R_in_err=Rm_f - Rm_ode,
        R_out_err=Rp_f - Rp_ode,
        R_fit_inner=Rfit_in, R_fit_outer=Rfit_out,
        R_fit_inner_err=Rfit_in - Rm_ode,
        center_inner=c_in, center_outer=c_out,
        center_drift_inner=math.hypot(*c_in),
        radial_cv_inner=rcv_in, radial_cv_outer=rcv_out,
        cv_l_inner_final=(seg_in.std() / seg_in.mean()).item(),
        sum_step_stage_dH_inner=cum_dH_in,
        stage_dH_relax_step=relax_step,
        stage_dH_before_relax=S_before,
        stage_dH_tail_after_relax=S_after,
        rows=rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/d2b_redist"))
    ap.add_argument("--t-end", type=float, default=0.004)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    v0, _, _ = snapshot(0.5)
    delta, tau = compute_recommended_params(v0.positions)
    out = dict(meta=dict(sigma=SIGMA, delta=delta, tau=tau,
                         t_end=args.t_end))

    print("=== resolved track (T=%.4f) ===" % args.t_end)
    for dt in (2e-5, 1e-5):
        n_steps = round(args.t_end / dt)
        for warp in (0.0, 0.35):
            for rule in RULES:
                key = f"{rule}|warp{warp}|dt{dt}"
                r = run_one(rule, warp, dt, n_steps, delta, tau)
                out[key] = r
                last_q = (r["rows"][-1].get("q_inner_quants")
                          if r["rows"] else None)
                print(f"  {key:45s} R_err={r['R_in_err']:+.2e} "
                      f"cv_in={r['cv_l_inner_final']:.4f} "
                      f"stage_dH_in={r['sum_step_stage_dH_inner']:.2e} "
                      f"inv={r['invocations']} "
                      f"errs={len(r['timeline_errors'])}", flush=True)

    print("=== closure-layer stress track (R_-=0.06, warped, 50 steps,"
          " ALGORITHMIC data) ===")
    for rule in RULES:
        key = f"stress|{rule}"
        r = run_one(rule, 0.35, 2e-5, 50, delta, tau, Rm0=0.06)
        out[key] = r
        print(f"  {key:30s} R_in_final={r['R_in_final']:.4f} "
              f"cv_in={r['cv_l_inner_final']:.4f} "
              f"stage_dH_in={r['sum_step_stage_dH_inner']:.2e} "
              f"errs={len(r['timeline_errors'])}", flush=True)

    path = args.out / "d2b1_active_rules.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
