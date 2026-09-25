"""D2b-2: redistribution SCHEDULING as an accuracy-cost-relaxation
trade-off (deletion OFF, rule fixed = legacy_hybrid).

Not a safety experiment: D2b-1R established that legacy + every-step is
dt-refinement stable on the resolved track. The question here is
whether fixed-physical-interval or an adaptive trigger can cut
invocations without losing accuracy or slowing the warp relaxation.

Schedules
    every_step               legacy behavior (interval = 1)
    fixed_physical_interval  fires on accumulated physical time
                             (Delta t_post = 1e-4: every 5th step at
                             dt = 2e-5, every 10th at 1e-5 -- the
                             dt-INDEPENDENT invocation rate)
    adaptive_cv_trigger      fires when the PERMUTATION-INVARIANT
                             self-excluded KDE-density CV exceeds 0.05
                             (calibrated: uniform 0.0005, warp-0.1
                             0.16, warp-0.35 0.56; never ordered
                             edge-length CV)

Grids
    A  early resolved (R_-=0.5), N=128: dt {2e-5, 1e-5} x
       {uniform, warped} x {none + 3 schedules}; dt 5e-6 warped only.
    B  later resolved snapshot (R_-=0.35), warped, dt 2e-5.
    C  N=256 spot check (warped, early, dt 2e-5):
       every_step vs adaptive.

Per run: wall clock, invocations, best-fit radius/center/radial-CV,
spacing CV, stage-dH proxy + relaxation decomposition, q-bias rows,
and the NET final-support difference vs the matching no-redistribution
run (sampled per-loop Hausdorff on the final committed positions).

Usage
-----
    uv run python scripts/experiments/d2b2_schedule_tradeoff.py \
        --out results/d2b_redist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import d2b0_timelevel_shadow as d2b0  # noqa: E402
from d2a_redistribution_calibration import (  # noqa: E402
    sampled_polyline_hausdorff,
)
from d2b1_active_rules import run_one  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)

PHYS_INTERVAL = 1e-4
TRIGGER_CV = 0.05
T_END = 0.004

SCHEDULES = (("none", {}),
             ("every_step", dict(schedule="every_step")),
             ("fixed_physical_interval",
              dict(schedule="fixed_physical_interval",
                   phys_interval=PHYS_INTERVAL)),
             ("adaptive_cv_trigger",
              dict(schedule="adaptive_cv_trigger",
                   trigger_cv=TRIGGER_CV)))


def _net_vs_none(r, r_none, n_out):
    P, P0 = r.pop("final_positions"), r_none["final_positions"]
    if P.shape != P0.shape:
        return None
    return dict(
        outer=sampled_polyline_hausdorff(P[:n_out], P0[:n_out]),
        inner=sampled_polyline_hausdorff(P[n_out:], P0[n_out:]))


def _block(out, key_prefix, dt, warp, delta, tau, Rm0, n_out):
    n_steps = round(T_END / dt)
    r_none = None
    for name, kw in SCHEDULES:
        rule = "none" if name == "none" else "legacy_hybrid"
        r = run_one(rule, warp, dt, n_steps, delta, tau, Rm0=Rm0,
                    return_final_positions=True, **kw)
        if name == "none":
            r_none = dict(final_positions=r.pop("final_positions"))
            net = None
        else:
            net = _net_vs_none(r, r_none, n_out)
        r["net_final_dH_vs_none"] = net
        out[f"{key_prefix}|{name}"] = r
        print(f"  {key_prefix}|{name:24s} inv={r['invocations']:4d} "
              f"wall={r['wall_seconds']:6.1f}s "
              f"Rfit_err={r['R_fit_inner_err']:+.2e} "
              f"cv_in={r['cv_l_inner_final']:.4f} "
              f"net_dH_in={net['inner'] if net else None} "
              f"errs={len(r['timeline_errors'])}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/d2b_redist"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    v0, n_out, _ = d2b0.snapshot(0.5)
    delta, tau = compute_recommended_params(v0.positions)
    out = dict(meta=dict(t_end=T_END, phys_interval=PHYS_INTERVAL,
                         trigger_cv=TRIGGER_CV, delta_n128=delta))

    print("=== grid A: early resolved, N=128 ===")
    for dt in (2e-5, 1e-5):
        for warp in (0.0, 0.35):
            _block(out, f"A|dt{dt}|warp{warp}", dt, warp, delta, tau,
                   0.5, n_out)
    _block(out, "A|dt5e-06|warp0.35", 5e-6, 0.35, delta, tau, 0.5,
           n_out)

    print("=== grid B: later resolved (R_-=0.35), warped, dt=2e-5 ===")
    _block(out, "B|dt2e-05|warp0.35", 2e-5, 0.35, delta, tau, 0.35,
           n_out)

    print("=== grid C: N=256 spot (warped, early, dt=2e-5) ===")
    d2b0.N_OUTER = 256
    v0b, n_out256, _ = d2b0.snapshot(0.5)
    delta256, tau256 = compute_recommended_params(v0b.positions)
    out["meta"]["delta_n256"] = delta256
    n_steps = round(T_END / 2e-5)
    r_none = run_one("none", 0.35, 2e-5, n_steps, delta256, tau256,
                     return_final_positions=True)
    base_fp = r_none.pop("final_positions")
    out["C|none"] = r_none
    for name, kw in SCHEDULES[1:]:
        if name == "fixed_physical_interval":
            continue
        r = run_one("legacy_hybrid", 0.35, 2e-5, n_steps, delta256,
                    tau256, return_final_positions=True, **kw)
        P = r.pop("final_positions")
        r["net_final_dH_vs_none"] = (
            dict(outer=sampled_polyline_hausdorff(P[:n_out256],
                                                  base_fp[:n_out256]),
                 inner=sampled_polyline_hausdorff(P[n_out256:],
                                                  base_fp[n_out256:]))
            if P.shape == base_fp.shape else None)
        out[f"C|{name}"] = r
        print(f"  C|{name:24s} inv={r['invocations']:4d} "
              f"wall={r['wall_seconds']:6.1f}s "
              f"Rfit_err={r['R_fit_inner_err']:+.2e} "
              f"cv_in={r['cv_l_inner_final']:.4f}", flush=True)
    d2b0.N_OUTER = 128

    path = args.out / "d2b2_schedule_tradeoff.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
