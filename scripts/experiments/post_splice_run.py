"""S2: post-splice continuation -- the single-loop point cloud under
the unchanged production MM stack (CC+rows; with one loop the contact
branch is identically the self branch). Verifies the operational
point-cloud topology change end state: one loop, conserved merged
area / first moment, relaxation toward the circle R_eq = sqrt(2ab).

Telemetry per step: A, M, bar, P_frozen, W, isoperimetric ratio
4 pi A / L^2 -> 1, radius mean/std about the centroid vs R_eq,
radial Fourier modes 2..6. Stops: sanity |dA|/A > 1e-3 or
|dbar| > 1e-2 (coarse, as in L0), polygon self-intersection
(checked every 5 steps), --steps cap.

Usage:
    uv run python scripts/experiments/post_splice_run.py \
        --spliced two_ellipses_s1_768_spliced_state.pt \
        --steps 200 --tag _s2_768
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import two_ellipses_benchmark as teb  # noqa: E402
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.solver.arc_splice import (  # noqa: E402
    _has_self_intersection,
    _polygon_A_M,
)
from src.torch.solver.mm_solver import MMSolver  # noqa: E402

R = Path("results/two_ellipses")
R_EQ = math.sqrt(2.0 * teb.A_EL * teb.B_EL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spliced", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--grid", type=int, default=768)
    ap.add_argument("--dt", type=float, default=1e-5)
    ap.add_argument("--tag", default="_s2_768")
    ap.add_argument("--moment-rows-form", default="polygon",
                    choices=["polygon", "current", "current_centered"])
    ap.add_argument("--area-sanity", type=float, default=1e-3,
                    help="diagnostic probes may relax this to observe "
                         "the neck-opening transient (labeled)")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    blob = torch.load(R / args.spliced, weights_only=False)
    pos0, ang0 = blob["positions"], blob["angles"]
    act = blob["activation_state"]
    target0 = float(act["target_volume_initial"])
    sprec = blob.get("splice_record", {})
    A_ref = sprec.get("A_pre")
    M_ref = sprec.get("M_pre")
    # t=0 cloud only for delta/tau (production convention)
    v0, _, _, _ = teb.build_cloud(0.0)
    delta, tau = map(float, compute_recommended_params(v0.positions))
    cfg = teb.build_config(teb.production_args(
        grid=args.grid, redist_monotone=True, quotient_mode="off",
        q_mode="self_renormalized"), delta, tau)
    cfg.time_step = args.dt
    cfg.grid_bulk_first_moment_rows = True
    cfg.grid_bulk_first_moment_rows_form = args.moment_rows_form
    A0, Mx0, My0 = (float(x) for x in _polygon_A_M(pos0))
    print(f"S2: single-loop N={pos0.shape[0]}, A0 {A0:.8f} "
          f"(ref {A_ref}), R_eq {R_EQ:.6f}", flush=True)
    series = []
    states = {}
    state = dict(t0=time.time(), stop=None)
    out_json = R / f"two_ellipses{args.tag}.json"

    def dump(err=None):
        out_json.write_text(json.dumps(dict(
            meta=dict(source=args.spliced, grid=args.grid, dt=args.dt,
                      A_ref=A_ref, M_ref=M_ref, r_eq=R_EQ,
                      stop_reason=state["stop"], error=err,
                      splice_record={k: v for k, v in sprec.items()
                                     if k != "certificates"}),
            series=series), indent=1, default=str))
        torch.save(states, R / f"two_ellipses{args.tag}_states.pt")

    def cb(step, res):
        vv = (res.committed_varifold
              if res.committed_varifold is not None else res.varifold)
        p = vv.positions
        A, Mx, My = (float(x) for x in _polygon_A_M(p))
        c = torch.tensor([Mx / A, My / A])
        d = p - c
        r = d.norm(dim=1)
        th = torch.atan2(d[:, 1], d[:, 0])
        L = float((p.roll(-1, 0) - p).norm(dim=1).sum())
        modes = {int(k): float((r * torch.exp(-1j * k * th)).sum()
                               .abs() / (r.shape[0] * r.mean()))
                 for k in range(2, 7)}
        rec = dict(step=step, t=(step + 1) * args.dt, A=A,
                   dA_rel=(A - A0) / A0, M=(Mx, My),
                   bar=(float(c[0]), float(c[1])), L=L,
                   isoperimetric=4 * math.pi * A / L ** 2,
                   r_mean=float(r.mean()), r_std=float(r.std()),
                   r_mean_over_req=float(r.mean()) / R_EQ,
                   modes=modes,
                   P_frozen=float(res.perimeter),
                   W=float(res.wasserstein), n_iter=int(res.n_iter),
                   max_s=float(res.displacements.abs().max()))
        series.append(rec)
        if step % 25 == 0:
            states[step] = {"positions": p.clone(),
                            "angles": vv.angles.clone()}
            print(f"step {step:4d}: A drift {rec['dA_rel']:+.2e} "
                  f"bar ({rec['bar'][0]:+.1e},{rec['bar'][1]:+.1e}) "
                  f"isoper {rec['isoperimetric']:.5f} r/R_eq "
                  f"{rec['r_mean_over_req']:.5f} r_std "
                  f"{rec['r_std']:.4f} m2 {modes[2]:.4f} "
                  f"({(time.time()-state['t0'])/(step+1):.1f}s/step)",
                  flush=True)
            dump()
        stop = None
        prev = series[-2] if len(series) > 1 else None
        rec["dA_step"] = (rec["dA_rel"] - prev["dA_rel"]
                          if prev else rec["dA_rel"])
        rec["dth_step"] = float(teb.wrap(
            vv.angles - state["prev_ang"]).abs().max()) \
            if state.get("prev_ang") is not None else None
        state["prev_ang"] = vv.angles.clone()
        if abs(rec["dA_rel"]) > args.area_sanity:
            stop = (f"area sanity: {rec['dA_rel']:.2e} "
                    f"(bound {args.area_sanity:g})")
        elif math.hypot(*rec["bar"]) > 1e-2:
            stop = f"barycenter sanity: {rec['bar']}"
        elif step % 5 == 0 and _has_self_intersection(p):
            stop = "self-intersection"
        if stop:
            state["stop"] = stop
            states[step] = {"positions": p.clone(),
                            "angles": vv.angles.clone()}
            print(f"== stop at {step}: {stop} ==", flush=True)
            dump()
            return True
        return False

    sv = MMSolver(cfg)
    sv._pending_target_volume = target0
    err = None
    try:
        hist = sv.solve(OrientedPointCloudVarifold(
            positions=pos0.clone(), angles=ang0.clone()),
            args.steps, callback=cb)
        if getattr(hist, "stop_message", None):
            err = (f"{getattr(hist, 'stop_exception_type', '?')}: "
                   f"{hist.stop_message}")
    except BaseException as ex:                  # noqa: BLE001
        err = f"{type(ex).__name__}: {ex}"
        print(f"    [FAIL-CLOSED] {err}", flush=True)
    if series:
        states[series[-1]["step"]] = {
            "positions": None}      # placeholder removed below
        states.pop(series[-1]["step"], None)
    dump(err)
    last = series[-1] if series else {}
    print(f"done {len(series)} steps; stop {state['stop']}; err {err};"
          f" final isoper {last.get('isoperimetric')}, r/R_eq "
          f"{last.get('r_mean_over_req')}", flush=True)


if __name__ == "__main__":
    main()
