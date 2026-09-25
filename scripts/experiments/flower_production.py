"""Single-component sanity run for the paper: five-petal flower under
the PRODUCTION stack (grid-Poisson metric H=768, box (-2,2)^2,
eps_fill 0.04, WB self-renormalized coherence, order-free
current_centered first-moment rows, monotone loopwise redistribution),
arclength-sampled at the benchmark spacing h_ref = 2 pi / 256.

Telemetry per step: perimeter P_frozen, order-free area 1/2 sum m x.n,
circularity 4 pi A / P^2, area drift, barycenter, max_s. States every
25 steps.

Usage:
  uv run python scripts/experiments/flower_production.py --steps 4000
Output: results/flower/flower_production.json / _states.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import two_ellipses_benchmark as teb  # noqa: E402
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.shapes.generator import generate_oriented_flower  # noqa
from src.torch.solver.mm_solver import MMSolver  # noqa: E402

H_REF = teb.H_REF
OUT = Path("results/flower")


def flower_perimeter(n_petals, r_in, r_out, n=200000):
    t = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)
    base = 0.5 * (r_in + r_out)
    amp = 0.5 * (r_out - r_in)
    r = base + amp * torch.cos(n_petals * t)
    x, y = r * torch.cos(t), r * torch.sin(t)
    return float(torch.hypot(x[1:] - x[:-1], y[1:] - y[:-1]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--dt", type=float, default=1e-5)
    ap.add_argument("--grid", type=int, default=768)
    ap.add_argument("--petals", type=int, default=5)
    ap.add_argument("--r-in", type=float, default=0.5)
    ap.add_argument("--r-out", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--metric", default="grid", choices=("grid", "bie"))
    ap.add_argument("--bridge-gap", type=float, default=1.0)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    OUT.mkdir(parents=True, exist_ok=True)
    L = flower_perimeter(args.petals, args.r_in, args.r_out)
    n = int(round(L / H_REF))
    v0 = generate_oriented_flower(n, n_petals=args.petals,
                                  inner_radius=args.r_in,
                                  outer_radius=args.r_out,
                                  device="cpu", dtype=torch.float64,
                                  initial_sampling="arc_length")
    pos, ang = v0.positions.clone(), v0.angles.clone()
    seg = (pos.roll(-1, 0) - pos).norm(dim=1)
    print(f"flower: L={L:.5f} N={n} spacing {seg.mean():.5f} "
          f"(CV {float(seg.std() / seg.mean()):.2e})", flush=True)
    delta, tau = map(float, compute_recommended_params(pos))
    cfg = teb.build_config(teb.production_args(
        grid=args.grid, redist_monotone=True, quotient_mode="off",
        metric=args.metric, bridge_gap=args.bridge_gap),
        delta, tau)
    cfg.time_step = args.dt
    cfg.grid_bulk_first_moment_rows = True
    cfg.grid_bulk_first_moment_rows_form = "current_centered"
    meta = dict(kind="flower", petals=args.petals, r_in=args.r_in,
                r_out=args.r_out, L=L, N=n, h_ref=H_REF, delta=delta,
                tau=tau, dt=args.dt, grid=args.grid,
                metric=args.metric,
                fill_epsilon=(cfg.grid_metric.phase.fill_epsilon
                              if cfg.grid_metric is not None else None),
                box=([list(cfg.grid_metric.phase.box_min),
                      list(cfg.grid_metric.phase.box_max)]
                     if cfg.grid_metric is not None else None),
                bie=(dict(vars(cfg.bie_metric))
                     if cfg.bie_metric is not None else None),
                sigma=teb.SIGMA, stop_reason=None, error=None)
    series, states = [], {0: dict(positions=pos.clone(), angles=ang.clone())}
    out_json = OUT / f"flower_production{args.tag}.json"
    t0 = time.time()
    A0 = None

    def dump(err=None):
        meta["error"] = err
        tmp = out_json.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(meta=meta, series=series),
                                  indent=1, default=str))
        tmp.replace(out_json)
        torch.save(states, OUT / f"flower_production{args.tag}_states.pt")

    def cb(step, res):
        nonlocal A0
        vv = (res.committed_varifold
              if res.committed_varifold is not None else res.varifold)
        p, nor = vv.positions, vv.normals
        m = teb.resolve_m(p, nor, delta, tau)
        A = float(0.5 * (m * (p * nor).sum(-1)).sum())
        if A0 is None:
            A0 = A
        bar = [float(0.5 * (m * (p * nor).sum(-1) * p[:, k]).sum() / A)
               for k in range(2)]
        P = float(res.perimeter)
        rec = dict(step=step, t=(step + 1) * args.dt, P=P, A=A,
                   dA_rel=(A - A0) / A0, circ=4 * math.pi * A / P ** 2,
                   bar=bar, max_s=float(res.displacements.abs().max()),
                   n_iter=int(res.n_iter), N=int(p.shape[0]))
        series.append(rec)
        if step % 25 == 0:
            states[step] = dict(positions=p.clone(),
                                angles=vv.angles.clone())
            print(f"step {step:4d}: P {P:.5f} circ {rec['circ']:.4f} "
                  f"dA {rec['dA_rel']:+.2e} max_s {rec['max_s']:.2e} "
                  f"({(time.time() - t0) / (step + 1):.1f}s/step)",
                  flush=True)
            dump()
        if abs(rec["dA_rel"]) > 5e-3:
            meta["stop_reason"] = f"area sanity {rec['dA_rel']:.2e}"
            dump()
            return True
        return False

    sv = MMSolver(cfg)
    err = None
    try:
        hist = sv.solve(v0, args.steps, callback=cb)
        if getattr(hist, "stop_message", None):
            err = (f"{getattr(hist, 'stop_exception_type', '?')}: "
                   f"{hist.stop_message}")
    except BaseException as ex:  # noqa: BLE001
        err = f"{type(ex).__name__}: {ex}"
        print(f"    [FAIL-CLOSED] {err}", flush=True)
        if isinstance(ex, (KeyboardInterrupt, SystemExit)):
            dump(err)
            raise
    dump(err)
    print(f"done {len(series)} steps; stop {meta['stop_reason']}; err {err}",
          flush=True)


if __name__ == "__main__":
    main()
