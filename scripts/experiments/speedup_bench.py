"""Definitive speedup benchmark (quiet machine required).

Verification contract (user): outputs unchanged AND wall-time reduced.
Correctness is pinned in tests/test_solver_speedups.py; this script
measures the TIME side and records the machine state alongside.

Ablation ladder (per config: min-of-REPS over N_STEPS-step trajectory):
    off            both flags off (baseline)
    quad           wlin_analytic_gradient only, kernel op disabled
    quad+kern      wlin_analytic_gradient (includes the kernel-grad op)
    quad+kern+lob  + bem_svd_method="lobpcg"

Shapes: paper pair at n_per = 128 (NK = 768) and n_per = 256
(NK = 1536) -- the second is the production-refinement regime where
the SVD share grows and the speedup is expected to be larger.

All progress is line-printed (flush=True); run under stdbuf -oL and
tee to a log for live monitoring.

Usage
-----
    uv run python scripts/experiments/speedup_bench.py \
        --out results/speedup_bench
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys

import torch

from src.torch.oriented_varifold.mass import compute_recommended_params
from src.torch.shapes.generator import generate_oriented_two_ellipses
from src.torch.solver.mm_step import MMStepper

sys.path.insert(0, str(Path(__file__).parent))
from p1_production_comparison import make_cfg  # noqa: E402

N_STEPS = 4
REPS = 3

CONFIGS = (
    ("off", dict()),
    ("quad", dict(wlin_analytic_gradient=True)),
    ("quad+lob", dict(wlin_analytic_gradient=True,
                      bem_svd_method="lobpcg")),
)
# (the hand-written kernel second-derivative op was benchmarked at
# 0.89x on BOTH sizes -- python-layer custom-Function overhead loses to
# the fused C++ double backward -- and was removed)


def pair(n_per):
    return generate_oriented_two_ellipses(
        n_per, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
        a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu",
        dtype=torch.float64)


def bench_one(v, delta, tau, overrides):
    cfg = make_cfg("C3", delta, tau, 2)
    for k, val in overrides.items():
        setattr(cfg, k, val)
    st = MMStepper(cfg)
    st.step(v)                      # warmup
    t0 = time.perf_counter()
    vv = v
    for _ in range(N_STEPS):
        r = st.step(vv)
        vv = r.varifold
    wall = (time.perf_counter() - t0) / N_STEPS
    return wall, vv, st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("results/speedup_bench"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    out = dict(meta=dict(
        n_steps=N_STEPS, reps=REPS,
        omp=os.environ.get("OMP_NUM_THREADS"),
        loadavg=os.getloadavg(), torch=torch.__version__))
    print(f"machine: load={os.getloadavg()} "
          f"omp={os.environ.get('OMP_NUM_THREADS')}", flush=True)

    for n_per in (128, 256):
        v = pair(n_per)
        delta, tau = compute_recommended_params(v.positions)
        key = f"pair_nper{n_per}"
        out[key] = {}
        base_final = None
        for name, ov in CONFIGS:
            best, final = None, None
            for rep in range(REPS):
                wall, vv, st = bench_one(v, delta, tau, dict(ov))
                if best is None or wall < best:
                    best, final = wall, vv
                print(f"  {key} {name:14s} rep{rep}: "
                      f"{wall:.3f}s/step", flush=True)
            row = dict(per_step=best)
            if name == "off":
                base_final = final
                base = best
            else:
                row["speedup_vs_off"] = base / best
                row["traj_max_dx_vs_off"] = float(
                    (final.positions - base_final.positions)
                    .abs().max())
            out[key][name] = row
            print(f"  {key} {name:14s} BEST {best:.3f}s/step"
                  + (f"  speedup={row.get('speedup_vs_off'):.2f}x "
                     f"dx={row.get('traj_max_dx_vs_off'):.1e}"
                     if name != "off" else ""), flush=True)

    path = args.out / "speedup_bench.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"raw results -> {path}", flush=True)


if __name__ == "__main__":
    main()
