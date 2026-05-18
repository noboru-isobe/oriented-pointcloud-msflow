"""Mullins–Sekerka flow via oriented point cloud varifolds — entry point.

Evolves a user-selected initial shape under the Wasserstein gradient flow of
perimeter (Mullins–Sekerka), using the BEM-linearised minimizing-movements
scheme, and optionally renders an mp4.

Examples
--------
    # Two ellipses merging, 5000 steps, save video
    python run.py --shape two-ellipses --n-steps 5000 --time-step 1e-5 \
        --output two_ellipses.mp4

    # Flower rounding, no video, just final perimeter/volume
    python run.py --shape flower --n-points 128 --n-steps 2000 --no-video

    # Quick circle sanity check
    python run.py --shape circle --n-points 64 --n-steps 10 --no-video

All `MMConfig` fields are exposed as `--<field>` flags (underscores →
hyphens). See `src/torch/solver/mm_step.py::MMConfig` for the full list.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from src.torch.shapes.generator import (
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_flower,
    generate_oriented_two_ellipses,
)
from src.torch.solver.mm_step import MMConfig
from src.torch.solver.mm_solver import compute_volume_divergence
from src.torch.oriented_varifold.mass import compute_masses, compute_recommended_params
from src.torch.visualization.animation import VarifoldAnimator


SHAPES = ("circle", "ellipse", "flower", "two-ellipses")


def build_varifold(args, device: str, dtype: torch.dtype):
    """Dispatch --shape to the matching factory in shapes.generator."""
    if args.shape == "circle":
        return generate_oriented_circle(
            args.n_points, radius=args.radius, device=device, dtype=dtype,
        )
    if args.shape == "ellipse":
        return generate_oriented_ellipse(
            args.n_points, a=args.a, b=args.b, device=device, dtype=dtype,
        )
    if args.shape == "flower":
        return generate_oriented_flower(
            args.n_points, n_petals=args.n_petals,
            inner_radius=args.inner_radius, outer_radius=args.outer_radius,
            device=device, dtype=dtype,
        )
    if args.shape == "two-ellipses":
        cx = (args.a + args.a + args.gap) / 2.0
        return generate_oriented_two_ellipses(
            args.n_points, a1=args.a, b1=args.b, center1=(-cx, 0.0),
            a2=args.a, b2=args.b, center2=(cx, 0.0),
            device=device, dtype=dtype,
        )
    raise ValueError(f"unknown shape {args.shape!r}")


def config_from_args(args) -> MMConfig:
    """Build MMConfig from any --<field> args that match its dataclass
    fields. argparse `dest`s are named to mirror MMConfig field names, so
    a plain dict-comprehension picks them up; ``None`` defaults are
    skipped so they fall back to the MMConfig defaults."""
    valid = {f.name for f in dataclasses.fields(MMConfig)}
    kw = {k: v for k, v in vars(args).items() if k in valid and v is not None}
    return MMConfig(**kw)


def main():
    p = argparse.ArgumentParser(
        description="Mullins–Sekerka flow on an oriented point cloud varifold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- shape ---
    p.add_argument("--shape", choices=SHAPES, default="two-ellipses")
    p.add_argument("--n-points", type=int, default=128,
                   help="Points on the boundary (per ellipse for two-ellipses)")
    p.add_argument("--radius", type=float, default=1.0, help="circle radius")
    p.add_argument("--a", type=float, default=0.4, help="ellipse semi-axis x")
    p.add_argument("--b", type=float, default=1.0, help="ellipse semi-axis y")
    p.add_argument("--gap", type=float, default=0.1,
                   help="two-ellipses: gap between the two ellipses")
    p.add_argument("--n-petals", type=int, default=5, help="flower petals")
    p.add_argument("--inner-radius", type=float, default=0.5, help="flower")
    p.add_argument("--outer-radius", type=float, default=1.0, help="flower")
    # --- evolution ---
    p.add_argument("--n-steps", type=int, default=5000)
    p.add_argument("--time-step", type=float, default=1e-5,
                   help="MM scheme time step h")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    # --- common MMConfig knobs (full list mirrors MMConfig fields) ---
    p.add_argument("--optimizer-method", type=str, default="trust-ncg")
    p.add_argument("--gtol", dest="optimizer_tol", type=float, default=1e-8)
    p.add_argument("--redistribute", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--mass-knn-k", type=int, default=9)
    p.add_argument("--backend", choices=["naive", "keops"], default="naive")
    # dead-point removal (used by the two-ellipses showcase video)
    p.add_argument("--remove-dead", dest="remove_dead_points",
                   action="store_true",
                   help="enable dead-point removal (m·q below threshold)")
    p.add_argument("--dead-threshold", dest="dead_point_threshold",
                   type=float, default=None,
                   help="dead-point threshold (MMConfig default 1e-4; "
                        "the two-ellipses showcase uses 0.15)")
    # BEM endpoint collocation (the flower showcase uses 3 = MMConfig default)
    p.add_argument("--bem-n-endpoints", dest="bem_n_endpoints",
                   type=int, default=None,
                   help="BEM collocation points per segment "
                        "(1=center only, 3=endpoints; default 3)")
    # --- output ---
    p.add_argument("--output", type=Path, default=None,
                   help="mp4 path; if omitted (or --no-video) just evolve")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--skip-frames", type=int, default=10,
                   help="render every N-th frame for the mp4")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print(f"shape={args.shape}  N={args.n_points}  steps={args.n_steps}  "
          f"device={device}  dtype={args.dtype}")

    varifold = build_varifold(args, device, dtype)
    config = config_from_args(args)
    animator = VarifoldAnimator(varifold, config=config)

    make_video = (args.output is not None) and (not args.no_video)
    if make_video:
        animator.create_animation(
            args.n_steps, save_path=args.output, skip_frames=args.skip_frames,
        )
        print(f"Saved video: {args.output.resolve()}")
    else:
        animator.evolve_varifold(args.n_steps)

    # --- summary (works for both paths; order-independent volume) ---
    cfg = config
    delta, tau = compute_recommended_params(
        varifold.positions, kernel=cfg.mass_kernel, k_min=cfg.mass_k_min,
    )
    md = cfg.mass_delta if cfg.mass_delta is not None else delta
    mt = cfg.mass_tau if cfg.mass_tau is not None else tau
    v0 = animator.varifolds[0]
    vf = animator.varifolds[-1]
    m0 = compute_masses(v0.positions, md, mt, cfg.mass_kernel)
    mf = compute_masses(vf.positions, md, mt, cfg.mass_kernel)
    P0, Pf = animator.history["perimeters"][0], animator.history["perimeters"][-1]
    V0 = compute_volume_divergence(v0, m0)
    Vf = compute_volume_divergence(vf, mf)
    print(f"perimeter: {P0:.4f} -> {Pf:.4f}  ({(P0-Pf)/P0*100:+.2f}%)")
    print(f"volume   : {V0:.4f} -> {Vf:.4f}  (err {abs(Vf-V0)/abs(V0)*100:.3f}%)")


if __name__ == "__main__":
    main()
