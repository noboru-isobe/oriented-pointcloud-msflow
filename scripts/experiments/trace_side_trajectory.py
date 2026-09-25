"""Evolve the same initial shape under both single-layer traces.

The static audits (scripts/experiments/bem_static_tests.py) show the solver
was assembling the exterior trace, and T6 shows a single MM step already moves
appreciably differently. This runs the two side by side for real so the
trajectories can be compared -- perimeter history, area drift, and the shapes
themselves.

Writes one .npz per trace in the layout VarifoldAnimator.save_history uses, so
the existing renderers apply directly:

    uv run python scripts/experiments/replay_history.py \
        --npz <out>/flower_interior.npz --out <out>/flower_interior.mp4 \
        --time-step 1e-5

Usage
-----
    uv run python scripts/experiments/trace_side_trajectory.py \
        --shape flower --n-steps 2000 --out results/trace_compare
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.torch.oriented_varifold import compute_masses, compute_recommended_params
from src.torch.perimeter.coherence_perimeter import compute_recommended_sigma
from src.torch.transport.bem_wasserstein import compute_coherence
from src.torch.shapes.generator import (
    generate_oriented_ellipse,
    generate_oriented_flower,
    generate_oriented_star,
    generate_oriented_two_ellipses,
)
from src.torch.solver.mm_solver import MMSolver, compute_volume_divergence
from src.torch.solver.mm_step import MMConfig

TRACES = ("legacy_exterior", "interior")


def build(shape: str, n: int, device: str, dtype):
    if shape == "flower":
        return generate_oriented_flower(n, 5, 0.5, 1.0, (0, 0), device, dtype)
    if shape == "star":
        return generate_oriented_star(n, 5, 0.4, 1.0, (0, 0), device, dtype)
    if shape == "ellipse":
        return generate_oriented_ellipse(n, 1.0, 0.5, (0, 0), device, dtype)
    if shape == "two-ellipses":
        return generate_oriented_two_ellipses(n // 2, device=device, dtype=dtype)
    raise ValueError(f"unknown shape {shape!r}")


def geometric_area(positions: torch.Tensor,
                   labels: torch.Tensor | None = None) -> float:
    """Shoelace in the sampled order, summed over components.

    Valid because these shapes are generated in parameter order and only move
    along their normals, so index order stays the boundary order. It is an
    area diagnostic independent of the KDE carrier weights, which matters
    because `compute_volume_divergence` shares those weights and cannot tell
    "the geometry lost area" from "the carrier weights drifted".

    `labels` is required for multi-component clouds: rolling across the whole
    concatenated array would close a spurious edge from the last point of one
    component to the first point of the next.
    """
    if labels is None:
        labels = torch.zeros(len(positions), dtype=torch.long)
    total = 0.0
    for a in range(int(labels.max().item()) + 1):
        p = positions[labels == a]
        x, y = p[:, 0], p[:, 1]
        total += 0.5 * torch.abs(
            (x * y.roll(-1) - x.roll(-1) * y).sum()
        ).item()
    return total


def component_labels(shape: str, n_points: int) -> torch.Tensor:
    """Which generated component each sample belongs to."""
    if shape == "two-ellipses":
        half = n_points // 2
        return torch.cat([torch.zeros(half, dtype=torch.long),
                          torch.ones(half, dtype=torch.long)])
    return torch.zeros(n_points, dtype=torch.long)


def run_one(shape, trace_side, args) -> dict:
    device = args.device
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    varifold = build(shape, args.n_points, device, dtype)

    config = MMConfig(
        time_step=args.time_step,
        bem_trace_side=trace_side,
        optimizer_method="trust-ncg",
        optimizer_tol=args.gtol,
        redistribute=args.redistribute,
        mass_knn_k=args.mass_knn_k,
    )

    labels = component_labels(shape, varifold.n_points)

    def coherent_perimeter(v) -> float:
        """P = sum_i m_i q_i recomputed from the current cloud."""
        delta, tau = compute_recommended_params(v.positions)
        m = compute_masses(v.positions, delta, tau)
        sigma = compute_recommended_sigma(v.positions)
        q = compute_coherence(v, m, sigma).clamp(0.0, 1.0)
        return (m * q).sum().item()

    perimeter_at_start = coherent_perimeter(varifold)

    t0 = time.time()
    history = MMSolver(config).solve(varifold, args.n_steps)
    elapsed = time.time() - t0

    n_frames = len(history)
    positions, angles, geom_area, div_area = [], [], [], []
    eff_masses, masses_all, coherences = [], [], []
    for k in range(n_frames):
        v = history.get_varifold(k)
        pos = v.positions.detach().cpu()
        positions.append(pos.numpy())
        angles.append(v.angles.detach().cpu().numpy())
        # labels only stay valid while N is constant (no dead-point removal)
        lab = labels if len(pos) == len(labels) else None
        geom_area.append(geometric_area(pos, lab))
        delta, tau = compute_recommended_params(v.positions)
        m = compute_masses(v.positions, delta, tau)
        sigma = compute_recommended_sigma(v.positions)
        q = compute_coherence(v, m, sigma).clamp(0.0, 1.0)
        div_area.append(compute_volume_divergence(v, m))
        masses_all.append(m.detach().cpu().numpy())
        coherences.append(q.detach().cpu().numpy())
        eff_masses.append((m * q).detach().cpu().numpy())

    stem = f"{shape}_{trace_side}"
    out_npz = args.out / f"{stem}.npz"
    same_n = len({p.shape[0] for p in positions}) == 1

    def pack(seq):
        return np.stack(seq) if same_n else np.array(seq, dtype=object)

    np.savez_compressed(
        out_npz,
        positions=pack(positions), angles=pack(angles),
        # per-point m, q and m*q, so questions like "did that detached point
        # still count toward the perimeter?" are answerable post hoc without
        # recomputing the estimator
        masses=pack(masses_all), coherence=pack(coherences),
        effective_masses=pack(eff_masses),
        labels=labels.numpy(),
        perimeters=np.asarray(history.perimeters[:history.n_completed]),
        volumes=np.asarray(history.volumes[:history.n_completed + 1]),
        geometric_areas=np.asarray(geom_area),
        divergence_areas=np.asarray(div_area),
    )

    summary = dict(
        shape=shape, trace_side=trace_side,
        n_points=args.n_points, n_steps=args.n_steps,
        n_completed=int(history.n_completed), time_step=args.time_step,
        seconds=elapsed,
        # recomputed on the initial cloud; history.perimeters[0] is the value
        # after the first MM step, not before it
        perimeter_at_start=perimeter_at_start,
        perimeter_after_first_step=float(history.perimeters[0]),
        perimeter_final=float(history.perimeters[history.n_completed - 1]),
        geometric_area_initial=geom_area[0],
        geometric_area_final=geom_area[-1],
        geometric_area_max_drift=max(abs(a - geom_area[0]) for a in geom_area)
        / geom_area[0],
        divergence_area_initial=div_area[0],
        divergence_area_final=div_area[-1],
        npz=str(out_npz),
    )
    print(f"  [{trace_side:>16}] {history.n_completed}/{args.n_steps} steps in "
          f"{elapsed:.0f}s   P {summary['perimeter_at_start']:.4f} -> "
          f"{summary['perimeter_final']:.4f}   "
          f"A_geom drift max {summary['geometric_area_max_drift']:.2%}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="flower",
                    choices=["flower", "star", "ellipse", "two-ellipses"])
    ap.add_argument("--n-points", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=2000)
    ap.add_argument("--time-step", type=float, default=1e-5)
    ap.add_argument("--gtol", type=float, default=1e-8)
    ap.add_argument("--mass-knn-k", type=int, default=9)
    ap.add_argument("--redistribute", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--trace", default=None, choices=list(TRACES),
                    help="run only one trace (for parallel invocation)")
    ap.add_argument("--out", type=Path, default=Path("results/trace_compare"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    traces = (args.trace,) if args.trace else TRACES

    print(f"{args.shape}  N={args.n_points}  {args.n_steps} steps  "
          f"dt={args.time_step}  device={args.device}")
    summaries = [run_one(args.shape, side, args) for side in traces]

    tag = f"_{args.trace}" if args.trace else ""
    path = args.out / f"{args.shape}{tag}_summary.json"
    path.write_text(json.dumps(summaries, indent=2))
    print(f"summary -> {path}")


if __name__ == "__main__":
    main()
