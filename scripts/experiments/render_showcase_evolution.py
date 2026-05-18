"""Paper-style static figure for a showcase trajectory — same 2×2
layout as the mp4 (boundary, perimeter, volume error, circularity), but
with the boundary panel showing **all five snapshots overlaid** instead
of just the final frame. Time is encoded by transparency (early = faint,
late = solid); a single solid colour is used so the legend opacities
match the markers without competing with any other colour scale.

Example
-------
    uv run python scripts/experiments/render_showcase_evolution.py \\
        --npz scripts/outputs/movies/flower_p5_history.npz \\
        --out results/flower_evolution \\
        --time-step 1e-5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ablation_summary import SNAPSHOT_STEPS, _save_png_pdf   # noqa: E402, F401
from src.torch.visualization.static import setup_matplotlib_style   # noqa: E402

# Match the ablation figures' typography (Computer Modern / LaTeX).
setup_matplotlib_style()


SNAPSHOT_ALPHAS = [0.20, 0.35, 0.55, 0.78, 1.00]
SCATTER_COLOR = "C0"


def _autoscale(positions_arr) -> tuple[tuple[float, float], tuple[float, float]]:
    xs, ys = [], []
    for frame in positions_arr:
        xs.extend(frame[:, 0]); ys.extend(frame[:, 1])
    if not xs:
        return (-1.2, 1.2), (-1.2, 1.2)
    pad = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys))
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Base path (no extension); writes .png and .pdf")
    p.add_argument("--time-step", type=float, required=True)
    p.add_argument("--target-perimeter", type=float, default=None,
                   help=r"If supplied, drawn as a dashed reference on the "
                        r"perimeter panel (e.g. $P_\star = 2\sqrt{\pi V_0}$).")
    args = p.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    positions_arr = npz["positions"]
    perimeters = np.asarray(npz["perimeters"])
    volumes = np.asarray(npz["volumes"])
    n_frames = len(positions_arr)
    h = args.time_step
    t = np.arange(n_frames) * h

    xlim, ylim = _autoscale(positions_arr)
    target_P = args.target_perimeter
    if target_P is None and len(volumes):
        # Single-disk asymptote, the relevant target for one-component runs.
        target_P = 2.0 * float(np.sqrt(np.pi * volumes[0]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # ---- (0,0) Boundary overlay ----
    ax = axes[0, 0]
    ax.set_aspect("equal")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Boundary evolution")
    from matplotlib.lines import Line2D
    legend_handles = []
    for step, alpha in zip(SNAPSHOT_STEPS, SNAPSHOT_ALPHAS):
        idx = min(step, n_frames - 1)
        pos = positions_arr[idx]
        ax.scatter(pos[:, 0], pos[:, 1], color=SCATTER_COLOR, alpha=alpha,
                   s=12, edgecolors="none")
        # Proxy artist (not added to any axes) — keeps the legend handles
        # local to this panel; bare `plt.scatter([], [], label=...)` would
        # silently attach to gca() and leak into other panels' legends.
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=SCATTER_COLOR, markeredgecolor="none",
                   alpha=alpha, markersize=7,
                   label=rf"$\text{{time}} = {step * h:.4f}$")
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10,
              framealpha=0.9, title="snapshot")

    # ---- (0,1) Perimeter ----
    ax = axes[0, 1]
    ax.plot(t, perimeters, color="C0", lw=1.7, label="$P(t)$")
    if target_P is not None:
        ax.axhline(target_P, color="k", ls="--", alpha=0.5,
                   label=rf"target $P_\star = 2\sqrt{{\pi V_0}} = {target_P:.4f}$")
    ax.set_xlabel(r"$\text{time} = n\tau$")
    ax.set_ylabel(r"$P$")
    ax.set_title("Perimeter evolution")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    # ---- (1,0) Volume relative error ----
    ax = axes[1, 0]
    V0 = volumes[0] if len(volumes) else 1.0
    err_pct = (volumes - V0) / V0 * 100 if V0 else np.zeros_like(volumes)
    ax.plot(t, err_pct, color="C3", lw=1.7, label=r"$(V(t) - V_0)/V_0$")
    ax.axhline(0.0, color="k", ls="--", alpha=0.5,
               label="exact conservation")
    ax.set_xlabel(r"$\text{time} = n\tau$")
    ax.set_ylabel(r"$(V - V_0)/V_0\ [\%]$")
    ax.set_title("Area conservation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    # ---- (1,1) Circularity 4π V / P² ----
    ax = axes[1, 1]
    circ = 4 * np.pi * volumes / np.maximum(perimeters ** 2, 1e-30)
    ax.plot(t, circ, color="C2", lw=1.7, label=r"$C(t) = 4\pi V/P^2$")
    ax.axhline(1.0, color="k", ls="--", alpha=0.5, label="perfect circle")
    ax.set_xlabel(r"$\text{time} = n\tau$")
    ax.set_ylabel(r"$4\pi V / P^2$")
    ax.set_title("Circularity")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _save_png_pdf(fig, args.out.with_suffix(".png"),
                  dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out.with_suffix('.png')} + .pdf")


if __name__ == "__main__":
    main()
