"""Render a boundary-only animated GIF from a saved trajectory (.npz).

Reuses the rendering helpers that produce the static evolution figures
(scatter coloured by m_i·q_i + subsampled outward-normal arrows) so the
animated showcase frames look like the static counterparts.

Output is GIF only (no mp4 intermediate, no ffmpeg) via matplotlib's
PillowWriter. Handles both constant-N and variable-N (object array) npz.

Example
-------
    uv run python scripts/experiments/render_boundary_gif.py \\
        --npz scripts/outputs/movies/flower_p5_history.npz \\
        --out results/flower_boundary.gif \\
        --time-step 1e-5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ablation_summary import _eff_mass, _draw_normal_arrows   # noqa: E402
from render_showcase_evolution import _autoscale                # noqa: E402
from src.torch.visualization.static import setup_matplotlib_style  # noqa: E402

setup_matplotlib_style()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Output .gif path")
    p.add_argument("--time-step", type=float, required=True)
    p.add_argument("--skip-frames", type=int, default=10)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--figsize", type=float, default=4.5,
                   help="Square figure side in inches")
    args = p.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    positions_arr = npz["positions"]
    angles_arr = npz["angles"]
    n_frames = len(positions_arr)
    h = args.time_step

    xlim, ylim = _autoscale(positions_arr)

    # Frame indices we'll actually render. Clip skip_frames so we end on
    # the final frame instead of stopping just short.
    indices = list(range(0, n_frames, args.skip_frames))
    if indices[-1] != n_frames - 1:
        indices.append(n_frames - 1)

    # Global vmax for the m·q colourmap, taken over all the frames we'll
    # actually draw — keeps the colour scale fixed for the whole gif.
    vmax = 0.0
    for k in indices:
        eff_k = _eff_mass(positions_arr[k], angles_arr[k])
        if eff_k.size:
            vmax = max(vmax, float(eff_k.max()))
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(args.figsize, args.figsize))

    def draw(k: int):
        pos = positions_arr[k]
        ang = angles_arr[k]
        eff = _eff_mass(pos, ang)
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.scatter(pos[:, 0], pos[:, 1], c=eff, cmap="viridis",
                   s=14, alpha=0.9, vmin=0.0, vmax=vmax,
                   edgecolors="none")
        _draw_normal_arrows(ax, pos, ang, xlim, ylim, num_arrows=28)
        ax.set_title(rf"$\text{{time}} = {k * h:.4f}$", fontsize=14)

    anim = FuncAnimation(fig, draw, frames=indices,
                         interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    size_mb = args.out.stat().st_size / 1e6
    print(f"saved {args.out}  ({len(indices)} frames, {size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
