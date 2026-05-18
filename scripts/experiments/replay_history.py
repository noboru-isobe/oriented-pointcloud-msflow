"""Re-render an MS-flow mp4 from a saved trajectory (.npz from
VarifoldAnimator.save_history). Skips the (expensive) solver step and only
re-runs the matplotlib + PyAV animation pipeline — useful when the trajectory
is unchanged but the visualisation code has been updated (axis labels,
colour maps, etc.).

Handles both constant-N (3D positions array) and variable-N (object array,
e.g. from dead-point-removal runs) histories.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.torch.solver.mm_step import MMConfig
from src.torch.oriented_varifold.state import OrientedPointCloudVarifold
from src.torch.visualization.animation import VarifoldAnimator


def replay(npz_path: Path, mp4_out: Path, *, time_step: float,
           skip_frames: int = 10) -> None:
    npz = np.load(npz_path, allow_pickle=True)
    positions_arr = npz["positions"]
    angles_arr = npz["angles"]
    perimeters = list(npz["perimeters"])
    volumes = list(npz["volumes"])
    n_frames = len(positions_arr)

    # Rebuild the per-frame varifold list. Object-array (variable N) and
    # 3D ndarray (constant N) both iterate frame-by-frame the same way.
    varifolds = []
    for k in range(n_frames):
        pos = torch.from_numpy(np.asarray(positions_arr[k]))
        ang = torch.from_numpy(np.asarray(angles_arr[k]))
        varifolds.append(OrientedPointCloudVarifold(positions=pos, angles=ang))

    cfg = MMConfig(time_step=time_step)
    animator = VarifoldAnimator(varifolds[0], config=cfg)
    animator.varifolds = varifolds
    animator.history["perimeters"] = perimeters
    animator.history["volumes"] = volumes
    animator.history["times"] = [k * time_step for k in range(n_frames)]
    if volumes:
        animator.target_volume = float(volumes[0])

    # `_compute_plot_limits` normally runs at the end of `evolve_varifold`
    # and seeds `self.plot_limits` for the boundary/mass/perimeter/volume
    # axes; since we're bypassing the solver, call it explicitly here.
    animator._compute_plot_limits()

    mp4_out.parent.mkdir(parents=True, exist_ok=True)
    animator.create_animation(n_steps=n_frames - 1, save_path=mp4_out,
                              skip_frames=skip_frames)
    print(f"replayed: {npz_path}  ->  {mp4_out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--time-step", type=float, required=True)
    p.add_argument("--skip-frames", type=int, default=10)
    args = p.parse_args()
    replay(args.npz, args.out, time_step=args.time_step,
           skip_frames=args.skip_frames)


if __name__ == "__main__":
    main()
