"""Render a Phase P dumbbell run in the standard VarifoldAnimator
2x2 style (boundary colored by effective mass with normals,
perimeter, area conservation) with the 4th panel showing the neck
width w_fit against the pre-registered admissibility floor w*.

Usage:
    uv run python scripts/experiments/render_dumbbell_movie.py \
        --tag _p12_base [--title "..."]
Output:
    results/reports/dumbbell<tag>_movie.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa
from src.torch.solver import MMConfig  # noqa: E402
from src.torch.visualization.animation import (  # noqa: E402
    VarifoldAnimator,
    setup_animation_backend,
)

R = Path("results/pinchoff")
W_STAR = 0.075


class DumbbellAnimator(VarifoldAnimator):
    def __init__(self, varifolds, steps, series, config, title):
        super().__init__(varifolds[0], config=config, figsize=(12, 8))
        self.varifolds = varifolds
        self.steps = steps
        self.series = series
        self.title = title
        self.history = {
            "perimeters": [r["P"] for r in series],
            "volumes": [r["A"] for r in series],
            "times": [r["t"] for r in series]}
        self.target_volume = float(series[0]["A"])
        self._compute_plot_limits()

    def setup_figure(self):
        super().setup_figure()
        self.axes[0].set_title(self.title)
        self.axes[3].set_title("Neck width (order-free)")
        self.axes[3].set_xlabel(r"$\text{time} = n\tau$")
        self.axes[3].set_ylabel(r"$w_{\mathrm{fit}}$")
        xr = max(float(v.positions[:, 0].abs().max())
                 for v in self.varifolds) * 1.1
        self.axes[0].set_xlim(-xr, xr)
        self.axes[0].set_ylim(-xr / 2.4, xr / 2.4)
        self._info_txt = self.axes[0].text(
            0.02, 0.975, "", transform=self.axes[0].transAxes,
            fontsize=9, va="top", family="monospace")

    def animate_frame(self, frame):
        arts = super().animate_frame(frame)
        k = self.steps[frame]
        t_now = (k + 1) * self.config.time_step
        self.axes[0].set_title(
            f"Boundary Evolution ($t = {t_now:.5f}$)")
        self._info_txt.set_text(
            f"step {k}   $N = "
            f"{self.varifolds[frame].n_points}$")
        xr = self.axes[0].get_xlim()[1]
        self.axes[0].set_xlim(-xr, xr)
        self.axes[0].set_ylim(-xr / 2.4, xr / 2.4)
        return arts

    def _update_diagnostic_plot(self, frame):
        k = self.steps[frame]
        t = np.array(self.history["times"])
        n = int(np.searchsorted(np.array([r["step"]
                                          for r in self.series]), k,
                                side="right"))
        if n < 2:
            return
        P = np.array(self.history["perimeters"])
        V = np.array(self.history["volumes"])
        wf = np.array([r["w_fit"] for r in self.series])
        err = (V - self.target_volume) / self.target_volume * 100
        for key, ax, y, style, lab in (
                ("perimeter_line", self.axes[1], P, "b-", "P"),
                ("volume_line", self.axes[2], err, "r-",
                 "area error"),
                ("neck_line", self.axes[3], wf, "g-",
                 r"$w_{\mathrm{fit}}$")):
            if key not in self.artists:
                self.artists[key], = ax.plot(
                    t[:n], y[:n], style, linewidth=2, label=lab)
                if key == "volume_line":
                    ax.axhline(0.0, color="r", linestyle="--",
                               alpha=0.7)
                if key == "neck_line":
                    ax.axhline(W_STAR, color="r", linestyle=":",
                               label=r"$w^* = 0.075$")
                ax.legend(fontsize=8)
            else:
                self.artists[key].set_data(t[:n], y[:n])
            ax.set_xlim(0, t[-1] * 1.02)
        self.axes[1].set_ylim(P.min() * 0.995, P.max() * 1.005)
        pad = max((err.max() - err.min()) * 0.05, 0.005)
        self.axes[2].set_ylim(min(err.min(), 0) - pad,
                              max(err.max(), 0) + pad)
        self.axes[3].set_ylim(min(wf.min(), W_STAR) - 0.02,
                              wf.max() + 0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    setup_animation_backend()
    d = json.load(open(R / f"dumbbell{args.tag}.json"))
    series = d["series"]
    ck = torch.load(R / f"dumbbell{args.tag}_states.pt",
                    weights_only=True)
    steps = sorted(x for x in ck if isinstance(x, int))
    varifolds = [OrientedPointCloudVarifold(
        positions=ck[k]["positions"], angles=ck[k]["angles"])
        for k in steps]
    cfg = MMConfig(perimeter_sigma=0.1, time_step=d["meta"]["dt"])
    title = args.title or (
        f"MS flow: capsule dumbbell {args.tag} "
        f"(P_eff = {d['meta']['fixture'].get('p_eff_right', 0):.1f})")
    anim = DumbbellAnimator(varifolds, steps, series, cfg, title)
    out = Path(f"results/reports/dumbbell{args.tag}_movie.mp4")
    anim.setup_figure()
    anim._set_initial_axis_limits()
    anim._save_video_pyav(out, list(range(len(varifolds))),
                          fps=args.fps)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(steps)} frames)")


if __name__ == "__main__":
    main()
