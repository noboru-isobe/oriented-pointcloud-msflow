"""Render the FINAL 1024 t=0 benchmark run in the standard
VarifoldAnimator style (same 2x2 layout as the two_ellipses ablation
movies: boundary colored by the effective mass m q with normals,
perimeter evolution, area conservation, circularity), with the four
topology events (t_grid / t_quot / t_vis / t_comp) annotated.

Frames = the 25-step checkpoints + the event states (1919, 2292) from
exact_merger_final_1024_states.pt; the diagnostic curves use the FULL
per-step series from the run JSON (not just the checkpoints).

Usage:
    uv run python scripts/experiments/render_final_movie.py
Output:
    results/reports/final_1024_movie.mp4
"""

from __future__ import annotations

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

from exact_merger_benchmark import OUT  # noqa: E402
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa: E402
from src.torch.solver import MMConfig  # noqa: E402
from src.torch.visualization.animation import (  # noqa: E402
    VarifoldAnimator,
    setup_animation_backend,
)

DT_STEP = 1e-5
T_C = 0.020347043818204673
R_EQ = 0.9055385138149781
EVENTS = {
    1521: ("t_grid (support merger)", "0.55"),
    1919: ("t_quot (QUOTIENT: topology 2→1)", "tab:red"),
    2167: ("t_vis (visible current 2→1)", "tab:orange"),
    2292: ("t_comp (COMPRESSION N 487→256)", "tab:purple"),
}


class FinalRunAnimator(VarifoldAnimator):
    """Checkpoint-frame animator over a completed run: the boundary
    panel renders the stored states; the diagnostics use the full
    per-step series sliced up to each frame's TRUE step index."""

    def __init__(self, varifolds, steps, perimeters, volumes, config):
        super().__init__(varifolds[0], config=config, figsize=(12, 8))
        self.varifolds = varifolds
        self.steps = steps
        self.history = {"perimeters": list(perimeters),
                        "volumes": list(volumes),
                        "times": [s * config.time_step
                                  for s in range(len(perimeters))]}
        self.target_volume = float(volumes[0])
        # normally filled by evolve_varifold(); frames are precomputed
        self._compute_plot_limits()

    def setup_figure(self):
        super().setup_figure()
        self.axes[0].set_title(
            "MS flow: exact concentric merger (H = 1024, t = 0 run)")
        h = self.config.time_step
        for es, (lab, col) in EVENTS.items():
            for ax in self.axes[1:]:
                ax.axvline(es * h, color=col, linestyle=":",
                           linewidth=1.1, alpha=0.85)
        th = np.linspace(0, 2 * np.pi, 256)
        self.axes[0].plot(R_EQ * np.cos(th), R_EQ * np.sin(th), "--",
                          color="0.6", lw=0.9, zorder=1)
        self.axes[0].set_xlim(-1.25, 1.25)
        self.axes[0].set_ylim(-1.25, 1.25)
        self._info_txt = self.axes[0].text(
            0.02, 0.975, "", transform=self.axes[0].transAxes,
            fontsize=9, va="top", family="monospace")
        self._event_txt = self.axes[0].text(
            0.02, 0.02, "", transform=self.axes[0].transAxes,
            fontsize=8, va="bottom", family="monospace", color="0.25")

    def animate_frame(self, frame):
        arts = super().animate_frame(frame)
        k = self.steps[frame]
        v = self.varifolds[frame]
        self.axes[0].set_title(
            f"Boundary Evolution (t = {(k + 1) * self.config.time_step:.5f})")
        self._info_txt.set_text(
            f"step {k}   t/T_c = {(k + 1) * self.config.time_step / T_C:.3f}"
            f"   N = {v.n_points}")
        fired = [lab for es, (lab, col) in EVENTS.items() if es <= k]
        self._event_txt.set_text("\n".join(fired))
        self.axes[0].set_xlim(-1.25, 1.25)
        self.axes[0].set_ylim(-1.25, 1.25)
        return arts

    def _update_diagnostic_plot(self, frame):
        # base-class panels/styling, but sliced by the TRUE step index
        # of this checkpoint frame (the base slices by frame index)
        k = self.steps[frame]
        n = min(k + 1, len(self.history["perimeters"]))
        if n < 2:
            return
        h = self.config.time_step
        t = np.arange(n) * h
        P = np.array(self.history["perimeters"][:n])
        V = np.array(self.history["volumes"][:n])
        if "perimeter_line" not in self.artists:
            self.artists["perimeter_line"], = self.axes[1].plot(
                t, P, "b-", linewidth=2, label="Perimeter")
            self.axes[1].legend(fontsize=8)
        else:
            self.artists["perimeter_line"].set_data(t, P)
        err = (V - self.target_volume) / self.target_volume * 100
        if "volume_line" not in self.artists:
            self.artists["volume_line"], = self.axes[2].plot(
                t, err, "r-", linewidth=2, label="Relative error")
            self.axes[2].axhline(0.0, color="r", linestyle="--",
                                 alpha=0.7, label="Exact conservation")
            self.axes[2].legend(fontsize=8)
        else:
            self.artists["volume_line"].set_data(t, err)
        circ = 4 * np.pi * V / (P ** 2)
        if "circularity_line" not in self.artists:
            self.artists["circularity_line"], = self.axes[3].plot(
                t, circ, "g-", linewidth=2, label="Circularity")
            self.axes[3].axhline(1.0, color="g", linestyle="--",
                                 alpha=0.7, label="Perfect Circle")
            self.axes[3].legend(fontsize=8)
        else:
            self.artists["circularity_line"].set_data(t, circ)
        allP = np.array(self.history["perimeters"])
        allV = np.array(self.history["volumes"])
        t_tot = len(allP) * h
        for ax in (self.axes[1], self.axes[2], self.axes[3]):
            ax.set_xlim(0, t_tot)
        self.axes[1].set_ylim(allP.min() * 0.98, allP.max() * 1.02)
        allerr = (allV - self.target_volume) / self.target_volume * 100
        pad = max((allerr.max() - allerr.min()) * 0.05, 0.02)
        self.axes[2].set_ylim(min(allerr.min(), 0) - pad,
                              max(allerr.max(), 0) + pad)
        allc = 4 * np.pi * allV / (allP ** 2)
        self.axes[3].set_ylim(min(allc.min(), 0.95) - 0.02,
                              max(allc.max(), 1.05) + 0.02)


def main():
    torch.set_default_dtype(torch.float64)
    setup_animation_backend()
    ck = torch.load(OUT / "exact_merger_final_1024_states.pt",
                    weights_only=True)
    steps = sorted(ck)
    varifolds = [OrientedPointCloudVarifold(
        positions=ck[k]["positions"], angles=ck[k]["angles"])
        for k in steps]
    d = json.load(open(OUT / "exact_merger_final_1024.json"))
    per = {r["step"]: r for r in d["series"]}
    last = max(per)
    perimeters = [per[s]["perimeter"] for s in range(last + 1)]
    volumes = [per[s]["vol_current"] for s in range(last + 1)]
    cfg = MMConfig(perimeter_sigma=0.1, time_step=DT_STEP)
    anim = FinalRunAnimator(varifolds, steps, perimeters, volumes, cfg)
    out = Path("results/reports/final_1024_movie.mp4")
    # create_animation() derives fps = 5 from skip_frames = 1, which is
    # too slow for ~100 frames; render + encode explicitly at 15 fps
    anim.setup_figure()
    anim._set_initial_axis_limits()
    anim._save_video_pyav(out, list(range(len(anim.varifolds))), fps=15)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(steps)} frames)")


if __name__ == "__main__":
    main()
