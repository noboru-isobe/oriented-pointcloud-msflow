"""Render the two-ellipses LOCAL merger campaign end-to-end in the
standard VarifoldAnimator 2x2 style (same layout as the Part-1 final
movie): boundary colored by effective mass with normals, perimeter,
merged-area conservation, circularity, with the topology events
annotated:

    t_act    1433  WB -> CC activation (certified, g/sigma = 0.5495)
    t_grid   1440  grid support merger (b_finer)
    t_quot   1825  QUOTIENT: bulk classes 2 -> 1
    t_splice 2014  ARC SPLICE: loop complex 2 -> 1 (N 376 -> 304)
    then the single-loop relaxation toward R_eq = sqrt(2ab).

Frames: checkpoints of _la3_768 (0..1826) + _l1q_768 (..2014) + the
spliced state + _s2_768 (200 steps, global 2015..2214). Diagnostic
curves use the FULL per-step series of the three runs stitched on the
global step axis; merged area = sum of the polygon areas (single-loop
area after the splice).

Canonical variant: reads the ONE-STROKE run _canonical_768 (a
single solver process; no stitching) with its automatic events.

Usage:
    uv run python scripts/experiments/render_canonical_movie.py
Output:
    results/reports/canonical_768_movie.mp4
"""

from __future__ import annotations

import json
import math
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

R = Path("results/two_ellipses")
DT_STEP = 1e-5
R_EQ = math.sqrt(2.0 * 0.4 * 1.0)
SPLICE_STEP = 2017
EVENTS = {
    1435: ("t_act (WB - CC activation)", "0.55"),
    1443: ("t_grid (support merger)", "0.7"),
    1829: ("t_quot (QUOTIENT: bulks 2 - 1)", "tab:red"),
    2017: ("t_splice (SPLICE: loops 2 - 1, N 376 - 304)", "tab:purple"),
}


class LocalMergerAnimator(VarifoldAnimator):
    def __init__(self, varifolds, steps, labels, perimeters, volumes,
                 config):
        super().__init__(varifolds[0], config=config, figsize=(12, 8))
        self.varifolds = varifolds
        self.steps = steps
        self.frame_labels = labels
        self.history = {"perimeters": list(perimeters),
                        "volumes": list(volumes),
                        "times": [s * config.time_step
                                  for s in range(len(perimeters))]}
        self.target_volume = float(volumes[0])
        self._compute_plot_limits()

    def setup_figure(self):
        super().setup_figure()
        self.axes[0].set_title(
            "MS flow: local merger, ONE automatic run (H = 768, t = 0)")
        h = self.config.time_step
        for es, (lab, col) in EVENTS.items():
            for ax in self.axes[1:]:
                ax.axvline(es * h, color=col, linestyle=":",
                           linewidth=1.1, alpha=0.85)
        th = np.linspace(0, 2 * np.pi, 256)
        self.axes[0].plot(R_EQ * np.cos(th), R_EQ * np.sin(th), "--",
                          color="0.6", lw=0.9, zorder=1)
        self.axes[0].set_xlim(-1.2, 1.2)
        self.axes[0].set_ylim(-1.2, 1.2)
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
        t_now = (k + 1) * self.config.time_step
        marker = ("   [SPLICE]" if self.frame_labels[frame]
                  == "spliced (zero-time)" else "")
        self.axes[0].set_title(
            f"Boundary Evolution ($t = {t_now:.5f}$)" + marker)
        self._info_txt.set_text(
            f"step {k}   $N = {v.n_points}$   "
            f"{self.frame_labels[frame]}")
        fired = [lab for es, (lab, col) in EVENTS.items() if es <= k]
        self._event_txt.set_text("\n".join(fired))
        self.axes[0].set_xlim(-1.2, 1.2)
        self.axes[0].set_ylim(-1.2, 1.2)
        return arts

    def _update_diagnostic_plot(self, frame):
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
                t, P, "b-", linewidth=2, label="P (frozen q)")
            self.axes[1].legend(fontsize=8)
        else:
            self.artists["perimeter_line"].set_data(t, P)
        err = (V - self.target_volume) / self.target_volume * 100
        if "volume_line" not in self.artists:
            self.artists["volume_line"], = self.axes[2].plot(
                t, err, "r-", linewidth=2, label="Merged area error")
            self.axes[2].axhline(0.0, color="r", linestyle="--",
                                 alpha=0.7)
            self.axes[2].legend(fontsize=8)
        else:
            self.artists["volume_line"].set_data(t, err)
        circ = 4 * np.pi * V / (P ** 2)
        if "circularity_line" not in self.artists:
            self.artists["circularity_line"], = self.axes[3].plot(
                t, circ, "g-", linewidth=2, label=r"$4\pi A/P^2$")
            self.axes[3].axhline(1.0, color="g", linestyle="--",
                                 alpha=0.7, label="Circle")
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
        pad = max((allerr.max() - allerr.min()) * 0.05, 0.005)
        self.axes[2].set_ylim(min(allerr.min(), 0) - pad,
                              max(allerr.max(), 0) + pad)
        allc = 4 * np.pi * allV / (allP ** 2)
        self.axes[3].set_ylim(allc.min() - 0.03,
                              max(allc.max(), 1.0) + 0.03)


def main():
    torch.set_default_dtype(torch.float64)
    setup_animation_backend()
    # ---- single-process series (no stitching) -------------------
    ser = json.load(open(R / "two_ellipses_canonical_768.json"))["series"]
    per, vol = {}, {}
    for r in ser:
        per[r["step"]] = r["P_frozen"]
        vol[r["step"]] = (r["A"] if r.get("phase") == "single_loop"
                          else r["A"][0] + r["A"][1])
    last = max(per)
    steps_all = sorted(per)
    # fill tiny gaps by carrying forward
    P_series, V_series = [], []
    pcur, vcur = per[steps_all[0]], vol[steps_all[0]]
    for s in range(last + 1):
        pcur, vcur = per.get(s, pcur), vol.get(s, vcur)
        P_series.append(pcur)
        V_series.append(vcur)
    # ---- frames (one states file) -------------------------------
    frames = []          # (global_step, varifold, label)
    ck = torch.load(R / "two_ellipses_canonical_768_states.pt",
                    weights_only=True)
    for k in sorted(x for x in ck if isinstance(x, int)):
        lab = "single loop" if k > SPLICE_STEP else ""
        frames.append((k, OrientedPointCloudVarifold(
            positions=ck[k]["positions"], angles=ck[k]["angles"]),
            lab))
    key = f"spliced_{SPLICE_STEP}"
    if key in ck:
        frames.append((SPLICE_STEP, OrientedPointCloudVarifold(
            positions=ck[key]["positions"],
            angles=ck[key]["angles"]), "spliced (zero-time)"))
    seen = {}
    for k, v, lab in frames:
        seen[(k, lab)] = (k, v, lab)      # dedupe (l1q 1825 = la3 1825)
    frames = sorted(seen.values(), key=lambda x: (x[0], x[2]))
    steps = [f[0] for f in frames]
    varifolds = [f[1] for f in frames]
    labels = [f[2] for f in frames]
    cfg = MMConfig(perimeter_sigma=0.1, time_step=DT_STEP)
    anim = LocalMergerAnimator(varifolds, steps, labels, P_series,
                               V_series, cfg)
    out = Path("results/reports/canonical_768_movie.mp4")
    anim.setup_figure()
    anim._set_initial_axis_limits()
    anim._save_video_pyav(out, list(range(len(varifolds))), fps=15)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(steps)} frames)")


if __name__ == "__main__":
    main()
