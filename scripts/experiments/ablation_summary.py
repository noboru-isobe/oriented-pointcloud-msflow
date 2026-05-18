"""Aggregate the 4 cumulative-ablation cells (two-ellipses, 5000 steps) into:

  - ablation_summary.{csv,json} — per-cell metrics
  - ablation_comparison.png    — 2×2 quantitative comparison
                                 (Perimeter, Volume err %, Circularity, table)
  - ablation_snapshots.png     — 4 cells × 5 time snapshots, points coloured
                                 by the per-point perimeter contribution m_i·q_i
  - ablation_final_{cell}.png  — per-cell final frame (copy of the run output)
  - {cell}.mp4 + {cell}_data.json — copied per-cell

Reads from `scripts/outputs/experiments/ablation/` (produced by
`scripts/experiments/run_ablation.sh`), writes to `results/ablation/`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.torch.oriented_varifold.state import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import compute_masses, compute_recommended_params
from src.torch.perimeter.coherence_perimeter import compute_recommended_sigma
from src.torch.transport.bem_wasserstein import compute_coherence
from src.torch.visualization.static import setup_matplotlib_style

# Match the showcase mp4s' style: serif font + LaTeX rendering. All text
# strings below are written to be LaTeX-safe (no bare `_`, no raw `%`,
# no unicode `✗` / `≡` / `—`; specials go through math mode).
setup_matplotlib_style()


@dataclass
class Cell:
    name: str       # short tag for filenames
    label: str      # human-readable label for plots
    base: str       # run_two_ellipses_batch.py auto-tagged filename base
    color: str      # matplotlib color for trajectory lines
    use_uc: bool
    redistribute: bool
    remove_dead: bool


CELLS = [
    Cell("cell1_none", r"none ($q\equiv1$, no redist, no dead)",
         "two_ellipses_s5000_rdF_ucT_k9_compF_gap0.1_opt-trust-ncg_gtol1e-08",
         "C0", True,  False, False),
    Cell("cell2_coh", r"$+$coh",
         "two_ellipses_s5000_rdF_ucF_k9_compF_gap0.1_opt-trust-ncg_gtol1e-08",
         "C1", False, False, False),
    Cell("cell3_coh_redist", r"$+$coh $+$redist",
         "two_ellipses_s5000_rdT_ucF_k9_compF_gap0.1_opt-trust-ncg_gtol1e-08",
         "C2", False, True,  False),
    Cell("cell4_full", r"full ($+$ dead removal)",
         "two_ellipses_s5000_rdT_ucF_k9_dp0.15_compF_gap0.1_opt-trust-ncg_gtol1e-08",
         "C3", False, True,  True),
    Cell("cell5_full_no_coh", r"full $-$ coh ($q\equiv1$)",
         "two_ellipses_s5000_rdT_ucT_k9_dp0.15_compF_gap0.1_opt-trust-ncg_gtol1e-08",
         "C4", True,  True,  True),
]

# Step indices for the snapshot grid (clipped per cell to length of history).
SNAPSHOT_STEPS = [0, 1000, 2500, 3750, 5000]
KERNEL = "wendland_c2"   # MMConfig default; matches the showcase runs


def load_cell(cell: Cell, src_dir: Path):
    data = json.loads((src_dir / f"{cell.base}_data.json").read_text())
    npz_path = src_dir / f"{cell.base}.npz"
    npz = np.load(npz_path, allow_pickle=True) if npz_path.exists() else None
    return data, npz


def _save_png_pdf(fig, out_path: Path, dpi: int = 150,
                  bbox_inches=None) -> None:
    """Save a figure as both ``.png`` (raster, dpi) and ``.pdf`` (vector,
    paper-friendly). ``out_path`` is treated as the .png path; the .pdf
    sibling is derived from it."""
    fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches=bbox_inches)


def summary_for(cell: Cell, data: dict, npz=None) -> dict:
    """Per-cell summary. ``blow_up`` flags either kind of failure:
      - early-LU exception (n_completed < n_steps), or
      - silent position-divergence (final ‖x‖∞ ≥ 5 × initial ‖x‖∞)
    The latter is detectable only from the .npz (data.json's scalar
    history doesn't expose positions, and the BB-corrected perimeter
    stays bounded even when points explode because divergent points get
    near-zero mass)."""
    params = data["parameters"]
    theo = data.get("theoretical", {})
    hist = data["history"]
    perims = list(hist.get("perimeters", []))
    vols = list(hist.get("volumes", []))
    n_steps = params["n_steps"]
    n_completed = data.get("n_completed", max(len(perims) - 1, 0))
    early_break = (n_completed < n_steps)

    position_diverged = False
    final_pos_max = float("nan")
    if npz is not None:
        pos = npz["positions"]
        n_frames = len(pos)
        p0 = pos[0]
        pN = pos[n_frames - 1]
        init_scale = float(np.max(np.abs(p0))) if p0.size else 1.0
        final_pos_max = float(np.max(np.abs(pN))) if pN.size else float("nan")
        if init_scale > 0 and final_pos_max > 5.0 * init_scale:
            position_diverged = True

    P0 = perims[0] if perims else float("nan")
    Pf = perims[-1] if perims else float("nan")
    V0 = vols[0] if vols else float("nan")
    Vf = vols[-1] if vols else float("nan")
    # Single-disk asymptote (the actually-relevant target for the merging
    # scenario): perimeter of one disk that has the initial volume.
    # P★ = 2π√(V₀/π) = 2√(π V₀). data.json's `theoretical.target_perimeter`
    # in run_two_ellipses_batch.py instead reports the *two equal disks*
    # steady state (~7.95), which is misleading for cases that merge into
    # a single component, so we don't use it here.
    P_target_one_disk = (2.0 * math.sqrt(math.pi * V0)
                         if V0 and V0 > 0 else float("nan"))
    return {
        "cell": cell.name,
        "label": cell.label,
        "use_unit_coherence": cell.use_uc,
        "redistribute": cell.redistribute,
        "remove_dead": cell.remove_dead,
        "n_steps": n_steps,
        "n_completed": n_completed,
        "blow_up": bool(early_break or position_diverged),
        "blow_up_kind": (
            "early-LU" if early_break else
            "silent-divergence" if position_diverged else
            "none"
        ),
        "final_pos_max_abs": final_pos_max,
        "P_initial": P0,
        "P_final": Pf,
        "P_target": P_target_one_disk,        # single-disk asymptote
        "P_error": (abs(Pf - P_target_one_disk)
                    if not (math.isnan(P_target_one_disk) or math.isnan(Pf))
                    else float("nan")),
        "V_initial": V0,
        "V_final": Vf,
        "V_err_pct": (abs(Vf - V0) / abs(V0) * 100 if V0 else float("nan")),
        "time_step": params["time_step"],
    }


# ---------------------------------------------------------------- comparison

def plot_comparison(cells_loaded, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    target_P = None
    # Collect y-range only from cells that ran to completion; the blow-up
    # cells (rdF) reach 10^3-10^7 magnitudes and would swamp the scale.
    P_min_conv, P_max_conv = float("inf"), float("-inf")
    Verr_min_conv, Verr_max_conv = float("inf"), float("-inf")
    for cell, (data, npz) in cells_loaded:
        hist = data["history"]
        h = data["parameters"]["time_step"]
        perims = np.array(hist["perimeters"])
        vols = np.array(hist["volumes"])
        t = np.arange(len(perims)) * h
        s = summary_for(cell, data, npz)
        blow_up = s["blow_up"]
        ls = ":" if blow_up else "-"
        lw = 1.0 if blow_up else 1.7
        label = cell.label + (rf"  ($\times$ at step {len(perims)-1})" if blow_up else "")

        axes[0, 0].plot(t, perims, color=cell.color, lw=lw, ls=ls, label=label)
        V0 = vols[0] if len(vols) else 1.0
        if V0:
            err = (vols - V0) / V0 * 100
            axes[0, 1].plot(t, err, color=cell.color, lw=lw, ls=ls, label=label)
        circ = 4 * np.pi * vols / np.maximum(perims ** 2, 1e-30)
        axes[1, 0].plot(t, circ, color=cell.color, lw=lw, ls=ls, label=label)
        if target_P is None and not math.isnan(s["P_target"]):
            target_P = s["P_target"]   # one-disk asymptote 2√(π V₀)
        if not blow_up:
            P_min_conv = min(P_min_conv, float(perims.min()))
            P_max_conv = max(P_max_conv, float(perims.max()))
            Verr_min_conv = min(Verr_min_conv, float(err.min()))
            Verr_max_conv = max(Verr_max_conv, float(err.max()))
        # Mark the blow-up termination
        if blow_up and len(t):
            axes[0, 0].scatter([t[-1]], [perims[-1]], color=cell.color,
                               marker="x", s=60, zorder=10, clip_on=False)

    if target_P is not None:
        axes[0, 0].axhline(target_P, color="k", ls="--", alpha=0.4,
                           label=rf"target $P_\star = 2\sqrt{{\pi V_0}}={target_P:.4f}$  (single disk)")
    axes[0, 1].axhline(0.0, color="k", ls="--", alpha=0.4,
                       label="exact conservation")
    axes[1, 0].axhline(1.0, color="k", ls="--", alpha=0.4, label="perfect circle")

    # Clip y-axes to the converged-cell range so cells 3, 4 are visible;
    # the dotted trajectories of cells 1, 2 will exit the top of the frame
    # — that itself signals divergence.
    if P_max_conv > P_min_conv:
        pad = 0.1 * (P_max_conv - P_min_conv)
        target_lo = target_P if target_P is not None else P_min_conv
        axes[0, 0].set_ylim(min(target_lo, P_min_conv) - pad,
                            P_max_conv + pad)
    if Verr_max_conv > Verr_min_conv:
        pad = 0.2 * (Verr_max_conv - Verr_min_conv)
        axes[0, 1].set_ylim(min(0.0, Verr_min_conv) - pad,
                            max(0.0, Verr_max_conv) + pad)
    axes[1, 0].set_ylim(-0.05, 1.1)

    for ax, title, ylabel in [
        (axes[0, 0], "Perimeter", "P"),
        (axes[0, 1], "Volume relative error", r"$(V-V_0)/V_0$ [%]"),
        (axes[1, 0], "Circularity", r"$4\pi V/P^2$"),
    ]:
        ax.set_xlabel(r"$\text{time} = n\tau$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # (1,1) summary table — rendered as a matplotlib axes table so the
    # cells can carry math-mode labels without fighting LaTeX's text
    # parser over `_` and `%` chars.
    ax = axes[1, 1]
    ax.axis("off")
    col_labels = [r"cell", r"$n_{\rm compl}$",
                  r"$P_{\rm err}$", r"$V_{\rm err}\ [\%]$", r"failure"]
    row_labels = []
    cell_text = []
    for cell, (data, npz) in cells_loaded:
        s = summary_for(cell, data, npz)
        cell_text.append([
            cell.label,
            rf"${s['n_completed']}/{s['n_steps']}$",
            rf"${s['P_error']:.4f}$",
            rf"${s['V_err_pct']:.3f}$",
            s["blow_up_kind"].replace("-", "--"),
        ])
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    fig.suptitle("Cumulative ablation: two-ellipses, 5000 steps", fontsize=13)
    fig.tight_layout()
    _save_png_pdf(fig, out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- snapshot grid

def _eff_mass(positions: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Per-point perimeter contribution m_i·q_i for a single snapshot."""
    pos = torch.from_numpy(positions)
    ang = torch.from_numpy(angles)
    varifold = OrientedPointCloudVarifold(positions=pos, angles=ang)
    delta, tau = compute_recommended_params(pos)
    masses = compute_masses(pos, delta, tau)
    sigma = compute_recommended_sigma(pos)
    q = compute_coherence(varifold, masses, sigma, KERNEL).clamp(0.0, 1.0)
    return (masses * q).cpu().numpy()


def _common_limits(cells_loaded):
    """Plot range from converged cells only; blown-up cells (which fly off
    to ±10^3-10^6) would otherwise dominate the auto-range and shrink the
    interesting trajectories to invisible dots."""
    xs, ys = [], []
    for cell, (data, npz) in cells_loaded:
        if npz is None or summary_for(cell, data, npz)["blow_up"]:
            continue
        for frame in npz["positions"]:
            xs.extend(frame[:, 0]); ys.extend(frame[:, 1])
    if not xs:
        # Fallback for the two-ellipses initial geometry
        return (-1.2, 1.2), (-1.2, 1.2)
    pad = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys))
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


def _global_vmax(cells_loaded) -> float:
    """Per-cell initial-frame max ‖m·q‖∞, then take the max across cells.
    Used so every snapshot shares one colour scale."""
    vmax = 0.0
    for _cell, (_data, npz) in cells_loaded:
        if npz is None:
            continue
        vmax = max(vmax, float(_eff_mass(npz["positions"][0],
                                         npz["angles"][0]).max()))
    return vmax if vmax > 0 else 1.0


def plot_cell_evolution(cell: Cell, data: dict, npz, out_path: Path,
                        xlim, ylim, global_vmax: float):
    """Per-cell time-evolution strip: 1 row × 5 snapshots of the boundary,
    points coloured by m_i·q_i. Replaces the bare 'final frame' image so
    each cell's figure conveys *evolution* rather than the endpoint alone.
    """
    n_cols = len(SNAPSHOT_STEPS)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.8 * n_cols, 3.0),
                             gridspec_kw={"wspace": 0.05})
    h = data["parameters"]["time_step"]
    for j, step in enumerate(SNAPSHOT_STEPS):
        ax = axes[j]
        ax.set_aspect("equal")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(rf"$\text{{time}} = {step * h:.4f}$", fontsize=11)
        if npz is None:
            ax.text(0.5, 0.5, "(no .npz)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            continue
        n_frames = len(npz["positions"])
        idx = min(step, n_frames - 1)
        pos = npz["positions"][idx]
        ang = npz["angles"][idx]
        eff = _eff_mass(pos, ang)
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=eff, cmap="viridis",
                        s=14, alpha=0.85, vmin=0.0, vmax=global_vmax)
    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.85, pad=0.02,
                        label=r"$m_i\,q_i$")
    fig.suptitle(rf"{cell.label} -- boundary evolution (points coloured by $m_i\,q_i$)",
                 fontsize=12)
    _save_png_pdf(fig, out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_snapshot_grid(cells_loaded, out_path: Path,
                       xlim=None, ylim=None, vmax=None):
    n_rows = len(cells_loaded)
    n_cols = len(SNAPSHOT_STEPS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.6 * n_cols, 2.6 * n_rows),
                             squeeze=False,
                             gridspec_kw={"wspace": 0.05, "hspace": 0.12})
    if xlim is None or ylim is None:
        xlim, ylim = _common_limits(cells_loaded)
    if vmax is None:
        vmax = _global_vmax(cells_loaded)

    for i, (cell, (data, npz)) in enumerate(cells_loaded):
        h = data["parameters"]["time_step"]
        for j, step in enumerate(SNAPSHOT_STEPS):
            ax = axes[i, j]
            ax.set_aspect("equal")
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(rf"$\text{{time}} = {step * h:.4f}$", fontsize=15)
            if j == 0:
                ax.text(-0.05, 0.5, cell.label, transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=14)
            if npz is None:
                ax.text(0.5, 0.5, "(no .npz)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, alpha=0.6)
                continue
            n_frames = len(npz["positions"])
            idx = min(step, n_frames - 1)
            pos = npz["positions"][idx]
            ang = npz["angles"][idx]
            eff = _eff_mass(pos, ang)
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=eff, cmap="viridis",
                            s=12, alpha=0.85, vmin=0.0, vmax=vmax)
            # If the BEM blew up by this step the bulk of the points flew
            # off-screen — write "Overflowed" so readers don't puzzle over
            # the few stragglers still inside the frame. Trigger only when
            # >50% of points are outside (silent-divergence cases like
            # cell 3 have just a handful of escapees while the bulk of
            # the boundary remains visible — those should NOT be marked).
            xmin, xmax = xlim; ymin, ymax = ylim
            outside = ((pos[:, 0] < xmin) | (pos[:, 0] > xmax)
                       | (pos[:, 1] < ymin) | (pos[:, 1] > ymax))
            if outside.mean() > 0.5:
                ax.text(0.5, 0.5, "Overflowed",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=13, color="C3",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", alpha=0.85,
                                  edgecolor="C3"))

    # Shared colourbar — span the full grid height (no title above), with
    # larger label/tick fonts for paper readability.
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=1.0,
                        pad=0.02, label=r"$m_i\,q_i$")
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r"$m_i\,q_i$", fontsize=15)
    _save_png_pdf(fig, out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", type=Path,
                   default=Path("scripts/outputs/experiments/ablation"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/ablation"))
    p.add_argument("--copy-artifacts", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Also copy per-cell mp4/data.json/final-frame to out-dir.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells_loaded = []
    for c in CELLS:
        data, npz = load_cell(c, args.src_dir)
        cells_loaded.append((c, (data, npz)))

    # ---- summary tables
    summaries = [summary_for(c, d, npz) for c, (d, npz) in cells_loaded]
    with open(args.out_dir / "ablation_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    with open(args.out_dir / "ablation_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print("\n=== summary ===")
    for s in summaries:
        print(f"  {s['cell']:14s}  n_completed={s['n_completed']:5d}/{s['n_steps']}  "
              f"P_err={s['P_error']:.4f}  V_err%={s['V_err_pct']:.3f}  "
              f"blow_up={s['blow_up']}")

    # ---- shared plot range / colour scale (computed once)
    xlim, ylim = _common_limits(cells_loaded)
    vmax = _global_vmax(cells_loaded)

    # ---- plots
    plot_comparison(cells_loaded, args.out_dir / "ablation_comparison.png")
    print(f"saved {args.out_dir / 'ablation_comparison.png'}")
    # Snapshot grid: cells 1 and 2 both show "Overflowed" everywhere after
    # t=0 (both abort with LU singular within ~10 steps), so one is enough
    # to convey "redistribute missing → early break". Drop cell 2 from the
    # grid to keep the figure compact; it stays in the comparison plot and
    # the summary table where its slightly later breaking step still adds
    # information.
    snapshot_cells = [(c, d) for (c, d) in cells_loaded
                      if c.name != "cell2_coh"]
    plot_snapshot_grid(snapshot_cells, args.out_dir / "ablation_snapshots.png",
                       xlim=xlim, ylim=ylim, vmax=vmax)
    print(f"saved {args.out_dir / 'ablation_snapshots.png'}")

    # ---- per-cell time-evolution strips (replace bare final frames)
    for cell, (data, npz) in cells_loaded:
        out = args.out_dir / f"ablation_evolution_{cell.name}.png"
        plot_cell_evolution(cell, data, npz, out, xlim, ylim, vmax)
        print(f"saved {out}")

    # ---- per-cell mp4 + data.json copy (no longer copying _final_frame.png —
    # superseded by the time-evolution strips above)
    if args.copy_artifacts:
        for cell, _ in cells_loaded:
            for src_suffix, dst_name in [
                (".mp4", f"{cell.name}.mp4"),
                ("_data.json", f"{cell.name}_data.json"),
            ]:
                src = args.src_dir / f"{cell.base}{src_suffix}"
                if src.exists():
                    shutil.copy(src, args.out_dir / dst_name)
                    print(f"  {src.name}  ->  {dst_name}")
                else:
                    print(f"  [missing] {src.name}")
        # Remove stale single-frame stills from a previous run, if present.
        for cell, _ in cells_loaded:
            stale = args.out_dir / f"ablation_final_{cell.name}.png"
            if stale.exists():
                stale.unlink()


if __name__ == "__main__":
    main()
