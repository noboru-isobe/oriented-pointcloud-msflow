"""Shared helpers for the paper figures (static, PDF)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import two_ellipses_benchmark as teb  # noqa: E402
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.transport.bem_wasserstein import compute_coherence  # noqa

SIGMA = teb.SIGMA
import os  # noqa: E402
os.environ["PATH"] = "/usr/local/texlive/2026/bin/x86_64-linux:" + os.environ.get("PATH", "")
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                     "axes.titlesize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "text.usetex": True, "font.family": "serif",
                     "text.latex.preamble": r"\usepackage{amsmath,amssymb}"})


def effective_mass(pos: torch.Tensor, ang: torch.Tensor):
    """m_i q_i on the committed cloud (WB q on the full cloud)."""
    v = OrientedPointCloudVarifold(positions=pos, angles=ang)
    nor = v.normals
    delta, tau = map(float, compute_recommended_params(pos))
    m = teb.resolve_m(pos, nor, delta, tau)
    q = compute_coherence(v, m, SIGMA, "wendland_c2")
    return m, q, nor


def draw_cloud(ax, pos, ang, lim, title, normals_every=6,
               vmax=None, s=4.0):
    m, q, nor = effective_mass(pos, ang)
    mq = (m * q).numpy()
    p = pos.numpy()
    sc = ax.scatter(p[:, 0], p[:, 1], c=mq, s=s, cmap="viridis",
                    vmin=0.0, vmax=vmax, linewidths=0, zorder=3)
    idx = np.arange(0, len(p), normals_every)
    n = nor.numpy()
    ax.quiver(p[idx, 0], p[idx, 1], n[idx, 0], n[idx, 1],
              angles="xy", scale_units="xy", scale=6.5, width=0.006,
              headwidth=3.5, headlength=4.5, headaxislength=4.0,
              color="0.35", zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, pad=3)
    return sc


def nearest_state(ck, k, prefer_ge=True):
    """Integer key of the stored state closest to step k (ties and
    prefer_ge: the first stored step >= k if one exists)."""
    keys = sorted(kk for kk in ck if isinstance(kk, int))
    if k in ck:
        return k
    ge = [kk for kk in keys if kk >= k]
    if prefer_ge and ge:
        return ge[0]
    return min(keys, key=lambda kk: abs(kk - k))


def run_args(default_run, default_out):
    """--run <basename> (JSON/states under the results dir),
    --out <figure basename>; defaults reproduce the paper files."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=default_run)
    ap.add_argument("--out", default=default_out)
    return ap.parse_args()
