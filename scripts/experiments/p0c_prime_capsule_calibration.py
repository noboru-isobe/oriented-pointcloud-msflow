"""P0-c': capsule-based admissibility re-calibration (reviewer
2026-08-26 section 7). The stadium fixture confounded the wall
separation with the cap curvature (r_cap = w/2). Here the C^2 capsule
dumbbell keeps the LOBE radius and junction curvature FIXED and
varies only the neck width w, attributing the admissibility boundary
to the wall separation alone. w/sigma in {1.0, 1.25, 1.5, 2.0};
w/sigma in {1.0, 1.25} additionally at H=1024.

Same acceptance as P0-c (amended): |1-qbar_neck| <= 0.05,
|P_sigma - P_geom|/P_geom <= 0.05, |phase_grid_volume - A|/A <= 0.05,
w >= 3h.

Usage:
    uv run python scripts/experiments/p0c_prime_capsule_calibration.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import two_ellipses_benchmark as teb  # noqa: E402
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.shapes.dumbbell import build_capsule_dumbbell  # noqa
from src.torch.transport.bem_wasserstein import compute_coherence  # noqa

H_REF = teb.H_REF
SIGMA = teb.SIGMA
EPS_FILL = 0.04
OUT = Path("results/pinchoff")


def measure(w_over_sigma: float, grid: int):
    w = w_over_sigma * SIGMA
    # lobes and blend FIXED across the sweep (only w varies); no area
    # normalization here -- the calibration is about scales, not
    # dynamics
    c = build_capsule_dumbbell(R_l=0.55, R_r=0.55, L=2.0, w=w,
                               h=H_REF)
    pos, ang = c["positions"], c["angles"]
    v = OrientedPointCloudVarifold(positions=pos, angles=ang)
    nor = v.normals
    delta, tau = map(float, compute_recommended_params(pos))
    m = teb.resolve_m(pos, nor, delta, tau)
    q = compute_coherence(v, m, SIGMA, "wendland_c2")
    neck = (pos[:, 0].abs() <= c["L"] / 4)
    qbar = float(q[neck].mean())
    P_geom = c["perimeter"]
    A = c["area"]
    rec = dict(w=w, w_over_sigma=w_over_sigma, grid=grid,
               w_over_h=w / H_REF, w_over_fill=w / EPS_FILL,
               N=int(pos.shape[0]), qbar_neck=qbar,
               one_minus_qbar=1.0 - qbar,
               P_sigma_rel=float((m * q).sum()) / P_geom - 1.0)
    try:
        from src.torch.solver.mm_step import MMStepper
        from src.torch.transport.grid_wasserstein import GridMetricConfig
        from src.torch.transport.phase_grid import PhaseGridConfig
        cfg = teb.build_config(teb.production_args(
            grid=grid, redist_monotone=True, quotient_mode="off"),
            delta, tau)
        # Phase P domain: the dumbbell exceeds the production box
        # (-2,2); extend to (-3,3) and scale H by 1.5 so dx and
        # eps_fill/dx are BIT-IDENTICAL to the production grid
        # (768 -> 1152, 1024 -> 1536): the P0-c admissibility window
        # transfers unchanged.
        cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
            grid_shape=(int(grid * 1.5), int(grid * 1.5)),
            box_min=(-3.0, -3.0), box_max=(3.0, 3.0),
            fill_epsilon=EPS_FILL,
            support_threshold=3e-3, projection_rel_tol=5e-2),
            compatibility_components="conservative_sweep")
        cfg.grid_bulk_first_moment_rows = True
        cfg.grid_bulk_first_moment_rows_form = "current_centered"
        st = MMStepper(cfg)
        st._setup_step(OrientedPointCloudVarifold(
            positions=pos.clone(), angles=ang.clone()))
        sn = st._last_grid_snapshot
        rec["phase_grid_volume_rel"] = sn.phase_grid_volume / A - 1.0
        rec["n_components"] = sn.n_components
        rec["grid_status"] = "ok"
    except Exception as e:                        # noqa: BLE001
        rec["grid_status"] = f"{type(e).__name__}: {str(e)[:100]}"
        rec["phase_grid_volume_rel"] = None
    rec["window_pass"] = bool(
        abs(rec["one_minus_qbar"]) <= 0.05
        and abs(rec["P_sigma_rel"]) <= 0.05
        and rec["phase_grid_volume_rel"] is not None
        and abs(rec["phase_grid_volume_rel"]) <= 0.05
        and w >= 3 * H_REF)
    return rec


def main():
    torch.set_default_dtype(torch.float64)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = [(0.5, 768), (0.75, 768), (1.0, 768), (1.25, 768),
             (1.5, 768), (2.0, 768), (0.75, 1024), (1.0, 1024),
             (1.25, 1024)]
    for ws, grid in cases:
        r = measure(ws, grid)
        rows.append(r)
        print(f"  w/sigma {ws:4.2f} H={grid}: 1-qbar "
              f"{r['one_minus_qbar']:+.5f} P_sig {r['P_sigma_rel']:+.5f}"
              f" phase_vol "
              f"{r['phase_grid_volume_rel'] if r['phase_grid_volume_rel'] is None else round(r['phase_grid_volume_rel'], 5)}"
              f" [{r['grid_status'][:45]}] pass={r['window_pass']}",
              flush=True)
    (OUT / "p0c_prime_capsule.json").write_text(
        json.dumps(dict(h=H_REF, sigma=SIGMA, eps_fill=EPS_FILL,
                        fixture="capsule R=0.55 L=2.0 (derived blend) "
                                "(lobes/junction fixed, only w varies)",
                        rows=rows), indent=1, default=str))
    print(f"wrote {OUT / 'p0c_prime_capsule.json'}")


if __name__ == "__main__":
    main()
