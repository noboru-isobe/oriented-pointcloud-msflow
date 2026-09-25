"""P0-c: static parallel-wall validity calibration (Phase P plan,
reviewer 2026-08-26). A single closed loop cannot be protected by the
0J loopwise correction (q^WB = q^full when there is one loop), so the
anti-parallel neck walls feel the coherence cancellation directly at
O(sigma) separation. This fixes, BEFORE any dynamic run, the window

    w >= max(c_h h, c_sigma sigma, c_f eps_fill)        (P.3)

inside which neck telemetry may be read as sharp-interface PDE
evidence.

Fixture: stadium (two straight anti-parallel walls of length L_s at
distance w + two semicircular caps), arc-length sampled at the
production spacing h_ref. Static measurements per w:
  qbar_neck   mean q_full over the straight-wall particles
  P_sigma     sum m q (the sigma-regularized perimeter) vs exact P
  Q_grid      phase-grid volume vs analytic area (stepper setup);
              grid fail-closed exceptions recorded as such
Pre-registered acceptance: |1 - qbar_neck| <= 0.05 AND
|P_sigma - P_geom|/P_geom <= 0.05 AND |phase_grid_volume - A|/A <=
0.05 (plus w >= 3h). NOTE (reviewer): phase_grid_volume is the phase
RECONSTRUCTION volume, not the tangent metric Q_grid -- this static
window is the carrier/incidence/phase-reconstruction ADMISSIBILITY
window; "dynamic sharp-regime" needs the P1-2 rate pin. The smallest
passing w/sigma
and w/eps_fill are recorded into docs/pinchoff_study.md section P0-c.

Usage:
    uv run python scripts/experiments/p0c_wall_calibration.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import two_ellipses_benchmark as teb  # noqa: E402  (build_config etc.)
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.transport.bem_wasserstein import compute_coherence  # noqa

H_REF = teb.H_REF                     # 2 pi / 256 ~ 0.02454
SIGMA = teb.SIGMA                     # 0.1
EPS_FILL = 0.04                       # production fill_epsilon
OUT = Path("results/pinchoff")


def stadium(w: float, L_s: float, h: float):
    """Closed stadium: walls y = +-w/2 for |x| <= L_s/2, semicircle
    caps radius w/2. CCW order, outward normals. Returns positions,
    angles, wall mask (straight-wall particles)."""
    r = w / 2.0
    n_wall = max(int(round(L_s / h)), 4)
    n_cap = max(int(round(math.pi * r / h)), 4)
    xs = torch.linspace(-L_s / 2, L_s / 2, n_wall + 1,
                        dtype=torch.float64)[:-1]
    # bottom wall, left -> right, outward normal (0,-1)
    bot = torch.stack([xs, torch.full_like(xs, -r)], 1)
    a_bot = torch.full((n_wall,), -math.pi / 2, dtype=torch.float64)
    # right cap, -pi/2 -> pi/2
    t = torch.linspace(-math.pi / 2, math.pi / 2, n_cap + 1,
                       dtype=torch.float64)[:-1]
    cap_r = torch.stack([L_s / 2 + r * t.cos(), r * t.sin()], 1)
    a_r = t.clone()
    # top wall, right -> left, outward normal (0,+1)
    top = torch.stack([xs.flip(0), torch.full_like(xs, r)], 1)
    a_top = torch.full((n_wall,), math.pi / 2, dtype=torch.float64)
    # left cap, pi/2 -> 3pi/2
    t2 = torch.linspace(math.pi / 2, 3 * math.pi / 2, n_cap + 1,
                        dtype=torch.float64)[:-1]
    cap_l = torch.stack([-L_s / 2 + r * t2.cos(), r * t2.sin()], 1)
    a_l = t2.clone()
    pos = torch.cat([bot, cap_r, top, cap_l])
    ang = torch.cat([a_bot, a_r, a_top, a_l])
    wall = torch.zeros(pos.shape[0], dtype=torch.bool)
    wall[:n_wall] = True
    wall[n_wall + n_cap:2 * n_wall + n_cap] = True
    # keep only the CENTRAL half of each wall (junction-free zone)
    central = torch.zeros_like(wall)
    central[:n_wall] = xs.abs() <= L_s / 4
    central[n_wall + n_cap:2 * n_wall + n_cap] = \
        xs.flip(0).abs() <= L_s / 4
    return pos, ang, wall & central


def main():
    torch.set_default_dtype(torch.float64)
    OUT.mkdir(parents=True, exist_ok=True)
    L_s = 2.0
    rows = []
    print(f"P0-c stadium calibration: h={H_REF:.4f} sigma={SIGMA} "
          f"eps_fill={EPS_FILL} L_s={L_s}", flush=True)
    for w_over_sigma in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0,
                         4.0, 6.0):
        w = w_over_sigma * SIGMA
        pos, ang, wallc = stadium(w, L_s, H_REF)
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        nor = v.normals
        delta, tau = map(float, compute_recommended_params(pos))
        m = teb.resolve_m(pos, nor, delta, tau)
        q = compute_coherence(v, m, SIGMA, "wendland_c2")
        qbar = float(q[wallc].mean())
        A = L_s * w + math.pi * (w / 2) ** 2
        P_ex = 2 * L_s + math.pi * w
        P_sig = float((m * q).sum())
        rec = dict(w=w, w_over_sigma=w_over_sigma,
                   w_over_h=w / H_REF, w_over_fill=w / EPS_FILL,
                   N=int(pos.shape[0]), qbar_neck=qbar,
                   one_minus_qbar=1.0 - qbar,
                   P_sigma_rel=P_sig / P_ex - 1.0, A_exact=A)
        # grid reconstruction via a production stepper setup
        try:
            from src.torch.solver.mm_step import MMStepper
            cfg = teb.build_config(teb.production_args(
                grid=768, redist_monotone=True, quotient_mode="off"),
                delta, tau)
            cfg.grid_bulk_first_moment_rows = True
            cfg.grid_bulk_first_moment_rows_form = "current_centered"
            st = MMStepper(cfg)
            st._setup_step(OrientedPointCloudVarifold(
                positions=pos.clone(), angles=ang.clone()))
            sn = st._last_grid_snapshot
            rec["phase_grid_volume_rel"] = sn.phase_grid_volume / A - 1.0
            rec["n_components"] = sn.n_components
            rec["grid"] = "ok"
        except Exception as e:                    # noqa: BLE001
            rec["grid"] = f"{type(e).__name__}: {str(e)[:110]}"
            rec["phase_grid_volume_rel"] = None
        ok = (abs(rec["one_minus_qbar"]) <= 0.05
              and abs(rec["P_sigma_rel"]) <= 0.05
              and rec["phase_grid_volume_rel"] is not None
              and abs(rec["phase_grid_volume_rel"]) <= 0.05
              and w >= 3 * H_REF)
        rec["window_pass"] = bool(ok)
        rows.append(rec)
        print(f"  w/sigma {w_over_sigma:4.2f} (w/h {w/H_REF:5.2f}, "
              f"w/fill {w/EPS_FILL:5.2f}): 1-qbar "
              f"{rec['one_minus_qbar']:+.4f} P_sig rel "
              f"{rec['P_sigma_rel']:+.4f} phase_vol "
              f"{rec['phase_grid_volume_rel'] if rec['phase_grid_volume_rel'] is None else round(rec['phase_grid_volume_rel'], 5)}"
              f" [{rec['grid'][:40]}] pass={ok}", flush=True)
    passing = [r for r in rows if r["window_pass"]]
    summary = dict(
        h=H_REF, sigma=SIGMA, eps_fill=EPS_FILL, L_s=L_s,
        min_pass_w_over_sigma=(min(r["w_over_sigma"] for r in passing)
                               if passing else None),
        min_pass_w_over_fill=(min(r["w_over_fill"] for r in passing)
                              if passing else None),
        rows=rows)
    (OUT / "p0c_wall_calibration.json").write_text(
        json.dumps(summary, indent=1, default=str))
    print(f"window: min passing w/sigma = "
          f"{summary['min_pass_w_over_sigma']}, w/fill = "
          f"{summary['min_pass_w_over_fill']}")
    print(f"wrote {OUT / 'p0c_wall_calibration.json'}")


if __name__ == "__main__":
    main()
