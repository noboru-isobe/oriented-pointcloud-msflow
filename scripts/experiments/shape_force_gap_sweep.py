"""0H-0: moving-minimizer gap sweep + gradient channel attribution.

Reviewer protocol (2026-08-14): before any continuation past the 0F
shape stop, test the SELF-EXTINCTION hypothesis of the
estimator-induced shape drift statically, and attribute the force to
the mass channel vs the sigma-energy structure.

For synthetic states X_d (the 1400 wall contracted radially to mean
gap d/sigma in {0.6, 0.45, 0.30, 0.20, 0.10, 0}) measure along the
aligned k=2 mode xi_2:

    g2(d)  = D P_hat(X_d)[xi_2]        (production P: frozen q)
    H2(d)  = D^2 J_d(X_d)[xi_2, xi_2]  (full objective curvature)
    a*(d)  = -g2(d) / H2(d)            (per-step equilibrium displacement)

Necessary condition for self-extinction: g2 -> 0 and a* -> 0 as
d/sigma -> 0; the a*-integrated cumulative amplitude along the
observed gap trajectory must stay below the isolated-disk shape
envelope (from the 0F-cal states) plus one spacing.

Channel attribution. STRUCTURAL FACT: the production objective
freezes q (P = sum m~(Y) q_X, mm_step.py:1668), so the within-step
shape force is 100% the mass-estimator channel by construction. The
decisive control is therefore the ARC-LENGTH cell: replace m~(Y) by
polygon arclength weights m_arc(Y) (loop order known in this
benchmark) and re-measure g2. g_arc ~ 0 with g_full = O(1) pins the
oriented-KDE mass response as the culprit (-> omega_2 smoothstep
trial); g_arc comparable means the sigma-energy q_X-weighted
arclength response itself carries the bias.

Usage:
    uv run python scripts/experiments/shape_force_gap_sweep.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import (  # noqa: E402
    OUT,
    build_cloud,
    masses_for,
)
from incidence_rows_onestep import make_config  # noqa: E402
from shape_mode_audit import disk_sheet, fourier, manual_objective  # noqa: E402

from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.solver.mm_step import MMStepper  # noqa: E402

DT = torch.float64
SIGMA = 0.1
GAP_FRACS = (0.6, 0.45, 0.30, 0.20, 0.10, 0.0)


def synthetic_state(v, d_target):
    """Contract the two wall sheets radially (about the wall midline,
    preserving each sheet's shape) to mean gap d_target; outer ring
    untouched."""
    pos = v.positions.clone()
    r = pos.norm(dim=1)
    rhat = pos / r[:, None]
    wall = r < 0.6
    outward = (pos * v.normals).sum(1) > 0
    disk_m = wall & outward
    hole_m = wall & ~outward
    rd = float(r[disk_m].mean())
    rh = float(r[hole_m].mean())
    mid = 0.5 * (rd + rh)
    shift_d = (mid - 0.5 * d_target) - rd     # move disk sheet
    shift_h = (mid + 0.5 * d_target) - rh     # move hole sheet
    pos[disk_m] += shift_d * rhat[disk_m]
    pos[hole_m] += shift_h * rhat[hole_m]
    return OrientedPointCloudVarifold(positions=pos, angles=v.angles)


def arc_masses(v, loops):
    """Polygon arclength weights: order each loop by angle about its
    centroid (valid for these near-circular sheets), half the sum of
    adjacent edge lengths."""
    N = v.positions.shape[0]
    m = torch.zeros(N, dtype=DT)
    for lb in loops.unique():
        idx = torch.nonzero(loops == lb).flatten()
        p = v.positions[idx]
        c = p.mean(0)
        order = torch.argsort(torch.atan2(p[:, 1] - c[1],
                                          p[:, 0] - c[0]))
        q = p[order]
        nxt = torch.roll(q, -1, dims=0)
        e = (nxt - q).norm(dim=1)
        mm = 0.5 * (e + torch.roll(e, 1, dims=0))
        m[idx[order]] = mm
    return m


def main():
    ck = torch.load(OUT / "exact_merger_0f_oriented_states.pt",
                    weights_only=True)
    st = ck[1400]
    v0me = OrientedPointCloudVarifold(positions=st["positions"],
                                      angles=st["angles"])
    v0, _ = build_cloud()
    delta, tau = map(float, compute_recommended_params(v0.positions))
    m0 = masses_for("oriented_kde", v0.positions, v0.normals,
                    delta, tau)
    target0 = float(
        0.5 * (m0 * (v0.positions * v0.normals).sum(-1)).sum())

    # k=2 pattern (aligned with the current deformation at 1400)
    m1400 = masses_for("oriented_kde", v0me.positions, v0me.normals,
                       delta, tau)
    mask, w, rad, th = disk_sheet(v0me, m1400)
    idx = torch.nonzero(mask).flatten()
    _, four = fourier(w, rad, th)
    a2, b2, amp2 = four[2]

    # isolated-disk shape envelope from the 0F-cal states
    cal = torch.load(OUT / "exact_merger_0fcal_states.pt",
                     weights_only=True)
    amps = []
    for s_, stc in sorted(cal.items()):
        vc = OrientedPointCloudVarifold(positions=stc["positions"],
                                        angles=stc["angles"])
        mc = masses_for("oriented_kde", vc.positions, vc.normals,
                        delta, tau)
        _, wc, rc, tc = disk_sheet(vc, mc)
        _, fc = fourier(wc, rc, tc)
        amps.append(fc[2][2])
    env_A2 = max(amps)
    print(f"isolated-disk A2 envelope (0F-cal): {env_A2:.3e} "
          f"(current 1400 A2 = {amp2:.3e})", flush=True)

    from src.torch.transport.incidence import oriented_graph_components

    report = dict(env_A2=env_A2, A2_1400=amp2, cells=[])
    for frac in GAP_FRACS:
        d = frac * SIGMA
        vd = synthetic_state(v0me, d)
        cell = dict(d_over_sigma=frac, d=d)
        m = masses_for("oriented_kde", vd.positions, vd.normals,
                       delta, tau)
        maskd, wd, radd, thd = disk_sheet(vd, m)
        idxd = torch.nonzero(maskd).flatten()
        xi = torch.zeros(vd.positions.shape[0], dtype=DT)
        xi[idxd] = (a2 * torch.cos(2 * thd)
                    + b2 * torch.sin(2 * thd)) / amp2
        nrmM = float((m * xi * xi).sum())
        try:
            cfg = make_config("augment", delta, tau, target0)
            cfg.mass_estimator = "oriented_kde"
            stp = MMStepper(cfg)
            stp._grid_target_volume_initial = target0
            stp._setup_step(vd)
            p = stp.param
            xi_a = p.Q @ (p.Q.T @ xi)
            dth = p.AB_solve @ xi_a
            t = 1e-4
            J0, P0, _ = manual_objective(
                stp, torch.zeros_like(xi), torch.zeros_like(xi))
            Jp_, Pp_, _ = manual_objective(stp, t * xi_a, t * dth)
            Jm_, Pm_, _ = manual_objective(stp, -t * xi_a, -t * dth)
            g2 = float((Pp_ - Pm_) / (2 * t))
            H2 = float((Jp_ + Jm_ - 2 * J0) / (t * t * nrmM))
            cell.update(g2=g2, H2=H2,
                        a_star=(-g2 / (H2 * nrmM)
                                if H2 > 0 else None))
            # arc-length control (same xi_a; frozen q from stepper)
            loops = oriented_graph_components(
                vd.positions, vd.normals, 4.0 * float(m.median()))

            def p_arc(s, dthv):
                pos = p.prev_positions + s[:, None] * p.prev_normals
                from src.torch.math_utils.angles import wrap_angles
                ang = wrap_angles(p.prev_angles + dthv)
                va = OrientedPointCloudVarifold(positions=pos,
                                                angles=ang)
                ma = arc_masses(va, loops)
                return float((ma * stp.fixed_coherence).sum())

            g_arc = float((p_arc(t * xi_a, t * dth)
                           - p_arc(-t * xi_a, -t * dth)) / (2 * t))
            cell["g2_arc_control"] = g_arc
        except Exception as exc:      # noqa: BLE001 record and go on
            cell["error"] = f"{type(exc).__name__}: {exc}"
        report["cells"].append(cell)
        print(json.dumps(cell, default=str), flush=True)

    ok = [c for c in report["cells"] if "g2" in c]
    if len(ok) >= 2:
        g_ends = (ok[0]["g2"], ok[-1]["g2"])
        print(f"\ng2: {g_ends[0]:+.3e} (d/sigma={ok[0]['d_over_sigma']}) "
              f"-> {g_ends[1]:+.3e} (d/sigma={ok[-1]['d_over_sigma']})",
              flush=True)
    out = Path("results/reports/phase3c0h_gap_sweep.json")
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
