"""0L-theta1: kinematic consistency of the angle transport.

Arms: U_q (union kernel + m q^full, current production) / L_q
(loopwise + m q^full) / L_m (loopwise + raw loopwise mass) /
L_m_cons (+ constrained mean-zero solve).

(a) constant mode |D 1|_inf and weighted mean |m^T D s| on clean
    circles and the organic states 1300/1525 (VALIDATION only);
(b) Fourier 2x2 transfer matrices T_k on the mass-weighted
    orthonormalized {cos k, sin k} basis vs the exact
    (k/R) [[0,-1],[1,0]] -- amplitude, phase, cos/sin asymmetry and
    orientation sign in one number ||T_k - exact||_2; measured on
    the OUTER circle and on an INWARD-normal hole circle (sign
    pin); resolved band k <= R/sigma_angle; plus the max gain
    ||D||_{m->m} on the retained subspace;
(c) oracle audits at endgame-#3 1525: the normal-only fixture
    (pure hole radial motion, true dtheta = -d_tau v_n with
    v_t = 0) AND the full geometric oracle
    dtheta = -d_tau v_n + kappa^poly v_t for the ACTUAL MM
    displacement (v_n, v_t decomposed in the certified polygon
    frame) -- distinguishing a consistency failure from a genuine
    tangential contribution;
(d) d/sigma_angle scaling on a CLEAN synthetic concentric family
    d = delta sigma_angle, delta in {0.8,0.6,0.5,0.3,0.15},
    sigma_angle in {0.06,0.08,0.10} (delta_KDE and sampling fixed/
    recorded; organic states never calibrate the law).

Usage:
    uv run python scripts/experiments/angle_kinematics_audit.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import OUT, build_cloud  # noqa: E402

from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.loopwise_mass import (  # noqa: E402
    resolve_loopwise_oriented_mass_source,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.perimeter.angle_constraint import (  # noqa: E402
    compute_angle_constraint_matrices_from_weights,
)
from src.torch.solver.mm_step import (  # noqa: E402
    solve_angle_map,
    solve_angle_map_blocks,
)
from src.torch.transport.bem_wasserstein import (  # noqa: E402
    compute_coherence,
)
from src.torch.transport.loop_geometry import (  # noqa: E402
    certified_cyclic_order,
    polygon_dual,
)

DT = torch.float64
ARMS = ("U_q", "L_q", "L_m", "L_m_cons")


def build_map(arm, v, tang, m, q, lbl, sigma_angle):
    w = m if arm in ("L_m", "L_m_cons") else m * q
    if arm == "U_q":
        A, B = compute_angle_constraint_matrices_from_weights(
            v.positions, tang, w, sigma_angle, "wendland_c2")
        return solve_angle_map(A, B).AB_solve
    return solve_angle_map_blocks(
        v.positions, tang, w, lbl, sigma_angle, "wendland_c2",
        consistency=("mean_zero" if arm == "L_m_cons"
                     else "none")).AB_solve


def circle_state(R, n, inward=False):
    t = torch.arange(n, dtype=DT) * 2 * math.pi / n
    pos = torch.stack([R * t.cos(), R * t.sin()], 1)
    ang = t + (math.pi if inward else 0.0)
    v = OrientedPointCloudVarifold(positions=pos, angles=ang)
    tang = torch.stack([-v.normals[:, 1], v.normals[:, 0]], 1)
    m = torch.full((n,), 2 * math.pi * R / n, dtype=DT)
    q = torch.ones(n, dtype=DT)
    lbl = torch.zeros(n, dtype=torch.long)
    return v, tang, m, q, lbl, t


def fourier_transfer(D, m, t, R, k, orient=1.0):
    """2x2 transfer on the m-weighted orthonormalized {c_k, s_k};
    exact = orient * (k/R) [[0,-1],[1,0]] for the stored-tangent
    convention (orient flips on inward-normal loops)."""
    c = torch.cos(k * t)
    s = torch.sin(k * t)

    def ip(a, b):
        return float((m * a * b).sum())

    nc, ns = math.sqrt(ip(c, c)), math.sqrt(ip(s, s))
    c, s = c / nc, s / ns
    T = torch.tensor([[ip(c, D @ c), ip(c, D @ s)],
                      [ip(s, D @ c), ip(s, D @ s)]], dtype=DT)
    exact = orient * (k / R) * torch.tensor([[0.0, -1.0],
                                             [1.0, 0.0]], dtype=DT)
    return float(torch.linalg.matrix_norm(T - exact, ord=2)), \
        float(torch.linalg.matrix_norm(exact, ord=2))


def audit_circle(name, R, n, inward, sigma_angle, report):
    v, tang, m, q, lbl, t = circle_state(R, n, inward=inward)
    k_res = max(2, int(R / sigma_angle))
    rec = dict(fixture=name, R=R, n=n, inward=inward,
               sigma_angle=sigma_angle, k_resolved=k_res, arms={})
    # orientation of the exact operator: stored tangent on an
    # inward-normal circle runs clockwise -> sign flips
    orient = -1.0 if inward else 1.0
    for arm in ARMS:
        D = build_map(arm, v, tang, m, q, lbl, sigma_angle)
        const = float((D @ torch.ones(n, dtype=DT)).abs().max())
        errs = {}
        for k in range(1, k_res + 1):
            e, scale = fourier_transfer(D, m, t, R, k, orient)
            errs[k] = e / max(scale, 1e-30)
        gain = float(torch.linalg.matrix_norm(
            torch.diag(m.sqrt()) @ D @ torch.diag(1 / m.sqrt()),
            ord=2))
        rec["arms"][arm] = dict(const_mode=const,
                                fourier_rel=errs,
                                worst_fourier=max(errs.values()),
                                gain_m=gain)
        print(f"  [{name}/{arm}] |D1| {const:.2e}  worst-k rel "
              f"{max(errs.values()):.3f} (k1 {errs[1]:.3f})  "
              f"gain {gain:.1f}", flush=True)
    report.append(rec)


def main():
    torch.set_default_dtype(DT)
    ck = torch.load(OUT / "exact_merger_endgame3_states.pt",
                    weights_only=True)
    v0, _ = build_cloud()
    DELTA, TAU = map(float, compute_recommended_params(v0.positions))
    report = dict(circles=[], organic=[], oracle=None, scaling=[])

    print("== (a)+(b) clean circles ==", flush=True)
    audit_circle("outer_R1", 1.0, 256, False, 0.1,
                 report["circles"])
    audit_circle("disk_R035", 0.35, 90, False, 0.1,
                 report["circles"])
    audit_circle("hole_R042_inward", 0.42, 141, True, 0.1,
                 report["circles"])

    print("== (a) organic states (validation) ==", flush=True)
    for step in (1300, 1525):
        st = ck[step]
        v = OrientedPointCloudVarifold(positions=st["positions"],
                                       angles=st["angles"])
        res = resolve_loopwise_oriented_mass_source(
            v.positions, v.normals, DELTA, TAU)
        m, lbl = res.m_loop, res.loop_labels_pre
        q = compute_coherence(v, m, 0.1, "wendland_c2")
        tang = torch.stack([-v.normals[:, 1], v.normals[:, 0]], 1)
        rec = dict(step=step, arms={})
        for arm in ARMS:
            D = build_map(arm, v, tang, m, q, lbl, 0.1)
            const = float((D @ torch.ones(v.n_points,
                                          dtype=DT)).abs().max())
            rec["arms"][arm] = dict(const_mode=const)
            print(f"  [{step}/{arm}] |D1| {const:.3e}", flush=True)
        report["organic"].append(rec)

    print("== (c) oracle audits at 1525 ==", flush=True)
    st = ck[1525]
    v = OrientedPointCloudVarifold(positions=st["positions"],
                                   angles=st["angles"])
    res = resolve_loopwise_oriented_mass_source(
        v.positions, v.normals, DELTA, TAU)
    m, lbl = res.m_loop, res.loop_labels_pre
    q = compute_coherence(v, m, 0.1, "wendland_c2")
    tang = torch.stack([-v.normals[:, 1], v.normals[:, 0]], 1)
    orders = certified_cyclic_order(v.positions, lbl, v.normals, m,
                                    float(m.median()))
    # actual-motion proxy: previous->this committed displacement
    st_prev = ck[1500]
    dx = v.positions - st_prev["positions"]
    orc = dict(states="1500->1525", loops={})
    r = v.positions.norm(dim=1)
    wall = r < 0.6
    outw = (v.positions * v.normals).sum(1) > 0
    roles = {}
    for lb, lo in orders.items():
        sel = lbl == lb
        rmean = float(r[sel].mean())
        roles[lb] = ("disk" if rmean < 0.4 and outw[sel].float()
                     .mean() > 0.5 else
                     "hole" if rmean < 0.6 else "outer")
    for lb, lo in orders.items():
        idx = lo.order
        m_poly, nu = polygon_dual(v.positions, idx)
        tau_p = torch.stack([-nu[:, 1], nu[:, 0]], 1)
        v_n = (dx[idx] * nu).sum(1)
        v_t = (dx[idx] * tau_p).sum(1)
        # polygon derivative d_tau f ~ (f_{i+1}-f_{i-1}) / (2 sbar)
        e = v.positions[idx].roll(-1, 0) - v.positions[idx]
        sbar = 0.5 * (e.norm(dim=1) + e.roll(1, 0).norm(dim=1))
        dtau_vn = (v_n.roll(-1) - v_n.roll(1)) / (2 * sbar)
        ang_e = torch.atan2(e[:, 1], e[:, 0])
        alpha = torch.remainder(ang_e - ang_e.roll(1) + math.pi,
                                2 * math.pi) - math.pi
        kap = alpha / sbar.clamp_min(1e-30)
        oracle_full = -dtau_vn + kap * v_t
        oracle_normal_only = -dtau_vn
        # arms applied to the stored-normal displacement field
        s_field = torch.zeros(v.n_points, dtype=DT)
        s_field[idx] = (dx[idx] * v.normals[idx]).sum(1)
        row = {}
        for arm in ("U_q", "L_m"):
            D = build_map(arm, v, tang, m, q, lbl, 0.1)
            pred = (D @ s_field)[idx]
            row[arm] = dict(
                rms_vs_full=float(((pred - oracle_full) ** 2)
                                  .mean() ** 0.5),
                rms_vs_normal_only=float(
                    ((pred - oracle_normal_only) ** 2)
                    .mean() ** 0.5),
                pred_rms=float((pred ** 2).mean() ** 0.5),
                oracle_full_rms=float((oracle_full ** 2)
                                      .mean() ** 0.5),
                vt_rms=float((v_t ** 2).mean() ** 0.5),
            )
        orc["loops"][roles[lb]] = row
        print(f"  [{roles[lb]}] U_q rms-vs-full "
              f"{row['U_q']['rms_vs_full']:.2e}  L_m "
              f"{row['L_m']['rms_vs_full']:.2e}  (oracle rms "
              f"{row['L_m']['oracle_full_rms']:.2e}, v_t rms "
              f"{row['L_m']['vt_rms']:.2e})", flush=True)
    report["oracle"] = orc

    print("== (d) d/sigma_angle scaling (clean family) ==",
          flush=True)
    for sig_a in (0.06, 0.08, 0.10):
        for dl in (0.8, 0.6, 0.5, 0.3, 0.15):
            gap = dl * sig_a
            p1, a1 = (lambda R, n: (
                torch.stack([R * (torch.arange(n, dtype=DT) * 2
                                  * math.pi / n).cos(),
                             R * (torch.arange(n, dtype=DT) * 2
                                  * math.pi / n).sin()], 1),
                torch.arange(n, dtype=DT) * 2 * math.pi / n))(
                    0.35, 90)
            R2 = 0.35 + gap
            n2 = 141
            t2 = torch.arange(n2, dtype=DT) * 2 * math.pi / n2
            p2 = torch.stack([R2 * t2.cos(), R2 * t2.sin()], 1)
            a2 = t2 + math.pi
            pos = torch.cat([p1, p2])
            ang = torch.cat([a1, a2])
            vv = OrientedPointCloudVarifold(positions=pos,
                                            angles=ang)
            lblx = torch.cat([torch.zeros(90, dtype=torch.long),
                              torch.ones(n2, dtype=torch.long)])
            mm = torch.cat([
                torch.full((90,), 2 * math.pi * 0.35 / 90,
                           dtype=DT),
                torch.full((n2,), 2 * math.pi * R2 / n2, dtype=DT)])
            qq = torch.ones(pos.shape[0], dtype=DT)
            tng = torch.stack([-vv.normals[:, 1], vv.normals[:, 0]],
                              1)
            A, B = compute_angle_constraint_matrices_from_weights(
                vv.positions, tng, mm, sig_a, "wendland_c2")
            D = solve_angle_map(A, B).AB_solve
            mo = torch.zeros(pos.shape[0], dtype=DT)
            mo[lblx == 1] = 1e-4
            leak = float((D @ mo)[lblx == 0].abs().max())
            report["scaling"].append(dict(sigma_angle=sig_a,
                                          d_over_sig=dl, gap=gap,
                                          disk_leak=leak))
            print(f"  [sig_a {sig_a} d/sig {dl}] disk leak "
                  f"{leak:.3e}", flush=True)

    out_p = Path("results/reports/phase3c0l_theta1.json")
    out_p.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
