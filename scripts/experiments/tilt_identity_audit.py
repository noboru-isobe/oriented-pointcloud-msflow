"""0L-A0: exact tilt/quadrature decomposition + local sigma-sweep.

(a) On the real endgame-#2 states (1320 / 1328 / 1340 / deepest
    saved), per loop, the EXACT decomposition (reviewer 0L section 4):

        V_cur - A_poly = T_tilt + T_quad,
        T_tilt = 1/2 sum m~_i x_i . (n_i - nu^poly_i)
        T_quad = 1/2 sum (m~_i - m^poly_i) x_i . nu^poly_i

    closed to machine precision, plus eps_n and the transversality
    telemetry. This closes the attribution "the V_cur / V_geom
    divergence is normal tilt" quantitatively, separated from the
    H^1-quadrature channel.

(b) Local sigma-sweep WITH a tilt field (reviewer section 7: pure
    radial synthetic states have eps_n = 0 identically and cannot
    probe the anomaly): the dimensionless per-sheet tilt profile
    phi(theta) measured at endgame-#2 step 1340 is transplanted onto
    the synthetic (sigma, N) cells (m10/m075/m05) at the SAME
    d/sigma, then eps_n, V_cur - A_poly, |C^geom|, and the one-step
    rush rate (production current-rows baseline) are compared across
    sigma. Observation only -- no gates.

Usage:
    uv run python scripts/experiments/tilt_identity_audit.py
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
from sigma_n_refinement import (  # noqa: E402
    CELLS,
    cell_config,
    organic_reference,
    synthetic_cell_state,
)

import sigma_n_refinement as snr  # noqa: E402

from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.loopwise_mass import (  # noqa: E402
    resolve_loopwise_oriented_mass_source,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.transport.loop_geometry import (  # noqa: E402
    certified_cyclic_order,
    polygon_area,
    polygon_dual,
    polygon_gradient,
)

DT = torch.float64
# reviewer asked for 1320/1328/1340/1354; the endgame-#2 checkpoints
# are on the 25-step grid, so the nearest saved states are used
# (documented substitution): 1300 pre-rush, 1325 rush onset, 1350
# mid-rush, plus the deepest saved state.
STEPS = (1300, 1325, 1350)
PROFILE_STEP = 1350             # tilt-profile source (1340 unsaved)
DELTA_OVER_SIGMA = 0.71         # endgame2 step-1350 gap / sigma


def loop_roles(labels, positions, m):
    r = positions.norm(dim=1)
    rad = {int(lb): float((m[labels == lb] * r[labels == lb]).sum()
                          / m[labels == lb].sum())
           for lb in labels.unique()}
    order = sorted(rad, key=rad.get)
    roles = {order[0]: "disk", order[-1]: "outer"}
    for lb in order[1:-1]:
        roles[lb] = "hole"
    return roles


def decompose_state(v, delta, tau):
    """Per-loop exact decomposition record + tilt profiles."""
    res = resolve_loopwise_oriented_mass_source(
        v.positions, v.normals, delta, tau)
    labels = res.loop_labels_pre
    m = res.m_loop
    orders = certified_cyclic_order(v.positions, labels, v.normals,
                                    m, float(m.median()))
    roles = loop_roles(labels, v.positions, m)
    recs, profiles = {}, {}
    for lb, lo in orders.items():
        idx = lo.order
        mm = m[idx]
        x = v.positions[idx]
        n = v.normals[idx]
        m_poly, nu_poly = polygon_dual(v.positions, idx)
        a_poly = float(polygon_area(v.positions, idx))
        v_cur = float(0.5 * (mm * (x * n).sum(1)).sum())
        t_tilt = float(0.5 * (mm * (x * (n - nu_poly)).sum(1)).sum())
        t_quad = float(0.5 * ((mm - m_poly)
                              * (x * nu_poly).sum(1)).sum())
        resid = abs(v_cur - a_poly - t_tilt - t_quad)
        eps_n = float(((mm * ((n - nu_poly) ** 2).sum(1)).sum()
                       / mm.sum()) ** 0.5)
        recs[roles[lb]] = dict(
            A_poly=a_poly, V_cur=v_cur, T_tilt=t_tilt, T_quad=t_quad,
            identity_resid=resid, eps_n=eps_n,
            min_transversality=lo.min_transversality,
            mean_transversality=lo.mean_transversality,
        )
        # dimensionless tilt profile phi(theta) (signed angle from
        # nu_poly to n) vs polar angle about the loop center
        c = lo.center
        theta = torch.atan2(x[:, 1] - c[1], x[:, 0] - c[0])
        phi = torch.atan2(
            n[:, 0] * nu_poly[:, 1] - n[:, 1] * nu_poly[:, 0],
            (n * nu_poly).sum(1))
        profiles[roles[lb]] = (theta, -phi)
        # NOTE sign: phi is the rotation FROM n TO nu_poly; applying
        # -phi to a radial normal reproduces the stored tilt
    return recs, profiles


def transplant_tilt(v, labels, roles_by_label, profiles):
    """Rotate each sheet's stored normals by the transplanted
    dimensionless profile phi(theta) (nearest-theta interpolation)."""
    ang = v.angles.clone()
    for lb, role in roles_by_label.items():
        if role not in profiles:
            continue
        th_src, phi_src = profiles[role]
        sel = labels == lb
        x = v.positions[sel]
        c = x.mean(0)
        th = torch.atan2(x[:, 1] - c[1], x[:, 0] - c[0])
        d = torch.remainder(th[:, None] - th_src[None, :] + math.pi,
                            2 * math.pi) - math.pi
        nearest = d.abs().argmin(dim=1)
        ang[sel] = ang[sel] + phi_src[nearest]
    return OrientedPointCloudVarifold(positions=v.positions,
                                      angles=ang)


def main():
    torch.set_default_dtype(DT)
    ck = torch.load(OUT / "exact_merger_endgame2_states.pt",
                    weights_only=True)
    deepest = max(k for k in ck.keys())
    steps = list(STEPS) + [deepest]
    v0, _ = build_cloud()
    delta, tau = map(float, compute_recommended_params(v0.positions))
    report = dict(delta=delta, tau=tau, states={}, sweep=[])
    profiles_1340 = None
    for step in steps:
        st = ck[step]
        v = OrientedPointCloudVarifold(positions=st["positions"],
                                       angles=st["angles"])
        recs, profiles = decompose_state(v, delta, tau)
        if step == PROFILE_STEP:
            profiles_1340 = profiles
        report["states"][step] = recs
        for role, r in recs.items():
            print(f"[{step}/{role}] A_poly {r['A_poly']:+.6f}  "
                  f"V_cur {r['V_cur']:+.6f}  T_tilt {r['T_tilt']:+.3e}"
                  f"  T_quad {r['T_quad']:+.3e}  resid "
                  f"{r['identity_resid']:.1e}  eps_n {r['eps_n']:.4f} "
                  f" min_t {r['min_transversality']:.4f}", flush=True)

    # (b) sigma-sweep with the transplanted 1340 tilt field
    R_d_star, A_ann, phi0 = organic_reference()
    for cell in CELLS[:3]:
        try:
            v_syn, _ = synthetic_cell_state(
                R_d_star, A_ann, cell["sigma"], DELTA_OVER_SIGMA,
                cell["n_out"], 0.0, phi0, k=2)
            kde = tuple(map(float, compute_recommended_params(
                snr.build_cloud_n(cell["n_out"]).positions)))
            res = resolve_loopwise_oriented_mass_source(
                v_syn.positions, v_syn.normals, *kde)
            labels = res.loop_labels_pre
            roles = loop_roles(labels, v_syn.positions, res.m_loop)
            v_t = transplant_tilt(v_syn, labels, roles,
                                  profiles_1340)
            recs, _ = decompose_state(v_t, *kde)
            row = dict(cell=cell["name"], sigma=cell["sigma"],
                       static=recs)
            # geometric row magnitude on the disk sheet
            res_t = resolve_loopwise_oriented_mass_source(
                v_t.positions, v_t.normals, *kde)
            orders = certified_cyclic_order(
                v_t.positions, res_t.loop_labels_pre, v_t.normals,
                res_t.m_loop, float(res_t.m_loop.median()))
            for lb, role in loop_roles(res_t.loop_labels_pre,
                                       v_t.positions,
                                       res_t.m_loop).items():
                if role == "disk":
                    g = polygon_gradient(v_t.positions,
                                         orders[lb].order)
                    sel = res_t.loop_labels_pre == lb
                    row["C_geom_disk_norm"] = float(
                        ((g[sel] * v_t.normals[sel]).sum(1) ** 2)
                        .sum() ** 0.5)
            # one-step production baseline (current rows/target)
            m0 = res.m_loop  # t=0-like target from the UNtilted state
            target0 = float(0.5 * (m0 * (v_syn.positions
                                         * v_syn.normals)
                                   .sum(-1)).sum())
            cfg = cell_config(cell, *kde, target0,
                              q_mode="self_renormalized")
            cfg.mass_estimator = "loopwise_oriented_kde"
            from src.torch.solver.mm_step import MMStepper
            stp = MMStepper(cfg)
            stp._grid_target_volume_initial = target0
            r_before = {ro: rec["A_poly"]
                        for ro, rec in recs.items()}
            out = stp.step(v_t)
            vv = (out.committed_varifold
                  if out.committed_varifold is not None
                  else out.varifold)
            recs_a, _ = decompose_state(vv, *kde)
            row["rush_dA_disk"] = (recs_a["disk"]["A_poly"]
                                   - r_before["disk"])
            row["rush_eps_n_disk"] = (recs_a["disk"]["eps_n"]
                                      - recs["disk"]["eps_n"])
            print(f"[sweep {cell['name']}] eps_n(disk) "
                  f"{recs['disk']['eps_n']:.4f}  V-A "
                  f"{recs['disk']['V_cur'] - recs['disk']['A_poly']:+.3e}"
                  f"  dA_disk/step {row['rush_dA_disk']:+.3e}",
                  flush=True)
        except Exception as exc:      # noqa: BLE001
            row = dict(cell=cell["name"], sigma=cell["sigma"],
                       error=f"{type(exc).__name__}: {exc}")
            print(f"[sweep {cell['name']}] ERROR {row['error']}",
                  flush=True)
        report["sweep"].append(row)

    out_p = Path("results/reports/phase3c0l_a0_tilt_audit.json")
    out_p.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
