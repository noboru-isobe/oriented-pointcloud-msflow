"""0L-R1: redistribution integrability -- true 2^3 factorial + oracle.

Arms: density {global, loopwise} x tangent {stored(angles), PCA} x
retraction {off, on} = 8, plus E (redistribution OFF) and O
(loopwise + certified-polygon-oracle tangents + retraction, AUDIT
ONLY -- the polygon order never becomes a production update rule
outside this arm). From endgame-#2 checkpoints 1300 (healthy
control) and 1325 (rush onset), 15 production steps each; 50-step
windows for {A, D', E, O}. The MM stage keeps the CURRENT
rows/target (reviewer section 9: fix redistribution first; the
geometric-rows parity test comes after R2). q policy is FIXED across
all arms (stale full q -- the q_redist policy is the registered
pre-R2 blocker).

Per-step measurements (per bulk, disk highlighted):
- Delta A_redist and its EXACT split Sum_m L_A^m + Sum_m Q_A^m
  (L_A^m = DA(X^m)[dX^m] in the certified-polygon frame,
  checkpoint_every=1; the reviewer blocker: PCA leaves a FIRST-order
  leakage L_A -- substep refinement only removes Q_A);
- normalized leakage rate |Sum L_A| / Sum |dX|;
- eps_n, CV_l before/after, cap audit (clip count);
- retraction audit with the SHARED actual energy
  P#(X, n) = sum m~^loop(X,n) q^WB(X,n) (0K mass + 0J template
  refreshed on the state): Delta_ret = P#(X+, n_ret) - P#(X+, n_pre)
  and the one-step split defect R_split = P#(X^{n+1}) + W_n - P#(X^n)
  -- (R_split)+ is the primary energy criterion;
- r_l, M_l/H1, R_disk.

R1b: (eta, N_sub) vs (eta/2, 2 N_sub) at fixed T_red with early stop
disabled + cap audit (comparison valid only if cap-free or matched
effective displacement); static spatial ladder N, 2N, 4N on clean
fixtures (circle + flower) with h_N ~ N^{-1/2} measuring
E_proj(N) and the normalized leakage rate -- both must decrease.

Usage:
    uv run python scripts/experiments/redistribution_integrability.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import OUT, build_cloud  # noqa: E402
from loopwise_mass_ablation import qwb  # noqa: E402
from tilt_identity_audit import decompose_state, loop_roles  # noqa: E402
from tilt_row_ablation import (  # noqa: E402
    bulk_areas_and_orders,
    production_stepper,
)

from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.loopwise_mass import (  # noqa: E402
    resolve_loopwise_oriented_mass_source,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.solver.redistribution_rules import (  # noqa: E402
    _local_pca_tangents,
    redistribute_with_rule,
)
from src.torch.transport.loop_geometry import (  # noqa: E402
    certified_cyclic_order,
    polygon_area,
    polygon_gradient,
)

DT = torch.float64
STATES = (1300, 1325)
WINDOW = 15
LONG_WINDOW = 50
RHO_MAX, NEFF_MIN = 0.25, 2.5
ORI = math.cos(math.radians(30.0))

ARMS = {
    # name: (density_scope, tangent_source, retraction)
    "A_gso": ("global", "angles", False),      # current production
    "B_lso": ("loopwise", "angles", False),
    "C_gpo": ("global", "local_pca", False),
    "D_lpo": ("loopwise", "local_pca", False),
    "Ar_gsr": ("global", "angles", True),
    "Br_lsr": ("loopwise", "angles", True),
    "Cr_gpr": ("global", "local_pca", True),
    "Dr_lpr": ("loopwise", "local_pca", True),  # MAIN CANDIDATE
    "E_off": None,                              # no redistribution
    "O_oracle": ("loopwise", "ordered_loop_oracle", True),
}
LONG_ARMS = ("A_gso", "Dr_lpr", "E_off", "O_oracle")


def p_sharp(v):
    """Shared actual energy P# = sum m~^loop q^WB, everything
    refreshed on the state (0K mass, 0J template, labels)."""
    res = resolve_loopwise_oriented_mass_source(
        v.positions, v.normals, DELTA, TAU)
    q, rs, _ = qwb(v, res.m_loop, res.loop_labels_pre)
    return float((res.m_loop * q).sum()), rs, res


def redistribution_call(stp, vv, arm, n_iters=None, step_size=None,
                        tol=None, curvature_scope="union"):
    """One production-parity redistribution under the arm's knobs
    (labels re-certified via the shared resolver, as the solver
    does). Returns (v_new, info, labels, orders)."""
    density, tangent, retract = arm
    cfg = stp.config
    res_post = resolve_loopwise_oriented_mass_source(
        vv.positions.detach(), vv.normals.detach(), DELTA, TAU)
    labels = res_post.loop_labels_pre
    orders = certified_cyclic_order(
        vv.positions.detach(), labels, vv.normals.detach(),
        res_post.m_loop, float(res_post.m_loop.median()))
    loops = [lo.order for lo in orders.values()]
    pos, ang, info = redistribute_with_rule(
        vv.positions, vv.angles, cfg.redistribution_rule,
        delta=stp.mass_delta, kernel=cfg.mass_kernel,
        n_iters=(n_iters if n_iters is not None
                 else cfg.redistribute_n_iters),
        step_size=(step_size if step_size is not None
                   else cfg.redistribute_step_size),
        tol=(tol if tol is not None else cfg.redistribute_tol),
        max_disp_ratio=cfg.redistribute_max_disp_ratio,
        mass_tau=stp.mass_tau,
        delta_redist=stp.mass_delta * cfg.redistribute_delta_ratio,
        coherence=stp.fixed_coherence, sigma=stp._sigma,
        coherence_kernel=cfg.perimeter_kernel,
        coherence_backend=cfg.backend,
        loops=loops,
        loop_labels=labels,
        density_scope=density,
        curvature_scope=curvature_scope,
        tangent_source=tangent,
        normal_retraction=retract,
        pca_rho_max=RHO_MAX, pca_neff_min=NEFF_MIN,
        pca_delta_tan=stp.mass_delta,
        checkpoint_every=1)
    return (OrientedPointCloudVarifold(positions=pos, angles=ang),
            info, labels, orders)


def la_qa_split(v0, info, orders, roles):
    """Exact per-subiteration split Delta A = L_A + Q_A in the
    certified-polygon frame of the STEP-START state's orders (vertex
    correspondence preserved -- redistribution moves, never resamples
    indices)."""
    from src.torch.transport.loop_geometry import area_quadratic_term
    chain = [v0.positions] + [p for _, p, _ in info["checkpoints"]]
    L = {"disk": 0.0, "ann": 0.0}
    Q = {"disk": 0.0, "ann": 0.0}
    disp = 0.0
    for a, b in zip(chain[:-1], chain[1:]):
        dx = b - a
        disp += float(dx.norm(dim=1).sum())
        for lb, lo in orders.items():
            key = "disk" if roles[lb] == "disk" else "ann"
            g = polygon_gradient(a, lo.order)
            L[key] += float((g * dx).sum())
            Q[key] += float(area_quadratic_term(dx, lo.order))
    return L, Q, disp


def run_arm(name, arm, v_src, window):
    stp = production_stepper("current", "current", "augment",
                             TARGET0)
    v = v_src
    t0 = time.time()
    series = []
    p0, _, _ = p_sharp(v)
    for k in range(window):
        res = stp.step(v)
        vv = (res.committed_varifold
              if res.committed_varifold is not None else res.varifold)
        p_pre_red, _, res_pre = p_sharp(vv)
        rec = dict(k=k, converged=bool(res.converged))
        if arm is not None:
            roles = loop_roles(res_pre.loop_labels_pre, vv.positions,
                               res_pre.m_loop)
            a_before, orders_b, roles_b, _ = bulk_areas_and_orders(
                vv, DELTA, TAU)
            ang_pre = vv.angles
            v_new, info, labels, orders = redistribution_call(
                stp, vv, arm)
            L, Q, disp = la_qa_split(vv, info, orders, roles)
            a_after, _, _, _ = bulk_areas_and_orders(v_new, DELTA,
                                                     TAU)
            rec.update(
                dA_redist_disk=a_after["disk"] - a_before["disk"],
                L_disk=L["disk"], Q_disk=Q["disk"],
                leak_rate=(abs(L["disk"]) / (disp + 1e-30)),
                disp=disp,
                clip_count=sum(1 for t in info["trace"]
                               if t["clip_active"]),
                cv_first=info["cv_history"][0],
                cv_last=info["cv_history"][-1],
                cv_final_by_loop=info["cv_final_by_loop"],
            )
            if arm[2]:      # retraction arm: Delta_ret with P#
                p_ret, _, _ = p_sharp(v_new)
                v_nore = OrientedPointCloudVarifold(
                    positions=v_new.positions, angles=ang_pre)
                p_nore, _, _ = p_sharp(v_nore)
                rec["delta_ret"] = p_ret - p_nore
                rec["retraction"] = info["retraction"]
            v = v_new
        else:
            v = vv
        p_now, rs, res_now = p_sharp(v)
        recs, _ = decompose_state(v, DELTA, TAU)
        rec.update(
            P_sharp=p_now,
            R_split=p_now + float(res.wasserstein) - p0,
            eps_n_disk=recs["disk"]["eps_n"],
            A_disk=recs["disk"]["A_poly"],
            r_l=rs,
        )
        p0 = p_now
        series.append(rec)
    out = dict(
        arm=name, window=window, wall_s=time.time() - t0,
        A_disk_final=series[-1]["A_disk"],
        dA_disk_total=series[-1]["A_disk"] - series[0]["A_disk"],
        eps_n_final=series[-1]["eps_n_disk"],
        sum_L_disk=sum(r.get("L_disk", 0.0) for r in series),
        sum_Q_disk=sum(r.get("Q_disk", 0.0) for r in series),
        max_leak_rate=max((r.get("leak_rate", 0.0) for r in series),
                          default=0.0),
        sum_dA_redist=sum(r.get("dA_redist_disk", 0.0)
                          for r in series),
        clip_total=sum(r.get("clip_count", 0) for r in series),
        max_R_split_plus=max(max(r["R_split"], 0.0) for r in series),
        sum_delta_ret_plus=sum(max(r.get("delta_ret", 0.0), 0.0)
                               for r in series),
        series=series,
    )
    print(f"  [{name}] dA_disk {out['dA_disk_total']:+.5f} "
          f"(redist {out['sum_dA_redist']:+.2e} = L "
          f"{out['sum_L_disk']:+.2e} + Q {out['sum_Q_disk']:+.2e}) "
          f"eps_n {out['eps_n_final']:.4f}  leak "
          f"{out['max_leak_rate']:.2e}  R_split+ "
          f"{out['max_R_split_plus']:.2e}  clips {out['clip_total']}",
          flush=True)
    return out


def substep_refinement(v_src):
    """R1b: fixed T_red, early stop disabled, cap audit."""
    stp = production_stepper("current", "current", "augment",
                             TARGET0)
    res = stp.step(v_src)
    vv = (res.committed_varifold
          if res.committed_varifold is not None else res.varifold)
    arm = ARMS["Dr_lpr"]
    out = {}
    for tag, (eta_f, n_f) in (("base", (1.0, 1)),
                              ("half", (0.5, 2))):
        v_new, info, labels, orders = redistribution_call(
            stp, vv, arm,
            n_iters=stp.config.redistribute_n_iters * n_f,
            step_size=stp.config.redistribute_step_size * eta_f,
            tol=0.0)
        res_pre = resolve_loopwise_oriented_mass_source(
            vv.positions.detach(), vv.normals.detach(), DELTA, TAU)
        roles = loop_roles(res_pre.loop_labels_pre, vv.positions,
                           res_pre.m_loop)
        L, Q, disp = la_qa_split(vv, info, orders, roles)
        ang_jump = (info["retraction"]["max_angle_rad"]
                    if info["retraction"] else None)
        out[tag] = dict(L_disk=L["disk"], Q_disk=Q["disk"],
                        disp=disp,
                        clip_count=sum(1 for t in info["trace"]
                                       if t["clip_active"]),
                        retraction_max_angle=ang_jump)
        print(f"  [R1b {tag}] Q {Q['disk']:+.3e}  L {L['disk']:+.3e}"
              f"  disp {disp:.4f}  clips {out[tag]['clip_count']}  "
              f"ret_angle {ang_jump}", flush=True)
    return out


def spatial_ladder():
    """R1b static ladder: E_proj(N) and normalized leakage rate must
    both decrease with h_N ~ N^{-1/2} (clean fixtures; 1325 is
    validation only)."""
    from src.torch.shapes.generator import (
        generate_oriented_circle,
        generate_oriented_flower,
    )
    rows = []
    for name, maker, n0 in (
            ("circle", lambda n: generate_oriented_circle(
                n, 1.0, (0.0, 0.0), "cpu", DT), 128),
            ("flower", lambda n: generate_oriented_flower(
                n, device="cpu", dtype=DT), 160)):
        for scale in (1, 2, 4):
            n = n0 * scale
            v = maker(n)
            lbl = torch.zeros(n, dtype=torch.long)
            m = torch.full((n,), 2 * math.pi / n, dtype=DT)
            spacing = float((v.positions.roll(-1, 0)
                             - v.positions).norm(dim=1).median())
            h = spacing * 3.0 * math.sqrt(scale)   # h ~ N^{-1/2}
            orders = certified_cyclic_order(v.positions, lbl,
                                            v.normals, m, spacing)
            lo = orders[0]
            t_pca = _local_pca_tangents(v.positions, v.angles, h,
                                        "wendland_c2",
                                        loop_labels=lbl)
            p = v.positions[lo.order]
            t_or = p.roll(-1, 0) - p.roll(1, 0)
            t_or = t_or / t_or.norm(dim=1, keepdim=True)
            cosang = (t_pca[lo.order] * t_or).sum(1).abs().clamp(0, 1)
            e_proj = float(torch.acos(cosang).max())
            # one unit-q subiteration on a jittered resampling
            g = torch.Generator().manual_seed(5)
            jit = v.angles + 0.0  # positions jitter via angle offset
            pos_j = v.positions + (0.15 * spacing) * torch.randn(
                v.positions.shape, generator=g, dtype=DT)
            p1, _, _ = redistribute_with_rule(
                pos_j, v.angles, "legacy_hybrid", delta=h,
                n_iters=1, tol=0.0, coherence=None,
                q_override="unit", loop_labels=lbl,
                density_scope="loopwise",
                tangent_source="local_pca", pca_delta_tan=h,
                pca_rho_max=None, pca_neff_min=None)
            dx = p1 - pos_j
            gg = polygon_gradient(pos_j, lo.order)
            L = abs(float((gg * dx).sum()))
            rate = L / (float(dx.norm(dim=1).sum()) + 1e-30)
            rows.append(dict(fixture=name, N=n, h=h, E_proj=e_proj,
                             leak_rate=rate))
            print(f"  [ladder {name} N={n}] E_proj {e_proj:.5f}  "
                  f"leak_rate {rate:.2e}", flush=True)
    return rows


def main():
    global DELTA, TAU, TARGET0
    torch.set_default_dtype(DT)
    ck = torch.load(OUT / "exact_merger_endgame2_states.pt",
                    weights_only=True)
    v0, _ = build_cloud()
    DELTA, TAU = map(float, compute_recommended_params(v0.positions))
    import tilt_row_ablation as tra
    tra.DELTA, tra.TAU = DELTA, TAU
    m0 = resolve_loopwise_oriented_mass_source(
        v0.positions, v0.normals, DELTA, TAU).m_loop
    TARGET0 = float(0.5 * (m0 * (v0.positions
                                 * v0.normals).sum(-1)).sum())
    report = dict(delta=DELTA, tau=TAU, states={}, r1b={},
                  ladder=None)
    for step in STATES:
        st = ck[step]
        v = OrientedPointCloudVarifold(positions=st["positions"],
                                       angles=st["angles"])
        print(f"== state {step} (15-step factorial) ==", flush=True)
        arms = {}
        for name, arm in ARMS.items():
            try:
                arms[name] = run_arm(name, arm, v, WINDOW)
            except Exception as exc:      # noqa: BLE001
                arms[name] = dict(arm=name,
                                  error=f"{type(exc).__name__}: "
                                        f"{exc}")
                print(f"  [{name}] ERROR {arms[name]['error']}",
                      flush=True)
        report["states"][step] = arms

    print("== 50-step windows (1325) ==", flush=True)
    st = ck[1325]
    v = OrientedPointCloudVarifold(positions=st["positions"],
                                   angles=st["angles"])
    long = {}
    for name in LONG_ARMS:
        try:
            long[name] = run_arm(name, ARMS[name], v, LONG_WINDOW)
        except Exception as exc:      # noqa: BLE001
            long[name] = dict(arm=name,
                              error=f"{type(exc).__name__}: {exc}")
            print(f"  [{name}] ERROR {long[name]['error']}",
                  flush=True)
    report["long_1325"] = long

    print("== R1b substep refinement (1325) ==", flush=True)
    report["r1b"] = substep_refinement(v)
    print("== R1b spatial ladder ==", flush=True)
    report["ladder"] = spatial_ladder()

    out_p = Path("results/reports/phase3c0l_r1.json")
    out_p.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
