"""0K-B: first-failure state ablation of the loopwise oriented mass.

Reviewer-amended protocol (2026-08-15):
- THREE static arms at the endgame-#1 failure states (1300, 1325):
    P_FF(Y|X) = sum m^or(Y)   q^WB,or(X)   (control)
    P_LF(Y|X) = sum m^loop(Y) q^WB,or(X)   (candidate mass only 0K)
    P_LL(Y|X) = sum m^loop(Y) q^WB,loop(X) (full 0K)
  DP_FF - DP_LF isolates the candidate-side cross-loop mass response
  (the decisive quantity); DP_LF - DP_LL isolates the source-template
  change.
- TWO-TERM decomposition (eq 4.1), h_tau(s) = chi_tau(s)/s:
    D m^or - D m^loop = (1/N)[ h'(th_or) D th_cross
                              + (h'(th_or) - h'(th_self)) D th_self ]
  valid verbatim only where chi_tau = 1 (audited and asserted on the
  disk/hole loops; h' = -s^-2 there). Dth_cross alone is NEVER used
  to claim closure.
- ADMISSIBLE pump directions: primary = the measured control one-step
  normal displacement field; secondary = the raw radial pump field
  projected onto the admissible space (param.Q, q-weighted).
- Volume telemetry in three systems (geom / current-or / current-loop);
  verdicts read the GEOMETRIC one.
- omega2 (smoothstep) secondary control: shows an O(eps^2) orientation
  kernel shrinks but does NOT annihilate the cross term (script-local
  only, never production).
- candidate perturbations act on POSITIONS with normals frozen (the
  0H-1/2 structural fact: the within-step pump force is carried by
  the mass response at frozen q; the angle channel is not the pump).

Directional derivatives: autograd (exact) for scalar DP; central
differences (h = 1e-6) for per-point D theta fields; the eq-4.1
residual is therefore FD-limited (~1e-9 rel), reported as such.

The 30-step continuation from 1325 is launched separately via the
driver (--resume 1325) and is PRE-REGISTERED as a hot-swap diagnostic:
(a) a first-step guard trip after the swap is NOT a production
failure; (b) mechanism attribution rests on the one-step derivatives
and the LF arm; (c) the production verdict comes from the t=0
endgame #2 run only.

Usage:
    uv run python scripts/experiments/loopwise_mass_ablation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import (  # noqa: E402
    DT_STEP,
    OUT,
    build_cloud,
    masses_for,
    shoelace_area,
)

from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.loopwise_mass import (  # noqa: E402
    resolve_loopwise_oriented_mass_source,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    KERNEL_CONSTANTS,
    chi_tau,
    compute_masses_oriented,
    compute_masses_oriented_loopwise,
    compute_recommended_params,
    oriented_density_cross_split,
    wendland_c2,
)
from src.torch.perimeter.coherence_perimeter import (  # noqa: E402
    compute_coherence_loopwise,
)
from src.torch.transport.bem_wasserstein import (  # noqa: E402
    compute_coherence,
)

SIGMA = 0.1
KER = "wendland_c2"
STATES = (1300, 1325)
FD_H = 1e-6
RES_DIR = Path("results/reports")


def qwb(v, m, labels):
    """q^WB = r_l q^self on the given raw masses -- same algebra as
    MMStepper._corrected_perimeter_q (0J-B pinned to machine
    precision); returns (q_wb, per-loop r)."""
    q_full = compute_coherence(v, m, SIGMA, KER)
    q_self, cross_max = compute_coherence_loopwise(
        v.positions, v.normals, m, SIGMA, KER, loop_labels=labels)
    q = torch.empty_like(q_full)
    rs = {}
    for lb in labels.unique():
        sel = labels == lb
        A = float((m[sel] * q_full[sel]).sum())
        B = float((m[sel] * q_self[sel]).sum())
        r = A / B
        rs[int(lb)] = r
        q[sel] = r * q_self[sel]
    return q, rs, cross_max


def classify_loops(labels, positions, m):
    """label -> role by mass-weighted mean radius: disk (smallest),
    outer (largest), hole (the remaining one)."""
    radii = {}
    r = positions.norm(dim=1)
    for lb in labels.unique():
        sel = labels == lb
        radii[int(lb)] = float((m[sel] * r[sel]).sum() / m[sel].sum())
    order = sorted(radii, key=radii.get)
    roles = {order[0]: "disk", order[-1]: "outer"}
    for lb in order[1:-1]:
        roles[lb] = "hole"
    return roles, radii


def dp_autograd(mass_fn, q_frozen, positions, xi):
    pos = positions.clone().requires_grad_(True)
    P = (mass_fn(pos) * q_frozen).sum()
    (g,) = torch.autograd.grad(P, pos)
    return float((g * xi).sum())


def fd_field(fn, positions, xi, h=FD_H):
    """Central-difference directional derivative of a vector field."""
    return (fn(positions + h * xi) - fn(positions - h * xi)) / (2 * h)


def omega2_masses(positions, normals, delta, tau):
    """Secondary control: endpoint-flat orientation kernel
    omega2 = 3t^2 - 2t^3, t = (1+s)/2 (O(eps^2) near antiparallel)."""
    N = positions.shape[0]
    C = KERNEL_CONSTANTS[KER]
    d = torch.cdist(positions, positions)
    ker = wendland_c2(d / delta)
    t = 0.5 * (1.0 + normals @ normals.T)
    om2 = 3 * t * t - 2 * t ** 3
    th = (ker * om2).sum(1) / (N * C * delta)
    chi = chi_tau(th, tau)
    m = torch.where(chi > 0, chi / th, torch.zeros_like(th)) / N
    return m, th


def production_stepper(est, target0):
    """The endgame-#1 production stack with the mass estimator as the
    single moving part (mirrors the driver's config assembly)."""
    from p1_production_comparison import make_cfg

    from src.torch.solver.mm_step import MMStepper
    from src.torch.transport.grid_wasserstein import GridMetricConfig
    from src.torch.transport.phase_grid import PhaseGridConfig

    cfg = make_cfg("C3", DELTA, TAU, 2)
    cfg.time_step = DT_STEP
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(512, 512), fill_epsilon=0.04,
        support_threshold=3e-3, projection_rel_tol=5e-2),
        compatibility_components="conservative_sweep")
    cfg.redistribute = True
    cfg.remove_dead_points = False
    cfg.enforce_support_gates = False
    cfg.grid_volume_target_mode = "initial_fixed"
    cfg.grid_phase_support_projection = False
    cfg.mass_estimator = est
    cfg.grid_bulk_rows_mode = "augment"
    cfg.perimeter_q_mode = "self_renormalized"
    st = MMStepper(cfg)
    # reviewer section 8: seed the volume target from the ORIGINAL
    # t=0 cloud (same estimator) -- a checkpoint-built target would
    # absorb past drift and void the comparison
    st._grid_target_volume_initial = target0
    return st


def one_step_record(est, v, target0, tag):
    st = production_stepper(est, target0)
    t0 = time.time()
    res = st.step(v)
    vv = (res.committed_varifold if res.committed_varifold is not None
          else res.varifold)
    gs = res.grid_setup_snapshot
    disp = res.displacements.detach()
    if disp.dim() == 1:
        xi_vec = disp[:, None] * v.normals
    else:
        xi_vec = disp
    m_after = masses_for(est, vv.positions, vv.normals, DELTA, TAU)
    r_after = vv.positions.norm(dim=1)
    wall = r_after < 0.6
    outward = (vv.positions * vv.normals).sum(1) > 0
    disk_m = wall & outward
    xn = (vv.positions * vv.normals).sum(-1)

    def wmean(mask):
        w = m_after[mask]
        return (float((w * r_after[mask]).sum() / w.sum())
                if w.numel() else None)

    rec = dict(
        tag=tag, est=est, wall_s=time.time() - t0,
        converged=bool(res.converged),
        perimeter=float(res.perimeter),
        R_disk_after=wmean(disk_m),
        R_hole_after=wmean(wall & ~outward),
        R_outer_after=wmean(~wall),
        vol_geom_disk_after=shoelace_area(vv.positions[disk_m]),
        vol_cur_disk_after=float(
            0.5 * (m_after[disk_m] * xn[disk_m]).sum()),
        vol_cur_ann_after=float(
            0.5 * (m_after[~disk_m] * xn[~disk_m]).sum()),
        max_s=float(disp.abs().max()),
        inactive_rhs=(gs.max_inactive_rhs_fraction
                      if gs is not None and hasattr(
                          gs, "max_inactive_rhs_fraction") else None),
        bulk_flux_residuals=(
            [float(x) for x in res.bulk_flux_residuals]
            if getattr(res, "bulk_flux_residuals", None) is not None
            else None),
        qmode_stats=st._qmode_stats,
    )
    snap = getattr(st, "_last_mass_loop_snapshot", None)
    if snap is not None:
        rec["mass_loop_stats"] = snap.loop_stats
    return rec, xi_vec, st


def admissible_projection(st, xi_scalar):
    """Project a scalar normal-displacement field onto the admissible
    space actually used by the step (s = [q *] Q y)."""
    Q = st.param.Q
    B = (st.param.coherence[:, None] * Q
         if st.config.displacement_q_suppress else Q)
    coef = torch.linalg.lstsq(B, xi_scalar[:, None]).solution
    return (B @ coef).flatten()


def static_arms(v, roles_by_label, xi_name, xi, res):
    """All static directional quantities along one direction xi."""
    labels = res.loop_labels_pre
    nor = v.normals
    N = v.n_points

    def m_or_fn(pos):
        return compute_masses_oriented(pos, nor, DELTA, TAU, KER)

    def m_loop_fn(pos):
        return compute_masses_oriented_loopwise(pos, nor, labels,
                                                DELTA, TAU, KER)

    q_ff, r_ff, _ = qwb(v, res.m_pre, labels)     # control template
    q_ll, r_ll, _ = qwb(v, res.m_loop, labels)    # full-0K template

    dp_ff = dp_autograd(m_or_fn, q_ff, v.positions, xi)
    dp_lf = dp_autograd(m_loop_fn, q_ff, v.positions, xi)
    dp_ll = dp_autograd(m_loop_fn, q_ll, v.positions, xi)

    # eq 4.1 decomposition (FD fields; chi audited below)
    def th_split(pos):
        return oriented_density_cross_split(pos, nor, DELTA, KER,
                                            loop_labels=labels)

    dth_self = fd_field(lambda p: th_split(p)[0], v.positions, xi)
    dth_cross = fd_field(lambda p: th_split(p)[1], v.positions, xi)
    th_self, th_cross = th_split(v.positions)
    th_or = th_self + th_cross
    chi_or = chi_tau(th_or, TAU)
    chi_self = chi_tau(th_self, TAU)
    n_chi = int(((chi_or < 1) | (chi_self < 1)).sum())
    hp_or = -1.0 / th_or ** 2      # h_tau' with chi = 1
    hp_self = -1.0 / th_self ** 2
    t1 = (hp_or * dth_cross) / N
    t2 = ((hp_or - hp_self) * dth_self) / N
    lhs = float((q_ff * (t1 + t2)).sum())
    gap_ff_lf = dp_ff - dp_lf
    resid = abs(lhs - gap_ff_lf) / max(abs(gap_ff_lf), 1e-30)

    # omega2 secondary control (candidate-mass arm, control template)
    def m_w2_fn(pos):
        return omega2_masses(pos, nor, DELTA, TAU)[0]

    dp_w2 = dp_autograd(m_w2_fn, q_ff, v.positions, xi)

    per_loop = {}
    for lb, role in roles_by_label.items():
        sel = labels == lb
        per_loop[role] = dict(
            T1=float((q_ff[sel] * t1[sel]).sum()),
            T2=float((q_ff[sel] * t2[sel]).sum()),
        )
    return dict(
        direction=xi_name,
        DP_FF=dp_ff, DP_LF=dp_lf, DP_LL=dp_ll, DP_w2=dp_w2,
        cross_response_FF_minus_LF=gap_ff_lf,
        template_change_LF_minus_LL=dp_lf - dp_ll,
        eq41_sum=lhs, eq41_rel_residual_fd=resid,
        eq41_terms_per_loop=per_loop,
        n_chi_below_one=n_chi,
        r_ff=r_ff, r_ll=r_ll,
    )


def main():
    global DELTA, TAU
    torch.set_default_dtype(torch.float64)
    ck = torch.load(OUT / "exact_merger_endgame1_states.pt",
                    weights_only=True)
    v0, _ = build_cloud()
    DELTA, TAU = map(float, compute_recommended_params(v0.positions))
    print(f"(delta, tau) from t=0 cloud: ({DELTA:.6f}, {TAU:.6f})")
    report = dict(delta=DELTA, tau=TAU, sigma=SIGMA, fd_h=FD_H,
                  states={})
    for step in STATES:
        st0 = ck[step]
        v = OrientedPointCloudVarifold(positions=st0["positions"],
                                       angles=st0["angles"])
        res = resolve_loopwise_oriented_mass_source(
            v.positions, v.normals, DELTA, TAU, KER)
        labels = res.loop_labels_pre
        roles, radii = classify_loops(labels, v.positions, res.m_loop)
        print(f"\n== state {step}: loops "
              f"{[(roles[l], round(radii[l], 4)) for l in roles]} ==")

        # per-sheet estimator comparison
        sheet = {}
        for lb, role in roles.items():
            sel = labels == lb
            ratio = res.m_pre[sel] / res.m_loop[sel]
            zc = [s for s in res.loop_stats if s["label"] == lb][0]
            sheet[role] = dict(
                m_or_over_m_loop_min=float(ratio.min()),
                m_or_over_m_loop_mean=float(ratio.mean()),
                zeta_cross=zc["zeta_cross"], M_l=zc["M_l"])
        rec = dict(loop_roles={int(k): v_ for k, v_ in roles.items()},
                   sheets=sheet,
                   cutoff_count_loop=res.cutoff_count_loop,
                   cutoff_count_oriented=res.cutoff_count_oriented)

        # volume targets from t=0 per estimator (reviewer section 8)
        t0v = {}
        for est in ("oriented_kde", "loopwise_oriented_kde"):
            m0 = masses_for(est, v0.positions, v0.normals, DELTA, TAU)
            t0v[est] = float(0.5 * (m0 * (v0.positions
                                          * v0.normals).sum(-1)).sum())
        rec["target0"] = t0v

        # geometric volumes BEFORE
        r = v.positions.norm(dim=1)
        wall = r < 0.6
        outward = (v.positions * v.normals).sum(1) > 0
        rec["vol_geom_disk_before"] = shoelace_area(
            v.positions[wall & outward])

        # one-step FF (control) -- also yields xi_ctrl and the
        # admissible space
        ff, xi_ctrl_vec, st_ff = one_step_record(
            "oriented_kde", v, t0v["oriented_kde"], f"FF@{step}")
        print(f"  one-step FF: R_disk {ff['R_disk_after']:.6f} "
              f"Vgeom {ff['vol_geom_disk_after']:.6f}")
        ll, _, _ = one_step_record(
            "loopwise_oriented_kde", v, t0v["loopwise_oriented_kde"],
            f"LL@{step}")
        print(f"  one-step LL: R_disk {ll['R_disk_after']:.6f} "
              f"Vgeom {ll['vol_geom_disk_after']:.6f}")
        rec["one_step"] = dict(FF=ff, LL=ll)

        # directions
        rhat = v.positions / r[:, None]
        xi_raw = torch.zeros_like(v.positions)
        for lb, role in roles.items():
            sel = labels == lb
            if role == "disk":
                xi_raw[sel] = rhat[sel]
            elif role == "hole":
                xi_raw[sel] = -rhat[sel]
        xi_raw = xi_raw / xi_raw.norm()
        xi_ctrl = xi_ctrl_vec / xi_ctrl_vec.norm()
        s_adm = admissible_projection(
            st_ff, (xi_raw * v.normals).sum(1))
        xi_adm = s_adm[:, None] * v.normals
        xi_adm = xi_adm / xi_adm.norm()

        rec["static"] = [
            static_arms(v, roles, "xi_ctrl_measured", xi_ctrl, res),
            static_arms(v, roles, "xi_pump_admissible", xi_adm, res),
            static_arms(v, roles, "xi_pump_raw_secondary", xi_raw,
                        res),
        ]
        for s in rec["static"]:
            print(f"  [{s['direction']}] DP_FF {s['DP_FF']:+.4e}  "
                  f"DP_LF {s['DP_LF']:+.4e}  DP_LL {s['DP_LL']:+.4e}  "
                  f"cross-resp {s['cross_response_FF_minus_LF']:+.4e} "
                  f"(eq4.1 resid {s['eq41_rel_residual_fd']:.1e}, "
                  f"chi<1: {s['n_chi_below_one']})")
        report["states"][step] = rec

    out = RES_DIR / "phase3c0k_ablation.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
