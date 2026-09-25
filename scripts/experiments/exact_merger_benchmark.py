"""Exact merger benchmark: concentric disk + annulus (Stage-5
alternative, docs/annulus_benchmark_plan.md 2026-08-02 note).

    E = {r < R1}  u  {R2 < r < R3},   R1 < R2 < R3.

One-phase MS potentials are per-component Neumann problems, so the
disk is EXACTLY stationary and the annulus obeys the Run-A radial ODE
dR/dt = -a/R with (R+, R-) = (R3, R2) -- component interaction is
exactly zero. The hole shrinks monotonically and CONTACTS the disk at
finite T_c solving R2(T_c) = R1 (same quadrature as Run A). At
contact the union is { r < R3(T_c) } by area conservation
(pi R1^2 + pi (R3^2 - R2^2) = pi R3^2 when R2 = R1), i.e. the merged
state is a disk = EXACT MS equilibrium. Before (exact ODE), when
(exact T_c) and after (exactly stationary) are all analytic: this is
the only setting where "which part of the merger is discretization
error" decomposes quantitatively. The contact circle carries two
antiparallel sheets (disk-outward + hole-inward) -- the hidden
boundary annihilation itself -- so the whole vertical machinery
(gate, coupled da, GC) is exercised against an exact answer.

Known non-genericity: contact is simultaneous around the circle;
discrete noise breaks the symmetry (expected; the D1b-P perturbation
R(theta) = R (1 + eps cos k theta) is the deliberate version, not
implemented here).

Usage:
    uv run python scripts/experiments/exact_merger_benchmark.py \
        --vertical --lam 1.0 --advect --freeze-dead
    uv run python scripts/experiments/exact_merger_benchmark.py \
        --postprocess results/exact_merger/exact_merger_v2.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import compute_recommended_params
from src.torch.shapes.generator import (
    generate_oriented_annulus,
    generate_oriented_circle,
)

DT = torch.float64
DT_STEP = 1e-5
R1, R2, R3 = 0.35, 0.55, 1.0
N_OUT = 256                      # outer-ring count; others by spacing
OUT = Path("results/exact_merger")


def exact_reference():
    """Dense annulus trajectory (R3(t), R2(t)) with terminal contact
    event R2(t) = R1; the disk is exactly stationary. Returns
    (solution, T_c, R_eq = R3(T_c))."""
    from scipy.integrate import solve_ivp

    from annulus_run_a import ode_rhs

    def rhs(t, y):
        return list(ode_rhs(y[0], y[1]))

    def contact(t, y):
        return y[1] - R1
    contact.terminal = True
    contact.direction = -1
    sol = solve_ivp(rhs, [0.0, 1.0], [R3, R2], rtol=1e-12,
                    atol=1e-14, dense_output=True, events=contact)
    t_c = float(sol.t_events[0][0])
    r_eq = float(sol.sol(t_c)[0])
    return sol, t_c, r_eq


def build_cloud():
    spacing = 2.0 * math.pi * R3 / N_OUT
    n_disk = max(3, round(2.0 * math.pi * R1 / spacing))
    disk = generate_oriented_circle(n_disk, R1, (0.0, 0.0), "cpu", DT)
    ann = generate_oriented_annulus(N_OUT, R3, R2, (0.0, 0.0),
                                    "cpu", DT)
    v = OrientedPointCloudVarifold(
        positions=torch.cat([disk.positions, ann.positions]),
        angles=torch.cat([disk.angles, ann.angles]))
    return v, n_disk


def radius_clusters(positions, gap=0.06):
    """Label-free radial clustering (measurement only): sorted radii
    split at gaps > `gap`. Returns list of (mean_r, count)."""
    r = positions.norm(dim=1).sort().values
    if r.numel() == 0:
        return []
    splits = (r[1:] - r[:-1] > gap).nonzero().flatten() + 1
    out, start = [], 0
    for s in [int(x) for x in splits] + [r.numel()]:
        seg = r[start:s]
        out.append((float(seg.mean()), int(seg.numel())))
        start = s
    return out


def masses_for(est, positions, normals, delta, tau):
    """Same estimator dispatch as MMStepper._masses_for (IIIc-0D).
    0K: loopwise goes through the SHARED two-stage-certificated
    resolver -- the simulation and every diagnostic read the same
    masses by construction (no independent dispatch)."""
    from src.torch.oriented_varifold.mass import (
        compute_masses,
        compute_masses_oriented,
    )
    if est == "loopwise_oriented_kde":
        return loopwise_resolution(positions, normals, delta,
                                   tau).m_loop
    if est == "oriented_kde":
        return compute_masses_oriented(positions, normals, delta, tau,
                                       "wendland_c2")
    return compute_masses(positions, delta, tau, "wendland_c2")


def loopwise_resolution(positions, normals, delta, tau):
    """0K source resolution (masses + labels + theta split + per-loop
    stats) via the shared resolver used by MMStepper._setup_step."""
    from src.torch.oriented_varifold.loopwise_mass import (
        resolve_loopwise_oriented_mass_source,
    )
    return resolve_loopwise_oriented_mass_source(
        positions, normals, delta, tau, "wendland_c2")


def _sorted_by_angle(pts):
    c = pts.mean(0)
    ang = torch.atan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[ang.argsort()]


def shoelace_area(pts):
    """Estimator-INDEPENDENT polygon area of one (star-shaped) sheet:
    angular sort about the centroid, then the shoelace formula
    (reviewer 0K section 7: geometric volume for success gates)."""
    if pts.shape[0] < 3:
        return None
    p = _sorted_by_angle(pts)
    x, y = p[:, 0], p[:, 1]
    x2, y2 = torch.roll(x, -1, 0), torch.roll(y, -1, 0)
    return float(0.5 * torch.abs((x * y2 - x2 * y).sum()))


def polygon_perimeter(pts):
    """Estimator-independent H^1 length of one sheet (angular sort +
    closed polyline) -- denominator of the M_l / H^1_geom telemetry."""
    if pts.shape[0] < 2:
        return None
    p = _sorted_by_angle(pts)
    return float((torch.roll(p, -1, 0) - p).norm(dim=1).sum())


def loop_m4(vv, masses, labels):
    """0M-0 observable: complex m=4 shape coefficient per loop,
        z4 = sum_i m_i (|x_i-c|-rbar) e^{-4 i theta_i} / sum_i m_i
    with mass-weighted centroid c and mean radius rbar. A4 = |z4|,
    phi4 = -(1/4) arg z4 mod pi/2. The complex components are
    recorded so phase tracking never has to unwrap the mod-90
    angle (reviewer 0M-0: 89 deg -> 1 deg is a wrap, not a move)."""
    out = {}
    for lb in labels.unique():
        sel = labels == lb
        m = masses[sel]
        M = float(m.sum())
        if int(sel.sum()) < 8 or M <= 0:
            continue
        x = vv.positions[sel]
        c = (m.unsqueeze(1) * x).sum(0) / M
        d = x - c
        r = d.norm(dim=1)
        rbar = float((m * r).sum() / M)
        th = torch.atan2(d[:, 1], d[:, 0])
        z_re = float((m * (r - rbar) * torch.cos(4 * th)).sum() / M)
        z_im = float(-(m * (r - rbar) * torch.sin(4 * th)).sum() / M)
        a4 = math.hypot(z_re, z_im)
        phi4 = (-math.atan2(z_im, z_re) / 4.0) % (math.pi / 2.0)
        nsel = vv.normals[sel]
        outward = float(((x * nsel).sum(1) > 0).float().mean())
        rmean = float(x.norm(dim=1).mean())
        role = ("disk" if rmean < 0.6 and outward > 0.5
                else "hole" if rmean < 0.6 else "outer")
        out[role] = dict(A4=a4, phi4_deg=math.degrees(phi4),
                         re=z_re, im=z_im, rbar=rbar,
                         A4_over_R=a4 / max(rbar, 1e-30))
    return out


def contact_certificate_record(vv, ctx, delta, tau, sigma=0.1):
    """0L-B1 telemetry: evaluate the contact certificate for every
    State-II candidate pair (same grid component, different raw bulk)
    on the committed state. Returns a JSON-able list (possibly empty)
    or None when the bookkeeping context is unavailable."""
    if ctx is None:
        return None
    from src.torch.transport.contact_certificate import (
        ContactCertificateError,
        evaluate_all_candidates,
    )
    try:
        lw = loopwise_resolution(vv.positions, vv.normals, delta, tau)
        labels = lw.loop_labels_pre
        # the source-state maps are keyed by source loop label; the
        # committed state's certified labels are re-resolved -- match
        # by majority particle overlap (labels are particle-aligned
        # within a step; the resolver is deterministic)
        certs = evaluate_all_candidates(
            vv.positions, vv.normals, lw.m_loop, labels,
            ctx["loop_to_grid_component"], ctx["loop_to_raw_bulk"],
            sigma)
        return [c.as_dict() for c in certs]
    except ContactCertificateError as ex:
        return {"error": str(ex)[:160]}
    except Exception as ex:                # noqa: BLE001
        return {"error": f"{type(ex).__name__}: {str(ex)[:120]}"}


def wall_diagnostics(vv, est, delta, tau, probe_curvature=False):
    """IIIc-0D per-step record: q_wall, alpha_wall/alpha_total,
    cutoff audit, and (optionally) the perimeter-only directional
    curvature along a common e_x translation of the wall marks.
    Diagnostic sigma fixed at 0.1 (IIIb-4 rule); wall = r < 0.6."""
    from src.torch.oriented_varifold.mass import (
        chi_tau,
        compute_kde_density,
        compute_kde_density_oriented,
    )
    from src.torch.transport.bem_wasserstein import compute_coherence

    wall = vv.positions.norm(dim=1) < 0.6
    if est == "loopwise_oriented_kde":
        _res = loopwise_resolution(vv.positions, vv.normals, delta, tau)
        m, th = _res.m_loop, _res.theta_self
    elif est == "oriented_kde":
        m = masses_for(est, vv.positions, vv.normals, delta, tau)
        th = compute_kde_density_oriented(vv.positions, vv.normals,
                                          delta, "wendland_c2")
    else:
        m = masses_for(est, vv.positions, vv.normals, delta, tau)
        th = compute_kde_density(vv.positions, delta, "wendland_c2")
    q = compute_coherence(vv, m, 0.1, "wendland_c2")
    alpha = m * q
    out = {
        "p_full_committed": float((m * q).sum()),
        "n_wall": int(wall.sum()),
        "q_wall_mean": (float(q[wall].mean()) if wall.any() else None),
        "alpha_wall_frac": (float(alpha[wall].sum() / alpha.sum())
                            if wall.any() else None),
        "n_cutoff": int((chi_tau(th, tau) < 1).sum()),
        "min_th_over_tau": float((th / tau).min()),
    }
    if probe_curvature and wall.any():
        # P-only curvature (NOT the full objective -- the metric term
        # is what gauges null-DOF; this matches the IIIa-3 probe)
        t = 1e-3
        xi = torch.zeros_like(vv.positions)
        xi[wall, 0] = 1.0

        def phat(dx):
            vp = OrientedPointCloudVarifold(
                positions=vv.positions + dx, angles=vv.angles)
            return float((m * compute_coherence(
                vp, m, 0.1, "wendland_c2")).sum())

        p0 = phat(0.0 * xi)
        curv = ((phat(t * xi) + phat(-t * xi) - 2.0 * p0)
                / (t * t * float((xi * xi).sum())))
        out["wall_ct_curvature_P"] = curv
    return out


def extension_diagnostics(vv, est, delta, tau, state, h):
    """IIIc-0D extension record (reviewer spec): mass-weighted sheet
    radii/stds, geometric gap, eta_wall = A_wall/M_wall, smoothed wall
    current L2 norm, the three volumes, radial-ODE residuals, and disk
    drift. Wall = r < 0.6, sheets split by outward/inward normal.
    Diagnostic sigma fixed at 0.1."""
    from annulus_run_a import ode_rhs

    from src.torch.transport.bem_wasserstein import compute_coherence

    r = vv.positions.norm(dim=1)
    wall = r < 0.6
    mass_res = None
    if est == "loopwise_oriented_kde":
        mass_res = loopwise_resolution(vv.positions, vv.normals,
                                       delta, tau)
        m = mass_res.m_loop
    else:
        m = masses_for(est, vv.positions, vv.normals, delta, tau)
    q = compute_coherence(vv, m, 0.1, "wendland_c2")
    outward = (vv.positions * vv.normals).sum(1) > 0

    def wmean(mask):
        w = m[mask]
        if w.numel() == 0:
            return None, None
        mu = float((w * r[mask]).sum() / w.sum())
        var = float((w * (r[mask] - mu) ** 2).sum() / w.sum())
        return mu, math.sqrt(max(var, 0.0))

    R_d, s_d = wmean(wall & outward)
    R_h, s_h = wmean(wall & ~outward)
    R_o, s_o = wmean(~wall)
    gap = (R_h - R_d) if (R_h is not None and R_d is not None) else None
    M_wall = float(m[wall].sum())
    A_wall = float((m * q)[wall].sum())
    # ||eta_sigma * T_wall||_L2 via the SHARED Gaussian self-convolution
    # Gram (0L-B1 contact_certificate helper; bitwise the former inline
    # formula)
    from src.torch.transport.contact_certificate import (
        mollified_current_energy,
    )
    sig = 0.1
    widx = wall.nonzero().flatten()
    if widx.numel() > 0:
        t_wall_l2 = math.sqrt(mollified_current_energy(
            vv.positions, vv.normals, m, sig, widx))
    else:
        t_wall_l2 = 0.0
    out = dict(
        R_disk=R_d, R_hole=R_h, R_outer=R_o,
        std_disk=s_d, std_hole=s_h, gap=gap,
        M_wall=M_wall, A_wall=A_wall,
        eta_wall=(A_wall / M_wall if M_wall > 0 else None),
        t_wall_current_l2=t_wall_l2,
        vol_current=float(0.5 * (m * (vv.positions
                                      * vv.normals).sum(-1)).sum()),
        vol_radial=(math.pi * (R_o ** 2 - R_h ** 2 + R_d ** 2)
                    if None not in (R_o, R_h, R_d) else None),
    )
    out["h_wall"] = (float(m[wall].median()) if wall.any() else None)
    # corrected volume gate (reviewer 0F verdict): per-bulk current
    # volumes V_b = 0.5 sum m x.n (disk bulk = outward wall sheet;
    # annulus bulk = inward wall sheet + outer ring), equivalent-area
    # radius, and shape telemetry (radial variance about the centroid
    # vs R_A, circularity 4 pi V / P^2, centroid)
    xn = (vv.positions * vv.normals).sum(-1)
    disk_m = wall & outward
    ann_m = ~disk_m
    v_disk = float(0.5 * (m[disk_m] * xn[disk_m]).sum())
    out["vol_cur_disk"] = v_disk
    out["vol_cur_ann"] = float(0.5 * (m[ann_m] * xn[ann_m]).sum())
    # 0K (reviewer section 7): estimator-INDEPENDENT geometric areas
    # -- the success gates read these, never a current-volume built
    # from the estimator under test
    hole_m = wall & ~outward
    a_disk = shoelace_area(vv.positions[disk_m])
    a_hole = shoelace_area(vv.positions[hole_m])
    a_out = shoelace_area(vv.positions[~wall])
    out["vol_geom_disk"] = a_disk
    out["vol_geom_ann"] = (a_out - a_hole
                           if None not in (a_out, a_hole) else a_out)
    if mass_res is not None:
        # shadow telemetry: the OLD estimator's current volumes keep
        # running so the cross-contamination onset stays comparable
        from src.torch.oriented_varifold.mass import (
            compute_masses_oriented,
        )
        m_or = compute_masses_oriented(vv.positions, vv.normals,
                                       delta, tau, "wendland_c2")
        out["vol_current_or"] = float(0.5 * (m_or * xn).sum())
        out["vol_cur_disk_or"] = float(
            0.5 * (m_or[disk_m] * xn[disk_m]).sum())
        out["vol_cur_ann_or"] = float(
            0.5 * (m_or[ann_m] * xn[ann_m]).sum())
        loops = mass_res.loop_labels_pre
        lstats = []
        for st_l in mass_res.loop_stats:
            sel = loops == st_l["label"]
            per = polygon_perimeter(vv.positions[sel])
            lstats.append(dict(
                st_l, H1_geom=per,
                M_over_H1=(st_l["M_l"] / per if per else None)))
        out["mass_loop_stats"] = lstats
        out["cutoff_count_loop"] = mass_res.cutoff_count_loop
        out["cutoff_count_oriented"] = mass_res.cutoff_count_oriented
    if v_disk > 0 and disk_m.any():
        w = m[disk_m]
        c_d = (w[:, None] * vv.positions[disk_m]).sum(0) / w.sum()
        r_c = (vv.positions[disk_m] - c_d[None, :]).norm(dim=1)
        rbar = float((w * r_c).sum() / w.sum())
        p_d = float(w.sum())
        out["R_A_disk"] = math.sqrt(v_disk / math.pi)
        # PURE shape variance: anchored at the mean radius about the
        # centroid, NOT at R_A -- the deep-layer estimator inflates
        # the sheet arclength (~15% at 1300), pushing R_A away from
        # the geometric mean radius; anchoring at R_A would couple
        # that estimation channel into the shape telemetry
        out["S_disk"] = float((w * (r_c - rbar) ** 2).sum() / w.sum())
        out["circularity_disk"] = 4 * math.pi * v_disk / p_d ** 2
        out["centroid_disk"] = [float(c_d[0]), float(c_d[1])]
    prev = state.get("prev_radii")
    if prev is not None and None not in (R_o, R_h, R_d):
        rhs_o, rhs_h = ode_rhs(prev["R_o"], prev["R_h"])
        out["ode_residual_outer"] = (R_o - prev["R_o"]) / h - rhs_o
        out["ode_residual_hole"] = (R_h - prev["R_h"]) / h - rhs_h
        out["disk_drift"] = R_d - state["R_disk_0"]
        out["rdot_disk"] = (R_d - prev["R_d"]) / h
    if R_d is not None:
        state.setdefault("R_disk_0", R_d)
        state["prev_radii"] = dict(R_o=R_o, R_h=R_h, R_d=R_d)
    return out


def incidence_diagnostics(vv, est, delta, tau, s=None, dtheta=None,
                          residuals=None):
    """0F: raw vs visible winding incidence, compared as CO-MEMBERSHIP
    partitions (label renaming must not trigger a stop). When the
    realized step (s, dtheta) and the stepper's bulk-row residuals are
    given, also computes the reviewer's per-bulk relative flux
    eps_flux_b = |C_b| / (sum_{i in b} m|s| + (m^2/3)|dtheta|)
    (K=3 symmetric endpoints: mean |r_ik| = m/3). Relies on the loop/
    bulk labelling being deterministic and identical to the stepper's
    (same functions, same state, same ell)."""
    from src.torch.transport.bem_wasserstein import compute_coherence
    from src.torch.transport.incidence import (
        oriented_graph_components,
        partition_relation,
        winding_bulk_labels,
    )

    m = masses_for(est, vv.positions, vv.normals, delta, tau)
    q = compute_coherence(vv, m, 0.1, "wendland_c2")
    spacing = float(m.median())
    loops = oriented_graph_components(vv.positions, vv.normals,
                                      4.0 * spacing)
    bulk_raw, margin_raw = winding_bulk_labels(
        vv.positions, vv.normals, m, loops, spacing)
    bulk_vis, margin_vis = winding_bulk_labels(
        vv.positions, vv.normals, m * q, loops, spacing)
    out = dict(
        n_loops=int(loops.max()) + 1,
        incidence_margin_raw=margin_raw,
        incidence_margin_vis=margin_vis,
        n_bulk_raw=(int(bulk_raw.max()) + 1
                    if bulk_raw is not None else None),
        n_bulk_vis=(int(bulk_vis.max()) + 1
                    if bulk_vis is not None else None),
    )
    if bulk_raw is None or bulk_vis is None:
        out["raw_vis_agree"] = None      # certificate failure upstream
    else:
        out["raw_vis_agree"] = (partition_relation(
            bulk_raw[loops], bulk_vis[loops]) == "equal")
    if (residuals is not None and bulk_raw is not None
            and s is not None and dtheta is not None):
        bulk_pp = bulk_raw[loops]
        eps = []
        for b, g in enumerate(residuals):
            sel = bulk_pp == b
            den = float((m[sel] * s[sel].abs()).sum()
                        + (m[sel] ** 2 / 3.0
                           * dtheta[sel].abs()).sum())
            eps.append(abs(g) / (den + 1e-30))
        out["eps_flux"] = eps
        out["eps_flux_max"] = max(eps)
    return out


# pre-registered 2026-08-16 (clean-fixture envelope + bandwidth
# spread + eps_num; flower curvature dominates both): fail-closed
# thresholds of the PCA shadow certificate. Never re-fit from runs.
E2_STAR = 0.0641
EINF_STAR = 0.2274


def e_int_of(vv, delta, tau):
    """PCA shadow certificate quantities: mass-weighted RMS and max
    angle between the stored normals and the loopwise-PCA normals
    (weights = RAW loopwise mass -- q must not hide a vanishing
    sheet's normal error)."""
    from src.torch.solver.redistribution_rules import (
        _local_pca_tangents,
    )
    res = loopwise_resolution(vv.positions, vv.normals, delta, tau)
    t = _local_pca_tangents(vv.positions, vv.angles, delta,
                            "wendland_c2",
                            loop_labels=res.loop_labels_pre)
    nu = torch.stack([t[:, 1], -t[:, 0]], 1)
    ang = torch.acos(((vv.normals * nu).sum(1)).abs().clamp(0, 1))
    m = res.m_loop
    e2 = float(((m * ang ** 2).sum() / m.sum()) ** 0.5)
    return e2, float(ang.max())


def _bulk_rows_stop(rec, state, args, step):
    """0F pre-registered stop conditions, corrected-gate version
    (reviewer verdict): the pump gates are CONSERVATION quantities --
    per-bulk relative flux eps_flux and the finite-step component
    volume drift against a quadratic envelope -- never mean-radius
    rates (not a volume coordinate away from exact circularity).
    Shape telemetry (radial variance, centroid) stops on growth, not
    on volume-neutral relaxation. The ODE tempo factor is accepted at
    fixed sigma (attribution deferred to the refinement sweep)."""
    def consec(key, cond, k):
        state[key] = state.get(key, 0) + 1 if cond else 0
        return state[key] >= k

    # PCA shadow certificate (fail-closed HARD guard, both bounds
    # must hold; never converted to an event under --endgame)
    e2, einf = rec.get("e2"), rec.get("einf")
    if e2 is not None and (e2 > E2_STAR or einf > EINF_STAR):
        return (f"PCA shadow certificate: e2 {e2:.4f} / einf "
                f"{einf:.4f} exceed pre-registered ({E2_STAR}, "
                f"{EINF_STAR})")
    gap, hw = rec.get("gap"), rec.get("h_wall")
    if gap is not None and hw is not None and gap <= 2.0 * hw:
        if not args.endgame:
            return (f"contact layer reached (gap {gap:.4f} <= "
                    f"2 h_wall {2 * hw:.4f})")
        if state.get("contact_layer_at") is None:
            state["contact_layer_at"] = step
            print(f"== EVENT contact layer entered (gap <= 2 h_wall) "
                  f"at step {step} ==", flush=True)
    # B_raw vs B_vis: stop only when BOTH certify and the partitions
    # disagree. A decertified VISIBLE winding (margin_vis 0.21 -> 0.44
    # measured across the layer while raw stays < 0.06) is the
    # sigma-regularized visibility transition in progress -- margin_vis
    # is recorded as the continuous t_invisible order parameter, not a
    # stop. Raw-certificate failure raises in the stepper (fail-closed).
    if rec.get("raw_vis_agree") is False:
        if not args.endgame:
            return "raw vs visible incidence partitions diverged"
        if state.get("raw_vis_diverged_at") is None:
            state["raw_vis_diverged_at"] = step
            print(f"== EVENT raw/vis incidence diverged at step "
                  f"{step} ==", flush=True)
    prev_gap = state.get("prev_gap")
    state["prev_gap"] = gap
    bie = getattr(args, "metric", "grid") == "bie"
    # BIE backend: after the quotient the cancelling pair is held
    # fixed, so the gap floor shows as a CONSTANT gap (non-decreasing),
    # not as a re-expansion
    gap_up = None
    if gap is not None and prev_gap is not None:
        gap_up = ((gap >= prev_gap * (1.0 - 1e-9))
                  if (bie and state.get("quotient_at") is not None)
                  else (gap > prev_gap))
    if gap_up is not None and consec("n_gap_up", gap_up, 3):
        # State-dependent semantics (reviewer 2026-08-18): before the
        # quotient, or after it while the VISIBLE incidence still reads
        # two bulks, a re-expanding gap can flag pump / constraint
        # failure / sheet separation -> hard failure stop (unchanged).
        # After the quotient AND the raw/vis divergence (visible
        # incidence = 1), with the switch-scale certificate holding and
        # g90 decreasing, a min-gap reversal is the fixed-N ghost
        # representation reaching its gap floor: recorded and stopped
        # as a REPRESENTATION EVENT (compression eligibility), not a
        # failure. t_comp is reserved for a committed compression.
        # BIE backend: the raw/visible divergence is recorded as
        # telemetry only (the frozen pair no longer evolves, so a
        # later divergence cannot be waited for)
        if (state.get("quotient_at") is not None
                and (state.get("raw_vis_diverged_at") is not None
                     or bie)):
            state["comp_entry_at"] = step
            state["stop_stage"] = "representation_event"
            print(f"== EVENT GhostCompressionEligibilityEvent "
                  f"(t_comp-entry) at step {step}: min-gap floor "
                  f"reached ==", flush=True)
            return ("GhostCompressionEligibilityEvent: min-gap floor "
                    "(representation_event, t_comp-entry)")
        return "gap re-expanded 3 consecutive steps"
    # first-order constraint gate: max_b eps_flux < 1e-10
    ef = rec.get("eps_flux_max")
    if ef is not None and ef > 1e-10:
        return f"bulk flux residual eps_flux = {ef:.3e} > 1e-10"
    # NOTE (measured, exact): the per-step change of ANY
    # 0.5 sum m x.n volume is dominated by the estimator's ROTATION
    # response on tilted sheets (97.5% at 1300; transport term 1e-21),
    # so no per-step delta-V gate is a conservation observable here.
    # dv_mm_* + dv_mm_rot_* are recorded as telemetry; the finite-step
    # conservation monitor is the SIGNATURE gate below.
    # pump-SIGNATURE gate: anti-correlated committed-volume exchange
    # (disk down AND annulus up) at >= 10% of the measured 0Dx pump
    # rate (2e-4/step) for 5 consecutive steps, pre-registered
    for key in ("vol_cur_disk", "vol_cur_ann"):
        v = rec.get(key)
        state["d_" + key] = (None if v is None
                             or state.get("prev_" + key) is None
                             else v - state["prev_" + key])
        state["prev_" + key] = v
    dd, da = state.get("d_vol_cur_disk"), state.get("d_vol_cur_ann")
    if dd is not None and da is not None:
        if consec("n_pump", dd < -2e-5 and da > 2e-5, 5):
            if state.get("quotient_at") is None:
                return (f"pump signature: dV_disk {dd:.2e} < -2e-5 and "
                        f"dV_ann {da:.2e} > +2e-5 for 5 consecutive "
                        "steps")
            # 0L-B2 State III: old-component exchange is no longer a
            # pump -- shadow telemetry only (merged volume is the gate)
            if state.get("pump_shadow_noted") is None:
                state["pump_shadow_noted"] = step
                print(f"== (shadow) old-component exchange after "
                      f"quotient at step {step}: dV_disk {dd:.2e}, "
                      f"dV_ann {da:.2e} ==", flush=True)
    # 0L-B2 State III primary volume gate: merged geometric volume.
    # Comp-4: after a committed compression the ghost pair is gone --
    # the SAME physical merged volume continues under watch as the
    # outer loop's area alone (vd = 0 read; pre-compression path
    # bitwise unchanged)
    if state.get("quotient_at") is not None:
        vd, va = rec.get("vol_geom_disk"), rec.get("vol_geom_ann")
        if vd is None and state.get("comp_committed_at") is not None:
            vd = 0.0
        if vd is not None and va is not None:
            vm = vd + va
            base = state.setdefault("V_merged_base", vm)
            if abs(vm - base) / base > 5e-3:
                return (f"merged geometric volume drift "
                        f"{(vm - base) / base:+.2e} > 5e-3 after quotient")
    # shape telemetry: radial variance must not grow, centroid must
    # not drift systematically (volume-neutral circularization is OK)
    sd = rec.get("S_disk")
    if sd is not None:
        base = state.setdefault("S_disk_base", sd)
        # absolute floor 1e-8: physical layer signals are ~1e-4 while
        # a perfect t=0 circle sits at round-off (~1e-13) where any
        # jitter is a large RATIO but no signal
        if consec("n_sdisk", sd > 1.5 * base + 1e-8, 10):
            return (f"disk radial variance rising ({sd:.3e} vs "
                    f"baseline {base:.3e})")
    cd = rec.get("centroid_disk")
    if cd is not None and hw is not None:
        c0 = state.setdefault("centroid_disk_0", cd)
        drift = math.hypot(cd[0] - c0[0], cd[1] - c0[1])
        if drift > 5.0 * hw:
            return f"disk centroid drift {drift:.4f} > 5 h_wall"
    f = rec.get("inactive_rhs_fraction")
    if f is not None:
        base = state.setdefault("irf_base", f)
        if consec("n_irf", f > max(2.0 * base, base + 0.005), 5):
            return (f"inactive-RHS fraction rising ({f:.4f} vs "
                    f"baseline {base:.4f})")
    return None


def new_stepper_with_frozen_state(cfg, target0):
    """Comp-4 factory (reviewer sec. 7): every stepper created inside
    the atomic compression transaction and the post-compression
    production solver inherits the SAME frozen t=0 volume target
    (initial_fixed semantics -- a fresh instance must never re-derive
    the target from its own first cloud, or the volume gate becomes
    trivially self-referential). delta/tau/sigma live in cfg."""
    from src.torch.solver.mm_step import MMStepper
    st = MMStepper(cfg)
    st._grid_target_volume_initial = target0
    return st


def reset_after_compression(state, p_sharp_comp):
    """Comp-4 (reviewer sec. 8): after the zero-time 487 -> 256 event
    the rolling driver state refers to the pre-compression cloud and
    must be re-seeded from C^n; history markers are kept, and
    V_merged_base is kept (the SAME physical merged volume stays under
    watch across the event). The compression energy drop is event
    telemetry -- p_full_prev is set to P#(C^n) so the next step's
    dissipation does not absorb the jump."""
    for k in ("prev_committed", "prev_radii", "prev_gap",
              "prev_vol_cur_disk", "prev_vol_cur_ann", "n_gap_up",
              "n_pump", "n_sdisk", "n_irf", "irf_base", "S_disk_base",
              "centroid_disk_0", "R_disk_0", "pump_shadow_noted",
              "stop_stage", "stop_reason", "_last_committed"):
        state.pop(k, None)
    state["p_full_prev"] = p_sharp_comp


def run_compression_transaction(pos, ang, quot, cfg, delta, tau,
                                target0, event_step):
    """Comp-4 atomic transaction from the committed event state X^n
    (zero-time representation event; reviewer 2026-08-19 spec).

    preconditions -> C^n = conservative_compression(X^n) ->
    event-level hard gates (V#_geom conservation, P# non-increase,
    transfer-map residual, homothety certificates, 0 < c* <= 1) ->
    shadow arms X_K^{n+1} = Advance(X^n), X_C^{n+1} = Advance(C^n)
    (both steppers from the frozen-target factory) -> dynamics gates
    (existing calibrated constants, drop-style comparisons only).
    Returns (passed, record, compressed_varifold, committed_C_next).
    Nothing outside this function is mutated (failure purity)."""
    import torch as _t

    from src.torch.oriented_varifold.loopwise_mass import (
        resolve_loopwise_oriented_mass_source,
    )
    from src.torch.solver.ghost_compression import (
        EPS_ALG,
        GhostCompressionError,
        conservative_compression,
        current_distance,
        phase_field_distance,
        transfer_map_residual,
    )
    from src.torch.solver.mm_solver import MMSolver
    from src.torch.solver.quotient_gates import QUOTIENT_GATES
    from src.torch.transport.bem_wasserstein import compute_coherence

    rec = dict(step=int(event_step), type="ghost_compression",
               eps_alg=EPS_ALG, thresholds=dict(QUOTIENT_GATES))
    gates = {}
    rec["gates"] = gates

    def fail(reason, exc=None):
        rec["passed"] = False
        rec["reason"] = reason
        if exc is not None:
            rec["exception"] = f"{type(exc).__name__}: {exc}"
        return False, rec, None, None

    # ---- preconditions (reviewer sec. 6) --------------------------
    ref = (quot or {}).get("contact_reference") or {}
    if not quot or quot.get("quotient_at") is None:
        return fail("no committed quotient at the event state")
    if ref.get("particles_a_mask") is None \
            or ref.get("particles_b_mask") is None:
        return fail("contact_reference particle masks unresolved")
    mask_a = _t.as_tensor(ref["particles_a_mask"]).bool()
    mask_b = _t.as_tensor(ref["particles_b_mask"]).bool()
    mask_o = ~(mask_a | mask_b)
    vK = OrientedPointCloudVarifold(positions=pos.clone(),
                                    angles=ang.clone())
    solver = MMSolver(cfg)
    try:
        # ---- keep arm (also provides labels/masses on X^n) --------
        st_k = new_stepper_with_frozen_state(cfg, target0)
        st_k.import_quotient_state(quot, vK.n_points)
        res_k, committed_k = solver._advance_committed(st_k, vK)
        m_src, labels_src = st_k._resolve_state_masses_labels(
            pos, vK.normals)
        # mask completeness: pair loops + outer = the FULL certified
        # loop partition, each mask one loop exactly (reviewer sec. 4)
        for name, mm in (("pair_a", mask_a), ("pair_b", mask_b),
                         ("outer", mask_o)):
            labs = labels_src[mm].unique()
            if labs.numel() != 1:
                return fail(f"{name} mask spans {labs.numel()} "
                            "certified loops")
            if not bool(((labels_src == labs[0]) == mm).all()):
                return fail(f"{name} mask != its certified loop "
                            "(inflow/dropout)")
        if labels_src.unique().numel() != 3:
            return fail(f"{labels_src.unique().numel()} certified "
                        "loops at the event (need exactly 3)")
        cert_src = st_k.contact_reference_on_state(vK)
        if cert_src is None or not cert_src.certified:
            return fail("State III contact reference does not hold "
                        "on the event state")
        # ---- the zero-time map C(X^n) -----------------------------
        comp = conservative_compression(pos, ang, vK.normals,
                                        mask_o, mask_a, mask_b)
        vC = OrientedPointCloudVarifold(positions=comp["positions"],
                                        angles=comp["angles"])
        h_wall = float(st_k.fixed_masses.median())
        # ---- compress arm -----------------------------------------
        st_c = new_stepper_with_frozen_state(cfg, target0)
        res_c, committed_c = solver._advance_committed(st_c, vC)
        # ---- event-level hard gates (eps_alg; no new calibrated
        # thresholds) -----------------------------------------------
        P_src = st_k.sharp_perimeter(vK)
        P_comp = st_c.sharp_perimeter(vC)
        gates["volume_event"] = (
            abs(comp["V_comp_geom"] - comp["V_sharp_geom"])
            <= EPS_ALG * max(comp["V_sharp_geom"], 1.0) * 10.0)
        gates["perimeter_non_increase"] = P_comp <= P_src + 1e-9
        eps_tr = transfer_map_residual(vC.positions,
                                       comp["positions"], h_wall)
        gates["transfer_map_residual"] = eps_tr <= EPS_ALG * 1e3
        gates["certificates"] = bool(comp["certificates"]["all"])
        gates["scale_in_unit"] = 0.0 < comp["scale"] <= 1.0
        # ---- dynamics gates (existing constants, drop-style) ------
        def pos_(x):
            return max(x, 0.0)
        W_k, W_c = float(res_k.wasserstein), float(res_c.wasserstein)
        P_k1 = st_k.sharp_perimeter(committed_k)
        P_c1 = st_c.sharp_perimeter(committed_c)
        split_k = P_k1 + W_k - P_src
        split_c = P_c1 + W_c - P_comp
        gates["split_residual"] = (
            pos_(split_c) <= max(pos_(split_k),
                                 QUOTIENT_GATES["eps_split"]))
        from src.torch.solver.ghost_compression import (
            merged_geom_volume,
            star_shoelace_area,
        )
        V_k1 = merged_geom_volume(
            committed_k.positions, committed_k.normals,
            [mask_o, mask_a, mask_b])
        V_c1 = star_shoelace_area(committed_c.positions)
        dV_k = (V_k1 - comp["V_sharp_geom"]) / comp["V_sharp_geom"]
        dV_c = (V_c1 - comp["V_sharp_geom"]) / comp["V_sharp_geom"]
        gates["volume_step"] = abs(dV_c) <= max(
            abs(dV_k), QUOTIENT_GATES["eps_volume"])
        dth = float((_wrap_angle_t(committed_k.angles[mask_o]
                                   - committed_c.angles)).abs().max())
        gates["switch_theta"] = dth <= QUOTIENT_GATES[
            "eps_switch_theta"]
        s_k = float(res_k.displacements.abs().max()) / h_wall
        s_c = float(res_c.displacements.abs().max()) / h_wall
        gates["fixed_point_residual"] = s_c <= s_k
        gates["grid_components"] = int(
            st_c._last_grid_snapshot.n_components) == 1
        cert_k1 = st_k.contact_reference_on_state(committed_k)
        gates["keep_reference_certified"] = bool(
            cert_k1 is not None and cert_k1.certified)
        # ---- telemetry --------------------------------------------
        sig = cfg.perimeter_sigma
        q_src = compute_coherence(vK, m_src, sig, cfg.perimeter_kernel)
        m_c = resolve_loopwise_oriented_mass_source(
            vC.positions, vC.normals, delta, tau).m_loop
        q_c = compute_coherence(vC, m_c, sig, cfg.perimeter_kernel)
        st_k2 = new_stepper_with_frozen_state(cfg, target0)
        st_k2.import_quotient_state(quot, vK.n_points)
        st_k2._setup_step(committed_k)
        st_c2 = new_stepper_with_frozen_state(cfg, target0)
        st_c2._setup_step(committed_c)
        if st_k.grid_wasserstein is not None:
            rho_k = st_k.grid_wasserstein.phase.rho_metric
            rho_c = st_c.grid_wasserstein.phase.rho_metric
            d_rho_event = phase_field_distance(rho_k, rho_c)
            d_rho_step = phase_field_distance(
                st_k2.grid_wasserstein.phase.rho_metric,
                st_c2.grid_wasserstein.phase.rho_metric)
        else:                       # BIE backend: no phase field
            d_rho_event = None
            d_rho_step = None
        rec.update(
            scale=comp["scale"], z=comp["z"],
            root=dict(comp["root"]),
            V_sharp_geom=comp["V_sharp_geom"],
            V_pair_geom=comp["V_pair_geom"],
            V_comp_geom=comp["V_comp_geom"],
            V_cur_shadow_src=float(
                (0.5 * m_src * (pos * vK.normals).sum(-1)).sum()),
            V_cur_shadow_comp=float(
                (0.5 * m_c * (vC.positions * vC.normals)
                 .sum(-1)).sum()),
            transfer_inf_over_h=comp["transfer_inf"] / h_wall,
            transfer_map_residual=eps_tr,
            P_sharp_src=P_src, P_sharp_comp=P_comp,
            dP_sharp_event=P_comp - P_src,
            D_T_raw=current_distance(pos, vK.normals, m_src,
                                     vC.positions, vC.normals, m_c,
                                     sig),
            D_T_vis_qfull=current_distance(
                pos, vK.normals, m_src * q_src,
                vC.positions, vC.normals, m_c * q_c, sig),
            D_rho_event=d_rho_event, D_rho_step=d_rho_step,
            keep=dict(split_residual=split_k, dV_rel=dV_k,
                      max_s_over_h=s_k, n_iter=int(res_k.n_iter),
                      P_sharp_committed=P_k1, wasserstein=W_k),
            compress=dict(split_residual=split_c, dV_rel=dV_c,
                          max_s_over_h=s_c, n_iter=int(res_c.n_iter),
                          P_sharp_committed=P_c1, wasserstein=W_c),
            certificates={k: v for k, v in
                          comp["certificates"].items()},
            h_wall=h_wall, n_before=int(pos.shape[0]),
            n_after=int(vC.n_points))
    except (GhostCompressionError, Exception) as exc:  # noqa: BLE001
        return fail("transaction exception", exc)
    passed = all(gates.values())
    rec["passed"] = passed
    if not passed:
        rec["reason"] = [k for k, v in gates.items() if not v]
        return False, rec, None, None
    return True, rec, vC, committed_c


def _wrap_angle_t(a):
    import math as _m
    return (a + _m.pi) % (2 * _m.pi) - _m.pi


def build_config(args, delta, tau):
    """Production MMConfig of the exact-merger driver (single source
    for the driver, the B2 stability audit and any offline harness)."""
    from p1_production_comparison import make_cfg

    from src.torch.transport.grid_wasserstein import GridMetricConfig
    from src.torch.transport.phase_grid import PhaseGridConfig

    cfg = make_cfg("C3", delta, tau, 2)
    cfg.time_step = DT_STEP
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(args.grid, args.grid),
        fill_epsilon=args.fill_epsilon,
        support_threshold=3e-3, projection_rel_tol=5e-2),
        compatibility_components="conservative_sweep")
    if getattr(args, "metric", "grid") == "bie":
        from src.torch.transport.bie_wasserstein import BIEMetricConfig
        cfg.metric_backend = "bie"
        cfg.grid_metric = None
        cfg.bie_metric = BIEMetricConfig(
            bridge_gap_over_ell=float(getattr(args, "bridge_gap", 1.0)))
    cfg.redistribute = not args.no_redistribute
    cfg.remove_dead_points = False
    cfg.enforce_support_gates = False
    cfg.grid_volume_target_mode = "initial_fixed"
    cfg.grid_phase_support_projection = False
    cfg.grid_fossil_advection = args.advect
    cfg.grid_freeze_dead_dof = args.freeze_dead
    cfg.mass_estimator = args.mass_estimator
    if args.bulk_rows:
        cfg.grid_bulk_rows_mode = "augment"
    cfg.grid_bulk_rows_functional = args.bulk_functional
    cfg.grid_volume_target_functional = args.volume_target_functional
    cfg.grid_contact_rows_mode = args.contact_rows_mode
    cfg.grid_gauge_hold_bulk_seeded = args.gauge_hold
    cfg.grid_quotient_mode = args.quotient_mode
    if args.quotient_mode != "off":
        if getattr(args, "metric", "grid") == "bie":
            from src.torch.solver.quotient_gates_bie import (
                QUOTIENT_GATES_BIE as QUOTIENT_GATES,
            )
        else:
            from src.torch.solver.quotient_gates import QUOTIENT_GATES
        cfg.grid_quotient_eps_split = QUOTIENT_GATES["eps_split"]
        cfg.grid_quotient_eps_switch_x = QUOTIENT_GATES["eps_switch_x"]
        cfg.grid_quotient_eps_switch_theta = \
            QUOTIENT_GATES["eps_switch_theta"]
        cfg.grid_quotient_eps_volume = QUOTIENT_GATES["eps_volume"]
    cfg.angle_constraint_scope = args.angle_scope
    cfg.angle_constraint_measure = args.angle_measure
    cfg.angle_constraint_consistency = args.angle_consistency
    cfg.redistribution_density_scope = args.redist_scope
    cfg.redistribution_tangent_source = args.redist_tangent
    cfg.redistribution_curvature_scope = args.redist_curvature
    cfg.redistribution_q_policy = args.redist_q_policy
    cfg.redistribution_normal_retraction = args.redist_retraction
    cfg.redistribution_monotone = getattr(args, "redist_monotone", False)
    cfg.redistribution_monotone_parts = getattr(args, "redist_monotone_parts",
                                                "abc")
    cfg.perimeter_q_mode = args.q_mode
    if args.vertical:
        from src.torch.transport.carrier_amplitude import (
            CarrierAmplitudeConfig,
        )
        cfg.grid_carrier_amplitude_fade = True
        cfg.grid_vertical_dof = True
        cfg.grid_amplitude_config = CarrierAmplitudeConfig(
            lam=args.lam)
    if getattr(args, "optimizer_max_iter", None) is not None:
        cfg.optimizer_max_iter = args.optimizer_max_iter
    if getattr(args, "optimizer_tol", None) is not None:
        cfg.optimizer_tol = args.optimizer_tol
    return cfg


def run(args):
    from p1_production_comparison import make_cfg

    from src.torch.solver.mm_solver import MMSolver
    from src.torch.transport.grid_wasserstein import GridMetricConfig
    from src.torch.transport.phase_grid import PhaseGridConfig

    sol, t_c, r_eq = exact_reference()
    v0, n_disk = build_cloud()
    n_steps = (args.steps if args.steps is not None
               else int(round(1.3 * t_c / DT_STEP)))
    print(f"exact: T_c = {t_c:.6e} ({t_c / DT_STEP:.0f} steps), "
          f"R_eq = R3(T_c) = {r_eq:.6f}; running {n_steps} steps, "
          f"N = {v0.n_points} (disk {n_disk})")

    delta, tau = compute_recommended_params(v0.positions)
    delta, tau = float(delta), float(tau)
    cfg = build_config(args, delta, tau)

    series = []
    states = {}
    step_offset = 0
    v_start = v0
    _resume_quotient = None
    _resume_compression = None
    OUT.mkdir(parents=True, exist_ok=True)
    if args.resume is not None:
        src_tag = (args.states_from if args.states_from is not None
                   else args.tag)
        ck = torch.load(OUT / f"exact_merger{src_tag}_states.pt",
                        weights_only=True)
        st0 = ck[args.resume]
        pos0, ang0 = st0["positions"], st0["angles"]
        # 0M-1 discriminators: rigid transforms of the RESUME state
        # relative to the fixed grid (isotropic-continuum physics is
        # invariant; only grid-locked mechanisms can tell)
        if (args.resume_rotate_deg != 0.0
                or any(args.resume_shift)
                or args.resume_jitter != 0.0):
            al = math.radians(args.resume_rotate_deg)
            if al != 0.0:
                rot = torch.tensor(
                    [[math.cos(al), -math.sin(al)],
                     [math.sin(al), math.cos(al)]], dtype=pos0.dtype)
                pos0 = pos0 @ rot.T
                ang0 = ang0 + al
            if any(args.resume_shift):
                pos0 = pos0 + torch.tensor(args.resume_shift,
                                           dtype=pos0.dtype)
            if args.resume_jitter != 0.0:
                g = torch.Generator().manual_seed(
                    args.resume_jitter_seed)
                pos0 = pos0 + args.resume_jitter * torch.randn(
                    pos0.shape, generator=g, dtype=pos0.dtype)
            print(f"resume transform: rotate {args.resume_rotate_deg}"
                  f" deg, shift {tuple(args.resume_shift)}, jitter "
                  f"{args.resume_jitter} (seed "
                  f"{args.resume_jitter_seed})", flush=True)
        v_start = OrientedPointCloudVarifold(
            positions=pos0, angles=ang0)
        # 0L-B2: an irreversible quotient survives a resume
        _resume_quotient = st0.get("quotient")
        # Comp-4: a committed compression survives a resume (the
        # one-loop gate semantics must apply from the first step)
        _resume_compression = st0.get("compression_state")
        if args.repair_normals_on_resume:
            from src.torch.solver.redistribution_rules import (
                repair_boundary_normals,
            )
            new_ang, rep = repair_boundary_normals(
                v_start.positions, v_start.angles, delta, tau, 0.1)
            v_start = OrientedPointCloudVarifold(
                positions=v_start.positions, angles=new_ang)
            print(f"repaired resume normals: max angle "
                  f"{rep['retraction']['max_angle_rad']:.4f}, "
                  f"dP+ rel {rep['delta_p_plus_rel']:.2e}, "
                  f"dV_cur {rep['delta_v_cur']:+.2e}", flush=True)
        step_offset = args.resume + 1
        print(f"resuming from step {args.resume} "
              f"(states of tag {src_tag!r}, N={v_start.n_points})")
    target0 = None
    if args.resume is not None or args.compression == "atomic":
        # initial_fixed semantics: the volume target stays the
        # DETERMINISTIC t=0 cloud's (same driver-local seeding as
        # grid_pair_contact), estimator-consistent (IIIc-0D: the
        # oriented run must not chase a position-mass target). Comp-4:
        # the atomic transaction's fresh steppers and the
        # post-compression solver must inherit the SAME frozen target,
        # so the seeding applies to every stepper of an atomic run too.
        m0 = masses_for(args.mass_estimator, v0.positions, v0.normals,
                        delta, tau)
        target0 = float(
            0.5 * (m0 * (v0.positions * v0.normals).sum(-1)).sum())
        from src.torch.solver.mm_step import MMStepper as _MS
        _orig_init = _MS.__init__

        def _seeded_init(s, *a, **k):
            _orig_init(s, *a, **k)
            s._grid_target_volume_initial = target0
        _MS.__init__ = _seeded_init
    state = {"merged_at": None, "t_prev": time.time()}
    if _resume_compression:
        state["comp_committed_at"] = _resume_compression.get(
            "committed_at")
        state["_compression_state"] = _resume_compression
        print(f"imported compression state: committed_at "
              f"{state['comp_committed_at']}", flush=True)
    save_steps = set(args.save_steps or [])
    _solver_ref = {}

    def shadow_projection_rel(res):
        """0L-B0 cell C gate calibration: on the fail-closed forward
        path, how much of the committed step's flux the componentwise
        projection WOULD discard (shadow only; the projected forward
        records the same quantity live via max_projection_rel)."""
        sv = _solver_ref.get("s")
        gm = getattr(getattr(sv, "_stepper", None),
                     "grid_wasserstein", None)
        if gm is None or gm.poisson is None or gm.flux is None:
            return None
        try:
            with torch.no_grad():
                drho = gm.flux(res.displacements, res.delta_angles)
                b = gm.poisson.project_componentwise_zero_mean(drho)
                nd = float(drho.norm())
                return float((drho - b).norm()) / nd if nd > 0 else 0.0
        except Exception:
            return None
    t0 = time.time()
    out = OUT / f"exact_merger{args.tag}.json"
    states_path = OUT / f"exact_merger{args.tag}_states.pt"

    def dump(err=None):
        # atomic periodic checkpoint: the 11 h v2 run wrote only at
        # exit and one 7 h optimizer grind lost the whole series
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(
            meta=dict(R1=R1, R2=R2, R3=R3, dt=DT_STEP, n_out=N_OUT,
                      n_disk=n_disk, n_points=v0.n_points,
                      grid=args.grid, t_c_exact=t_c, r_eq_exact=r_eq,
                      n_steps=n_steps, vertical=args.vertical,
                      lam=args.lam, advect=args.advect,
                      freeze_dead=args.freeze_dead,
                      merged_at_step=state["merged_at"], error=err,
                      stop_reason=state.get("stop_reason"),
                      stop_stage=state.get("stop_stage"),
                      quotient_at=state.get("quotient_at"),
                      raw_vis_diverged_at=state.get(
                          "raw_vis_diverged_at"),
                      comp_entry_at=state.get("comp_entry_at"),
                      comp_committed_at=state.get("comp_committed_at"),
                      shadow_parity_max_abs=state.get(
                          "shadow_parity_max_abs"),
                      events=state.get("events")),
            series=series), indent=1))
        tmp.replace(out)
        torch.save(states, states_path)

    _off = {"v": step_offset}

    def cb(step, res):
        step = step + _off["v"]
        vv = (res.committed_varifold
              if res.committed_varifold is not None else res.varifold)
        snap = res.grid_setup_snapshot
        t = (step + 1) * DT_STEP
        now = time.time()
        if args.compression == "atomic":
            # Comp-4: the transaction reads the committed EVENT state;
            # kept every step (tiny) so the event needs no re-solve
            state["_last_committed"] = (vv.positions.clone(),
                                        vv.angles.clone(),
                                        res.quotient_state)
            shadow_c = state.pop("_shadow_c_next", None)
            if shadow_c is not None:
                # pin 10 telemetry: the first re-run committed state
                # must reproduce the shadow X_C^{n+1}
                state["shadow_parity_max_abs"] = float(
                    (vv.positions - shadow_c.positions).abs().max())
        if step % 25 == 0 or step in save_steps or \
                res.quotient_switch is not None:
            states[step] = {"positions": vv.positions.clone(),
                            "angles": vv.angles.clone(),
                            "quotient": res.quotient_state}
            if state.get("_compression_state") is not None:
                states[step]["compression_state"] = \
                    state["_compression_state"]
        disp = res.displacements
        ds = disp.norm(dim=1) if disp.dim() == 2 else disp.abs()
        m_loc = masses_for(args.mass_estimator, vv.positions,
                           vv.normals, delta, tau)
        rec = {
            "step": step,
            "t": t,
            "wall_s": now - state["t_prev"],
            "n_points": vv.n_points,
            "perimeter": float(res.perimeter),
            "wasserstein": float(res.wasserstein),
            "objective": float(res.objective),
            "converged": bool(res.converged),
            "n_iter": int(res.n_iter),
            "max_disp": float(ds.max()),
            "max_disp_over_spacing": float((ds / m_loc).max()),
            "n_components": (snap.n_components
                             if snap is not None else None),
            "projection_l2_relative": (snap.projection_l2_relative
                                       if snap is not None else None),
            "phase_grid_volume": (snap.phase_grid_volume
                                  if snap is not None else None),
            "clusters": radius_clusters(vv.positions),
            "vertical": res.vertical_stats,
            "amplitude": res.amplitude_stats,
        }
        rec.update(wall_diagnostics(
            vv, args.mass_estimator, delta, tau,
            probe_curvature=(step % 25 == 0)))
        # 0M-0: per-loop complex m4 observable + grid-scale ratios
        # (d/h and d/eps_fill independently; d/delta_KDE is the mass
        # channel's scale, not the grid channel's)
        try:
            lw_ = loopwise_resolution(vv.positions, vv.normals,
                                      delta, tau)
            rec["m4"] = loop_m4(vv, lw_.m_loop, lw_.loop_labels_pre)
            hw_ = rec.get("h_wall")
            if hw_:
                for role_ in rec["m4"]:
                    rec["m4"][role_]["A4_over_hwall"] = \
                        rec["m4"][role_]["A4"] / hw_
        except Exception as ex_:
            rec["m4"] = {"error": str(ex_)[:120]}
        # 0J-E refresh-defect audit: Delta_refresh = P_full(X^{n+1})
        # - P_WB(X^{n+1}|X^n) (the latter = res.perimeter, up to the
        # redistribution between minimizer and committed state), and
        # the one-step dissipation R_n on the FULL energy
        p_full_now = rec["p_full_committed"]
        p_full_prev = state.get("p_full_prev")
        rec["delta_refresh"] = p_full_now - float(res.perimeter)
        if p_full_prev is not None:
            rec["dissipation_full"] = (p_full_now
                                       + float(res.wasserstein)
                                       - p_full_prev)
        state["p_full_prev"] = p_full_now
        rec.update(extension_diagnostics(
            vv, args.mass_estimator, delta, tau, state, DT_STEP))
        # 0M-0: grid-scale gap ratios (d/h and d/eps_fill are the
        # grid channel's dimensionless coordinates)
        if rec.get("gap") is not None:
            rec["gap_over_h"] = rec["gap"] / (4.0 / args.grid)
            rec["gap_over_eps_fill"] = rec["gap"] / args.fill_epsilon
        if args.redist_curvature == "loopwise" or \
                args.redist_scope == "loopwise":
            # 0L R2: PCA shadow certificate (pre-registered
            # thresholds E2_STAR/EINF_STAR; committed state gates,
            # the MM candidate is recorded for stage attribution --
            # preMM equals the previous step's committed record)
            rec["e2_postMM"], rec["einf_postMM"] = e_int_of(
                res.varifold, delta, tau)
            rec["e2"], rec["einf"] = e_int_of(vv, delta, tau)
        if args.bulk_rows:
            # transport-channel volume drift: FROZEN pre-step masses,
            # MM minimizer BEFORE redistribution (res.varifold) -- the
            # quantity the constraint actually controls to O(|s|^2).
            # The committed-state V drift additionally carries the
            # redistribution and mass re-estimation channels (known,
            # pre-existing; recorded via vol_cur_* as telemetry).
            prev = state.get("prev_committed")
            if prev is not None:
                p_pos, p_ang, p_m = prev
                mmv = res.varifold

                def _vols(pos, ang, mw):
                    va_ = OrientedPointCloudVarifold(positions=pos,
                                                     angles=ang)
                    xn_ = (va_.positions * va_.normals).sum(-1)
                    dk = ((va_.positions.norm(dim=1) < 0.6)
                          & (xn_ > 0))
                    return (float(0.5 * (mw[dk] * xn_[dk]).sum()),
                            float(0.5 * (mw[~dk] * xn_[~dk]).sum()))

                d0_, a0_ = _vols(p_pos, p_ang, p_m)
                d1_, a1_ = _vols(mmv.positions, mmv.angles, p_m)
                rec["dv_mm_disk"] = d1_ - d0_
                rec["dv_mm_ann"] = a1_ - a0_
                # rotation-response decomposition: the estimator
                # 0.5 sum m x.n responds to angle updates on tilted
                # sheets (0.5 sum m (x.t) dtheta) while the true swept
                # volume of a rotating symmetric segment is zero --
                # measured 97.5% of dv_mm at 1300; the transport term
                # 0.5 sum m s is the constraint functional (~1e-21)
                pv = OrientedPointCloudVarifold(positions=p_pos,
                                                angles=p_ang)
                nn = pv.normals
                tt = torch.stack([-nn[:, 1], nn[:, 0]], 1)
                xt = (pv.positions * tt).sum(1)
                dth_ = res.delta_angles
                dk_ = ((pv.positions.norm(dim=1) < 0.6)
                       & ((pv.positions * nn).sum(1) > 0))
                rec["dv_mm_rot_disk"] = float(
                    0.5 * (p_m[dk_] * xt[dk_] * dth_[dk_]).sum())
                rec["dv_mm_rot_ann"] = float(
                    0.5 * (p_m[~dk_] * xt[~dk_] * dth_[~dk_]).sum())
            state["prev_committed"] = (vv.positions.clone(),
                                       vv.angles.clone(), m_loc)
            sn = res.grid_setup_snapshot
            gs = res.grid_step_stats
            rec.update(
                augmentation_fired=(sn.augmentation_fired
                                    if sn else None),
                partition_relation=(sn.partition_relation_grid_bulk
                                    if sn else None),
                r_grid=(sn.r_grid if sn else None),
                r_bulk=(sn.r_bulk if sn else None),
                r_stack=(sn.r_stack if sn else None),
                delta_compat=(sn.delta_compat if sn else None),
                bulk_flux_residuals=res.bulk_flux_residuals,
                inactive_rhs_fraction=(gs.max_inactive_rhs_fraction
                                       if gs else None),
                # 0L-B0 contact-layer telemetry
                contact_rows_mode=(sn.contact_rows_mode if sn else None),
                eps_row=(sn.eps_row if sn else None),
                eps_row_composed=(sn.eps_row_composed if sn else None),
                gauge_hold_fired=(sn.gauge_hold_fired if sn else None),
                gamma_cross=(sn.gamma_cross if sn else None),
                n_compat_base=(sn.n_compat_base if sn else None),
                n_compat_used=(sn.n_compat_used if sn else None),
                partition_relation_base=(sn.partition_relation_base
                                         if sn else None),
                constraint_sv=(list(sn.constraint_row_singular_values)
                               if sn else None),
                max_projection_rel=(gs.max_projection_rel
                                    if gs else None),
                contact_rows_telemetry=res.contact_rows_telemetry,
                objective_initial=float(res.objective_initial),
                proj_rel_shadow=shadow_projection_rel(res),
            )
            # 0L-B1: contact certificate telemetry (observe only). The
            # loop->grid/bulk maps are the SOURCE state's (State-II
            # bookkeeping); the metrics are read on the committed
            # state with the 0K loopwise masses/labels. Certificate
            # gates: uniqueness AND gap90/h AND coverage AND anti AND
            # mass AND max_sigma eps_cur2 -- no row is dropped here.
            rec["contact_cert"] = contact_certificate_record(
                vv, res.contact_pair_context, delta, tau)
            rec["contact_cert_source"] = res.contact_certificate_source
            rec["contact_cert_committed"] = \
                res.contact_certificate_committed
            # 0N-4: State III reference invariant -- switch-scale is
            # the hard gate, live-scale is shadow telemetry
            if res.contact_certificate_committed:
                _mref = res.contact_certificate_committed[0].get(
                    "metrics", {})
                if "gap90_over_h_switch" in _mref:
                    for k in ("gap90_over_h_switch", "gap90_over_h_live",
                              "coverage_switch_scale",
                              "coverage_live_scale", "h_ab_live"):
                        rec[k] = _mref.get(k)
                    if step % 25 == 0:
                        print(f"    gap90/h* {_mref['gap90_over_h_switch']:.4f}"
                              f" (live {_mref['gap90_over_h_live']:.4f}) "
                              f"cov* {_mref['coverage_switch_scale']:.3f}",
                              flush=True)
            rec["quotient_switch"] = res.quotient_switch
            rec["quotient_state"] = (
                {"n_groups": len(res.quotient_state["quotient_bulk_groups"]),
                 "quotient_at": res.quotient_state.get("quotient_at")}
                if res.quotient_state else None)
            if res.quotient_switch is not None:
                state["quotient_at"] = step
                print(f"== EVENT quotient switch committed at step {step} "
                      f"(bulks {res.quotient_switch['bulk_pair']}) ==",
                      flush=True)
            rec.update(incidence_diagnostics(
                vv, args.mass_estimator, delta, tau,
                s=res.displacements, dtheta=res.delta_angles,
                residuals=res.bulk_flux_residuals))
        # event tracking (reviewer spec: three separate event times)
        if (state.get("contact_at") is None and rec["gap"] is not None
                and rec["gap"] <= args.contact_eps):
            state["contact_at"] = step
            print(f"== geometric contact (gap <= {args.contact_eps}) "
                  f"at step {step}, t = {t:.5f} ==", flush=True)
        for thr, key in ((0.1, "invisible01_at"), (0.05, "invisible005_at")):
            if (state.get(key) is None and rec["eta_wall"] is not None
                    and rec["eta_wall"] < thr):
                state[key] = step
                print(f"== eta_wall < {thr} at step {step} ==",
                      flush=True)
        state["t_prev"] = now
        if (state["merged_at"] is None and snap is not None
                and snap.n_components == 1):
            state["merged_at"] = step
            print(f"== support merged (C 2 -> 1) at step {step}, "
                  f"t = {t:.5f} (exact T_c {t_c:.5f}) ==")
        series.append(rec)
        if step % 25 == 0:
            cl = " ".join(f"{m:.4f}x{c}" for m, c in rec["clusters"])
            print(f"step {step:4d}: N={vv.n_points} "
                  f"C={rec['n_components']} P={rec['perimeter']:.4f} "
                  f"W={rec['wasserstein']:.4f} "
                  f"conv={rec['converged']} "
                  f"maxs={rec['max_disp']:.2e} "
                  f"proj={rec['projection_l2_relative']:.1e} "
                  f"aw={rec['alpha_wall_frac']} "
                  f"curvP={rec.get('wall_ct_curvature_P')} "
                  f"({rec['wall_s']:.1f}s) r=[{cl}]", flush=True)
            dump()
        if (args.stop_after_contact is not None
                and state.get("contact_at") is not None
                and step >= state["contact_at"] + args.stop_after_contact):
            print(f"== stopping: contact + {args.stop_after_contact} "
                  f"steps reached at step {step} ==", flush=True)
            dump()
            return True
        if args.bulk_rows:
            reason = _bulk_rows_stop(rec, state, args, step)
            if reason:
                print(f"== 0F stop at step {step}: {reason} ==",
                      flush=True)
                state["stop_reason"] = reason
                dump()
                return True
        return False

    err = None
    solver = MMSolver(cfg)
    _solver_ref["s"] = solver
    if args.resume is not None and _resume_quotient:
        solver._pending_quotient_state = _resume_quotient
        # State III resumes as State III: the post-quotient gate set
        # (pump -> shadow, merged geometric volume primary) applies
        # from the first resumed step
        state["quotient_at"] = _resume_quotient.get("quotient_at")
        print(f"imported quotient state: {len(_resume_quotient.get('quotient_bulk_groups', []))} group(s), "
              f"quotient_at {_resume_quotient.get('quotient_at')}",
              flush=True)
    def _solve_segment(sv, vst):
        seg_err = None
        try:
            history = sv.solve(vst, n_steps - _off["v"], callback=cb)
            if getattr(history, "stop_details", None):
                (OUT / f"exact_merger{args.tag}_quotient_failure.json"
                 ).write_text(json.dumps(history.stop_details, indent=1,
                                         default=str))
                seg_err = (f"{history.stop_exception_type}: "
                           f"{history.stop_message}")
                print(f"    [FAIL-CLOSED] {seg_err} (details -> "
                      f"exact_merger{args.tag}_quotient_failure.json)")
            elif getattr(history, "stop_message", None):
                seg_err = (f"{history.stop_exception_type}: "
                           f"{history.stop_message}")
        except BaseException as exc:    # noqa: BLE001 -- record & save
            seg_err = f"{type(exc).__name__}: {exc}"
            det = getattr(exc, "details", None)
            if det:
                (OUT / f"exact_merger{args.tag}_quotient_failure.json"
                 ).write_text(json.dumps(det, indent=1, default=str))
            print(f"    [ERROR] at step {len(series)}: {seg_err}")
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                dump(seg_err)
                raise
        return seg_err

    if args.compression == "off":
        # legacy control flow verbatim: one solve call, no controller
        err = _solve_segment(solver, v_start)
    else:
        while True:
            err = _solve_segment(solver, v_start)
            if not (err is None
                    and state.get("stop_stage") == "representation_event"
                    and state.get("comp_committed_at") is None
                    and state.get("_last_committed") is not None):
                break
            # ---- Comp-4 atomic zero-time transaction at event step n
            n_evt = int(state["comp_entry_at"])
            posE, angE, quotE = state["_last_committed"]
            print(f"== Comp-4 atomic transaction at event step "
                  f"{n_evt} ==", flush=True)
            okc, ev_rec, vC, committed_c = run_compression_transaction(
                posE, angE, quotE, cfg, delta, tau, target0, n_evt)
            if not okc:
                (OUT / f"exact_merger{args.tag}"
                 "_compression_failure.json").write_text(
                    json.dumps(ev_rec, indent=1, default=str))
                err = (f"GhostCompressionRejected at step {n_evt}: "
                       f"{ev_rec.get('reason')}")
                print(f"    [FAIL-CLOSED] {err} (details -> "
                      f"exact_merger{args.tag}"
                      "_compression_failure.json)")
                break
            # commit the ZERO-TIME C^n at event step n (eq. 5.1): the
            # event record is attached to step n's rec + meta events,
            # never appended as a physical series row; the resumed
            # solver generates step n+1
            ref_q = quotE["contact_reference"]
            mask_o_cpu = ~(torch.as_tensor(
                ref_q["particles_a_mask"]).bool()
                | torch.as_tensor(ref_q["particles_b_mask"]).bool())
            comp_state = dict(
                committed_at=n_evt, source_step=n_evt,
                transfer_scale=ev_rec["scale"],
                V_target_geom=ev_rec["V_sharp_geom"],
                V_target_current_shadow=ev_rec["V_cur_shadow_src"],
                outer_source_mask=mask_o_cpu.cpu(),
                event_certificate=ev_rec["certificates"])
            state["_compression_state"] = comp_state
            state["comp_committed_at"] = n_evt
            state.setdefault("events", []).append(ev_rec)
            if series and series[-1]["step"] == n_evt:
                series[-1]["compression_event"] = ev_rec
            states[n_evt] = {"positions": vC.positions.clone(),
                             "angles": vC.angles.clone(),
                             "quotient": None,
                             "compression_state": comp_state}
            reset_after_compression(state, ev_rec["P_sharp_comp"])
            state["_shadow_c_next"] = committed_c
            print(f"== EVENT t_comp: ghost compression COMMITTED at "
                  f"step {n_evt} (N {ev_rec['n_before']} -> "
                  f"{ev_rec['n_after']}, c* {ev_rec['scale']:.8f}, "
                  f"transfer {ev_rec['transfer_inf_over_h']:.4f} "
                  f"h_wall) ==", flush=True)
            v_start = vC
            _off["v"] = n_evt + 1
            solver = MMSolver(cfg)
            _solver_ref["s"] = solver
            dump()
            if _off["v"] >= n_steps:
                break

    dump(err)
    print(f"\ncompleted {len(series)}/{n_steps} steps in "
          f"{(time.time() - t0) / 3600.0:.2f} h; "
          f"merged_at={state['merged_at']} err={err}")
    print(f"wrote {out}")


def postprocess(path: str):
    """Errors against the exact reference: pre-contact ODE tracking,
    contact time, post-contact equilibrium drift."""
    d = json.loads(Path(path).read_text())
    meta, series = d["meta"], d["series"]
    sol, t_c, r_eq = exact_reference()
    rows_pre, rows_post = [], []
    merged = meta.get("merged_at_step")
    for rec in series:
        t = rec["t"]
        cl = rec["clusters"]
        if merged is None or rec["step"] < merged:
            if t >= t_c or len(cl) < 3:
                continue
            r3x, r2x = (float(x) for x in sol.sol(t))
            rows_pre.append(dict(
                step=rec["step"], t=t,
                e_disk=abs(cl[0][0] - meta["R1"]) / meta["R1"],
                e_hole=abs(cl[1][0] - r2x) / r2x,
                e_outer=abs(cl[-1][0] - r3x) / r3x))
        else:
            rows_post.append(dict(
                step=rec["step"], t=t,
                e_eq=abs(cl[-1][0] - r_eq) / r_eq,
                perimeter_rel=(rec["perimeter"]
                               / (2.0 * math.pi * r_eq) - 1.0),
                n_clusters=len(cl)))
    summary = dict(t_c_exact=t_c, r_eq_exact=r_eq)
    if merged is not None:
        summary["t_c_numeric"] = (merged + 1) * meta["dt"]
        summary["t_c_rel_error"] = (summary["t_c_numeric"] - t_c) / t_c
    if rows_pre:
        summary["pre_contact_max_e_hole"] = max(
            r["e_hole"] for r in rows_pre)
        summary["pre_contact_max_e_outer"] = max(
            r["e_outer"] for r in rows_pre)
        summary["pre_contact_max_e_disk"] = max(
            r["e_disk"] for r in rows_pre)
    if rows_post:
        tail = rows_post[-min(50, len(rows_post)):]
        summary["post_contact_e_eq_tail_mean"] = (
            sum(r["e_eq"] for r in tail) / len(tail))
        summary["post_contact_perimeter_rel_tail"] = (
            sum(r["perimeter_rel"] for r in tail) / len(tail))
        summary["post_contact_final_n_clusters"] = (
            rows_post[-1]["n_clusters"])
    print(json.dumps(summary, indent=1))
    outp = Path(path).with_suffix(".summary.json")
    outp.write_text(json.dumps(dict(summary=summary, pre=rows_pre,
                                    post=rows_post), indent=1))
    print(f"wrote {outp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--metric", default="grid", choices=("grid", "bie"),
                    help="Wasserstein metric backend: grid Poisson "
                         "(H^2 cells, fill epsilon) or block-diagonal "
                         "boundary integral (grid-free)")
    ap.add_argument("--bridge-gap", type=float, default=1.0,
                    help="BIE only: bulk components merge into one "
                         "metric component when two loops come within "
                         "bridge_gap * ell (ell = median mass)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--vertical", action="store_true")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--advect", action="store_true")
    ap.add_argument("--freeze-dead", action="store_true")
    ap.add_argument("--mass-estimator", default="kde",
                    choices=("kde", "oriented_kde",
                             "loopwise_oriented_kde"),
                    help="IIIc-0D estimator flag; loopwise_oriented_"
                         "kde = 0K (same-certified-loop mass mask)")
    ap.add_argument("--endgame", action="store_true",
                    help="annihilation-endgame run: the pre-contact "
                         "PROTOCOL stops (gap <= 2 h_wall, raw/vis "
                         "divergence) become recorded events; every "
                         "fail-closed GUARD (inactive-RHS, incidence "
                         "certificates, eps_flux, pump signature, "
                         "shape stops) stays active and unchanged")
    ap.add_argument("--q-mode", default="full",
                    choices=("full", "loop_mean", "self_renormalized"),
                    help="0J: perimeter-energy coherence mode")
    ap.add_argument("--angle-scope", default="union",
                    choices=("union", "loopwise"))
    ap.add_argument("--angle-measure", default="visible_full",
                    choices=("visible_full", "raw_loopwise"))
    ap.add_argument("--angle-consistency", default="none",
                    choices=("none", "mean_zero"))
    ap.add_argument("--redist-scope", default="global",
                    choices=("global", "loopwise"))
    ap.add_argument("--redist-tangent", default="angles",
                    choices=("angles", "local_pca"))
    ap.add_argument("--redist-curvature", default="union",
                    choices=("union", "loopwise"))
    ap.add_argument("--redist-q-policy", default="stale_full",
                    choices=("stale_full", "r_loop", "q_wb"))
    ap.add_argument("--redist-retraction", action="store_true")
    ap.add_argument("--redist-monotone", action="store_true",
                    help="0N: per-loop active set + monotone "
                         "backtracking + stage-total cap")
    ap.add_argument("--redist-monotone-parts", default="abc",
                    help="0N-2 diagnostic subset of {a,b,c}")
    ap.add_argument("--repair-normals-on-resume",
                    action="store_true",
                    help="one-shot boundary-normal repair applied to "
                         "the RESUME state (never during dynamics)")
    ap.add_argument("--bulk-functional", default="current",
                    choices=("current", "geometric_area"),
                    help="0L-A: physical bulk-row volume functional")
    ap.add_argument("--volume-target-functional", default="current",
                    choices=("current", "geometric_area"),
                    help="0L-A: phase-filling volume target "
                         "functional")
    ap.add_argument("--no-redistribute", action="store_true",
                    help="0G-0a attribution: disable the "
                         "redistribution stage entirely")
    ap.add_argument("--bulk-rows", action="store_true",
                    help="0F: grid_bulk_rows_mode='augment' + "
                         "incidence diagnostics + pre-contact stops")
    ap.add_argument("--e-disk", type=float, default=None,
                    help="0F-cal isolated-disk |Rdot| envelope")
    ap.add_argument("--e-ann", type=float, default=None,
                    help="0F-cal annulus ODE-residual envelope (p95)")
    ap.add_argument("--stop-after-contact", type=int, default=None,
                    help="event-driven stop: run this many steps past "
                         "geometric contact (gap <= contact-eps)")
    ap.add_argument("--contact-eps", type=float, default=0.01,
                    help="geometric-contact threshold on the "
                         "mass-weighted sheet gap")
    ap.add_argument("--resume", type=int, default=None,
                    help="restart from this checkpoint step")
    ap.add_argument("--fill-epsilon", type=float, default=0.04,
                    help="0M-4: diffuse wall width of the phase grid")
    ap.add_argument("--resume-rotate-deg", type=float, default=0.0,
                    help="0M-1: rotate the resume state (deg, about "
                         "the origin) relative to the fixed grid")
    ap.add_argument("--resume-shift", type=float, nargs=2,
                    default=(0.0, 0.0), metavar=("DX", "DY"),
                    help="0M-1: subcell shift of the resume state")
    ap.add_argument("--resume-jitter", type=float, default=0.0,
                    help="0M-1: iid position jitter amplitude on the "
                         "resume state")
    ap.add_argument("--resume-jitter-seed", type=int, default=0)
    ap.add_argument("--contact-rows-mode", default="aligned_prequotient",
                    choices=("stack", "aligned_prequotient",
                             "bulk_projected", "grid_only"),
                    help="0L-B0 cells A-D: admissible rows inside the "
                         "contact layer (b_finer branch only)")
    ap.add_argument("--gauge-hold", action="store_true",
                    help="0L-B0 cell E (diagnostic): bulk-seeded "
                         "two-component gauge/deflation hold")
    ap.add_argument("--save-steps", type=int, nargs="*", default=None,
                    help="extra checkpoint steps (besides every 25)")
    ap.add_argument("--optimizer-max-iter", type=int, default=None,
                    help="B2 stability audit: override the MM optimizer "
                         "iteration budget (default: production 300)")
    ap.add_argument("--optimizer-tol", type=float, default=None,
                    help="B2 stability audit: override optimizer_tol")
    ap.add_argument("--compression", default="off",
                    choices=["off", "atomic"],
                    help="Comp-4 atomic ghost-compression transaction "
                         "at the eligibility event ('off' keeps the "
                         "legacy control flow verbatim)")
    ap.add_argument("--quotient-mode", default="off",
                    choices=("off", "certificate_shadow"),
                    help="0L-B2: certificate-gated shadow quotient switch")
    ap.add_argument("--states-from", default=None,
                    help="tag whose _states.pt supplies the resume "
                         "checkpoint (default: own tag; gate-empty "
                         "prefixes are bitwise-shared across stacks)")
    ap.add_argument("--postprocess", default=None)
    args = ap.parse_args()
    if args.postprocess:
        postprocess(args.postprocess)
    else:
        run(args)


if __name__ == "__main__":
    main()
