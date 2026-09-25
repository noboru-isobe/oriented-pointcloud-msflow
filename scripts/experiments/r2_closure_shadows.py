"""Pre-R2 closure shadows (reviewer 2026-08-16 checklist §6–7).

(A) q_redist shadow:
  A1 algebraic pin -- one cap-free subiteration: the r_loop policy
     rescales each loop's update EXACTLY by r_l (path-preserving,
     pseudo-time only), while q^full changes the within-loop
     DIRECTION (contrast pin).
  A2 dynamic comparison at matched per-loop CV target: policies
     {stale_full, r_loop, q_wb} on states {1300, repaired-1325,
     deep clean synthetics d/sigma in {0.6, 0.3, 0.15}} --
     iterations to target, cap counts, disk-area drift (L+Q in the
     certified frame), outer-loop drift, final CV. Comparison
     accepted only cap-free or matched-CV (recorded).

(B) composed-row parity: for each state build the production stepper
    under bulk_functional in {current, geometric_area} (target
    current both), and compare the FULL composed admissible spaces
    via the kernel projector difference
        d = || Q_cur Q_cur^T - Q_geo Q_geo^T ||_2,
    max principal angle = arcsin(d); plus the one-step displacement
    and objective difference. Gate states: t=0, 1000, 1300,
    REPAIRED-1325; raw-1325 is a breakdown DIAGNOSTIC only (its
    inherited tilt is the old union-kappa's product, not the
    production candidate's).

Usage:
    uv run python scripts/experiments/r2_closure_shadows.py
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
from sigma_n_refinement import organic_reference  # noqa: E402
from sigma_n_refinement import synthetic_cell_state  # noqa: E402
from tilt_identity_audit import decompose_state, loop_roles  # noqa: E402

import redistribution_integrability as ri  # noqa: E402
import tilt_row_ablation as tra  # noqa: E402

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
    redistribute_with_rule,
    repair_boundary_normals,
)
from src.torch.transport.loop_geometry import (  # noqa: E402
    certified_cyclic_order,
    polygon_area,
)

DT = torch.float64
CV_TARGET = 0.02
POLICIES = ("stale_full", "r_loop", "q_wb")


def q_vectors(v, m, labels):
    q_full = None
    from src.torch.transport import compute_coherence
    q_full = compute_coherence(v, m, 0.1, "wendland_c2")
    q_wb_vec, rs, _ = qwb(v, m, labels)
    r_pp = torch.ones_like(q_full)
    for lb, r in rs.items():
        r_pp[labels == lb] = r
    return dict(stale_full=q_full, r_loop=r_pp, q_wb=q_wb_vec), rs


def one_redistribution(v, labels, q_vec, n_iters, tol,
                       step_size=None):
    return redistribute_with_rule(
        v.positions, v.angles, "legacy_hybrid",
        delta=DELTA, kernel="wendland_c2", n_iters=n_iters,
        step_size=(step_size if step_size is not None else 0.01),
        tol=tol, max_disp_ratio=0.05, mass_tau=TAU,
        delta_redist=DELTA * 0.5,
        coherence=q_vec, sigma=0.1,
        loop_labels=labels, density_scope="loopwise",
        curvature_scope="loopwise",
        kappa_den_beta_min=0.48, kappa_den_gamma_min=0.48)


def algebraic_pin(v, labels, qs):
    """One CAP-FREE subiteration: v^r = r_l * v^1 exactly. The
    contrast metric is the WITHIN-LOOP magnitude profile f_i =
    |u_i| / sum_loop |u_j| (a positive pointwise scalar never
    changes the pointwise unit vector, so the field-shape profile is
    the right object): r_loop leaves it invariant, q^full deforms
    it (path change over substeps)."""
    outs = {}
    for tag, q in (("unit", torch.ones_like(qs["r_loop"])),
                   ("r_loop", qs["r_loop"]),
                   ("full", qs["stale_full"])):
        p1, _, _ = one_redistribution(
            v, labels, q, n_iters=1, tol=0.0, step_size=1e-4)
        outs[tag] = p1 - v.positions
    ok = True
    prof_dev_full = 0.0
    prof_dev_r = 0.0
    for lb in labels.unique():
        sel = labels == lb
        r = float(qs["r_loop"][sel][0])
        resid = float((outs["r_loop"][sel]
                       - r * outs["unit"][sel]).abs().max())
        scale = float(outs["unit"][sel].abs().max())
        ok = ok and resid <= 1e-14 + 1e-12 * scale

        def prof(u):
            n = u[sel].norm(dim=1)
            return n / n.sum().clamp_min(1e-30)

        prof_dev_full = max(prof_dev_full, float(
            (prof(outs["full"]) - prof(outs["unit"])).abs().max()))
        prof_dev_r = max(prof_dev_r, float(
            (prof(outs["r_loop"]) - prof(outs["unit"])).abs().max()))
    return dict(r_loop_exact=bool(ok),
                r_loop_profile_dev=prof_dev_r,
                full_profile_dev=prof_dev_full)


def tangential_jitter(v, frac=0.35, seed=9):
    """Rotate each point about the origin by a small random angle:
    every point stays EXACTLY on its circle while the sampling
    density becomes nonuniform (creates a genuine equalization task
    for concentric-circle states)."""
    n = v.n_points
    r = v.positions.norm(dim=1)
    spacing_ang = 2 * math.pi / n * 3.0
    g = torch.Generator().manual_seed(seed)
    dphi = frac * spacing_ang * torch.randn(n, generator=g, dtype=DT)
    th = torch.atan2(v.positions[:, 1], v.positions[:, 0]) + dphi
    pos = torch.stack([r * th.cos(), r * th.sin()], 1)
    return OrientedPointCloudVarifold(positions=pos,
                                      angles=v.angles + dphi)


def dynamic_compare(name, v):
    res = resolve_loopwise_oriented_mass_source(
        v.positions, v.normals, DELTA, TAU)
    labels = res.loop_labels_pre
    qs, rs = q_vectors(v, res.m_loop, labels)
    orders = certified_cyclic_order(v.positions, labels, v.normals,
                                    res.m_loop,
                                    float(res.m_loop.median()))
    roles = loop_roles(labels, v.positions, res.m_loop)
    a0 = {roles[lb]: float(polygon_area(v.positions, lo.order))
          for lb, lo in orders.items()}
    # matched-CV protocol: per-state target = half the initial max
    # per-loop CV (floor 1e-3) so every arm has genuine work to do
    from src.torch.solver.redistribute import (
        _compute_density_log_gradient,
    )
    _, th0 = _compute_density_log_gradient(v.positions, DELTA * 0.5,
                                           "wendland_c2", labels)
    cv0 = max((th0[labels == lb]).std().item()
              / ((th0[labels == lb]).mean().item() + 1e-30)
              for lb in labels.unique())
    cv_target = max(0.5 * cv0, 1e-3)
    out = dict(state=name, r_by_loop=rs, cv0=cv0,
               cv_target=cv_target, arms={})
    for pol in POLICIES:
        p1, a1, info = one_redistribution(
            v, labels, qs[pol], n_iters=300, tol=cv_target)
        a_after = {}
        for lb, lo in orders.items():
            a_after[roles[lb]] = float(polygon_area(p1, lo.order))
        clips = sum(1 for t in info["trace"] if t["clip_active"])
        out["arms"][pol] = dict(
            iters=info["n_iters"], converged=info["converged"],
            cv_final=info["cv_history"][-1], clips=clips,
            dA_disk=a_after.get("disk", 0.0) - a0.get("disk", 0.0),
            dA_outer=(a_after.get("outer", a_after.get("ann", 0.0))
                      - a0.get("outer", a0.get("ann", 0.0))),
        )
        a = out["arms"][pol]
        print(f"  [{name}/{pol}] iters {a['iters']} (conv "
              f"{a['converged']}) cv {a['cv_final']:.4f} clips "
              f"{a['clips']}  dA_disk {a['dA_disk']:+.2e}  dA_outer "
              f"{a['dA_outer']:+.2e}", flush=True)
    return out


def parity_state(name, v, gate):
    """Composed-row parity (reviewer section 7): compare the FULL
    stacked row spaces of EQUAL construction,
        R_cur = row[C_grid~ ; C_cur~],  R_geo = row[C_grid~ ; C_geo~]
    with C~ = C_s + C_theta AB. One geometric-mode stepper supplies
    all three raw matrices (C_grid from the grid metric, C_cur as
    the shadow rows, C_geo as the active rows); the production
    partition-equal branch of the current mode carries NO physical
    row, so a naive production-vs-production projector diff is a
    trivial dimension mismatch, not the reviewer's metric."""
    st = ri.production_stepper("geometric_area", "current",
                               "augment", ri.TARGET0)
    st._setup_step(v)
    gm = st.grid_wasserstein
    c_grid = gm.flux.component_rows(gm.compat_labels,
                                    gm.phase.cell_area)
    c_cur = st._bulk_rows_shadow
    c_geo = st._bulk_rows_last
    AB = st.param.AB_solve
    N = v.n_points

    def composed(rows):
        return rows[:, :N] + rows[:, N:] @ AB

    def rowbasis(mat):
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        r = int((s > s.max() * 1e-10).sum())
        return vh[:r]

    Rc = rowbasis(torch.cat([composed(c_grid), composed(c_cur)]))
    Rg = rowbasis(torch.cat([composed(c_grid), composed(c_geo)]))
    d = float(torch.linalg.matrix_norm(
        Rc.T @ Rc - Rg.T @ Rg, ord=2))
    ang = math.degrees(math.asin(min(d, 1.0)))
    # one-step comparison under the two ACTIVE-row choices
    st2 = ri.production_stepper("current", "current", "augment",
                                ri.TARGET0)
    res_g = st.step(v)
    res_c = st2.step(v)
    sc = res_c.displacements.detach()
    sg = res_g.displacements.detach()
    ds = float((sc - sg).norm() / sc.norm().clamp_min(1e-30))
    dobj = abs(float(res_c.objective) - float(res_g.objective)) \
        / max(abs(float(res_c.objective)), 1e-30)
    rec = dict(state=name, gate=gate,
               rank_cur=int(Rc.shape[0]), rank_geo=int(Rg.shape[0]),
               proj_diff=d, angle_deg=ang,
               rel_disp_diff=ds, rel_obj_diff=dobj)
    print(f"  [parity {name}{' (gate)' if gate else ' (diag)'}] "
          f"ranks {rec['rank_cur']}/{rec['rank_geo']}  |dP| {d:.2e} "
          f"angle {ang:.4f} deg  rel_ds {ds:.2e}  rel_dobj "
          f"{dobj:.2e}", flush=True)
    return rec


def main():
    global DELTA, TAU
    torch.set_default_dtype(DT)
    ck = torch.load(OUT / "exact_merger_endgame2_states.pt",
                    weights_only=True)
    v0, _ = build_cloud()
    DELTA, TAU = map(float, compute_recommended_params(v0.positions))
    ri.DELTA, ri.TAU = DELTA, TAU
    tra.DELTA, tra.TAU = DELTA, TAU
    m0 = resolve_loopwise_oriented_mass_source(
        v0.positions, v0.normals, DELTA, TAU).m_loop
    ri.TARGET0 = float(0.5 * (m0 * (v0.positions
                                    * v0.normals).sum(-1)).sum())
    report = dict(delta=DELTA, tau=TAU, cv_target=CV_TARGET)

    def load(step):
        st = ck[step]
        return OrientedPointCloudVarifold(positions=st["positions"],
                                          angles=st["angles"])

    v1300, v1325 = load(1300), load(1325)
    new_ang, rep = repair_boundary_normals(
        v1325.positions, v1325.angles, DELTA, TAU, 0.1)
    v1325r = OrientedPointCloudVarifold(positions=v1325.positions,
                                        angles=new_ang)
    print(f"repaired 1325: max angle "
          f"{rep['retraction']['max_angle_rad']:.4f}  dP+rel "
          f"{rep['delta_p_plus_rel']:.2e}", flush=True)
    report["repair_1325"] = {k: v for k, v in rep.items()
                             if k != "retraction"} | {
        "max_angle_rad": rep["retraction"]["max_angle_rad"]}

    # (A) q shadow
    print("== A1 algebraic pin (1300) ==", flush=True)
    resq = resolve_loopwise_oriented_mass_source(
        v1300.positions, v1300.normals, DELTA, TAU)
    qs, _ = q_vectors(v1300, resq.m_loop, resq.loop_labels_pre)
    report["A1"] = algebraic_pin(v1300, resq.loop_labels_pre, qs)
    print(f"  r_loop exact: {report['A1']['r_loop_exact']}  "
          f"profile dev r_loop {report['A1']['r_loop_profile_dev']:.2e}"
          f" vs full {report['A1']['full_profile_dev']:.2e}",
          flush=True)

    print("== A2 dynamic (matched CV) ==", flush=True)
    R_d_star, A_ann, phi = organic_reference()
    dyn = [dynamic_compare("1300", v1300),
           dynamic_compare("1325_repaired", v1325r)]
    for dsig in (0.6, 0.3, 0.15):
        v_syn, _ = synthetic_cell_state(R_d_star, A_ann, 0.10, dsig,
                                        256, 0.0, phi, k=2)
        # tangential jitter: creates a genuine equalization task
        # (the pristine generators are already uniform, CV ~ 0)
        v_syn = tangential_jitter(v_syn)
        dyn.append(dynamic_compare(f"synth_d{dsig}", v_syn))
    report["A2"] = dyn

    # (B) parity
    print("== B composed-row parity ==", flush=True)
    par = [parity_state("t0", v0, True),
           parity_state("1000", load(1000), True),
           parity_state("1300", v1300, True),
           parity_state("1325_repaired", v1325r, True),
           parity_state("1325_raw", v1325, False)]
    report["B"] = par

    out_p = Path("results/reports/phase3c0l_r2_shadows.json")
    out_p.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
