"""0L-A2: causal decomposition of the tilt-driven volume anomaly.

Three stages (reviewer 0L section 9), states 1320 (pre-rush) and
1340 (mid-rush), one-step + 15-step windows, all arms from the SAME
source state:

A2a row mechanism (target = current everywhere):
    a1  current rows                (production control)
    a2  [C_grid; C_geom] stacked    (blocker-1 semantics)
    a3  C_geom only + projected Poisson (diagnostic: separates the
        geometric-row effect from C_grid overconstraint)
A2b target consistency (geometric rows):
    a2  geometric rows + current target      (== A2a arm 2)
    b2  geometric rows + polygon-area target (production candidate)
A2c normal integrability (geometric rows + geometric target):
    b2  angle transport as-is                (== A2b arm b2)
    c2  post-commit projection n <- nu_poly(X^{n+1}) each step

Pre-registered gates (exact, not "reduced rush"):
    DA_b[s] = 0 to machine precision on the geometric arms
    (bulk_flux_residuals), and per MM step
    |Delta A_b(candidate) - Q_{A,b}(dx)| <= eps_alg -- the polygon
    area is exactly quadratic, so a wired row leaves EXACTLY the
    second-order drift. The 15-step cumulative Delta A_b must equal
    sum Q_{A,b} + the redistribution channel (recorded separately:
    redistribution resamples vertices, which changes the polygon
    without moving the curve; it is NOT a first-order constraint
    failure).
Readings (pre-registered):
  - candidate holds  -> DA=0 + quadratic envelope + rush inside it
  - eps_n still grows under geometric conservation -> volume closure
    succeeds but angle integrability is a SEPARATE open channel (do
    not proceed to 0L-B)
  - rush persists in a3 -> the tilt x volume-row attribution is
    insufficient; persists only in a2 -> C_grid finite-discretization
    incompatibility.

Usage:
    uv run python scripts/experiments/tilt_row_ablation.py
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

from exact_merger_benchmark import DT_STEP, OUT, build_cloud  # noqa: E402
from tilt_identity_audit import decompose_state, loop_roles  # noqa: E402

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
    area_quadratic_term,
    certified_cyclic_order,
    polygon_area,
)

DT = torch.float64
# saved checkpoints are on the 25-step grid; 1300 = pre-rush and
# 1325 = rush onset (the 15-step window then covers 1326-1340, the
# rush build-up, and stays clear of the 1354 merger boundary)
STATES = (1300, 1325)
WINDOW = 15
EPS_ALG = 1e-12


def production_stepper(rows_functional, target_functional,
                       rows_mode, target0):
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
    cfg.mass_estimator = "loopwise_oriented_kde"
    cfg.grid_bulk_rows_mode = rows_mode
    cfg.grid_bulk_rows_functional = rows_functional
    cfg.grid_volume_target_functional = target_functional
    cfg.perimeter_q_mode = "self_renormalized"
    st = MMStepper(cfg)
    st._grid_target_volume_initial = target0
    return st


def bulk_areas_and_orders(v, delta, tau):
    res = resolve_loopwise_oriented_mass_source(
        v.positions, v.normals, delta, tau)
    labels = res.loop_labels_pre
    orders = certified_cyclic_order(v.positions, labels, v.normals,
                                    res.m_loop,
                                    float(res.m_loop.median()))
    roles = loop_roles(labels, v.positions, res.m_loop)
    a = {"disk": 0.0, "ann": 0.0}
    for lb, lo in orders.items():
        val = float(polygon_area(v.positions, lo.order))
        a["disk" if roles[lb] == "disk" else "ann"] += val
    return a, orders, roles, labels


def project_normals(v, delta, tau):
    """Post-commit projection arm: n <- nu_poly(X)."""
    from src.torch.transport.loop_geometry import polygon_dual
    res = resolve_loopwise_oriented_mass_source(
        v.positions, v.normals, delta, tau)
    orders = certified_cyclic_order(
        v.positions, res.loop_labels_pre, v.normals, res.m_loop,
        float(res.m_loop.median()))
    ang = v.angles.clone()
    for lo in orders.values():
        _, nu = polygon_dual(v.positions, lo.order)
        ang[lo.order] = torch.atan2(nu[:, 1], nu[:, 0])
    return OrientedPointCloudVarifold(positions=v.positions,
                                      angles=ang)


def apply_redistribution(stp, vv):
    """Replicate MMSolver's post-step redistribution EXACTLY (the
    stepper itself never redistributes -- discovering this is itself
    an A2 result: the bare MM step conserves the polygon area to
    ~1e-9/step even under current rows, so the production rush must
    enter through THIS stage). legacy_hybrid, interval 1, solver
    defaults, stale pre-MM coherence -- as in MMSolver._redistribute."""
    from src.torch.solver.redistribution_rules import (
        redistribute_with_rule,
    )
    cfg = stp.config
    new_pos, new_angles, info = redistribute_with_rule(
        vv.positions, vv.angles, cfg.redistribution_rule,
        delta=stp.mass_delta, kernel=cfg.mass_kernel,
        n_iters=cfg.redistribute_n_iters,
        step_size=cfg.redistribute_step_size,
        tol=cfg.redistribute_tol,
        max_disp_ratio=cfg.redistribute_max_disp_ratio,
        mass_tau=stp.mass_tau,
        delta_redist=stp.mass_delta * cfg.redistribute_delta_ratio,
        coherence=stp.fixed_coherence, sigma=stp._sigma,
        coherence_kernel=cfg.perimeter_kernel,
        coherence_backend=cfg.backend)
    return OrientedPointCloudVarifold(positions=new_pos,
                                      angles=new_angles)


def run_arm(name, v_src, rows_functional, target_functional,
            rows_mode, target0, project_each_step=False):
    stp = production_stepper(rows_functional, target_functional,
                             rows_mode, target0)
    v = v_src
    a0, orders0, roles0, labels0 = bulk_areas_and_orders(v, DELTA,
                                                         TAU)
    series = []
    cumQ = {"disk": 0.0, "ann": 0.0}
    t0 = time.time()
    for k in range(WINDOW):
        a_before, orders, roles, labels = bulk_areas_and_orders(
            v, DELTA, TAU)
        res = stp.step(v)
        cand = res.varifold
        dx = cand.positions.detach() - v.positions
        # exact per-bulk quadratic term + candidate-step gate
        q_b = {"disk": 0.0, "ann": 0.0}
        a_cand = {"disk": 0.0, "ann": 0.0}
        da_gate = {}
        for lb, lo in orders.items():
            key = "disk" if roles[lb] == "disk" else "ann"
            q_b[key] += float(area_quadratic_term(dx, lo.order))
            a_cand[key] += float(polygon_area(cand.positions.detach(),
                                              lo.order))
        for key in ("disk", "ann"):
            da = a_cand[key] - a_before[key]
            resid = da - q_b[key]
            if rows_functional == "geometric_area":
                # subtract the first-order part actually constrained:
                # DA_b[s] is the bulk residual (should be ~0)
                pass
            da_gate[key] = dict(dA_cand=da, Q=q_b[key],
                                lin_resid=resid)
            cumQ[key] += q_b[key]
        vv = (res.committed_varifold
              if res.committed_varifold is not None else res.varifold)
        vv = apply_redistribution(stp, vv)
        a_comm, _, _, _ = bulk_areas_and_orders(vv, DELTA, TAU)
        recs, _ = decompose_state(vv, DELTA, TAU)
        rec = dict(
            k=k,
            A_disk_committed=a_comm["disk"],
            A_ann_committed=a_comm["ann"],
            dA_redist_disk=a_comm["disk"] - a_cand["disk"],
            gate=da_gate,
            eps_n_disk=recs["disk"]["eps_n"],
            R_disk=None,
            bulk_residuals=(list(res.bulk_flux_residuals)
                            if res.bulk_flux_residuals is not None
                            else None),
            bulk_residuals_shadow=(
                list(res.bulk_flux_residuals_shadow)
                if getattr(res, "bulk_flux_residuals_shadow", None)
                is not None else None),
            converged=bool(res.converged),
        )
        m_c = resolve_loopwise_oriented_mass_source(
            vv.positions, vv.normals, DELTA, TAU).m_loop
        r = vv.positions.norm(dim=1)
        wall = r < 0.6
        outw = (vv.positions * vv.normals).sum(1) > 0
        sel = wall & outw
        rec["R_disk"] = float((m_c[sel] * r[sel]).sum()
                              / m_c[sel].sum())
        series.append(rec)
        v = vv
        if project_each_step:
            v = project_normals(v, DELTA, TAU)
    out = dict(
        arm=name, wall_s=time.time() - t0,
        A_disk_0=a0["disk"],
        A_disk_final=series[-1]["A_disk_committed"],
        dA_disk_total=series[-1]["A_disk_committed"] - a0["disk"],
        cumQ_disk=cumQ["disk"],
        eps_n_final=series[-1]["eps_n_disk"],
        R_disk_final=series[-1]["R_disk"],
        max_lin_resid=max(abs(r_["gate"]["disk"]["lin_resid"])
                          for r_ in series),
        max_bulk_resid=max(
            (max(abs(x) for x in r_["bulk_residuals"])
             if r_["bulk_residuals"] else 0.0) for r_ in series),
        series=series,
    )
    print(f"  [{name}] dA_disk {out['dA_disk_total']:+.5f} "
          f"(cumQ {out['cumQ_disk']:+.2e})  R_disk "
          f"{out['R_disk_final']:.5f}  eps_n {out['eps_n_final']:.4f}"
          f"  max|DA_b[s]| {out['max_bulk_resid']:.1e}  "
          f"max lin-resid {out['max_lin_resid']:.1e}", flush=True)
    return out


def main():
    global DELTA, TAU
    torch.set_default_dtype(DT)
    ck = torch.load(OUT / "exact_merger_endgame2_states.pt",
                    weights_only=True)
    v0, _ = build_cloud()
    DELTA, TAU = map(float, compute_recommended_params(v0.positions))
    m0 = resolve_loopwise_oriented_mass_source(
        v0.positions, v0.normals, DELTA, TAU).m_loop
    target0_cur = float(0.5 * (m0 * (v0.positions
                                     * v0.normals).sum(-1)).sum())
    a0, _, _, _ = bulk_areas_and_orders(v0, DELTA, TAU)
    target0_geom = a0["disk"] + a0["ann"]
    print(f"targets: current {target0_cur:.6f}  polygon "
          f"{target0_geom:.6f}")
    report = dict(delta=DELTA, tau=TAU, window=WINDOW,
                  target0_current=target0_cur,
                  target0_geometric=target0_geom, states={})
    for step in STATES:
        st = ck[step]
        v = OrientedPointCloudVarifold(positions=st["positions"],
                                       angles=st["angles"])
        print(f"== state {step} ==", flush=True)
        arms = {}
        arms["a1_current"] = run_arm(
            "a1 current rows", v, "current", "current", "augment",
            target0_cur)
        arms["a2_geom_stack"] = run_arm(
            "a2 geom stacked", v, "geometric_area", "current",
            "augment", target0_cur)
        arms["a3_geom_only"] = run_arm(
            "a3 geom only projected", v, "geometric_area", "current",
            "bulk_only_diagnostic", target0_cur)
        arms["b2_geom_target"] = run_arm(
            "b2 geom rows+target", v, "geometric_area",
            "geometric_area", "augment", target0_geom)
        arms["c2_geom_project"] = run_arm(
            "c2 geom+post-commit projection", v, "geometric_area",
            "geometric_area", "augment", target0_geom,
            project_each_step=True)
        report["states"][step] = arms

    out_p = Path("results/reports/phase3c0l_a2_ablation.json")
    out_p.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
