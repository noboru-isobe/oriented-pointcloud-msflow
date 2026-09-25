"""Compression-2 (reviewer 2026-08-18 GO): committed shadow one-step
keep/compress comparison from the t_comp-entry states.

From the SAME source X (768 @ 2294 / 1024 @ 2292):
    keep      full cloud + imported quotient bookkeeping, one
              production committed step (MM + redistribution)
    compress  conservative compression FIRST (atomic ghost-pair
              removal + exact-area outer projection, as in
              Compression-1 arm C), then one production committed
              step with one-bulk bookkeeping (no quotient groups)
Gates follow the B2 shadow pattern (existing calibrated envelopes,
NO new thresholds; drop-style comparisons against the keep arm):
    (R_compress)+ <= max((R_keep)+, eps_split)         split residual
    |dV_compress| <= max(|dV_keep|, eps_volume)        merged volume
    ||X_keep,outer - X_compress||_inf / h_wall <= eps_switch_x
    ||dtheta_outer||_inf <= eps_switch_theta
    keep arm's State III contact reference still certified
    any exception in either arm -> fail-closed
This is an AUDIT script: nothing is committed to a production run
here; Compression-3 (continuation) resumes from the compressed state
only if these gates pass.

Usage:
    uv run python scripts/experiments/ghost_compression_shadow.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import OUT, build_cloud, build_config  # noqa
from ghost_compression_static import loop_masks, resolve_m  # noqa
from quotient_switch_stability import production_args, wrap  # noqa
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa
from src.torch.oriented_varifold.mass import (  # noqa
    compute_recommended_params,
)
from src.torch.solver.mm_solver import MMSolver  # noqa
from src.torch.solver.mm_step import MMStepper  # noqa
from src.torch.solver.quotient_gates import QUOTIENT_GATES  # noqa

DT = torch.float64


def radial_volume(pos, nor, m):
    return float((0.5 * m * (pos * nor).sum(-1)).sum())


def arm(cfg, v_src, target0, delta, tau, quotient=None):
    solver = MMSolver(cfg)
    st = MMStepper(cfg)
    st._grid_target_volume_initial = target0
    if quotient is not None:
        st.import_quotient_state(quotient, v_src.n_points)
    m_src = resolve_m(v_src.positions, v_src.normals, delta, tau)
    V_src = radial_volume(v_src.positions, v_src.normals, m_src)
    res, committed = solver._advance_committed(st, v_src)
    P_src = float(res.frozen_perimeter_initial)
    W = float(res.wasserstein)
    P_new = st.sharp_perimeter(committed)
    m_new = resolve_m(committed.positions, committed.normals, delta, tau)
    V_new = radial_volume(committed.positions, committed.normals, m_new)
    h_wall = float(st.fixed_masses.median())
    tel = dict(
        n_points=v_src.n_points, n_iter=int(res.n_iter),
        converged=bool(res.converged),
        P_sharp_src=P_src, P_sharp_committed=P_new, wasserstein=W,
        split_residual=P_new + W - P_src,
        V_src=V_src, V_committed=V_new,
        dV_rel=(V_new - V_src) / V_src,
        max_s_over_h=float(res.displacements.abs().max()) / h_wall,
        max_dtheta=float(res.delta_angles.abs().max()),
        h_wall=h_wall,
        r_stack=st._last_grid_snapshot.r_stack,
        grid_components=int(st._last_grid_snapshot.n_components),
        phase_grid_volume=float(st._last_grid_snapshot.phase_grid_volume),
    )
    if quotient is not None:
        cert = st.contact_reference_on_state(committed)
        tel["reference_certified"] = bool(cert is not None
                                          and cert.certified)
        tel["reference_failing"] = list(cert.failing) if cert else None
        tel["gap90_over_h_switch"] = (cert.metrics["gap90_over_h_switch"]
                                      if cert else None)
    return tel, committed


def run_grid(tag, step, grid, delta, tau, target0):
    ck = torch.load(OUT / f"exact_merger{tag}_states.pt",
                    weights_only=True)
    stt = ck[step]
    pos, ang, quot = stt["positions"], stt["angles"], stt["quotient"]
    ma, mb, mo = loop_masks(pos, quot)
    vK = OrientedPointCloudVarifold(positions=pos.clone(),
                                    angles=ang.clone())
    m0 = resolve_m(pos, vK.normals, delta, tau)
    V_pre = radial_volume(pos, vK.normals, m0)
    V_out = radial_volume(pos[mo], vK.normals[mo], m0[mo])
    c = math.sqrt(V_pre / V_out)
    vC = OrientedPointCloudVarifold(positions=pos[mo].clone() * c,
                                    angles=ang[mo].clone())
    cfg = build_config(production_args(grid=grid, redist_monotone=True),
                       delta, tau)
    rep = dict(grid=grid, step=step, C_scale=c, V_pre=V_pre)
    try:
        rep["keep"], committed_K = arm(cfg, vK, target0, delta, tau,
                                       quotient=quot)
        rep["keep"]["exception"] = None
    except Exception as ex:                       # noqa: BLE001
        rep["keep"] = {"exception": f"{type(ex).__name__}: {ex}"}
        committed_K = None
    try:
        rep["compress"], committed_C = arm(cfg, vC, target0, delta, tau)
        rep["compress"]["exception"] = None
    except Exception as ex:                       # noqa: BLE001
        rep["compress"] = {"exception": f"{type(ex).__name__}: {ex}"}
        committed_C = None
    gates = {}
    gates["both_ran"] = (rep["keep"].get("exception") is None
                         and rep["compress"].get("exception") is None)
    if gates["both_ran"]:
        def pos_(x):
            return max(x, 0.0)
        K, C = rep["keep"], rep["compress"]
        gates["split_residual"] = (
            pos_(C["split_residual"])
            <= max(pos_(K["split_residual"]),
                   QUOTIENT_GATES["eps_split"]))
        gates["volume"] = (abs(C["dV_rel"])
                          <= max(abs(K["dV_rel"]),
                                 QUOTIENT_GATES["eps_volume"]))
        h_wall = K["h_wall"]
        dX = float((committed_K.positions[mo]
                    - committed_C.positions).abs().max()) / h_wall
        dTh = float(wrap(committed_K.angles[mo]
                         - committed_C.angles).abs().max())
        rep["dX_outer_over_h"] = dX
        rep["dtheta_outer_inf"] = dTh
        gates["switch_x"] = dX <= QUOTIENT_GATES["eps_switch_x"]
        gates["switch_theta"] = dTh <= QUOTIENT_GATES["eps_switch_theta"]
        gates["keep_reference_certified"] = bool(
            K.get("reference_certified"))
    rep["gates"] = gates
    rep["passed"] = all(gates.values()) if gates else False
    rep["thresholds"] = dict(QUOTIENT_GATES)
    if committed_C is not None:
        rep["_committed_C"] = committed_C
    return rep


def main():
    torch.set_default_dtype(DT)
    v0, _ = build_cloud()
    delta, tau = map(float, compute_recommended_params(v0.positions))
    m0 = resolve_m(v0.positions, v0.normals, delta, tau)
    target0 = float((0.5 * m0 * (v0.positions * v0.normals)
                     .sum(-1)).sum())
    out = {}
    comp_states = {}
    for tag, step, grid in (("_0n4c_768", 2294, 768),
                            ("_0n4c_1024", 2292, 1024)):
        if not (OUT / f"exact_merger{tag}_states.pt").exists():
            print(f"-- {tag} states missing, skipped")
            continue
        rep = run_grid(tag, step, grid, delta, tau, target0)
        cc = rep.pop("_committed_C", None)
        out[tag] = rep
        print(f"== {tag} step {step} (grid {grid}) passed={rep['passed']}")
        for a in ("keep", "compress"):
            d = rep[a]
            if d.get("exception"):
                print(f"  {a}: EXCEPTION {d['exception']}")
                continue
            print(f"  {a}: N {d['n_points']} iter {d['n_iter']} split_R "
                  f"{d['split_residual']:+.3e} dV_rel {d['dV_rel']:+.3e}"
                  f" max_s/h {d['max_s_over_h']:.2e} P# "
                  f"{d['P_sharp_committed']:.6f} comps "
                  f"{d['grid_components']} r_stack {d['r_stack']}"
                  + (f" ref_cert {d.get('reference_certified')}"
                     if a == "keep" else ""))
        if rep.get("gates", {}).get("both_ran"):
            print(f"  dX_outer/h {rep['dX_outer_over_h']:.3e} "
                  f"dtheta_outer {rep['dtheta_outer_inf']:.3e} gates "
                  f"{rep['gates']}")
        if cc is not None and rep["passed"]:
            comp_states[grid] = dict(step=step,
                                     positions=cc.positions.clone(),
                                     angles=cc.angles.clone())
    # persist the compressed one-step states for Compression-3 resume
    for grid, d in comp_states.items():
        p = OUT / f"exact_merger_comp_{grid}_states.pt"
        torch.save({d["step"] + 1: {"positions": d["positions"],
                                    "angles": d["angles"],
                                    "quotient": None}}, p)
        print(f"wrote {p} (step {d['step'] + 1}, N "
              f"{d['positions'].shape[0]})")
    Path("results/reports/ghost_compression_shadow.json").write_text(
        json.dumps(out, indent=1, default=str))
    print("wrote results/reports/ghost_compression_shadow.json")


if __name__ == "__main__":
    main()
