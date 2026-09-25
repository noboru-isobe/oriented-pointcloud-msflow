"""0L-B2-amend-1: optimizer stability of the DROP arm at the first
certified state (768, state 1920; reviewer 2026-08-17 section 4).

The drop arm's optimizer reported converged=False after 3 iterations
while every substantive gate passed. Before the irreversible switch is
re-attempted, verify that the drop-arm committed state is insensitive
to the iteration budget: arms M (production 300) and 2M (600), plus a
tighter-tolerance arm (tol/10) as extra information, all from the SAME
source state 1920 with the SAME tentative quotient. Requirement (no
new thresholds -- the calibrated switch envelopes are reused):
    |X^(2M) - X^(M)|_inf / h_wall <= eps_switch_x,
    |theta^(2M) - theta^(M)|_inf  <= eps_switch_theta,
and identical verdicts for objective decrease, split residual,
certificate and merged geometric volume.

Usage:
    uv run python scripts/experiments/quotient_switch_stability.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import OUT, build_cloud, build_config  # noqa
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa
from src.torch.oriented_varifold.mass import compute_recommended_params  # noqa
from src.torch.solver.mm_solver import MMSolver  # noqa
from src.torch.solver.mm_step import MMStepper  # noqa
from src.torch.solver.quotient_gates import QUOTIENT_GATES  # noqa

DT = torch.float64


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def production_args(**over):
    ns = argparse.Namespace(
        grid=768, fill_epsilon=0.04, no_redistribute=False, advect=False,
        freeze_dead=False, mass_estimator="loopwise_oriented_kde",
        bulk_rows=True, bulk_functional="current",
        volume_target_functional="current",
        contact_rows_mode="aligned_prequotient", gauge_hold=False,
        quotient_mode="certificate_shadow", angle_scope="loopwise",
        angle_measure="raw_loopwise", angle_consistency="none",
        redist_scope="loopwise", redist_tangent="angles",
        redist_curvature="loopwise", redist_q_policy="r_loop",
        redist_retraction=False, redist_monotone=False,
        q_mode="self_renormalized",
        vertical=False, lam=1.0, optimizer_max_iter=None,
        optimizer_tol=None,
        # metric backend: "grid" (Poisson on H^2 cells) or "bie"
        # (block-diagonal boundary integral; bridge_gap in units of
        # the median particle mass ell)
        metric="grid", bridge_gap=1.0, cut_gap=0.08)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def drop_arm(cfg, v_src, target0, mask, tag):
    solver = MMSolver(cfg)
    st = MMStepper(cfg)
    st._grid_target_volume_initial = target0
    st.commit_quotient(mask, 1920, {"audit": tag}, (0, 1))
    res, committed = solver._advance_committed(st, v_src)
    P_src = float(res.frozen_perimeter_initial)
    P_new = st.sharp_perimeter(committed)
    certs, labels_c, m_c = st.contact_certificates_on_state(
        committed, st._contact_pair_context, st._loop_labels_last)
    V_new = solver._geometric_merged_volume(st, committed, labels_c, m_c)
    return dict(
        tag=tag, n_iter=int(res.n_iter), converged=bool(res.converged),
        objective=float(res.objective),
        objective_initial=float(res.objective_initial),
        wasserstein=float(res.wasserstein),
        split_residual=P_new + float(res.wasserstein) - P_src,
        certified=bool(certs and certs[0].certified),
        failing=(list(certs[0].failing) if certs else None),
        V_merged=V_new, r_stack=st._last_grid_snapshot.r_stack,
        quotient_classes=st._last_grid_snapshot.quotient_classes,
        positions=committed.positions, angles=committed.angles,
        h_wall=float(st.fixed_masses.median()))


def main():
    torch.set_default_dtype(DT)
    v0, _ = build_cloud()
    delta, tau = map(float, compute_recommended_params(v0.positions))
    from src.torch.oriented_varifold.loopwise_mass import (
        resolve_loopwise_oriented_mass_source,
    )
    m0 = resolve_loopwise_oriented_mass_source(
        v0.positions, v0.normals, delta, tau).m_loop
    target0 = float(0.5 * (m0 * (v0.positions * v0.normals).sum(-1)).sum())
    ck = torch.load(OUT / "exact_merger_b2_st1920_states.pt",
                    weights_only=True)
    st0 = ck[1920]
    v = OrientedPointCloudVarifold(positions=st0["positions"],
                                   angles=st0["angles"])
    # the raw-bulk pair = both bulks (disk + annulus) -> whole cloud
    mask = torch.ones(v.n_points, dtype=torch.bool)
    arms = {}
    for tag, over in (("M300", dict(optimizer_max_iter=300)),
                      ("M600", dict(optimizer_max_iter=600)),
                      ("M300_tol1e-9", dict(optimizer_max_iter=300,
                                            optimizer_tol=1e-9))):
        cfg = build_config(production_args(**over), delta, tau)
        for k, key in (("grid_quotient_eps_split", "eps_split"),
                       ("grid_quotient_eps_switch_x", "eps_switch_x"),
                       ("grid_quotient_eps_switch_theta",
                        "eps_switch_theta"),
                       ("grid_quotient_eps_volume", "eps_volume")):
            setattr(cfg, k, QUOTIENT_GATES[key])
        arms[tag] = drop_arm(cfg, v, target0, mask, tag)
        a = arms[tag]
        print(f"  [{tag}] n_iter {a['n_iter']} conv {a['converged']} "
              f"obj {a['objective_initial']:.6f}->{a['objective']:.6f} "
              f"split {a['split_residual']:+.3e} cert {a['certified']} "
              f"{a['failing']} V {a['V_merged']:.6f} r_stack "
              f"{a['r_stack']} classes {a['quotient_classes']}",
              flush=True)
    base = arms["M300"]
    rep = {"arms": {k: {kk: vv for kk, vv in a.items()
                        if kk not in ("positions", "angles")}
                    for k, a in arms.items()},
           "gates": QUOTIENT_GATES, "compare": {}}
    for tag in ("M600", "M300_tol1e-9"):
        a = arms[tag]
        dx = float((a["positions"] - base["positions"]).abs().max()) \
            / base["h_wall"]
        dth = float(wrap(a["angles"] - base["angles"]).abs().max())
        same = (a["certified"] == base["certified"]
                and (a["split_residual"] <= 0) == (base["split_residual"]
                                                   <= 0)
                and (a["objective"] < a["objective_initial"])
                == (base["objective"] < base["objective_initial"]))
        dV = abs(a["V_merged"] - base["V_merged"]) / base["V_merged"]
        rep["compare"][tag] = dict(
            dX_over_h=dx, dtheta=dth, dV_rel=dV, same_verdicts=same,
            pass_x=dx <= QUOTIENT_GATES["eps_switch_x"],
            pass_theta=dth <= QUOTIENT_GATES["eps_switch_theta"],
            pass_V=dV <= QUOTIENT_GATES["eps_volume"])
        print(f"  {tag} vs M300: |dX|/h {dx:.3e} (<= "
              f"{QUOTIENT_GATES['eps_switch_x']:.3e}) |dtheta| {dth:.3e} "
              f"(<= {QUOTIENT_GATES['eps_switch_theta']:.3e}) dV {dV:.2e} "
              f"same verdicts {same}")
    Path("results/reports/phase3c0l_b2_stability.json").write_text(
        json.dumps(rep, indent=1, default=str))
    print("wrote results/reports/phase3c0l_b2_stability.json")


if __name__ == "__main__":
    main()
