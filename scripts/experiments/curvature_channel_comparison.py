"""0L-kappa1: dynamic comparison of the normal-update channel.

All arms share loopwise density + stored tangents (fixed); ONLY the
normal update varies:

    K_union    kappa^union Frenet   (current production; contact-layer
                                     ANTI-correction)
    K_loop     kappa^loop  Frenet   (first candidate: same-loop
                                     estimator + 0K loopwise masses)
    R_PCA      no Frenet + PCA re-anchoring (validated fallback
                                     Br_lsr; curvature-estimator-free
                                     projection correction)
    K_shadow   kappa^loop Frenet + PCA SHADOW certificate: the state
               follows Frenet; e_int = max angle(n^Frenet, nu^PCA) is
               recorded only (predictor-corrector reading; a full
               retraction after Frenet would make the predictor
               redundant -- reviewer)
    N          no normal update (angles reset to the input each step)

From endgame-#2 checkpoints 1300/1325, 15 steps each; 50 steps at
1325 for {K_union, K_loop, R_PCA}. Per-step: eps_n, A_disk (polygon),
R_disk, P#/R_split, dA_redist, mean |kappa_est - kappa^poly| on the
disk sheet (the arm's own estimator vs the certified-polygon discrete
curvature kappa^poly_i = alpha_i / sbar_i), CV_l, e_int (shadow).

Pre-registered decision branches (reviewer):
 1. K_loop passes (eps_n non-increasing AND Delta A within the
    Br_lsr envelope AND 50-step survival) -> production = loopwise
    density + stored tangent + loopwise-kappa; PCA re-anchoring
    demoted to shadow certificate;
 2. improves but accumulates -> PCA production with predictor
    monitoring;
 3. dynamically unstable -> Br_lsr.

Usage:
    uv run python scripts/experiments/curvature_channel_comparison.py
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
from redistribution_integrability import (  # noqa: E402
    p_sharp,
    redistribution_call,
)
from tilt_identity_audit import decompose_state, loop_roles  # noqa: E402

import redistribution_integrability as ri  # noqa: E402
import tilt_row_ablation as tra  # noqa: E402

from src.torch.math_utils.curvature import (  # noqa: E402
    compute_regularized_curvature,
)
from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.loopwise_mass import (  # noqa: E402
    resolve_loopwise_oriented_mass_source,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_masses,
    compute_masses_oriented_loopwise,
    compute_recommended_params,
)
from src.torch.solver.redistribution_rules import (  # noqa: E402
    _local_pca_tangents,
)
from src.torch.transport.loop_geometry import (  # noqa: E402
    certified_cyclic_order,
)

DT = torch.float64
STATES = (1300, 1325)
WINDOW = 15
LONG_WINDOW = 50
LONG_ARMS = ("K_union", "K_loop", "R_PCA")

ARMS = {
    # (density, tangent, retraction, curvature_scope, post)
    "K_union": ("loopwise", "angles", False, "union", None),
    "K_loop": ("loopwise", "angles", False, "loopwise", None),
    "R_PCA": ("loopwise", "angles", True, "union", None),
    "K_shadow": ("loopwise", "angles", False, "loopwise", "shadow"),
    "N_none": ("loopwise", "angles", False, "union", "reset"),
}


def kappa_poly_disk(v, labels, orders, roles):
    """Certified-polygon discrete curvature on the disk sheet:
    kappa_i = alpha_i / sbar_i (turning angle over dual length);
    exact 1/R on a uniform circle."""
    for lb, lo in orders.items():
        if roles[lb] != "disk":
            continue
        p = v.positions[lo.order]
        e = p.roll(-1, 0) - p
        ang = torch.atan2(e[:, 1], e[:, 0])
        alpha = torch.remainder(ang - ang.roll(1) + math.pi,
                                2 * math.pi) - math.pi
        sbar = 0.5 * (e.norm(dim=1) + e.roll(1, 0).norm(dim=1))
        k = alpha / sbar.clamp_min(1e-30)
        out = torch.zeros(v.n_points, dtype=DT)
        out[lo.order] = k
        return out, lo.order
    raise RuntimeError("no disk loop")


def kappa_estimator_error(v, arm_scope):
    """Mean |kappa_est - kappa_poly| on the disk sheet, using the
    arm's OWN estimator inputs."""
    res = resolve_loopwise_oriented_mass_source(
        v.positions, v.normals, DELTA, TAU)
    labels = res.loop_labels_pre
    orders = certified_cyclic_order(v.positions, labels, v.normals,
                                    res.m_loop,
                                    float(res.m_loop.median()))
    roles = loop_roles(labels, v.positions, res.m_loop)
    kp, disk_idx = kappa_poly_disk(v, labels, orders, roles)
    if arm_scope == "loopwise":
        m = compute_masses_oriented_loopwise(
            v.positions, v.normals, labels, DELTA, TAU)
        ke, _ = compute_regularized_curvature(
            v.positions, v.normals, m, epsilon=DELTA,
            loop_labels=labels)
    else:
        m = compute_masses(v.positions, DELTA, TAU, "wendland_c2")
        ke, _ = compute_regularized_curvature(
            v.positions, v.normals, m, epsilon=DELTA)
    err = float((ke[disk_idx] - kp[disk_idx]).abs().mean())
    return err, labels


def run_arm(name, spec, v_src, window):
    density, tangent, retract, curv, post = spec
    stp = ri.production_stepper("current", "current", "augment",
                                ri.TARGET0)
    v = v_src
    t0 = time.time()
    series = []
    p0, _, _ = p_sharp(v)
    for k in range(window):
        res = stp.step(v)
        vv = (res.committed_varifold
              if res.committed_varifold is not None else res.varifold)
        ang_in = vv.angles
        v_new, info, labels, orders = redistribution_call(
            stp, vv, (density, tangent, retract),
            curvature_scope=curv)
        rec = dict(k=k, converged=bool(res.converged))
        if post == "reset":
            v_new = OrientedPointCloudVarifold(
                positions=v_new.positions, angles=ang_in)
        elif post == "shadow":
            t_pca = _local_pca_tangents(
                v_new.positions, v_new.angles, stp.mass_delta,
                "wendland_c2", loop_labels=labels)
            nu = torch.stack([t_pca[:, 1], -t_pca[:, 0]], 1)
            n_fr = v_new.normals
            cosang = (n_fr * nu).sum(1).abs().clamp(0.0, 1.0)
            rec["e_int"] = float(torch.acos(cosang).max())
        kerr, _ = kappa_estimator_error(vv, curv)
        rec["kappa_err_disk"] = kerr
        p_now, rs, _ = p_sharp(v_new)
        recs, _ = decompose_state(v_new, DELTA, TAU)
        rec.update(
            P_sharp=p_now,
            R_split=p_now + float(res.wasserstein) - p0,
            eps_n_disk=recs["disk"]["eps_n"],
            A_disk=recs["disk"]["A_poly"],
            r_l=rs,
            cv_last=info["cv_history"][-1],
        )
        p0 = p_now
        series.append(rec)
        v = v_new
    out = dict(
        arm=name, window=window, wall_s=time.time() - t0,
        A_disk_final=series[-1]["A_disk"],
        dA_disk_total=series[-1]["A_disk"] - series[0]["A_disk"],
        eps_n_final=series[-1]["eps_n_disk"],
        eps_n_max=max(r["eps_n_disk"] for r in series),
        kappa_err_mean=sum(r["kappa_err_disk"] for r in series)
        / len(series),
        max_R_split_plus=max(max(r["R_split"], 0.0) for r in series),
        e_int_max=max((r.get("e_int", 0.0) for r in series),
                      default=None),
        series=series,
    )
    print(f"  [{name}] dA_disk {out['dA_disk_total']:+.5f}  eps_n "
          f"{series[0]['eps_n_disk']:.4f}->{out['eps_n_final']:.4f} "
          f"(max {out['eps_n_max']:.4f})  kappa_err "
          f"{out['kappa_err_mean']:.3f}  R_split+ "
          f"{out['max_R_split_plus']:.1e}"
          + (f"  e_int_max {out['e_int_max']:.4f}"
             if out["e_int_max"] else ""), flush=True)
    return out


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
    report = dict(delta=DELTA, tau=TAU, states={}, long_1325={})
    for step in STATES:
        st = ck[step]
        v = OrientedPointCloudVarifold(positions=st["positions"],
                                       angles=st["angles"])
        print(f"== state {step} (15-step) ==", flush=True)
        arms = {}
        for name, spec in ARMS.items():
            try:
                arms[name] = run_arm(name, spec, v, WINDOW)
            except Exception as exc:      # noqa: BLE001
                arms[name] = dict(arm=name,
                                  error=f"{type(exc).__name__}: "
                                        f"{exc}")
                print(f"  [{name}] ERROR {arms[name]['error']}",
                      flush=True)
        report["states"][step] = arms
    print("== 50-step (1325) ==", flush=True)
    st = ck[1325]
    v = OrientedPointCloudVarifold(positions=st["positions"],
                                   angles=st["angles"])
    for name in LONG_ARMS:
        try:
            report["long_1325"][name] = run_arm(name, ARMS[name], v,
                                                LONG_WINDOW)
        except Exception as exc:      # noqa: BLE001
            report["long_1325"][name] = dict(
                arm=name, error=f"{type(exc).__name__}: {exc}")
            print(f"  [{name}] ERROR "
                  f"{report['long_1325'][name]['error']}", flush=True)
    out_p = Path("results/reports/phase3c0l_kappa1.json")
    out_p.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
