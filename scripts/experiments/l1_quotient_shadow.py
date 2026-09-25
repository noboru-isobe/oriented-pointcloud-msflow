"""L1-1: local (two-ellipses) quotient switch -- committed shadow
one-step keep vs drop from the SAME certified source (reviewer
2026-08-23 revised verdict: L1 dynamic quotient GO).

  keep = CC + first-moment rows, State II constraint space
         [G; P_rel C_A; C_M1; C_M2]
  drop = same stepper with commit_quotient(both raw bulks -> 1 class):
         State III constraint space [G; C_M1 + C_M2]
         (P_rel over one class vanishes; moment rows aggregate via
         (Q_quot (x) I_2), L0J-M implementation)

Existing gates only (no new thresholds):
  split residual     (R_drop)+ <= eps_split          [0L-B2 QUOTIENT_GATES]
  |dX_drop-dX_keep|/h <= eps_switch_x                [0L-B2]
  |dtheta|_inf        <= eps_switch_theta            [0L-B2]
  merged area         |A_drop - A_keep|/A <= eps_volume  [0L-B2 eps_volume]
  merged first moment drop-style vs keep with the L-A EPS_M floor
  post-step L1 window certificate on BOTH arms      [l1_thresholds.json]
  optimizer: no exception, objective decrease, n_iter < cap

PASS -> the irreversible 2 bulk -> 1 bulk quotient may be committed
(the driver's --commit-quotient resume path), then 100-200 steps.

Usage:
    uv run python scripts/experiments/l1_quotient_shadow.py \
        --states-tag _la3_768 --step 1826 --tag _l1s_768
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import two_ellipses_benchmark as teb  # noqa: E402
from l1_certificate_calibration import (  # noqa: E402
    OUT as L1OUT,
    l1_certificate,
    measure_geom,
    measure_raw,
)
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.solver.mm_solver import MMSolver  # noqa: E402
from src.torch.solver.mm_step import MMStepper  # noqa: E402
from src.torch.solver.quotient_gates import QUOTIENT_GATES  # noqa: E402

R = Path("results/two_ellipses")


def window_certificate(pos, ang, m1, delta, tau, thr, gamma=1.5):
    """L1 h-window extraction (cyclic canonicalization, certified
    order, no flip) + certificate; returns (rec or None, certified,
    failing)."""
    nor = torch.stack([ang.cos(), ang.sin()], 1)
    m = teb.resolve_m(pos, nor, delta, tau)
    pa, pb = pos[m1], pos[~m1]
    D = torch.cdist(pa, pb)
    h_ab = max(float(m[m1].median()), float(m[~m1].median()))
    wa = D.min(dim=1).values <= gamma * h_ab
    wb = D.min(dim=0).values <= gamma * h_ab
    if int(wa.sum()) < 3 or int(wb.sum()) < 3:
        return None, False, ["no_window"]

    def cyc(w):
        idx = w.nonzero().flatten()
        n = w.shape[0]
        starts = [int(i) for i in idx if not bool(w[(int(i) - 1) % n])]
        if len(starts) != 1:
            return None
        return torch.tensor([(starts[0] + k) % n
                             for k in range(int(w.sum()))],
                            dtype=torch.long)
    ia, ib = cyc(wa), cyc(wb)
    if ia is None or ib is None:
        return None, False, ["split_window"]
    A = (pa[ia], nor[m1][ia], m[m1][ia])
    B = (pb[ib], nor[~m1][ib], m[~m1][ib])
    rec, pair = measure_geom(A, B, h_ab)
    rec.update(measure_raw(A, B, pair, h_ab))
    rec.update(h_ab=h_ab, g_min=float(D.min()),
               g_over_h=float(D.min()) / h_ab,
               n_W=(int(wa.sum()), int(wb.sum())))
    ok, failing = l1_certificate(rec, thr)
    return rec, ok, failing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-tag", default="_la3_768")
    ap.add_argument("--step", type=int, default=1826)
    ap.add_argument("--grid", type=int, default=768)
    ap.add_argument("--rotate-deg", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=1e-5)
    ap.add_argument("--tag", default="_l1s")
    ap.add_argument("--moment-rows-form", default="polygon",
                    choices=["polygon", "current", "current_centered"])
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    thr = json.loads((L1OUT / "l1_thresholds.json").read_text())
    v0, m1, n_el, _ = teb.build_cloud(args.rotate_deg)
    delta, tau = map(float, compute_recommended_params(v0.positions))
    ck = torch.load(R / f"two_ellipses{args.states_tag}_states.pt",
                    weights_only=True)
    src = ck[args.step]
    act = next(ck[k]["activation_state"] for k in sorted(
        k for k in ck if isinstance(k, int))
        if "activation_state" in ck[k])
    target0 = float(act["target_volume_initial"])
    pos, ang = src["positions"].clone(), src["angles"].clone()
    N = pos.shape[0]
    labels = (~m1).long()

    def make_cfg():
        c = teb.build_config(teb.production_args(
            grid=args.grid, redist_monotone=True, quotient_mode="off",
            q_mode="contact_complex_renormalized"), delta, tau)
        c.time_step = args.dt
        c.grid_bulk_first_moment_rows = True
        c.grid_bulk_first_moment_rows_form = args.moment_rows_form
        return c

    # ---- source certificate (L1-0 window) ----------------------
    src_rec, src_ok, src_fail = window_certificate(pos, ang, m1, delta,
                                                   tau, thr)
    print(f"source {args.states_tag}@{args.step}: L1 certificate "
          f"{'PASS' if src_ok else 'FAIL ' + str(src_fail)} "
          f"g/h {src_rec['g_over_h']:.4f} n_W {src_rec['n_W']} "
          f"eps_anti {src_rec['eps_anti_max']:.2e} cur "
          f"{src_rec['raw_current_rel']:.2e}", flush=True)
    cert = dict(kind="L1-0 window certificate", metrics=src_rec,
                thresholds=thr, certified=src_ok, failing=src_fail,
                source_step=args.step, source_tag=args.states_tag)
    rec = dict(step=args.step, source_tag=args.states_tag,
               source_certificate=cert, gates={}, arms={})
    if not src_ok:
        rec["verdict"] = "source not certified -- no shadow"
        (R / f"two_ellipses{args.tag}_l1shadow.json").write_text(
            json.dumps(rec, indent=1, default=str))
        print(rec["verdict"])
        return 1
    # ---- arms --------------------------------------------------
    v_src = OrientedPointCloudVarifold(positions=pos.clone(),
                                       angles=ang.clone())
    out = {}
    for arm in ("keep", "drop"):
        cfg = make_cfg()
        sv = MMSolver(cfg)
        st = MMStepper(cfg)
        st._grid_target_volume_initial = target0
        if arm == "drop":
            st.commit_quotient(torch.ones(N, dtype=torch.bool),
                               args.step, cert, (0, 1))
        try:
            res, com = sv._advance_committed(st, v_src)
        except Exception as ex:                 # noqa: BLE001
            rec["arms"][arm] = dict(exception=f"{type(ex).__name__}: "
                                    f"{str(ex)[:200]}")
            rec["verdict"] = f"{arm} arm raised -- FAIL"
            (R / f"two_ellipses{args.tag}_l1shadow.json").write_text(
                json.dumps(rec, indent=1, default=str))
            print(rec["verdict"], rec["arms"][arm])
            return 1
        p = com.positions
        A = [teb.signed_area_order(p[:n_el]),
             teb.signed_area_order(p[n_el:])]
        M = [teb.polygon_moments_order(p[:n_el]),
             teb.polygon_moments_order(p[n_el:])]
        post_rec, post_ok, post_fail = window_certificate(
            p, com.angles, m1, delta, tau, thr)
        sn = st._last_grid_snapshot
        out[arm] = dict(
            pos=p, ang=com.angles,
            n_iter=int(res.n_iter), converged=bool(res.converged),
            objective=float(res.objective),
            objective_initial=float(res.objective_initial),
            wasserstein=float(res.wasserstein),
            P_sharp_committed=st.sharp_perimeter(com),
            P_sharp_src=st.sharp_perimeter(v_src),
            A=A, A_merged=A[0] + A[1],
            M=M, M_merged=(M[0][0] + M[1][0], M[0][1] + M[1][1]),
            post_cert=post_ok, post_failing=post_fail,
            post_metrics=post_rec,
            r_stack=getattr(sn, "r_stack", None),
            quotient_classes=getattr(sn, "quotient_classes", None),
            constraint_rank=getattr(sn, "constraint_rank", None),
            h_wall=float(st.fixed_masses.median()))
        print(f"  [{arm}] n_iter {out[arm]['n_iter']} conv "
              f"{out[arm]['converged']} obj "
              f"{out[arm]['objective_initial']:.8f}->"
              f"{out[arm]['objective']:.8f} r_stack {out[arm]['r_stack']}"
              f" classes {out[arm]['quotient_classes']} rank "
              f"{out[arm]['constraint_rank']} post-cert {post_ok} "
              f"{post_fail}", flush=True)
    # ---- source polygons ---------------------------------------
    A0 = [teb.signed_area_order(pos[:n_el]),
          teb.signed_area_order(pos[n_el:])]
    M0 = [teb.polygon_moments_order(pos[:n_el]),
          teb.polygon_moments_order(pos[n_el:])]
    A0m = A0[0] + A0[1]
    M0m = (M0[0][0] + M0[1][0], M0[0][1] + M0[1][1])
    k, d = out["keep"], out["drop"]
    h = k["h_wall"]
    g = rec["gates"]
    split_drop = d["P_sharp_committed"] + d["wasserstein"] - d["P_sharp_src"]
    split_keep = k["P_sharp_committed"] + k["wasserstein"] - k["P_sharp_src"]
    g["split_residual"] = max(split_drop, 0.0) <= QUOTIENT_GATES["eps_split"]
    dx = float((d["pos"] - k["pos"]).abs().max()) / h
    dth = float(teb.wrap(d["ang"] - k["ang"]).abs().max())
    g["switch_x"] = dx <= QUOTIENT_GATES["eps_switch_x"]
    g["switch_theta"] = dth <= QUOTIENT_GATES["eps_switch_theta"]
    dA_rel = abs(d["A_merged"] - k["A_merged"]) / A0m
    g["merged_area"] = dA_rel <= QUOTIENT_GATES["eps_volume"]
    dMm_drop = [d["M_merged"][i] - M0m[i] for i in (0, 1)]
    dMm_keep = [k["M_merged"][i] - M0m[i] for i in (0, 1)]
    g["merged_first_moment"] = all(
        abs(dMm_drop[i]) <= max(abs(dMm_keep[i]), teb.EPS_M_FLOOR)
        for i in (0, 1))
    g["post_cert_both"] = bool(k["post_cert"] and d["post_cert"])
    g["objective_decrease"] = all(
        a["objective"] <= a["objective_initial"] + 1e-12
        for a in (k, d))
    g["iterations_under_cap"] = all(a["n_iter"] < 300 for a in (k, d))
    g["drop_is_one_class"] = (d["quotient_classes"] is not None
                              and len(set(d["quotient_classes"])) == 1)
    rec["measures"] = dict(
        split_drop=split_drop, split_keep=split_keep,
        dX_over_h=dx, dtheta=dth, dA_merged_rel=dA_rel,
        dM_merged_drop=dMm_drop, dM_merged_keep=dMm_keep,
        dA_components_drop=[d["A"][i] - A0[i] for i in (0, 1)],
        dA_components_keep=[k["A"][i] - A0[i] for i in (0, 1)],
        gates_ref=QUOTIENT_GATES, eps_M_floor=teb.EPS_M_FLOOR,
        h_wall=h)
    rec["arms"] = {a: {kk: vv for kk, vv in out[a].items()
                       if kk not in ("pos", "ang")} for a in out}
    rec["passed"] = all(g.values())
    rec["verdict"] = ("PASS -- quotient commit admissible"
                      if rec["passed"] else
                      f"FAIL {[kk for kk, vv in g.items() if not vv]}")
    (R / f"two_ellipses{args.tag}_l1shadow.json").write_text(
        json.dumps(rec, indent=1, default=str))
    torch.save(dict(keep=dict(positions=k["pos"], angles=k["ang"]),
                    drop=dict(positions=d["pos"], angles=d["ang"])),
               R / f"two_ellipses{args.tag}_l1shadow_arms.pt")
    print(json.dumps(dict(gates=g, measures={
        kk: vv for kk, vv in rec["measures"].items()
        if kk not in ("gates_ref",)}), indent=1, default=str))
    print(rec["verdict"])
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
