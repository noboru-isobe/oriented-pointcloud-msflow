"""S0/S1: certified arc splice at the post-quotient state (reviewer
2026-08-24 GO): static splice + audit, then the atomic committed
one-step keep vs splice.

  X          = l1q 768 state 2014 (g ~ h, certified window 23/23)
  X_keep+    = Advance(X)            [CC+rows+committed quotient]
  X_splice0  = S(X)                  [arc_splice: 2 cycles -> 1 cycle]
  X_splice+  = Advance(X_splice0)    [CC+rows, single loop]

Event gates (S0): single loop (resolver + certified order) / simple /
orientation / |dA|,|dM| <= eps_alg / P#_new <= P#_pre / spacing /
transversality / unique crosswise pairing.
One-step gates (S1): no exception / objective decrease / n_iter cap /
split residual (drop-style vs keep) / merged area & first moment
one-step (drop-style vs keep with the L-A floors) / post-step simple,
single grid component, single resolved loop / angle-impulse scale.
NOTE (reviewer): keep-vs-splice |dX| is NOT a gate -- the splice is an
intentional zero-time representation change.

Usage:
    uv run python scripts/experiments/l4_splice_shadow.py \
        --states-tag _l1q_768 --step 2014 --tag _s1_768
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
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.solver.arc_splice import ArcSpliceError, arc_splice  # noqa
from src.torch.solver.mm_solver import MMSolver  # noqa: E402
from src.torch.solver.mm_step import MMStepper  # noqa: E402

R = Path("results/two_ellipses")


def h_window(pos, m1, m, gamma=1.5, abs_gap=None):
    """Contiguous cyclic windows: gamma*h threshold (the calibrated
    L1 certificate window) or an ABSOLUTE gap threshold (the CUT
    window). Design lesson (S2 probe, both dt): cutting only the
    1.5h window leaves slot walls at 0.034-0.1 apart -- a channel the
    GRID cannot resolve (it bridged at the certified b_finer scale
    1.35 eps_fill = 0.054 at step 1440), so the rhs of the remaining
    walls overflows the active support (2e-2, dt-independent). The
    cut must remove exactly the arcs the grid cannot separate: cut
    window = {cross-gap <= 1.35 eps_fill}."""
    pa, pb = pos[m1], pos[~m1]
    D = torch.cdist(pa, pb)
    h_ab = max(float(m[m1].median()), float(m[~m1].median()))
    thr = abs_gap if abs_gap is not None else gamma * h_ab
    wa = D.min(dim=1).values <= thr
    wb = D.min(dim=0).values <= thr

    def cyc(w, base):
        idx = w.nonzero().flatten()
        n = w.shape[0]
        starts = [int(i) for i in idx if not bool(w[(int(i) - 1) % n])]
        assert len(starts) == 1, "split window"
        return base + torch.tensor(
            [(starts[0] + k) % n for k in range(int(w.sum()))],
            dtype=torch.long)
    n1 = int(m1.sum())
    return cyc(wa, 0), cyc(wb, n1), h_ab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-tag", default="_l1q_768")
    ap.add_argument("--step", type=int, default=2014)
    ap.add_argument("--grid", type=int, default=768)
    ap.add_argument("--dt", type=float, default=1e-5)
    ap.add_argument("--tag", default="_s1_768")
    ap.add_argument("--moment-rows-form", default="polygon",
                    choices=["polygon", "current", "current_centered"])
    ap.add_argument("--cut-gap", type=float, default=1.35 * 0.04,
                    help="cut-window gap threshold; default = the "
                         "certified b_finer/E4 grid-bridging scale "
                         "1.35 * fill_epsilon (pre-registered, L0)")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    v0, m1, n_el, _ = teb.build_cloud(0.0)
    delta, tau = map(float, compute_recommended_params(v0.positions))
    ck = torch.load(R / f"two_ellipses{args.states_tag}_states.pt",
                    weights_only=True)
    st_src = ck[args.step]
    act = st_src["activation_state"]
    quo = st_src["quotient_state"]
    target0 = float(act["target_volume_initial"])
    pos, ang = st_src["positions"].clone(), st_src["angles"].clone()
    labels = (~m1).long()
    nor = torch.stack([ang.cos(), ang.sin()], 1)
    m = teb.resolve_m(pos, nor, delta, tau)

    def make_cfg(q_mode="contact_complex_renormalized"):
        # post-splice branch is WB (== full coherence on a single
        # loop, r == 1): the splice consumes the contact complex and
        # the CC scope certificate CORRECTLY refuses the spliced neck
        # (same-loop sigma-proximity at arc distance > 2 sigma) --
        # recorded as part of the event semantics, not bypassed
        c = teb.build_config(teb.production_args(
            grid=args.grid, redist_monotone=True, quotient_mode="off",
            q_mode=q_mode), delta, tau)
        c.time_step = args.dt
        c.grid_bulk_first_moment_rows = True
        c.grid_bulk_first_moment_rows_form = args.moment_rows_form
        return c

    rec = dict(step=args.step, source_tag=args.states_tag,
               type="arc_splice", gates={}, s0={})
    out_json = R / f"two_ellipses{args.tag}_splice.json"

    def dump(err=None):
        rec["error"] = err
        out_json.write_text(json.dumps(
            {k: v for k, v in rec.items()}, indent=1, default=str))

    # ================= S0: static splice =========================
    # certificate window (calibrated 1.5h core) -- must be certified
    w1c, w2c, h_ab = h_window(pos, m1, m, gamma=1.5)
    # cut window (grid-bridging scale)
    w1, w2, _ = h_window(pos, m1, m, abs_gap=args.cut_gap)
    rec["s0"]["window"] = dict(
        n_cert=(int(w1c.shape[0]), int(w2c.shape[0])),
        n_cut=(int(w1.shape[0]), int(w2.shape[0])),
        h_ab=h_ab, cut_gap=args.cut_gap,
        cut_gap_provenance="1.35 * fill_epsilon (b_finer/E4)")
    try:
        sp = arc_splice(pos, ang, labels, w1, w2, h_ab)
    except ArcSpliceError as e:
        dump(f"S0 ArcSpliceError: {e}")
        print(rec["error"])
        return 1
    Xs, ANs = sp.pop("positions"), sp.pop("angles")
    prov = sp.pop("provenance")
    rec["s0"].update({k: v for k, v in sp.items()})
    print(f"S0 splice: N {pos.shape[0]} -> {Xs.shape[0]} "
          f"(removed {sp['n_removed']}, bridge {sp['n_bridge']}), "
          f"pairing {sp['pairing']}, corr residual "
          f"{sp['corr_residual']}, corr disp/h "
          f"{sp['corr_disp_over_h']:.3f}, edges/h {sp['edge_over_h']},"
          f" transversality {sp['transversality_min']:.3f}",
          flush=True)
    # production sharp perimeter, pre vs post (same functional)
    stp = MMStepper(make_cfg())
    stp._loop_labels_last = None      # fresh-stepper evaluation
    stp.fixed_coherence = None
    P_pre = stp.sharp_perimeter(OrientedPointCloudVarifold(
        positions=pos.clone(), angles=ang.clone()))
    stp2 = MMStepper(make_cfg("self_renormalized"))
    stp2._loop_labels_last = None
    stp2.fixed_coherence = None
    P_new = stp2.sharp_perimeter(OrientedPointCloudVarifold(
        positions=Xs.clone(), angles=ANs.clone()))
    rec["s0"]["P_sharp"] = dict(
        pre=P_pre, post=P_new,
        branch=dict(pre="contact_complex_renormalized",
                    post="self_renormalized (== full, single loop)"))
    rec["gates"]["s0_certificates"] = bool(sp["certified"])
    rec["gates"]["s0_perimeter_drop"] = P_new <= P_pre + 1e-9
    # resolver sees ONE loop + certified order exists
    nor_s = torch.stack([ANs.cos(), ANs.sin()], 1)
    m_s = teb.resolve_m(Xs, nor_s, delta, tau)
    from src.torch.oriented_varifold.loopwise_mass import (
        resolve_loopwise_oriented_mass_source,
    )
    lb_s = resolve_loopwise_oriented_mass_source(
        Xs, nor_s, delta, tau).loop_labels_pre
    rec["s0"]["n_loops_resolved"] = int(lb_s.unique().numel())
    rec["gates"]["s0_single_loop"] = rec["s0"]["n_loops_resolved"] == 1
    try:
        from src.torch.transport.loop_geometry import (
            certified_cyclic_order,
        )
        certified_cyclic_order(Xs, lb_s, nor_s, m_s,
                               float(m_s.median()))
        rec["gates"]["s0_certified_order"] = True
    except Exception as e:                       # noqa: BLE001
        rec["s0"]["order_error"] = str(e)[:200]
        rec["gates"]["s0_certified_order"] = False
    print(f"S0 audit: P# {P_pre:.6f} -> {P_new:.6f}, resolved loops "
          f"{rec['s0']['n_loops_resolved']}, certified order "
          f"{rec['gates']['s0_certified_order']}", flush=True)
    if not all(rec["gates"].values()):
        dump("S0 gates failed")
        print("S0 FAIL:", [k for k, v in rec["gates"].items()
                           if not v])
        return 1
    torch.save(dict(positions=Xs, angles=ANs, provenance=prov,
                    splice_record={k: v for k, v in sp.items()},
                    source=dict(tag=args.states_tag, step=args.step),
                    activation_state=act, quotient_state=quo),
               R / f"two_ellipses{args.tag}_spliced_state.pt")
    # ================= S1: atomic one-step =======================
    arms = {}
    A_pre = sp["A_pre"]
    M_pre = sp["M_pre"]
    for arm in ("keep", "splice"):
        cfg = make_cfg("contact_complex_renormalized" if arm == "keep"
                       else "self_renormalized")
        sv = MMSolver(cfg)
        st = MMStepper(cfg)
        st._grid_target_volume_initial = target0
        if arm == "keep":
            st.import_quotient_state(quo, pos.shape[0])
            v = OrientedPointCloudVarifold(positions=pos.clone(),
                                           angles=ang.clone())
        else:
            v = OrientedPointCloudVarifold(positions=Xs.clone(),
                                           angles=ANs.clone())
        try:
            res, com = sv._advance_committed(st, v)
        except Exception as ex:                  # noqa: BLE001
            rec["arms"] = arms
            dump(f"S1 {arm} arm raised: {type(ex).__name__}: "
                 f"{str(ex)[:300]}")
            print(rec["error"])
            return 1
        p = com.positions
        if arm == "keep":
            Am = sum(float(teb.signed_area_order(p[labels == lb]))
                     for lb in (0, 1))
            Mm = [sum(t) for t in zip(
                *[teb.polygon_moments_order(p[labels == lb])
                  for lb in (0, 1)])]
        else:
            from src.torch.solver.arc_splice import (
                _has_self_intersection,
                _polygon_A_M,
            )
            Aq, Mxq, Myq = _polygon_A_M(p)
            Am, Mm = float(Aq), [float(Mxq), float(Myq)]
        sn = st._last_grid_snapshot
        arms[arm] = dict(
            n_iter=int(res.n_iter), converged=bool(res.converged),
            objective=float(res.objective),
            objective_initial=float(res.objective_initial),
            wasserstein=float(res.wasserstein),
            P_sharp_committed=st.sharp_perimeter(com),
            P_sharp_src=st.sharp_perimeter(v),
            A_merged=Am, M_merged=Mm,
            dA_merged=Am - A_pre,
            dM_merged=[Mm[0] - M_pre[0], Mm[1] - M_pre[1]],
            dtheta_max=float(teb.wrap(com.angles - v.angles)
                             .abs().max()),
            dX_max_over_h=float((com.positions - v.positions)
                                .norm(dim=1).max()) / h_ab,
            n_components=(sn.n_components if sn else None),
            r_stack=getattr(sn, "r_stack", None),
            post_simple=(None if arm == "keep"
                         else not _has_self_intersection(p)))
        print(f"  [{arm}] n_iter {arms[arm]['n_iter']} conv "
              f"{arms[arm]['converged']} obj "
              f"{arms[arm]['objective_initial']:.8f}->"
              f"{arms[arm]['objective']:.8f} dA_m "
              f"{arms[arm]['dA_merged']:+.2e} dM_m "
              f"{arms[arm]['dM_merged'][0]:+.2e} dth "
              f"{arms[arm]['dtheta_max']:.2e} ncomp "
              f"{arms[arm]['n_components']} r_stack "
              f"{arms[arm]['r_stack']}", flush=True)
        if arm == "splice":
            arms[arm]["committed_positions"] = com.positions
            arms[arm]["committed_angles"] = com.angles
    rec["arms"] = {a: {k: v for k, v in arms[a].items()
                       if not k.startswith("committed_")}
                   for a in arms}
    k, s = arms["keep"], arms["splice"]
    g = rec["gates"]
    g["objective_decrease"] = all(
        a["objective"] <= a["objective_initial"] + 1e-12
        for a in (k, s))
    g["iterations_under_cap"] = all(a["n_iter"] < 300 for a in (k, s))
    split_s = s["P_sharp_committed"] + s["wasserstein"] \
        - s["P_sharp_src"]
    split_k = k["P_sharp_committed"] + k["wasserstein"] \
        - k["P_sharp_src"]
    rec["split"] = dict(keep=split_k, splice=split_s)
    g["split_residual"] = max(split_s, 0.0) <= max(
        max(split_k, 0.0), 1e-9)
    g["one_step_area"] = abs(s["dA_merged"]) <= max(
        abs(k["dA_merged"]), teb.EPS_A_FLOOR)
    g["one_step_moment"] = all(
        abs(s["dM_merged"][i]) <= max(abs(k["dM_merged"][i]),
                                      teb.EPS_M_FLOOR)
        for i in (0, 1))
    g["post_simple"] = bool(s["post_simple"])
    g["one_component"] = s["n_components"] == 1
    g["angle_impulse"] = s["dtheta_max"] <= 10 * k["dtheta_max"]
    rec["passed"] = all(g.values())
    rec["verdict"] = ("PASS -- splice commit admissible"
                      if rec["passed"] else
                      f"FAIL {[kk for kk, vv in g.items() if not vv]}")
    if rec["passed"]:
        torch.save(dict(positions=arms["splice"]["committed_positions"],
                        angles=arms["splice"]["committed_angles"]),
                   R / f"two_ellipses{args.tag}_splice_committed.pt")
    dump(None)
    print(json.dumps(dict(gates=g, split=rec["split"]), indent=1,
                     default=str))
    print(rec["verdict"])
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
