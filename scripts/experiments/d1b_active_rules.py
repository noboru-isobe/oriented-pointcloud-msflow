"""D1b: the three dead-point rules ACTIVE in the solver.

Two parts, per review:

  D1b-R  exact radial wiring audit. Starting from a prescribed
         near-threshold annulus snapshot (R_- ~ 0.145, just above the
         rho = 0.3 crossing), each rule runs ACTIVELY. By symmetry the
         whole inner ring deletes in one event, so this part checks
         WIRING, not feedback: the active event step must equal the
         rule's own shadow prediction; IDs/tombstones must record the
         inner loop's disappearance; the geometric area jump must be
         ~ pi R_-^2 (the set interpretation fills the hole instantly --
         a finite-resolution topology post-processing event, NOT
         continuum closure). The scientific trajectory ENDS at the
         deletion; the post-event single outer loop is recorded, and the
         relative lag of the legacy event (one step, from D1a) is
         confirmed in the active setting.

  D1b-P  progressive-deletion stress. The inner ring of the SAME
         near-threshold snapshot carries a deterministic k = 2
         perturbation R_-(theta) = R_- (1 + eps cos 2 theta), so scores
         vary along the ring and the rules can genuinely diverge:
         different masks feed back into carrier, coherence and
         geometry. First/half/all-inner deletion are ALGORITHMIC event
         steps; after the first partial deletion the support is
         open_after_deletion, the scientific MS track ends, and the
         continuation is an explicitly invalid-support stress run
         tracked via the timeline's support states.

CLAIM SCOPE (review, D1b acceptance):
  - The exact one-step statement e_lag^{n+1} == e_ref^n applies only to
    the FIRST legacy/refreshed event, evaluated on the common
    pre-divergence trajectory. Later event steps (the 31/32 pair in the
    P runs) occur AFTER the trajectories have diverged through feedback
    and are stress coincidences, not consequences of the index-shift
    identity.
  - semi_implicit is NOT a time shift of refreshed: on the common valid
    pre-deletion geometry of the P fixture it selects a genuinely
    different mask (24 vs 28 removed) at the same step. The
    `cross_rule` section pins the differing IDs with their margins.
  - Everything after the first partial deletion in D1b-P is
    invalid-support stress data (sticky requires_reconstruction), not a
    branching of three valid discrete MS trajectories.
  - eps normalization: r = R(1 + eps cos 2 theta) inflates the enclosed
    area by (1 + eps^2/2) relative to the radial control -- 4.5e-4 at
    eps = 0.03, immaterial here. Future eps SWEEPS should use
    r = R (1 + eps cos 2 theta) / sqrt(1 + eps^2/2) to equalize areas.

eta_lag = |Rdot| dt / R_- is recorded (at these parameters ~5e-3, far
below the 0.1 gate the joint (N, dt) refinement would control).

Usage
-----
    uv run python scripts/experiments/d1b_active_rules.py \
        --out results/d1b_active
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from src.torch.diagnostics import SupportTimeline
from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import compute_recommended_params
from src.torch.shapes.generator import (
    annulus_point_counts,
    generate_oriented_annulus,
)
from src.torch.solver.dead_point_rules import (
    DEAD_POINT_RULES,
    evaluate_dead_points,
)
from src.torch.solver.mm_solver import MMSolver

sys.path.insert(0, str(Path(__file__).parent))
from d1a_mm_shadow import make_config  # noqa: E402
from d1a_ode_calibration import (  # noqa: E402
    _SOL,
    _T_STAR,
    ode_radii,
    rdot_inner,
)

DT_TENSOR = torch.float64
RHO = 0.3
DT = 2e-5
N_OUTER = 128
R_START = 0.145      # just above the rho=0.3 crossing (~0.1377)


def t_at_inner_radius(R_target: float) -> float:
    lo, hi = 0.0, 0.999 * _T_STAR
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ode_radii(mid)[1] > R_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def start_cloud(eps: float = 0.0):
    """Near-threshold annulus; optional k=2 inner perturbation with
    exact tangent-derived normals (inward = outward normal of E)."""
    n_out, n_in = annulus_point_counts(N_OUTER, 1.0, 0.5)
    t0 = t_at_inner_radius(R_START)
    Rp, Rm = ode_radii(t0)
    base = generate_oriented_annulus(n_out, Rp, Rm, (0.0, 0.0), "cpu",
                                     DT_TENSOR, n_inner=n_in)
    if eps == 0.0:
        return base, n_out, n_in
    th = (torch.arange(n_in, dtype=DT_TENSOR) + 0.5) / n_in * 2 * math.pi
    r = Rm * (1 + eps * torch.cos(2 * th))
    P = torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)
    tang = P.roll(-1, 0) - P.roll(1, 0)
    # inner ring: normals point INTO the hole (E's outward normal);
    # the inner ring is stored CCW, so that is the -90 deg rotation
    # flipped: n = (-t_y, t_x)/|t| points inward for CCW
    ang = torch.atan2(tang[:, 0], -tang[:, 1])
    positions = torch.cat([base.positions[:n_out], P])
    angles = torch.cat([base.angles[:n_out], ang])
    return (OrientedPointCloudVarifold(positions=positions, angles=angles),
            n_out, n_in)


def run_rule(rule: str, eps: float, n_steps: int, delta, tau,
             return_timeline: bool = False):
    v, n_out, n_in = start_cloud(eps)
    cfg = make_config(DT, delta, tau)
    cfg.remove_dead_points = True
    cfg.dead_point_threshold = RHO
    cfg.dead_point_rule = rule
    solver = MMSolver(cfg)
    solver.support_timeline = SupportTimeline(
        initial_loops=[torch.arange(n_out),
                       torch.arange(n_out, n_out + n_in)])
    hist = solver.solve(v, n_steps)
    tl = solver.support_timeline

    def _rec(level, step):
        return next(r for r in tl.records
                    if r.level == level and r.step == step)

    events = []
    for r in tl.records:
        if r.level == "post_deletion_raw" and r.n_removed > 0:
            cand = _rec("post_mm_candidate", r.step)
            deleted = [pid for pid, k in zip(cand.particle_ids,
                                             r.keep_mask) if not k]
            events.append(dict(
                step=r.step, n_removed=r.n_removed,
                survivors_inner=r.loop_survivor_counts[1],
                vanished=list(r.vanished_loop_ids),
                deleted_ids=deleted,
                state=(r.report.state if r.report else None)))
    # geometric area around the first event (pre = committed final of
    # the previous step; post = post_deletion_final of the event step)
    area_jump = None
    hole_area_pre = None
    if events:
        e0 = events[0]["step"]
        pre = _rec("post_mm_candidate", e0)
        post = _rec("post_deletion_final", e0)
        if pre.report and post.report:
            area_jump = (sum(post.report.areas_geom)
                         - sum(pre.report.areas_geom))
            if len(pre.report.areas_geom) > 1:
                hole_area_pre = pre.report.areas_geom[1]
    # shadow prediction on THIS rule's own trajectory
    shadow_first = None
    for r in tl.records:
        if r.level != "post_mm_candidate":
            continue
        if r.m_used is None:
            continue
        vf = OrientedPointCloudVarifold(positions=r.positions,
                                        angles=r.angles)
        dec = evaluate_dead_points(vf, r.m_used, r.q_used, rule, RHO,
                                   delta_for_kde=delta, tau_for_kde=tau,
                                   sigma=0.1)
        if dec.n_removed > 0:
            shadow_first = r.step
            break
    states = [(r.step, r.report.state if r.report else None)
              for r in tl.records if r.level == "post_deletion_final"]
    out = dict(rule=rule, eps=eps,
               n_steps_requested=n_steps,
               n_steps_completed=hist.n_completed,
               timeline_errors=[(r.step, r.level, r.error)
                                for r in tl.records if r.error],
               requires_reconstruction=tl.requires_reconstruction,
               pending_topology_events=list(tl.pending_topology_events),
               events=events, area_jump_at_first_event=area_jump,
               hole_area_pre_first_event=hole_area_pre,
               shadow_first_event=shadow_first,
               active_first_event=(events[0]["step"] if events
                                   else None),
               support_states=states)
    if return_timeline:
        return out, tl
    return out


def cross_rule_first_event(delta, tau, eps: float = 0.03,
                           n_steps: int = 11) -> dict:
    """D1b.1 item 1: the semi vs refreshed mask difference, pinned on
    the COMMON valid pre-deletion geometry. Deletion only acts after the
    MM candidate, so with the first event at step 9 every rule's
    candidate at steps <= 10 (legacy deletes at 10, after its candidate)
    is bitwise the no-deletion trajectory -- one shadow run supplies the
    common geometry for all three decisions."""
    v, n_out, n_in = start_cloud(eps)
    cfg = make_config(DT, delta, tau)          # deletion OFF
    solver = MMSolver(cfg)
    solver.support_timeline = SupportTimeline(
        initial_loops=[torch.arange(n_out),
                       torch.arange(n_out, n_out + n_in)])
    hist = solver.solve(v, n_steps)
    tl = solver.support_timeline
    # fail-closed (review): MMSolver catches internal exceptions, so the
    # shadow run must prove it actually completed before its records are
    # used as the common geometry
    if hist.n_completed != n_steps:
        raise RuntimeError(
            f"shadow run completed {hist.n_completed}/{n_steps} steps")
    errs = [(r.step, r.level, r.error) for r in tl.records if r.error]
    if errs:
        raise RuntimeError(f"timeline errors in shadow run: {errs}")

    def _dec(step, rule):
        rec = next(r for r in tl.records
                   if r.level == "post_mm_candidate" and r.step == step)
        vf = OrientedPointCloudVarifold(positions=rec.positions,
                                        angles=rec.angles)
        dec = evaluate_dead_points(vf, rec.m_used, rec.q_used, rule, RHO,
                                   delta_for_kde=delta, tau_for_kde=tau,
                                   sigma=0.1)
        return rec, dec

    rec9, ref9 = _dec(9, "refreshed")
    _, semi9 = _dec(9, "semi_implicit")
    _, lag10 = _dec(10, "legacy_lagged")

    ids9 = rec9.particle_ids

    def _removed(dec):
        return sorted(pid for pid, k in zip(ids9, dec.keep_mask.tolist())
                      if not k)

    ref_ids, semi_ids = _removed(ref9), _removed(semi9)
    lag_ids = _removed(lag10)
    diff = sorted(set(ref_ids) ^ set(semi_ids))
    pos_of = {pid: k for k, pid in enumerate(ids9)}
    P9 = rec9.positions

    def _row(dec, pid):
        k = pos_of[pid]
        return dict(score=dec.score[k].item(),
                    margin=dec.threshold_margin[k].item())

    theta = {pid: math.atan2(P9[pos_of[pid], 1].item(),
                             P9[pos_of[pid], 0].item()) for pid in diff}
    return dict(
        eps=eps, common_geometry_step=9,
        removed_refreshed=ref_ids, removed_semi=semi_ids,
        removed_legacy_step10=lag_ids,
        symmetric_difference_ids=diff,
        diff_theta=[theta[p] for p in diff],
        diff_margins_refreshed=[_row(ref9, p)["margin"] for p in diff],
        diff_margins_semi=[_row(semi9, p)["margin"] for p in diff],
        diff_scores_refreshed=[_row(ref9, p)["score"] for p in diff],
        diff_scores_semi=[_row(semi9, p)["score"] for p in diff],
        threshold_refreshed=ref9.threshold,
        threshold_semi=semi9.threshold,
        # one-step shift on the common trajectory (first event only)
        legacy10_equals_refreshed9_score=bool(
            torch.equal(lag10.score, ref9.score)),
        legacy10_equals_refreshed9_mask=bool(
            torch.equal(lag10.keep_mask, ref9.keep_mask)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/d1b_active"))
    ap.add_argument("--cross-rule-only", action="store_true",
                    help="append the P_cross_rule section to the "
                         "existing JSON without redoing the six runs")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "d1b_active_rules.json"

    if args.cross_rule_only:
        out = json.loads(path.read_text())
        n_out, n_in = out["meta"]["n_outer"], out["meta"]["n_inner"]
        delta, tau = out["meta"]["delta"], out["meta"]["tau"]
        cr = cross_rule_first_event(delta, tau)
        out["P_cross_rule"] = cr
        path.write_text(json.dumps(out, indent=1))
        print(f"removed: ref {len(cr['removed_refreshed'])} / semi "
              f"{len(cr['removed_semi'])} / legacy@10 "
              f"{len(cr['removed_legacy_step10'])}")
        print(f"diff ids: {cr['symmetric_difference_ids']}")
        print(f"diff theta: {[f'{t:+.3f}' for t in cr['diff_theta']]}")
        print(f"margins ref:  "
              f"{[f'{m:+.3e}' for m in cr['diff_margins_refreshed']]}")
        print(f"margins semi: "
              f"{[f'{m:+.3e}' for m in cr['diff_margins_semi']]}")
        print(f"legacy@10 == refreshed@9: score="
              f"{cr['legacy10_equals_refreshed9_score']} mask="
              f"{cr['legacy10_equals_refreshed9_mask']}")
        print(f"raw results -> {path}")
        return

    n_out, n_in = annulus_point_counts(N_OUTER, 1.0, 0.5)
    v_scale = generate_oriented_annulus(n_out, 1.0, 0.5, (0.0, 0.0),
                                        "cpu", DT_TENSOR, n_inner=n_in)
    delta, tau = compute_recommended_params(v_scale.positions)
    t0 = t_at_inner_radius(R_START)
    Rp0, Rm0 = ode_radii(t0)
    eta_lag = abs(rdot_inner(Rp0, Rm0)) * DT / Rm0
    out = dict(meta=dict(n_outer=n_out, n_inner=n_in, dt=DT, rho=RHO,
                         R_start=R_START, delta=delta, tau=tau,
                         t_start_ode=t0, eta_lag_start=eta_lag))
    print(f"start snapshot: t_ode={t0:.6f} R=({Rp0:.4f}, {Rm0:.4f}) "
          f"eta_lag=|Rdot| dt / R_- = {eta_lag:.2e}")

    print(f"=== D1b-R: radial active wiring (R_start={R_START}, "
          f"rho={RHO}) ===")
    for rule in DEAD_POINT_RULES:
        r = run_rule(rule, 0.0, 60, delta, tau)
        out[f"R_{rule}"] = r
        ev = r["events"][0] if r["events"] else None
        print(f"  {rule}: active event step={r['active_first_event']} "
              f"shadow={r['shadow_first_event']} "
              f"n_removed={ev['n_removed'] if ev else 0} "
              f"vanished_loops={ev['vanished'] if ev else []} "
              f"area_jump={r['area_jump_at_first_event']} "
              f"post_state={r['support_states'][-1][1]}", flush=True)

    print(f"\n=== D1b-P: progressive deletion (eps=0.03, k=2) ===")
    for rule in DEAD_POINT_RULES:
        r = run_rule(rule, 0.03, 80, delta, tau)
        out[f"P_{rule}"] = r
        evs = r["events"]
        total = sum(e["n_removed"] for e in evs)
        first = evs[0]["step"] if evs else None
        half = next((e["step"] for e in evs
                     if sum(x["n_removed"] for x in evs
                            if x["step"] <= e["step"]) >= n_in / 2), None)
        alld = next((e["step"] for e in evs
                     if e["survivors_inner"] == 0), None)
        states = sorted({s for _, s in r["support_states"] if s})
        print(f"  {rule}: events={[(e['step'], e['n_removed']) for e in evs][:6]}"
              f" first/half/all={first}/{half}/{alld} "
              f"total_removed={total} states={states}", flush=True)

    out["P_cross_rule"] = cross_rule_first_event(delta, tau)
    path.write_text(json.dumps(out, indent=1))
    print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
