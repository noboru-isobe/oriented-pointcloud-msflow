"""D3: full post-processing bundles (deletion x redistribution x
schedule), the final audit before the production choice.

Bundles (review selection -- no 3x3 sweep; post_mm_frozen +
semi_implicit dropped for lack of robust advantage in D1/D2):

    L-E   legacy_hybrid + every_step        + legacy_lagged   (baseline)
    L-A   legacy_hybrid + adaptive_cv_trigger + legacy_lagged (efficiency)
    L-F   legacy_hybrid + fixed_physical_interval + legacy_lagged
    R-E   substep_refreshed + every_step    + refreshed
          -- a REFRESHED-TIME-LEVEL / RAW-CURVATURE sensitivity bundle,
          NOT the exact manuscript Algorithm-3 implementation (that
          would pair visible q*m curvature weights, available only as
          the audit knob curvature_visible_qm).

Event time levels (review closure patch 2): deletion acts on the
POST-REDISTRIBUTION geometry (the per-step redistribution runs between
the MM candidate and the deletion), so every event row uses
    pre   = post_redistribution
    raw   = post_deletion_raw       (deletion-only jump = raw - pre)
    final = post_deletion_final     (post-removal-redistribution jump
                                     = final - raw)
Polygon-area jumps are reported ONLY for whole-loop (closed-support)
events; partial/open deletions carry area_jump = None.

Certificates here are POST-HOC AUDIT certificates: the solver does not
yet consume gated_permissions, so a clean certificate documents that
the event WOULD certify -- certificate-gated continuation belongs to
the production-integration stage.

Tracks:
    D3-R  radial near-threshold annulus (R_- = 0.145, rho = 0.3): the
          whole inner ring deletes in one event -- wiring/event-timing
          comparison; whole-loop extinction certificate; the scientific
          trajectory continues on the certified single loop.
    D3-P  k = 2 perturbed inner ring (eps = 0.03): the FIRST partial
          deletion is the scientific endpoint (open support after);
          deleted-ID sets, threshold margins and pre/post-event density
          CV compared across bundles; everything beyond the first event
          is stress data.

Per run: first/all deletion steps, R_- just before deletion, deletion
score margins (from the raw-record payload), geometric area jump,
regular vs post-removal redistribution invocation counts, trigger
diagnostics, support states, stop reason.

Usage
-----
    uv run python scripts/experiments/d3_bundles.py \
        --out results/d3_bundles
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from src.torch.diagnostics import SupportTimeline
from src.torch.oriented_varifold.mass import compute_recommended_params
from src.torch.solver.mm_solver import MMSolver

sys.path.insert(0, str(Path(__file__).parent))
from d1a_mm_shadow import make_config  # noqa: E402
from d1b_active_rules import start_cloud  # noqa: E402
from d2b0_timelevel_shadow import snapshot  # noqa: E402

RHO = 0.3
DT = 2e-5

BUNDLES = dict(
    LE=dict(redistribution_rule="legacy_hybrid",
            redistribution_schedule="every_step",
            dead_point_rule="legacy_lagged"),
    LA=dict(redistribution_rule="legacy_hybrid",
            redistribution_schedule="adaptive_cv_trigger",
            redistribute_trigger_cv=0.05,
            dead_point_rule="legacy_lagged"),
    LF=dict(redistribution_rule="legacy_hybrid",
            redistribution_schedule="fixed_physical_interval",
            redistribute_physical_interval=1e-4,
            dead_point_rule="legacy_lagged"),
    RE=dict(redistribution_rule="substep_refreshed",
            redistribution_schedule="every_step",
            dead_point_rule="refreshed"),
)


def run_bundle(bundle: str, eps: float, n_steps: int, delta, tau):
    v, n_out, n_in = start_cloud(eps)
    cfg = make_config(DT, delta, tau)
    cfg.redistribute = True
    cfg.remove_dead_points = True
    cfg.dead_point_threshold = RHO
    cfg.redistribute_n_iters_after_removal = 3
    for k, val in BUNDLES[bundle].items():
        setattr(cfg, k, val)
    solver = MMSolver(cfg)
    solver.support_timeline = SupportTimeline(
        initial_loops=[torch.arange(n_out),
                       torch.arange(n_out, n_out + n_in)])
    hist = solver.solve(v, n_steps)
    tl = solver.support_timeline

    def _rec(level, step):
        return next((r for r in tl.records
                     if r.level == level and r.step == step), None)

    events = []
    for r in tl.records:
        if r.level == "post_deletion_raw" and r.n_removed > 0:
            pre = _rec("post_redistribution", r.step)
            fin = _rec("post_deletion_final", r.step)
            deleted = [pid for pid, k in zip(pre.particle_ids,
                                             r.keep_mask) if not k]
            margins = r.rank_fields["dead_point_margin"]
            kept = torch.tensor(r.keep_mask)
            whole_loop = r.loop_survivor_counts[1] == 0
            ev = dict(step=r.step, n_removed=r.n_removed,
                      survivors_inner=r.loop_survivor_counts[1],
                      deleted_ids=deleted,
                      R_in_pre=pre.positions[n_out:]
                      .norm(dim=1).mean().item(),
                      margin_min_removed=margins[~kept].max().item(),
                      margin_min_kept=(margins[kept].min().item()
                                       if kept.any() else None),
                      state=(r.report.state if r.report else None))
            # polygon-area jumps only for whole-loop (closed-support)
            # events; partial deletion leaves an open remnant where the
            # polygon area is not meaningful
            if whole_loop and pre.report and r.report and fin.report:
                ev["area_jump_deletion"] = (
                    sum(r.report.areas_geom)
                    - sum(pre.report.areas_geom))
                ev["area_jump_post_removal_redist"] = (
                    sum(fin.report.areas_geom)
                    - sum(r.report.areas_geom))
            else:
                ev["area_jump_deletion"] = None
                ev["area_jump_post_removal_redist"] = None
            events.append(ev)

    # invocation counts, split regular vs post-removal (the latter is
    # inferred from deletion events with n_iters_after_removal > 0)
    reg_inv = sum(1 for r in tl.records
                  if r.level == "post_redistribution"
                  and r.stage_executed)
    post_removal_inv = sum(1 for e in events if e["n_removed"] > 0)

    # whole-loop extinction certificate on the radial track
    cert = None
    if events and events[0]["survivors_inner"] == 0:
        try:
            c = tl.certify_and_accept_loop_extinction(
                events[0]["step"], phase_rank_before=1,
                phase_rank_after=1)
            cert = dict(loop_index=c.loop_index, n_removed=c.n_removed,
                        hole_area_pre=c.hole_area_pre,
                        area_jump=c.area_jump)
        except ValueError as e:
            cert = dict(error=str(e))

    states = [(r.step, r.report.state if r.report else None)
              for r in tl.records if r.level == "post_deletion_final"]
    return dict(
        bundle=bundle, eps=eps, **BUNDLES[bundle],
        n_steps_requested=n_steps, n_steps_completed=hist.n_completed,
        stop_reason=(dict(step=hist.stop_step, stage=hist.stop_stage,
                          exception=hist.stop_exception_type)
                     if hist.stop_step is not None else None),
        timeline_errors=[(r.step, r.level, r.error)
                         for r in tl.records if r.error],
        regular_redist_invocations=reg_inv,
        post_removal_redist_events=post_removal_inv,
        events=events,
        first_event=(events[0]["step"] if events else None),
        extinction_certificate=cert,
        requires_reconstruction=tl.requires_reconstruction,
        pending_topology_events=list(tl.pending_topology_events),
        support_states_tail=states[-3:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/d3_bundles"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    v0, _, _ = snapshot(0.5)
    delta, tau = compute_recommended_params(v0.positions)
    out = dict(meta=dict(rho=RHO, dt=DT, delta=delta, tau=tau))

    print("=== D3-R: radial whole-loop extinction (R_-=0.145) ===")
    for b in BUNDLES:
        r = run_bundle(b, 0.0, 25, delta, tau)
        out[f"R_{b}"] = r
        ev = r["events"][0] if r["events"] else None
        print(f"  {b}: first={r['first_event']} "
              f"n_removed={ev['n_removed'] if ev else 0} "
              f"R_pre={ev['R_in_pre'] if ev else None} "
              f"areaJ={ev.get('area_jump_deletion') if ev else None} "
              f"cert={'ok' if r['extinction_certificate'] and 'error' not in (r['extinction_certificate'] or {}) else r['extinction_certificate']} "
              f"reg_inv={r['regular_redist_invocations']} "
              f"pr_inv={r['post_removal_redist_events']} "
              f"errs={len(r['timeline_errors'])}", flush=True)

    print("=== D3-P: k=2 partial deletion (eps=0.03; first event = "
          "scientific endpoint) ===")
    for b in BUNDLES:
        r = run_bundle(b, 0.03, 14, delta, tau)
        out[f"P_{b}"] = r
        ev = r["events"][0] if r["events"] else None
        print(f"  {b}: first={r['first_event']} "
              f"n_removed={ev['n_removed'] if ev else 0} "
              f"margin_rm={ev['margin_min_removed'] if ev else None} "
              f"margin_kp={ev['margin_min_kept'] if ev else None} "
              f"sticky={r['requires_reconstruction']} "
              f"errs={len(r['timeline_errors'])}", flush=True)
    # cross-bundle mask comparison at first event
    ids = {b: set(out[f"P_{b}"]["events"][0]["deleted_ids"])
           for b in BUNDLES if out[f"P_{b}"]["events"]}
    if "LE" in ids:
        for b in ids:
            if b != "LE":
                out[f"P_maskdiff_LE_{b}"] = sorted(ids["LE"] ^ ids[b])
                print(f"  mask diff LE vs {b}: "
                      f"{len(ids['LE'] ^ ids[b])} ids")

    path = args.out / "d3_bundles.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
