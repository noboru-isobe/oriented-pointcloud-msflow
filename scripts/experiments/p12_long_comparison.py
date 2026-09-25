"""P1.2: full-SVD long-trajectory comparison across the production
chain (F14 protocol, long-run stage).

Configurations per shape (review):
    C1    legacy exterior + legacy global projection   (published)
    C2    interior + legacy global projection          (trace fix)
    C3-A  interior + spectral full-SVD + carrier + M variant,
          UNSEEDED adaptive_state_machine
          == the PRIMARY production candidate
    C3-S  (paper pair only) same as C3-A but with the integer rank
          certificate C0=2 -- an AUDIT/SENSITIVITY track. If A refuses
          mid-run it STOPS THERE (structured stop reason); there is
          never a fallback from A to S inside one trajectory.

The support gate is a METRIC-INDEPENDENT safety layer and is applied
uniformly: gated main line for ALL kinds (C1/C2/C3), --no-gates for
the ungated stress track (reviewer, fair comparison). All kinds run
with the production post-processing bundle (redistribute + dead-point
removal): without it the y-space W_lin metric loses positive
definiteness and the comparison measures an artifact of the missing
bundle, not the metric.

Shapes: paper two-ellipses (n_per=128, N_total=256), flower (N=256),
star (N=256), ellipse 2:1 control (N=256). The N=128 C2-C2S
sensitivity found in P1.1 is a decaying finite-resolution effect, so
long runs use N >= 256 throughout.

Per step (from the SupportTimeline + rank telemetry): working rank &
action, gap/null levels, C_u row-space angle, support state, per-loop
best-fit radius/center where meaningful, pair gap for the two-ellipse
runs. Scientific windows: the pair runs report up to the support
gate / first contact-layer event; everything after is stress data.

Usage
-----
    uv run python scripts/experiments/p12_long_comparison.py \
        --out results/p12_long [--shape pair|flower|star|ellipse]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

from src.torch.diagnostics import SupportTimeline
from src.torch.oriented_varifold.mass import compute_recommended_params
from src.torch.shapes.generator import (
    generate_oriented_ellipse,
    generate_oriented_flower,
    generate_oriented_star,
    generate_oriented_two_ellipses,
)
from src.torch.solver.mm_solver import MMSolver

sys.path.insert(0, str(Path(__file__).parent))
from p1_production_comparison import make_cfg  # noqa: E402

DT = 1e-5           # paper time step (stability floor for N=64 was
                    # the published constraint; N>=256 is comfortable)
DTYPE = torch.float64


def make_shape(name, n_per=128):
    if name == "pair":
        v = generate_oriented_two_ellipses(
            n_per, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
            a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu",
            dtype=DTYPE)
        loops = [torch.arange(n_per), torch.arange(n_per, 2 * n_per)]
        return v, 2, loops
    if name == "flower":
        v = generate_oriented_flower(256, 5, 0.7, 1.3, (0.0, 0.0),
                                     "cpu", DTYPE)
    elif name == "star":
        v = generate_oriented_star(256, device="cpu", dtype=DTYPE)
    elif name == "ellipse":
        v = generate_oriented_ellipse(256, 1.0, 0.5, (0.0, 0.0),
                                      "cpu", DTYPE)
    else:
        raise ValueError(name)
    return v, 1, [torch.arange(v.n_points)]


def pair_gap(P):
    """Signed x-gap between the two sheets. Split by sign of x (robust
    to deletions changing indices), not by the initial index blocks."""
    left = P[P[:, 0] < 0.0]
    right = P[P[:, 0] >= 0.0]
    if left.numel() == 0 or right.numel() == 0:
        return float("nan")
    return (right[:, 0].min() - left[:, 0].max()).item()


def _san(x):
    """JSON-safe: SupportReport is already scalarized (Explore audit);
    only nan/inf floats need mapping (-> str) plus tensor stragglers."""
    if isinstance(x, dict):
        return {k: _san(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_san(v) for v in x]
    if isinstance(x, torch.Tensor):
        return _san(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


def report_dict(rec):
    if rec is None or rec.report is None:
        return None
    return _san(dataclasses.asdict(rec.report))


def loop_spacing_stats(v, loops):
    segs = []
    for c in loops:
        P = v.positions[c]
        segs.append((P.roll(-1, 0) - P).norm(dim=1))
    s = torch.cat(segs)
    return dict(min=float(s.min()), median=float(s.median()),
                max=float(s.max()))


def run_config(shape, kind, n_steps, seed_cert=None, gates=True,
               out_dir=None, n_per=128, sigma_p=None, sigma_a=None,
               theta_const=None, save_states_every=None):
    v, rank, loops = make_shape(shape, n_per=n_per)
    delta, tau = compute_recommended_params(v.positions)
    if kind in ("C1", "C2"):
        cfg = make_cfg(kind, delta, tau, rank)
        cfg.time_step = DT
    else:
        cfg = make_cfg("C3", delta, tau, rank)
        cfg.time_step = DT
        cfg.bem_rank_mode = "adaptive_state_machine"
        cfg.bem_component_rank = None
        cfg.initial_rank_certificate = seed_cert
        if seed_cert is not None:
            cfg.initial_rank_certificate_provenance = (
                "paper generator: two disjoint closed loops at t=0 "
                "(integer count only)")
    # FAIR gate comparison (reviewer): the support gate is a
    # metric-independent safety layer, so it is switched per TRACK
    # (gated main line vs --no-gates stress track), not per metric.
    # Known limitation for C1/C2: on a whole-loop extinction the gate
    # fail-closes with "NO rank evidence" (legacy mode has no rank
    # machinery) -- documented behaviour, not a bug.
    cfg.enforce_support_gates = bool(gates)
    # Phase IV: independently chosen bandwidths for the quantitative
    # pre-contact track (sigma from the sigma-sweep, angle from the
    # angle-only sweep); None keeps the production defaults.
    if sigma_p is not None:
        cfg.perimeter_sigma = sigma_p
    if sigma_a is not None:
        cfg.angle_sigma = sigma_a
    if theta_const is not None:
        cfg.rank_theta_const_deg = theta_const

    # P1.2 correction (2026-08-05): long runs REQUIRE the production
    # post-processing bundle.  Without redistribution, tangential drift
    # collapses adjacent spacing (star: 0.0098 -> 0.0004 in 8 steps),
    # the y-space W_lin Hessian loses positive-definiteness
    # (eig_min = -10.75 measured on the bunched frame) and trust-ncg
    # diverges along the negative-curvature direction.  The P1
    # one-step comparison configs (redistribute=False) must not be
    # reused for trajectories.
    cfg.redistribute = True
    cfg.remove_dead_points = True
    # observational BEM-setup telemetry (bitwise-neutral, pinned by
    # tests/test_bem_setup_telemetry.py); scalars go to every row,
    # full arrays only to selected frames (memory policy)
    cfg.bem_setup_telemetry = True

    solver = MMSolver(cfg)
    solver.support_timeline = SupportTimeline(
        initial_loops=loops, store_pointwise_fields=False)
    t0 = time.perf_counter()
    _last = [t0]
    step_walls = []
    step_niters = []
    prev_v = [v]
    snapshots = []
    setup_scalars = {}
    first_snapshot = [None]
    last_snapshot = [None]

    def _progress(step, result):
        now = time.perf_counter()
        step_walls.append(now - _prev_t[0])
        _prev_t[0] = now
        step_niters.append(result.n_iter)
        sn = result.bem_setup_snapshot
        if sn is not None:
            if first_snapshot[0] is None:
                first_snapshot[0] = sn
            last_snapshot[0] = sn
            sc = dict(delta_used=sn.delta_used, tau_used=sn.tau_used,
                      sigma_used=sn.perimeter_sigma_used,
                      angle_sigma_used=sn.angle_sigma_used,
                      eps_bem_used=sn.epsilon_bem_used,
                      n_fev=result.n_fev)
            # collocation proximity for ALL modes (hang-attribution
            # requirement). raw includes BY-DESIGN adjacency (segment
            # endpoints of neighbouring particles meet); the DANGER
            # metric excludes each particle's 2 nearest neighbours and
            # measures NONLOCAL near-duplication only.
            if (sn.endpoint_positions is not None
                    and sn.epsilon_bem_used):
                E = sn.endpoint_positions
                K = sn.endpoints_per_particle
                pid = torch.arange(E.shape[0]) // K
                dm = torch.cdist(E, E).masked_fill(
                    pid[:, None] == pid[None, :], float("inf"))
                sc["min_collocation_over_eps_raw"] = float(
                    dm.min() / sn.epsilon_bem_used)
                P = sn.positions
                dp = torch.cdist(P, P)
                dp.fill_diagonal_(float("inf"))
                nn2 = dp.argsort(dim=1)[:, :2]
                adj = torch.zeros_like(dp, dtype=torch.bool)
                adj.scatter_(1, nn2, True)
                adj = adj | adj.T
                adj_e = adj.repeat_interleave(K, 0) \
                           .repeat_interleave(K, 1)
                dm2 = dm.masked_fill(adj_e, float("inf"))
                sc["min_collocation_over_eps_nonlocal"] = float(
                    dm2.min() / sn.epsilon_bem_used)
            # raw singular values at the working rank (never only the
            # derived ratios; ascending convention fixed at creation)
            s = sn.singular_values_ascending
            C = result.detected_rank
            if s is not None and C is not None and 1 <= C < len(s):
                sc.update(
                    singular_values_order="ascending",
                    s_null_edge=float(s[C - 1]),
                    s_range_edge=float(s[C]),
                    s_max=float(s[-1]),
                    rho_rank=float(s[C - 1] / s[C]),
                    # BIE range conditioning -- NOT a KKT condition no.
                    kappa_A_range=float(s[-1] / s[C]))
            setup_scalars[step] = sc
        # slow-step snapshot for hang attribution (reviewer 8): dump the
        # PRE-step varifold when a step takes > 5x the running median
        if (out_dir is not None and len(step_walls) > 10
                and len(snapshots) < 3):
            med = statistics.median(step_walls[:-1])
            if step_walls[-1] > 5 * med:
                sd = Path(out_dir) / "snapshots"
                sd.mkdir(parents=True, exist_ok=True)
                p = sd / f"{shape}_{kind}_step{step}.pt"
                torch.save(dict(
                    positions=prev_v[0].positions,
                    angles=prev_v[0].angles, shape=shape, kind=kind,
                    step=step, wall=step_walls[-1], median=med,
                    loops=[c.clone() for c in
                           solver.support_timeline.current_loops()]), p)
                snapshots.append(dict(step=step, path=str(p),
                                      wall=step_walls[-1], median=med))
                print(f"    [{shape}|{kind}] SLOW step {step}: "
                      f"{step_walls[-1]:.1f}s vs median {med:.1f}s "
                      f"-- snapshot saved", flush=True)
        prev_v[0] = result.varifold
        # G3.1: periodic state checkpoints for cross-backend post-hoc
        # comparison (positions/angles, same cadence as the grid
        # longrun driver)
        if save_states_every and (
                step % save_states_every == 0
                or step in (1, n_steps - 1)):
            _state_ckpts[step] = dict(
                positions=result.varifold.positions.clone(),
                angles=result.varifold.angles.clone())
        if (step + 1) % 25 == 0:
            print(f"    [{shape}|{kind}] step {step + 1}/{n_steps} "
                  f"{(now - _last[0]) / 25:.2f}s/step "
                  f"n_iter={result.n_iter}", flush=True)
            _last[0] = now
        return False

    _prev_t = [t0]
    _state_ckpts = {}
    hist = solver.solve(v, n_steps, callback=_progress)
    wall = time.perf_counter() - t0
    if save_states_every and out_dir is not None:
        torch.save(_state_ckpts,
                   Path(out_dir) / f"{shape}_{kind}_states.pt")
    tl = solver.support_timeline

    by_level = {}
    for r in tl.records:
        by_level[(r.level, r.step)] = r

    rows = []
    first_unresolved = None
    for step in range(hist.n_completed):
        rec = by_level.get(("post_deletion_final", step))
        row = dict(step=step)
        if step < len(step_walls):
            row["wall_s"] = round(step_walls[step], 3)
            row["n_iter"] = step_niters[step]
        if rec is not None and rec.report is not None:
            row["state"] = rec.report.state
            if shape == "pair":
                row["min_gap"] = rec.report.min_gap
            # SELF-CONTAINED artifact (reviewer B): the full committed
            # SupportReport -- l_res_contact, gap_over_delta/sigma/
            # resolution, carrier_bias_active, facing quantiles, ...
            row["report"] = report_dict(rec)
            if (first_unresolved is None
                    and rec.report.state == "touching_or_unresolved"):
                first_unresolved = step
        raw = by_level.get(("post_deletion_raw", step))
        if raw is not None:
            row["n_removed"] = raw.n_removed
        if rec is not None and rec.rank_fields:
            for k in ("detected_rank", "rank_status"):
                if k in rec.rank_fields:
                    row[k] = rec.rank_fields[k]
        # FULL scalar rank telemetry from post_mm_candidate (reviewer T:
        # everything the machine saw -- spectrum head, evidence, action,
        # align/contain angles; scalars and float lists only, no basis
        # tensors reach the JSON)
        rec_c = by_level.get(("post_mm_candidate", step))
        if rec_c is not None and rec_c.rank_fields:
            rank_all = {k: v for k, v in rec_c.rank_fields.items()
                        if isinstance(v, (int, float, str, bool,
                                          type(None), list))}
            row["rank_fields"] = _san(rank_all)
        # BEM-setup scalars actually USED by this step's solve
        if step in setup_scalars:
            row["bem_setup"] = _san(setup_scalars[step])
        rows.append(row)

    # stop context (reviewer 2): the last VALID committed frame and the
    # REJECTED candidate are different objects with different gaps --
    # never mix them in the narrative
    stop_context = None
    if hist.stop_step is not None:
        lv = hist.last_valid_step
        lv_rec = (by_level.get(("post_deletion_final", lv))
                  if lv is not None else None)
        cand_rec = by_level.get(("post_mm_candidate", hist.stop_step))
        stop_context = dict(
            last_valid_step=lv,
            last_valid_report=report_dict(lv_rec),
            last_valid_gap=(lv_rec.report.min_gap
                            if lv_rec is not None
                            and lv_rec.report is not None else None),
            rejected_candidate_report=report_dict(cand_rec),
            rejected_candidate_gap=(cand_rec.report.min_gap
                                    if cand_rec is not None
                                    and cand_rec.report is not None
                                    else None))

    # selected-frame FULL snapshot arrays (memory policy: never the
    # whole history) -- first step + last step, as .pt next to the JSON
    if out_dir is not None:
        sd = Path(out_dir) / "snapshots"
        for tag, sn in (("first", first_snapshot[0]),
                        ("last", last_snapshot[0])):
            if sn is not None:
                sd.mkdir(parents=True, exist_ok=True)
                torch.save(sn, sd / f"{shape}_{kind}_bemsetup_{tag}.pt")

    final = hist.get_final_valid_varifold()
    out = dict(
        shape=shape, kind=kind, seed_cert=seed_cert, dt=DT,
        gates_enforced=bool(gates),
        post_processing=dict(redistribute=cfg.redistribute,
                             remove_dead_points=cfg.remove_dead_points),
        # SELF-CONTAINED artifact (reviewer B): full config + scales
        mm_config=_san({k: (val if isinstance(
            val, (int, float, str, bool, type(None))) else str(val))
            for k, val in dataclasses.asdict(cfg).items()}),
        scales=dict(delta=float(delta), tau=float(tau),
                    sigma=cfg.perimeter_sigma,
                    angle_sigma=cfg.angle_sigma,
                    h_face0=loop_spacing_stats(v, loops),
                    gap0=(pair_gap(v.positions) if shape == "pair"
                          else None)),
        # scientific window (reviewer 3): trajectory data are usable as
        # resolved one-phase dynamics ONLY strictly BEFORE this step
        first_touching_or_unresolved=first_unresolved,
        n_points=v.n_points, rank_oracle_reference=rank,
        wall_seconds=wall,
        n_steps_requested=n_steps, n_steps_completed=hist.n_completed,
        stop=dict(step=hist.stop_step, stage=hist.stop_stage,
                  exc=hist.stop_exception_type,
                  msg=(hist.stop_message or "")[:300])
        if hist.stop_step is not None else None,
        stop_context=stop_context,
        slow_step_snapshots=snapshots,
        terminal_candidate_valid=hist.terminal_candidate_valid,
        timeline_errors=[(r.step, r.level, r.error)
                         for r in tl.records if r.error][:10],
        perimeters_head_tail=[float(hist.perimeters[0]),
                              float(hist.perimeters[
                                  max(hist.n_completed - 1, 0)])],
        final_positions=final.positions,
        rows=rows)
    if shape == "pair":
        out["final_gap"] = pair_gap(final.positions)
        out["gap_series"] = [
            (r["step"], r.get("min_gap")) for r in rows
            if r.get("min_gap") is not None][::20]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/p12_long"))
    ap.add_argument("--shape", default="pair",
                    choices=["pair", "flower", "star", "ellipse"])
    ap.add_argument("--n-steps", type=int, default=2000)
    ap.add_argument("--kind", default=None,
                    choices=[None, "C1", "C2", "C3A", "C3S"],
                    help="run ONE config and write its JSON "
                         "immediately (parallel orchestration)")
    ap.add_argument("--n-per", type=int, default=128,
                    help="points per component (pair shape)")
    ap.add_argument("--sigma-p", type=float, default=None,
                    help="override perimeter_sigma (Phase IV "
                         "quantitative track)")
    ap.add_argument("--sigma-a", type=float, default=None,
                    help="override angle_sigma")
    ap.add_argument("--tag", default="",
                    help="suffix for the output JSON filename")
    ap.add_argument("--save-states-every", type=int, default=None,
                    help="G3.1: save positions/angles checkpoints "
                         "every N steps to <shape>_<kind>_states.pt")
    ap.add_argument("--theta-const", type=float, default=None,
                    help="AUDIT-ONLY: rank-machine phase-alignment "
                         "threshold in degrees (default keeps the "
                         "production 15)")
    ap.add_argument("--no-gates", action="store_true",
                    help="UNGATED STRESS track (applies to ALL kinds): "
                         "disable the metric-independent support gate "
                         "to observe how each configuration fails past "
                         "first unresolved contact; the gated run is "
                         "the production behavior and the scientific "
                         "window ends at first_touching_or_unresolved")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.kind is not None:
        kind, cert = args.kind, (2 if args.kind == "C3S" else None)
        base_kind = {"C3A": "C3", "C3S": "C3"}.get(kind, kind)
        r = run_config(args.shape, base_kind, args.n_steps,
                       seed_cert=cert, gates=not args.no_gates,
                       out_dir=args.out, n_per=args.n_per,
                       sigma_p=args.sigma_p, sigma_a=args.sigma_a,
                       theta_const=args.theta_const,
                       save_states_every=args.save_states_every)
        fp = r.pop("final_positions")
        r["final_positions_list"] = fp.tolist()
        suffix = ("_nogates" if args.no_gates else "") + args.tag
        pth = args.out / f"p12_{args.shape}_{kind}{suffix}.json"
        pth.write_text(json.dumps(r, indent=1, default=str))
        print(f"  {args.shape}|{kind}: completed="
              f"{r['n_steps_completed']}/{args.n_steps} "
              f"stop={r['stop']} wall={r['wall_seconds']:.0f}s")
        print(f"raw results -> {pth}")
        return

    shape = args.shape
    kinds = [("C1", None), ("C2", None), ("C3A", None)]
    if shape == "pair":
        kinds.append(("C3S", 2))

    path = args.out / f"p12_{shape}.json"
    out = dict(meta=dict(shape=shape, dt=DT, n_steps=args.n_steps))
    finals = {}
    for kind, cert in kinds:
        label = kind
        base_kind = {"C3A": "C3", "C3S": "C3"}.get(kind, kind)
        r = run_config(shape, base_kind, args.n_steps, seed_cert=cert,
                       out_dir=args.out)
        finals[label] = r.pop("final_positions")
        out[label] = r
        gap = (f" gap={r.get('final_gap'):.4f}"
               if r.get("final_gap") is not None else "")
        print(f"  {shape}|{label}: completed="
              f"{r['n_steps_completed']}/{args.n_steps} "
              f"stop={r['stop']}{gap} wall={r['wall_seconds']:.0f}s",
              flush=True)

    # pairwise final-state distances on the common completed window
    for a in finals:
        for b in finals:
            if a < b and finals[a].shape == finals[b].shape:
                out[f"final_dx_{a}_{b}"] = float(
                    (finals[a] - finals[b]).abs().max())
    for k, val in list(out.items()):
        if k.startswith("final_dx"):
            print(f"  {k} = {val:.3e}", flush=True)

    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
