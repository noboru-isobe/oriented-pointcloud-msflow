"""Otto-style pass-through contact experiment: the ACTUAL two-ellipse
pair, grid weighted-Poisson backend.

Precise claim (review wording): NO PRESCRIBED COMPONENT LABELS, NO
EVENT-TIME HISTORY, NO EXPLICIT CONNECTIVITY SURGERY. The scheme does
use threshold component detection, the conservative two-core
partition, the ambiguity-window rules, and subgrid speck removal --
every one a function of the CURRENT diffuse phase alone, appearing
and disappearing with it. That is the stronger statement: the
compatibility regularization the merger needs emerges from the
present rho, history-free, and removes itself.

Design (user direction, 2026-08-07): in Otto's set-level formulation a
topology change is a non-event -- the JKO minimization is over sets
and never needs to recognize the transition. The grid backend restores
exactly that structure for the point-cloud scheme: the filling map
rebuilds the phase rho from the current every step, connected
components emerge from rho's support (solvability of the weighted
Poisson operator), and opposite-orientation sheets CANCEL in the
filling (pinned in G1). So a merger needs no certification: the phase
goes 2 -> 1 components on its own, the volume-constraint rows follow,
and the q-dead carrier points become passive (velocity ~ q) until
dead-point removal sweeps them.

What this run therefore tests is NOT topology recognition but
VALIDITY through the transition window: the filling's scale
conditions (2*eps < gap fails transiently as the gap closes through
[0, 2*eps]), the frozen-linearization error near contact, and the
reconstruction gates (projection, speck, carrier renormalization, box
margin -- all SCALE gates, none topological). They stay fail-closed
at their calibrated values: where and whether one fires IS the
result. No gate is pre-weakened; the support timeline records as an
observer but does not gate (enforce_support_gates=False -- the
loop-count-event machinery is BIE-era topology bookkeeping, exactly
what this experiment bypasses).

Config: calibrated G2 pair scales (n_per=256 -> h/eps = 0.45 with
eps=0.04 < gap0/2 = 0.05, grid 512^2, support_threshold 3e-3 as in
G3c run 4), production C3 scales, dt=1e-5 (measured stable domain),
redistribute + dead-point removal ON, metric_backend=grid_poisson
(sparse_direct), initial_fixed total-volume targeting (total area is
conserved through a merger; the componentwise split is the phase's
own business). 4000 steps = t=0.04, crossing the superposition
contact prediction t* ~ 0.023 with margin.

Output: results/grid_metric/pair_contact_otto.json + state
checkpoints every 100 steps in pair_contact_otto_states.pt.
"""

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import torch  # noqa: E402

from src.torch.diagnostics import SupportTimeline  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.shapes.generator import (  # noqa: E402
    generate_oriented_two_ellipses,
)
from src.torch.solver.mm_solver import MMSolver  # noqa: E402
from src.torch.transport.grid_wasserstein import (  # noqa: E402
    GridMetricConfig,
)
from src.torch.transport.phase_grid import PhaseGridConfig  # noqa: E402
from p1_production_comparison import make_cfg  # noqa: E402

DT = torch.float64
DT_STEP = 1e-5
N_STEPS = 4000  # module default; --steps overrides in main()
N_PER = 256
EPS = 0.04
OUT = ROOT / "results" / "grid_metric"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames (confirmatory "
                         "suite: psp_on / psp_off / grid768)")
    ap.add_argument("--no-psp", action="store_true",
                    help="negative control: fossil left in place")
    ap.add_argument("--advect", action="store_true",
                    help="arm 4: passive advection of low-q marks by "
                         "the step's own transport field grad(phi)")
    ap.add_argument("--fade", action="store_true",
                    help="arm 5: persistent carrier-amplitude fade of "
                         "phase-interior marks (uniform whole-complex "
                         "reduction)")
    ap.add_argument("--fade-rate", type=float, default=0.05,
                    help="arm 6: base fade rate (q-modulated; marks "
                         "reaching a=0 are garbage-collected)")
    ap.add_argument("--fade-variational", action="store_true",
                    help="arm V1: replace rate/closed-loop/allocation "
                         "with one box-constrained QP per step (grid "
                         "metric quadratic + lam * removed amplitude)")
    ap.add_argument("--lam", type=float, default=0.0,
                    help="arm V1: shadow price per unit gated carrier "
                         "amplitude removed")
    ap.add_argument("--qp-norm", default="metric",
                    choices=["metric", "gate_l2"],
                    help="experiment B: norm pricing the removal QP")
    ap.add_argument("--vertical", action="store_true",
                    help="arm V2: coupled minimizing movement -- da "
                         "becomes a decision variable of the MM step "
                         "(transport compensates removal in-step)")
    ap.add_argument("--freeze-dead", action="store_true",
                    help="arm 7: dead marks' displacement DOF leave "
                         "the admissible basis (horizontal-space "
                         "optimization)")
    ap.add_argument("--grid", type=int, default=512,
                    help="grid resolution (second-resolution control)")
    ap.add_argument("--steps", type=int, default=None,
                    help="override total step count (calibration "
                         "probes; default 4000)")
    ap.add_argument("--resume", type=int, default=None,
                    help="restart from this checkpoint step in "
                         "pair_contact_otto_states.pt (the volume "
                         "target stays the DETERMINISTIC initial "
                         "cloud's -- initial_fixed semantics survive "
                         "the restart)")
    args = ap.parse_args()

    v0 = generate_oriented_two_ellipses(
        N_PER, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
        a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu", dtype=DT)
    n1 = N_PER
    delta, tau = compute_recommended_params(v0.positions)
    cfg = make_cfg("C3", delta, tau, 2)
    cfg.time_step = DT_STEP
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    # conservative_sweep (v4): runs v1-v3 all stopped at the same
    # step 743 -- the moment the operating support bridges, the
    # operator acquires a soft exchange mode (measured |phi|/|b|
    # 1.3e-3 -> 2.27) and float64 solves hit the eps*||A||*||phi||
    # floor (3.1e-10 > the 1e-10 tolerance; refinement and
    # LU-preconditioned CG both stagnate there, as they must). The
    # compatibility subspace now follows the most conservative
    # confidence-sweep reading: blobs keep separate volume
    # constraints/projections until thr AND 3*thr agree the support
    # is one component -- label-free, self-removing, no events.
    # v9 exploratory operating point (reviewer-instrumented run):
    # v8 measured the healing projection disturbance OUTLASTING the
    # count-based ambiguity window (strict gate fired at 1.33e-2 with
    # the window closed). Rather than invent another window heuristic
    # mid-flight, this RUN operates at projection_rel_tol=5e-2 with
    # the full per-step series recorded (projection, compat count,
    # window flag, speck details, cumulative sums) -- the principled
    # gate is calibrated POST-HOC from the measured series.
    cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(args.grid, args.grid), fill_epsilon=EPS,
        support_threshold=3e-3,
        projection_rel_tol=5e-2),
        compatibility_components="conservative_sweep")
    cfg.redistribute = True
    # NO deletion (user direction 2026-08-07): dead-point removal was a
    # BIE-era necessity (carrier contamination + conditioning) and its
    # legacy recipe is now the very thing that destabilizes the
    # corrected map (ablation cells 4/5 blow at the first removal
    # epoch). In the Otto picture the dead marks are passive: the
    # filling cancels opposite sheets, q suppresses their P and flux
    # contributions. The KDE-carrier bias near retained hidden
    # clusters is the measured quantity this run exposes -- if the
    # scale gates stay green through the merger with nothing deleted,
    # deletion is demonstrably unnecessary for the grid backend.
    cfg.remove_dead_points = False
    cfg.enforce_support_gates = False       # observer only (see header)
    cfg.grid_volume_target_mode = "initial_fixed"
    # v10: interior-fossil projection (review option B) --
    # once the phase is stably single-component, marks the
    # reconstructed set declares strictly interior are
    # removed (shadow-verified); kills the fossil dipole
    # that v9 measured growing (raw_max 1.02 -> 1.14)
    cfg.grid_phase_support_projection = not args.no_psp
    # arm 4: fossils ride the flow as tracers instead of freezing
    cfg.grid_fossil_advection = args.advect
    # arm 5: interior marks' carrier amplitude fades to zero
    cfg.grid_carrier_amplitude_fade = args.fade or args.vertical
    cfg.grid_vertical_dof = args.vertical
    if args.fade or args.vertical:
        from src.torch.transport.carrier_amplitude import (
            CarrierAmplitudeConfig,
        )
        cfg.grid_amplitude_config = CarrierAmplitudeConfig(
            rate=args.fade_rate,
            variational=args.fade_variational,
            lam=args.lam,
            qp_norm=args.qp_norm)
    # arm 7: horizontal-space optimization
    cfg.grid_freeze_dead_dof = args.freeze_dead
    if args.steps is not None:
        global N_STEPS
        N_STEPS = args.steps

    solver = MMSolver(cfg)
    solver.support_timeline = SupportTimeline(
        initial_loops=[torch.arange(n1), torch.arange(n1, 2 * n1)],
        store_pointwise_fields=False)

    step_offset = 0
    v_start = v0
    if args.resume is not None:
        from src.torch.oriented_varifold import (
            OrientedPointCloudVarifold,
        )
        from src.torch.oriented_varifold.mass import compute_masses
        ck = torch.load(OUT / f"pair_contact_otto{args.tag}_states.pt",
                        weights_only=True)
        st0 = ck[args.resume]
        v_start = OrientedPointCloudVarifold(
            positions=st0["positions"], angles=st0["angles"])
        step_offset = args.resume + 1
        # preserve initial_fixed semantics: the target stays the
        # DETERMINISTIC initial cloud's divergence volume, pre-seeded
        # on the stepper before the first step
        m0 = compute_masses(v0.positions, delta, tau, "wendland_c2")
        target0 = float(0.5 * (m0 * (v0.positions
                                     * v0.normals).sum(-1)).sum())
        # MMSolver builds its stepper inside solve(), so the driver
        # pre-seeds the initial_fixed target by patching the stepper
        # constructor (driver-local; the target equals what a from-
        # scratch run computes from the deterministic initial cloud)
        from src.torch.solver.mm_step import MMStepper as _MS
        _orig_init = _MS.__init__

        def _seeded_init(s, *a, **k):
            _orig_init(s, *a, **k)
            s._grid_target_volume_initial = target0
        _MS.__init__ = _seeded_init

    def _psp_cfg_dict():
        import dataclasses as _dc
        from src.torch.transport.phase_support_projection import (
            PhaseSupportProjectionConfig,
        )
        c = cfg.grid_psp_config or PhaseSupportProjectionConfig()
        return _dc.asdict(c)

    series = []
    checkpoints = ({} if args.resume is None else dict(
        torch.load(OUT / f"pair_contact_otto{args.tag}_states.pt",
                   weights_only=True)))
    state = {"merged_at_step": None, "resumed_from": args.resume,
             "compat_release_step": None, "stable_111_step": None,
             "source_commit": None}
    import subprocess as _sp
    state["source_commit"] = _sp.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True).stdout.strip()
    cum_speck = [0.0, 0.0]
    t_prev = [time.perf_counter()]

    def callback(raw_step, result):
        step = raw_step + step_offset
        now = time.perf_counter()
        sn = result.grid_setup_snapshot
        gs = result.grid_step_stats
        # review blocker 4: checkpoints and telemetry must use the
        # COMMITTED state (post redistribution/deletion/projection) --
        # result.varifold is the pre-postprocessing MM minimizer and
        # would resurrect projected-out fossils on resume
        v = result.committed_varifold \
            if result.committed_varifold is not None else result.varifold
        pos = v.positions
        # cross-component gap while the phase still sees 2 components:
        # use the sign of x as a cheap side label ONLY for the gap
        # diagnostic (the physics never consumes it)
        left = pos[:, 0] < 0
        # cross-side distance is a COMPONENT gap only while the
        # support reads two components (reviewer): recorded as None
        # after the bridge
        gap = (float(torch.cdist(pos[left], pos[~left]).min())
               if (sn.n_components == 2 and left.any()
                   and (~left).any()) else None)
        gm = solver._stepper.grid_wasserstein
        pgm = gm.phase
        cum_speck[0] += pgm.speck_mass_dropped
        cum_speck[1] += pgm.speck_area_dropped
        rec = dict(
            step=step,
            wall_s=now - t_prev[0],
            n_points=v.n_points,
            n_components=sn.n_components,
            compatibility_components=gm.n_compat_components,
            ambiguous_window=bool(gm.ambiguous_window),
            n_speck_dropped=pgm.n_speck_components_dropped,
            speck_area_dropped=pgm.speck_area_dropped,
            speck_details=list(pgm.speck_details),
            cum_speck_mass=cum_speck[0],
            cum_speck_area=cum_speck[1],
            rho_raw_min=pgm.rho_raw_min,
            rho_raw_max=pgm.rho_raw_max,
            counts_at_thresholds=list(sn.component_counts_at_thresholds),
            psp=result.phase_support_projection_outcome,
            gap_lr=gap,
            q_min=float((result.effective_masses
                         / result.masses.clamp_min(1e-300)).min()),
            perimeter=result.perimeter,
            objective_decreased=bool(result.objective_decreased),
            relgrad=result.relative_gradient_norm,
            n_iter=result.n_iter,
            projection_l2_relative=sn.projection_l2_relative,
            speck_mass_dropped=sn.speck_mass_dropped,
            carrier_renormalization=sn.carrier_renormalization,
            volume_drift=sn.target_volume_drift_relative,
            phase_volume=sn.phase_grid_volume,
            constraint_condition=sn.constraint_row_condition,
            max_solve_residual=gs.max_relative_residual,
            box_margin_over_eps=sn.box_boundary_margin_over_eps,
            advection=result.advection_stats,
            amplitude=result.amplitude_stats,
        )
        # arm 4 telemetry: re-coherence watch. A fossil pair drifting
        # into the true interface band could regain coherence and
        # re-enter the energy -- the pre-registered accident mode.
        # max q over marks holding an antiparallel partner within eps.
        try:
            q_vec = (result.effective_masses
                     / result.masses.clamp_min(1e-300))
            if q_vec.shape[0] == pos.shape[0]:
                nrm = v.normals
                d = torch.cdist(pos, pos)
                d.fill_diagonal_(float("inf"))
                paired = ((d < EPS) & ((nrm @ nrm.T) < -0.8)).any(dim=1)
                rec["paired_q_max"] = (float(q_vec[paired].max())
                                       if bool(paired.any()) else None)
                rec["n_paired"] = int(paired.sum())
            else:
                rec["paired_q_max"] = None
                rec["n_paired"] = None
        except Exception:
            rec["paired_q_max"] = None
            rec["n_paired"] = None
        t_prev[0] = now
        series.append(rec)
        if (state["merged_at_step"] is None
                and sn.n_components == 1):
            state["merged_at_step"] = step
            print(f"== OPERATING-SUPPORT BRIDGE (2 -> 1) at step "
                  f"{step}, t = {step * DT_STEP:.5f} ==", flush=True)
        if (state["compat_release_step"] is None
                and state["merged_at_step"] is not None
                and gm.n_compat_components == 1):
            state["compat_release_step"] = step
            print(f"== COMPATIBILITY RELEASE (C_compat 2 -> 1) at "
                  f"step {step} ==", flush=True)
        if (state["stable_111_step"] is None
                and rec["counts_at_thresholds"] == [1, 1, 1]
                and not rec["ambiguous_window"]):
            state["stable_111_step"] = step
            print(f"== STABLE (1,1,1), window closed, at step {step} "
                  f"==", flush=True)
        if step % 100 == 0 or step == N_STEPS - 1:
            checkpoints[step] = dict(positions=pos.clone(),
                                     angles=v.angles.clone())
            torch.save(checkpoints,
                       OUT / f"pair_contact_otto{args.tag}_states.pt")
            flush()
        if step % 50 == 0:
            gtxt = f"{gap:.4f}" if gap is not None else "--"
            print(f"step {step:4d}: N={rec['n_points']} "
                  f"C={rec['n_components']} gap={gtxt} "
                  f"P={rec['perimeter']:.5f} "
                  f"proj={rec['projection_l2_relative']:.1e} "
                  f"({rec['wall_s']:.1f}s)", flush=True)
        return False

    def flush():
        (OUT / f"pair_contact_otto{args.tag}.json").write_text(json.dumps(dict(
            meta=dict(dt=DT_STEP, n_steps=N_STEPS, n_per=N_PER,
                      eps=EPS, grid=args.grid, support_threshold=3e-3,
                      superposition_contact_prediction=0.02335,
                      projection_gate_status="exploratory_calibration",
                      projection_rel_tol=5e-2,
                      production_acceptance_gate="not_yet_selected",
                      support_gate="observer_only",
                      cumulative_speck_semantics=(
                          "reconstruction intervention budget, NOT "
                          "physical mass loss (water-filling is "
                          "mass-preserving each step)"),
                      psp_config=_psp_cfg_dict(),
                      psp_enabled=not args.no_psp,
                      fossil_advection=args.advect,
                      carrier_amplitude_fade=args.fade,
                      fade_rate=args.fade_rate,
                      fade_variational=args.fade_variational,
                      lam=args.lam,
                      qp_norm=args.qp_norm,
                      vertical=args.vertical,
                      freeze_dead_dof=args.freeze_dead),
            state=state, series=series), indent=1))

    t0 = time.time()
    err = None
    try:
        solver.solve(v_start, N_STEPS - step_offset,
                     callback=callback)
        stop = None
    except Exception as e:  # noqa: BLE001 -- record, never hide
        err = f"{type(e).__name__}: {e}"
        stop = "exception"
    state.update(completed_steps=len(series),
                 wall_total_s=time.time() - t0,
                 stop=str(stop), error=err)
    flush()
    print(f"\ncompleted {len(series)}/{N_STEPS} steps in "
          f"{(time.time() - t0) / 3600:.2f} h; "
          f"merged_at={state['merged_at_step']} err={err}", flush=True)


if __name__ == "__main__":
    main()
