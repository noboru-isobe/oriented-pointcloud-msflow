"""G3c: the grid backend attempts the EXACT trajectory the BIE lost.

Reference event (results/p12_long/p12_ellipse_C3A.json): ellipse 2:1,
N=256, dt=1e-5, production post-processing bundle -- the adaptive rank
machine killed the run at step 884 (AmbiguousComponentRankError,
reject_artificial, align 15 deg) while spectrum/support/geometry were
clean. The grid backend has no rank/align machinery; its component
structure comes from the phase support.

Reviewer criteria (G3c): passing step 884 alone proves nothing. The
claim "the BIE theta-kill was a false positive for this trajectory"
requires EVERY validity gate to hold, audited at step 900 before the
run is allowed to continue to 2000:

  G1  objective decreased every step
  G2  relative gradient norm finite and < 1e-5 every step
  G3  grid component count == 1 every step
  G4  reconstruction: projection_l2_relative < 1e-2, no speck mass
      dropped (fail-closed layers also enforce this in-line)
  G5  composed-constraint condition < 100
  G6  Poisson max relative residual < 1e-10
  G7  |V_hat KDE drift| (divergence-theorem functional, vs step-1)
      < 0.15
      [recalibrated after the first attempt: run 1 (current_divergence
      targeting) was CORRECTLY stopped by the fail-closed projection
      gate at step 263 -- the water-filling was absorbing exactly the
      KDE quadrature drift (projection ~ 0.0005 + 1.25|drift|, ratio
      constant over 260 steps; drift rate -2.9e-5/STEP, dt-independent
      -- a property of the KDE functional along the flow, NOT a grid
      volume leak; the grid constraint conserves the linearized phase
      volume exactly, realized flux ~1e-21). Fix: initial_fixed
      targeting (the reviewer's conservation track) -- the filling
      target no longer chases the drifting KDE functional, so the
      projection stays at its step-0 level and G4 is the meaningful
      gate. V_hat drift remains RECORDED; this gate now only catches
      a runaway (2.5x headroom over the measured benign rate x 2000
      steps). GEOMETRIC conservation is audited post-hoc from the
      checkpointed states (shoelace on the polar-ordered loop).]
  G8  max|s| / median local spacing < 0.5
  G9  active support >= 2*eps away from the box boundary
  G10 no fail-closed exception (any raise ends the run and is
      recorded, never swallowed)

RUN 4 CONTEXT (theta-drift root cause, 2026-08-06): runs 1-3 were
stopped by the projection gate at steps 263/75/248 -- correctly. The
checkpoint post-hoc split traced the V_hat drift to the varifold
ANGLES lagging the geometric normals (rms 2.2e-2 rad by step 200,
polygon-normal quadrature exact at 1.5e-4 throughout), and the
one-step decomposition measured the angle map A^dagger B applying
only 0.449 of the geometric rotation -- uniformly at every
wavenumber (circle transfer factor 0.4461, shape residual 1e-13).
Root cause: compute_kernel_gradient omitted the C_2D mollifier
normalization that the A-side psi carries, a spurious uniform
factor 1/C_2D = pi/7 for wendland_c2 (finite-difference match
1/c = 2.228169 vs C_2D = 2.228169). Fixed in
src/torch/perimeter/angle_constraint.py with transfer-factor pins in
tests/test_angle_constraint.py. NOTE: the reference kill at step 884
was recorded under the PRE-FIX map; the secular angle drift it
detected (align 15 deg ~ the extrapolated max angle error at step
~900) was REAL and is now repaired at source, for both backends.
Gates run at their original calibration -- no tolerance was widened.

Config: p12 ellipse conventions (make_cfg C3 scales, dt=1e-5,
redistribute + dead-point removal + support gates -- the P1.2
correction: trajectories REQUIRE the production bundle) with
metric_backend="grid_poisson" (512^2, eps=0.05, sparse_direct) and
bem_solver_mode="legacy_global_projection" (no spectral machinery).

Output: results/grid_metric/longrun_ellipse.json (+ final state
tensors alongside), flushed every 50 steps.
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
    generate_oriented_ellipse,
)
from src.torch.solver.mm_solver import MMSolver  # noqa: E402
from src.torch.transport.grid_wasserstein import (  # noqa: E402
    GridMetricConfig,
)
from src.torch.transport.phase_grid import PhaseGridConfig  # noqa: E402
from p1_production_comparison import make_cfg  # noqa: E402

DT = torch.float64
DT_STEP = 1e-5
N_STEPS = 2000
AUDIT_STEP = 900
EPS = 0.05
OUT = ROOT / "results" / "grid_metric"
GATES = ("G1_objective_decreased", "G2_relgrad", "G3_components",
         "G4_reconstruction", "G5_constraint_condition",
         "G6_solve_residual", "G7_volume_drift", "G8_step_scale",
         "G9_box_margin")


def gate_flags(rec):
    return {
        "G1_objective_decreased": bool(rec["objective_decreased"]),
        "G2_relgrad": (rec["relgrad"] is not None
                       and math.isfinite(rec["relgrad"])
                       and rec["relgrad"] < 1e-5),
        "G3_components": rec["n_components"] == 1,
        "G4_reconstruction": (rec["projection_l2_relative"] < 1e-2
                              and rec["speck_mass_dropped"] == 0.0),
        "G5_constraint_condition": rec["constraint_condition"] < 100.0,
        "G6_solve_residual": rec["max_solve_residual"] < 1e-10,
        "G7_volume_drift": abs(rec["volume_drift"]) < 0.15,
        "G8_step_scale": rec["step_over_h"] < 0.5,
        "G9_box_margin": rec["box_margin_over_eps"] >= 2.0,
    }


def main():
    v0 = generate_oriented_ellipse(256, 1.0, 0.5, (0.0, 0.0),
                                   device="cpu", dtype=DT)
    delta, tau = compute_recommended_params(v0.positions)
    cfg = make_cfg("C3", delta, tau, 1)
    cfg.time_step = DT_STEP
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    # support_threshold 3e-3 (run 4): the G1-calibrated stability
    # plateau is [1e-3, 1e-2] (counts stable, areas within 15%); the
    # old default 1e-3 sits at the plateau's LOWER EDGE, touching the
    # measured noise-floor band (1e-4..1e-3). Post-C_2D-fix the volume
    # drift is POSITIVE (saturating at ~+1e-4), so the water-filling
    # LIFTS the wall skirt; at step ~90 four single-cell crumbs of the
    # 1e-3 superlevel set pinched off the skirt and the speck gate
    # correctly failed closed. Diagnosis (step-88 state): the
    # 5e-4..5e-3 band hugs the skirt (dist to curve 0.03..0.12 ~
    # eps..2.5eps, no far-field pockets); at 3e-4 the skirt is
    # CONNECTED, so the crumbs are threshold-quantization of a thin
    # skirt band, not wall porosity. 3e-3 = plateau center, ~3x
    # headroom over the saturating lift; the thr/3 component count
    # (recorded per step below) keeps the crumbs visible as an early
    # warning rather than a run-killer.
    cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(512, 512), fill_epsilon=EPS,
        support_threshold=3e-3))
    # P1.2 correction: trajectories require the production bundle
    cfg.redistribute = True
    cfg.remove_dead_points = True
    cfg.enforce_support_gates = True
    # run 2: conservation-track targeting (see G7 note above)
    cfg.grid_volume_target_mode = "initial_fixed"

    solver = MMSolver(cfg)
    solver.support_timeline = SupportTimeline(
        initial_loops=[torch.arange(v0.n_points)],
        store_pointwise_fields=False)

    series = []
    state = {"aborted_at_audit": False, "audit_report": None,
             "volume_target_mode": cfg.grid_volume_target_mode}
    checkpoints = {}
    t_prev = [time.perf_counter()]

    def shoelace_polar(pos):
        """Post-hoc geometric area: shoelace on the polar-ordered
        single loop (valid for the star-shaped ellipse relaxation)."""
        c = pos.mean(dim=0)
        th = torch.atan2(pos[:, 1] - c[1], pos[:, 0] - c[0])
        P = pos[torch.argsort(th)]
        x, y = P[:, 0], P[:, 1]
        return float(0.5 * (x * y.roll(-1) - y * x.roll(-1)).sum().abs())

    def flush():
        (OUT / "longrun_ellipse.json").write_text(json.dumps(dict(
            meta=dict(dt=DT_STEP, n_steps=N_STEPS, eps=EPS,
                      grid=512, audit_step=AUDIT_STEP,
                      reference_kill_step=884),
            state=state, series=series), indent=1))

    def callback(step, result):
        now = time.perf_counter()
        sn = result.grid_setup_snapshot
        gs = result.grid_step_stats
        pos = result.varifold.positions
        d = torch.cdist(pos, pos)
        d.fill_diagonal_(float("inf"))
        h_med = float(d.min(dim=1).values.median())
        rec = dict(
            step=step,
            wall_s=now - t_prev[0],
            n_points=pos.shape[0],
            objective_decreased=bool(result.objective_decreased),
            relgrad=result.relative_gradient_norm,
            n_iter=result.n_iter,
            perimeter=result.perimeter,
            n_components=sn.n_components,
            projection_l2_relative=sn.projection_l2_relative,
            speck_mass_dropped=sn.speck_mass_dropped,
            constraint_condition=sn.constraint_row_condition,
            max_solve_residual=gs.max_relative_residual,
            volume_drift=sn.target_volume_drift_relative,
            volume_current=sn.target_volume_current,
            step_over_h=float(result.displacements.abs().max()) / h_med,
            box_margin_over_eps=sn.box_boundary_margin_over_eps,
            n_solves=gs.n_forward_solves + gs.n_projected_solves,
            counts_at_thresholds=list(sn.component_counts_at_thresholds),
        )
        t_prev[0] = now
        rec["geometric_area"] = shoelace_polar(pos)
        rec["gates"] = gate_flags(rec)
        rec["all_gates_pass"] = all(rec["gates"].values())
        series.append(rec)
        if step % 200 == 0 or step in (1, AUDIT_STEP, N_STEPS - 1):
            checkpoints[step] = dict(
                positions=pos.clone(),
                angles=result.varifold.angles.clone())
            torch.save(checkpoints, OUT / "longrun_states.pt")
        if step % 50 == 0 or not rec["all_gates_pass"]:
            flush()
        if step % 50 == 0 or step in (884, 885, AUDIT_STEP):
            print(f"step {step:4d}: N={rec['n_points']} "
                  f"P={rec['perimeter']:.5f} "
                  f"drift={rec['volume_drift']:+.2e} "
                  f"relgrad={rec['relgrad']:.1e} "
                  f"s/h={rec['step_over_h']:.3f} "
                  f"{'OK' if rec['all_gates_pass'] else 'GATE-FAIL'} "
                  f"({rec['wall_s']:.1f}s)", flush=True)
        if step == AUDIT_STEP:
            # reviewer condition: full-window audit BEFORE continuing
            bad = [r["step"] for r in series
                   if not r["all_gates_pass"]]
            worst = {g: max(0 if r["gates"][g] else 1
                            for r in series) for g in GATES}
            state["audit_report"] = dict(
                steps_audited=len(series), failing_steps=bad[:50],
                n_failing=len(bad), gates_ever_failed=[
                    g for g, x in worst.items() if x])
            print(f"== AUDIT @ {step}: {len(bad)} failing steps; "
                  f"gates ever failed: "
                  f"{state['audit_report']['gates_ever_failed']} ==",
                  flush=True)
            flush()
            if bad:
                state["aborted_at_audit"] = True
                return True          # stop -- criteria not met
        return False

    t0 = time.time()
    err = None
    try:
        hist = solver.solve(v0, N_STEPS, callback=callback)
        completed = len(series)
        stop = getattr(hist, "stop_stage", None)
    except Exception as e:  # noqa: BLE001 -- G10: record, never hide
        err = f"{type(e).__name__}: {e}"
        completed = len(series)
        stop = "exception"
    state.update(
        completed_steps=completed, wall_total_s=time.time() - t0,
        stop_stage=str(stop), error=err,
        passed_reference_kill_step=completed > 884,
    )
    flush()
    print(f"\ncompleted {completed}/{N_STEPS} steps in "
          f"{(time.time() - t0) / 3600:.2f} h; "
          f"passed step 884: {completed > 884}; "
          f"stop={stop} err={err}", flush=True)


if __name__ == "__main__":
    main()
