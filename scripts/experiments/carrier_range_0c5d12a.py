"""0C-5d-1.2a/b: closing the carrier-overlap range mechanism.

Checks completing the 0C-5d-1.2 causal identification (delta varies at
FIXED N so nothing else moves), extended per review (1.2b) with the
perimeter-vs-metric pathway split, cutoff-saturation assertions, the
time-consistent diagnostic q, the rejected-candidate bookkeeping, and
the Q dt-refinement.

MEASURED SUMMARY: delta = gap is the exact switch-on threshold FOR THE
INITIAL PAPER CONFIGURATION AND THE COMPACT WENDLAND KERNEL (below it,
omega_cross and the global-vs-componentwise mass difference are EXACTLY
zero; response above it is non-monotone: -85/-183/-97 at delta/g0 =
1.25/1.5/2.0 -- range threshold and pathway closed, universal force law
not). The 2x2 call-site split, read against the independent-relaxation
baseline -1.5: the GLOBAL PERIMETER CARRIER is necessary and dominant
(alone: -81.9 of attraction under componentwise setup); the SETUP side
is an interaction/AMPLIFICATION term (-39.1 additional under the global
perimeter carrier, but essentially nothing, -0.1, once the perimeter
carrier is componentwise) -- NOT an independent additive attraction
channel. Note "metric carrier" is a broad call-site label: the setup
masses feed coherence, angle matrices, BEM quadrature and compatibility
alike, so the honest name is SETUP-SIDE carrier. Q's violating candidate appears at every tested dt at
gaps that do not shrink proportionally (0.034/0.017/0.015), with the
checkpoint velocity dt-stable (~80-95): the fast dynamics is not a
step-size artifact, though a continuous-time rejection of Q remains
unclaimed. Factual term: finite-delta carrier-overlap ATTRACTION/BIAS;
"contact-layer regularisation" is a deliberate modeling interpretation,
not the default name.

Original four checks:

  1. EXPLICIT-DELTA SWEEP, N = 128, delta/g0 in {0.5 .. 2.0}: the early
     attraction must switch on across delta ~ g0. Recorded per delta:
     g_dot(first), g_dot(early), the cross-component kernel fraction
         omega_i = sum_{j in other} eta_ij / sum_j eta_ij  (max over i),
     and sum(m_global)/sum(m_componentwise); actual mass_delta/mass_tau
     go to the JSON (review: they were never logged).

  2. MACHINE EQUALITY: the Wendland kernel has support exactly [0, delta],
     so delta < g0 must give m_global == m_componentwise to machine
     precision -- asserted, not eyeballed.

  3. DIAGNOSTIC REAL q ON THE UNIT-q TRAJECTORY: mode B evolves with
     q == 1; recomputing the real coherence on its snapshots (never fed
     back) shows q_min collapsing as the gap closes -- the direct
     visualisation that the q drop is a CONSEQUENCE of the carrier
     attraction, not a cause.

  4. M/Q PRE-CONTACT HARD STOP: both variants stop when
     max|s|/gap > eta_contact = 0.25 (a normal-graph step at contact
     scale means the support comparison is over; negative-gap frames are
     invalid-support data). Where each variant stops is the recorded
     discriminator -- expectation from 1.2: Q degenerates earlier.

Usage
-----
    uv run python scripts/experiments/carrier_range_0c5d12a.py \
        --out results/carrier_range_0c5d12a
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from src.torch.oriented_varifold.mass import (
    compute_kde_density,
    compute_masses,
    compute_recommended_params,
)
from src.torch.solver.mm_step import MMStepper
from src.torch.transport import compute_coherence

sys.path.insert(0, str(Path(__file__).parent))
from spectral_rank_auto_0c5b import cat  # noqa: E402
from two_ellipses_causal_ablation_0c5d12 import run_mode  # noqa: E402
from two_ellipses_paper_precontact import (  # noqa: E402
    D_CENTERS,
    cfg_for,
    make_ellipse,
)

G0 = 0.1
N_PER = 128
ETA_CONTACT = 0.25


def pair_cloud():
    parts = [make_ellipse(N_PER, -D_CENTERS / 2),
             make_ellipse(N_PER, D_CENTERS / 2)]
    return cat(*parts), parts[0].n_points


def carrier_mixing(positions, n1, delta, tau):
    """(max cross-kernel fraction, sum m_global / sum m_componentwise,
    max |m_global - m_cw|, min theta/tau) at this delta, with the SAME
    tau the dynamics uses -- and the cutoff saturation is measured, not
    assumed (review 1.2b-4)."""
    N = positions.shape[0]
    th_all = compute_kde_density(positions, delta)
    th_self = torch.cat([
        compute_kde_density(positions[:n1], delta) * (n1 / N),
        compute_kde_density(positions[n1:], delta) * ((N - n1) / N)])
    omega = (1 - th_self / th_all).clamp(min=0.0)
    m_gl = compute_masses(positions, delta, tau)
    m_cw = torch.cat([compute_masses(positions[:n1], delta, tau),
                      compute_masses(positions[n1:], delta, tau)])
    return (omega.max().item(),
            (m_gl.sum() / m_cw.sum()).item(),
            (m_gl - m_cw).abs().max().item(),
            (th_all.min() / tau).item())


def delta_sweep() -> list:
    v, n1 = pair_cloud()
    # the KDE density magnitude is delta-independent for a curve (the
    # 1/delta normalisation cancels the growing support), so the auto tau
    # is valid across the whole sweep; explicit mass_delta REQUIRES an
    # explicit mass_tau (the stepper skips auto-calibration otherwise)
    _, tau_auto = compute_recommended_params(v.positions)
    rows = []
    from two_ellipses_causal_ablation_0c5d12 import (
        _componentwise_carrier_ctx,
        _null_ctx,
    )

    def three_step_gdot(delta, componentwise):
        cfg = cfg_for(1e-4, "oracle", 2)
        cfg.mass_delta = delta
        cfg.mass_tau = tau_auto
        stepper = MMStepper(cfg)
        vv = v
        gaps = [(vv.positions[n1:, 0].min()
                 - vv.positions[:n1, 0].max()).item()]
        ctx = (_componentwise_carrier_ctx(n1) if componentwise
               else _null_ctx())
        with ctx:
            for _ in range(3):
                r = stepper.step(vv)
                vv = r.varifold
                gaps.append((vv.positions[n1:, 0].min()
                             - vv.positions[:n1, 0].max()).item())
        return ((gaps[1] - gaps[0]) / 1e-4,
                (gaps[3] - gaps[0]) / 3e-4, stepper)

    for ratio in (0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0):
        delta = ratio * G0
        omega, mass_ratio, mass_diff, sat = carrier_mixing(
            v.positions, n1, delta, tau_auto)
        gd1, gd3, stepper = three_step_gdot(delta, False)
        _, gd3_cw, _ = three_step_gdot(delta, True)
        rows.append(dict(
            delta_over_g0=ratio, delta=delta,
            mass_delta_used=stepper.mass_delta,
            mass_tau_used=stepper.mass_tau,
            omega_cross_max=omega,
            mass_ratio_global_over_cw=mass_ratio,
            mass_maxdiff=mass_diff,
            min_theta_over_tau=sat,
            g_dot_first=gd1,
            g_dot_early=gd3,
            g_dot_early_componentwise=gd3_cw,
            delta_g_dot=gd3 - gd3_cw,
        ))
    return rows


def unit_q_diagnostic(n_steps: int = 10) -> list:
    """Mode B trajectory with the real coherence recomputed per snapshot
    (diagnostic only, never fed back). Both time levels are saved
    (review 1.2b-1): `q_min_lagged` mixes the post-step geometry with the
    pre-step fixed masses (the earlier version's inconsistency, kept for
    comparison), while `q_min_current` recomputes the masses on the
    CURRENT geometry with the stepper's own delta/tau first."""
    v, n1 = pair_cloud()
    cfg = cfg_for(1e-4, "oracle", 2)
    cfg.use_unit_coherence = True
    stepper = MMStepper(cfg)
    rows = []
    for k in range(n_steps):
        r = stepper.step(v)
        v = r.varifold
        gap = (v.positions[n1:, 0].min()
               - v.positions[:n1, 0].max()).item()
        kern = stepper.config.perimeter_kernel
        q_lag = compute_coherence(v, r.masses, 0.1, kern)
        m_cur = compute_masses(v.positions, stepper.mass_delta,
                               stepper.mass_tau)
        q_cur = compute_coherence(v, m_cur, 0.1, kern)
        rows.append(dict(t=(k + 1) * 1e-4, gap=gap,
                         q_min_lagged=q_lag.min().item(),
                         q_min_current=q_cur.min().item(),
                         q_med_current=q_cur.median().item()))
    return rows


def hard_stop_run(variant: str, n_steps: int = 40,
                  dt: float = 1e-4) -> dict:
    """M or Q with the pre-contact support/step-size hard stop."""
    v, n1 = pair_cloud()
    cfg = cfg_for(dt, "oracle", 2)
    if variant == "Q":
        cfg.wlin_use_coherence_velocity = True
    stepper = MMStepper(cfg)
    frames = []
    rejected = None
    dt = cfg.time_step
    for k in range(n_steps):
        r = stepper.step(v)
        gap = (r.varifold.positions[n1:, 0].min()
               - r.varifold.positions[:n1, 0].max()).item()
        ratio = r.displacements.abs().max().item() / max(gap, 1e-12)
        if gap <= 0 or ratio > ETA_CONTACT:
            # the violating step is a REJECTED CANDIDATE, not a valid
            # frame: production must roll back / reduce dt / hand off to
            # a support-event handler instead of committing this geometry
            rejected = dict(t=(k + 1) * dt, gap=gap, s_over_gap=ratio,
                            reason=("gap<=0" if gap <= 0
                                    else f"max|s|/gap > {ETA_CONTACT}"))
            break
        v = r.varifold
        frames.append(dict(t=(k + 1) * dt, gap=gap, s_over_gap=ratio,
                           max_v=(r.displacements.abs().max() / dt).item(),
                           q_min=stepper.fixed_coherence.min().item()))
    last = frames[-1] if frames else None
    return dict(variant=variant, dt=dt, frames=frames,
                last_valid_t=last["t"] if last else None,
                last_valid_gap=last["gap"] if last else None,
                rejected_candidate=rejected,
                stop_reason=(rejected["reason"] if rejected
                             else "step cap"))


def two_by_two_split(n_steps: int = 3) -> dict:
    """Item 2: split the carrier effect between the PERIMETER energy and
    the METRIC side. Mode D patched compute_masses everywhere; here the
    componentwise substitution is applied selectively by call site:
    _setup_step's call feeds the frozen masses (BEM operator, quadrature,
    compatibility, coherence input) = the METRIC side; the objective's
    call on current positions feeds the frozen perimeter P = the ENERGY
    side."""
    from unittest.mock import patch as _patch

    from two_ellipses_causal_ablation_0c5d12 import mm_step_mod

    v, n1 = pair_cloud()
    orig_cm = mm_step_mod.compute_masses
    orig_setup = MMStepper._setup_step
    out = {}
    for per_cw, met_cw in ((False, False), (True, False),
                           (False, True), (True, True)):
        state = {"in_setup": False}

        def dispatch(points, delta, tau, kernel="wendland_c2",
                     _state=state, _p=per_cw, _m=met_cw):
            cw = _m if _state["in_setup"] else _p
            if cw and points.shape[0] == 2 * N_PER:
                return torch.cat(
                    [orig_cm(points[:n1], delta, tau, kernel),
                     orig_cm(points[n1:], delta, tau, kernel)])
            return orig_cm(points, delta, tau, kernel)

        def setup_wrap(self, varifold, _state=state):
            _state["in_setup"] = True
            try:
                return orig_setup(self, varifold)
            finally:
                _state["in_setup"] = False

        stepper = MMStepper(cfg_for(1e-4, "oracle", 2))
        vv = v
        gaps = [(vv.positions[n1:, 0].min()
                 - vv.positions[:n1, 0].max()).item()]
        with _patch.object(mm_step_mod, "compute_masses", dispatch),                 _patch.object(MMStepper, "_setup_step", setup_wrap):
            for _ in range(n_steps):
                r = stepper.step(vv)
                vv = r.varifold
                gaps.append((vv.positions[n1:, 0].min()
                             - vv.positions[:n1, 0].max()).item())
        key = f"perimeter_{'cw' if per_cw else 'gl'}_metric_"               f"{'cw' if met_cw else 'gl'}"
        out[key] = (gaps[n_steps] - gaps[0]) / (n_steps * 1e-4)
    return out


def q_dt_refinement() -> list:
    """Item 5: max|s|/gap ~ dt * max|v|/gap, so the fixed-dt hard stop
    shows failure of THAT step size, not yet a continuous-time
    singularity. At each dt, record the VELOCITY max|s|/dt at the first
    valid frame with gap <= the fixed checkpoint: dt-stable velocity that
    grows as q -> 0 means genuine metric degeneracy; a shrinking step
    ratio alone would be a controllable numerical issue."""
    checkpoint = 0.06
    rows = []
    for dt in (1e-4, 5e-5, 2e-5):
        run = hard_stop_run("Q", n_steps=int(round(4e-3 / dt)), dt=dt)
        probe = next((f for f in run["frames"] if f["gap"] <= checkpoint),
                     None)
        rows.append(dict(
            dt=dt, checkpoint_gap=checkpoint,
            v_at_checkpoint=(probe["max_v"] if probe else None),
            s_over_gap_at_checkpoint=(probe["s_over_gap"] if probe
                                      else None),
            q_min_at_checkpoint=(probe["q_min"] if probe else None),
            rejected=run["rejected_candidate"],
            last_valid_gap=run["last_valid_gap"],
        ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print("=== 1: explicit-delta sweep (N = 128, dt = 1e-4; response is "
          "NON-monotone in delta -- threshold closed, force law not) ===")
    print(f"{'d/g0':>5} {'omega_max':>10} {'Sm_gl/Sm_cw':>12} "
          f"{'max|dm|':>9} {'gdot3_gl':>9} {'gdot3_cw':>9} "
          f"{'Dgdot':>8} {'minth/tau':>9}")
    sweep = delta_sweep()
    for r in sweep:
        print(f"{r['delta_over_g0']:>5} {r['omega_cross_max']:>10.2e} "
              f"{r['mass_ratio_global_over_cw']:>12.6f} "
              f"{r['mass_maxdiff']:>9.1e} {r['g_dot_early']:>9.1f} "
              f"{r['g_dot_early_componentwise']:>9.1f} "
              f"{r['delta_g_dot']:>8.1f} "
              f"{r['min_theta_over_tau']:>9.2f}", flush=True)
    for r in sweep:
        assert r["min_theta_over_tau"] > 1.0, \
            f"cutoff NOT saturated at delta/g0={r['delta_over_g0']}"
        if r["delta_over_g0"] <= 1.0:
            assert r["mass_maxdiff"] < 1e-14, \
                f"machine equality violated at delta/g0={r['delta_over_g0']}"
    print("  cutoff saturation min(theta) > tau at every delta: PASS")
    print("  machine equality m_global == m_cw for delta <= g0: PASS")

    print("\n=== 2: perimeter-carrier vs metric-carrier (2x2, 3 steps) "
          "===")
    split = two_by_two_split()
    for k, gd in split.items():
        print(f"  {k}: g_dot = {gd:+.1f}", flush=True)

    print("\n=== 3: diagnostic real q on the unit-q (mode B) trajectory "
          "===")
    diag = unit_q_diagnostic()
    for r in diag[::3] + [diag[-1]]:
        print(f"  t={r['t']:.4f}: gap={r['gap']:+.4f} "
              f"q_min_current={r['q_min_current']:.3f} "
              f"(lagged {r['q_min_lagged']:.3f})", flush=True)

    print("\n=== 4: M/Q pre-contact hard stop (eta_contact = "
          f"{ETA_CONTACT}) ===")
    stops = {}
    for variant in ("M", "Q"):
        s = hard_stop_run(variant)
        stops[variant] = s
        rej = s["rejected_candidate"]
        print(f"  {variant}: last valid t={s['last_valid_t']} "
              f"gap={s['last_valid_gap']} "
              + (f"| REJECTED candidate at t={rej['t']:.4f} "
                 f"gap={rej['gap']:+.4f} ({rej['reason']})" if rej
                 else "| no rejection (step cap)"), flush=True)

    print("\n=== 5: Q dt-refinement at fixed gap checkpoint 0.06 ===")
    qref = q_dt_refinement()
    for r in qref:
        print(f"  dt={r['dt']}: v(checkpoint)={r['v_at_checkpoint']} "
              f"s/gap={r['s_over_gap_at_checkpoint']} "
              f"q_min={r['q_min_at_checkpoint']}", flush=True)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "carrier_range_0c5d12a.json"
        path.write_text(json.dumps(dict(
            meta=dict(N_per_ellipse=N_PER, g0=G0, dt=1e-4,
                      eta_contact=ETA_CONTACT,
                      kernel="wendland_c2 (compact support [0, delta])"),
            delta_sweep=sweep, two_by_two_split=split,
            unit_q_diagnostic=diag,
            hard_stops=stops, q_dt_refinement=qref), indent=2))
        print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
