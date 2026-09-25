"""D1a-MM: shadow audit of the dead-point rules on the corrected
annulus MM trajectory.

The exact-snapshot calibration (D1a-ODE) established the one-step-delay
structure on the prescribed radial path. Here the SAME shared helper
shadow-evaluates all three rules on each post-MM geometry of the
corrected rank-1 annulus evolution -- deletion and redistribution OFF,
fixed scales, fixed particle IDs -- so the identity and the margin
curves can be checked where the geometry is produced by the actual
solver.

WIRING GATE (review): with no post-processing the committed post-MM
geometry of step n IS the pre-MM geometry of step n+1, so

    e_lag^{n+1} == e_ref^n     (bitwise: same function, same tensors)

must hold on the MM path exactly as on the ODE path. If it fails, the
time-level wiring or the scale handling is inconsistent -- the run
aborts.

Comparisons are reported at EQUAL R_- (not equal time): margin curves
R_- -> margin_rule(R_-), with the ODE-calibration curves as reference,
which separates MM time-integration error from the rule time-level
difference. R_+^{MM}(R_-) is recorded next to sqrt(C + R_-^2) so area
drift is visible in the same frame.

Validity separation (review): frames are classified against the Run-A
style window -- spatial R_- >= 1.2 delta, step ratio, shape CV, area
drift -- and any shadow crossing beyond it is ALGORITHMIC CLOSURE-LAYER
data, not a Mullins-Sekerka statement.

Usage
-----
    uv run python scripts/experiments/d1a_mm_shadow.py \
        --out results/d1a_mm [--n-outer 256] [--dt 2e-5]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from src.torch.oriented_varifold.mass import compute_recommended_params
from src.torch.shapes.generator import (
    annulus_point_counts,
    generate_oriented_annulus,
)
from src.torch.solver.dead_point_rules import (
    DEAD_POINT_RULES,
    evaluate_dead_points,
    mask_symmetric_difference,
)
from src.torch.solver.mm_step import MMConfig, MMStepper

sys.path.insert(0, str(Path(__file__).parent))
from d1a_ode_calibration import ode_radii, rdot_inner  # noqa: E402

DT_TENSOR = torch.float64
R_OUT0, R_IN0 = 1.0, 0.5
SIGMA = 0.1
RHOS = (0.2, 0.3)


def make_config(dt, delta, tau):
    return MMConfig(
        time_step=dt,
        bem_trace_side="interior",
        bem_solver_mode="spectral_bordered",
        bem_rank_mode="oracle",
        bem_component_rank=1,
        bem_operator_measure="carrier",
        wlin_use_coherence_velocity=False,      # M variant
        bem_epsilon_mode="carrier_segment_length",
        perimeter_sigma=SIGMA,
        angle_sigma=SIGMA,
        use_unit_coherence=False,               # real q: scores need it
        mass_bandwidth_type="global",
        mass_delta=delta, mass_tau=tau,
        mass_delta_adaptive=False,
        redistribute=False, remove_dead_points=False,
        optimizer_method="trust-ncg", optimizer_tol=1e-8,
        optimizer_max_iter=300,
    )


def run(n_outer: int, dt: float, t_end: float, out_dir: Path) -> dict:
    n_out, n_in = annulus_point_counts(n_outer, R_OUT0, R_IN0)
    v = generate_oriented_annulus(n_out, R_OUT0, R_IN0, (0.0, 0.0),
                                  "cpu", DT_TENSOR, n_inner=n_in)
    delta, tau = compute_recommended_params(v.positions)
    stepper = MMStepper(make_config(dt, delta, tau))
    Csq = R_OUT0 ** 2 - R_IN0 ** 2

    frames = []
    prev_ref = None
    identity_checked = 0
    n_steps = round(t_end / dt)
    for k in range(n_steps):
        r = stepper.step(v)
        post = r.varifold
        m_pre, q_pre = r.masses, stepper.fixed_coherence

        # --- one-step identity gate: e_lag^{n+1} == e_ref^n ----------
        # NOTE (review): with no post-processing this identity follows
        # ALGEBRAICALLY from the definitions -- the gate verifies that
        # the IMPLEMENTATION realizes the intended time levels, it is
        # not a dynamical discovery. Score, margin AND mask are pinned,
        # for every rho, and it holds only between lagged and refreshed
        # (semi-implicit has no index-shift identity).
        if prev_ref is not None:
            for rho in RHOS:
                lag_now = evaluate_dead_points(
                    post, m_pre, q_pre, "legacy_lagged", rho)
                pr = prev_ref[rho]
                if not (torch.equal(lag_now.score, pr.score)
                        and torch.equal(lag_now.threshold_margin,
                                        pr.threshold_margin)
                        and torch.equal(lag_now.keep_mask,
                                        pr.keep_mask)):
                    raise RuntimeError(
                        f"one-step identity BROKEN at step {k}, "
                        f"rho={rho}: max|e_lag^(n+1) - e_ref^n| = "
                        f"{(lag_now.score - pr.score).abs().max():.2e}"
                        " -- time-level wiring or scale handling is "
                        "inconsistent")
            identity_checked += 1

        decs = {}
        for rho in RHOS:
            for rule in DEAD_POINT_RULES:
                decs[(rule, rho)] = evaluate_dead_points(
                    post, m_pre, q_pre, rule, rho,
                    delta_for_kde=delta, tau_for_kde=tau, sigma=SIGMA)
        prev_ref = {rho: decs[("refreshed", rho)] for rho in RHOS}

        pos = post.positions
        Rm = pos[n_out:].norm(dim=1).mean().item()
        Rp = pos[:n_out].norm(dim=1).mean().item()
        cv_in = (pos[n_out:].norm(dim=1).std() / Rm).item()
        area = Rp ** 2 - Rm ** 2   # relative proxy; polygon below
        X, Y = pos[:, 0], pos[:, 1]

        try:
            rdot = rdot_inner(Rp, Rm)
        except ValueError:
            rdot = float("nan")
        smax = r.displacements.abs().max().item()
        fr = dict(
            step=k, t=(k + 1) * dt, R_in=Rm, R_out=Rp,
            R_out_area_pred=math.sqrt(Csq + Rm * Rm),
            R_in_over_delta=Rm / delta, R_in_over_sigma=Rm / SIGMA,
            eta_step=smax / Rm,
            eta_lag=abs(rdot) * dt / Rm,
            cv_inner=cv_in,
            relgrad=r.relative_gradient_norm,
            objective_decreased=bool(r.objective_decreased),
            q_min_used_pre=q_pre[n_out:].min().item(),
            q_min_current_post=decs[("refreshed", RHOS[0])]
            .q_used[n_out:].min().item(),
            m_min_used_pre=m_pre[n_out:].min().item(),
            m_min_current_post=decs[("refreshed", RHOS[0])]
            .m_used[n_out:].min().item(),
        )
        for rho in RHOS:
            for rule in DEAD_POINT_RULES:
                d = decs[(rule, rho)]
                fr[f"margin_{rule}_rho{rho}"] = \
                    d.threshold_margin[n_out:].min().item()
            fr[f"symdiff_lag_ref_rho{rho}"] = len(
                mask_symmetric_difference(decs[("legacy_lagged", rho)],
                                          decs[("refreshed", rho)]))
            fr[f"symdiff_semi_ref_rho{rho}"] = len(
                mask_symmetric_difference(decs[("semi_implicit", rho)],
                                          decs[("refreshed", rho)]))
        # CONSERVATIVE carrier-bandwidth gate (review: NOT the full
        # Run-A t_valid -- local h, eps_BEM, area/ODE error, optimizer
        # and support diagnostics would be needed for that)
        fr["carrier_resolved"] = bool(
            Rm >= 1.2 * delta and fr["eta_step"] <= 0.1
            and cv_in <= 0.05)
        frames.append(fr)
        v = post
        if Rm < 0.115:            # past the rho=0.3 ODE crossing radius
            break

    return dict(n_outer=n_out, n_inner=n_in, N_total=n_out + n_in,
                dt=dt, delta=delta, tau=tau, sigma=SIGMA,
                identity_checked_steps=identity_checked,
                frames=frames)


def crossing_roots(d: dict) -> dict:
    """Interpolated margin roots in R_in per (rule, rho), with brackets
    (the first-negative-frame convention carries an O(|Rdot| dt)
    frame-width error -- review correction 3)."""
    fr = d["frames"]
    out = {}
    for rho in RHOS:
        for rule in DEAD_POINT_RULES:
            key = f"margin_{rule}_rho{rho}"
            idx = next((i for i, f in enumerate(fr) if f[key] < 0), None)
            if idx is None or idx == 0:
                out[f"{rule}_rho{rho}"] = None
                continue
            a, b = fr[idx - 1], fr[idx]
            w = a[key] / (a[key] - b[key])
            out[f"{rule}_rho{rho}"] = (
                a["R_in"] + w * (b["R_in"] - a["R_in"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/d1a_mm"))
    ap.add_argument("--n-outer", type=int, default=256)
    ap.add_argument("--dt", type=float, default=2e-5)
    ap.add_argument("--t-end", type=float, default=0.029)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = run(args.n_outer, args.dt, args.t_end, args.out)
    fr = d["frames"]
    print(f"n_outer={d['n_outer']} dt={d['dt']}: {len(fr)} frames, "
          f"one-step identity verified on {d['identity_checked_steps']} "
          f"steps")
    resolved = [f for f in fr if f["carrier_resolved"]]
    print(f"resolved frames: {len(resolved)} "
          f"(last resolved R_in = "
          f"{resolved[-1]['R_in'] if resolved else None})")
    crossings = {}
    for rho in RHOS:
        for rule in DEAD_POINT_RULES:
            key = f"margin_{rule}_rho{rho}"
            idx = next((i for i, f in enumerate(fr) if f[key] < 0), None)
            if idx is None or idx == 0:
                crossings[f"{rule}_rho{rho}"] = dict(
                    status="not_reached_in_sweep_window",
                    last_R_in=fr[-1]["R_in"],
                    last_margin=fr[-1][key])
                continue
            a, b = fr[idx - 1], fr[idx]
            w = a[key] / (a[key] - b[key])
            R_root = a["R_in"] + w * (b["R_in"] - a["R_in"])
            crossings[f"{rule}_rho{rho}"] = dict(
                status="crossed", R_cross_interpolated=R_root,
                R_before=a["R_in"], R_after=b["R_in"],
                margin_before=a[key], margin_after=b[key],
                carrier_resolved_at_cross=b["carrier_resolved"])
        c = crossings[f"refreshed_rho{rho}"]
        if c["status"] == "crossed":
            tag = ("carrier-resolved" if c["carrier_resolved_at_cross"]
                   else "closure-layer (algorithmic)")
            print(f"rho={rho}: refreshed root R_in="
                  f"{c['R_cross_interpolated']:.5f} "
                  f"(bracket [{c['R_after']:.5f}, {c['R_before']:.5f}])"
                  f" [{tag}]")
        else:
            print(f"rho={rho}: no shadow crossing in window "
                  f"(last R_in={fr[-1]['R_in']:.4f})")

    path = args.out / f"d1a_mm_n{d['n_outer']}_dt{d['dt']}.json"
    path.write_text(json.dumps(d, indent=1))
    print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
