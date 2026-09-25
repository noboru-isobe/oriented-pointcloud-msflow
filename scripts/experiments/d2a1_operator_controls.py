"""D2a.1: compact operator controls before D2b (review items 1-8).

A. q-confound controls: unit-q / fixed-real-q / refreshed-q for both
   structurally distinct rules at the 800-iteration budget -- the D2a
   long run compared legacy without q-suppression against substep with
   refreshed real q, so the instability attribution needed a
   geometry-only control.
B. step-size / clipping sweep at FIXED redistribution pseudo-time
   s = n_iters * lambda: lambda in {0.01, 0.005, 0.0025} with
   n = s/lambda. If the drift is a property of the (clipped explicit
   map) rather than the underlying vector field, it changes materially
   with lambda at fixed s. Clipping fractions recorded per run
   (preclip/postclip norms are in the trace).
C. one long call vs repeated production-sized calls: 1 x 800 versus
   80 x 10 with the angle correction applied BETWEEN calls (the
   production pattern -- tangents are rebuilt from corrected angles at
   every invocation). The single-call divergence does NOT by itself
   predict production dt-refinement accumulation; this is the direct
   F4 calibration.
D. equal-CV Pareto: drift compared at the same achieved spacing CV
   (not the same iteration count), normalized: |dA|/A, |dP|/P,
   dH/diam.
E. cv naming separation: cv_theta_excluding_self (the operator's
   stopping criterion) vs cv_theta_including_self (the carrier
   diagnostic) -- these differ by more than 2x on the circle fixture.
F. stored-angle vs geometric-tangent mismatch (max degrees, from the
   trace).
G. Algorithm-3 curvature pairing: substep with visible q*m curvature
   weights vs the default raw-m variant (which is deliberately NOT
   called an exact Algorithm-3 implementation).

Usage
-----
    uv run python scripts/experiments/d2a1_operator_controls.py \
        --out results/d2a_redist
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import (
    compute_kde_density,
    compute_masses,
    compute_recommended_params,
)
from src.torch.solver.redistribute import _compute_density_log_gradient
from src.torch.solver.redistribution_rules import redistribute_with_rule
from src.torch.transport import compute_coherence

import sys
sys.path.insert(0, str(Path(__file__).parent))
from d2a_redistribution_calibration import (  # noqa: E402
    DELTA_RATIO,
    MAX_DISP_RATIO,
    SIGMA,
    STEP_SIZE,
    TOL,
    curve_metrics,
    make_fixture,
    sampled_polyline_hausdorff,
)

PSEUDO_TIME = 8.0          # s = n_iters * lambda, fixed for the sweep


def _cv_theta_excl(P, d_redist):
    _, theta = _compute_density_log_gradient(P, d_redist, "wendland_c2")
    return (theta.std() / theta.mean()).item()


def _drift(P0, P1, before):
    m = curve_metrics(P1)
    # TRUE sample diameter (review: the previous bounding-box diagonal
    # over-normalized by sqrt(2) on the circle)
    diam = torch.cdist(P0, P0).max().item()
    return dict(
        cv_l=m["cv_l"],
        dA_rel=abs(m["area"] - before["area"]) / abs(before["area"]),
        dP_rel=abs(m["perimeter"] - before["perimeter"])
        / before["perimeter"],
        dH_over_diam=sampled_polyline_hausdorff(P0, P1) / diam)


def _run(P0, ang0, rule, n_iters, kw, **extra):
    P1, A1, info = redistribute_with_rule(
        P0, ang0, rule, n_iters=n_iters, **kw, **extra)
    return P1, A1, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/d2a_redist"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    P0, ang0 = make_fixture("circle")
    delta, tau = compute_recommended_params(P0)
    d_redist = DELTA_RATIO * delta
    kw = dict(delta=delta, step_size=STEP_SIZE, tol=TOL,
              max_disp_ratio=MAX_DISP_RATIO, mass_tau=tau,
              delta_redist=d_redist)
    before = curve_metrics(P0)
    m0 = compute_masses(P0, delta, tau)
    q0 = compute_coherence(
        OrientedPointCloudVarifold(positions=P0, angles=ang0),
        m0, SIGMA, "wendland_c2")
    out = dict(meta=dict(fixture="circle", n=P0.shape[0], delta=delta,
                         q_input_min=q0.min().item(),
                         pseudo_time=PSEUDO_TIME))

    # E. the two theta-CVs are different quantities
    theta_incl = compute_kde_density(P0, d_redist)
    out["cv_naming"] = dict(
        cv_theta_including_self=(theta_incl.std()
                                 / theta_incl.mean()).item(),
        cv_theta_excluding_self=_cv_theta_excl(P0, d_redist),
        cv_spacing=before["cv_l"])
    print(f"E. cv_theta incl={out['cv_naming']['cv_theta_including_self']:.3f} "
          f"excl={out['cv_naming']['cv_theta_excluding_self']:.3f} "
          f"cv_spacing={before['cv_l']:.3f}")

    # A. q-confound controls at the 800-iteration budget
    print("A. q controls (circle, 800 iters)")
    out["q_controls"] = {}
    for rule in ("legacy_hybrid", "substep_refreshed"):
        for mode, extra in (("unit", dict(q_override="unit")),
                            ("fixed_real",
                             dict(q_override="fixed", coherence=q0)),
                            ("refreshed",
                             dict(sigma=SIGMA)
                             if rule == "substep_refreshed" else None)):
            if extra is None:
                continue
            # declared cyclic order (single-loop generator fixture):
            # the leakage frame is an ordered-curve oracle diagnostic
            loops = [torch.arange(P0.shape[0])]
            P1, _, info = _run(P0, ang0, rule, 800, kw, loops=loops,
                               **extra)
            d = _drift(P0, P1, before)
            d["max_angle_mismatch_deg"] = max(
                r["stored_vs_geom_tangent_max_deg"] for r in info["trace"])
            out["q_controls"][f"{rule}:{mode}"] = d
            print(f"  {rule}:{mode:11s} cv_l={d['cv_l']:.4f} "
                  f"dP/P={d['dP_rel']:.3e} dH/diam={d['dH_over_diam']:.3e} "
                  f"angle_mismatch={d['max_angle_mismatch_deg']:.2f}deg",
                  flush=True)

    # B. lambda/clip sweep at fixed pseudo-time (q == 1)
    print("B. step-size sweep at fixed pseudo-time s = n*lambda = "
          f"{PSEUDO_TIME}")
    out["lambda_sweep"] = {}
    for lam in (0.01, 0.005, 0.0025):
        n_it = round(PSEUDO_TIME / lam)
        for rule in ("legacy_hybrid", "substep_refreshed"):
            kw_l = dict(kw, step_size=lam)
            P1, _, info = _run(P0, ang0, rule, n_it, kw_l,
                               q_override="unit")
            d = _drift(P0, P1, before)
            fr = [r["fraction_above_threshold"] for r in info["trace"]]
            d["mean_fraction_above_threshold"] = sum(fr) / max(len(fr), 1)
            out["lambda_sweep"][f"{rule}:lam{lam}"] = d
            print(f"  {rule}:lam={lam:.4f} n={n_it:4d} "
                  f"cv_l={d['cv_l']:.4f} dP/P={d['dP_rel']:.3e} "
                  f"dH/diam={d['dH_over_diam']:.3e} "
                  f"clip_frac={d['mean_fraction_above_threshold']:.2f}",
                  flush=True)

    # C. one long call vs repeated production-sized calls (q == 1)
    print("C. 1x800 vs 80x10 (angle correction between calls)")
    out["repeated_calls"] = {}
    for rule in ("legacy_hybrid", "substep_refreshed"):
        P_long, _, _ = _run(P0, ang0, rule, 800, kw, q_override="unit")
        Pk, Ak = P0.clone(), ang0.clone()
        for _ in range(80):
            Pk, Ak, _ = redistribute_with_rule(
                Pk, Ak, rule, n_iters=10, q_override="unit", **kw)
        out["repeated_calls"][rule] = dict(
            single_call=_drift(P0, P_long, before),
            repeated_80x10=_drift(P0, Pk, before))
        s, r = (out["repeated_calls"][rule][k]
                for k in ("single_call", "repeated_80x10"))
        print(f"  {rule}: 1x800 cv_l={s['cv_l']:.4f} dP/P={s['dP_rel']:.3e}"
              f" dH/diam={s['dH_over_diam']:.3e} | 80x10 cv_l={r['cv_l']:.4f} "
              f"dP/P={r['dP_rel']:.3e} dH/diam={r['dH_over_diam']:.3e}",
              flush=True)

    # C2. repeated-call saturation: does the K-call map saturate or
    # eventually degrade? (review fix 4)
    print("C2. repeated-call saturation, legacy, K in "
          "(20, 40, 80, 160, 320)")
    out["repeated_call_saturation"] = {}
    Pk, Ak = P0.clone(), ang0.clone()
    K_marks = (20, 40, 80, 160, 320)
    for k in range(1, max(K_marks) + 1):
        Pk, Ak, _ = redistribute_with_rule(
            Pk, Ak, "legacy_hybrid", n_iters=10, q_override="unit", **kw)
        if k in K_marks:
            d = _drift(P0, Pk, before)
            out["repeated_call_saturation"][f"K{k}"] = d
            print(f"  K={k:3d} cv_l={d['cv_l']:.4f} "
                  f"dP/P={d['dP_rel']:.3e} "
                  f"dH/diam={d['dH_over_diam']:.3e}", flush=True)

    # D. equal-CV Pareto: drift at the first checkpoint reaching the
    # target spacing CV (q == 1)
    target_cv = 0.12
    print(f"D. equal-CV comparison (first checkpoint with CV_l <= "
          f"{target_cv})")
    out["equal_cv"] = dict(target_cv=target_cv)
    for rule in ("legacy_hybrid", "substep_refreshed"):
        _, _, info = _run(P0, ang0, rule, 800, kw, q_override="unit",
                          checkpoint_every=25)
        hit = None
        for it, Pc, _a in info["checkpoints"]:
            d = _drift(P0, Pc, before)
            if d["cv_l"] <= target_cv:
                hit = dict(iteration=it, **d)
                break
        out["equal_cv"][rule] = hit
        print(f"  {rule}: {hit}", flush=True)

    # G. Algorithm-3 curvature pairing: visible q*m vs raw m
    print("G. substep curvature weights: raw m vs visible q*m "
          "(800 iters, real refreshed q)")
    out["curvature_pairing"] = {}
    for label, extra in (("raw_m", dict(sigma=SIGMA)),
                         ("visible_qm", dict(sigma=SIGMA,
                                             curvature_visible_qm=True))):
        P1, _, _ = _run(P0, ang0, "substep_refreshed", 800, kw, **extra)
        out["curvature_pairing"][label] = _drift(P0, P1, before)
        d = out["curvature_pairing"][label]
        print(f"  {label:10s} cv_l={d['cv_l']:.4f} "
              f"dP/P={d['dP_rel']:.3e} dH/diam={d['dH_over_diam']:.3e}",
              flush=True)

    path = args.out / "d2a1_operator_controls.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
