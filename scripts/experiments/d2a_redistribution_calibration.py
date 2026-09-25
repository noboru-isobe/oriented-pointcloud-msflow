"""D2a: redistribution-only operator calibration (no MM evolution).

The redistribution operator is a REPARAMETERIZATION: it must equalize
the spacing (CV_l down) while leaving the geometric support unmoved.
This script measures exactly that, per time-level rule, on nonuniformly
sampled closed analytic curves with EXACT normals.

Rules (shared helper `redistribute_with_rule`, whose legacy branch is
bitwise the production `redistribute_points` -- pinned here as the
operator wiring gate):

    legacy_hybrid      fixed input tangents, caller (stale) q, final
                       angle correction from input geometry
    post_mm_frozen     same loop; m, q recomputed ONCE on the input
    substep_refreshed  tangent/q/curvature refreshed every subiteration,
                       incremental angle correction

SCOPING (stated before measurement): at operator level there is no MM
step, so when the caller passes q(input) the legacy_hybrid and
post_mm_frozen branches COINCIDE BY CONSTRUCTION -- their difference is
purely the staleness of q across an MM step and is measured in the
in-solver audit (D2b), not here. The operator-level axis that can
genuinely differ is frozen-tangent vs substep-refresh.

Observables per fixture x rule:
    CV_l before/after           consecutive-spacing CV (NOT the KDE
                                theta-CV the operator minimizes -- both
                                recorded)
    max_i |dx_i . n_i|, ||dx_n||_2, ||dx_t||_2
                                displacement split against the INPUT
                                polygon's central-difference normals
    d_H(Gamma_before, Gamma_after)   symmetric polyline Hausdorff
    dA_geom, dP_geom, d c_phase      shoelace area / polygon perimeter /
                                     phase-centroid drift
    quality = (CV_l reduction) / (normal leakage + geometric drift)

Usage
-----
    uv run python scripts/experiments/d2a_redistribution_calibration.py \
        --out results/d2a_redist
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.torch.diagnostics import polygon_stats
from src.torch.oriented_varifold.mass import (
    compute_kde_density,
    compute_recommended_params,
)
from src.torch.solver.redistribute import redistribute_points
from src.torch.solver.redistribution_rules import (
    REDISTRIBUTION_RULES,
    redistribute_with_rule,
)

DT = torch.float64
SIGMA = 0.1
WARP = 0.35            # spacing ratio (1+a)/(1-a) ~ 2.1
N_ITERS = 50
STEP_SIZE = 0.01       # production default
TOL = 1e-4             # production default
MAX_DISP_RATIO = 0.05  # production default
DELTA_RATIO = 0.5      # production delta_redist = 0.5 * delta


def _warped(n: int) -> torch.Tensor:
    u = (torch.arange(n, dtype=DT) + 0.5) / n
    return 2 * math.pi * u + WARP * torch.sin(2 * math.pi * u)


def _from_param(x: torch.Tensor, y: torch.Tensor):
    """Positions + exact outward-normal angles from analytic tangents
    (CCW curve: n = (t_y, -t_x))."""
    P = torch.stack([x, y], dim=1)
    t = P.roll(-1, 0) - P.roll(1, 0)          # central difference
    ang = torch.atan2(-t[:, 0], t[:, 1])
    return P, ang


def make_fixture(name: str):
    if name == "circle":
        t = _warped(128)
        x, y = torch.cos(t), torch.sin(t)
        P = torch.stack([x, y], 1)
        return P, t.remainder(2 * math.pi)     # exact normal angle = t
    if name == "ellipse":
        t = _warped(128)
        a, b = 1.2, 0.6
        x, y = a * torch.cos(t), b * torch.sin(t)
        P = torch.stack([x, y], 1)
        # exact outward normal of the ellipse: (b cos t, a sin t)
        ang = torch.atan2(a * torch.sin(t), b * torch.cos(t))
        return P, ang
    if name == "flower":
        t = _warped(192)
        r = 1.0 + 0.3 * torch.cos(5 * t)
        x, y = r * torch.cos(t), r * torch.sin(t)
        # exact tangent: x' = r'(cos,sin) + r(-sin,cos); n = (t_y, -t_x)
        rp = -1.5 * torch.sin(5 * t)
        tx = rp * torch.cos(t) - r * torch.sin(t)
        ty = rp * torch.sin(t) + r * torch.cos(t)
        P = torch.stack([x, y], 1)
        return P, torch.atan2(-tx, ty)
    raise ValueError(name)


def curve_metrics(P: torch.Tensor) -> dict:
    seg = (P.roll(-1, 0) - P).norm(dim=1)
    A, c = polygon_stats(P)
    return dict(cv_l=(seg.std() / seg.mean()).item(),
                perimeter=seg.sum().item(), area=A, centroid=c)


def sampled_polyline_hausdorff(P: torch.Tensor, Q: torch.Tensor) -> float:
    """Symmetric VERTEX-to-polyline distance (review: not the exact
    polyline-polyline Hausdorff -- segment-interior-to-segment-interior
    maxima are not sampled; adequate at these resolutions)."""
    def one_way(A, B):
        b0, b1 = B, B.roll(-1, 0)
        d = b1 - b0
        ap = A[:, None, :] - b0[None, :, :]
        tt = ((ap * d[None]).sum(-1)
              / (d * d).sum(-1)[None].clamp_min(1e-30)).clamp(0, 1)
        proj = b0[None] + tt[..., None] * d[None]
        return (A[:, None, :] - proj).norm(dim=-1).min(dim=1).values.max()
    return max(one_way(P, Q).item(), one_way(Q, P).item())


def displacement_split(P0: torch.Tensor, P1: torch.Tensor) -> dict:
    t = P0.roll(-1, 0) - P0.roll(1, 0)
    t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-30)
    n = torch.stack([-t[:, 1], t[:, 0]], dim=1)
    dx = P1 - P0
    dn, dtt = (dx * n).sum(1), (dx * t).sum(1)
    return dict(max_abs_dx_n=dn.abs().max().item(),
                l2_dx_n=dn.norm().item(), l2_dx_t=dtt.norm().item())


def run_fixture(name: str) -> dict:
    P0, ang0 = make_fixture(name)
    delta, tau = compute_recommended_params(P0)
    d_redist = DELTA_RATIO * delta
    kw = dict(delta=delta, n_iters=N_ITERS, step_size=STEP_SIZE, tol=TOL,
              max_disp_ratio=MAX_DISP_RATIO, mass_tau=tau,
              delta_redist=d_redist)
    theta0 = compute_kde_density(P0, d_redist)
    before = curve_metrics(P0)
    before["cv_theta"] = (theta0.std() / theta0.mean()).item()

    out = dict(fixture=name, n=P0.shape[0], delta=delta, tau=tau,
               delta_redist=d_redist, before=before, rules={})

    # q(input) once: legacy's caller-supplied q for the parity scoping
    from src.torch.oriented_varifold.mass import compute_masses
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    from src.torch.transport import compute_coherence
    m0 = compute_masses(P0, delta, tau)
    q0 = compute_coherence(
        OrientedPointCloudVarifold(positions=P0, angles=ang0),
        m0, SIGMA, "wendland_c2")
    out["q_input_min"] = q0.min().item()

    # operator wiring gate: legacy branch == production, bitwise
    p_prod, a_prod, _ = redistribute_points(
        P0, ang0, delta, "wendland_c2", N_ITERS, STEP_SIZE, TOL,
        MAX_DISP_RATIO, tau, d_redist, coherence=q0)
    p_leg, a_leg, info_leg = redistribute_with_rule(
        P0, ang0, "legacy_hybrid", coherence=q0, **kw)
    out["parity_production_bitwise"] = bool(
        torch.equal(p_prod, p_leg) and torch.equal(a_prod, a_leg))

    for rule in REDISTRIBUTION_RULES:
        if rule == "legacy_hybrid":
            P1, A1, info = p_leg, a_leg, info_leg
        else:
            P1, A1, info = redistribute_with_rule(
                P0, ang0, rule, coherence=q0, sigma=SIGMA, **kw)
        after = curve_metrics(P1)
        theta1 = compute_kde_density(P1, d_redist)
        after["cv_theta"] = (theta1.std() / theta1.mean()).item()
        split = displacement_split(P0, P1)
        dH = sampled_polyline_hausdorff(P0, P1)
        drift = dict(
            dA=after["area"] - before["area"],
            dP=after["perimeter"] - before["perimeter"],
            dc=math.hypot(after["centroid"][0] - before["centroid"][0],
                          after["centroid"][1] - before["centroid"][1]),
            hausdorff=dH)
        cv_red = before["cv_l"] - after["cv_l"]
        leak = split["max_abs_dx_n"] + dH
        out["rules"][rule] = dict(
            after=after, split=split, drift=drift,
            n_iters=info["n_iters"], converged=info["converged"],
            cv_theta_history_first_last=[info["cv_history"][0],
                                         info["cv_history"][-1]],
            max_normal_update_per_iter=max(
                (r["normal_update_norm"] for r in info["trace"]),
                default=0.0),
            quality=cv_red / leak if leak > 0 else float("inf"),
            trace=info["trace"])
    # scoping pin: legacy == frozen bitwise at operator level
    pf, af, _ = redistribute_with_rule(
        P0, ang0, "post_mm_frozen", coherence=q0, sigma=SIGMA, **kw)
    out["legacy_equals_frozen_bitwise"] = bool(
        torch.equal(p_leg, pf) and torch.equal(a_leg, af))
    return out


def convergence_limit(n_iters: int = 2000) -> dict:
    """Where does the operator actually converge? The loop equalizes
    the SMOOTHED density theta_{delta_redist}, not the spacing, so with
    delta_redist >> h the spacing CV should hit a bandwidth-limited
    floor > 0 long before theta-CV reaches the 1e-4 tolerance -- and
    the geometric drift keeps accumulating while it grinds. Measured on
    the circle fixture for the two structurally distinct rules."""
    P0, ang0 = make_fixture("circle")
    delta, tau = compute_recommended_params(P0)
    d_redist = DELTA_RATIO * delta
    out = dict(n_iters=n_iters, checkpoints={})
    for rule in ("legacy_hybrid", "substep_refreshed"):
        # q_override="unit" for BOTH rules: the earlier version compared
        # legacy without q-suppression against substep with refreshed
        # real q -- a coherence confound (review); the geometry-only
        # control removes it
        P1, A1, info = redistribute_with_rule(
            P0, ang0, rule, q_override="unit", delta=delta,
            n_iters=n_iters, step_size=STEP_SIZE, tol=TOL,
            max_disp_ratio=MAX_DISP_RATIO, mass_tau=tau,
            delta_redist=d_redist)
        after = curve_metrics(P1)
        out["checkpoints"][rule] = dict(
            cv_l_final=after["cv_l"],
            cv_theta_history_sampled=info["cv_history"][::100]
            + [info["cv_history"][-1]],
            n_iters=info["n_iters"], converged=info["converged"],
            hausdorff=sampled_polyline_hausdorff(P0, P1),
            dA=after["area"] - curve_metrics(P0)["area"],
            dP=after["perimeter"] - curve_metrics(P0)["perimeter"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/d2a_redist"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = dict(meta=dict(warp=WARP, n_iters=N_ITERS,
                             step_size=STEP_SIZE, tol=TOL,
                             max_disp_ratio=MAX_DISP_RATIO,
                             delta_ratio=DELTA_RATIO, sigma=SIGMA))
    for name in ("circle", "ellipse", "flower"):
        r = run_fixture(name)
        results[name] = r
        print(f"=== {name} (N={r['n']}, delta={r['delta']:.4f}, "
              f"CV_l before={r['before']['cv_l']:.4f}, "
              f"CV_theta before={r['before']['cv_theta']:.4f}) ===")
        print(f"  parity production bitwise: "
              f"{r['parity_production_bitwise']}   "
              f"legacy==frozen bitwise: "
              f"{r['legacy_equals_frozen_bitwise']}")
        for rule, d in r["rules"].items():
            print(f"  {rule:18s} CV_l {d['after']['cv_l']:.4f}  "
                  f"CV_th {d['after']['cv_theta']:.4f}  "
                  f"max|dx.n| {d['split']['max_abs_dx_n']:.2e}  "
                  f"dH {d['drift']['hausdorff']:.2e}  "
                  f"dA {d['drift']['dA']:+.2e}  "
                  f"dP {d['drift']['dP']:+.2e}  "
                  f"iters {d['n_iters']}  q {d['quality']:.1f}",
                  flush=True)

    cl = convergence_limit()
    results["convergence_limit_circle"] = cl
    print("=== convergence limit (circle, q=1, 2000 iters) ===")
    for rule, d in cl["checkpoints"].items():
        print(f"  {rule:18s} CV_l floor {d['cv_l_final']:.4f}  "
              f"converged(theta tol)={d['converged']} "
              f"@{d['n_iters']}  dH {d['hausdorff']:.2e}  "
              f"dA {d['dA']:+.2e}  dP {d['dP']:+.2e}", flush=True)

    path = args.out / "d2a_calibration.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
