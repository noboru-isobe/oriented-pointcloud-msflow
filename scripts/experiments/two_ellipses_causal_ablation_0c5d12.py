"""0C-5d-1.2: causal ablation of the fast finite-sigma pair attraction.

The paper-configuration pair run (gap 0.1 ~ sigma) closes its gap several
times faster than the isolated-relaxation prediction while q_min
collapses. That measured DEVIATION from superposition bundles four
couplings; correlation with q_min does not identify the cause. Here the
same initial cloud, same (N, dt, eps) runs under five modes that unbundle
them:

    A  real q, global BIE, global carrier          (baseline, variant M)
    B  q == 1, global BIE, global carrier          -> coherent-perimeter
                                                      contribution
    C  real q, BLOCK-DIAGONAL S/K*, global carrier -> BIE cross-block
    D  real q, global BIE, COMPONENTWISE carrier   -> KDE cross-component
       (KDE densities computed per component;         carrier coupling
        the normalisation cancels so D == A exactly
        when the kernels do not reach across)
    E  real q, global BIE, global carrier, VARIANT Q (V = q v_geom)

plus (N, dt) refinement of mode A. Stated predictions before
measurement: g_dot_early(A) ~ -100, B at the independent-relaxation scale
~ -1 (coherence channel), C ~ A, D between, E faster.

MEASURED OUTCOME (the prediction was WRONG about the channel): A -122.5,
B -124.2, C -131.1, E -125.1 -- but D = -1.5. The effect is the
FINITE-DELTA CARRIER-OVERLAP ATTRACTION (a finite-bandwidth contact-layer
regularisation induced by the global inverse-density carrier estimator:
two sheets within delta share density -> masses drop -> the estimated
perimeter drops, the dynamic realisation of F8), NOT the coherent energy;
the q_min collapse seen earlier was a consequence of the attraction
pulling the boundaries into sigma range, not a cause. Consistent with the
threshold picture, at N = 256 (delta ~ 0.09 < gap) mode A itself runs at
-1.5; the fixed-N explicit-delta sweep of 0C-5d-1.2a closes the range
mechanism. Claim discipline (review-fixed): (i) the effect vanishes at
fixed positive gap as delta -> 0 and sits OUTSIDE the presently proved
fully discrete evolution theory -- it is not plain physical contact
dynamics; (ii) the archived N = 64 run starts deep inside the
carrier-overlap regime (delta ~ 0.36), but it also used the legacy
trace/global metric and post-processing, so its timing is a COMPOSITE
finite-resolution quantity, not attributable solely to the carrier;
(iii) the -122.5 rate is causal-contrast data at fixed dt, NOT a
time-converged contact velocity (dt sweep: -122.5/-114.2/-100.1);
(iv) mode E establishes PRE-CONTACT DEGENERATION of variant Q (step/gap
reaches O(1) before contact; the negative-gap frames are invalid-support
data, not post-contact dynamics) -- Q is demoted to an exploratory
relaxed metric, M is the post-contact baseline.

Per-run diagnostics (review 3): worst relative gradient, max optimizer
iterations, max |s|/h, max |s|/gap, q quantiles, per-component polygon
centroid velocity.

Usage
-----
    uv run python scripts/experiments/two_ellipses_causal_ablation_0c5d12.py \
        --out results/two_ellipses_causal_0c5d12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

import src.torch.transport.bem_wasserstein as bw
from src.torch.solver.mm_step import MMStepper

# `import src.torch.solver.mm_step as X` would bind the package ATTRIBUTE
# mm_step, which the solver __init__ rebinds to the legacy function of the
# same name -- fetch the module object itself for patching.
mm_step_mod = sys.modules[MMStepper.__module__]

sys.path.insert(0, str(Path(__file__).parent))
from spectral_rank_auto_0c5b import cat  # noqa: E402
from two_ellipses_paper_precontact import (  # noqa: E402
    D_CENTERS,
    PERI,
    cfg_for,
    make_ellipse,
    poly_slice,
)

MODES = ("A", "B", "C", "D", "E")


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _blockdiag_ctx(cut):
    orig = bw.build_bem_matrices_point

    def blocked(pos, nor, w, eps):
        S, K_star = orig(pos, nor, w, eps)
        S = S.clone()
        K_star = K_star.clone()
        S[:cut, cut:] = 0.0
        S[cut:, :cut] = 0.0
        K_star[:cut, cut:] = 0.0
        K_star[cut:, :cut] = 0.0
        return S, K_star

    return patch.object(bw, "build_bem_matrices_point", blocked)


def _componentwise_carrier_ctx(n1):
    orig = mm_step_mod.compute_masses

    def cw(points, delta, tau, kernel):
        # per-component KDE: the 1/N_c normalisation cancels against the
        # density's, so this equals the global call exactly when the
        # kernels do not reach across the gap
        return torch.cat([orig(points[:n1], delta, tau, kernel),
                          orig(points[n1:], delta, tau, kernel)])

    return patch.object(mm_step_mod, "compute_masses", cw)


def run_mode(mode: str, n: int, dt: float, n_steps: int = 10) -> dict:
    parts = [make_ellipse(n, -D_CENTERS / 2), make_ellipse(n, D_CENTERS / 2)]
    v = cat(*parts)
    n1 = parts[0].n_points
    h = PERI / n
    K = 3

    cfg = cfg_for(dt, "oracle", 2)
    if mode == "B":
        cfg.use_unit_coherence = True
    if mode == "E":
        cfg.wlin_use_coherence_velocity = True   # variant Q
    ctx = _null_ctx()
    if mode == "C":
        ctx = _blockdiag_ctx(n1 * K)
    if mode == "D":
        ctx = _componentwise_carrier_ctx(n1)

    stepper = MMStepper(cfg)
    gap0 = (v.positions[n1:, 0].min() - v.positions[:n1, 0].max()).item()
    gaps, frames = [gap0], []
    c0 = [poly_slice(v, 0, n1)[1][0], poly_slice(v, n1, v.n_points)[1][0]]
    with ctx:
        for k in range(n_steps):
            r = stepper.step(v)
            v = r.varifold
            gap = (v.positions[n1:, 0].min()
                   - v.positions[:n1, 0].max()).item()
            gaps.append(gap)
            q = stepper.fixed_coherence
            s = r.displacements
            frames.append(dict(
                t=(k + 1) * dt, gap=gap,
                q_quantiles=[q.min().item(),
                             q.quantile(0.1).item(),
                             q.median().item()],
                max_s_over_h=(s.abs().max().item() / h),
                max_s_over_gap=(s.abs().max().item() / max(gap, 1e-12)),
                relative_gradient_norm=r.relative_gradient_norm,
                n_iter=int(r.n_iter),
                objective_decreased=bool(r.objective_decreased),
            ))
    c1 = [poly_slice(v, 0, n1)[1][0], poly_slice(v, n1, v.n_points)[1][0]]
    T = n_steps * dt
    return dict(
        mode=mode, N_per_ellipse=n, dt=dt, h=h, gap0=gap0,
        g_dot_first_step=(gaps[1] - gaps[0]) / dt,
        g_dot_early=(gaps[3] - gaps[0]) / (3 * dt),
        gap_final=gaps[-1],
        centroid_velocity_x=[(c1[0] - c0[0]) / T, (c1[1] - c0[1]) / T],
        worst_relative_gradient=max(f["relative_gradient_norm"]
                                    for f in frames),
        max_n_iter=max(f["n_iter"] for f in frames),
        worst_s_over_h=max(f["max_s_over_h"] for f in frames),
        worst_s_over_gap=max(f["max_s_over_gap"] for f in frames),
        all_decreased=all(f["objective_decreased"] for f in frames),
        q_min_final=frames[-1]["q_quantiles"][0],
        frames=frames,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = {}

    print("=== modes A-E, N = 128, dt = 1e-4, 10 steps ===")
    hdr = (f"{'mode':>5} {'gdot(1st)':>10} {'gdot(3)':>9} {'gap_end':>8} "
           f"{'q_min':>6} {'|s|/h':>6} {'relgrad':>8} {'cx_vel':>18}")
    print(hdr)
    for mode in MODES:
        r = run_mode(mode, 128, 1e-4)
        out[f"mode_{mode}"] = r
        cx = ", ".join(f"{c:+.2f}" for c in r["centroid_velocity_x"])
        print(f"{mode:>5} {r['g_dot_first_step']:>10.1f} "
              f"{r['g_dot_early']:>9.1f} {r['gap_final']:>8.4f} "
              f"{r['q_min_final']:>6.3f} {r['worst_s_over_h']:>6.3f} "
              f"{r['worst_relative_gradient']:>8.1e} ({cx})", flush=True)

    print("\n=== mode A refinement ===")
    for n, dt in ((128, 5e-5), (128, 2e-5), (256, 1e-4), (256, 5e-5)):
        r = run_mode("A", n, dt)
        out[f"A_N{n}_dt{dt}"] = r
        print(f"  N={n} dt={dt}: gdot(1st)={r['g_dot_first_step']:.1f} "
              f"gdot(3)={r['g_dot_early']:.1f} q_min={r['q_min_final']:.3f}"
              f" |s|/h={r['worst_s_over_h']:.3f}", flush=True)

    a = out["mode_A"]["g_dot_early"]
    b = out["mode_B"]["g_dot_early"]
    print(f"\n  attribution: gdot_early A - B = {a - b:.1f} "
          f"(coherence channel), C - A = "
          f"{out['mode_C']['g_dot_early'] - a:.1f} (BIE cross-block), "
          f"D - A = {out['mode_D']['g_dot_early'] - a:.1f} (KDE carrier), "
          f"E - A = {out['mode_E']['g_dot_early'] - a:.1f} (M vs Q)")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "two_ellipses_causal_ablation_0c5d12.json"
        path.write_text(json.dumps(dict(
            meta=dict(config="paper: a=0.4 b=1.0 centers +-0.45 gap 0.1",
                      modes=dict(
                          A="real q, global BIE, global carrier, M",
                          B="unit q, global, global, M",
                          C="real q, block-diagonal S/K*, global, M",
                          D="real q, global BIE, componentwise carrier, M",
                          E="real q, global, global, Q")),
            **out), indent=2))
        print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
