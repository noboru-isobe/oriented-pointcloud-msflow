"""0C-5c: carrier operator fixed, m-flux vs qm-flux compared.

0C-2c showed that q*m operator quadrature can destroy the Neumann nullspace
rank, so BOTH variants here build the panel geometry, quadrature weights,
K*, rank detection, and the bordered solve from the CARRIER measure m
(`bem_operator_measure="carrier"`). The only difference is the flux measure
of the Neumann data, i.e. which velocity the metric transports:

    variant M   V_BEM = v_geom       compatibility = m-flux
    variant Q   V_BEM = q * v_geom   compatibility = qm-flux

The same flux_scale tensor enters the spectral compatibility matrix and the
BEM RHS (hard invariant enforced in the solver: BEMWasserstein.flux_scale
is the single source for both).

This stage does NOT select a production metric. The qm-flux variant is the
candidate consistent with the current-based perimeter interpretation, but
no theorem promotes either tangent metric to THE finite-resolution
extension of one-phase MS; the goal is to establish the behavior of the two
internally consistent variants:

    - M keeps the m-flux leakage at discretisation level,
    - Q keeps the qm-flux leakage at discretisation level,
    - the rank sensor is IDENTICAL between variants (same carrier operator),
    - at q = 1 the two variants coincide exactly,
    - divergence between them is measured under controlled q-dips and under
      real pre-contact coherence.

Leakage is dimensionless per component (never compare an area/time flux
with a point speed):

    ell_alpha^(u) = |sum_{i in E_alpha} u_i s_i| / sum_{i in E_alpha} u_i |s_i|,
    u = m or q*m,

and is interpreted per ACTIVE component only: the activity
a_alpha = sum u_i |s_i| is the leakage denominator, so a near-stationary
component's ratio is residual noise over residual noise. (Measured in the
dip tests: on the ACTIVE dipped component M holds m-leakage at ~1e-5 and Q
holds qm-leakage at ~1e-4 even at depth 0.75; the large ratios live
entirely on the inactive undipped disk.)

Also logs per-step wall-clock cost (BEM assembly / full SVD / bordered LU /
optimizer) -- the full SVD is the candidate bottleneck for long dynamic
runs.

Usage
-----
    uv run python scripts/experiments/spectral_flux_measure_0c5c.py \
        --out results/spectral_0c5c
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from unittest.mock import patch

import torch

from src.torch.solver.mm_step import MMConfig, MMStepper
from src.torch.transport.bem_wasserstein import AmbiguousComponentRankError

import sys
sys.path.insert(0, str(Path(__file__).parent))
from spectral_rank_auto_0c5b import (  # noqa: E402
    cat, circle, component_slices, ellipse, flux_leakage,
    generate_oriented_annulus, n_for,
)


def flux_activity(s, u, slices) -> list:
    """Per-component activity a_alpha = sum_{i in E_alpha} u_i |s_i| -- the
    leakage denominator. A component whose activity is far below the most
    active one has a leakage ratio made of residual noise over residual
    noise; interpret leakage ONLY on active components."""
    return [(u[i:j] * s[i:j].abs()).sum().item() for i, j in slices]


RHO_ACT = 0.05  # component is active if a_alpha >= RHO_ACT * max_beta a_beta

DTYPE = torch.float64
DT = 1e-5
H = 0.05
SIGMA = 0.1


def config(variant: str, unit_q: bool = False) -> MMConfig:
    """M: geometric velocity (m-flux); Q: q-scaled velocity (qm-flux).
    Both on the carrier operator."""
    return MMConfig(time_step=DT,
                    use_unit_coherence=unit_q,
                    perimeter_sigma=SIGMA,
                    angle_sigma=SIGMA,
                    bem_epsilon_mode="carrier_segment_length",
                    bem_solver_mode="spectral_bordered",
                    bem_trace_side="interior",
                    bem_rank_mode="auto",
                    bem_operator_measure="carrier",
                    wlin_use_coherence_velocity=(variant == "Q"),
                    optimizer_method="trust-ncg", optimizer_tol=1e-8,
                    optimizer_max_iter=300)


def run_case(name: str, parts, variant: str, n_steps: int = 3,
             unit_q: bool = False, coherence_fn=None) -> dict:
    """Short evolution; per-step rank evidence, both leakages, timings.
    coherence_fn(varifold, masses, sigma, kernel, backend=...) optionally
    replaces the production coherence (synthetic q-dip injection)."""
    v = cat(*parts)
    slices = component_slices(parts)
    stepper = MMStepper(config(variant, unit_q))
    ctx = (patch("src.torch.solver.mm_step.compute_coherence", coherence_fn)
           if coherence_fn is not None else _null_ctx())
    frames = []
    with ctx:
        for step_idx in range(n_steps):
            t0 = time.perf_counter()
            try:
                r = stepper.step(v)
            except AmbiguousComponentRankError as e:
                frames.append(dict(step=step_idx, ambiguous=str(e)))
                break
            wall = time.perf_counter() - t0
            m = r.masses
            qm = r.effective_masses
            s = r.displacements
            timings = dict(stepper.bem_wasserstein.last_timings or {})
            timings["step_total"] = wall
            frames.append(dict(
                step=step_idx,
                C=r.detected_rank,
                rank_gap_ratio=r.rank_gap_ratio,
                rank_abs_level=r.rank_abs_level,
                subspace_angle_deg=r.subspace_angle_deg,
                max_s_over_dt=(s.abs().max() / DT).item(),
                leakage_m=flux_leakage(s, m, slices),
                leakage_qm=flux_leakage(s, qm, slices),
                activity_m=flux_activity(s, m, slices),
                activity_qm=flux_activity(s, qm, slices),
                q_min=stepper.fixed_coherence.min().item(),
                q_mean=stepper.fixed_coherence.mean().item(),
                objective_final=r.objective,
                gradient_norm_final=r.gradient_norm_final,
                objective_decreased=bool(r.objective_decreased),
                timings=timings,
            ))
            v = r.varifold
    return dict(name=name, variant=variant, N=cat(*parts).n_points,
                frames=frames)


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def dip_coherence(slices, comp: int, depth: float, center):
    """Synthetic q-dip: Gaussian (width 0.4 rad) in polar angle around
    `center` on component `comp`; q = 1 elsewhere."""
    i0, j0 = slices[comp]
    cx, cy = center

    def fn(varifold, masses, sigma, kernel, backend=None):
        pos = varifold.positions
        q = torch.ones(pos.shape[0], dtype=pos.dtype)
        th = torch.atan2(pos[i0:j0, 1] - cy, pos[i0:j0, 0] - cx)
        q[i0:j0] = 1.0 - depth * torch.exp(-(th / 0.4) ** 2)
        return q

    return fn


def fmt_l(ls):
    return "(" + ", ".join(f"{x:.1e}" for x in ls) + ")"


def _worst_active(frames, leak_key, act_key):
    """Worst leakage over ACTIVE components only (activity gate RHO_ACT):
    a near-stationary component's leakage ratio is denominator noise and
    must not dominate the summary. NaN if no component is ever active."""
    vals = []
    for f in frames:
        amax = max(f[act_key])
        vals.extend(l for l, a in zip(f[leak_key], f[act_key])
                    if a >= RHO_ACT * amax)
    return max(vals) if vals else float("nan")


def summarize(run: dict) -> dict:
    ok = [f for f in run["frames"] if "ambiguous" not in f]
    if not ok:
        return dict(ambiguous=True)
    return dict(
        C=sorted({f["C"] for f in ok}),
        worst_leak_m=_worst_active(ok, "leakage_m", "activity_m"),
        worst_leak_qm=_worst_active(ok, "leakage_qm", "activity_qm"),
        q_min=min(f["q_min"] for f in ok),
        ambiguous=len(ok) < len(run["frames"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    all_runs = []

    def case_parts(name):
        if name == "two_ellipses":
            return [ellipse(0.7, 0.4, -1.6), ellipse(1.2, 0.8, 1.8)]
        if name == "annulus_disk":
            return [generate_oriented_annulus(n_for(2 * math.pi), 1.0, 0.5,
                                              (-1.8, 0.0), "cpu", DTYPE),
                    circle(0.6, 1.8)]
        if name == "three_disks":
            return [circle(0.5, -2.5), circle(0.8, 0.0, 2.2),
                    circle(1.0, 2.7)]
        raise KeyError(name)

    print("=== q = 1 equality gate: variants must coincide exactly ===")
    s_ref = {}
    for variant in ("M", "Q"):
        run = run_case("two_ellipses_q1", case_parts("two_ellipses"),
                       variant, n_steps=1, unit_q=True)
        all_runs.append(run)
        s_ref[variant] = run
    lm = s_ref["M"]["frames"][0]
    lq = s_ref["Q"]["frames"][0]
    eq = abs(lm["objective_final"] - lq["objective_final"])
    print(f"  |F_M - F_Q| = {eq:.3e}  "
          f"leak_m: M={fmt_l(lm['leakage_m'])} Q={fmt_l(lq['leakage_m'])}")

    print("\n=== main cases, real coherence (sigma = 0.1), 3 steps ===")
    for name in ("two_ellipses", "annulus_disk", "three_disks"):
        row = {}
        for variant in ("M", "Q"):
            run = run_case(name, case_parts(name), variant)
            all_runs.append(run)
            row[variant] = summarize(run)
        for variant in ("M", "Q"):
            s = row[variant]
            print(f"  {name:>13} {variant}: C={s['C']} "
                  f"leak_m={s['worst_leak_m']:.2e} "
                  f"leak_qm={s['worst_leak_qm']:.2e} "
                  f"q_min={s['q_min']:.3f}", flush=True)
        assert row["M"]["C"] == row["Q"]["C"], "rank sensor must not differ"

    print("\n=== synthetic q-dip on one arc of the large disk "
          "(two disks, gap 2) ===")
    parts = [circle(0.5, -1.5), circle(1.0, 2.0)]
    slices = component_slices(parts)
    for depth in (0.25, 0.5, 0.75):
        for variant in ("M", "Q"):
            fn = dip_coherence(slices, 1, depth, (2.0, 0.0))
            run = run_case(f"qdip_{depth}", parts, variant, n_steps=3,
                           coherence_fn=fn)
            run["dip_depth"] = depth
            all_runs.append(run)
            s = summarize(run)
            print(f"  depth={depth} {variant}: C={s.get('C')} "
                  f"leak_m={s.get('worst_leak_m', float('nan')):.2e} "
                  f"leak_qm={s.get('worst_leak_qm', float('nan')):.2e}",
                  flush=True)

    print("\n=== near-tangent real coherence (two ellipses, shrinking "
          "gap) ===")
    # sigma = 0.1: the coherence dip only appears once gap <~ sigma; the
    # last entries deliberately cross into the KDE-coupled regime
    # (delta ~ 0.25) -- ambiguous-rank stops there are legitimate output.
    for gap in (0.6, 0.3, 0.15, 0.08):
        parts = [ellipse(0.7, 0.4, -0.7 - gap / 2),
                 ellipse(1.2, 0.8, 1.2 + gap / 2)]
        for variant in ("M", "Q"):
            run = run_case(f"near_tangent_gap{gap}", parts, variant,
                           n_steps=3)
            run["gap"] = gap
            all_runs.append(run)
            s = summarize(run)
            tag = " AMBIGUOUS-STOP" if s.get("ambiguous") else ""
            print(f"  gap={gap} {variant}: C={s.get('C')} "
                  f"leak_m={s.get('worst_leak_m', float('nan')):.2e} "
                  f"leak_qm={s.get('worst_leak_qm', float('nan')):.2e} "
                  f"q_min={s.get('q_min', float('nan')):.3f}{tag}",
                  flush=True)

    print("\n=== timing profile (three_disks variant M, per step) ===")
    tim = [f["timings"] for r in all_runs
           if r["name"] == "three_disks" and r["variant"] == "M"
           for f in r["frames"] if "timings" in f]
    for k in ("assembly", "svd", "lu", "step_total"):
        vals = [t.get(k, float("nan")) for t in tim]
        print(f"  {k:>10}: " + " ".join(f"{v:.3f}s" for v in vals))

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "spectral_flux_measure_0c5c.json"
        path.write_text(json.dumps(dict(
            meta=dict(dt=DT, h=H, sigma=SIGMA,
                      operator_measure="carrier",
                      rank_mode="auto", rank_null_tol=0.05,
                      rank_gap_min=30.0,
                      note="no production metric selected at this stage"),
            runs=all_runs), indent=2))
        print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
