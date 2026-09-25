"""Annulus Run A: q = 1 benchmark of the MM step against the exact radial ODE.

The connected-phase (C = 1) benchmark that runs independently of the 0C
production questions. Exact solution (F1):

    a = (1/R+ + 1/R-) / log(R+/R-),   dR+/dt = -a/R+,   dR-/dt = -a/R-,

area pi (R+^2 - R-^2) conserved, finite closure time T* = 0.029864 for
R+ = 1, R- = 0.5.

Configuration, per the 0A gate: interior trace, carrier-segment epsilon
(q-independent), fixed angle_sigma, unit coherence, no redistribution, no
deletion. The one-step pilot must pass before any trajectory is run:

    (R_pm^1 - R_pm^0) / dt   vs   -a / R_pm,

together with max|s| / R-, per-ring noncircularity, centroid drift,
componentwise geometric area, and the 0A final-point optimizer diagnostics
(none of which existed when earlier runs quietly returned converged=False).

Claims discipline (review corrections to the 84ecff0 commit message):

  * "between second and third order" was premature -- two spatial levels at
    fixed dt cannot assign an order, and the formal per-quantity exponents
    disagree (p ~ 1.7 for R+, 3.1 for R-, 3.7 for area), suggesting mixed
    symmetry cancellations and a floor from the fixed temporal error. Until
    a third spatial level and a temporal refinement exist, say only:
    errors decrease strongly under spatial refinement.
  * machine-precision noncircularity proves the EXACTLY RADIAL subspace is
    numerically invariant, not that the annulus is stable to nonradial
    perturbations; --perturb-k / --perturb-eps runs measure the latter.
  * the disappearance of the old chronic converged=False cannot be
    attributed to the trace fix alone: trace side and gtol changed
    simultaneously. The corrected radial setup with gtol=1e-8 has clean
    final-point diagnostics; the two contributions remain to be isolated.
  * outer-ring k=2: the measured endpoint ratio x0.991 is a 0.9% change
    over the window -- too small to separate from optimizer/spatial noise,
    so say "no growth was observed over the tested interval", not "decay".
    (Inner-ring and higher-k decays are unambiguous.)
  * the validity block initially used eps_BEM ~ 0.1 h_out as an
    approximation; postprocess now reconstructs the ACTUAL per-frame scales
    (delta, tau_cut, eps_BEM, h_-) from the stored radii, which is exact for
    these runs because the clouds stay radial to 1e-16.

Usage
-----
    uv run python scripts/experiments/annulus_run_a.py --one-step
    uv run python scripts/experiments/annulus_run_a.py \
        --n-steps 400 --out results/annulus_run_a
    uv run python scripts/experiments/annulus_run_a.py \
        --postprocess results/annulus_run_a/annulus_run_a_n128_dt1e-05.json
    uv run python scripts/experiments/annulus_run_a.py \
        --summarize results/annulus_run_a
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.torch.shapes.generator import annulus_point_counts, generate_oriented_annulus
from src.torch.solver.mm_step import MMConfig, MMStepper

DTYPE = torch.float64
R_OUT, R_IN = 1.0, 0.5


def exact_coefficient(rp: float, rm: float) -> float:
    return (1.0 / rp + 1.0 / rm) / math.log(rp / rm)


def ode_rhs(rp: float, rm: float) -> tuple[float, float]:
    a = exact_coefficient(rp, rm)
    return -a / rp, -a / rm


def closure_time(rp0: float = R_OUT, rm0: float = R_IN) -> float:
    """T* from the exact quadrature (F1)."""
    from scipy.integrate import quad
    Csq = rp0 * rp0 - rm0 * rm0
    f = (lambda r: r * math.log(math.sqrt(Csq + r * r) / r)
         / ((Csq + r * r) ** -0.5 + 1.0 / r))
    val, _ = quad(f, 0.0, rm0, limit=200)
    return val


def exact_trajectory(t_end: float):
    """Dense exact solution of the radial ODE on [0, t_end]."""
    from scipy.integrate import solve_ivp
    return solve_ivp(lambda t, y: list(ode_rhs(*y)), [0.0, t_end],
                     [R_OUT, R_IN], rtol=1e-12, atol=1e-14,
                     dense_output=True)


def postprocess(json_path: Path) -> None:
    """Attach the exact comparison to a saved run, in the committed pipeline
    (previously the endpoint errors were computed in ad-hoc post-processing,
    so a third party could not regenerate the table from the commit)."""
    d = json.loads(Path(json_path).read_text())
    dt = d["time_step"]
    frames = d["frames"]
    t_end = dt * (len(frames) - 1)
    sol = exact_trajectory(t_end)
    t_star = closure_time()
    for i, f in enumerate(frames):
        t = i * dt
        rp, rm = (float(x) for x in sol.sol(t))
        f["t"] = t
        f["t_over_Tstar"] = t / t_star
        f["R_out_exact"] = rp
        f["R_in_exact"] = rm
        f["rel_error_out"] = abs(f["R_out"] - rp) / rp
        f["rel_error_in"] = abs(f["R_in"] - rm) / rm
    last = frames[-1]
    d["T_star"] = t_star
    d["endpoint"] = dict(
        t=last["t"], t_over_Tstar=last["t_over_Tstar"],
        rel_error_out=last["rel_error_out"],
        rel_error_in=last["rel_error_in"],
        area_drift=last.get("area_drift"),
    )
    # validity analysis: which of the t_valid criteria actually binds. The
    # scales delta, tau_cut and eps_BEM are RECONSTRUCTED per sampled frame
    # by rebuilding the radially symmetric annulus at the stored radii and
    # running the actual estimators -- exact for these runs, whose clouds
    # stay radial to 1e-16 (would not be valid after any symmetry breaking).
    h_out = 2 * math.pi * R_OUT / d["n_outer"]
    steps = [f for f in frames if "rel_error_in" in f and "step" in f]
    cross = next((f for f in steps if f["rel_error_in"] > 1e-2), None)

    def measured_scales(frame) -> dict:
        from src.torch.oriented_varifold import (
            compute_masses, compute_recommended_params)
        n_o, n_i = d["n_outer"], d["n_inner"]
        rp, rm = frame["R_out"], frame["R_in"]
        th_o = (torch.arange(n_o, dtype=DTYPE) + 0.5) / n_o * 2 * math.pi
        th_i = (torch.arange(n_i, dtype=DTYPE) + 0.5) / n_i * 2 * math.pi
        pos = torch.cat([
            torch.stack([rp * torch.cos(th_o), rp * torch.sin(th_o)], 1),
            torch.stack([rm * torch.cos(th_i), rm * torch.sin(th_i)], 1)])
        delta, tau = compute_recommended_params(pos)
        m = compute_masses(pos, delta, tau)
        eps = 0.1 * m.median().item()      # carrier_segment_length mode
        return dict(t_over_Tstar=frame["t_over_Tstar"],
                    delta=float(delta), tau_cut=float(tau),
                    eps_bem=eps,
                    h_in=2 * math.pi * rm / n_i,
                    R_in=rm)

    sample_idx = list(range(0, len(steps), max(1, len(steps) // 8)))
    scales = [measured_scales(steps[i]) for i in sample_idx]
    cross_scales = None if cross is None else measured_scales(cross)
    angle_cross = next(
        (f for f in steps
         if f["R_in_exact"] < d.get("angle_sigma", 0.1)), None)
    d["validity"] = dict(
        h_out=h_out,
        measured_scales=scales,
        max_eta_step=max(f["max_step_over_Rin"] for f in steps),
        max_noncirc_in=max(f["noncirc_in"] for f in steps),
        n_conv_false=sum(1 for f in steps if not f["converged"]),
        all_objective_decreased=all(f["objective_decreased"] for f in steps),
        angle_sigma_crossing=None if angle_cross is None
        else angle_cross["t_over_Tstar"],
        e_in_1pct_crossing=None if cross is None else dict(
            t_over_Tstar=cross["t_over_Tstar"],
            R_in_exact=cross["R_in_exact"],
            R_in_over_h_out=cross["R_in_exact"] / h_out,
            R_in_over_h_in=cross["R_in_exact"] / cross_scales["h_in"],
            R_in_over_eps_measured=cross["R_in_exact"]
            / cross_scales["eps_bem"],
            R_in_over_delta_measured=cross["R_in_exact"]
            / cross_scales["delta"],
            scales=cross_scales,
        ),
    )
    Path(json_path).write_text(json.dumps(d, indent=2))
    print(f"postprocessed {json_path}: t/T* = {last['t_over_Tstar']:.3f}  "
          f"e_out = {last['rel_error_out']:.2e}  "
          f"e_in = {last['rel_error_in']:.2e}")


def summarize(directory: Path) -> None:
    """Cross-run endpoint table with observed refinement ratios. Ratios are
    reported as ratios; no convergence order is fitted here (see the claims
    block above)."""
    runs = []
    for p in sorted(Path(directory).glob("annulus_run_a_*.json")):
        d = json.loads(p.read_text())
        if "endpoint" not in d:
            print(f"  (skipping {p.name}: not postprocessed)")
            continue
        runs.append(dict(name=p.name, n=d["n_outer"], dt=d["time_step"],
                         **d["endpoint"]))
    runs.sort(key=lambda r: (r["dt"], -r["n"]))
    print(f"{'run':>38} {'t/T*':>6} {'e_out':>10} {'e_in':>10} {'areadrift':>10}")
    for r in runs:
        print(f"{r['name']:>38} {r['t_over_Tstar']:>6.3f} "
              f"{r['rel_error_out']:>10.2e} {r['rel_error_in']:>10.2e} "
              f"{r['area_drift']:>10.2e}")
    for a, b in zip(runs, runs[1:]):
        if a["dt"] == b["dt"] and abs(a["t"] - b["t"]) < 1e-12:
            print(f"  ratio {b['name']} -> {a['name']}: "
                  f"e_out x{b['rel_error_out'] / a['rel_error_out']:.1f}, "
                  f"e_in x{b['rel_error_in'] / a['rel_error_in']:.1f}")

    # common-time comparison: runs with different end times must never be
    # compared at their endpoints, so slice all runs at shared t/T* marks
    print("\ncommon-time errors (e_in at t/T* marks):")
    marks = (0.27, 0.5, 0.7, 0.8)
    common = {}
    for p in sorted(Path(directory).glob("annulus_run_a_*.json")):
        dd = json.loads(p.read_text())
        if "endpoint" not in dd or dd.get("perturb_k"):
            continue
        row = {}
        fr = [f for f in dd["frames"] if "rel_error_in" in f]
        for mtag in marks:
            f = min(fr, key=lambda x: abs(x["t_over_Tstar"] - mtag))
            if abs(f["t_over_Tstar"] - mtag) < 0.02:
                row[mtag] = f["rel_error_in"]
        common[p.name] = row
    hdr = "".join(f"{m:>11}" for m in marks)
    print(f"{'run':>38}{hdr}")
    for name, row in common.items():
        cells = "".join(f"{row[m]:>11.2e}" if m in row else f"{'--':>11}"
                        for m in marks)
        print(f"{name:>38}{cells}")

    out = Path(directory) / "summary.json"
    out.write_text(json.dumps(dict(endpoints=runs, common_time=common),
                              indent=2))
    print(f"summary -> {out}")


def perturbed_annulus(n_outer: int, k: int, eps: float):
    """Annulus with r -> R (1 + eps cos(k theta)) on both rings, with the
    EXACT normals of the perturbed curves (a polar curve r(theta) has
    tangent (r' cos - r sin, r' sin + r cos) and outward-normal angle
    atan2(-t_x, t_y); the inner ring is flipped by pi as usual)."""
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    n_out, n_in = annulus_point_counts(n_outer, R_OUT, R_IN)
    pos, ang = [], []
    for n, R, inward in ((n_out, R_OUT, False), (n_in, R_IN, True)):
        th = (torch.arange(n, dtype=DTYPE) + 0.5) / n * 2 * math.pi
        r = R * (1 + eps * torch.cos(k * th))
        dr = -R * eps * k * torch.sin(k * th)
        x, y = r * torch.cos(th), r * torch.sin(th)
        tx = dr * torch.cos(th) - r * torch.sin(th)
        ty = dr * torch.sin(th) + r * torch.cos(th)
        a = torch.atan2(-tx, ty) + (math.pi if inward else 0.0)
        pos.append(torch.stack([x, y], 1))
        ang.append(a)
    return OrientedPointCloudVarifold(
        positions=torch.cat(pos), angles=torch.cat(ang)), n_out, n_in


def fourier_amplitude(positions: torch.Tensor, sel: torch.Tensor,
                      k: int) -> float:
    """|r_hat_k| of one ring about its centroid."""
    p = positions[sel]
    c = p.mean(0)
    d = p - c
    th = torch.atan2(d[:, 1], d[:, 0])
    r = d.norm(dim=1)
    rc = r - r.mean()
    n = len(r)
    a = (rc * torch.cos(k * th)).sum() * 2 / n
    b = (rc * torch.sin(k * th)).sum() * 2 / n
    return float(torch.hypot(a, b))


def make_config(args) -> MMConfig:
    return MMConfig(
        time_step=args.time_step,
        use_unit_coherence=True,
        bem_trace_side="interior",
        bem_epsilon_mode="carrier_segment_length",
        angle_sigma=args.angle_sigma,
        redistribute=False,
        remove_dead_points=False,
        optimizer_method="trust-ncg",
        optimizer_tol=args.gtol,
        optimizer_max_iter=args.max_iter,
    )


def ring_stats(positions: torch.Tensor, sel: torch.Tensor) -> dict:
    """Weighted best-fit-circle statistics of one ring (uniform weights are
    adequate here: the sampling is uniform and stays so without
    redistribution)."""
    p = positions[sel]
    centroid = p.mean(0)
    r = (p - centroid).norm(dim=1)
    return dict(radius=r.mean().item(),
                noncircularity=(r.std() / r.mean()).item(),
                centroid_drift=centroid.norm().item())


def ring_area(positions: torch.Tensor, sel: torch.Tensor) -> float:
    """Shoelace of one ring in its sampling order (valid: each ring is a
    single closed curve sampled in parameter order)."""
    p = positions[sel]
    x, y = p[:, 0], p[:, 1]
    return 0.5 * torch.abs((x * y.roll(-1) - x.roll(-1) * y).sum()).item()


def diagnostics(varifold, outer_sel, inner_sel) -> dict:
    so = ring_stats(varifold.positions, outer_sel)
    si = ring_stats(varifold.positions, inner_sel)
    return dict(
        R_out=so["radius"], R_in=si["radius"],
        noncirc_out=so["noncircularity"], noncirc_in=si["noncircularity"],
        centroid_out=so["centroid_drift"], centroid_in=si["centroid_drift"],
        area_geom=ring_area(varifold.positions, outer_sel)
        - ring_area(varifold.positions, inner_sel),
    )


def run(args) -> dict:
    if getattr(args, "perturb_k", 0):
        varifold, n_out, n_in = perturbed_annulus(
            args.n_outer, args.perturb_k, args.perturb_eps)
    else:
        n_out, n_in = annulus_point_counts(args.n_outer, R_OUT, R_IN)
        varifold = generate_oriented_annulus(args.n_outer, R_OUT, R_IN,
                                             device="cpu", dtype=DTYPE)
    outer_sel = torch.arange(n_out + n_in) < n_out
    inner_sel = ~outer_sel

    stepper = MMStepper(make_config(args))
    d0 = diagnostics(varifold, outer_sel, inner_sel)
    if getattr(args, "perturb_k", 0):
        # baseline at t = 0, so ratios are |r_k(t)| / |r_k(0)| exactly
        d0["fourier_out"] = fourier_amplitude(
            varifold.positions, outer_sel, args.perturb_k)
        d0["fourier_in"] = fourier_amplitude(
            varifold.positions, inner_sel, args.perturb_k)
    area0 = d0["area_geom"]

    frames = [d0]
    t0 = time.time()
    for k in range(args.n_steps):
        result = stepper.step(varifold)
        varifold = result.varifold
        d = diagnostics(varifold, outer_sel, inner_sel)
        prev = frames[-1]
        slope_out = (d["R_out"] - prev["R_out"]) / args.time_step
        slope_in = (d["R_in"] - prev["R_in"]) / args.time_step
        ode_out, ode_in = ode_rhs(prev["R_out"], prev["R_in"])
        d.update(
            step=k + 1,
            slope_out=slope_out, slope_in=slope_in,
            ode_out=ode_out, ode_in=ode_in,
            slope_ratio_out=slope_out / ode_out,
            slope_ratio_in=slope_in / ode_in,
            max_step_over_Rin=(result.displacements.abs().max()
                               / prev["R_in"]).item(),
            area_drift=abs(d["area_geom"] - area0) / area0,
            relative_gradient_norm=result.relative_gradient_norm,
            objective_decreased=result.objective_decreased,
            converged=bool(result.converged),
            n_iter=int(result.n_iter),
        )
        if getattr(args, "perturb_k", 0):
            d["fourier_out"] = fourier_amplitude(
                varifold.positions, outer_sel, args.perturb_k)
            d["fourier_in"] = fourier_amplitude(
                varifold.positions, inner_sel, args.perturb_k)
        frames.append(d)
        if args.one_step or (k + 1) % args.report_every == 0:
            print(f"step {k+1:>4}: R+ {d['R_out']:.6f} (dR/dt {slope_out:+.4f}"
                  f" vs ODE {ode_out:+.4f}, ratio {d['slope_ratio_out']:.4f})"
                  f"  R- {d['R_in']:.6f} ({slope_in:+.4f} vs {ode_in:+.4f},"
                  f" ratio {d['slope_ratio_in']:.4f})", flush=True)
            print(f"          noncirc ({d['noncirc_out']:.2e},"
                  f" {d['noncirc_in']:.2e})  centroid ({d['centroid_out']:.1e},"
                  f" {d['centroid_in']:.1e})  area drift {d['area_drift']:.2e}"
                  f"  max|s|/R- {d['max_step_over_Rin']:.2e}")
            print(f"          rel_grad {d['relative_gradient_norm']:.2e}"
                  f"  decreased={d['objective_decreased']}"
                  f"  conv={d['converged']}  n_iter={d['n_iter']}")
    elapsed = time.time() - t0

    if getattr(args, "perturb_k", 0):
        a0_out, a0_in = frames[0]["fourier_out"], frames[0]["fourier_in"]
        aN_out = frames[-1].get("fourier_out", float("nan"))
        aN_in = frames[-1].get("fourier_in", float("nan"))
        print(f"perturbation k={args.perturb_k}, eps={args.perturb_eps}: "
              f"|r_k| outer {a0_out:.3e} -> {aN_out:.3e} "
              f"(x{aN_out / a0_out:.3f}), inner {a0_in:.3e} -> {aN_in:.3e} "
              f"(x{aN_in / a0_in:.3f})")

    summary = dict(
        n_outer=n_out, n_inner=n_in, time_step=args.time_step,
        gtol=args.gtol, angle_sigma=args.angle_sigma,
        n_steps=args.n_steps, seconds=elapsed,
        perturb_k=getattr(args, "perturb_k", 0),
        perturb_eps=getattr(args, "perturb_eps", 0.0),
        config=dict(trace_side="interior",
                    epsilon_mode="carrier_segment_length",
                    unit_coherence=True, redistribution=False,
                    deletion=False),
        frames=frames,
    )
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        tag = f"n{args.n_outer}_dt{args.time_step:g}"
        if getattr(args, "perturb_k", 0):
            tag += f"_k{args.perturb_k}_eps{args.perturb_eps:g}"
        path = args.out / f"annulus_run_a_{tag}.json"
        path.write_text(json.dumps(summary, indent=2))
        if not getattr(args, "perturb_k", 0):
            postprocess(path)      # exact comparison lives in the pipeline
        print(f"raw results -> {path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-outer", type=int, default=128)
    ap.add_argument("--time-step", type=float, default=1e-5)
    ap.add_argument("--gtol", type=float, default=1e-8)
    ap.add_argument("--angle-sigma", type=float, default=0.1)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--one-step", action="store_true")
    ap.add_argument("--n-steps", type=int, default=1)
    ap.add_argument("--report-every", type=int, default=25)
    ap.add_argument("--perturb-k", type=int, default=0,
                    help="Fourier mode of an initial radial perturbation")
    ap.add_argument("--perturb-eps", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--postprocess", type=Path, default=None,
                    help="attach the exact ODE comparison to a saved run")
    ap.add_argument("--summarize", type=Path, default=None,
                    help="cross-run endpoint table for a results directory")
    args = ap.parse_args()
    if args.postprocess is not None:
        postprocess(args.postprocess)
        return
    if args.summarize is not None:
        summarize(args.summarize)
        return
    if args.one_step:
        args.n_steps = 1
    run(args)


if __name__ == "__main__":
    main()
