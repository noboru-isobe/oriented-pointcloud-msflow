"""D1a-ODE: pure time-level calibration of the dead-point rules.

No optimizer, no BEM: consecutive EXACT annulus ODE snapshots
E(t - dt), E(t) provide the pre/post geometries, so the ONLY difference
between the three rules is the time level of (m, q). Particle layout is
frozen at t = 0 (fixed n_outer/n_inner, fixed angles -- the radial flow
preserves them), so IDs coincide across all snapshots.

Symmetry note (review): on the exactly concentric, uniformly sampled
annulus every inner point carries the SAME score, so first = half = all
deletion. That is the clean calibration, not a failure -- progressive
deletion needs a deliberate nonradial perturbation later.

Primary observable: the threshold-crossing radius R_-^{del} per rule
(zero crossing of the minimum normalized margin, INTERPOLATED in R_-,
which removes most of the snapshot-grid error), swept over dt, N and
rho_dead. The central question: do R_lag / R_semi / R_ref approach each
other as dt -> 0? Linear-in-dt offsets mean the time-level choice is a
finite-dt discretisation difference; distinct limits mean the rule is
part of the discrete model.

Usage
-----
    uv run python scripts/experiments/d1a_ode_calibration.py \
        --out results/d1a_ode
"""

from __future__ import annotations

import argparse
import json
import math
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
)

DT = torch.float64
R_OUT0, R_IN0 = 1.0, 0.5
SIGMA = 0.1


# high-accuracy dense ODE solution shared with Run A (review D1a.1-7:
# a fixed-step RK4 must not be called "exact" near the singular closure)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from annulus_run_a import closure_time, exact_trajectory  # noqa: E402

_T_STAR = closure_time()
_SOL = exact_trajectory(0.999 * _T_STAR)


def rdot_inner(Rp: float, Rm: float) -> float:
    a = (1 / Rp + 1 / Rm) / math.log(Rp / Rm)
    return -a / Rm


def ode_radii(t: float):
    if t > 0.999 * _T_STAR:
        raise ValueError("past closure")
    Rp, Rm = _SOL.sol(t)
    if Rm <= 0.0:
        raise ValueError("past closure")
    return float(Rp), float(Rm)


def snapshot(n_out, n_in, Rp, Rm):
    return generate_oriented_annulus(n_out, Rp, Rm, (0.0, 0.0), "cpu",
                                     DT, n_inner=n_in)


def crossing_radius(n_out: int, dt: float, rho: float) -> dict:
    """First threshold crossing per rule, margin-interpolated in R_-."""
    n_out_, n_in = annulus_point_counts(n_out, R_OUT0, R_IN0)
    v0 = snapshot(n_out_, n_in, R_OUT0, R_IN0)
    delta, tau = compute_recommended_params(v0.positions)

    prev_margin = {r: None for r in DEAD_POINT_RULES}
    prev_Rm = {r: None for r in DEAD_POINT_RULES}
    out = {}
    inner_spread = 0.0
    t = dt
    while len(out) < len(DEAD_POINT_RULES):
        try:
            Rp_pre, Rm_pre = ode_radii(t - dt)
            Rp_post, Rm_post = ode_radii(t)
        except ValueError:
            break                     # marched past hole closure (T*)
        if Rm_post < 0.02:
            # algebraic sweep floor; validity vs the KDE bandwidth is
            # REPORTED (R_del/delta), not silently enforced -- at coarse
            # N the crossing may sit below the Run-A validity boundary
            # R_- >~ 1.2 delta, which is itself a finding
            break
        v_pre = snapshot(n_out_, n_in, Rp_pre, Rm_pre)
        v_post = snapshot(n_out_, n_in, Rp_post, Rm_post)
        from src.torch.oriented_varifold.mass import compute_masses
        from src.torch.transport import compute_coherence
        m_pre = compute_masses(v_pre.positions, delta, tau)
        q_pre = compute_coherence(v_pre, m_pre, SIGMA, "wendland_c2")
        for rule in DEAD_POINT_RULES:
            if rule in out:
                continue
            dec = evaluate_dead_points(
                v_post, m_pre, q_pre, rule, rho,
                delta_for_kde=delta, tau_for_kde=tau, sigma=SIGMA)
            inner = dec.score[n_out_:]
            inner_spread = max(inner_spread,
                               ((inner.max() - inner.min())
                                / inner.max()).item())
            # crossing detected on the INNER-RING margin explicitly
            # (review D1a.1-6): the audit question is inner deletion
            mmin = dec.threshold_margin[n_out_:].min().item()
            if mmin < 0:
                if prev_margin[rule] is not None:
                    w = prev_margin[rule] / (prev_margin[rule] - mmin)
                    R_del = (prev_Rm[rule]
                             + w * (Rm_post - prev_Rm[rule]))
                else:
                    R_del = Rm_post
                inner_all = bool((~dec.keep_mask[n_out_:]).all())
                outer_kept = bool(dec.keep_mask[:n_out_].all())
                if not outer_kept:
                    raise RuntimeError(
                        f"{rule}: outer points deleted at crossing")
                # margin-root time (linear in t between the two frames)
                if prev_margin[rule] is not None:
                    w = prev_margin[rule] / (prev_margin[rule] - mmin)
                    t_root = (t - dt) + w * dt
                else:
                    t_root = t
                out[rule] = dict(
                    status="crossed",
                    R_del=R_del, t_cross=t, t_root=t_root,
                    R_del_over_delta=R_del / delta,
                    n_removed_at_cross=dec.n_removed,
                    all_inner_removed=inner_all,
                    outer_all_kept=outer_kept)
            else:
                prev_margin[rule] = mmin
                prev_Rm[rule] = Rm_post
        t += dt
    for rule in DEAD_POINT_RULES:
        if rule not in out:
            # the sweep stops at the radius floor / 0.999 T*, so absence
            # of a crossing is a statement about THIS WINDOW only
            out[rule] = dict(status="not_reached_in_sweep_window",
                             R_floor=0.02, t_max=0.999 * _T_STAR,
                             last_R_in=prev_Rm[rule],
                             last_margin=prev_margin[rule])
    # exact one-step-delay prediction (review D1a.1-correction 1):
    # Delta R_pred = R_-(t_ref + dt) - R_-(t_ref) with t_ref the
    # refreshed-margin root; Rdot*dt is only its first-order expansion
    ref = out.get("refreshed", {})
    if ref.get("status") == "crossed":
        t_ref = ref["t_root"]
        try:
            R_a = ode_radii(t_ref)[1]
            R_b = ode_radii(t_ref + dt)[1]
            ref_exact_pred = R_b - R_a
        except ValueError:
            ref_exact_pred = None
        if out.get("legacy_lagged", {}).get("status") == "crossed":
            out["legacy_lagged"]["lag_predicted_exact"] = ref_exact_pred
            out["legacy_lagged"]["lag_predicted_first_order"] = (
                rdot_inner(*ode_radii(t_ref)) * dt
                if ref_exact_pred is not None else None)
    return dict(n_outer=n_out_, n_inner=n_in,
                N_total=n_out_ + n_in, dt=dt, rho=rho,
                delta=delta, tau=tau, sigma=SIGMA,
                inner_score_spread_max=inner_spread,
                crossings=out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/d1a_ode"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    print("=== D1a-ODE: threshold-crossing radius per rule ===")
    print(f"{'n_out':>5} {'n_in':>4} {'N_tot':>5} {'dt':>7} {'rho':>4} | "
          f"{'R_lag':>8} {'R_semi':>8} {'R_ref':>8} | {'lag-ref':>9} "
          f"{'pred':>9} {'semi-ref':>9} {'sym':>5}")
    for n_out in (64, 128, 256):
        for dt in (4e-4, 2e-4, 1e-4):
            for rho in (0.2, 0.3):
                r = crossing_radius(n_out, dt, rho)
                rows.append(r)
                c = r["crossings"]
                done = [k for k in c if c[k].get("status") == "crossed"]
                if len(done) < 3:
                    missing = [k for k in c
                               if c[k].get("status") != "crossed"]
                    print(f"{r['n_outer']:>5} {r['n_inner']:>4} "
                          f"{r['N_total']:>5} {dt:>7} {rho:>4} | "
                          f"not_reached_in_sweep_window: {missing}")
                    continue
                Rl = c["legacy_lagged"]["R_del"]
                Rs = c["semi_implicit"]["R_del"]
                Rr = c["refreshed"]["R_del"]
                pred = c["legacy_lagged"].get("lag_predicted_exact")
                pred = pred if pred is not None else float("nan")
                sym = all(c[k]["all_inner_removed"] for k in done)
                print(f"{r['n_outer']:>5} {r['n_inner']:>4} "
                      f"{r['N_total']:>5} {dt:>7} {rho:>4} | {Rl:>8.5f} "
                      f"{Rs:>8.5f} {Rr:>8.5f} | {Rl - Rr:>+9.2e} "
                      f"{pred:>+9.2e} {Rs - Rr:>+9.2e} {str(sym):>5}",
                      flush=True)

    # TWO fit families (review: the earlier version did independent
    # fits but called them common-intercept): (a) independent
    # R_r = R0_r + c_r dt, intercepts compared; (b) genuinely
    # CONSTRAINED R_r = R0 + c_r dt with ONE shared intercept, with both
    # residuals reported. Cells fitted from only two points are flagged.
    print("\n=== linear-in-dt fits (independent vs common-intercept) ===")
    import numpy as np
    groups = {}
    for r in rows:
        groups.setdefault((r["n_outer"], r["rho"]), []).append(r)
    fits = {}
    for (n_out, rho), grp in sorted(groups.items()):
        pts = {rule: [(g["dt"], g["crossings"][rule]["R_del"])
                      for g in grp
                      if g["crossings"][rule].get("status") == "crossed"]
               for rule in DEAD_POINT_RULES}
        cell = dict(n_points={r: len(p) for r, p in pts.items()},
                    two_point_rules=[r for r, p in pts.items()
                                     if len(p) == 2])
        # independent fits
        indep = {}
        for rule, p in pts.items():
            if len(p) < 2:
                continue
            dts = np.array([x[0] for x in p])
            Rs_ = np.array([x[1] for x in p])
            A = np.vstack([np.ones_like(dts), dts]).T
            sol, res, *_ = np.linalg.lstsq(A, Rs_, rcond=None)
            indep[rule] = dict(R0=float(sol[0]), slope=float(sol[1]),
                               residual=float(res[0]) if len(res) else 0.0)
        cell["independent"] = indep
        # constrained common intercept
        if all(len(p) >= 2 for p in pts.values()):
            rules = list(DEAD_POINT_RULES)
            rowsA, y = [], []
            for k, rule in enumerate(rules):
                for dtv, Rv in pts[rule]:
                    a = [1.0] + [0.0] * len(rules)
                    a[1 + k] = dtv
                    rowsA.append(a)
                    y.append(Rv)
            A = np.array(rowsA)
            y = np.array(y)
            sol, res, *_ = np.linalg.lstsq(A, y, rcond=None)
            cell["common"] = dict(
                R0=float(sol[0]),
                slopes={r: float(sol[1 + k])
                        for k, r in enumerate(rules)},
                residual=float(res[0]) if len(res) else 0.0)
        fits[f"{n_out}_{rho}"] = cell
        if indep and len(indep) == 3:
            r0s = [indep[r]["R0"] for r in DEAD_POINT_RULES]
            spread = max(r0s) - min(r0s)
            com = cell.get("common", {})
            flag = (f" [2-pt: {cell['two_point_rules']}]"
                    if cell["two_point_rules"] else "")
            print(f"  n_outer={n_out} rho={rho}: indep R0 spread = "
                  f"{spread:.2e}; common R0 = "
                  f"{com.get('R0', float('nan')):.5f} "
                  f"(resid {com.get('residual', float('nan')):.1e})"
                  f"{flag}", flush=True)
        else:
            print(f"  n_outer={n_out} rho={rho}: insufficient crossings",
                  flush=True)

    # R_del/delta per rho (review D1a.1-2: the two sequences differ)
    print("\n=== R_ref/delta by rho ===")
    for rho in (0.2, 0.3):
        vals = []
        for n_out in (64, 128, 256):
            rr = [r for r in rows if r["n_outer"] >= n_out
                  and r["rho"] == rho and r["dt"] == 1e-4
                  and r["n_outer"] == max(x["n_outer"] for x in rows
                                          if x["n_outer"] <= n_out * 2
                                          and x["n_outer"] >= n_out)]
            row = [r for r in rows
                   if r["rho"] == rho and r["dt"] == 1e-4][0]
        seq = [(r["n_outer"],
                r["crossings"]["refreshed"].get("R_del_over_delta"))
               for r in rows if r["rho"] == rho and r["dt"] == 1e-4]
        print(f"  rho={rho}: " + ", ".join(
            f"n_out={n}: {v:.2f}" if v else f"n_out={n}: --"
            for n, v in seq))

    path = args.out / "d1a_ode_calibration.json"
    path.write_text(json.dumps(dict(
        meta=dict(R_out0=R_OUT0, R_in0=R_IN0, sigma=SIGMA,
                  note=("exact-snapshot calibration: no optimizer, no "
                        "BEM; layout frozen at t=0; symmetric annulus "
                        "=> first = all deletion by design")),
        fits=fits, rows=rows), indent=1))
    print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
