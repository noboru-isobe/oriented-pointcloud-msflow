"""D2b-2.1: Grid-B regeneration (corrected ODE reference) and the
N-refinement separation of the warp-relaxation initialization layer.

1. Grid B rerun: run_one now references ode_radii_from_state (the
   committed generator previously compared an R_-(t0)=0.35 start
   against the canonical R_-(0)=0.5 trajectory -- a 1e-1-scale
   reference error; the +1.2e-3 row in the earlier JSON came from a
   post-hoc patch, not the generator).

2. Refinement of the warp layer, in TWO modes and TWO measures
   (review closure patch 1 + 6):
     E_N        |R_fit_err| of the every-step warped run -- the TOTAL
                radius error (MM + BEM + sampling + redistribution +
                bandwidth regularization), used only for the coarse
                mode comparison;
     B_N^red    R_fit(warped, every_step) - R_fit(warped, none) -- the
                DIFFERENTIAL against the matching no-redistribution
                trajectory: the redistribution-attributable part.
   Runs are FAIL-CLOSED: a row is valid only if it completed all
   requested steps with no stop_reason and no timeline errors; E/B are
   None on invalid rows and adjacent rates are computed only between
   valid rows. Modes:
     production_coupled   recommended delta/tau at each N
                          (N=128 row reused from grid A; N=256 from
                          grid C; N=512 run here)
     fixed_delta_sigma    the N=128 delta/tau reused at N=256, 512
   Rates p = log2(E_coarse/E_fine) per adjacent valid pair and mode.
   Interpretation ceiling (review): "consistent with a
   bandwidth-regularization initialization layer" -- E_N is a total
   error, sigma is never varied, and no rate or joint-limit claim is
   made.

Resolution labels: n_outer / n_inner / N_total (never a bare N).

Usage
-----
    uv run python scripts/experiments/d2b21_refinement.py \
        --out results/d2b_redist
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import d2b0_timelevel_shadow as d2b0  # noqa: E402
from d2b1_active_rules import ode_radii_from_state, run_one  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)


def _row_valid(r) -> bool:
    """Fail-closed validity (review closure patch 1): a row enters the
    E/rate computation only if it completed all requested steps with no
    structured stop reason and no timeline errors."""
    return (r["n_steps_completed"] == r["n_steps_requested"]
            and r.get("stop_reason") is None
            and r["timeline_errors"] == [])

T_END = 0.004
DT = 2e-5
SCHEDULES_B = (("none", {}),
               ("every_step", dict(schedule="every_step")),
               ("fixed_physical_interval",
                dict(schedule="fixed_physical_interval",
                     phys_interval=1e-4)),
               ("adaptive_cv_trigger",
                dict(schedule="adaptive_cv_trigger", trigger_cv=0.05)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/d2b_redist"))
    args = ap.parse_args()
    path = args.out / "d2b2_schedule_tradeoff.json"
    out = json.loads(path.read_text())

    n_steps = round(T_END / DT)

    # reference-state pin: t0 shift reproduces the initial state
    Rp0, Rm0 = ode_radii_from_state(0.0, 0.35)
    assert abs(Rm0 - 0.35) < 1e-9
    assert abs(Rp0 - math.sqrt(0.75 + 0.35 ** 2)) < 1e-9
    out["meta"]["gridB_reference_pin"] = dict(Rm0=Rm0, Rp0=Rp0)

    print("=== grid B regeneration (corrected in-generator reference)"
          " ===")
    v0, n_out, _ = d2b0.snapshot(0.5)
    delta, tau = compute_recommended_params(v0.positions)
    for name, kw in SCHEDULES_B:
        rule = "none" if name == "none" else "legacy_hybrid"
        r = run_one(rule, 0.35, DT, n_steps, delta, tau, Rm0=0.35, **kw)
        r["valid"] = _row_valid(r)
        out[f"B|dt2e-05|warp0.35|{name}"] = r
        print(f"  B|{name:26s} inv={r['invocations']:4d} "
              f"Rfit_err={r['R_fit_inner_err']:+.2e} "
              f"cv_in={r['cv_l_inner_final']:.4f} "
              f"valid={r['valid']}", flush=True)

    print("=== N refinement of the warp layer (every_step, warped, "
          "dt=2e-5; fail-closed) ===")
    ref = dict(production_coupled={}, fixed_delta_sigma={})
    delta128, tau128 = delta, tau

    def _pair(r_red, r_none):
        """Validity-gated row: total E plus the redistribution
        differential B_red against the matching none run."""
        v_red, v_none = _row_valid(r_red), _row_valid(r_none)
        return dict(
            valid=v_red,
            E=(abs(r_red["R_fit_inner_err"]) if v_red else None),
            B_red=((r_red["R_fit_inner"] - r_none["R_fit_inner"])
                   if (v_red and v_none) else None),
            delta=r_red.get("_delta"),
            n_outer=r_red["n_outer"], n_inner=r_red["n_inner"],
            N_total=r_red["N_total"],
            invalid_reason=(None if v_red else dict(
                completed=r_red["n_steps_completed"],
                requested=r_red["n_steps_requested"],
                stop=r_red["stop_reason"],
                errs=len(r_red["timeline_errors"]))))

    for n in (128, 256, 512):
        d2b0.N_OUTER = n
        v0n, _, _ = d2b0.snapshot(0.5)
        dn, tn = compute_recommended_params(v0n.positions)
        for mode, dd, tt in (("production_coupled", dn, tn),
                             ("fixed_delta_sigma", delta128, tau128)):
            if mode == "fixed_delta_sigma" and n == 128:
                ref[mode]["n128"] = dict(
                    same_as="production_coupled n128 (delta128 by "
                            "definition)")
                continue
            r_none = run_one("none", 0.35, DT, n_steps, dd, tt)
            r_red = run_one("legacy_hybrid", 0.35, DT, n_steps, dd, tt)
            r_red["_delta"] = dd
            row = _pair(r_red, r_none)
            ref[mode][f"n{n}"] = row
            print(f"  {mode:20s} n_outer={n} valid={row['valid']} "
                  f"E={row['E']} B_red={row['B_red']} delta={dd:.4f}",
                  flush=True)
    d2b0.N_OUTER = 128
    ref["fixed_delta_sigma"]["n128"] = dict(
        ref["production_coupled"]["n128"])

    def _rate(mode, a, b):
        ra, rb = ref[mode].get(a), ref[mode].get(b)
        Ea = ra.get("E") if ra else None
        Eb = rb.get("E") if rb else None
        if Ea is None or Eb is None or Ea <= 0 or Eb <= 0:
            return None
        return math.log2(Ea / Eb)

    for mode in ref:
        ref[mode]["p_128_256"] = _rate(mode, "n128", "n256")
        ref[mode]["p_256_512"] = _rate(mode, "n256", "n512")
    ref["note"] = ("consistent with a bandwidth-regularization "
                   "initialization layer; E is a TOTAL radius error "
                   "(sigma fixed, never varied) -- no rate or "
                   "joint-limit claim; B_red is the redistribution "
                   "differential vs the matching none run")
    out["warp_bias_refinement"] = ref
    for mode in ("production_coupled", "fixed_delta_sigma"):
        print(f"  {mode}: p(128->256)={ref[mode].get('p_128_256')} "
              f"p(256->512)={ref[mode].get('p_256_512')}")

    path.write_text(json.dumps(out, indent=1))
    print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
