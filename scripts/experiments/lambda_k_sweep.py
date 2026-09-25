"""0J-C: 4-arm complex-mode growth-rate sweep.

Arms: full (current frozen-q, control) / loop_mean (circular-limit
mechanism check) / self_renormalized (main candidate) /
isolated-self (the perturbed disk sheet ALONE -- the intrinsic
stability baseline the correction is supposed to restore).

Complex mode rates (reviewer spec): with z_k = a_k + i b_k,

    lambda_k = Re[(z_k^{n+1} - z_k^n) / (tau z_k^n)],
    omega_k  = Im[ ... ]                       (phase rotation),

separating growth from rotation. The acceptance threshold is a noise
envelope PRE-FIXED from the zero-seed runs (lambda_noise,k =
|dz_k(zero seed)| / (tau A_seed)); half-seed checks
lambda_k(A/2) ~ lambda_k(A). The essential verdict: under
self_renormalized the contact-induced sign reversal disappears and
lambda_k returns to the isolated-self stable side.

Grid: m10 cell, k in {2..8} x delta in {0.45, 0.30, 0.20, 0.10},
all 3 contact arms + isolated (delta-independent); m075/m05 at
k in {2,4,6} x delta in {0.45, 0.20} (sigma scaling); linearity at
k=2, m10 (zero/half seeds).

Usage:
    uv run python scripts/experiments/lambda_k_sweep.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import masses_for  # noqa: E402
from shape_mode_audit import disk_sheet, fourier  # noqa: E402
from sigma_n_refinement import (  # noqa: E402
    CELLS,
    SEED_RATIO,
    build_cloud_n,
    cell_config,
    organic_reference,
    synthetic_cell_state,
)

from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.solver.mm_step import MMStepper  # noqa: E402

DT = torch.float64
ARMS = ("full", "loop_mean", "self_renormalized")
KS_MAIN = (2, 3, 4, 5, 6, 7, 8)
KS_SIDE = (2, 4, 6)
DELTAS_MAIN = (0.45, 0.30, 0.20, 0.10)
DELTAS_SIDE = (0.45, 0.20)

# 0K-C: mass estimator under test (settable via --mass-estimator)
MASS_EST = "oriented_kde"


def z_of(v, m, k):
    _, w, rad, th = disk_sheet(v, m)
    _, four = fourier(w, rad, th)
    a, b, _ = four[k]
    return complex(a, b)


def one_rate(cell, ref, arm, delta, k, seed, kde, target0):
    R_d_star, A_ann, phi = ref
    isolated = arm == "isolated"
    v, _ = synthetic_cell_state(R_d_star, A_ann, cell["sigma"], delta,
                                cell["n_out"], seed, phi, k=k,
                                isolated=isolated)
    delta_kde, tau_kde = kde
    cfg = cell_config(cell, delta_kde, tau_kde, target0)
    cfg.mass_estimator = MASS_EST
    if isolated:
        cfg.grid_volume_target_mode = "current_divergence"
        cfg.grid_bulk_rows_mode = "off"
    else:
        cfg.perimeter_q_mode = (arm if arm != "full" else "full")
    stp = MMStepper(cfg)
    if not isolated:
        stp._grid_target_volume_initial = target0
    m = masses_for(MASS_EST, v.positions, v.normals,
                   delta_kde, tau_kde)
    z0 = z_of(v, m, k)
    res = stp.step(v)
    vv = (res.committed_varifold if res.committed_varifold is not None
          else res.varifold)
    m1 = masses_for(MASS_EST, vv.positions, vv.normals,
                    delta_kde, tau_kde)
    z1 = z_of(vv, m1, k)
    tau = cfg.time_step
    out = dict(arm=arm, delta=delta, k=k, seed=seed,
               z0=[z0.real, z0.imag], z1=[z1.real, z1.imag], tau=tau)
    if abs(z0) > 1e-10:
        rate = (z1 - z0) / (tau * z0)
        out["lam"] = rate.real
        out["omega"] = rate.imag
    else:
        # zero seed: report the FORCING |dz|/tau (a rate would divide
        # by round-off); the noise envelope is dz_over_tau / A_seed
        out["dz_over_tau"] = abs(z1 - z0) / tau
    if not isolated and stp._qmode_stats:
        out["min_r"] = stp._qmode_stats.get("min_r")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--extended", action="store_true",
                    help="reviewer coverage: m075/m05 x k{2,3,4} x "
                         "all four deltas, full + self_renormalized")
    ap.add_argument("--mass-estimator", default="oriented_kde",
                    choices=("oriented_kde", "loopwise_oriented_kde"))
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()
    global MASS_EST
    MASS_EST = args.mass_estimator

    ref = organic_reference()
    report = {"rows": []}
    out_path = Path(f"results/reports/phase3c0j_lambda_k"
                    f"{args.out_tag}.json")

    def run(cell, arm, delta, k, seed, kde, target0):
        try:
            row = one_rate(cell, ref, arm, delta, k, seed, kde,
                           target0)
        except Exception as exc:      # noqa: BLE001
            row = dict(arm=arm, delta=delta, k=k, seed=seed,
                       cell=cell["name"],
                       error=f"{type(exc).__name__}: {exc}")
        row["cell"] = cell["name"]
        report["rows"].append(row)
        msg = (f"[{cell['name']}/{arm}] k={k} d/s={delta} "
               + (f"lam={row['lam']:+.3e} om={row.get('omega'):+.2e}"
                  if "lam" in row else
                  f"dz/tau={row.get('dz_over_tau', float('nan')):.2e}"
                  if "dz_over_tau" in row else "ERROR " + row["error"]))
        print(msg, flush=True)
        out_path.write_text(json.dumps(report, indent=1,
                                       default=float))

    cells = {c["name"]: c for c in CELLS}
    plan = []
    m10 = cells["m10"]
    if args.extended:
        out_path = Path(f"results/reports/phase3c0j_lambda_k_ext"
                        f"{args.out_tag}.json")
        for name in ("m075", "m05"):
            for k in (2, 3, 4):
                for d in DELTAS_MAIN:
                    for arm in ("full", "self_renormalized"):
                        plan.append((cells[name], arm, d, k,
                                     "primary"))
        kde_cache, tgt_cache = {}, {}
        for cell, arm, d, k, seedname in plan:
            n_out = cell["n_out"]
            if n_out not in kde_cache:
                v0 = build_cloud_n(n_out)
                kde_cache[n_out] = tuple(map(
                    float, compute_recommended_params(v0.positions)))
                m0 = masses_for(MASS_EST, v0.positions,
                                v0.normals, *kde_cache[n_out])
                tgt_cache[n_out] = float(
                    0.5 * (m0 * (v0.positions
                                 * v0.normals).sum(-1)).sum())
            a_full = SEED_RATIO * cell["sigma"]
            run(cell, arm, d, k, a_full, kde_cache[n_out],
                tgt_cache[n_out])
        print(f"wrote {out_path}")
        return
    ks = KS_MAIN[:1] if args.smoke else KS_MAIN
    deltas = DELTAS_MAIN[:1] if args.smoke else DELTAS_MAIN
    for k in ks:
        for arm in ARMS:
            for d in deltas:
                plan.append((m10, arm, d, k, "primary"))
        plan.append((m10, "isolated", 0.45, k, "primary"))
        plan.append((m10, "isolated", 0.45, k, "zero"))
    if not args.smoke:
        # noise envelope: zero seed per (k, delta) on the main cell
        for k in KS_MAIN:
            for d in (0.45, 0.10):
                plan.append((m10, "self_renormalized", d, k, "zero"))
        # linearity at k=2
        plan.append((m10, "self_renormalized", 0.45, 2, "half"))
        plan.append((m10, "full", 0.45, 2, "half"))
        # sigma scaling cells
        for name in ("m075", "m05"):
            for k in KS_SIDE:
                for d in DELTAS_SIDE:
                    for arm in ("full", "self_renormalized"):
                        plan.append((cells[name], arm, d, k,
                                     "primary"))

    kde_cache, tgt_cache = {}, {}
    for cell, arm, d, k, seedname in plan:
        n_out = cell["n_out"]
        if n_out not in kde_cache:
            v0 = build_cloud_n(n_out)
            kde_cache[n_out] = tuple(map(float,
                                         compute_recommended_params(
                                             v0.positions)))
            m0 = masses_for(MASS_EST, v0.positions, v0.normals,
                            *kde_cache[n_out])
            tgt_cache[n_out] = float(
                0.5 * (m0 * (v0.positions
                             * v0.normals).sum(-1)).sum())
        a_full = SEED_RATIO * cell["sigma"]
        seed = {"primary": a_full, "half": 0.5 * a_full,
                "zero": 0.0}[seedname]
        run(cell, arm, d, k, seed, kde_cache[n_out], tgt_cache[n_out])

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
