"""Offline BIE diagnostics at the stored states of the production runs:
sigma_tail (second smallest / largest singular value of the weighted
operator per block), lambda_min / lambda_max of the metric Hessian on
the admissible subspace, number of masked particles, merged pairs.
Recomputed by _setup_step with the production configuration of each
driver (bitwise the setup of the run at those states).

Usage: uv run python scripts/experiments/bie_state_diagnostics.py
Writes results/reports/bie_state_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import two_ellipses_benchmark as teb  # noqa: E402
from exact_merger_benchmark import build_config  # noqa: E402
from quotient_switch_stability import production_args  # noqa: E402
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.solver.mm_step import MMStepper  # noqa: E402

torch.set_default_dtype(torch.float64)


def annulus_cfg(delta, tau):
    args = argparse.Namespace(
        grid=1024, fill_epsilon=0.04, metric="bie", bridge_gap=1.0,
        no_redistribute=False, advect=False, freeze_dead=False,
        mass_estimator="loopwise_oriented_kde", bulk_rows=True,
        bulk_functional="current", volume_target_functional="current",
        contact_rows_mode="aligned_prequotient", gauge_hold=False,
        quotient_mode="off", angle_scope="loopwise",
        angle_measure="raw_loopwise", angle_consistency="none",
        redist_scope="loopwise", redist_tangent="angles",
        redist_curvature="loopwise", redist_q_policy="r_loop",
        redist_retraction=False, redist_monotone=True,
        redist_monotone_parts="abc", q_mode="self_renormalized",
        vertical=False, lam=1.0, optimizer_max_iter=None,
        optimizer_tol=None)
    return build_config(args, delta, tau)


def ellipse_cfg(delta, tau, q_mode):
    c = build_config(production_args(
        grid=768, redist_monotone=True, quotient_mode="off",
        q_mode=q_mode, metric="bie", bridge_gap=1.0), delta, tau)
    c.time_step = 1e-5
    c.grid_bulk_first_moment_rows = True
    c.grid_bulk_first_moment_rows_form = "current_centered"
    return c


def flower_cfg(delta, tau):
    return ellipse_cfg(delta, tau, "self_renormalized")


def scan(name, states_path, cfg_fn, json_path=None):
    ck = torch.load(states_path, weights_only=True)
    keys = sorted(k for k in ck if isinstance(k, int))
    s0 = ck[keys[0]]
    delta, tau = map(float, compute_recommended_params(s0["positions"]))
    meta = json.load(open(json_path))["meta"] if json_path else {}
    act = meta.get("activated_at")
    quot = meta.get("quotient_at")
    rows = []
    for k in keys:
        st = ck[k]
        q_mode = ("contact_complex_renormalized"
                  if act is not None and k >= act
                  and st["positions"].shape[0] == s0["positions"].shape[0]
                  else "self_renormalized")
        cfg = cfg_fn(delta, tau) if cfg_fn is not flower_cfg and name == "annulus" \
            else (ellipse_cfg(delta, tau, q_mode) if name == "two_ellipses"
                  else flower_cfg(delta, tau))
        stp = MMStepper(cfg)
        v = OrientedPointCloudVarifold(positions=st["positions"].clone(),
                                       angles=st["angles"].clone())
        try:
            stp._setup_step(v)
        except Exception as ex:  # noqa: BLE001
            rows.append(dict(step=k, error=f"{type(ex).__name__}: {ex}"[:200]))
            continue
        sn = stp._last_grid_snapshot
        rows.append(dict(step=k, N=int(st["positions"].shape[0]),
                         n_components=sn.n_components,
                         sigma_tail_min=min(sn.bie_sigma_tail),
                         null_level_max=max(sn.bie_null_level),
                         lambda_min=sn.bie_lambda_min,
                         lambda_max=sn.bie_lambda_max,
                         lambda_ratio=sn.bie_lambda_min / sn.bie_lambda_max,
                         n_masked=sn.bie_n_masked,
                         merged=sn.bie_merged_pairs,
                         relation=sn.partition_relation_grid_bulk,
                         q_mode=q_mode))
    ok = [r for r in rows if "error" not in r]
    summ = dict(n_states=len(rows), n_errors=len(rows) - len(ok),
                sigma_tail_min=min(r["sigma_tail_min"] for r in ok),
                lambda_ratio_min=min(r["lambda_ratio"] for r in ok),
                lambda_ratio_min_step=min(ok, key=lambda r: r["lambda_ratio"])["step"],
                n_masked_max=max(r["n_masked"] for r in ok))
    print(name, summ, flush=True)
    return dict(summary=summ, rows=rows)


def main():
    out = {}
    out["flower"] = scan("flower", ROOT / "results/flower/flower_production_bie_states.pt", flower_cfg)
    out["annulus"] = scan("annulus", ROOT / "results/exact_merger/exact_merger_bie_states.pt", annulus_cfg,
                          ROOT / "results/exact_merger/exact_merger_bie.json")
    ell = ROOT / "results/two_ellipses/two_ellipses_bie_states.pt"
    ellj = ROOT / "results/two_ellipses/two_ellipses_bie.json"
    if ell.exists():
        out["two_ellipses"] = scan("two_ellipses", ell, ellipse_cfg, ellj)
    (ROOT / "results/reports/bie_state_diagnostics.json").write_text(
        json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
