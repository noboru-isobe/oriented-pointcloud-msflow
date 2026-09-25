"""0F-0: one-step attribution cells at checkpoint 1575.

Three cells from the SAME state (the last saved state before the
IIIc-0Dx guard stop), identical config except the constraint rows:

    control   grid_bulk_rows_mode="off"       (C_grid only)
    augment   "augment"                        ([C_grid; C_bulk])
    bulk_only "bulk_only_diagnostic"           (C_bulk + projected
                                                Poisson; NOT production)

Recorded per cell: bulk-row residuals g_disk/g_annulus of the realized
step, mass-weighted sheet radial rates Rdot_{disk,hole,outer} vs the
radial-ODE prediction, W_lin, P, projection, inactive-RHS fraction,
n_params, r_grid/r_bulk/r_stack, delta_compat, incidence margin.

Acceptance (pre-registered): |g_b| <= 1e-10 * sum(m~ |s|) and
|Rdot_disk| <= 2 E_disk (E_disk from the 0F-cal run). Branch-B
attribution table: see phase3c_plan / reviewer item 4.

Usage:
    uv run python scripts/experiments/incidence_rows_onestep.py \
        [--step 1575] [--e-disk <from 0F-cal>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import (  # noqa: E402
    DT_STEP,
    OUT,
    build_cloud,
    masses_for,
)

from annulus_run_a import ode_rhs  # noqa: E402
from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)


def make_config(mode, delta, tau, target0):
    from p1_production_comparison import make_cfg

    from src.torch.transport.grid_wasserstein import GridMetricConfig
    from src.torch.transport.phase_grid import PhaseGridConfig

    cfg = make_cfg("C3", delta, tau, 2)
    cfg.time_step = DT_STEP
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(512, 512), fill_epsilon=0.04,
        support_threshold=3e-3, projection_rel_tol=5e-2),
        compatibility_components="conservative_sweep")
    cfg.redistribute = True
    cfg.remove_dead_points = False
    cfg.enforce_support_gates = False
    cfg.grid_volume_target_mode = "initial_fixed"
    cfg.grid_phase_support_projection = False
    cfg.grid_bulk_rows_mode = mode
    return cfg


def sheet_rates(v, s, m):
    """Mass-weighted radial rates of the three sheets from the
    one-step normal displacement s (radial rate = s (n . rhat) / dt)."""
    r = v.positions.norm(dim=1)
    rhat = v.positions / r[:, None]
    proj = (v.normals * rhat).sum(1)
    rate = s * proj / DT_STEP
    wall = r < 0.6
    outward = (v.positions * v.normals).sum(1) > 0

    def wmean(mask):
        w = m[mask]
        return float((w * rate[mask]).sum() / w.sum())

    return dict(rdot_disk=wmean(wall & outward),
                rdot_hole=wmean(wall & ~outward),
                rdot_outer=wmean(~wall))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1575)
    ap.add_argument("--states-tag", default="_0dx_oriented")
    ap.add_argument("--e-disk", type=float, default=None)
    args = ap.parse_args()

    from src.torch.solver.mm_step import MMStepper as _MS

    ck = torch.load(OUT / f"exact_merger{args.states_tag}_states.pt",
                    weights_only=True)
    st0 = ck[args.step]
    v = OrientedPointCloudVarifold(positions=st0["positions"],
                                   angles=st0["angles"])
    v0, _ = build_cloud()
    delta, tau = compute_recommended_params(v0.positions)
    delta, tau = float(delta), float(tau)
    m0 = masses_for("oriented_kde", v0.positions, v0.normals,
                    delta, tau)
    target0 = float(
        0.5 * (m0 * (v0.positions * v0.normals).sum(-1)).sum())

    m = masses_for("oriented_kde", v.positions, v.normals, delta, tau)
    r = v.positions.norm(dim=1)
    wall = r < 0.6
    outward = (v.positions * v.normals).sum(1) > 0
    R_o = float((m[~wall] * r[~wall]).sum() / m[~wall].sum())
    hole_m = wall & ~outward
    R_h = float((m[hole_m] * r[hole_m]).sum() / m[hole_m].sum())
    rhs_o, rhs_h = ode_rhs(R_o, R_h)
    print(f"state {args.step}: R_o={R_o:.4f} R_h={R_h:.4f} "
          f"ODE Rdot_outer={rhs_o:.4e} Rdot_hole={rhs_h:.4e} "
          f"Rdot_disk=0 (exact)", flush=True)

    report = dict(step=args.step, e_disk=args.e_disk,
                  ode=dict(rdot_outer=rhs_o, rdot_hole=rhs_h,
                           rdot_disk=0.0),
                  cells={})
    for mode in ("off", "augment", "bulk_only_diagnostic"):
        cfg = make_config(mode, delta, tau, target0)
        cfg.mass_estimator = "oriented_kde"
        stepper = _MS(cfg)
        stepper._grid_target_volume_initial = target0
        cell = dict(mode=mode)
        try:
            res = stepper.step(v)
            sn = res.grid_setup_snapshot
            gs = res.grid_step_stats
            s = res.displacements
            cell.update(sheet_rates(v, s, m))
            flux_scale = float((m * s.abs()).sum())
            cell.update(
                converged=bool(res.converged),
                n_iter=int(res.n_iter),
                wasserstein=float(res.wasserstein),
                perimeter=float(res.perimeter),
                objective=float(res.objective),
                max_disp=float(s.abs().max()),
                projection=sn.projection_l2_relative if sn else None,
                inactive_rhs_fraction=(gs.max_inactive_rhs_fraction
                                       if gs else None),
                n_params=sn.n_params if sn else None,
                r_grid=sn.r_grid if sn else None,
                r_bulk=sn.r_bulk if sn else None,
                r_stack=sn.r_stack if sn else None,
                delta_compat=sn.delta_compat if sn else None,
                incidence_margin=sn.incidence_margin if sn else None,
                partition_relation=(sn.partition_relation_grid_bulk
                                    if sn else None),
                augmentation_fired=(sn.augmentation_fired
                                    if sn else None),
                bulk_flux_residuals=res.bulk_flux_residuals,
                flux_scale=flux_scale,
                g_over_scale=(max(abs(x) for x
                                  in res.bulk_flux_residuals)
                              / max(flux_scale, 1e-30)
                              if res.bulk_flux_residuals else None),
            )
        except Exception as exc:      # noqa: BLE001 record outcome
            cell["error"] = f"{type(exc).__name__}: {exc}"
        report["cells"][mode] = cell
        print(f"cell {mode}: " + json.dumps(cell, default=str),
              flush=True)

    out = Path("results/reports/phase3c0f_onestep.json")
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
