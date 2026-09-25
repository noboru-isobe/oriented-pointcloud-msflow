"""G3a-2: optimizer x variable benchmark on the grid backend
(cpu sparse_direct, the G3a-1 winner).

Reviewer discipline: velocity scaling and optimizer choice are
separate axes -- the reference is trust-ncg/displacement, and each
config is judged by the PRE-REGISTERED promotion rules (plan G3):

    |F - F_ref| / (1 + |F_ref|)   < 1e-8
    |s - s_ref| / (1 + |s_ref|)   < 1e-5
    residual/compatibility unchanged, fewer solves or wall improvement

Shapes: ellipse (smooth reference), annulus (C=1, two loops), pair
(C=2, gap 0.10). One full MM step each.

Output: results/grid_metric/optimizer_benchmark.json
"""

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import torch  # noqa: E402

from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.shapes.generator import (  # noqa: E402
    generate_oriented_annulus,
    generate_oriented_ellipse,
    generate_oriented_two_ellipses,
)
from src.torch.solver.mm_step import MMStepper  # noqa: E402
from src.torch.transport.grid_wasserstein import (  # noqa: E402
    GridMetricConfig,
)
from src.torch.transport.phase_grid import PhaseGridConfig  # noqa: E402
from p1_production_comparison import make_cfg  # noqa: E402

DT = torch.float64
METHODS = ["trust-ncg", "trust-krylov", "l-bfgs"]
VARIABLES = ["displacement", "velocity"]


def fixtures():
    return {
        "ellipse": (generate_oriented_ellipse(
            512, a=1.2, b=0.8, device="cpu", dtype=DT), 0.05),
        "annulus": (generate_oriented_annulus(
            256, R_outer=1.0, R_inner=0.5, device="cpu", dtype=DT),
            0.06),
        "pair": (generate_oriented_two_ellipses(
            256, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
            a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu",
            dtype=DT), 0.04),
    }


def run_one(v, eps, method, variable):
    delta, tau = compute_recommended_params(v.positions)
    cfg = make_cfg("C3", delta, tau, 1)
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    cfg.optimizer_method = method
    cfg.optimizer_variable = variable
    gm = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(512, 512), fill_epsilon=eps))
    gm.poisson = replace(gm.poisson, solver_backend="sparse_direct")
    cfg.grid_metric = gm
    st = MMStepper(cfg)
    t0 = time.time()
    try:
        r = st.step(v)
    except Exception as e:  # noqa: BLE001 -- record, judge later
        return dict(failed=True, error=f"{type(e).__name__}: {e}",
                    wall_s=time.time() - t0)
    wall = time.time() - t0
    gm_obj = st.grid_wasserstein
    rows = gm_obj.flux.component_rows(gm_obj.phase.component_labels,
                                      gm_obj.phase.cell_area)
    vec = torch.cat([r.displacements, r.delta_angles])
    gs = r.grid_step_stats
    return dict(
        failed=False, wall_s=wall,
        objective=r.objective, n_iter=r.n_iter, n_fev=r.n_fev,
        relgrad=r.relative_gradient_norm,
        objective_decreased=bool(r.objective_decreased),
        s=r.displacements,
        flux_leak=float((rows @ vec).abs().max()
                        / (rows.abs() @ vec.abs()).max()),
        n_solves=gs.n_forward_solves + gs.n_projected_solves,
        solve_wall_s=gs.solve_wall_seconds,
        max_residual=gs.max_relative_residual,
    )


def main():
    out = {}
    for shape, (v, eps) in fixtures().items():
        ref_F, ref_s = None, None
        out[shape] = {}
        for method in METHODS:
            for variable in VARIABLES:
                key = f"{method}/{variable}"
                r = run_one(v, eps, method, variable)
                if not r["failed"]:
                    if ref_s is None:    # trust-ncg/displacement first
                        ref_F, ref_s = r["objective"], r["s"].clone()
                    dF = abs(r["objective"] - ref_F) \
                        / (1.0 + abs(ref_F))
                    ds = float((r["s"] - ref_s).norm()) \
                        / (1.0 + float(ref_s.norm()))
                    r["dF_vs_ref"] = dF
                    r["ds_vs_ref"] = ds
                    r["meets_promotion_rules"] = bool(
                        dF < 1e-8 and ds < 1e-5
                        and r["objective_decreased"]
                        and r["flux_leak"] < 1e-12)
                    del r["s"]
                    print(f"{shape:8s} {key:24s} "
                          f"F={r['objective']:.8f} dF={r.get('dF_vs_ref', 0):.1e} "
                          f"ds={r.get('ds_vs_ref', 0):.1e} "
                          f"it={r['n_iter']:3d} solves={r['n_solves']:3d} "
                          f"wall={r['wall_s']:5.1f}s "
                          f"{'OK' if r['meets_promotion_rules'] else '--'}",
                          flush=True)
                else:
                    print(f"{shape:8s} {key:24s} FAILED: {r['error']}",
                          flush=True)
                out[shape][key] = r
    dest = ROOT / "results" / "grid_metric" / "optimizer_benchmark.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"-> {dest}", flush=True)


if __name__ == "__main__":
    main()
