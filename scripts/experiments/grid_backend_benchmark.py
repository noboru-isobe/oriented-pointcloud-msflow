"""G3a backend benchmark: one full MM step (ellipse N=512, 512^2 grid)
under four solver configurations, against the BEM C3 reference.

Measures what the reviewer's NO-GO was about: wall time per step, with
the full per-step Poisson telemetry (solve counts, PCG iterations,
solve wall) and the parity gate re-checked per configuration.

Configurations:
  cpu_pcg_generic  -- the G2b baseline (measured ~297 s/step)
  cpu_pcg_analytic -- analytic quadratic (fewer solves, same PCG)
  cpu_direct       -- sparse LU: factor once per step, ms per solve
  gpu_pcg_analytic -- T4, sync-light PCG (check interval 8)

Output: results/grid_metric/backend_benchmark.json
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
    generate_oriented_ellipse,
)
from src.torch.solver.mm_step import MMStepper  # noqa: E402
from src.torch.transport.grid_wasserstein import (  # noqa: E402
    GridMetricConfig,
)
from src.torch.transport.phase_grid import PhaseGridConfig  # noqa: E402
from p1_production_comparison import make_cfg  # noqa: E402

DT = torch.float64
N_GRID, EPS = 512, 0.05


def grid_cfg(delta, tau, path, backend, check_interval):
    cfg = make_cfg("C3", delta, tau, 1)
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    gm = GridMetricConfig(
        phase=PhaseGridConfig(grid_shape=(N_GRID, N_GRID),
                              fill_epsilon=EPS),
        quadratic_path=path)
    gm.poisson = replace(gm.poisson, solver_backend=backend,
                         pcg_check_interval=check_interval)
    cfg.grid_metric = gm
    return cfg


def main():
    v_cpu = generate_oriented_ellipse(512, a=1.2, b=0.8, device="cpu",
                                      dtype=DT)
    delta, tau = compute_recommended_params(v_cpu.positions)

    t0 = time.time()
    r_bem = MMStepper(make_cfg("C3", delta, tau, 1)).step(v_cpu)
    wall_bem = time.time() - t0
    s_bem = r_bem.displacements.cpu()
    out = {"bem_reference": {"wall_s": wall_bem,
                             "n_iter": r_bem.n_iter}}
    print(f"bem reference: {wall_bem:.1f}s ({r_bem.n_iter} iters)")

    configs = [
        ("cpu_pcg_generic", "cpu", "generic", "native_pcg", 1),
        ("cpu_pcg_analytic", "cpu", "analytic", "native_pcg", 1),
        ("cpu_direct", "cpu", "analytic", "sparse_direct", 1),
    ]
    if torch.cuda.is_available():
        configs.append(
            ("gpu_pcg_analytic", "cuda", "analytic", "native_pcg", 8))

    for name, device, path, backend, ck in configs:
        v = (v_cpu if device == "cpu" else
             generate_oriented_ellipse(512, a=1.2, b=0.8,
                                       device=device, dtype=DT))
        st = MMStepper(grid_cfg(delta, tau, path, backend, ck))
        t0 = time.time()
        r = st.step(v)
        if device == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
        rel_s = float((r.displacements.cpu() - s_bem).norm()
                      / s_bem.norm())
        gs = r.grid_step_stats
        out[name] = dict(
            wall_s=wall, rel_s_vs_bem=rel_s, n_iter=r.n_iter,
            relgrad=r.relative_gradient_norm,
            objective_decreased=bool(r.objective_decreased),
            n_forward_solves=gs.n_forward_solves,
            n_projected_solves=gs.n_projected_solves,
            pcg_iterations_total=gs.pcg_iterations_total,
            pcg_iterations_median=gs.pcg_iterations_median,
            pcg_iterations_max=gs.pcg_iterations_max,
            max_relative_residual=gs.max_relative_residual,
            solve_wall_seconds=gs.solve_wall_seconds,
        )
        print(f"{name:17s}: {wall:6.1f}s  rel_s={rel_s:.4f}  "
              f"solves {gs.n_forward_solves}+{gs.n_projected_solves}  "
              f"pcg_iters {gs.pcg_iterations_total}  "
              f"solve_wall {gs.solve_wall_seconds:.1f}s")

    dest = ROOT / "results" / "grid_metric" / "backend_benchmark.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
