"""G3b: short-horizon dynamic parity -- BEM C3 and the grid backend
evolve the SAME initial ellipse independently for K steps; trajectories
are compared at every step index (reviewer: one-step 0.36% does not
bound 100-step drift; measure it).

Per-step comparisons (reviewer list):
- max_i |x_i^grid - x_i^bem|, normal-angle RMS/max (wrapped)
- D_J: relative L2 difference of the mollified oriented currents
  eta * (J_grid - J_bem) on the audit grid (the scatter kernel IS
  eta, so the scattered field is the mollified current)
- D_rho: relative L1 difference of the reconstructed phases
- perimeter, divergence-theorem area, objective decrease, step/h and
  step/eps scales, grid component count + fail-closed gates (any
  violation raises and ends the run -- recorded, not swallowed)

Output: results/grid_metric/lockstep_parity.json (milestones at
20/50/100 plus the full per-step series).
"""

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import torch  # noqa: E402

from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_masses,
    compute_recommended_params,
)
from src.torch.shapes.generator import (  # noqa: E402
    generate_oriented_ellipse,
)
from src.torch.solver.mm_step import MMStepper  # noqa: E402
from src.torch.transport.grid_wasserstein import (  # noqa: E402
    GridMetricConfig,
)
from src.torch.transport.phase_grid import (  # noqa: E402
    CurrentToPhase,
    PhaseGridConfig,
    ScatterStencil,
)
from p1_production_comparison import make_cfg  # noqa: E402

DT = torch.float64
K_STEPS = 100
MILESTONES = (20, 50, 100)
AUDIT_CFG = PhaseGridConfig(grid_shape=(512, 512), fill_epsilon=0.05)


def _circ(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _masses(v, delta, tau):
    return compute_masses(v.positions, delta, tau, "wendland_c2")


def _current_field(v, m):
    st = ScatterStencil(v.positions, AUDIT_CFG)
    n = v.normals
    return torch.stack([st.apply(m * n[:, 0]), st.apply(m * n[:, 1])])


def _phase(v, m):
    vol = float(0.5 * (m * (v.positions * v.normals).sum(-1)).sum())
    return CurrentToPhase(AUDIT_CFG).reconstruct(v, m, vol).rho_metric


def compare(vg, vb, delta, tau):
    mg, mb = _masses(vg, delta, tau), _masses(vb, delta, tau)
    Jg, Jb = _current_field(vg, mg), _current_field(vb, mb)
    rg, rb = _phase(vg, mg), _phase(vb, mb)
    dth = _circ(vg.angles, vb.angles)
    return dict(
        max_pos_diff=float((vg.positions - vb.positions)
                           .norm(dim=1).max()),
        angle_rms=float(dth.pow(2).mean().sqrt()),
        angle_max=float(dth.abs().max()),
        D_J=float((Jg - Jb).norm() / Jb.norm()),
        D_rho=float((rg - rb).abs().sum() / rb.abs().sum()),
        area_grid=float(0.5 * (mg * (vg.positions
                                     * vg.normals).sum(-1)).sum()),
        area_bem=float(0.5 * (mb * (vb.positions
                                    * vb.normals).sum(-1)).sum()),
    )


def main():
    v0 = generate_oriented_ellipse(512, a=1.2, b=0.8, device="cpu",
                                   dtype=DT)
    delta, tau = compute_recommended_params(v0.positions)

    cfg_b = make_cfg("C3", delta, tau, 1)
    cfg_g = make_cfg("C3", delta, tau, 1)
    cfg_g.bem_solver_mode = "legacy_global_projection"
    cfg_g.metric_backend = "grid_poisson"
    gm = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(512, 512), fill_epsilon=0.05))
    gm.poisson = replace(gm.poisson, solver_backend="sparse_direct")
    cfg_g.grid_metric = gm

    st_b, st_g = MMStepper(cfg_b), MMStepper(cfg_g)
    vb, vg = v0, v0
    series, out = [], {}
    h = float(2 * math.pi * 1.0 / 512)   # nominal spacing scale
    t0 = time.time()
    for k in range(1, K_STEPS + 1):
        rb = st_b.step(vb)
        rg = st_g.step(vg)
        vb, vg = rb.varifold, rg.varifold
        rec = compare(vg, vb, delta, tau)
        rec.update(
            step=k,
            perimeter_grid=rg.perimeter, perimeter_bem=rb.perimeter,
            objective_decreased=bool(rb.objective_decreased
                                     and rg.objective_decreased),
            step_over_h=float(rg.displacements.abs().max()) / h,
            step_over_eps=float(rg.displacements.abs().max()) / 0.05,
            n_components=rg.grid_setup_snapshot.n_components,
            volume_drift_grid=(rg.grid_setup_snapshot
                               .target_volume_drift_relative),
        )
        series.append(rec)
        if k in MILESTONES:
            print(f"step {k:3d}: max|dx|={rec['max_pos_diff']:.2e} "
                  f"angle_rms={rec['angle_rms']:.2e} "
                  f"D_J={rec['D_J']:.4f} D_rho={rec['D_rho']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        elif k % 10 == 0:
            print(f"step {k:3d}: max|dx|={rec['max_pos_diff']:.2e} "
                  f"D_J={rec['D_J']:.4f}", flush=True)
    out["series"] = series
    out["milestones"] = {str(k): series[k - 1] for k in MILESTONES}
    out["wall_s_total"] = time.time() - t0
    dest = ROOT / "results" / "grid_metric" / "lockstep_parity.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"-> {dest}", flush=True)


if __name__ == "__main__":
    main()
