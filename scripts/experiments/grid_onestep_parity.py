"""G2b one-step optimizer parity: grid backend vs production BEM C3.

Pre-registered gate (plan go/no-go): on a smooth resolved single
component the full trust-ncg one-step velocity difference between the
two discretizations of the same tangent metric is < 2%. The separated
pair is REPORTED, not gated: near the contact layer the admissible
spaces (BIE spectral vs grid phase components) and the mollified
metric's attenuation both grow, and the plan excludes contact layers
from smooth-case gates.

Fixture laws (calibrated in G0/G1): h/eps <= 0.5 (porous wall),
dx <= eps/2, 2*eps < gap.

Output: results/grid_metric/onestep_parity.json
"""

import json
import sys
import time
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
    generate_oriented_two_ellipses,
)
from src.torch.solver.mm_step import MMStepper  # noqa: E402
from src.torch.transport.grid_wasserstein import (  # noqa: E402
    GridMetricConfig,
)
from src.torch.transport.phase_grid import PhaseGridConfig  # noqa: E402
from p1_production_comparison import make_cfg  # noqa: E402

DT = torch.float64


def grid_cfg(delta, tau, n, eps):
    cfg = make_cfg("C3", delta, tau, 1)
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(n, n), fill_epsilon=eps))
    return cfg


def one_step_pair(v, rank, n, eps):
    delta, tau = compute_recommended_params(v.positions)

    t0 = time.time()
    st_b = MMStepper(make_cfg("C3", delta, tau, rank))
    r_b = st_b.step(v)
    wall_b = time.time() - t0

    t0 = time.time()
    st_g = MMStepper(grid_cfg(delta, tau, n, eps))
    r_g = st_g.step(v)
    wall_g = time.time() - t0

    s_b, s_g = r_b.displacements, r_g.displacements
    gm = st_g.grid_wasserstein
    rows = gm.flux.component_rows(gm.phase.component_labels,
                                  gm.phase.cell_area)
    vec = torch.cat([s_g, r_g.delta_angles])
    sn = r_g.grid_setup_snapshot
    return dict(
        rel_s=float((s_g - s_b).norm() / s_b.norm()),
        rel_th=float((r_g.delta_angles - r_b.delta_angles).norm()
                     / r_b.delta_angles.norm()),
        s_norm_bem=float(s_b.norm()), s_norm_grid=float(s_g.norm()),
        n_iter_bem=r_b.n_iter, n_iter_grid=r_g.n_iter,
        relgrad_bem=r_b.relative_gradient_norm,
        relgrad_grid=r_g.relative_gradient_norm,
        objective_decreased=bool(r_b.objective_decreased
                                 and r_g.objective_decreased),
        realized_flux_max=float((rows @ vec).abs().max()),
        realized_flux_scale=float((rows.abs() @ vec.abs()).max()),
        n_components=sn.n_components, n_params=sn.n_params,
        target_volume=sn.target_volume_used,
        volume_error=sn.reconstruction_volume_error,
        projection_l2_relative=sn.projection_l2_relative,
        grid=n, eps=eps, wall_bem_s=wall_b, wall_grid_s=wall_g,
    )


def main():
    out = {}

    # A: smooth resolved ellipse, production scales (GATED < 2%)
    v = generate_oriented_ellipse(512, a=1.2, b=0.8, device="cpu",
                                  dtype=DT)
    out["ellipse_512"] = one_step_pair(v, 1, 512, 0.05)
    r = out["ellipse_512"]
    gate = r["rel_s"] < 0.02 and r["objective_decreased"]
    out["ellipse_512"]["gate_2pct"] = bool(gate)
    print(f"ellipse: rel_s={r['rel_s']:.4f} rel_th={r['rel_th']:.4f} "
          f"gate={'PASS' if gate else 'FAIL'} "
          f"(bem {r['wall_bem_s']:.0f}s / grid {r['wall_grid_s']:.0f}s)")

    # B: separated pair, gap 0.10, 2*eps=0.08 < gap (REPORTED)
    v = generate_oriented_two_ellipses(
        256, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
        a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu", dtype=DT)
    out["pair_gap010"] = one_step_pair(v, 2, 512, 0.04)
    r = out["pair_gap010"]
    print(f"pair   : rel_s={r['rel_s']:.4f} rel_th={r['rel_th']:.4f} "
          f"C={r['n_components']} (reported, not gated; "
          f"bem {r['wall_bem_s']:.0f}s / grid {r['wall_grid_s']:.0f}s)")

    dest = ROOT / "results" / "grid_metric" / "onestep_parity.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
