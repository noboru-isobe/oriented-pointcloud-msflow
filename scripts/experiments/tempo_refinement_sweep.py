"""0F post-hoc (reviewer-mandated): two-axis tempo attribution sweep.

At checkpoint 1575 (deep contact layer, gap 0.39 sigma-units), the
constrained one-step showed a uniform front-tempo factor eta ~ 0.69
vs the sharp-interface radial ODE. Candidates: (1) coherence/energy
bandwidth sigma, (2) phase filling bandwidth eps_fill, (3) diffuse-
bridge Poisson coupling, (4) finite N / angle error. This sweep
separates (1) from (2):

  axis A: sigma in {0.10, 0.075, 0.05}, eps_fill fixed 0.04
  axis B: eps_fill in {0.04, 0.03, 0.02}, sigma fixed 0.10

each cell = one augmented step, measuring eta_h = Rdot_hole/ODE and
eta_o = Rdot_outer/ODE. Scale-law guards: h_pc/sigma <= 0.35 at the
smallest sigma (marginal, reported); dx = 2.56/512 = 0.005 <=
eps_fill/2 at eps_fill = 0.02 (holds: 0.01 boundary -> use 512 grid).

Consistency claim requires eta -> 1 as the scales shrink; a plateau
means the scheme approximates a regularized flow with contact-layer
time dilation (report the branch, do not tune).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from annulus_run_a import ode_rhs  # noqa: E402
from exact_merger_benchmark import OUT, build_cloud, masses_for  # noqa: E402
from incidence_rows_onestep import make_config, sheet_rates  # noqa: E402

from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.solver.mm_step import MMStepper  # noqa: E402


def main():
    ck = torch.load(OUT / "exact_merger_0dx_oriented_states.pt",
                    weights_only=True)
    st0 = ck[1575]
    v = OrientedPointCloudVarifold(positions=st0["positions"],
                                   angles=st0["angles"])
    v0, _ = build_cloud()
    delta, tau = map(float, compute_recommended_params(v0.positions))
    m0 = masses_for("oriented_kde", v0.positions, v0.normals,
                    delta, tau)
    target0 = float(
        0.5 * (m0 * (v0.positions * v0.normals).sum(-1)).sum())
    m = masses_for("oriented_kde", v.positions, v.normals, delta, tau)

    r = v.positions.norm(dim=1)
    wall = r < 0.6
    outward = (v.positions * v.normals).sum(1) > 0
    R_o = float((m[~wall] * r[~wall]).sum() / m[~wall].sum())
    hole = wall & ~outward
    R_h = float((m[hole] * r[hole]).sum() / m[hole].sum())
    rhs_o, rhs_h = ode_rhs(R_o, R_h)
    print(f"ODE at 1575: Rdot_hole={rhs_h:.4f} Rdot_outer={rhs_o:.4f}",
          flush=True)

    cells = ([("sigma", s, 0.04) for s in (0.10, 0.075, 0.05)]
             + [("fill", 0.10, e) for e in (0.03, 0.02)])
    report = dict(ode=dict(rdot_hole=rhs_h, rdot_outer=rhs_o),
                  cells=[])
    for axis, sigma, eps in cells:
        from src.torch.transport.grid_wasserstein import (
            GridMetricConfig,
        )
        from src.torch.transport.phase_grid import PhaseGridConfig
        cfg = make_config("augment", delta, tau, target0)
        cfg.mass_estimator = "oriented_kde"
        cfg.perimeter_sigma = sigma
        cfg.angle_sigma = sigma
        cfg.grid_metric = GridMetricConfig(
            phase=PhaseGridConfig(
                grid_shape=(512, 512), fill_epsilon=eps,
                support_threshold=3e-3, projection_rel_tol=5e-2),
            compatibility_components="conservative_sweep")
        cell = dict(axis=axis, sigma=sigma, fill_epsilon=eps)
        try:
            stp = MMStepper(cfg)
            stp._grid_target_volume_initial = target0
            res = stp.step(v)
            rates = sheet_rates(v, res.displacements, m)
            cell.update(rates)
            cell["eta_h"] = rates["rdot_hole"] / rhs_h
            cell["eta_o"] = rates["rdot_outer"] / rhs_o
            cell["eps_flux_ok"] = (max(
                abs(x) for x in res.bulk_flux_residuals) < 1e-12)
        except Exception as exc:      # noqa: BLE001 record outcome
            cell["error"] = f"{type(exc).__name__}: {exc}"
        report["cells"].append(cell)
        print(json.dumps(cell, default=str), flush=True)

    out = Path("results/reports/phase3c0f_tempo_sweep.json")
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
