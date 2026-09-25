"""0H-1/2: full-q vs lagged-q gradient audit (reviewer-mandated).

The production objective uses the LAGGED surrogate
P_lag(Y) = sum m(Y) q(X^n); 0H-0 therefore measured only the
mass/arclength response. The full state functional
P_sigma(Y) = sum m(Y) q(Y) adds the coherence response
sum m Dq(Y)[xi], which may cancel it. Cells at each synthetic gap
state X_d (d/sigma in {0.45, 0.30, 0.20, 0.10}), all along the SAME
admissible-projected k=2 direction:

    A  g_lag,or   = D[sum m_or(Y)  q(X_d)]        (0H-0, re-measured)
    B  g_lag,arc  = D[sum m_arc(Y) q(X_d)]        (0H-0, re-measured)
    C  g_full,arc = D[sum m_arc(Y) q(Y; m_arc(Y))]
    C' g_full,or  = D[sum m_or(Y)  q(Y; m_or(Y))]  (extra telemetry)
    D  g_q        = C - B                          (coherence channel)

Branch (pre-registered): |C| << |B| -> the q-LAGGING is the culprit
(remedy = restore Dq: recompute q at the candidate / first-order
Taylor / one Picard update -- NOT an energy gate). |C| ~ |B| -> the
full sigma-energy's own layer interaction -> proceed to sigma-N
refinement. Also records the projection-mixing ratio mu_mix per cell
(cells with large mixing are excluded from convergence claims).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import (  # noqa: E402
    OUT,
    build_cloud,
    masses_for,
)
from incidence_rows_onestep import make_config  # noqa: E402
from shape_force_gap_sweep import arc_masses, synthetic_state  # noqa: E402
from shape_mode_audit import disk_sheet, fourier  # noqa: E402

from src.torch.math_utils.angles import wrap_angles  # noqa: E402
from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.solver.mm_step import MMStepper  # noqa: E402
from src.torch.transport.bem_wasserstein import (  # noqa: E402
    compute_coherence,
)
from src.torch.transport.incidence import (  # noqa: E402
    oriented_graph_components,
)

DT = torch.float64
SIGMA = 0.1
GAP_FRACS = (0.45, 0.30, 0.20, 0.10)


def main():
    ck = torch.load(OUT / "exact_merger_0f_oriented_states.pt",
                    weights_only=True)
    st = ck[1400]
    base = OrientedPointCloudVarifold(positions=st["positions"],
                                      angles=st["angles"])
    v0, _ = build_cloud()
    delta, tau = map(float, compute_recommended_params(v0.positions))
    m0 = masses_for("oriented_kde", v0.positions, v0.normals,
                    delta, tau)
    target0 = float(
        0.5 * (m0 * (v0.positions * v0.normals).sum(-1)).sum())
    m1400 = masses_for("oriented_kde", base.positions, base.normals,
                       delta, tau)
    _, w, rad, th = disk_sheet(base, m1400)
    _, four = fourier(w, rad, th)
    a2, b2, amp2 = four[2]

    report = {"cells": []}
    for frac in GAP_FRACS:
        d = frac * SIGMA
        vd = synthetic_state(base, d)
        m = masses_for("oriented_kde", vd.positions, vd.normals,
                       delta, tau)
        maskd, wd, radd, thd = disk_sheet(vd, m)
        idxd = torch.nonzero(maskd).flatten()
        xi = torch.zeros(vd.positions.shape[0], dtype=DT)
        xi[idxd] = (a2 * torch.cos(2 * thd)
                    + b2 * torch.sin(2 * thd)) / amp2

        cfg = make_config("augment", delta, tau, target0)
        cfg.mass_estimator = "oriented_kde"
        stp = MMStepper(cfg)
        stp._grid_target_volume_initial = target0
        stp._setup_step(vd)
        p = stp.param
        xi_a = p.Q @ (p.Q.T @ xi)
        dth = p.AB_solve @ xi_a
        mu_mix = float(torch.sqrt(
            (m[~maskd] * xi_a[~maskd] ** 2).sum()
            / (m * xi_a ** 2).sum()))
        loops = oriented_graph_components(
            vd.positions, vd.normals, 4.0 * float(m.median()))
        q_frozen = stp.fixed_coherence
        sigma_c = stp._sigma
        kern = cfg.perimeter_kernel

        def displaced(s, dthv):
            pos = p.prev_positions + s[:, None] * p.prev_normals
            ang = wrap_angles(p.prev_angles + dthv)
            return OrientedPointCloudVarifold(positions=pos,
                                              angles=ang)

        def cellval(kind, s, dthv):
            va = displaced(s, dthv)
            if kind.startswith("or"):
                mm = stp._masses_for(va.positions, va.normals)
            else:
                mm = arc_masses(va, loops)
            if kind.endswith("lag"):
                q = q_frozen
            else:
                q = compute_coherence(va, mm, sigma_c, kern)
            return float((mm * q).sum())

        t = 1e-4
        cell = dict(d_over_sigma=frac, mu_mix=mu_mix)
        for name, kind in (("g_lag_or", "or_lag"),
                           ("g_lag_arc", "arc_lag"),
                           ("g_full_arc", "arc_full"),
                           ("g_full_or", "or_full")):
            gp = cellval(kind, t * xi_a, t * dth)
            gm_ = cellval(kind, -t * xi_a, -t * dth)
            cell[name] = (gp - gm_) / (2 * t)
        cell["g_q_arc"] = cell["g_full_arc"] - cell["g_lag_arc"]
        cell["g_q_or"] = cell["g_full_or"] - cell["g_lag_or"]
        report["cells"].append(cell)
        print(json.dumps(cell, default=float), flush=True)

    out = Path("results/reports/phase3c0h_fullq.json")
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
