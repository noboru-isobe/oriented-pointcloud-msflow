"""0M-2: direct measurement of the radial -> m4 coupling in the
production grid metric.

The observed phenomenon is "pure radial motion produces an m=4
shape mode", so the causal object is the CROSS term of the metric
bilinear form, not the m4 diagonal energy:

    B_h(u, v) = < B_h u, L_{rho_h}^dag B_h v >

measured by polarization of the production quadratic form
q(u) = GridWassersteinMetric.quadratic(u, 0, dt):

    B(u, v) = (q(u+v) - q(u-v)) / 4.

For each loop: u0 = 1_Gamma (radial), u4c = cos(4 theta) 1_Gamma,
u4s = sin(4 theta) 1_Gamma (theta about the mass centroid).
Reported per (state, loop):

    c4     = (B(u0, u4c), B(u0, u4s))       [isotropic metric: = 0]
    eta04  = |c4| / sqrt(B00 * (B4c4c + B4s4s)/2)
    H4     = 2x2 m4 block
    phi(c4), phi(H4^{-1} c4)  in the phi4 convention of loop_m4
                              (deformation (a,b) -> atan2(b,a)/4)

plus a 22.5-deg rotated copy of each state (grid-locked coupling
keeps its WORLD-frame phase; particle-locked coupling rotates).

States: t=0 clean, endgame3 1300 / 1500 / 1525.

Usage:
    uv run python scripts/experiments/grid_m4_coupling.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import DT_STEP, OUT, build_cloud  # noqa: E402
import tilt_row_ablation as tra  # noqa: E402

from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.loopwise_mass import (  # noqa: E402
    resolve_loopwise_oriented_mass_source,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)

DT = torch.float64


def phase_deg(a, b):
    """(a, b) coefficients of a cos4t + b sin4t -> phi4 in deg
    (same convention as exact_merger_benchmark.loop_m4)."""
    return math.degrees((math.atan2(b, a) / 4.0) % (math.pi / 2.0))


def loop_fields(v, m, labels):
    """Per-loop (u0, u4c, u4s) scalar normal-displacement fields on
    the full cloud (zero off-loop), mass-centroid angular frame."""
    out = {}
    r_all = v.positions.norm(dim=1)
    outward = (v.positions * v.normals).sum(1) > 0
    for lb in labels.unique():
        sel = labels == lb
        if int(sel.sum()) < 8:
            continue
        x = v.positions[sel]
        mm = m[sel]
        c = (mm.unsqueeze(1) * x).sum(0) / mm.sum()
        th = torch.atan2(x[:, 1] - c[1], x[:, 0] - c[0])
        n = v.n_points
        u0 = torch.zeros(n, dtype=DT)
        u4c = torch.zeros(n, dtype=DT)
        u4s = torch.zeros(n, dtype=DT)
        u0[sel] = 1.0
        u4c[sel] = torch.cos(4 * th)
        u4s[sel] = torch.sin(4 * th)
        rmean = float(r_all[sel].mean())
        role = ("disk" if rmean < 0.6
                and float(outward[sel].float().mean()) > 0.5
                else "hole" if rmean < 0.6 else "outer")
        out[role] = (u0, u4c, u4s)
    return out


def rotate_state(v, deg):
    al = math.radians(deg)
    rot = torch.tensor([[math.cos(al), -math.sin(al)],
                        [math.sin(al), math.cos(al)]], dtype=DT)
    return OrientedPointCloudVarifold(
        positions=v.positions @ rot.T, angles=v.angles + al)


def couple_state(name, v, delta, tau, target0):
    st = tra.production_stepper("current", "current", "augment",
                                target0)
    st._setup_step(v)
    gm = st.grid_wasserstein
    # a bare u0 = 1_Gamma changes the component volume, so the
    # fail-closed forward correctly rejects it; the admissible-space
    # object is the PROJECTED form K = <P B u, Ldag P B u> (the
    # pre-existing 0F diagnostic path). P is symmetric, so the
    # polarized cross term is the anisotropic coupling of the
    # volume-neutral part of radial motion -- exactly what acts in
    # the MM step.
    gm.forward_solve = "projected"
    zeros = torch.zeros(v.n_points, dtype=DT)

    def q(u):
        with torch.no_grad():
            return float(gm.quadratic(u, zeros, DT_STEP))

    def bil(u, w):
        return 0.25 * (q(u + w) - q(u - w))

    res = resolve_loopwise_oriented_mass_source(
        v.positions, v.normals, delta, tau)
    fields = loop_fields(v, res.m_loop, res.loop_labels_pre)
    rec = {}
    for role, (u0, u4c, u4s) in fields.items():
        b00 = q(u0)
        b4c4c = q(u4c)
        b4s4s = q(u4s)
        b04c = bil(u0, u4c)
        b04s = bil(u0, u4s)
        b4c4s = bil(u4c, u4s)
        cnorm = math.hypot(b04c, b04s)
        eta04 = cnorm / math.sqrt(
            max(b00 * 0.5 * (b4c4c + b4s4s), 1e-300))
        H = torch.tensor([[b4c4c, b4c4s], [b4c4s, b4s4s]], dtype=DT)
        cv = torch.tensor([b04c, b04s], dtype=DT)
        resp = torch.linalg.solve(H, cv)
        rec[role] = dict(
            B00=b00, B4c4c=b4c4c, B4s4s=b4s4s, B4c4s=b4c4s,
            c4=(b04c, b04s), eta04=eta04,
            phi_c4_deg=phase_deg(b04c, b04s),
            phi_resp_deg=phase_deg(float(resp[0]), float(resp[1])),
        )
        print(f"  [{name}/{role}] eta04 {eta04:.3e}  "
              f"phi(c4) {rec[role]['phi_c4_deg']:5.1f}  "
              f"phi(H^-1 c4) {rec[role]['phi_resp_deg']:5.1f}  "
              f"H4 asym {(b4c4c - b4s4s) / max(b4c4c, 1e-300):+.2e} "
              f"offdiag {b4c4s / max(b4c4c, 1e-300):+.2e}",
              flush=True)
    return rec


def main():
    torch.set_default_dtype(DT)
    v0, _ = build_cloud()
    delta, tau = map(float, compute_recommended_params(v0.positions))
    tra.DELTA, tra.TAU = delta, tau
    m0 = resolve_loopwise_oriented_mass_source(
        v0.positions, v0.normals, delta, tau).m_loop
    target0 = float(0.5 * (m0 * (v0.positions
                                 * v0.normals).sum(-1)).sum())
    ck = torch.load(OUT / "exact_merger_endgame3_states.pt",
                    weights_only=True)
    report = {"meta": dict(delta=delta, tau=tau, target0=target0,
                           dt=DT_STEP), "states": {}}
    cases = [("t0", v0)]
    for s in (1300, 1500, 1525):
        st = ck[s]
        cases.append((str(s), OrientedPointCloudVarifold(
            positions=st["positions"], angles=st["angles"])))
    for name, v in cases:
        print(f"== {name} ==", flush=True)
        report["states"][name] = couple_state(name, v, delta, tau,
                                              target0)
        print(f"== {name} rotated 22.5 deg ==", flush=True)
        report["states"][name + "_rot22.5"] = couple_state(
            name + "_rot", rotate_state(v, 22.5), delta, tau,
            target0)
    out_p = Path("results/reports/phase3c0m2_coupling.json")
    out_p.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
