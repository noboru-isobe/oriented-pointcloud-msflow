"""Compression-0/1 (reviewer 2026-08-18 GO): static bookkeeping audit
and three-arm comparison at the t_comp-entry states (768 @ 2294,
1024 @ 2292).

Compression-0 (audit): closure of the signed-area bookkeeping
    A_out + A_disk + A_hole  vs  V_merged (driver vol_radial)
at the stop state, both grids. The ghost pair's signed area
A_pair = A_disk + A_hole < 0 must cancel the outer loop's excess over
the merged equilibrium.

Compression-1 (three static arms, NO dynamics):
    K  keep the ghost pair (reference)
    D  delete the pair only              -> NEGATIVE CONTROL
    C  delete the pair + conservative outer projection: scale the
       outer loop about the origin by c = sqrt(V_pre / A_out) (exact
       area projection; concentric oracle R_out -> sqrt(R_out^2 +
       R_d^2 - R_h^2) ~ R_eq)
Measured per arm: V (signed area), dV_merged, R_outer (mean radius),
centroid, P# (production sharp perimeter), phase-grid reconstruction
(components + phase volume from a production _setup_step), and the
visible-current distance at the perimeter mollifier scale
    D_T_raw = ||psi_s*(T_arm) - psi_s*(T_K)||_L2^2 / ||psi_s*T_K||^2 (RAW currents)
(masses re-resolved per arm by the production 0K estimator).

Usage:
    uv run python scripts/experiments/ghost_compression_static.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import OUT, build_cloud, build_config  # noqa
from quotient_switch_stability import production_args  # noqa
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa
from src.torch.oriented_varifold.mass import (  # noqa
    compute_recommended_params,
)
from src.torch.solver.mm_step import MMStepper  # noqa
from src.torch.transport.contact_certificate import (  # noqa
    mollified_current_gram,
)

DT = torch.float64
SIGMA = 0.1
R_EQ = 0.9055385138149781


def signed_area(pos):
    """Shoelace area of one loop, ordered by polar angle about the
    loop centroid (exact for the concentric-circle loops here)."""
    c = pos.mean(0)
    d = pos - c
    order = torch.atan2(d[:, 1], d[:, 0]).argsort()
    p = pos[order]
    q = torch.roll(p, -1, dims=0)
    return float(0.5 * (p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]).sum())


def loop_masks(pos, quotient):
    """Ghost pair from the stored contact_reference; outer = rest."""
    ref = quotient["contact_reference"]
    ma = torch.as_tensor(ref["particles_a_mask"]).bool()
    mb = torch.as_tensor(ref["particles_b_mask"]).bool()
    return ma, mb, ~(ma | mb)


def resolve_m(pos, nor, delta, tau):
    from src.torch.oriented_varifold.loopwise_mass import (
        resolve_loopwise_oriented_mass_source,
    )
    return resolve_loopwise_oriented_mass_source(
        pos, nor, delta, tau).m_loop


def raw_current_distance(vK, mK, vA, mA, sigma):
    """D_T_raw = ||psi*(T_A) - psi*(T_K)||^2 / ||psi*T_K||^2 for the
    RAW currents T = sum m n delta_x (terminology correction, reviewer
    2026-08-19: no coherence factor q is applied here; the
    q^WB-weighted visible distance lives in the Comp-4 transaction
    record as D_T_vis_qfull)."""
    pos = torch.cat([vK.positions, vA.positions])
    nor = torch.cat([vK.normals, vA.normals])
    m = torch.cat([mK, mA])
    nK, nA = vK.n_points, vA.n_points
    iK = torch.arange(nK)
    iA = torch.arange(nK, nK + nA)
    eK = mollified_current_gram(pos, nor, m, sigma, iK, iK)
    eA = mollified_current_gram(pos, nor, m, sigma, iA, iA)
    eKA = mollified_current_gram(pos, nor, m, sigma, iK, iA)
    return (eK + eA - 2.0 * eKA) / eK, eK


def probe_arm(tag, v, cfg, target0, delta, tau, quotient=None):
    st = MMStepper(cfg)
    st._grid_target_volume_initial = target0
    if quotient is not None:
        st.import_quotient_state(quotient, v.n_points)
    st._setup_step(v)
    sn = st._last_grid_snapshot
    P = st.sharp_perimeter(v)
    m = resolve_m(v.positions, v.normals, delta, tau)
    return dict(tag=tag, n_points=v.n_points, P_sharp=P,
                grid_components=int(sn.n_components)
                if hasattr(sn, "n_components") else None,
                phase_volume=(float(sn.phase_grid_volume)
                              if getattr(sn, "phase_grid_volume", None)
                              is not None else None),
                r_stack=getattr(sn, "r_stack", None),
                quotient_classes=getattr(sn, "quotient_classes", None),
                masses=m)


def run_grid(tag_states, step, grid, delta, tau, target0, vol_ref):
    ck = torch.load(OUT / f"exact_merger{tag_states}_states.pt",
                    weights_only=True)
    stt = ck[step]
    pos, ang, quot = stt["positions"], stt["angles"], stt["quotient"]
    v = OrientedPointCloudVarifold(positions=pos, angles=ang)
    ma, mb, mo = loop_masks(pos, quot)
    # orient: a/b = the ghost pair (disk / hole in some order)
    A_a, A_b = signed_area(pos[ma]), signed_area(pos[mb])
    # signed contribution needs the ORIENTATION: use the production
    # radial volume element 0.5 m x.n per loop instead of shoelace sign
    nor = v.normals
    m = resolve_m(pos, nor, delta, tau)
    xn = 0.5 * m * (pos * nor).sum(-1)
    V_a, V_b, V_o = (float(xn[ma].sum()), float(xn[mb].sum()),
                     float(xn[mo].sum()))
    V_pair = V_a + V_b
    V_tot = V_a + V_b + V_o
    R_out = float(pos[mo].norm(dim=1).mean())
    rep = dict(
        grid=grid, step=step,
        A_shoelace=dict(a=A_a, b=A_b, outer=signed_area(pos[mo])),
        V_radial=dict(a=V_a, b=V_b, outer=V_o, pair=V_pair, total=V_tot),
        vol_radial_run=vol_ref,
        closure_total_vs_run=V_tot - vol_ref,
        pair_share_of_V=V_pair / V_tot,
        R_outer=R_out,
        R_out_oracle=math.sqrt(max(R_out ** 2 + V_pair / math.pi, 0.0)),
    )
    # ---- arms ---------------------------------------------------
    # Comp-4 remeasurement (reviewer 2026-08-19): arm C uses the
    # production map (signed-shoelace V#, polygon-centroid homothety,
    # analytic root) -- the current-bookkeeping scale of the first
    # audit is superseded
    from src.torch.solver.ghost_compression import (
        conservative_compression,
        merged_geom_volume,
    )
    cfg = build_config(production_args(grid=grid, redist_monotone=True),
                       delta, tau)
    arms = {}
    vK = v
    arms["K"] = probe_arm("K", vK, cfg, target0, delta, tau,
                          quotient=quot)
    vD = OrientedPointCloudVarifold(positions=pos[mo].clone(),
                                    angles=ang[mo].clone())
    arms["D"] = probe_arm("D", vD, cfg, target0, delta, tau)
    comp = conservative_compression(pos, ang, nor, mo, ma, mb)
    vC = OrientedPointCloudVarifold(positions=comp["positions"],
                                    angles=comp["angles"])
    c = comp["scale"]
    arms["C"] = probe_arm("C", vC, cfg, target0, delta, tau)
    rep["C_scale"] = c
    rep["C_R_outer"] = float(vC.positions.norm(dim=1).mean())
    rep["V_sharp_geom"] = comp["V_sharp_geom"]
    rep["V_pair_geom"] = comp["V_pair_geom"]
    rep["C_V_geom_residual"] = comp["V_comp_geom"] - comp["V_sharp_geom"]
    rep["C_dV_geom_rel_D"] = (merged_geom_volume(
        vD.positions, vD.normals, [torch.ones(vD.n_points,
                                              dtype=torch.bool)])
        - comp["V_sharp_geom"]) / comp["V_sharp_geom"]
    mK = arms["K"].pop("masses")
    for a, va in (("K", vK), ("D", vD), ("C", vC)):
        mA = arms[a].pop("masses") if "masses" in arms[a] else mK
        xnA = 0.5 * mA * (va.positions * va.normals).sum(-1)
        arms[a]["V_merged"] = float(xnA.sum())
        arms[a]["dV_merged_rel"] = (arms[a]["V_merged"] - V_tot) / V_tot
        arms[a]["R_outer"] = float(
            va.positions[mo].norm(dim=1).mean()
            if a == "K" else va.positions.norm(dim=1).mean())
        arms[a]["centroid"] = [float(x) for x in va.positions.mean(0)]
        if a != "K":
            dT, eK = raw_current_distance(vK, mK, va, mA, SIGMA)
            arms[a]["D_T_raw"] = dT
        arms[a]["dP_sharp"] = arms[a]["P_sharp"] - arms["K"]["P_sharp"]
    rep["arms"] = arms
    return rep


def main():
    torch.set_default_dtype(DT)
    v0, _ = build_cloud()
    delta, tau = map(float, compute_recommended_params(v0.positions))
    from src.torch.oriented_varifold.loopwise_mass import (
        resolve_loopwise_oriented_mass_source,
    )
    m0 = resolve_loopwise_oriented_mass_source(
        v0.positions, v0.normals, delta, tau).m_loop
    target0 = float(0.5 * (m0 * (v0.positions * v0.normals)
                           .sum(-1)).sum())
    out = {}
    for tag, step, grid in (("_0n4c_768", 2294, 768),
                            ("_0n4c_1024", 2292, 1024)):
        if not (OUT / f"exact_merger{tag}_states.pt").exists():
            print(f"-- {tag} states missing, skipped")
            continue
        js = json.load(open(OUT / f"exact_merger{tag}.json"))
        vol_ref = [r for r in js["series"]
                   if r["step"] == step][0]["vol_radial"]
        rep = run_grid(tag, step, grid, delta, tau, target0, vol_ref)
        out[tag] = rep
        print(f"== {tag} step {step} (grid {grid})")
        print(f"  Comp-0: V_pair {rep['V_radial']['pair']:+.6e} "
              f"outer {rep['V_radial']['outer']:.6f} total "
              f"{rep['V_radial']['total']:.6f} (run vol_radial "
              f"{vol_ref:.6f}, closure {rep['closure_total_vs_run']:+.2e})"
              f" pair share {rep['pair_share_of_V']:+.4%}")
        print(f"  oracle R_out^comp {rep['R_out_oracle']:.6f} "
              f"(R_eq {R_EQ:.6f}); C scale {rep['C_scale']:.8f} -> "
              f"R_outer {rep['C_R_outer']:.6f}")
        for a, d in rep["arms"].items():
            print(f"  arm {a}: N {d['n_points']} V {d['V_merged']:.6f} "
                  f"dV_rel {d['dV_merged_rel']:+.3e} P# "
                  f"{d['P_sharp']:.6f} (dP {d['dP_sharp']:+.3e}) "
                  f"R_out {d['R_outer']:.6f} centroid "
                  f"[{d['centroid'][0]:+.2e},{d['centroid'][1]:+.2e}] "
                  f"D_T_raw {d.get('D_T_raw')}")
    Path("results/reports/ghost_compression_static.json").write_text(
        json.dumps(out, indent=1, default=str))
    print("wrote results/reports/ghost_compression_static.json")


if __name__ == "__main__":
    main()
