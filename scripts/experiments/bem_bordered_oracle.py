"""0C-3 clean prototype: compatibility-constrained bordered BIE, q = 1.

The label-free candidate solver for disconnected phases, restricted to the
regime where 0C-2b's modelling ambiguity is absent: q = 1 makes
m = q m = q^2 m, so there is exactly one compatibility scale. Everything is
static (Delta alpha = 0), rank C is supplied by the oracle (rank detection
and hysteresis are 0C-4), and NOTHING here is wired into MMStepper --
production defaults stay untouched until 0C-2c/0C-4 settle where q belongs
in the metric.

System, in weighted coordinates (A~ = W^{1/2} A W^{-1/2} = U S V^T,
U0/V0 the C smallest singular directions):

    [ A~   U0 ] [ lambda~ ]   [ v~ ]
    [ V0^T  0 ] [ eta     ] = [ 0  ]

The U0 column completes the range (left/compatibility side), the V0^T row
fixes the density gauge (right side) -- the two are NOT interchangeable
(0C-2a: on an annulus they sit 35 degrees apart).

An incompatible right-hand side is REJECTED before the solve via
r_comp = ||U0^T v~|| / ||v~||: the bordered variable eta must not be allowed
to absorb an exchange mode and return a finite energy for a velocity the
one-phase metric forbids.

Note on eta (review correction): eta is NOT an independent validation
metric. Multiplying the first block row by U0^T and using the gauge
V0^T lambda~ = 0 gives U0^T A~ lambda~ = Sigma_0 V0^T lambda~ = 0, hence

    eta = U0^T v~        (up to solver roundoff),

so eta_rel coincides with r_comp by construction -- the recorded tables show
them numerically identical. Report eta as the compatibility-discretisation
residual of the data; the independent evidence for the prototype is the
blockwise-oracle agreement (energy, gauge-fixed potentials), separation
independence, the rejection behaviour, and refinement convergence.

Oracle: the blockwise solve (each phase component on its own, C = 1 each),
compared in energy, componentwise gauge-fixed potential, eta decay,
separation independence, and N-refinement. Geometries deliberately include
noncircular disconnected cases; circles alone have hidden a sign error
before.

Usage
-----
    uv run python scripts/experiments/bem_bordered_oracle.py \
        --out results/bordered_0c3
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.transport.bem_wasserstein import BEMWasserstein

import sys

sys.path.insert(0, str(Path(__file__).parent))
from bem_disconnected_oracle import IncompatibleVelocity  # noqa: E402
from bem_spectral_audit import GEOMETRIES  # noqa: E402
from bem_visible_weight_audit import build_geometry_full  # noqa: E402

DTYPE = torch.float64
EPS_SCALE = 0.1
K_ENDPOINTS = 3


# --------------------------------------------------------------------------
# the prototype solver
# --------------------------------------------------------------------------

def solve_bordered_neumann_static(op, C: int, v_end: torch.Tensor,
                                  reject_tol: float = 0.1) -> dict:
    """Solve the interior Neumann problem on a possibly disconnected phase.

    Args:
        op: EndpointOperator from BEMWasserstein.endpoint_operator()
        C: number of phase components (oracle-supplied in this prototype)
        v_end: endpoint-level Neumann data (NK,)
        reject_tol: relative compatibility residual above which the data is
            refused as incompatible

    Returns dict with phi (endpoint potential), lam (density), eta, r_comp,
    energy = sum w phi v, and the ascending singular values.
    """
    w = op.weights
    sqrt_w = w.sqrt()
    At = op.weighted_trace_operator()
    U, S_desc, Vh = torch.linalg.svd(At)
    U0 = U[:, -C:]
    V0 = Vh[-C:, :].T

    vt = sqrt_w * v_end
    r_comp = ((U0.T @ vt).norm() / vt.norm()).item()
    if r_comp > reject_tol:
        raise IncompatibleVelocity(
            f"componentwise Neumann compatibility violated: "
            f"||U0^T v~||/||v~|| = {r_comp:.3f} > {reject_tol}")

    NK = len(w)
    Msys = torch.zeros(NK + C, NK + C, dtype=DTYPE)
    Msys[:NK, :NK] = At
    Msys[:NK, NK:] = U0
    Msys[NK:, :NK] = V0.T
    rhs = torch.cat([vt, torch.zeros(C, dtype=DTYPE)])
    sol = torch.linalg.solve(Msys, rhs)
    lam = sol[:NK] / sqrt_w
    eta = sol[NK:]

    phi = op.single_layer @ lam
    energy = (w * phi * v_end).sum().item()
    return dict(phi=phi, lam=lam, eta=eta, r_comp=r_comp, energy=energy,
                eta_rel=(eta.norm() / vt.norm()).item(),
                sigma=S_desc.flip(0))


# --------------------------------------------------------------------------
# geometry, velocities, oracle
# --------------------------------------------------------------------------

def _bem(varifold, lengths):
    bem = BEMWasserstein(method="point", epsilon_scale=EPS_SCALE,
                         n_endpoints=K_ENDPOINTS, trace_side="interior")
    bem.setup_for_step(varifold.positions, varifold.normals, lengths,
                       sigma=None)
    return bem


def compatible_velocity(varifold, m, phases, C) -> torch.Tensor:
    """cos(2 theta) about each phase's centroid, with the phase-weighted mean
    removed so that the componentwise flux vanishes EXACTLY. Works for
    noncircular components and multi-ring phases alike."""
    v = torch.zeros(len(m), dtype=DTYPE)
    for a in range(C):
        sel = phases == a
        p = varifold.positions[sel]
        centroid = (p * m[sel, None]).sum(0) / m[sel].sum()
        th = torch.atan2(p[:, 1] - centroid[1], p[:, 0] - centroid[0])
        v[sel] = torch.cos(2 * th)
        v[sel] -= (v[sel] * m[sel]).sum() / m[sel].sum()
    return v


def exchange_velocity(m, phases) -> torch.Tensor:
    """Zero global flux, componentwise fluxes (-1, +1): the mode the solver
    must refuse."""
    L0 = m[phases == 0].sum()
    L1 = m[phases == 1].sum()
    return torch.where(phases == 0,
                       torch.full_like(m, -1.0 / L0),
                       torch.full_like(m, +1.0 / L1))


def global_solve(varifold, m, phases, C, v_particle) -> dict:
    bem = _bem(varifold, m)
    op = bem.endpoint_operator()
    v_end = bem.expand_to_endpoints(v_particle)
    out = solve_bordered_neumann_static(op, C, v_end)
    out.update(op=op, v_end=v_end, bem=bem)
    return out


def blockwise_solve(varifold, m, phases, C, v_particle) -> dict:
    """Each phase on its own boundary (all of its rings), C = 1 per block."""
    energies, phis, etas = [], {}, []
    for a in range(C):
        sel = phases == a
        sub = OrientedPointCloudVarifold(
            positions=varifold.positions[sel], angles=varifold.angles[sel])
        bem = _bem(sub, m[sel])
        op = bem.endpoint_operator()
        v_end = bem.expand_to_endpoints(v_particle[sel])
        out = solve_bordered_neumann_static(op, 1, v_end)
        energies.append(out["energy"])
        etas.append(out["eta_rel"])
        phis[a] = dict(phi=out["phi"], w=op.weights, v=v_end)
    return dict(energy=sum(energies), energies=energies, phis=phis,
                eta_rel_max=max(etas))


def gauge_fixed_potential_error(g: dict, b: dict, phases, K: int) -> float:
    """Max over phases of the relative weighted-L2 difference between the
    global and blockwise potentials, each with its own per-phase mean
    removed (potentials on a disconnected phase are defined up to one
    additive constant PER COMPONENT, so raw phi must not be compared)."""
    lab_end = phases.repeat_interleave(K)
    worst = 0.0
    for a, blk in b["phis"].items():
        sel = lab_end == a
        w = blk["w"]
        pg = g["phi"][sel]
        pb = blk["phi"]
        pg = pg - (w * pg).sum() / w.sum()
        pb = pb - (w * pb).sum() / w.sum()
        err = ((w * (pg - pb) ** 2).sum().sqrt()
               / (w * pb ** 2).sum().sqrt().clamp_min(1e-300)).item()
        worst = max(worst, err)
    return worst


# --------------------------------------------------------------------------
# case matrix
# --------------------------------------------------------------------------

def register_local_geometries():
    GEOMETRIES.setdefault("two_unequal_disks_g1", dict(
        tier="primary", components=[
            ("circle", dict(R=0.5), (-1.0, 0.0), False, 0),
            ("circle", dict(R=1.0), (1.5, 0.0), False, 1)]))
    GEOMETRIES.setdefault("two_unequal_disks_g4", dict(
        tier="primary", components=[
            ("circle", dict(R=0.5), (-2.5, 0.0), False, 0),
            ("circle", dict(R=1.0), (3.0, 0.0), False, 1)]))
    GEOMETRIES.setdefault("two_unequal_ellipses_far", dict(
        tier="primary", components=[
            ("ellipse", dict(a=1.0, b=0.5), (-3.1, 0.0), False, 0),
            ("ellipse", dict(a=0.55, b=0.35), (2.9, 0.25), False, 1)]))


CASES = [
    "two_unequal_disks_g1",
    "two_unequal_disks_g4",
    "two_unequal_ellipses",
    "two_unequal_ellipses_far",
    "disk_ellipse_flower",
    "annulus_plus_ellipse",
]

# exact anchor for the disks: Q(cos 2 theta) on a disk of radius R is
# pi R^2 / 2, additive over components
EXACT_ENERGY = {
    "two_unequal_disks_g1": math.pi * (0.5 ** 2 + 1.0 ** 2) / 2,
    "two_unequal_disks_g4": math.pi * (0.5 ** 2 + 1.0 ** 2) / 2,
}


def run_case(name: str, spacing: float) -> dict:
    varifold, m, phases, bcomp, M, C = build_geometry_full(name, spacing)
    v = compatible_velocity(varifold, m, phases, C)

    g = global_solve(varifold, m, phases, C, v)
    b = blockwise_solve(varifold, m, phases, C, v)

    row = dict(
        geometry=name, spacing=spacing, C=C,
        n_endpoints=len(g["op"].weights),
        r_comp=g["r_comp"],
        eta_rel=g["eta_rel"],
        eta_rel_blockwise_max=b["eta_rel_max"],
        energy_global=g["energy"],
        energy_blockwise=b["energy"],
        energy_ratio=g["energy"] / b["energy"],
        phi_rel_err_max=gauge_fixed_potential_error(
            g, b, phases, K_ENDPOINTS),
        sigma_C=g["sigma"][C - 1].item(),
        sigma_Cp1=g["sigma"][C].item(),
    )
    if name in EXACT_ENERGY:
        row["energy_exact"] = EXACT_ENERGY[name]
        row["energy_vs_exact"] = g["energy"] / EXACT_ENERGY[name]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.06, 0.03, 0.015])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    register_local_geometries()
    rows = []
    print(f"{'geometry':>26} {'h':>7} {'NK':>6} {'r_comp':>9} {'eta_rel':>9} "
          f"{'E_glob/E_block':>14} {'phi_err':>9} {'E/exact':>9}")
    for name in CASES:
        for h in args.levels:
            r = run_case(name, h)
            rows.append(r)
            anchor = (f"{r['energy_vs_exact']:>9.5f}"
                      if "energy_vs_exact" in r else f"{'--':>9}")
            print(f"{r['geometry']:>26} {h:>7.4f} {r['n_endpoints']:>6} "
                  f"{r['r_comp']:>9.2e} {r['eta_rel']:>9.2e} "
                  f"{r['energy_ratio']:>14.6f} {r['phi_rel_err_max']:>9.2e} "
                  f"{anchor}", flush=True)
        print()

    # rejection check: the exchange mode must raise, not return an energy
    varifold, m, phases, bcomp, M, C = build_geometry_full(
        "two_unequal_disks_g1", args.levels[0])
    try:
        global_solve(varifold, m, phases, C, exchange_velocity(m, phases))
        rejection = dict(rejected=False)
        print("REJECTION CHECK FAILED: exchange mode was accepted")
    except IncompatibleVelocity as exc:
        rejection = dict(rejected=True, message=str(exc))
        print(f"rejection check: exchange mode refused ({exc})")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "bem_bordered_oracle.json"
        path.write_text(json.dumps(dict(
            meta=dict(levels=args.levels, epsilon_scale=EPS_SCALE,
                      n_endpoints=K_ENDPOINTS, trace_side="interior",
                      q="identically 1", C="oracle-supplied"),
            rejection=rejection, rows=rows), indent=2))
        print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
