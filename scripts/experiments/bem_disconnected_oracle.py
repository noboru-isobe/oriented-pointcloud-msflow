"""0C-1: does the discrete tangent space admit inter-component transport?

In the one-phase flow the tangent norm

    ||v||^2 = inf { int_E |u|^2 : div u = 0 in E, u.nu = v on dE }

is finite only when the flux through *each* connected component of the phase
vanishes, since div u = 0 forces int_{dE_a} v = 0 componentwise. The solver
imposes a single global constraint sum_i s_i w_i = 0, which is strictly
weaker. This file measures the gap using known component labels.

Known labels are an *oracle*: they measure how far the current global space is
from the correct one-phase tangent space. They are not a repair of the
production solver -- that has to construct the compatibility subspace without
labels, which is 0C-2 onward. They are also only valid before first contact;
carrying I_1, I_2 through a merger would treat one phase component as two.

Everything here is algebraic. No optimizer runs, so none of the convergence
caveats attached to the one-step trajectory comparison apply.

Tests
-----
A  exchange mode      s in Ran Q_global but not Ran Q_comp
B  first variation    DP[s] = -1/R1 + 1/R2 < 0 under Q_global, ~0 under Q_comp
C  oracle rejection   the componentwise solver must refuse an exchange mode
D  compatible mode    blockwise energy -> pi (R1^2 + R2^2) / 2, gap-independent

Usage
-----
    uv run python scripts/experiments/bem_disconnected_oracle.py
    uv run python scripts/experiments/bem_disconnected_oracle.py --out results/oracle_0c1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.shapes.generator import generate_oriented_circle
from src.torch.transport.bem_wasserstein import BEMWasserstein

DTYPE = torch.float64
DEV = "cpu"
DT = 1e-5
EPS_SCALE = 0.1
K_ENDPOINTS = 3


class IncompatibleVelocity(ValueError):
    """Raised when a velocity has nonzero flux through some phase component.

    The correct tangent norm is +inf there. Silently subtracting componentwise
    means would return a finite cost for a *different* velocity than the one
    supplied, which is exactly the failure this file exists to prevent.
    """


# --------------------------------------------------------------------------
# geometry: two disjoint disks with known labels
# --------------------------------------------------------------------------

def two_disks(R1: float, R2: float, gap: float, spacing: float):
    """Disks of radii R1, R2 separated by `gap`, with exact arc-length weights.

    Returns (varifold, masses, labels, (n1, n2)) where labels[i] in {0, 1}.
    """
    n1 = max(16, int(round(2 * math.pi * R1 / spacing)))
    n2 = max(16, int(round(2 * math.pi * R2 / spacing)))
    cx1 = -(R1 + gap / 2)
    cx2 = +(R2 + gap / 2)
    v1 = generate_oriented_circle(n1, R1, (cx1, 0.0), device=DEV, dtype=DTYPE)
    v2 = generate_oriented_circle(n2, R2, (cx2, 0.0), device=DEV, dtype=DTYPE)
    varifold = OrientedPointCloudVarifold(
        positions=torch.cat([v1.positions, v2.positions]),
        angles=torch.cat([v1.angles, v2.angles]),
    )
    masses = torch.cat([
        torch.full((n1,), 2 * math.pi * R1 / n1, dtype=DTYPE),
        torch.full((n2,), 2 * math.pi * R2 / n2, dtype=DTYPE),
    ])
    labels = torch.cat([torch.zeros(n1, dtype=torch.long),
                        torch.ones(n2, dtype=torch.long)])
    return varifold, masses, labels, (n1, n2)


def flux_matrix(masses: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """G with G[a, i] = m_i if i is in component a, else 0.

    Row a is the discrete int_{dE_a} v dH^1. With q == 1 this is exactly the
    continuum compatibility functional; the q < 1 case is a separate modelling
    question and is deliberately kept out of this sanity check.
    """
    C = int(labels.max().item()) + 1
    G = torch.zeros(C, masses.numel(), dtype=masses.dtype)
    for a in range(C):
        G[a] = masses * (labels == a)
    return G


def nullspace_basis(G: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of ker G, i.e. the componentwise-admissible space."""
    _, s, Vh = torch.linalg.svd(G, full_matrices=True)
    rank = int((s > s.max() * 1e-12).sum())
    return Vh[rank:].T.contiguous()


def projection_residual(Q: torch.Tensor, s: torch.Tensor) -> float:
    """||s - Q Q^T s|| / ||s||: zero iff s lies in the span of Q."""
    return ((s - Q @ (Q.T @ s)).norm() / s.norm()).item()


# --------------------------------------------------------------------------
# A: the exchange mode lives in the global space but not the componentwise one
# --------------------------------------------------------------------------

def exchange_mode(masses, labels, R1, R2) -> torch.Tensor:
    """Shrink the small disk, grow the large one, at zero *global* flux.

    s_i = -1/L_1 on the small disk, +1/L_2 on the large one, with
    L_a = 2 pi R_a, so the component fluxes are exactly (-1, +1) and their sum
    is zero. This is the Ostwald-ripening direction: forbidden by the
    one-phase metric, admitted by a single global constraint.
    """
    L1, L2 = 2 * math.pi * R1, 2 * math.pi * R2
    s = torch.where(labels == 0,
                    torch.full_like(masses, -1.0 / L1),
                    torch.full_like(masses, +1.0 / L2))
    return s


def test_exchange_mode(R1=0.5, R2=1.0, gap=4.0, spacing=0.03) -> dict:
    _, masses, labels, _ = two_disks(R1, R2, gap, spacing)
    G = flux_matrix(masses, labels)
    g_global = G.sum(0, keepdim=True)                 # the solver's one row

    s = exchange_mode(masses, labels, R1, R2)
    Q_global = nullspace_basis(g_global)
    Q_comp = nullspace_basis(G)

    return dict(
        R1=R1, R2=R2, gap=gap, n_points=masses.numel(),
        dim_global=Q_global.shape[1], dim_comp=Q_comp.shape[1],
        global_flux=(g_global @ s).item(),
        component_fluxes=[x.item() for x in (G @ s)],
        residual_in_global=projection_residual(Q_global, s),
        residual_in_comp=projection_residual(Q_comp, s),
    )


# --------------------------------------------------------------------------
# B: the perimeter first variation along that mode
# --------------------------------------------------------------------------

def test_first_variation(R1=0.5, R2=1.0, gap=4.0, spacing=0.03) -> dict:
    """On a circle of radius R the curvature is 1/R, so the exact first
    variation of perimeter is DP[s] = sum_a (1/R_a) int_{dE_a} s.

    For the exchange mode that is -1/R1 + 1/R2 < 0: it strictly decreases
    perimeter at fixed total area. The gradient g_i = m_i / R_a lies in the
    row space of G, so its componentwise projection vanishes while its global
    projection does not -- the descent direction survives exactly one
    constraint and is killed by the correct two.
    """
    _, masses, labels, _ = two_disks(R1, R2, gap, spacing)
    G = flux_matrix(masses, labels)
    g_global = G.sum(0, keepdim=True)
    s = exchange_mode(masses, labels, R1, R2)

    curvature = torch.where(labels == 0,
                            torch.full_like(masses, 1.0 / R1),
                            torch.full_like(masses, 1.0 / R2))
    grad = curvature * masses                          # dP/ds_i

    Q_global = nullspace_basis(g_global)
    Q_comp = nullspace_basis(G)
    scale = grad.norm().item()

    return dict(
        R1=R1, R2=R2, gap=gap,
        dP_along_exchange=(grad * s).sum().item(),
        dP_exact=-1.0 / R1 + 1.0 / R2,
        grad_norm=scale,
        proj_grad_global=(Q_global.T @ grad).norm().item() / scale,
        proj_grad_comp=(Q_comp.T @ grad).norm().item() / scale,
    )


# --------------------------------------------------------------------------
# C / D: energies from a blockwise interior-BEM oracle
# --------------------------------------------------------------------------

def _component_energy(varifold, masses, vel, index) -> float:
    """Interior-trace BEM energy of one component, solved on its own."""
    sub = OrientedPointCloudVarifold(
        positions=varifold.positions[index], angles=varifold.angles[index],
    )
    m, v = masses[index], vel[index]
    bem = BEMWasserstein(method="point", epsilon_scale=EPS_SCALE,
                         n_endpoints=K_ENDPOINTS, trace_side="interior")
    bem.setup_for_step(sub.positions, sub.normals, m, sigma=None)
    bem.setup_coherence(torch.ones(m.numel(), dtype=DTYPE))
    return (2.0 * bem(DT * v, torch.zeros_like(v), DT) / DT).item()


def blockwise_oracle_energy(varifold, masses, labels, vel,
                            tol: float = 1e-10) -> float:
    """Sum of per-component interior Neumann energies.

    Rejects velocities with nonzero componentwise flux instead of projecting
    them: see IncompatibleVelocity.
    """
    G = flux_matrix(masses, labels)
    fluxes = G @ vel
    scale = (masses * vel.abs()).sum().clamp_min(1e-300)
    if (fluxes.abs() / scale).max() > tol:
        raise IncompatibleVelocity(
            "componentwise Neumann compatibility violated: fluxes "
            f"{[round(f.item(), 6) for f in fluxes]}"
        )
    total = 0.0
    for a in range(int(labels.max().item()) + 1):
        total += _component_energy(varifold, masses, vel, labels == a)
    return total


def test_oracle_rejects_exchange(R1=0.5, R2=1.0, gap=4.0, spacing=0.03) -> dict:
    varifold, masses, labels, _ = two_disks(R1, R2, gap, spacing)
    s = exchange_mode(masses, labels, R1, R2)
    try:
        blockwise_oracle_energy(varifold, masses, labels, s)
        return dict(rejected=False, message="oracle accepted an exchange mode")
    except IncompatibleVelocity as exc:
        return dict(rejected=True, message=str(exc))


def test_compatible_mode(R1=0.5, R2=1.0, spacing=0.03) -> list[dict]:
    """v_a(theta) = cos(2 theta) on each disk: mean zero on each component, so
    admissible. The exact energy of cos(k theta) on a disk of radius R is
    pi R^2 / k, hence pi (R1^2 + R2^2) / 2 for k = 2, and it must not depend on
    how far apart the disks are -- in the continuum they do not interact.

    Sweeps the gap at fixed resolution (independence) and the resolution at
    fixed gap (convergence), so a constant offset cannot be mistaken for
    either.
    """
    exact = math.pi * (R1 ** 2 + R2 ** 2) / 2
    rows = []
    for gap, h in [(g, spacing) for g in (1.0, 2.0, 4.0, 8.0)] + \
                  [(4.0, h) for h in (0.06, 0.03, 0.015, 0.0075)]:
        varifold, masses, labels, (n1, n2) = two_disks(R1, R2, gap, h)
        t1 = torch.linspace(0, 2 * math.pi, n1 + 1, dtype=DTYPE)[:-1]
        t2 = torch.linspace(0, 2 * math.pi, n2 + 1, dtype=DTYPE)[:-1]
        vel = torch.cat([torch.cos(2 * t1), torch.cos(2 * t2)])
        energy = blockwise_oracle_energy(varifold, masses, labels, vel)
        rows.append(dict(gap=gap, spacing=h, n_points=n1 + n2,
                         energy=energy, energy_exact=exact,
                         ratio=energy / exact,
                         component_fluxes=[x.item()
                                           for x in (flux_matrix(masses, labels) @ vel)]))
    return rows


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--spacing", type=float, default=0.03)
    args = ap.parse_args()

    results = {}

    print("=" * 78)
    print("A  exchange mode: zero global flux, component fluxes (-1, +1)")
    results["A_exchange_mode"] = [
        test_exchange_mode(gap=g, spacing=args.spacing) for g in (1.0, 4.0)
    ]
    for r in results["A_exchange_mode"]:
        print(f"  gap={r['gap']}  N={r['n_points']}  "
              f"dim(global)={r['dim_global']}  dim(comp)={r['dim_comp']}")
        print(f"    global flux            {r['global_flux']:+.3e}   (must be 0)")
        print(f"    component fluxes       "
              f"[{r['component_fluxes'][0]:+.6f}, {r['component_fluxes'][1]:+.6f}]"
              f"   (must be -1, +1)")
        print(f"    residual in Ran Q_glob {r['residual_in_global']:.3e}"
              f"   (0 => admitted by the solver)")
        print(f"    residual in Ran Q_comp {r['residual_in_comp']:.3e}"
              f"   (O(1) => forbidden by the one-phase metric)")

    print("\n" + "=" * 78)
    print("B  perimeter first variation along the exchange mode")
    results["B_first_variation"] = [
        test_first_variation(gap=g, spacing=args.spacing) for g in (1.0, 4.0)
    ]
    for r in results["B_first_variation"]:
        print(f"  gap={r['gap']}  DP[s]={r['dP_along_exchange']:+.6f}  "
              f"exact={r['dP_exact']:+.6f}  (negative => decreases perimeter)")
        print(f"    ||Q_global^T grad||/||grad|| = {r['proj_grad_global']:.4f}"
              f"   (>0 => descent direction survives)")
        print(f"    ||Q_comp^T   grad||/||grad|| = {r['proj_grad_comp']:.3e}"
              f"   (~0 => killed by componentwise constraints)")

    print("\n" + "=" * 78)
    print("C  the oracle must refuse an exchange mode, not project it")
    results["C_oracle_rejection"] = test_oracle_rejects_exchange(
        spacing=args.spacing)
    r = results["C_oracle_rejection"]
    print(f"  rejected={r['rejected']}")
    print(f"  {r['message']}")

    print("\n" + "=" * 78)
    print("D  admissible mode cos(2 theta): additive and gap-independent")
    results["D_compatible_mode"] = test_compatible_mode(spacing=args.spacing)
    print(f"{'gap':>6} {'spacing':>9} {'N':>6} {'energy':>12} {'exact':>12} "
          f"{'ratio':>10}")
    for r in results["D_compatible_mode"]:
        print(f"{r['gap']:>6.1f} {r['spacing']:>9.4f} {r['n_points']:>6} "
              f"{r['energy']:>12.6f} {r['energy_exact']:>12.6f} {r['ratio']:>10.6f}")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "bem_disconnected_oracle.json"
        path.write_text(json.dumps(
            {"meta": dict(spacing=args.spacing, epsilon_scale=EPS_SCALE,
                          n_endpoints=K_ENDPOINTS, trace_side="interior"),
             **results}, indent=2))
        print(f"\nraw results -> {path}")


if __name__ == "__main__":
    main()
