"""0M-3/0M-4: causal factorization of the grid-metric m4 coupling.

Cell A: analytic RADIAL rho (smooth two-phase indicator sampled on
        the grid) + analytic ring RHS fields (radial / cos4 / sin4)
        + weighted Poisson only. No particles, no scatter. A nonzero
        eta04 here indicts the 5-point weighted FV stencil or the
        threshold active set.
Cell B: same analytic radial rho + production BoundaryFluxToGrid
        scatter from exact-circle particle clouds. B minus A is the
        scatter (lattice quadrature) contribution.
(Cell C = production reconstruction + production scatter is
 grid_m4_coupling.py; its t0 row is the same geometry as here.)

Also 0M-4 static scans of cell A and cell B over
    H in {512, 768, 1024} and eps_fill in {0.03, 0.04, 0.06}
at the near-jump geometry, plus a gap scan at production (512,
0.04): gaps {0.20, 0.10, 0.063, 0.055, 0.045} with R_disk = 0.35,
R_hole = R_disk + gap, R_outer = 0.93.

All forms are the projected admissible-space object
    K(u, v) = < P u, L^dag P v >   (componentwise zero-mean P),
polarized from solves; eta04 and phases as in grid_m4_coupling.

Usage:
    uv run python scripts/experiments/grid_anisotropy_cells.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.torch.transport.boundary_flux_grid import (  # noqa: E402
    BoundaryFluxPolicies,
    BoundaryFluxToGrid,
)
from src.torch.transport.phase_grid import PhaseGridConfig  # noqa: E402
from src.torch.transport.weighted_poisson import (  # noqa: E402
    WeightedPoissonConfig,
    WeightedPoissonOperator,
)

DT = torch.float64
R_DISK = 0.35
R_OUT = 0.93
GAPS = (0.20, 0.10, 0.063, 0.055, 0.045)
SCAN_GAP = 0.055


def phase_deg(a, b):
    return math.degrees((math.atan2(b, a) / 4.0) % (math.pi / 2.0))


def grid_xy(H):
    h = 4.0 / H
    c = torch.arange(H, dtype=DT) * h - 2.0 + 0.5 * h
    X, Y = torch.meshgrid(c, c, indexing="ij")
    return X, Y, h


def smooth_step(t, w):
    """C^inf-ish radial step of width w (erf profile)."""
    return 0.5 * (1.0 + torch.erf(t / (w / 2.0)))


def analytic_rho(H, eps, r_hole):
    """Radial two-phase indicator: disk (r < R_DISK) union annulus
    (r_hole < r < R_OUT), each wall smoothed over eps."""
    X, Y, h = grid_xy(H)
    r = torch.sqrt(X * X + Y * Y)
    rho = (smooth_step(R_DISK - r, eps)
           + smooth_step(r - r_hole, eps) * smooth_step(R_OUT - r,
                                                        eps))
    return rho.clamp(0.0, 1.0), X, Y, h


def ring(X, Y, R, eps, m=0, s=False):
    """Analytic ring RHS at radius R: radial bump x {1, cos4, sin4}."""
    r = torch.sqrt(X * X + Y * Y)
    g = torch.exp(-((r - R) / (eps / 2.0)) ** 2)
    if m == 0:
        return g
    ph = torch.atan2(Y, X)
    return g * (torch.sin(4 * ph) if s else torch.cos(4 * ph))


def coupling(op, u0, u4c, u4s):
    """Projected bilinear-form coupling of grid RHS fields."""
    def solve_p(b):
        bp = op.project_componentwise_zero_mean(b)
        phi, info = op.solve(bp)
        if not info.converged:
            raise RuntimeError(f"PCG not converged: {info}")
        return bp, phi

    cache = {}

    def K(a, bt):
        key = (id(a), id(bt))
        if key not in cache:
            ap, _ = solve_p(a)
            _, phib = solve_p(bt)
            cache[key] = float(op.inner(ap, phib))
        return cache[key]

    b00 = K(u0, u0)
    b4c4c = K(u4c, u4c)
    b4s4s = K(u4s, u4s)
    b04c = K(u0, u4c)
    b04s = K(u0, u4s)
    b4c4s = K(u4c, u4s)
    cnorm = math.hypot(b04c, b04s)
    eta04 = cnorm / math.sqrt(max(b00 * 0.5 * (b4c4c + b4s4s),
                                  1e-300))
    return dict(B00=b00, B4c4c=b4c4c, B4s4s=b4s4s, B4c4s=b4c4s,
                c4=(b04c, b04s), eta04=eta04,
                phi_c4_deg=phase_deg(b04c, b04s))


def make_op(rho, H, eps):
    pcfg = WeightedPoissonConfig(grid_shape=(H, H),
                                 support_threshold=3e-3)
    return WeightedPoissonOperator(rho, pcfg)


def circle_cloud(R, n):
    t = torch.arange(n, dtype=DT) * 2 * math.pi / n
    pos = torch.stack([R * t.cos(), R * t.sin()], 1)
    return pos, t


def cell_A(H, eps, r_hole, walls=("disk", "hole", "outer")):
    rho, X, Y, h = analytic_rho(H, eps, r_hole)
    op = make_op(rho, H, eps)
    out = {}
    radii = dict(disk=R_DISK, hole=r_hole, outer=R_OUT)
    for w in walls:
        R = radii[w]
        u0 = ring(X, Y, R, eps)
        u4c = ring(X, Y, R, eps, m=4)
        u4s = ring(X, Y, R, eps, m=4, s=True)
        out[w] = coupling(op, u0, u4c, u4s)
    return out, op.n_components


def cell_B(H, eps, r_hole, walls=("disk", "hole", "outer")):
    rho, X, Y, h = analytic_rho(H, eps, r_hole)
    op = make_op(rho, H, eps)
    pcfg = PhaseGridConfig(grid_shape=(H, H), fill_epsilon=eps,
                           support_threshold=3e-3,
                           projection_rel_tol=5e-2)
    spec = dict(disk=(R_DISK, 90, +1.0), hole=(r_hole, 141, -1.0),
                outer=(R_OUT, 256, +1.0))
    out = {}
    for w in walls:
        R, n, orient = spec[w]
        pos, t = circle_cloud(R, n)
        nrm = torch.stack([t.cos(), t.sin()], 1) * orient
        m = torch.full((n,), 2 * math.pi * R / n, dtype=DT)
        q = torch.ones(n, dtype=DT)
        flux = BoundaryFluxToGrid(pos, nrm, m, q, pcfg,
                                  BoundaryFluxPolicies())
        z = torch.zeros(n, dtype=DT)
        u0 = flux(torch.ones(n, dtype=DT), z)
        u4c = flux(torch.cos(4 * t), z)
        u4s = flux(torch.sin(4 * t), z)
        out[w] = coupling(op, u0, u4c, u4s)
    return out, op.n_components


def show(tag, rec):
    for w, r in rec.items():
        print(f"  [{tag}/{w}] eta04 {r['eta04']:.3e}  phi(c4) "
              f"{r['phi_c4_deg']:5.1f}  H4 asym "
              f"{(r['B4c4c'] - r['B4s4s']) / max(r['B4c4c'], 1e-300):+.2e}",
              flush=True)


def main():
    torch.set_default_dtype(DT)
    report = dict(gap_scan={}, H_scan={}, eps_scan={})

    print("== gap scan (H=512, eps=0.04) ==", flush=True)
    for gap in GAPS:
        rh = R_DISK + gap
        a, nc_a = cell_A(512, 0.04, rh)
        b, nc_b = cell_B(512, 0.04, rh)
        report["gap_scan"][gap] = dict(A=a, B=b, ncomp=(nc_a, nc_b))
        print(f" gap {gap} (ncomp {nc_a}/{nc_b}):")
        show("A", a)
        show("B", b)

    print(f"== H scan (gap={SCAN_GAP}, eps=0.04) ==", flush=True)
    for H in (512, 768, 1024):
        rh = R_DISK + SCAN_GAP
        a, _ = cell_A(H, 0.04, rh)
        b, _ = cell_B(H, 0.04, rh)
        report["H_scan"][H] = dict(A=a, B=b)
        print(f" H {H}:")
        show("A", a)
        show("B", b)

    print(f"== eps scan (H=512, gap={SCAN_GAP}) ==", flush=True)
    for eps in (0.03, 0.04, 0.06):
        rh = R_DISK + SCAN_GAP
        a, _ = cell_A(512, eps, rh)
        b, _ = cell_B(512, eps, rh)
        report["eps_scan"][eps] = dict(A=a, B=b)
        print(f" eps {eps}:")
        show("A", a)
        show("B", b)

    outp = Path("results/reports/phase3c0m3_cells.json")
    outp.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
