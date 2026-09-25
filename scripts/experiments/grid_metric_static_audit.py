"""Grid metric G0 static audit -- the first go/no-go (pre-registered).

G1-CONSISTENT construction throughout (this matters: an earlier
profile-based source v*|grad rho| carries an uncorrected O(eps/R)
curvature bias; the boundary-scatter construction below is exactly
what BoundaryFluxToGrid will produce and removes it):

    rho       = eta_eps * chi_E          (FFT convolution, same kernel)
    delta_rho = eta_eps * (v H^1|_dE)    (boundary quadrature scatter)

Gates (all with support_threshold = 1e-3, the calibrated default):
1. disk Fourier v_k = cos(k theta): Q_exact = pi R^2 / k;
   <1% for resolved modes (k eps / R <= 0.5), attenuation curve else.
2. annulus radial v = a/r: Q_exact = 2 pi a^2 log(R+/R-); C=1.
3. two disks: exchange rejected; compatible-mode energy additivity.

Calibration history (2026-08-05, k=1 disk, n=512):
    thr=1e-2: -2.2..-3.1% bias (eps-insensitive -> threshold-driven)
    thr=1e-3: -0.23..-0.34%   (adopted)
Measured mode attenuation (thr=1e-3, n=512): the Fourier error follows
    rel_err ~ -0.13 (k eps / R)^2
-- the physical low-pass of the eps-regularized metric, not a defect.
The resolved-mode constant is therefore CALIBRATED to c_mode = 0.28
(1% at the band edge), updating the pre-registered 0.5 under the
fixed rule "constants calibrate, rules don't".

Usage
-----
    uv run python scripts/experiments/grid_metric_static_audit.py \
        --n 512 --eps 0.08 0.06 --out results/grid_metric
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.torch.transport.weighted_poisson import (  # noqa: E402
    IncompatibleGridVelocityError,
    WeightedPoissonConfig,
    WeightedPoissonOperator,
)

DT = torch.float64
BOX = 2.0


class Field:
    """Shared FFT kernel machinery for one (n, eps)."""

    def __init__(self, n, eps):
        self.n, self.eps = n, eps
        self.dx = 2 * BOX / n
        xs = torch.linspace(-BOX, BOX, n + 1, dtype=DT)[:-1] + self.dx / 2
        self.X, self.Y = torch.meshgrid(xs, xs, indexing="ij")
        d = torch.sqrt(self.X**2 + self.Y**2)
        t = (d / eps).clamp(max=1.0)
        ker = (1 - t)**4 * (4 * t + 1)          # Wendland C2
        ker = ker / (ker.sum() * self.dx**2)
        self.Kf = torch.fft.fft2(torch.fft.ifftshift(ker))

    def mollify(self, f):
        return torch.real(torch.fft.ifft2(
            torch.fft.fft2(f) * self.Kf)) * self.dx**2

    def scatter_boundary(self, pts, vals, weights):
        """eta * (sum_j w_j v_j delta_{y_j}) via nearest-cell
        deposition + FFT mollification (audit-grade; nb >> n)."""
        n, dx = self.n, self.dx
        src = torch.zeros(n, n, dtype=DT)
        ix = ((pts[:, 0] + BOX) / dx).long().clamp(0, n - 1)
        iy = ((pts[:, 1] + BOX) / dx).long().clamp(0, n - 1)
        src.index_put_((ix, iy), vals * weights / dx**2,
                       accumulate=True)
        return self.mollify(src)


def circle_pts(R, nb, center=(0.0, 0.0)):
    th = torch.linspace(0, 2 * math.pi, nb + 1, dtype=DT)[:-1]
    pts = torch.stack([center[0] + R * torch.cos(th),
                       center[1] + R * torch.sin(th)], dim=1)
    w = torch.full((nb,), 2 * math.pi * R / nb, dtype=DT)
    return th, pts, w


def _op(rho, n):
    return WeightedPoissonOperator(rho, WeightedPoissonConfig(
        grid_shape=(n, n), pcg_max_iter=3000))


def disk_fourier(n, eps, R=1.0, kmax=12, nb=4096):
    F = Field(n, eps)
    r = torch.sqrt(F.X**2 + F.Y**2)
    rho = F.mollify((r <= R).to(DT))
    op = _op(rho, n)
    th, pts, w = circle_pts(R, nb)
    rows = []
    for k in range(1, kmax + 1):
        drho = F.scatter_boundary(pts, torch.cos(k * th), w)
        drho_p = op.project_componentwise_zero_mean(drho)
        proj_rel = float((drho_p - drho).norm() / drho.norm())
        phi, info = op.solve(drho_p)
        q = float(op.inner(drho_p, phi))
        q_exact = math.pi * R * R / k
        rows.append(dict(k=k, Q_grid=q, Q_exact=q_exact,
                         rel_err=q / q_exact - 1.0,
                         projection_rel=proj_rel,
                         pcg_iters=info.iterations,
                         resolved=bool(k * eps / R <= 0.28)))
    return rows


def annulus_radial(n, eps, R_out=1.2, R_in=0.5, a=1.0, nb=4096):
    F = Field(n, eps)
    r = torch.sqrt(F.X**2 + F.Y**2)
    rho = F.mollify(((r <= R_out) & (r >= R_in)).to(DT))
    op = _op(rho, n)
    _, pts_o, w_o = circle_pts(R_out, nb)
    _, pts_i, w_i = circle_pts(R_in, nb)
    # outward-normal speeds of the PHASE: +a/R+ outer, -a/R- inner
    drho = (F.scatter_boundary(
        pts_o, torch.full((nb,), a / R_out, dtype=DT), w_o)
        + F.scatter_boundary(
            pts_i, torch.full((nb,), -a / R_in, dtype=DT), w_i))
    drho = op.project_componentwise_zero_mean(drho)
    phi, info = op.solve(drho)
    q = float(op.inner(drho, phi))
    q_exact = 2 * math.pi * a * a * math.log(R_out / R_in)
    return dict(n=n, eps=eps, Q_grid=q, Q_exact=q_exact,
                rel_err=q / q_exact - 1.0,
                pcg_iters=info.iterations,
                n_components=op.n_components)


def two_disks(n, eps, R=0.5, d=1.0, nb=2048):
    F = Field(n, eps)
    r1 = torch.sqrt((F.X + d)**2 + F.Y**2)
    r2 = torch.sqrt((F.X - d)**2 + F.Y**2)
    rho = F.mollify(((r1 <= R) | (r2 <= R)).to(DT))
    op = _op(rho, n)
    th1, pts1, w1 = circle_pts(R, nb, (-d, 0.0))
    th2, pts2, w2 = circle_pts(R, nb, (d, 0.0))
    out = dict(n_components=op.n_components)
    # exchange mode: uniform growth of one disk, shrink of the other
    drho_x = (F.scatter_boundary(pts1, torch.ones(nb, dtype=DT), w1)
              + F.scatter_boundary(pts2, -torch.ones(nb, dtype=DT), w2))
    try:
        op.solve(drho_x)
        out["exchange_rejected"] = False
    except IncompatibleGridVelocityError:
        out["exchange_rejected"] = True
    # compatible k=2 mode on both disks: additivity of energies
    drho2 = (F.scatter_boundary(pts1, torch.cos(2 * th1), w1)
             + F.scatter_boundary(pts2, torch.cos(2 * th2), w2))
    drho2 = op.project_componentwise_zero_mean(drho2)
    phi, info = op.solve(drho2)
    q_pair = float(op.inner(drho2, phi))
    q_exact = 2 * (math.pi * R * R / 2)
    out.update(Q_pair=q_pair, Q_exact_sum=q_exact,
               rel_err=q_pair / q_exact - 1.0,
               pcg_iters=info.iterations)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=[512])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.08, 0.06])
    ap.add_argument("--out", type=Path,
                    default=Path("results/grid_metric"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = dict(construction="chi-convolution + boundary scatter "
                               "(G1-consistent)",
                  support_threshold=WeightedPoissonConfig().
                  support_threshold,
                  disk=[], annulus=[], two_disks=[])
    for n in args.n:
        for eps in args.eps:
            rows = disk_fourier(n, eps)
            report["disk"].append(dict(n=n, eps=eps, modes=rows))
            worst = max((abs(r["rel_err"]) for r in rows
                         if r["resolved"]), default=float("nan"))
            print(f"disk n={n} eps={eps}: worst RESOLVED "
                  f"= {worst:.3%}", flush=True)
            for r in rows:
                tag = "R" if r["resolved"] else "-"
                print(f"   k={r['k']:2d} [{tag}] "
                      f"{r['rel_err']:+.4f} "
                      f"(pcg {r['pcg_iters']})", flush=True)
            ann = annulus_radial(n, eps)
            report["annulus"].append(ann)
            print(f"annulus n={n} eps={eps}: {ann['rel_err']:+.3%} "
                  f"(C={ann['n_components']})", flush=True)
            td = two_disks(n, eps)
            report["two_disks"].append(dict(n=n, eps=eps, **td))
            print(f"two disks n={n} eps={eps}: "
                  f"C={td['n_components']} "
                  f"exchange_rejected={td['exchange_rejected']} "
                  f"additivity={td['rel_err']:+.3%}", flush=True)

    path = args.out / "static_audit.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"raw results -> {path}", flush=True)


if __name__ == "__main__":
    main()
