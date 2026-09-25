"""Static BEM audits that must pass before any annulus evolution is run.

These are cheap (seconds) and they gate the expensive experiments. They were
written after the annulus radial-energy benchmark diverged like O(N); the
cause turned out to be that the solver was assembling the *exterior* trace of
the single-layer potential rather than the interior one.

Tests
-----
T0  trace-side          which jump relation is implemented (1 line, decisive)
T1  annulus area        generator + inner-normal sign, exact arc-length weights
T2  annulus energy      2 W_lin / dt  vs  Q_E(v) = 2 pi a^2 log(R+/R-)
T3  potential profile   phi  vs  -a log r      (note the sign: phi_v = -p)
T4  shape sensitivity   how much the trace side changes W_lin, per shape
T5  nullity             dim ker A  vs  number of *phase* components C
T6  one-step step       does an actual MM step move differently? (not static)

Usage
-----
    uv run python scripts/experiments/bem_static_tests.py
    uv run python scripts/experiments/bem_static_tests.py --out results/bem_static
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.torch.shapes.generator import (
    annulus_exact_masses,
    annulus_point_counts,
    generate_oriented_annulus,
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_flower,
    generate_oriented_star,
)
from src.torch.solver.mm_solver import compute_volume_divergence
from src.torch.transport.bem_wasserstein import (
    BEMWasserstein,
    build_bem_matrices_point,
)

DTYPE = torch.float64
DEV = "cpu"
DT = 1e-5          # only sets the scale of s = dt * v; W_lin / dt is dt-free
EPS_SCALE = 0.1
K_ENDPOINTS = 3


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def annulus_pressure_coeff(R_out: float, R_in: float) -> float:
    """a in p(r) = a log r + b for the one-phase Mullins-Sekerka annulus."""
    return (1.0 / R_out + 1.0 / R_in) / math.log(R_out / R_in)


def annulus_setup(n_out: int, R_out: float, R_in: float):
    """Annulus varifold, exact arc-length weights, and the exact radial data.

    Outward-normal velocities: v_+ = -a/R_+ on the outer ring and v_- = +a/R_-
    on the inner one (its outward normal is -e_r, so v_- > 0 still shrinks the
    hole). Compatibility 2 pi R_+ v_+ + 2 pi R_- v_- = 0 holds exactly.
    """
    n_out, n_in = annulus_point_counts(n_out, R_out, R_in)
    varifold = generate_oriented_annulus(n_out, R_out, R_in, device=DEV, dtype=DTYPE)
    masses = annulus_exact_masses(n_out, n_in, R_out, R_in, device=DEV, dtype=DTYPE)
    a = annulus_pressure_coeff(R_out, R_in)
    vel = torch.cat([
        torch.full((n_out,), -a / R_out, dtype=DTYPE),
        torch.full((n_in,), +a / R_in, dtype=DTYPE),
    ])
    energy = 2 * math.pi * a * a * math.log(R_out / R_in)
    return varifold, masses, vel, a, energy, n_out, n_in


def bem_for(varifold, masses, trace_side: str) -> BEMWasserstein:
    bem = BEMWasserstein(
        method="point", epsilon_scale=EPS_SCALE,
        n_endpoints=K_ENDPOINTS, trace_side=trace_side,
    )
    bem.setup_for_step(varifold.positions, varifold.normals, masses, sigma=None)
    bem.setup_coherence(torch.ones(len(masses), dtype=DTYPE))
    return bem


def linearized_energy(bem: BEMWasserstein, vel: torch.Tensor) -> float:
    """2 W_lin / dt, which equals the tangent energy Q_E(v)."""
    zero = torch.zeros_like(vel)
    return (2.0 * bem(DT * vel, zero, DT) / DT).item()


def geometric_masses(positions: torch.Tensor) -> torch.Tensor:
    """Arc-length weights from neighbour distances, independent of the KDE."""
    d = (positions.roll(-1, 0) - positions).norm(dim=1)
    return 0.5 * (d + d.roll(1, 0))


# --------------------------------------------------------------------------
# T0: which trace is implemented?
# --------------------------------------------------------------------------

def test_trace_side() -> list[dict]:
    """A uniform density on a circle makes the single layer constant inside,
    so the *interior* trace (+1/2 I + K*) 1 must vanish. The exterior trace
    does not. This distinguishes the two in one line -- the energy test on a
    single circle cannot, because K* annihilates mean-zero Fourier modes there
    and the two sign flips cancel.
    """
    rows = []
    for n in (64, 128, 256, 512, 1024, 2048):
        v = generate_oriented_circle(n, 1.0, (0.0, 0.0), device=DEV, dtype=DTYPE)
        m = torch.full((n,), 2 * math.pi / n, dtype=DTYPE)
        _, K_star = build_bem_matrices_point(
            v.positions, v.normals, m, EPS_SCALE * m.median().item()
        )
        one = torch.ones(n, dtype=DTYPE)
        rows.append(dict(
            n=n,
            interior_residual=(0.5 * one + K_star @ one).abs().max().item(),
            exterior_residual=(-0.5 * one + K_star @ one).abs().max().item(),
            mean_Kstar_one=(K_star @ one).mean().item(),
        ))
    return rows


# --------------------------------------------------------------------------
# T1 / T2 / T3: annulus
# --------------------------------------------------------------------------

def test_annulus_area() -> list[dict]:
    """Exact arc-length weights must reproduce pi (R+^2 - R-^2) to roundoff.
    This checks the generator and the inner-normal sign, with no KDE error
    mixed in (the KDE carrier estimator gets its own refinement test).
    """
    rows = []
    for R_in in (0.5, 0.3, 0.1):
        exact = math.pi * (1.0 - R_in ** 2)
        for n_out in (64, 128, 256, 512, 1024):
            no, ni = annulus_point_counts(n_out, 1.0, R_in)
            v = generate_oriented_annulus(n_out, 1.0, R_in, device=DEV, dtype=DTYPE)
            m = annulus_exact_masses(no, ni, 1.0, R_in, device=DEV, dtype=DTYPE)
            area = compute_volume_divergence(v, m)
            n = v.normals
            contrib = 0.5 * (v.positions * n).sum(1) * m
            rows.append(dict(
                R_in=R_in, n_outer=no, n_inner=ni,
                area=area, area_exact=exact,
                rel_err=abs(area - exact) / exact,
                outer_contrib=contrib[:no].sum().item(),
                inner_contrib=contrib[no:].sum().item(),
                inner_contrib_exact=-math.pi * R_in ** 2,
            ))
    return rows


def test_annulus_energy() -> list[dict]:
    """The gate. Sweeps R-/R+ because eps_BEM is a *global* median segment
    length: the outer ring has more points, so as the hole shrinks the median
    stays pinned to the outer scale and eps/l_- grows. The inner boundary can
    therefore go sub-resolution in the BEM regularisation before it does in
    sigma or delta.
    """
    rows = []
    for R_in in (0.5, 0.3, 0.2, 0.1):
        for n_out in (64, 128, 256, 512):
            varifold, m, vel, a, exact, no, ni = annulus_setup(n_out, 1.0, R_in)
            row = dict(R_in=R_in, n_outer=no, n_inner=ni, energy_exact=exact,
                       compat=(vel * m).sum().item())
            for side in ("legacy_exterior", "interior"):
                bem = bem_for(varifold, m, side)
                row[f"energy_{side}"] = linearized_energy(bem, vel)
                row[f"ratio_{side}"] = row[f"energy_{side}"] / exact
                if side == "interior":
                    eps = bem.endpoint_operator().epsilon
                    row["eps_bem"] = eps
                    row["eps_over_inner_spacing"] = eps / (2 * math.pi * R_in / ni)
                    row["eps_over_R_in"] = eps / R_in
            rows.append(row)
    return rows


def test_potential_profile() -> list[dict]:
    """The Wasserstein tangent potential satisfies d_nu phi = v while the MS
    pressure satisfies -d_nu p = v, so phi = -p = -a log r + const. The target
    is a_BEM / (-a) -> 1; a *negative* fitted coefficient is correct.
    """
    rows = []
    for R_in in (0.5, 0.2):
        for n_out in (128, 256, 512):
            varifold, m, vel, a, _, no, ni = annulus_setup(n_out, 1.0, R_in)
            for side in ("legacy_exterior", "interior"):
                bem = bem_for(varifold, m, side)
                op = bem.endpoint_operator()
                w = op.weights
                V = bem.expand_to_endpoints(vel).clone()
                V = V - (w * V).sum() / w.sum()
                rhs = V if side == "interior" else -V
                Q, A = op.constraint_basis, op.trace_operator
                y = torch.linalg.solve(Q.T @ A @ Q, Q.T @ rhs)
                phi = op.single_layer @ (Q @ y)
                phi = phi - (w * phi).sum() / w.sum()
                logr = torch.log(op.positions.norm(dim=1))
                basis = torch.stack([logr, torch.ones_like(logr)], 1)
                sw = w.sqrt()[:, None]
                coef = torch.linalg.lstsq(basis * sw, phi * sw.squeeze()).solution
                rows.append(dict(
                    R_in=R_in, n_outer=no, trace_side=side,
                    a_exact=a, a_bem=coef[0].item(),
                    ratio_to_minus_a=(coef[0] / (-a)).item(),
                ))
    return rows


# --------------------------------------------------------------------------
# T4: how far does the trace side reach?
# --------------------------------------------------------------------------

def test_shape_sensitivity(n: int = 256) -> list[dict]:
    """K* v = 0 holds only for the circle. Everywhere else the two traces give
    materially different energies, so the trace fix is not confined to
    multi-component runs -- it touches every published shape.
    """
    shapes = {
        "circle": generate_oriented_circle(n, 1.0, (0, 0), DEV, DTYPE),
        "ellipse_2": generate_oriented_ellipse(n, 1.0, 0.5, (0, 0), DEV, DTYPE),
        "ellipse_4": generate_oriented_ellipse(n, 1.0, 0.25, (0, 0), DEV, DTYPE),
        "flower_5": generate_oriented_flower(n, 5, 0.5, 1.0, (0, 0), DEV, DTYPE),
        "star_5": generate_oriented_star(n, 5, 0.4, 1.0, (0, 0), DEV, DTYPE),
    }
    t = torch.linspace(0, 2 * math.pi, n + 1, dtype=DTYPE)[:-1]
    rows = []
    for name, v in shapes.items():
        m = geometric_masses(v.positions)
        vel = torch.cos(2 * t)
        vel = vel - (vel * m).sum() / m.sum()          # volume-preserving
        bem_i = bem_for(v, m, "interior")
        op = bem_i.endpoint_operator()
        w = op.weights
        V = bem_i.expand_to_endpoints(vel).clone()
        V = V - (w * V).sum() / w.sum()
        Kv = (op.adjoint_double_layer @ V).norm() / V.norm()
        e_ext = linearized_energy(bem_for(v, m, "legacy_exterior"), vel)
        e_int = linearized_energy(bem_i, vel)
        rows.append(dict(
            shape=name, Kstar_v_rel=Kv.item(),
            energy_legacy_exterior=e_ext, energy_interior=e_int,
            rel_difference=abs(e_int - e_ext) / abs(e_int),
        ))
    return rows


# --------------------------------------------------------------------------
# T5: nullity == number of phase components
# --------------------------------------------------------------------------

def _rings_to_varifold(rings, spacing: float):
    """rings: (R, cx, cy, inward). Inner boundaries carry angles t + pi."""
    pos, ang, mas = [], [], []
    for R, cx, cy, inward in rings:
        n = max(12, int(round(2 * math.pi * R / spacing)))
        v = generate_oriented_circle(n, R, (cx, cy), device=DEV, dtype=DTYPE)
        pos.append(v.positions)
        ang.append(v.angles + (math.pi if inward else 0.0))
        mas.append(torch.full((n,), 2 * math.pi * R / n, dtype=DTYPE))

    class _V:
        positions = torch.cat(pos)
        angles = torch.cat(ang)
        normals = torch.stack([torch.cos(angles), torch.sin(angles)], 1)

    return _V(), torch.cat(mas)


def test_nullity(spacing: float = 0.03) -> list[dict]:
    """dim ker A equals the number of connected components of the *phase* C,
    not the number of boundary components M. Read off the weighted operator
    W^{1/2} A W^{-1/2} at endpoint level; the gap s_(C)/s_(C+1) is what
    identifies C without any labels.

    Caveat: this is the clean q == 1 regime with positive, nondegenerate
    weights. Near a fusion event zeta ~ q_i m_i with q_i -> 0 and W^{-1/2}
    degrades; that case needs its own study.
    """
    cases = [
        ("disk", [(1.0, 0.0, 0.0, False)], 1, 1),
        ("annulus", [(1.0, 0, 0, False), (0.5, 0, 0, True)], 2, 1),
        ("two_disks", [(1.0, -3, 0, False), (1.0, 3, 0, False)], 2, 2),
        ("annulus_plus_disk",
         [(1.0, 0, 0, False), (0.5, 0, 0, True), (0.8, 4, 0, False)], 3, 2),
        ("three_disks",
         [(0.9, -3, 0, False), (0.9, 0, 2.6, False), (0.9, 3, 0, False)], 3, 3),
        ("disk_two_holes",
         [(1.5, 0, 0, False), (0.4, -0.6, 0, True), (0.4, 0.6, 0, True)], 3, 1),
    ]
    rows = []
    for name, rings, M, C in cases:
        v, m = _rings_to_varifold(rings, spacing)
        op = bem_for(v, m, "interior").endpoint_operator()
        s = torch.linalg.svdvals(op.weighted_trace_operator()).sort().values
        rows.append(dict(
            case=name, M=M, C=C, n_endpoints=len(op.weights),
            singular_values_low=[x.item() for x in s[:6]],
            gap_at_C=(s[C - 1] / s[C]).item(),
        ))
    return rows


# --------------------------------------------------------------------------
# T6: does an actual MM step move differently?
# --------------------------------------------------------------------------

def test_one_step_trajectory(n: int = 128, max_iter: int = 300,
                             gtol: float = 1e-10) -> list[dict]:
    """T4 measures a quadratic form against an artificial test velocity. This
    measures the returned one-step displacement instead.

    Read this as a *finite-budget* comparison, not a comparison of MM
    minimizers: with gtol=1e-10 neither trace converges within `max_iter`, so
    what differs is where trust-ncg stopped. That is still enough to conclude
    the traces induce different metrics -- T4 already settles that
    independently -- but the percentages are provisional until the optimizer
    audit lands.

    Two further cautions baked into the reported fields:
      * `displacement_rel_diff` divides by ||s_interior||, so a small
        denominator inflates it. `symmetric_rel_diff` and `cosine_similarity`
        separate "different direction" from "small reference".
      * `wasserstein` on MMStepResult is `_last_wasserstein`, the value at
        whichever point the optimizer last evaluated -- not necessarily
        `result.x`. Treat `wasserstein_rel_diff` as indicative only.

    The circle is stationary, so its displacement is ~0 and the ratio is 0/0;
    reported as None.
    """
    from src.torch.shapes.generator import generate_oriented_two_ellipses
    from src.torch.solver.mm_step import MMConfig, MMStepper

    builders = {
        "circle": lambda: generate_oriented_circle(n, 1.0, (0, 0), DEV, DTYPE),
        "ellipse_2": lambda: generate_oriented_ellipse(n, 1.0, 0.5, (0, 0), DEV, DTYPE),
        "flower_5": lambda: generate_oriented_flower(n, 5, 0.5, 1.0, (0, 0), DEV, DTYPE),
        "star_5": lambda: generate_oriented_star(n, 5, 0.4, 1.0, (0, 0), DEV, DTYPE),
        "two_ellipses": lambda: generate_oriented_two_ellipses(
            n // 2, device=DEV, dtype=DTYPE),
    }
    rows = []
    for name, build in builders.items():
        out, spacing = {}, None
        for side in ("legacy_exterior", "interior"):
            cfg = MMConfig(time_step=DT, bem_trace_side=side,
                           optimizer_method="trust-ncg", optimizer_tol=gtol,
                           optimizer_max_iter=max_iter)
            v = build()
            if spacing is None:
                spacing = geometric_masses(v.positions).median().item()
            out[side] = MMStepper(cfg).step(v)

        ext, interior = out["legacy_exterior"], out["interior"]
        se, si = ext.displacements, interior.displacements
        ne, ni_ = se.norm().item(), si.norm().item()
        diff = (se - si).norm().item()
        tiny = max(ne, ni_) < 1e-14
        rows.append(dict(
            shape=name,
            converged_exterior=bool(ext.converged),
            converged_interior=bool(interior.converged),
            n_iter_exterior=int(ext.n_iter), n_iter_interior=int(interior.n_iter),
            norm_exterior=ne, norm_interior=ni_, norm_difference=diff,
            # ratio to the interior run, i.e. what a naive report shows
            displacement_rel_diff=None if ni_ < 1e-14 else diff / ni_,
            # symmetric version, immune to a small denominator
            symmetric_rel_diff=None if tiny else 2 * diff / (ne + ni_),
            # separates "rotated" from "rescaled"
            cosine_similarity=None if tiny
            else (torch.dot(se, si) / (se.norm() * si.norm())).item(),
            # is the step even resolved against the point spacing?
            max_step_over_spacing=si.abs().max().item() / spacing,
            # indicative only: see the docstring on _last_wasserstein
            wasserstein_rel_diff=abs(ext.wasserstein - interior.wasserstein)
            / abs(interior.wasserstein),
        ))
    return rows


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None,
                    help="directory to write raw results (JSON)")
    args = ap.parse_args()

    results = {}

    print("=" * 78)
    print("T0  trace-side: uniform density on a circle")
    print("    single layer is constant inside => interior trace must vanish")
    results["T0_trace_side"] = test_trace_side()
    print(f"{'N':>6} {'(+1/2 I + K*)1':>16} {'(-1/2 I + K*)1':>16} {'mean K* 1':>11}")
    for r in results["T0_trace_side"]:
        print(f"{r['n']:>6} {r['interior_residual']:>16.3e} "
              f"{r['exterior_residual']:>16.3e} {r['mean_Kstar_one']:>11.6f}")

    print("\n" + "=" * 78)
    print("T1  annulus area with exact arc-length weights")
    results["T1_annulus_area"] = test_annulus_area()
    worst = max(r["rel_err"] for r in results["T1_annulus_area"])
    print(f"    worst relative error over all (R_in, N): {worst:.3e}")
    r0 = results["T1_annulus_area"][0]
    print(f"    inner-ring contribution at R_in={r0['R_in']}: "
          f"{r0['inner_contrib']:+.9f} (exact {r0['inner_contrib_exact']:+.9f})")

    print("\n" + "=" * 78)
    print("T2  annulus radial BEM energy,  2 W_lin/dt  vs  2 pi a^2 log(R+/R-)")
    results["T2_annulus_energy"] = test_annulus_energy()
    print(f"{'R-':>5} {'n_out':>6} {'legacy_exterior':>16} {'interior':>11} "
          f"{'eps/l_-':>9} {'eps/R_-':>9}")
    for r in results["T2_annulus_energy"]:
        print(f"{r['R_in']:>5.1f} {r['n_outer']:>6} "
              f"{r['ratio_legacy_exterior']:>16.4f} {r['ratio_interior']:>11.6f} "
              f"{r['eps_over_inner_spacing']:>9.3f} {r['eps_over_R_in']:>9.4f}")

    print("\n" + "=" * 78)
    print("T3  potential profile: phi = -p = -a log r + const")
    results["T3_potential"] = test_potential_profile()
    print(f"{'R-':>5} {'n_out':>6} {'trace_side':>16} {'a_BEM/(-a)':>12}")
    for r in results["T3_potential"]:
        print(f"{r['R_in']:>5.1f} {r['n_outer']:>6} {r['trace_side']:>16} "
              f"{r['ratio_to_minus_a']:>12.6f}")

    print("\n" + "=" * 78)
    print("T4  shape sensitivity: K* v = 0 only for the circle")
    results["T4_shape_sensitivity"] = test_shape_sensitivity()
    print(f"{'shape':>12} {'||K* v||/||v||':>15} {'rel. energy diff':>18}")
    for r in results["T4_shape_sensitivity"]:
        print(f"{r['shape']:>12} {r['Kstar_v_rel']:>15.4f} "
              f"{r['rel_difference']:>17.2%}")

    print("\n" + "=" * 78)
    print("T5  nullity of A_int == number of phase components C (not M)")
    results["T5_nullity"] = test_nullity()
    print(f"{'case':>20} {'M':>3} {'C':>3} {'s_(1)':>10} {'s_(2)':>10} "
          f"{'s_(3)':>10} {'s_(C)/s_(C+1)':>14}")
    for r in results["T5_nullity"]:
        s = r["singular_values_low"]
        print(f"{r['case']:>20} {r['M']:>3} {r['C']:>3} {s[0]:>10.2e} "
              f"{s[1]:>10.2e} {s[2]:>10.2e} {r['gap_at_C']:>14.3e}")

    print("\n" + "=" * 78)
    print("T6  one-step MM displacement (not static; takes ~30 s)")
    print("    PROVISIONAL: neither trace reaches gtol here, so this compares")
    print("    where the optimizer stopped, not two MM minimizers.")
    results["T6_one_step"] = test_one_step_trajectory()
    print(f"{'shape':>14} {'conv e/i':>9} {'|s_ext|':>10} {'|s_int|':>10} "
          f"{'sym rel':>9} {'cos':>8} {'max|s|/h':>9}")
    for r in results["T6_one_step"]:
        conv = f"{str(r['converged_exterior'])[0]}/{str(r['converged_interior'])[0]}"
        sym = r["symmetric_rel_diff"]
        cos = r["cosine_similarity"]
        sym_s = "(stat.)" if sym is None else f"{sym:.1%}"
        cos_s = "  --  " if cos is None else f"{cos:+.4f}"
        print(f"{r['shape']:>14} {conv:>9} {r['norm_exterior']:>10.3e} "
              f"{r['norm_interior']:>10.3e} {sym_s:>9} {cos_s:>8} "
              f"{r['max_step_over_spacing']:>9.3f}")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        meta = dict(dtype=str(DTYPE), dt=DT, epsilon_scale=EPS_SCALE,
                    n_endpoints=K_ENDPOINTS)
        path = args.out / "bem_static_tests.json"
        path.write_text(json.dumps({"meta": meta, **results}, indent=2))
        print(f"\nraw results -> {path}")


if __name__ == "__main__":
    main()
