"""0C-5b: label-free component rank inside the MM step.

Scope unchanged from 0C-5a (q = 1, u = m, valid separated boundaries, no
redistribution, no deletion); the ONLY new ingredient is that C is read from
the weighted interior operator's singular spectrum instead of being supplied
by the oracle. Fusion / rank hysteresis remain a later stage.

Rank detection uses TWO independent detectors and refuses to guess:

    absolute-null   #{k : s_(k)/s_max < rank_null_tol}
    max-gap         argmax_k s_(k+1)/s_(k), winning ratio >= rank_gap_min

Disagreement raises AmbiguousComponentRankError. r_comp is NOT used as rank
validation: once the displacement basis is built from the selected U0, it is
machine zero by construction even for a wrong rank.

Three parts:

    parity      auto vs oracle one-step on six geometries spanning
                (M, C) = (1,1), (2,1), (2,2), (2,2), (3,2), (3,3).
                Same detected rank implies the same SVD subspace, so
                displacements and objectives must agree to machine level.
    spectra     the leading relative spectrum + detector margins per
                geometry (the raw evidence the thresholds rest on).
    evolution   short pre-contact trajectories (two disks, two ellipses),
                auto and oracle stepped in lockstep with DIRECT per-step
                tensor comparison (max position / circular-angle diff):
                require C_hat = 2 at every step, no premature collapse,
                dimensionless m-leakage |sum m s|/sum m|s| at
                discretisation level, subspace continuity angle small.

Usage
-----
    uv run python scripts/experiments/spectral_rank_auto_0c5b.py \
        --out results/spectral_0c5b
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.shapes.generator import (
    generate_oriented_annulus,
    generate_oriented_circle,
    generate_oriented_ellipse,
)
from src.torch.solver.mm_step import MMConfig, MMStepper

DTYPE = torch.float64
DT = 1e-5
H = 0.05


def n_for(length: float, h: float = H) -> int:
    return max(16, round(length / h))


def circle(R, cx, cy=0.0, h=H):
    return generate_oriented_circle(n_for(2 * math.pi * R, h), R, (cx, cy),
                                    "cpu", DTYPE)


def ellipse(a, b, cx, cy=0.0, h=H):
    peri = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    return generate_oriented_ellipse(n_for(peri, h), a, b, (cx, cy),
                                     "cpu", DTYPE, "arc_length")


def cat(*parts):
    return OrientedPointCloudVarifold(
        positions=torch.cat([p.positions for p in parts]),
        angles=torch.cat([p.angles for p in parts]))


def component_slices(parts):
    out, k = [], 0
    for p in parts:
        out.append((k, k + p.n_points))
        k += p.n_points
    return out


def geometries(h: float = H) -> dict:
    """name -> (varifold, oracle C, component index ranges)."""
    disk = [circle(1.0, 0.0, h=h)]
    ann = [generate_oriented_annulus(n_for(2 * math.pi, h), 1.0, 0.5,
                                     (0.0, 0.0), "cpu", DTYPE)]
    two_disks = [circle(0.5, -1.5, h=h), circle(1.0, 2.0, h=h)]
    two_ellipses = [ellipse(0.7, 0.4, -1.6, h=h), ellipse(1.2, 0.8, 1.8, h=h)]
    ann_disk = [generate_oriented_annulus(n_for(2 * math.pi, h), 1.0, 0.5,
                                          (-1.8, 0.0), "cpu", DTYPE),
                circle(0.6, 1.8, h=h)]
    three = [circle(0.5, -2.5, h=h), circle(0.8, 0.0, 2.2, h=h),
             circle(1.0, 2.7, h=h)]
    return {
        "disk": (cat(*disk), 1, component_slices(disk)),
        "annulus": (cat(*ann), 1, [(0, ann[0].n_points)]),
        "two_disks": (cat(*two_disks), 2, component_slices(two_disks)),
        "two_ellipses": (cat(*two_ellipses), 2,
                         component_slices(two_ellipses)),
        "annulus_disk": (cat(*ann_disk), 2, component_slices(ann_disk)),
        "three_disks": (cat(*three), 3, component_slices(three)),
    }


def config(rank_mode: str, rank: int | None) -> MMConfig:
    return MMConfig(time_step=DT, use_unit_coherence=True,
                    bem_epsilon_mode="carrier_segment_length",
                    angle_sigma=0.1,
                    bem_solver_mode="spectral_bordered",
                    bem_trace_side="interior",
                    bem_rank_mode=rank_mode,
                    bem_component_rank=rank,
                    optimizer_method="trust-ncg", optimizer_tol=1e-8,
                    optimizer_max_iter=300)


def flux_leakage(s, u, slices) -> list:
    """Dimensionless per-component flux leakage

        ell_alpha = |sum_{i in E_alpha} u_i s_i| / sum_{i in E_alpha} u_i |s_i|

    with u = m (carrier) or u = q*m (current). A raw flux (area/time) must
    not be compared against a point speed (length/time); this normalisation
    makes cases of different shape and scale comparable."""
    out = []
    for i, j in slices:
        denom = (u[i:j] * s[i:j].abs()).sum().item()
        num = abs((u[i:j] * s[i:j]).sum().item())
        out.append(num / max(denom, 1e-300))
    return out


def circular_angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def parity_case(name: str, v, C_true: int, slices) -> dict:
    """One step with auto rank vs oracle rank; label used only here, for
    diagnostics -- the solver never sees it in auto mode."""
    st_auto = MMStepper(config("auto", None))
    r_auto = st_auto.step(v)
    st_orc = MMStepper(config("oracle", C_true))
    r_orc = st_orc.step(v)
    s_a, s_o = r_auto.displacements, r_orc.displacements
    m = r_auto.masses
    fluxes = [((s_a[i:j] * m[i:j]).sum() / DT).item() for i, j in slices]
    return dict(
        name=name, N=v.n_points, C_true=C_true,
        C_auto=r_auto.detected_rank,
        rank_gap_ratio=r_auto.rank_gap_ratio,
        rank_abs_level=r_auto.rank_abs_level,
        spectrum_head=r_auto.rank_spectrum_head,
        max_s_diff=(s_a - s_o).abs().max().item(),
        objective_diff=abs(r_auto.objective - r_orc.objective),
        max_s_over_dt=(s_a.abs().max() / DT).item(),
        fluxes_over_dt=fluxes,
        leakage_m=flux_leakage(s_a, m, slices),
        r_comp=r_auto.r_comp,
        objective_decreased=bool(r_auto.objective_decreased),
    )


def evolve_pair(name: str, parts, n_steps: int) -> dict:
    """Auto and oracle trajectories stepped in LOCKSTEP with direct
    per-step tensor comparison (max position and circular-angle
    differences) -- a scalar checksum cannot establish trajectory
    equality. Per-step rank evidence and dimensionless leakage recorded
    from the auto run."""
    va, vo = cat(*parts), cat(*parts)
    slices = component_slices(parts)
    st_a = MMStepper(config("auto", None))
    st_o = MMStepper(config("oracle", 2))
    frames = []
    for step in range(n_steps):
        ra = st_a.step(va)
        ro = st_o.step(vo)
        m, s = ra.masses, ra.displacements
        frames.append(dict(
            step=step,
            C=ra.detected_rank,
            rank_gap_ratio=ra.rank_gap_ratio,
            rank_abs_level=ra.rank_abs_level,
            subspace_angle_deg=ra.subspace_angle_deg,
            max_s_over_dt=(s.abs().max() / DT).item(),
            fluxes_over_dt=[((s[i:j] * m[i:j]).sum() / DT).item()
                            for i, j in slices],
            leakage_m=flux_leakage(s, m, slices),
            max_position_diff=(ra.varifold.positions
                               - ro.varifold.positions).abs().max().item(),
            max_angle_diff=circular_angle_diff(
                ra.varifold.angles, ro.varifold.angles).abs().max().item(),
            objective_decreased=bool(ra.objective_decreased),
        ))
        va, vo = ra.varifold, ro.varifold
    return dict(name=name, n_steps=n_steps, frames=frames,
                worst_position_diff=max(f["max_position_diff"]
                                        for f in frames),
                worst_angle_diff=max(f["max_angle_diff"] for f in frames))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    print("=== 0C-5b-2: auto-vs-oracle parity (one step, h = 0.05) ===")
    hdr = (f"{'geometry':>14} {'N':>4} {'C':>2} {'C^':>3} {'gap':>8} "
           f"{'abs':>8} {'|s_a-s_o|':>10} {'|F_a-F_o|':>10} "
           f"{'max|s|/dt':>10} fluxes/dt")
    print(hdr)
    parity = []
    for name, (v, C_true, slices) in geometries().items():
        row = parity_case(name, v, C_true, slices)
        parity.append(row)
        fl = ", ".join(f"{f:.1e}" for f in row["fluxes_over_dt"])
        print(f"{name:>14} {row['N']:>4} {C_true:>2} {row['C_auto']:>3} "
              f"{row['rank_gap_ratio']:>8.1f} {row['rank_abs_level']:>8.1e} "
              f"{row['max_s_diff']:>10.2e} {row['objective_diff']:>10.2e} "
              f"{row['max_s_over_dt']:>10.2e} ({fl})", flush=True)

    print("\n=== 0C-5b-3: short pre-contact evolution (auto rank) ===")
    h = H
    evo_cases = {
        "two_disks": [circle(0.5, -1.5, h=h), circle(1.0, 2.0, h=h)],
        "two_ellipses": [ellipse(0.7, 0.4, -1.6, h=h),
                         ellipse(1.2, 0.8, 1.8, h=h)],
    }
    evolutions = []
    for name, parts in evo_cases.items():
        run = evolve_pair(name, parts, args.steps)
        Cs = [f["C"] for f in run["frames"]]
        worst_leak = max(l for fr in run["frames"] for l in fr["leakage_m"])
        worst_angle = max((fr["subspace_angle_deg"] or 0.0)
                          for fr in run["frames"])
        evolutions.append(run)
        print(f"  {name}: C_hat = {sorted(set(Cs))} over {len(Cs)} steps, "
              f"worst m-leakage = {worst_leak:.2e}, "
              f"worst subspace angle = {worst_angle:.3f} deg, "
              f"auto-vs-oracle max|dx| = {run['worst_position_diff']:.2e}, "
              f"max|dtheta| = {run['worst_angle_diff']:.2e}", flush=True)

    print("\n=== flux refinement (one step, h = 0.05 -> 0.03) ===")
    refinement = []
    for h in (0.05, 0.03):
        for nm, parts in (
                ("two_ellipses", [ellipse(0.7, 0.4, -1.6, h=h),
                                  ellipse(1.2, 0.8, 1.8, h=h)]),
                ("two_disks", [circle(0.5, -1.5, h=h),
                               circle(1.0, 2.0, h=h)])):
            row = parity_case(nm, cat(*parts), 2, component_slices(parts))
            row["h"] = h
            refinement.append(row)
            fl = ", ".join(f"{f:.2e}" for f in row["fluxes_over_dt"])
            print(f"  {nm} h={h}: N={row['N']} fluxes/dt=({fl}) "
                  f"gap={row['rank_gap_ratio']:.0f}", flush=True)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "spectral_rank_auto_0c5b.json"
        path.write_text(json.dumps(dict(
            meta=dict(dt=DT, h=H, q="identically 1", u="m",
                      rank_null_tol=0.05, rank_gap_min=30.0),
            parity=parity, evolutions=evolutions,
            refinement=refinement), indent=2))
        print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
