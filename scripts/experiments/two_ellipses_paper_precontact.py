"""Corrected pre-contact validation of the PAPER two-ellipse experiment.

Configuration (verified against the archived run's parameters): two
vertical ellipses a = 0.4, b = 1.0 at centers +-0.45 -- minor axes
facing, initial gap g0 = 0.1. The paper ran N = 64 per ellipse
(h ~ 4.60/64 ~ 0.072, so g0/h ~ 1.4: marginal), dt = 1e-5, T = 0.05,
with the legacy exterior/global metric, real coherence and
redistribution. Here everything runs on the corrected spectral
componentwise metric (interior trace, carrier operator, variant M),
q real (sigma = 0.1), no redistribution/deletion, particle IDs fixed.

Three steps, per review; STOP at the first invalid-support/ambiguous
frame -- post-contact still waits for 0D/0E:

  1. ISOLATED refinement: one ellipse (the pair is symmetric), oracle
     C = 1, N in {64, 128, 256} at dt = 1e-4 plus N = 128 at dt = 2e-5.
     Saved: x_max(t), polygon area/centroid, predicted pair gap
     g(t) = 0.9 - 2 x_max(t), interpolated contact time, optimizer
     diagnostics. Prediction (from the committed gap-0.4 trajectory,
     reprocessed): contact near t ~ 0.021, inside the paper window.

  2. PAIR (oracle C = 2) vs isolated superposition, N = 128: gap must
     decrease monotonically, per-component polygon area/centroid held,
     pair-vs-isolated defect small until near contact. NOTE the paper
     gap sits at 1 sigma and below the KDE bandwidth delta from the
     START -- estimator coupling across components is expected to be
     visible here, unlike the resolved gap-0.6 audits; measuring it is
     part of the point.

  3. LABEL-FREE rank (state_machine mode) at N = 128 and 256: run until
     the first hard stop, recording the full evidence stream and the
     working-rank actions.

Usage
-----
    uv run python scripts/experiments/two_ellipses_paper_precontact.py \
        --out results/two_ellipses_paper
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from src.torch.shapes.generator import generate_oriented_ellipse
from src.torch.solver.mm_step import MMStepper
from src.torch.transport.bem_wasserstein import AmbiguousComponentRankError

sys.path.insert(0, str(Path(__file__).parent))
from spectral_flux_measure_0c5c import config as base_config  # noqa: E402
from spectral_rank_auto_0c5b import cat  # noqa: E402

A, B = 0.4, 1.0
D_CENTERS = 0.9          # paper: gap 0.1 -> centers +-0.45
PERI = math.pi * (3 * (A + B) - math.sqrt((3 * A + B) * (A + 3 * B)))


def make_ellipse(n, cx=0.0):
    return generate_oriented_ellipse(n, A, B, (cx, 0.0), "cpu",
                                     torch.float64, "arc_length")


def cfg_for(dt, rank_mode, rank):
    c = base_config("M")            # carrier operator, interior, sigma 0.1
    c.use_unit_coherence = False    # real coherence, as in the paper
    c.time_step = dt
    c.bem_rank_mode = rank_mode
    c.bem_component_rank = rank
    return c


def poly(v):
    X, Y = v.positions[:, 0], v.positions[:, 1]
    Xn, Yn = X.roll(-1), Y.roll(-1)
    c = X * Yn - Xn * Y
    Aa = 0.5 * c.sum()
    return (Aa.item(),
            [(((X + Xn) * c).sum() / (6 * Aa)).item(),
             (((Y + Yn) * c).sum() / (6 * Aa)).item()])


def isolated_run(n, dt, t_end=0.03):
    v = make_ellipse(n)
    stepper = MMStepper(cfg_for(dt, "oracle", 1))
    n_steps = round(t_end / dt)
    every = max(1, n_steps // 30)
    traj, worst_rg = [], 0.0
    for k in range(n_steps):
        r = stepper.step(v)
        v = r.varifold
        worst_rg = max(worst_rg, r.relative_gradient_norm)
        if k % every == 0 or k == n_steps - 1:
            area, cen = poly(v)
            xm = v.positions[:, 0].max().item()
            traj.append(dict(t=(k + 1) * dt, x_max=xm,
                             gap=D_CENTERS - 2 * xm,
                             area_geom=area, centroid_x=cen[0]))
    hit = None
    for p, q in zip(traj, traj[1:]):
        if p["gap"] > 0 >= q["gap"]:
            hit = p["t"] + (q["t"] - p["t"]) * p["gap"] / (p["gap"]
                                                           - q["gap"])
            break
    return dict(N=n, dt=dt, h=PERI / n, traj=traj, contact_time=hit,
                worst_relative_gradient=worst_rg,
                area_drift_at_end=abs(traj[-1]["area_geom"]
                                      - math.pi * A * B) / (math.pi * A * B))


def pair_run(n, dt, rank_mode, t_end=0.03, stop_gap=0.02):
    parts = [make_ellipse(n, -D_CENTERS / 2), make_ellipse(n, D_CENTERS / 2)]
    v = cat(*parts)
    n1 = parts[0].n_points
    stepper = MMStepper(cfg_for(dt, rank_mode,
                                2 if rank_mode == "oracle" else None))
    iso_v = make_ellipse(n)
    iso_stepper = MMStepper(cfg_for(dt, "oracle", 1))
    frames = []
    for k in range(round(t_end / dt)):
        try:
            r = stepper.step(v)
        except AmbiguousComponentRankError as e:
            frames.append(dict(step=k, hard_stop=str(e)[:200]))
            break
        iso_v = iso_stepper.step(iso_v).varifold
        v = r.varifold
        gap = (v.positions[n1:, 0].min() - v.positions[:n1, 0].max()).item()
        # superposition reference by symmetry: the isolated ellipse
        # translated to +D/2 must match the right pair component; the
        # deviation bundles ALL pair couplings (BIE cross-block +
        # coherence + KDE), which at gap ~ sigma is exactly the quantity
        # of interest
        dx_right = (v.positions[n1:]
                    - (iso_v.positions
                       + torch.tensor([D_CENTERS / 2, 0.0]))).abs().max()
        a1, c1 = poly_slice(v, 0, n1)
        a2, c2 = poly_slice(v, n1, v.n_points)
        frames.append(dict(
            step=k, t=(k + 1) * dt, gap=gap,
            dx_iso_right=dx_right.item(),
            q_min=stepper.fixed_coherence.min().item(),
            C=r.detected_rank, rank_status=r.rank_status,
            rank_action=r.rank_action,
            rank_gap_ratio=r.rank_gap_ratio,
            area_geom=[a1, a2], centroid_x=[c1[0], c2[0]],
            objective_decreased=bool(r.objective_decreased),
        ))
        if gap < stop_gap:
            frames.append(dict(step=k, stopped="gap below stop threshold",
                               gap=gap))
            break
    return dict(N_per_ellipse=n, dt=dt, rank_mode=rank_mode, frames=frames)


def poly_slice(v, i, j):
    X, Y = v.positions[i:j, 0], v.positions[i:j, 1]
    Xn, Yn = X.roll(-1), Y.roll(-1)
    c = X * Yn - Xn * Y
    Aa = 0.5 * c.sum()
    return Aa.item(), [(((X + Xn) * c).sum() / (6 * Aa)).item(),
                       (((Y + Yn) * c).sum() / (6 * Aa)).item()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = {}

    print("=== Step 1: isolated refinement (paper geometry, gap 0.1) ===")
    iso = []
    # MEASURED stability boundary: (N=64, dt=1e-4) blows up at step ~15
    # (max|s| grows x1.6/step from the marginally resolved tips, curvature
    # radius 0.16 ~ 2.2 h). N=64 needs dt <= 2e-5 HERE; scope: the
    # published dt = 1e-5 lies below the instability threshold observed in
    # THIS corrected no-redistribution control -- the published run used a
    # different metric, redistribution and coherence handling, so no
    # direct stability statement about it follows.
    for n, dt in ((64, 2e-5), (128, 1e-4), (256, 1e-4), (128, 2e-5)):
        r = isolated_run(n, dt)
        iso.append(r)
        print(f"  N={n} dt={dt}: h={r['h']:.4f} contact_time="
              f"{r['contact_time']} area_drift={r['area_drift_at_end']:.2e}"
              f" worst_relgrad={r['worst_relative_gradient']:.1e}",
              flush=True)
    out["isolated"] = iso

    print("\n=== Step 2: pair (oracle C=2) vs superposition, N=128 ===")
    p = pair_run(128, 1e-4, "oracle")
    out["pair_oracle_128"] = p
    ok = [f for f in p["frames"] if "gap" in f and "stopped" not in f]
    for f in ok[::5] + ok[-1:]:
        print(f"  t={f['t']:.4f}: gap={f['gap']:+.4f} "
              f"dx_iso={f['dx_iso_right']:.2e} q_min={f['q_min']:.3f} "
              f"areas={[f'{a:.4f}' for a in f['area_geom']]}", flush=True)

    print("\n=== Step 3: label-free rank (state_machine), N=128 and 256 "
          "===")
    for n in (128, 256):
        p = pair_run(n, 1e-4, "state_machine")
        out[f"pair_statemachine_{n}"] = p
        ok = [f for f in p["frames"] if "gap" in f and "stopped" not in f]
        stops = [f for f in p["frames"] if "hard_stop" in f]
        if ok:
            statuses = sorted({f["rank_status"] for f in ok})
            print(f"  N={n}: {len(ok)} evolved frames, final gap "
                  f"{ok[-1]['gap']:+.4f}, statuses={statuses}, "
                  f"hard stops: {len(stops)}"
                  + (f" ({stops[0]['hard_stop'][:80]}...)" if stops
                     else ""), flush=True)
        else:
            print(f"  N={n}: no evolved frames; first frame: "
                  f"{p['frames'][0]}", flush=True)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "two_ellipses_paper_precontact.json"
        path.write_text(json.dumps(dict(
            meta=dict(a=A, b=B, d_centers=D_CENTERS, paper_gap=0.1,
                      metric="spectral/interior, carrier operator, M",
                      coherence="real, sigma=0.1",
                      note=("pre-contact only; post-contact waits for "
                            "0D/0E support validation")),
            **out), indent=2))
        print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
