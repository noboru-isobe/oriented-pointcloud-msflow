"""0C-5d-2a: the adaptive noise-floor rank rule on valid prescribed
boundaries. No solver integration yet, no raw post-contact evolution.

Replaces the fixed absolute threshold (rejected: anti-convergent against
the thin-neck quasi-null mode) by the scale-invariant floor rule of
AdaptiveRankMachine: eta estimated from accepted clean frames, candidate
rank r from max-gap, floor-consistency kappa_null / kappa_sep, physical
validation (phase-constant alignment + containment), persistence in
cumulative ACTUAL geometric displacement |Delta g| (review 4.2).

Three checks, predictions stated before measurement:

  A. UNION SWEEP at h = 0.03 / 0.015 / 0.0075: switch location should
     move TOWARD geometric contact with refinement without any absolute
     tau (fixed-tau machine switched at g = -0.10 at every h). MEASURED
     with |Delta g| persistence: -0.07 / -0.02 / -0.01, monotone in h --
     roughly onset minus the 2 l_res persistence scale, both of which
     vanish with h.

  B. RAW CROSSING: unchanged refusal -- the artificial C = 1 modes pass
     the floor tests (they are genuinely near-null) but fail the
     phase-constant alignment; the machine must reject, never switch.

  C. RESCALING INVARIANCE x10 (geometry, spacing, gaps all scaled): the
     rule consumes singular-value RATIOS only, so every decision must be
     identical. Precision (review 4.1): the FIXED tau = 0.05 is also
     dimensionless and passes this check -- its defect is h-refinement
     anti-convergence, not scaling. What this check rules out is any
     dimensional tau ~ h formula.

  D. GRID INVARIANCE: the same physical sweep sampled on two different
     gap grids. MEASURED at h = 0.015: coarse grid switches at -0.02,
     dense grid at -0.04 -- agreement within the 2 l_res = 0.03
     persistence scale (the dense grid sees extra weak frames near onset
     that freeze the floor longer), NOT exact invariance. The |Delta g|
     persistence bounds the grid dependence by the persistence scale;
     a frame-count rule had no such bound.

Usage
-----
    uv run python scripts/experiments/adaptive_rank_rule_0c5d2a.py \
        --out results/adaptive_rank_0c5d2a
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.torch.transport.rank_state import (
    AdaptiveRankConfig,
    AdaptiveRankMachine,
)

sys.path.insert(0, str(Path(__file__).parent))
import bem_rank_transition as brt  # noqa: E402
from bem_rank_transition import sensor  # noqa: E402

UNION_GAPS = [0.4, 0.3, 0.2, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01,
              -0.01, -0.02, -0.025, -0.03, -0.035, -0.04, -0.05, -0.07,
              -0.10, -0.15, -0.20, -0.30, -0.40]
UNION_GAPS_FINE = [0.05, 0.02, 0.01, -0.01, -0.02, -0.025, -0.03, -0.04,
                   -0.05, -0.07, -0.10]
RAW_GAPS = [0.10, 0.05, -0.05, -0.10, -0.15, -0.20, -0.30, -0.40]


def run_sequence(kind: str, spacing: float, gaps, scale: float = 1.0):
    """Drive the adaptive machine over a prescribed geometry sequence.

    Persistence integrates the PRESCRIBED-FAMILY PARAMETER displacement
    |Delta g| between frames (review 4.2 / 5.1: a fixed
    one-l_res-per-frame convention makes the switch inherit the sampling
    grid; and |Delta g| is still not the true boundary displacement --
    the union family regenerates arcs, so production evolutions must use
    max|dx| or a transported Hausdorff-type distance). `scale` multiplies
    radii, spacing and gaps (check C); the module-level R1/R2 of the 0C-4
    family are patched for the scaled run and restored afterwards."""
    R1_0, R2_0 = brt.R1, brt.R2
    brt.R1, brt.R2 = R1_0 * scale, R2_0 * scale
    try:
        machine = AdaptiveRankMachine(config=AdaptiveRankConfig())
        rows = []
        prev_g = None
        for g in gaps:
            gs, sp = g * scale, spacing * scale
            if kind == "union":
                if g >= 0:
                    v, w = brt.separated_circles(gs, sp)
                else:
                    v, w = brt.union_boundary(brt.R1 + brt.R2 + gs, sp)
            else:
                v, w, _ = brt.raw_two_loops(brt.R1 + brt.R2 + gs, sp)
            disp = (sp if prev_g is None
                    else abs(gs - prev_g))
            prev_g = gs
            sen = sensor(v, w)
            dec = machine.update(sen["S_desc"], sen["U"], sen["sqrt_w"],
                                 step_displacement=disp, l_res=sp)
            rows.append(dict(
                g=g, N=v.n_points, action=dec.action, rank=dec.rank,
                floor=machine.floor, pending=dec.pending,
                align_const_deg=dec.align_const_deg,
                contain_deg=dec.contain_deg,
            ))
        return rows
    finally:
        brt.R1, brt.R2 = R1_0, R2_0


def summarize(rows):
    return dict(
        switch_g=next((r["g"] for r in rows if r["action"] == "switch"),
                      None),
        rejections=sum(r["action"] == "reject_artificial" for r in rows),
        budget_stops=sum(r["action"] == "ambiguity_budget_exceeded"
                         for r in rows),
        final_rank=rows[-1]["rank"],
        actions={a: sum(r["action"] == a for r in rows)
                 for a in {r["action"] for r in rows}},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = {}

    print("=== A. union sweep, adaptive floor rule ===")
    for spacing, gaps in ((0.03, UNION_GAPS), (0.015, UNION_GAPS),
                          (0.0075, UNION_GAPS_FINE)):
        rows = run_sequence("union", spacing, gaps)
        s = summarize(rows)
        out[f"union_h{spacing}"] = dict(rows=rows, summary=s)
        print(f"  h={spacing}: switch at g={s['switch_g']}, "
              f"final rank={s['final_rank']}, actions={s['actions']}",
              flush=True)

    print("\n=== B. raw crossing, adaptive floor rule ===")
    for spacing in (0.03, 0.015):
        rows = run_sequence("raw", spacing, RAW_GAPS)
        s = summarize(rows)
        out[f"raw_h{spacing}"] = dict(rows=rows, summary=s)
        rej = [r for r in rows if r["action"] == "reject_artificial"]
        aligns = [f"{r['align_const_deg']:.1f}" for r in rej]
        print(f"  h={spacing}: switch={s['switch_g']}, "
              f"rejections={s['rejections']} (aligns deg: {aligns}), "
              f"final rank={s['final_rank']}", flush=True)

    print("\n=== C. rescaling invariance (x10, h = 0.03 union) ===")
    base = run_sequence("union", 0.03, UNION_GAPS, scale=1.0)
    scaled = run_sequence("union", 0.03, UNION_GAPS, scale=10.0)
    mismatches = [(a["g"], a["action"], b["action"])
                  for a, b in zip(base, scaled)
                  if a["action"] != b["action"] or a["rank"] != b["rank"]]
    print(f"  decision mismatches: {len(mismatches)} "
          f"{mismatches if mismatches else '(all identical)'}")
    out["rescaling"] = dict(n_mismatches=len(mismatches),
                            mismatches=mismatches)

    print("\n=== D. gap-grid invariance (h = 0.015 union, two grids) ===")
    dense = [0.4, 0.3, 0.2, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03, 0.02,
             0.015, 0.01, 0.005, -0.005, -0.01, -0.015, -0.02, -0.025,
             -0.03, -0.035, -0.04, -0.045, -0.05, -0.06, -0.07, -0.085,
             -0.10, -0.15, -0.20, -0.30, -0.40]
    for name, gaps in (("coarse", UNION_GAPS), ("dense", dense)):
        rows = run_sequence("union", 0.015, gaps)
        s = summarize(rows)
        out[f"grid_{name}_h0.015"] = dict(rows=rows, summary=s)
        print(f"  {name} grid ({len(gaps)} frames): switch at "
              f"g={s['switch_g']}, actions={s['actions']}", flush=True)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "adaptive_rank_rule_0c5d2a.json"
        path.write_text(json.dumps(dict(
            meta=dict(config=vars(AdaptiveRankConfig()),
                      persistence="cumulative prescribed-family parameter displacement |Delta g| (production evolutions must use actual boundary displacement max|dx| or a transported Hausdorff-type distance instead)"),
            **out), indent=2))
        print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
