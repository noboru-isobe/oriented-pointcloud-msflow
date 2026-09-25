"""0C-5d-0: static audit of the hysteretic rank state machine.

No evolution here -- the machine is driven over PRESCRIBED geometry
sequences (the 0C-4 families) before it is allowed anywhere near a dynamic
fusion run. Two sequences, two spacings:

  A. CLEAN UNION: separated two disks (g > 0, valid raw loops) sweeping
     through tangency into resolved overlap (g < 0, the actual union
     boundary with interior arcs removed). Required behavior:
       - resolved separation: hold C = 2 on clean evidence;
       - thin-neck interval: weak/ambiguous evidence HOLDS the previous
         rank -- no chatter, no premature C = 1;
       - switch to C = 1 only after k_persist consecutive frames of clean
         C = 1 evidence whose candidate mode passes the phase-constant
         alignment test angle(W^{1/2}1, span U_cand) < theta_const.
     Scope caveats (measured): N changes along this family, so the
     subspace-containment check is INACTIVE here -- the union switch is
     validated by alignment + persistence only (containment is exercised
     by the fixed-N raw family below; the dynamic replacement is the C_u
     row-space continuation of 0C-5d-1). And the OVERLAP-side ambiguous
     band does NOT shrink under h-refinement: the fixed tau_null counts
     the thin-neck quasi-null mode (s_(2) -> lambda_neck(g) > 0 as h -> 0)
     as a second null direction whenever lambda_neck < tau, so only the
     separated-side boundary tightens with h. This is a finite-resolution
     REGULARIZATION, not a convergent transition rule -- do not report an
     O(l_res) interval; the threshold audit (rank_threshold_audit_0c5d)
     owns the replacement design.

  B. RAW TWO-LOOP CROSSING: both full circles kept through overlap (the
     production carrier before any hidden-point removal; N fixed, so the
     subspace-containment check is ACTIVE here). The phase is connected
     (C = 1) but the raw support is invalid. Required behavior: the machine
     must NEVER complete a switch to C = 1 -- candidate modes born of
     near-duplicate collocation must fail the physical-mode validation
     (reject_artificial) or the evidence must stay non-clean (hold).
     Hysteresis prevents chatter; it cannot and must not manufacture a
     valid merged boundary out of an invalid support. A rejection here is
     Gate-B-conditional territory for the caller (0D/0E support
     diagnostics), not something the machine papers over.

Usage
-----
    uv run python scripts/experiments/rank_state_machine_0c5d0.py \
        --out results/rank_state_0c5d0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.torch.transport.bem_wasserstein import rank_evidence
from src.torch.transport.rank_state import (
    RankStateConfig,
    RankStateMachine,
)

sys.path.insert(0, str(Path(__file__).parent))
from bem_rank_transition import (  # noqa: E402
    R1,
    R2,
    raw_two_loops,
    sensor,
    separated_circles,
    union_boundary,
)

TAU = 0.05
GAP_MIN = 30.0
RANK_MAX = 4

# descending signed gap: g > 0 separated, g < 0 resolved overlap depth
UNION_GAPS = [0.4, 0.3, 0.2, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01,
              -0.01, -0.02, -0.025, -0.03, -0.035, -0.04, -0.05, -0.07,
              -0.10, -0.15, -0.20, -0.30, -0.40]
RAW_GAPS = [0.10, 0.05, -0.05, -0.10, -0.15, -0.20, -0.30, -0.40]


def frame(v, w):
    sen = sensor(v, w)
    ev = rank_evidence(sen["S_desc"], RANK_MAX, TAU, GAP_MIN)
    U_cand = sen["U"][:, -ev.c_gap:]
    return sen, ev, U_cand


def run_sequence(kind: str, spacing: float, gaps) -> list:
    machine = RankStateMachine(rank=2, config=RankStateConfig())
    rows = []
    for g in gaps:
        if kind == "union":
            if g >= 0:
                v, w = separated_circles(g, spacing)
            else:
                v, w = union_boundary(R1 + R2 + g, spacing)
        else:
            v, w, _ = raw_two_loops(R1 + R2 + g, spacing)
        sen, ev, U_cand = frame(v, w)
        dec = machine.update(ev, U_cand, sen["sqrt_w"])
        rows.append(dict(
            g=g, N=v.n_points, status=ev.status,
            c_abs=ev.c_abs, c_gap=ev.c_gap,
            gap_ratio=ev.gap_ratio,
            abs_level=ev.abs_levels[max(ev.c_gap - 1, 0)],
            action=dec.action, rank=dec.rank, pending=dec.pending,
            align_const_deg=dec.align_const_deg,
            contain_deg=dec.contain_deg,
        ))
    return rows


def print_rows(rows):
    for r in rows:
        al = ("--" if r["align_const_deg"] is None
              else f"{r['align_const_deg']:6.2f}")
        co = ("--" if r["contain_deg"] is None
              else f"{r['contain_deg']:6.2f}")
        print(f"  g={r['g']:+.2f} N={r['N']:>4} "
              f"ev=({r['c_abs']},{r['c_gap']},{r['status']:>21}) "
              f"gap={r['gap_ratio']:>7.1f} -> {r['action']:>17} "
              f"rank={r['rank']} align={al} contain={co}", flush=True)


def transition_summary(rows):
    """Evidence-level interval (where the SPECTRUM is ambiguous or wrong)
    vs machine-level switch point (which additionally lags by k_persist)."""
    clean2 = [r["g"] for r in rows
              if r["status"] == "clean" and r["c_gap"] == 2]
    accepted1 = [r["g"] for r in rows
                 if r["status"] == "clean" and r["c_gap"] == 1
                 and r["action"] != "reject_artificial"]
    ambiguous = [r["g"] for r in rows if r["status"] != "clean"]
    switch_g = next((r["g"] for r in rows if r["action"] == "switch"), None)
    return dict(
        evidence_clean2_until=min(clean2, default=None),
        evidence_clean1_from=max(accepted1, default=None),
        ambiguous_gs=ambiguous,
        machine_switch_g=switch_g,
        switched=any(r["action"] == "switch" for r in rows),
        rejections=sum(r["action"] == "reject_artificial" for r in rows),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = {}

    for spacing in (0.03, 0.015):
        print(f"=== A. clean union sequence, spacing = {spacing} "
              f"(l_res ~ {spacing}) ===")
        rows = run_sequence("union", spacing, UNION_GAPS)
        print_rows(rows)
        s = transition_summary(rows)
        print(f"  summary: {s}\n")
        out[f"union_h{spacing}"] = dict(rows=rows, summary=s)

    for spacing in (0.03, 0.015):
        print(f"=== B. raw two-loop crossing, spacing = {spacing} ===")
        rows = run_sequence("raw", spacing, RAW_GAPS)
        print_rows(rows)
        s = transition_summary(rows)
        print(f"  summary: {s}\n")
        out[f"raw_h{spacing}"] = dict(rows=rows, summary=s)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "rank_state_machine_0c5d0.json"
        path.write_text(json.dumps(dict(
            meta=dict(tau=TAU, gap_min=GAP_MIN, rank_max=RANK_MAX,
                      k_persist=RankStateConfig().k_persist,
                      theta_const_deg=RankStateConfig().theta_const_deg,
                      theta_contain_deg=RankStateConfig().theta_contain_deg,
                      R1=R1, R2=R2),
            **out), indent=2))
        print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
