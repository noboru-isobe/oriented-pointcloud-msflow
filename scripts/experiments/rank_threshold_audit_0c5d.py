"""0C-5d threshold audit: fixed absolute tau vs h on the clean-union family.

Why (measured at 0C-5d-0/1): the OVERLAP-side ambiguity of the union sweep
does not shrink under h-refinement -- the fixed tau_null = 0.05 keeps
counting the thin-neck quasi-null mode (s_(2) -> lambda_neck(g) > 0 as
h -> 0) as a second null direction; and in the pre-contact dynamic run the
small ellipse's null level sat a hair under the fixed confidence threshold
(solo gap ratio 29.6 vs gap_min 30) and drifted across it during evolution.
Both point at the same design question this audit quantifies: how do the
TRUE null levels and the quasi-null neck levels scale with h, and which
tau separates them at each resolution?

Protocol: spectrum computed ONCE per (h, g) on the clean-union family;
evidence evaluated for every tau in TAUS (gap_min fixed at 30). The
phase-constant alignment of the max-gap C = 1 candidate is recorded on
EVERY frame including disagreements -- if it stays at a few degrees on the
frames the absolute count vetoes, the disagreement is a threshold
artifact, not a physical-mode failure.

Output guides the 0C-5d-2 replacement rule. Reading discipline for the
results (review-fixed):
- "no ambiguous frame on the SAMPLED gap grid" is the correct phrasing --
  g = 0 and finer gaps were not sampled, so an empty ambiguous set does
  not show the continuous transition interval is empty;
- tau ~ 1.5-2 h is NOT a production formula: the spectrum of K* is
  dimensionless in the geometry scale, so a threshold proportional to the
  ABSOLUTE spacing h is not invariant under rescaling, and it was tested
  on one circular union family only. The supported design is a null floor
  estimated from the previously ACCEPTED clean regime (or normalised by
  the current global-constant mode s_(1)), validated under rescaling and
  on a noncircular thin-neck family;
- the true-null scaling is "consistent with O(h)" on three resolutions,
  not established;
- this absolute-floor problem is SEPARATE from the dynamic pre-contact
  stop, which was a gap_ratio/rank_gap_min confidence event: tau(h) alone
  would not have prevented it. That one is handled by the state machine
  holding same-rank weak evidence (0C-5d-1.1).
This audit does not itself change any production or solver default.

Usage
-----
    uv run python scripts/experiments/rank_threshold_audit_0c5d.py \
        --out results/rank_threshold_0c5d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from src.torch.transport.bem_wasserstein import rank_evidence
from src.torch.transport.rank_state import angle_vec_in_span_deg

sys.path.insert(0, str(Path(__file__).parent))
from bem_rank_transition import (  # noqa: E402
    R1, R2, sensor, separated_circles, union_boundary,
)

GAP_MIN = 30.0
RANK_MAX = 4
TAUS = (0.05, 0.025, 0.0125)
GAPS_FULL = [0.4, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01,
             -0.01, -0.02, -0.03, -0.04, -0.05, -0.07, -0.10, -0.20, -0.40]
GAPS_FINE = [0.05, 0.02, 0.01, -0.01, -0.02, -0.03, -0.04, -0.05,
             -0.07, -0.10]           # h = 0.0075 is expensive: core band only
SPACINGS = (0.03, 0.015, 0.0075)


def frame(g, spacing):
    if g >= 0:
        v, w = separated_circles(g, spacing)
    else:
        v, w = union_boundary(R1 + R2 + g, spacing)
    sen = sensor(v, w)
    s_asc = sen["sigma"]
    rel = (s_asc / s_asc[-1]).tolist()
    U1 = sen["U"][:, -1:]
    align1 = angle_vec_in_span_deg(sen["sqrt_w"], U1)
    row = dict(g=g, N=v.n_points,
               s1_rel=rel[0], s2_rel=rel[1], s3_rel=rel[2],
               align_c1_deg=align1)
    for tau in TAUS:
        ev = rank_evidence(sen["S_desc"], RANK_MAX, tau, GAP_MIN)
        row[f"tau{tau}"] = dict(status=ev.status, c_abs=ev.c_abs,
                                c_gap=ev.c_gap, gap_ratio=ev.gap_ratio)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = {}

    for spacing in SPACINGS:
        gaps = GAPS_FINE if spacing == 0.0075 else GAPS_FULL
        print(f"=== union family, h = {spacing} ===", flush=True)
        hdr = (f"{'g':>6} {'N':>5} {'s1/smax':>9} {'s2/smax':>9} "
               f"{'align(C1)':>9} " +
               " ".join(f"tau={t:<6}" for t in TAUS))
        print(hdr)
        rows = []
        for g in gaps:
            row = frame(g, spacing)
            rows.append(row)
            stats = " ".join(
                f"{row[f'tau{t}']['status'][:9]:>9}" for t in TAUS)
            print(f"{g:>+6.2f} {row['N']:>5} {row['s1_rel']:>9.2e} "
                  f"{row['s2_rel']:>9.2e} {row['align_c1_deg']:>9.2f} "
                  f"{stats}", flush=True)
        out[f"h{spacing}"] = rows

        for tau in TAUS:
            key = f"tau{tau}"
            clean1 = [r["g"] for r in rows
                      if r[key]["status"] == "clean"
                      and r[key]["c_gap"] == 1]
            ambiguous = [r["g"] for r in rows
                         if r[key]["status"] not in ("clean",)]
            print(f"  tau={tau}: first clean C=1 at g="
                  f"{max(clean1) if clean1 else None}, "
                  f"ambiguous gs={ambiguous}")
        print()

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / "rank_threshold_audit_0c5d.json"
        path.write_text(json.dumps(dict(
            meta=dict(gap_min=GAP_MIN, taus=TAUS, spacings=SPACINGS,
                      R1=R1, R2=R2), **out), indent=2))
        print(f"raw results -> {path}")


if __name__ == "__main__":
    main()
