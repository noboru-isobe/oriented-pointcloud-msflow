"""Isolated-relaxation control runs under the corrected (C_2D-fixed)
angle map: SUPERPOSITION contact prediction, NOT an actual pair
trajectory.

What this computes (review correction, 2026-08-07): a SINGLE isolated
ellipse (oracle C=1, no redistribution, no deletion, no hidden
boundary) is evolved and the contact time is INTERPOLATED from
g_pred(t) = D_CENTERS - 2 x_max(t) -- the instant at which two
mirror-image copies placed at the paper's center distance WOULD touch
if the pair dynamics were the isolated superposition. The codebase
itself measures that at the paper gap (g0 ~ sigma) the actual pair is
NOT the isolated superposition (q_min collapses, superposition defect
> 5e-3 -- pinned in test_pair_at_paper_gap_shows_coherence_coupling),
so this artifact must never be cited as an actual pair contact time,
and it demonstrates NO topological change.

Output schema (results/contact_postfix/):
    superposition_contact_prediction_dt<dt>.json
        predicted_superposition_contact_time  -- linear interpolation
            of the g_pred sign change between recorded samples
        area_drift_at_end, worst_relative_gradient, traj samples
    config.json  -- full provenance: MMConfig fields, git SHA,
        package versions, dtype/device, command line, schema notes

Usage:
    uv run python scripts/experiments/contact_postfix.py --dt 2e-5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import scipy  # noqa: E402
import torch  # noqa: E402

from two_ellipses_paper_precontact import (  # noqa: E402
    cfg_for,
    isolated_run,
)

OUT = ROOT / "results" / "contact_postfix"


def provenance(dt: float, argv: list[str]) -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    cfg = cfg_for(dt, "oracle", 1)
    return dict(
        git_sha=sha,
        command=" ".join(argv),
        torch_version=torch.__version__,
        scipy_version=scipy.__version__,
        python_version=sys.version.split()[0],
        dtype="float64",
        device="cpu",
        seed="deterministic (no RNG in the isolated run)",
        contact_interpolation=(
            "linear in t between the last sample with g_pred > 0 and "
            "the first with g_pred <= 0; g_pred = D_CENTERS - 2*x_max"),
        mm_config={k: (v if isinstance(v, (int, float, str, bool,
                                           type(None))) else str(v))
                   for k, v in dataclasses.asdict(cfg).items()},
        semantics=(
            "SUPERPOSITION contact prediction from an isolated "
            "single-ellipse relaxation; NOT an actual pair trajectory; "
            "shows no topological change"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--t-end", type=float, default=0.03)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    t0 = time.time()
    r = isolated_run(args.n, args.dt, t_end=args.t_end)
    r["wall_s"] = time.time() - t0
    # review-corrected field name; keep the raw value once, under the
    # honest key only
    r["predicted_superposition_contact_time"] = r.pop("contact_time")
    p = OUT / f"superposition_contact_prediction_dt{args.dt:g}.json"
    p.write_text(json.dumps(r, indent=1))

    cfg_path = OUT / "config.json"
    cfg = (json.loads(cfg_path.read_text())
           if cfg_path.exists() else {"runs": {}})
    cfg["runs"][f"dt{args.dt:g}"] = provenance(args.dt, sys.argv)
    cfg_path.write_text(json.dumps(cfg, indent=1, default=str))

    print(f"dt={args.dt:.0e}: predicted superposition contact "
          f"{r['predicted_superposition_contact_time']} "
          f"area_drift={r['area_drift_at_end']:.2e} "
          f"wall={r['wall_s']/60:.1f}min -> {p.name}")


if __name__ == "__main__":
    main()
