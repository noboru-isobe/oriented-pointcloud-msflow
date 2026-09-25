"""Post-hoc per-loop r_l trajectory for an endgame run (reviewer
telemetry): from the 25-step saved states reconstruct, per loop,

    r_l, qbar_self_l, P_l^WB = sum m~ q^WB,

expecting r_disk, r_hole -> 0 (annihilation attenuation) and
r_outer -> 1 (no contamination of the surviving boundary). Exact
recomputation from states -- the in-run stepper enforces the same
quantities through its fail-closed guards.

Usage:
    uv run python scripts/experiments/endgame_loop_telemetry.py \
        [--tag _endgame1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from exact_merger_benchmark import OUT, build_cloud, masses_for  # noqa: E402

from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.perimeter.coherence_perimeter import (  # noqa: E402
    compute_coherence_loopwise,
)
from src.torch.transport.bem_wasserstein import (  # noqa: E402
    compute_coherence,
)
from src.torch.transport.incidence import (  # noqa: E402
    oriented_graph_components,
)

SIGMA = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_endgame1")
    ap.add_argument("--est", default="oriented_kde",
                    choices=("oriented_kde", "loopwise_oriented_kde"))
    args = ap.parse_args()
    ck = torch.load(OUT / f"exact_merger{args.tag}_states.pt",
                    weights_only=True)
    v0, _ = build_cloud()
    delta, tau = map(float, compute_recommended_params(v0.positions))
    rows = []
    for step in sorted(ck.keys()):
        st = ck[step]
        v = OrientedPointCloudVarifold(positions=st["positions"],
                                       angles=st["angles"])
        m = masses_for(args.est, v.positions, v.normals,
                       delta, tau)
        loops = oriented_graph_components(v.positions, v.normals,
                                          4.0 * float(m.median()))
        q_full = compute_coherence(v, m, SIGMA, "wendland_c2")
        q_self, cross = compute_coherence_loopwise(
            v.positions, v.normals, m, SIGMA, "wendland_c2",
            loop_labels=loops)
        rec = dict(step=step, n_loops=int(loops.max()) + 1,
                   cross_max=cross, loops=[])
        r_pts = v.positions.norm(dim=1)
        for lb in loops.unique():
            sel = loops == lb
            A = float((m[sel] * q_full[sel]).sum())
            B = float((m[sel] * q_self[sel]).sum())
            rec["loops"].append(dict(
                label=int(lb), n=int(sel.sum()),
                mean_radius=float((m[sel] * r_pts[sel]).sum()
                                  / m[sel].sum()),
                r=A / B if B > 0 else None,
                qbar_self=B / float(m[sel].sum()),
                p_wb=A))
        rows.append(rec)
        loop_str = " ".join(
            f"[r={l['mean_radius']:.3f}: r_l={l['r']:.3f}]"
            for l in sorted(rec["loops"],
                            key=lambda x: x["mean_radius"]))
        print(f"step {step}: loops={rec['n_loops']} {loop_str}",
              flush=True)
    out = Path(f"results/reports/endgame_loop_telemetry"
               f"{args.tag}.json")
    out.write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
