"""0K-C tempo / regular-regime check.

(a) Checkpoint sweep: per saved endgame-#1 state, max theta_cross and
    max |m_or/m_loop - 1| -- localizes where the exact identity
    regime ends (expected: theta_cross == 0 until the sheets enter
    the delta_KDE range, i.e. 0K is bitwise-inert on the regular
    part of the trajectory).
(b) One-step tempo: FF vs LL production steps at a regular-regime
    checkpoint (identity -> bitwise-equal displacements expected)
    and at a layer checkpoint (1200; gate: velocity ratio within 5%,
    the 0J-D tempo gate).

Usage:
    uv run python scripts/experiments/loopwise_tempo_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import loopwise_mass_ablation as abl  # noqa: E402
from exact_merger_benchmark import OUT, build_cloud, masses_for  # noqa: E402

from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.loopwise_mass import (  # noqa: E402
    resolve_loopwise_oriented_mass_source,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)

SWEEP = (700, 800, 900, 1000, 1100, 1200, 1300, 1325)
TEMPO_STATES = (1000, 1200)


def main():
    torch.set_default_dtype(torch.float64)
    ck = torch.load(OUT / "exact_merger_endgame1_states.pt",
                    weights_only=True)
    v0, _ = build_cloud()
    delta, tau = map(float, compute_recommended_params(v0.positions))
    abl.DELTA, abl.TAU = delta, tau
    print(f"(delta, tau) = ({delta:.6f}, {tau:.6f})")
    rows = []
    for step in SWEEP:
        st0 = ck[step]
        v = OrientedPointCloudVarifold(positions=st0["positions"],
                                       angles=st0["angles"])
        res = resolve_loopwise_oriented_mass_source(
            v.positions, v.normals, delta, tau)
        dev = float((res.m_pre / res.m_loop - 1.0).abs().max())
        tc = float(res.theta_cross.max())
        gapish = None
        r = v.positions.norm(dim=1)
        wall = r < 0.6
        outw = (v.positions * v.normals).sum(1) > 0
        if (wall & outw).any() and (wall & ~outw).any():
            gapish = float(r[wall & ~outw].mean()
                           - r[wall & outw].mean())
        rows.append(dict(step=step, gap=gapish, theta_cross_max=tc,
                         mass_rel_dev_max=dev))
        print(f"step {step}: gap {gapish:.4f}  theta_cross_max "
              f"{tc:.3e}  |m_or/m_loop-1|_max {dev:.3e}", flush=True)

    tempo = {}
    for step in TEMPO_STATES:
        st0 = ck[step]
        v = OrientedPointCloudVarifold(positions=st0["positions"],
                                       angles=st0["angles"])
        t0v = {}
        for est in ("oriented_kde", "loopwise_oriented_kde"):
            m0 = masses_for(est, v0.positions, v0.normals, delta, tau)
            t0v[est] = float(0.5 * (m0 * (v0.positions
                                          * v0.normals).sum(-1)).sum())
        recs = {}
        for est, key in (("oriented_kde", "FF"),
                         ("loopwise_oriented_kde", "LL")):
            st = abl.production_stepper(est, t0v[est])
            res = st.step(v)
            recs[key] = res.displacements.detach()
        d_ff, d_ll = recs["FF"], recs["LL"]
        bitwise = bool(torch.equal(d_ff, d_ll))
        ratio = float(d_ll.norm() / d_ff.norm())
        cos = float((d_ff * d_ll).sum()
                    / (d_ff.norm() * d_ll.norm()))
        tempo[step] = dict(bitwise_equal=bitwise,
                           speed_ratio_ll_over_ff=ratio,
                           direction_cosine=cos)
        print(f"tempo @ {step}: bitwise={bitwise} ratio={ratio:.6f} "
              f"cos={cos:.8f}", flush=True)

    out = Path("results/reports/phase3c0k_tempo.json")
    out.write_text(json.dumps(dict(delta=delta, tau=tau, sweep=rows,
                                   tempo=tempo), indent=1,
                              default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
