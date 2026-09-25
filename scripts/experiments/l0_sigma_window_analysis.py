"""L0 supplementary analysis (labeled SIGMA-SCALE, not the
pre-registered h-scale window, which had not yet formed when the
pre-registered K=100 stop fired): spatial decomposition of
q_full / q_self / q_WB at each run's stop state.

The central L0 question (reviewer 2026-08-20): does the loopwise
q^WB = r_l q^self localize contact attenuation? Measured answer at
the sigma scale: q_full dips locally at the facing arc while
q_WB applies a uniform loop-wide factor -- eta_loc ~ 0.2,
eta_spill ~ 24x the true outside attenuation.

Usage:
    uv run python scripts/experiments/l0_sigma_window_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from two_ellipses_benchmark import OUT, SIGMA, resolve_m  # noqa: E402
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)


def analyze(tag):
    from src.torch.perimeter.coherence_perimeter import (
        compute_coherence_loopwise,
    )
    from src.torch.transport.bem_wasserstein import compute_coherence
    p = OUT / f"two_ellipses{tag}_states.pt"
    if not p.exists():
        return None
    ck = torch.load(p, weights_only=True)
    step = max(ck)
    st = ck[step]
    pos, ang = st["positions"], st["angles"]
    n = pos.shape[0] // 2
    m1 = torch.zeros(pos.shape[0], dtype=torch.bool)
    m1[:n] = True
    v = OrientedPointCloudVarifold(positions=pos, angles=ang)
    delta, tau = map(float, compute_recommended_params(pos))
    m = resolve_m(pos, v.normals, delta, tau)
    q_full = compute_coherence(v, m, SIGMA, "wendland_c2")
    q_self, cross_max = compute_coherence_loopwise(
        pos, v.normals, m, SIGMA, "wendland_c2", (~m1).long())
    out = dict(tag=tag, step=int(step), sigma=SIGMA,
               cross_max=cross_max)
    for k, (sel, name) in enumerate(((m1, "loop1"), (~m1, "loop2"))):
        r_l = float((m[sel] * q_full[sel]).sum()
                    / (m[sel] * q_self[sel]).sum())
        q_wb = r_l * q_self
        other = ~sel if k == 0 else m1
        g = torch.cdist(pos[sel], pos[other]).min(dim=1).values
        W = g <= SIGMA
        rec = dict(r_l=r_l, q_full_min=float(q_full[sel].min()),
                   q_self_min=float(q_self[sel].min()),
                   n_W_sigma=int(W.sum()), n=int(sel.sum()),
                   g_min=float(g.min()))
        if int(W.sum()) >= 2:
            Wc = ~W

            def mb(q, s):
                return float((m[sel][s] * q[sel][s]).sum()
                             / m[sel][s].sum())
            qfW, qfC = mb(q_full, W), mb(q_full, Wc)
            qwW, qwC = mb(q_wb, W), mb(q_wb, Wc)
            rec.update(qbar_full_W=qfW, qbar_full_Wc=qfC,
                       qbar_wb_W=qwW, qbar_wb_Wc=qwC,
                       eta_loc_sigma=(1 - qwW) / (1 - qfW)
                       if qfW < 1 else None,
                       eta_spill_sigma=1 - qwC,
                       true_outside_attenuation=1 - qfC)
        out[name] = rec
    return out


def main():
    torch.set_default_dtype(torch.float64)
    rep = {}
    for tag in ("_l0_768", "_l0_1024", "_l0_768rot"):
        r = analyze(tag)
        if r is None:
            print(f"-- {tag}: states missing, skipped")
            continue
        rep[tag] = r
        l1 = r["loop1"]
        print(f"== {tag} @ {r['step']}: r_l {l1['r_l']:.4f} q_full_min "
              f"{l1['q_full_min']:.4f} eta_loc(sigma) "
              f"{l1.get('eta_loc_sigma')} eta_spill "
              f"{l1.get('eta_spill_sigma')} (true outside "
              f"{l1.get('true_outside_attenuation')})")
    Path("results/reports/l0_sigma_window_analysis.json").write_text(
        json.dumps(rep, indent=1, default=str))
    print("wrote results/reports/l0_sigma_window_analysis.json")


if __name__ == "__main__":
    main()
