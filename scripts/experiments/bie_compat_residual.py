"""Relative compatibility residual |u^T D^{1/2} V| / |D^{1/2} V| of the
bordered boundary integral system along the production runs: one MM
step is recomputed from every stored state with the production
configuration and the worst residual over the optimizer's evaluations
is recorded (the run stops above 0.1).

Usage: uv run python scripts/experiments/bie_compat_residual.py
Writes results/reports/bie_compat_residual.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from bie_state_diagnostics import annulus_cfg, ellipse_cfg, flower_cfg  # noqa: E402
from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa: E402
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.solver.mm_step import MMStepper  # noqa: E402

torch.set_default_dtype(torch.float64)


def scan(name, states_path, json_path=None):
    ck = torch.load(states_path, weights_only=True)
    keys = sorted(k for k in ck if isinstance(k, int))
    s0 = ck[keys[0]]
    delta, tau = map(float, compute_recommended_params(s0["positions"]))
    meta = json.load(open(json_path))["meta"] if json_path else {}
    act = meta.get("activated_at")
    rows = []
    for k in keys:
        st = ck[k]
        if name == "annulus":
            cfg = annulus_cfg(delta, tau)
        elif name == "two_ellipses":
            q_mode = ("contact_complex_renormalized"
                      if act is not None and k >= act
                      and st["positions"].shape[0] == s0["positions"].shape[0]
                      else "self_renormalized")
            cfg = ellipse_cfg(delta, tau, q_mode)
        else:
            cfg = flower_cfg(delta, tau)
        stp = MMStepper(cfg)
        v = OrientedPointCloudVarifold(positions=st["positions"].clone(),
                                       angles=st["angles"].clone())
        try:
            res = stp.step(v)
        except Exception as ex:  # noqa: BLE001
            rows.append(dict(step=k, error=f"{type(ex).__name__}: {ex}"[:200]))
            continue
        rows.append(dict(step=k, r_comp_max=stp.bie_wasserstein._max_r_comp,
                         r_comp_last=stp.bie_wasserstein._last_r_comp,
                         n_iter=res.n_iter))
    ok = [r for r in rows if "error" not in r]
    summ = dict(n_states=len(rows), n_errors=len(rows) - len(ok),
                r_comp_max=max(r["r_comp_max"] for r in ok),
                r_comp_max_step=max(ok, key=lambda r: r["r_comp_max"])["step"])
    print(name, summ, flush=True)
    return dict(summary=summ, rows=rows)


def main():
    out = {
        "flower": scan("flower", ROOT / "results/flower/flower_production_bie_states.pt"),
        "annulus": scan("annulus", ROOT / "results/exact_merger/exact_merger_bie_states.pt",
                        ROOT / "results/exact_merger/exact_merger_bie.json"),
        "two_ellipses": scan("two_ellipses", ROOT / "results/two_ellipses/two_ellipses_bie_states.pt",
                             ROOT / "results/two_ellipses/two_ellipses_bie.json"),
    }
    (ROOT / "results/reports/bie_compat_residual.json").write_text(
        json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
