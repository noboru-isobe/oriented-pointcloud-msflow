"""Sizing smoke for the RESOLVED two-ellipses run (n_per=512).

Prints delta vs initial gap (the resolved regime requires delta < gap)
and measures s/step for C3 with the speedup flags on, to size the
production merger run.  Findings recorded in docs/p12_pair_findings.md
(knowledge item 4): delta scales as 1/N; n_per=128 is inside the
carrier window from t=0, 256 is marginal, 512 is resolved.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

from src.torch.oriented_varifold.mass import compute_recommended_params  # noqa: E402
from src.torch.shapes.generator import generate_oriented_two_ellipses  # noqa: E402
from src.torch.solver.mm_step import MMStepper  # noqa: E402
from p1_production_comparison import make_cfg  # noqa: E402

GAP0 = 0.1  # centers 0.9 apart, semi-axes 0.4 -> initial gap

for n_per in (256, 512):
    v = generate_oriented_two_ellipses(
        n_per, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
        a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu",
        dtype=torch.float64)
    delta, tau = compute_recommended_params(v.positions)
    print(f"n_per={n_per}: delta={float(delta):.5f} gap0={GAP0} "
          f"resolved={'YES' if float(delta) < GAP0 else 'NO'}", flush=True)
    cfg = make_cfg("C3", delta, tau, 2)
    cfg.wlin_analytic_gradient = True
    cfg.bem_svd_method = "lobpcg"
    st = MMStepper(cfg)
    st.step(v)  # warmup
    t0 = time.perf_counter()
    vv = v
    for _ in range(3):
        vv = st.step(vv).varifold
    per = (time.perf_counter() - t0) / 3
    print(f"n_per={n_per}: {per:.2f}s/step (C3 + quad + lobpcg)", flush=True)
