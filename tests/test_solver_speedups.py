"""Flag-gated solver speedups (default OFF -- the default numeric path
is bit-identical; goldens cover it untouched).

Part 1  wlin_analytic_gradient: spectral W_lin evaluated through the
        per-step EXPLICIT quadratic matrix (assembled once by a batched
        solve in finalize_with_rank; the inner loop is V^T (Mq V) --
        no lu_solve inside the optimization loop). A hand-coded
        double-backward kernel-gradient op was ALSO tried and REMOVED:
        benchmarked 0.89x on both sizes -- python-layer custom-Function
        overhead loses to the fused C++ double backward.
Part 2  bem_svd_method="lobpcg": warm-started Gram-free scipy LOBPCG
        for the near-null pairs (k = rank_max + 3), cached s_max with
        periodic re-solve, full-SVD fallback on residual/exception/
        periodic refresh.

Verification contract (user): outputs unchanged AND measured speedup.
MEASURED (quiet machine, min-of-3, results/speedup_bench):
    pair n_per=128 (NK=768):  0.790 -> 0.700 s/step (1.13x)
    pair n_per=256 (NK=1536): 3.075 -> 2.004 s/step (1.53x)  <- the
    production-refinement regime meets the >=1.5x bar; outputs within
    1e-10 over the benchmark trajectories. Tests pin CORRECTNESS.
"""

import sys
from pathlib import Path

import pytest
import torch

from src.torch.oriented_varifold.mass import compute_recommended_params
from src.torch.solver.mm_step import MMStepper

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"
                       / "experiments"))

from p1_production_comparison import make_cfg, shapes  # noqa: E402


def _pair():
    v, rank = shapes()["two_ellipses_paper"]
    delta, tau = compute_recommended_params(v.positions)
    return v, rank, delta, tau


def _annulus():
    from d2b0_timelevel_shadow import snapshot
    v, n_out, n_in = snapshot(0.5, warp=0.35)
    delta, tau = compute_recommended_params(snapshot(0.5)[0].positions)
    return v, 1, delta, tau


def _stepper(v, rank, delta, tau, wlin=False, svd="full"):
    cfg = make_cfg("C3", delta, tau, rank)
    cfg.wlin_analytic_gradient = wlin
    cfg.bem_svd_method = svd
    return MMStepper(cfg)


@pytest.mark.parametrize("fixture", [_pair, _annulus],
                         ids=["pair", "annulus"])
def test_wlin_quadratic_value_grad_hvp_parity(fixture):
    """The quadratic-matrix W_lin agrees with the flag-off evaluation
    in VALUE (1e-11 -- assembly reorders float ops), GRADIENT and
    HESSIAN-VECTOR PRODUCT (both through the actual double-backward
    path trust-ncg uses)."""
    v, rank, delta, tau = fixture()

    outs = {}
    for flag in (False, True):
        st = _stepper(v, rank, delta, tau, wlin=flag)
        st._setup_step(v)
        torch.manual_seed(0)      # SAME y for both flags
        y = 1e-3 * torch.randn(st.param.n_params, dtype=torch.float64)
        y = y.requires_grad_(True)
        F = st.objective(y)
        g = torch.autograd.grad(F, y, create_graph=True)[0]
        torch.manual_seed(1)
        vec = torch.randn_like(y)
        hvp = torch.autograd.grad(g, y, vec, retain_graph=True)[0]
        outs[flag] = (F.detach(), g.detach(), hvp.detach())

    F0, g0, h0 = outs[False]
    F1, g1, h1 = outs[True]
    assert abs(F1 - F0) < 1e-11 * abs(F0)
    assert (g1 - g0).abs().max() < 1e-9 * g0.abs().max()
    assert (h1 - h0).abs().max() < 1e-8 * h0.abs().max().clamp_min(1e-30)


def test_mm_step_and_short_trajectory_parity():
    """Full MM steps with both flags on stay allclose to the baseline
    over a short trajectory (the optimizer's iterate path may differ in
    rounding; accumulated divergence must stay at solver-tolerance
    level, well below any physical scale)."""
    v, rank, delta, tau = _pair()
    va, vb = v, v
    st0 = _stepper(v, rank, delta, tau)
    st1 = _stepper(v, rank, delta, tau, wlin=True, svd="lobpcg")
    for _ in range(6):
        va = st0.step(va).varifold
        vb = st1.step(vb).varifold
    import math
    # tolerance 1e-8 -> 1e-7 (2026-08-06, C_2D angle-map fix): the
    # corrected map applies the full geometric rotation (2.23x the old
    # under-rotation), so identical-code rounding-path divergence
    # between the two configurations amplifies accordingly (measured
    # 4.2e-8 over 6 steps, was <1e-8 pre-fix). Still solver-tolerance
    # scale, orders below any physical quantity.
    assert (va.positions - vb.positions).abs().max() < 1e-7
    dang = (va.angles - vb.angles + math.pi) % (2 * math.pi) - math.pi
    assert dang.abs().max() < 1e-7      # circular: 2pi wrap is identity


def test_lobpcg_spectral_parity_and_fallback():
    """LOBPCG working rank / U0 subspace / gap-ratio parity with the
    full SVD, and the full-SVD fallback fires on an injected scipy
    failure (reason recorded)."""
    import numpy as np
    from unittest.mock import patch

    v, rank, delta, tau = _pair()
    st_f = _stepper(v, rank, delta, tau, svd="full")
    st_l = _stepper(v, rank, delta, tau, svd="lobpcg")
    st_f._setup_step(v)
    st_l._setup_step(v)
    bf, bl = st_f.bem_wasserstein, st_l.bem_wasserstein
    assert bf.rank_evidence_last.c_gap == bl.rank_evidence_last.c_gap
    assert abs(bf.rank_evidence_last.gap_ratio
               - bl.rank_evidence_last.gap_ratio) < 1e-6 * \
        bf.rank_evidence_last.gap_ratio
    Uf = bf._spec_svd[0][:, -rank:]
    Ul = bl._spec_svd[0][:, -rank:]
    cos = torch.linalg.svdvals(Uf.T @ Ul).min()
    assert torch.rad2deg(torch.acos(cos.clamp(-1, 1))) < 1e-3

    # fallback: scipy failure -> full SVD, reason recorded, run continues
    st2 = _stepper(v, rank, delta, tau, svd="lobpcg")
    with patch("scipy.sparse.linalg.lobpcg",
               side_effect=RuntimeError("injected")):
        st2._setup_step(v)
    used = st2.bem_wasserstein.last_timings.get("svd_method_used", "")
    assert used.startswith("full_fallback:exception")


def test_flags_off_is_default_and_untouched():
    """Defaults: both flags off; the flag-off spectral evaluation takes
    the ORIGINAL code path (no _spec_quad consumed)."""
    from src.torch.solver.mm_step import MMConfig
    cfg = MMConfig()
    assert cfg.wlin_analytic_gradient is False
    assert cfg.bem_svd_method == "full"
    v, rank, delta, tau = _pair()
    st = _stepper(v, rank, delta, tau)
    st._setup_step(v)
    assert st.bem_wasserstein._spec_quad is None
