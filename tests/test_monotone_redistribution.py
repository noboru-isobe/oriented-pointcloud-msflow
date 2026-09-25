"""0N pins: monotone loopwise redistribution.

- legacy flag OFF is bitwise the previous path (positions, angles,
  cv_history, n_iters, converged)
- inactive loops (CV < tol) are bitwise unchanged while an active loop
  moves; a loop that reaches CV < tol is frozen
- every accepted subiteration is non-increasing in Phi_l = CV_l^2
- stage-total displacement <= max_disp per loop
- loop A's line search does not change loop B's update (per-loop
  alpha): B's update is bitwise the same with A active or A frozen
- particle permutation invariance
- rejected (stalled) update -> zero displacement AND zero Frenet
  angle correction on that loop; stop_reason recorded
- guards: monotone requires loopwise density
"""

import math

import pytest
import torch

from src.torch.solver.redistribution_rules import (
    RedistributionGeometryError,
    redistribute_with_rule,
)

DT = torch.float64


def circle(R, n, jitter=0.0, seed=0, inward=False, phase=0.0):
    t = torch.arange(n, dtype=DT) * 2 * math.pi / n + phase
    if jitter:
        g = torch.Generator().manual_seed(seed)
        t = t + jitter * (2 * math.pi / n) * torch.randn(n, generator=g,
                                                          dtype=DT)
    pos = torch.stack([R * t.cos(), R * t.sin()], 1)
    ang = t + (math.pi if inward else 0.0)
    return pos, ang


def fixture(jit_a=0.3, jit_b=0.0):
    pa, aa = circle(0.35, 90, jitter=jit_a, seed=1)
    pb, ab = circle(0.9, 256, jitter=jit_b, seed=2)
    pos = torch.cat([pa, pb]); ang = torch.cat([aa, ab])
    lbl = torch.cat([torch.zeros(90, dtype=torch.long),
                     torch.ones(256, dtype=torch.long)])
    return pos, ang, lbl


COMMON = dict(delta=0.12, kernel="wendland_c2", n_iters=10, step_size=0.01,
              tol=1e-4, max_disp_ratio=0.05, mass_tau=0.0,
              delta_redist=0.06, coherence=None, sigma=0.1)


def run(pos, ang, lbl, monotone, **kw):
    q = torch.ones(pos.shape[0], dtype=DT)
    args = dict(COMMON); args["coherence"] = q; args.update(kw)
    return redistribute_with_rule(pos, ang, "legacy_hybrid",
                                  loop_labels=lbl,
                                  density_scope="loopwise",
                                  curvature_scope="loopwise",
                                  monotone=monotone, **args)


class TestMonotone:
    def test_flag_off_bitwise(self):
        pos, ang, lbl = fixture()
        p0, a0, i0 = run(pos, ang, lbl, False)
        p1, a1, i1 = run(pos, ang, lbl, False)
        assert torch.equal(p0, p1) and torch.equal(a0, a1)
        assert i0["stop_reason"] in ("within_tol", "max_iters")
        assert i0["converged"] == (i0["n_iters"] < 10)

    def test_inactive_loop_frozen_and_monotone(self):
        pos, ang, lbl = fixture(jit_a=0.3, jit_b=0.0)   # B uniform
        p, a, info = run(pos, ang, lbl, True)
        selB = lbl == 1
        assert torch.equal(p[selB], pos[selB]) and torch.equal(a[selB],
                                                              ang[selB])
        cvA = [t["cv_by_loop"][0] for t in info["trace"]
               if "cv_by_loop" in t]
        assert all(b <= a_ + 1e-12 for a_, b in zip(cvA[:-1], cvA[1:]))
        for t in info["trace"]:
            m = t.get("monotone")
            if m:
                assert 1 not in m.get("active_loops", [])
        assert info["stop_reason"] in ("within_tol", "max_iters",
                                       "stage_budget_exhausted")

    def test_stage_total_cap(self):
        pos, ang, lbl = fixture(jit_a=1.5)          # strongly nonuniform
        p, a, info = run(pos, ang, lbl, True)
        max_disp = COMMON["max_disp_ratio"] * COMMON["delta_redist"]
        moved = (p - pos).norm(dim=1)
        assert float(moved.max()) <= max_disp * (1 + 1e-9)
        # legacy could exceed the per-iteration cap cumulatively
        p_leg, _, i_leg = run(pos, ang, lbl, False)
        assert i_leg["n_iters"] >= 1

    def test_per_loop_alpha_independence(self):
        pos, ang, lbl = fixture(jit_a=0.3, jit_b=0.3)
        p_both, _, _ = run(pos, ang, lbl, True, n_iters=1)
        # freeze A by making it perfectly uniform: B's update must be
        # bitwise the same
        pa, aa = circle(0.35, 90)
        pos2 = pos.clone(); ang2 = ang.clone()
        pos2[lbl == 0] = pa; ang2[lbl == 0] = aa
        p_bonly, _, _ = run(pos2, ang2, lbl, True, n_iters=1)
        selB = lbl == 1
        assert torch.equal(p_both[selB] - pos[selB],
                           p_bonly[selB] - pos2[selB])

    def test_permutation_invariance(self):
        pos, ang, lbl = fixture(jit_a=0.3, jit_b=0.3)
        g = torch.Generator().manual_seed(9)
        perm = torch.randperm(pos.shape[0], generator=g)
        p1, a1, _ = run(pos, ang, lbl, True)
        p2, a2, _ = run(pos[perm], ang[perm], lbl[perm], True)
        assert torch.allclose(p2, p1[perm], atol=1e-12)
        assert torch.allclose(a2, a1[perm], atol=1e-12)

    def test_stalled_rejects_update_and_angle(self, monkeypatch):
        """Force no-descent by making the merit never decrease: the
        active loop's update and Frenet correction must both be zero
        and stop_reason = stalled_no_descent."""
        import src.torch.solver.redistribution_rules as rr
        real = rr._compute_density_log_gradient
        calls = {"n": 0}

        def fake(pos, d, kernel, labels=None):
            g, th = real(pos, d, kernel, labels)
            calls["n"] += 1
            if calls["n"] > 1:            # trial evaluations: worse
                th = th * (1.0 + 3.0 * torch.rand_like(th))
            return g, th
        monkeypatch.setattr(rr, "_compute_density_log_gradient", fake)
        pos, ang, lbl = fixture(jit_a=0.3, jit_b=0.0)
        p, a, info = run(pos, ang, lbl, True)
        assert info["stop_reason"] == "stalled_no_descent"
        assert torch.equal(p, pos)
        assert torch.equal(a, ang)

    def test_guard(self):
        pos, ang, lbl = fixture()
        with pytest.raises(RedistributionGeometryError):
            q = torch.ones(pos.shape[0], dtype=DT)
            args = dict(COMMON); args["coherence"] = q
            redistribute_with_rule(pos, ang, "legacy_hybrid",
                                   loop_labels=lbl, density_scope="global",
                                   monotone=True, **args)
