"""L0J-0 pins (reviewer 2026-08-21): sigma-scale energy contact
complex and the contact-complex well-balanced coherence q^CC.

0a graph/order algebra:
  distant-loops identity, full-contact reduction to q^WB, sidewise
  source-energy conservation, far-field exact identity, two-window
  independence, permutation (cyclic roll) + rigid rotation invariance,
  r/denominator/side-size guards, three-loop / crossing / self-contact
  fail-closed, source-frozen gradient check, default paths untouched.
0b support-boundary continuity:
  edge birth/death sweep (the hard-membership r jump is REAL and
  measured here; its acceptance against the 0J refresh envelope is
  the L0J-1 audit's job), merge -> three-loop fail-closed.
"""

import math

import pytest
import torch

from src.torch.perimeter.contact_complex import (
    SigmaContactComplexError,
    _monotone,
    certify_loop_orders,
    contact_complex_q,
    sigma_contact_complex,
)
from src.torch.perimeter.coherence_perimeter import (
    compute_coherence_loopwise,
)
from src.torch.transport.bem_wasserstein import compute_coherence

DT = torch.float64
SIG = 0.1


def circle(R, n, center=(0.0, 0.0), inward=False, phase=0.0):
    t = torch.arange(n, dtype=DT) * 2 * math.pi / n + phase
    pos = torch.stack([center[0] + R * t.cos(),
                       center[1] + R * t.sin()], 1)
    ang = t + (math.pi if inward else 0.0)
    m = torch.full((n,), 2 * math.pi * R / n, dtype=DT)
    return pos, ang, m


def assemble(parts):
    pos = torch.cat([p for p, _, _ in parts])
    ang = torch.cat([a for _, a, _ in parts])
    m = torch.cat([mm for _, _, mm in parts])
    lbl = torch.cat([torch.full((p.shape[0],), i, dtype=torch.long)
                     for i, (p, _, _) in enumerate(parts)])
    nor = torch.stack([ang.cos(), ang.sin()], 1)
    return pos, nor, m, lbl


class _V:
    def __init__(self, pos, nor):
        self.positions, self.normals = pos, nor


def qq(pos, nor, m, lbl):
    q_full = compute_coherence(_V(pos, nor), m, SIG, "wendland_c2")
    q_self, _ = compute_coherence_loopwise(pos, nor, m, SIG,
                                           "wendland_c2", lbl)
    return q_full, q_self


def cc(pos, nor, m, lbl):
    q_full, q_self = qq(pos, nor, m, lbl)
    return contact_complex_q(pos, nor, m, lbl, SIG, "wendland_c2",
                             q_full, q_self), q_full, q_self


def side_by_side(gap, R=0.35, n=90):
    """Two outward circles: facing normals point at each other
    (annihilating local contact when gap < sigma)."""
    return [circle(R, n, center=(-(R + gap / 2), 0.0)),
            circle(R, n, center=(R + gap / 2, 0.0))]


class TestAlgebra:
    def test_distant_loops_identity(self):
        pos, nor, m, lbl = assemble(side_by_side(gap=0.5))
        res, q_full, q_self = cc(pos, nor, m, lbl)
        assert res["cx"]["n_complexes"] == 0
        assert torch.equal(res["q_cc"], q_self)
        assert torch.equal(q_self, q_full)          # zero cross kernel

    def test_full_contact_reduces_to_qwb(self):
        pos, nor, m, lbl = assemble([circle(0.35, 90),
                                     circle(0.35, 141, inward=True)])
        res, q_full, q_self = cc(pos, nor, m, lbl)
        assert res["cx"]["n_complexes"] == 1
        for lb in (0, 1):
            sel = lbl == lb
            assert int(sel.sum()) == len(
                [s for s in [res["cx"]["complexes"][0].sides[i]
                             for i in (0, 1)] if s.loop == lb][0]
                .particles)                          # whole loop
            r_wb = float((m[sel] * q_full[sel]).sum()
                         / (m[sel] * q_self[sel]).sum())
            assert torch.allclose(res["q_cc"][sel],
                                  r_wb * q_self[sel], atol=0,
                                  rtol=0)            # exact reduction

    def test_sidewise_energy_and_far_field(self):
        pos, nor, m, lbl = assemble(side_by_side(gap=0.05))
        res, q_full, q_self = cc(pos, nor, m, lbl)
        assert res["cx"]["n_complexes"] == 1
        for side in res["cx"]["complexes"][0].sides:
            sel = side.mask
            lhs = float((m[sel] * res["q_cc"][sel]).sum())
            rhs = float((m[sel] * q_full[sel]).sum())
            assert abs(lhs - rhs) < 1e-13 * abs(rhs)   # eq. (3.1)
            assert side.n >= 2
        out = res["cx"]["complex_id"] == -1
        assert bool(out.any())
        assert torch.equal(res["q_cc"][out], q_full[out])
        assert torch.equal(q_self[out], q_full[out])   # exact far field

    def test_two_windows_independent(self):
        # big circle with two small circles at opposite poles: two
        # separate complexes; perturbing one must not touch the other
        A = circle(1.0, 256)
        B1 = circle(0.12, 40, center=(0.0, 1.12 + 0.05))
        B2 = circle(0.12, 40, center=(0.0, -(1.12 + 0.05)))
        pos, nor, m, lbl = assemble([A, B1, B2])
        res, _, _ = cc(pos, nor, m, lbl)
        assert res["cx"]["n_complexes"] == 2

        def top_rs(r):
            # the complex whose two sides are loops {0, 1} (A~B1); the
            # big loop 0 also has a side in the OTHER complex, so the
            # selection must be per complex, not per loop id
            for comp in r["cx"]["complexes"]:
                if sorted(s.loop for s in comp.sides) == [0, 1]:
                    sel = comp.sides[0].mask | comp.sides[1].mask
                    return ([float((m[s.mask] * r["q_cc"][s.mask])
                                   .sum()) for s in comp.sides], sel)
            raise AssertionError("A~B1 complex not found")
        r_top, sel_top = top_rs(res)
        B2b = circle(0.12, 40, center=(0.0, -(1.12 + 0.06)))
        pos2, nor2, m2, lbl2 = assemble([A, B1, B2b])
        res2, _, _ = cc(pos2, nor2, m2, lbl2)
        r_top2, sel_top2 = top_rs(res2)
        assert r_top == r_top2                       # bitwise
        assert torch.equal(res["q_cc"][sel_top],
                           res2["q_cc"][sel_top2])

    def test_full_pointcloud_equivariance(self):
        """Reviewer blocker A: the cyclic order is DERIVED from the
        source geometry (centroid-angle), so the pipeline must be
        equivariant under ARBITRARY within-loop permutations, label
        renaming, cyclic reversal, and rigid rotation -- not just
        order-preserving rolls."""
        parts = side_by_side(gap=0.05)
        pos, nor, m, lbl = assemble(parts)
        res, _, _ = cc(pos, nor, m, lbl)
        n = parts[0][0].shape[0]
        # (a) arbitrary within-loop permutation (fixed seed)
        g = torch.Generator().manual_seed(7)
        perm = torch.cat([torch.randperm(n, generator=g),
                          n + torch.randperm(n, generator=g)])
        res_p, _, _ = cc(pos[perm], nor[perm], m[perm], lbl[perm])
        assert torch.allclose(res_p["q_cc"], res["q_cc"][perm],
                              atol=1e-14, rtol=0)
        # (b) label renaming (swap loop ids)
        res_l, _, _ = cc(pos, nor, m, 1 - lbl)
        assert torch.allclose(res_l["q_cc"], res["q_cc"],
                              atol=1e-14, rtol=0)
        # (c) cyclic reversal of each loop's index order
        rev = torch.cat([torch.arange(n).flip(0),
                         torch.arange(n, 2 * n).flip(0)])
        res_v, _, _ = cc(pos[rev], nor[rev], m[rev], lbl[rev])
        assert torch.allclose(res_v["q_cc"], res["q_cc"][rev],
                              atol=1e-14, rtol=0)
        # (d) rigid rotation
        al = 0.37
        R = torch.tensor([[math.cos(al), -math.sin(al)],
                          [math.sin(al), math.cos(al)]], dtype=DT)
        res_r, _, _ = cc(pos @ R.T, nor @ R.T, m, lbl)
        assert torch.allclose(res_r["q_cc"], res["q_cc"],
                              atol=1e-12, rtol=0)

    def test_co_oriented_scope_and_delta(self):
        # concentric same-orientation sheets: r > 1 -> scope certificate
        pos, nor, m, lbl = assemble([circle(0.35, 90),
                                     circle(0.35 + 0.02, 141)])
        with pytest.raises(SigmaContactComplexError, match="scope"):
            cc(pos, nor, m, lbl)

    def test_fail_closed_three_loop_and_self_contact(self):
        # three SMALL circles around a triple point: the pairwise
        # windows are chained within one sigma ball -> a single
        # component meeting three loops (three well-separated pairwise
        # windows would be three legal 2-loop complexes instead)
        g = 0.03
        R = 0.06
        d = 2 * R + g
        h = d * math.sqrt(3) / 2
        pos, nor, m, lbl = assemble([
            circle(R, 40, center=(-d / 2, 0.0)),
            circle(R, 40, center=(d / 2, 0.0)),
            circle(R, 40, center=(0.0, h))])
        with pytest.raises(SigmaContactComplexError,
                           match="loops|interval|monotone"):
            cc(pos, nor, m, lbl)
        # horseshoe: same-loop sheets Euclid-close but arc-far
        t = torch.linspace(0.15 * math.pi, 1.85 * math.pi, 120,
                           dtype=DT)
        outer = torch.stack([torch.cos(t), torch.sin(t)], 1)
        inner = torch.stack([0.93 * torch.cos(t.flip(0)),
                             0.93 * torch.sin(t.flip(0))], 1)
        loop = torch.cat([outer, inner])
        ang = torch.atan2(loop[:, 1], loop[:, 0])
        mm = torch.full((loop.shape[0],), 0.02, dtype=DT)
        lb = torch.zeros(loop.shape[0], dtype=torch.long)
        orders = None
        with pytest.raises(SigmaContactComplexError,
                           match="self-contact|cyclic"):
            nor_h = torch.stack([ang.cos(), ang.sin()], 1)
            q_f, q_s = qq(loop, nor_h, mm, lb)
            contact_complex_q(loop, nor_h, mm, lb, SIG, "wendland_c2",
                              q_f, q_s)

    def test_monotone_helper(self):
        assert _monotone([0, 1, 1, 2])
        assert _monotone([5, 3, 3, 1])
        assert not _monotone([0, 2, 1, 3])           # crossing

    def test_source_frozen_gradient(self):
        pos, nor, m, lbl = assemble(side_by_side(gap=0.05))
        res, _, _ = cc(pos, nor, m, lbl)
        q_cc = res["q_cc"].detach()
        y = pos.clone().requires_grad_(True)
        c = pos.detach() - torch.tensor([0.3, 0.1], dtype=DT)
        w = (y - c).norm(dim=1) + m                  # smooth mock m(Y)
        P = (w * q_cc).sum()
        g = torch.autograd.grad(P, y)[0]
        # frozen q: the gradient sees only dm/dY, never dq/dY
        e = torch.zeros_like(pos)
        e[3, 0] = 1e-6
        w2 = ((pos + e) - c).norm(dim=1) + m
        P2 = float((w2 * q_cc).sum())
        assert abs((P2 - float(P)) - float(g[3, 0]) * 1e-6) < 1e-10

    def test_default_paths_untouched(self):
        # self_renormalized fixture values are governed by the
        # existing 0J suite (tests/test_shape_correction.py); here we
        # only assert the new mode does not alter loopwise q_self
        pos, nor, m, lbl = assemble(side_by_side(gap=0.05))
        q_self_a, _ = compute_coherence_loopwise(pos, nor, m, SIG,
                                                 "wendland_c2", lbl)
        _res, _, q_self_b = cc(pos, nor, m, lbl)
        assert torch.equal(q_self_a, q_self_b)


class TestSupportBoundaryContinuity:
    def test_edge_birth_jump_measured(self):
        """Hard membership: crossing the support edge changes side
        membership and jumps r by a finite amount even though the
        kernel vanishes smoothly there. The jump is MEASURED (and
        must be finite/small-scale); acceptance vs the 0J refresh
        envelope is decided in the L0J-1 audit, not here."""
        jumps = []
        # (beyond ~0.092 the support fringe produces SINGLE-PAIR
        # satellite components -- see the dedicated pin below)
        gaps = torch.linspace(0.05, 0.0918, 39, dtype=DT)
        prev = None
        for g in gaps:
            pos, nor, m, lbl = assemble(side_by_side(float(g)))
            res, _, _ = cc(pos, nor, m, lbl)
            n_mem = int((res["cx"]["complex_id"] >= 0).sum())
            P = float((m * res["q_cc"]).sum())
            if prev is not None and n_mem != prev[0]:
                jumps.append(abs(P - prev[1]))
            prev = (n_mem, P)
        assert jumps, "sweep must cross at least one birth/death"
        for j in jumps:
            assert j < 1e-2        # finite, small-scale (documented)

    def test_satellite_pair_fails_closed(self):
        """Hard-membership pathology at the support fringe (measured
        in the sweep above): near tangency of the kernel support, a
        mirror pair {L_k, R_k} can connect ONLY to each other,
        forming a single-pair satellite component whose sides violate
        |I| >= 2 -> the whole evaluation is fail-closed. This is a
        REAL hard-complex artifact and prime evidence for the tapered
        core/collar fallback (L0J-1 decision input)."""
        pos, nor, m, lbl = assemble(side_by_side(0.0929))
        with pytest.raises(SigmaContactComplexError, match="< 2"):
            cc(pos, nor, m, lbl)

    def test_merge_becomes_three_loop_fail_closed(self):
        # two satellites approaching each other while both touch the
        # big circle: the merged component meets three loops -> the
        # complex refuses (fail-closed) rather than silently merging
        A = circle(1.0, 256)
        th = 0.06
        B1 = circle(0.12, 40, center=(1.17 * math.cos(th),
                                      1.17 * math.sin(th)))
        B2 = circle(0.12, 40, center=(1.17 * math.cos(-th),
                                      1.17 * math.sin(-th)))
        pos, nor, m, lbl = assemble([A, B1, B2])
        with pytest.raises(SigmaContactComplexError):
            cc(pos, nor, m, lbl)
