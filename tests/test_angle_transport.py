"""0L-theta0 pins: loopwise (direct-sum) angle transport.

Pin priority (reviewer): (1) block solve == standalone per-loop
solve; (2) off-block entries exactly zero; (3) distant loops:
union vs loopwise agreement within a resolved-mode envelope
(bitwise only on clear-singular-gap fixtures -- the loop-local
rcond is the POINT of the block solve); plus permutation
invariance, guards, per-block telemetry, defaults bitwise, the
constrained mean-zero solve (D 1 = 0, m^T D = 0 exactly, residual
minimality preserved -- Z (A Z)^+ B P, never P A^+ B P), and the
static cross-contamination pin (union rotates the disk under a
pure hole motion; loopwise does not).
"""

import math

import pytest
import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.perimeter.angle_constraint import (
    compute_angle_constraint_matrices,
    compute_angle_constraint_matrices_from_weights,
)
from src.torch.solver.mm_step import (
    MMConfig,
    MMStepper,
    solve_angle_map,
    solve_angle_map_blocks,
)

DT = torch.float64
SIG = 0.1


def circle(R, n, inward=False):
    t = torch.arange(n, dtype=DT) * 2 * math.pi / n
    pos = torch.stack([R * t.cos(), R * t.sin()], 1)
    ang = t + (math.pi if inward else 0.0)
    return pos, ang


def pair(R2=0.42, n1=90, n2=141, inward2=True):
    p1, a1 = circle(0.35, n1)
    p2, a2 = circle(R2, n2, inward=inward2)
    pos = torch.cat([p1, p2])
    ang = torch.cat([a1, a2])
    lbl = torch.cat([torch.zeros(n1, dtype=torch.long),
                     torch.ones(n2, dtype=torch.long)])
    v = OrientedPointCloudVarifold(positions=pos, angles=ang)
    tang = torch.stack([-v.normals[:, 1], v.normals[:, 0]], 1)
    w = torch.rand(pos.shape[0], generator=torch.Generator()
                   .manual_seed(3), dtype=DT) * 0.01 + 0.02
    return v, tang, w, lbl


class TestTheta0:
    def test_block_equals_standalone(self):
        v, tang, w, lbl = pair()
        sol = solve_angle_map_blocks(v.positions, tang, w, lbl,
                                     SIG, "wendland_c2")
        for lb in (0, 1):
            sel = (lbl == lb).nonzero().flatten()
            A_a, B_a = compute_angle_constraint_matrices_from_weights(
                v.positions[sel], tang[sel], w[sel], SIG,
                "wendland_c2")
            alone = solve_angle_map(A_a, B_a)
            block = sol.AB_solve[sel.unsqueeze(1), sel.unsqueeze(0)]
            assert torch.equal(block, alone.AB_solve)

    def test_off_block_exactly_zero(self):
        v, tang, w, lbl = pair()
        sol = solve_angle_map_blocks(v.positions, tang, w, lbl,
                                     SIG, "wendland_c2")
        cross = (lbl.unsqueeze(1) != lbl.unsqueeze(0))
        assert float(sol.AB_solve[cross].abs().max()) == 0.0
        assert sol.block_stats is not None
        assert len(sol.block_stats) == 2
        for st in sol.block_stats:
            assert st["cond"] > 0

    def test_distant_loops_envelope(self):
        """Far-separated loops (zero kernel overlap): union solve vs
        direct sum agree on the resolved action D s within a small
        envelope (NOT bitwise -- loop-local rcond may retain
        different near-threshold modes; that is the feature)."""
        v, tang, w, lbl = pair(R2=1.0)
        A, B = compute_angle_constraint_matrices_from_weights(
            v.positions, tang, w, SIG, "wendland_c2")
        cross = (lbl.unsqueeze(1) != lbl.unsqueeze(0))
        assert float(A[cross].abs().max()) == 0.0   # no overlap
        u = solve_angle_map(A, B)
        s = solve_angle_map_blocks(v.positions, tang, w, lbl,
                                   SIG, "wendland_c2")
        g = torch.Generator().manual_seed(5)
        x = torch.randn(v.n_points, generator=g, dtype=DT)
        du = u.AB_solve @ x
        ds = s.AB_solve @ x
        assert float((du - ds).abs().max()) < 1e-6 * max(
            1.0, float(du.abs().max()))

    def test_permutation_invariance(self):
        v, tang, w, lbl = pair()
        g = torch.Generator().manual_seed(7)
        perm = torch.randperm(v.n_points, generator=g)
        sol = solve_angle_map_blocks(v.positions, tang, w, lbl,
                                     SIG, "wendland_c2")
        solp = solve_angle_map_blocks(v.positions[perm], tang[perm],
                                      w[perm], lbl[perm], SIG,
                                      "wendland_c2")
        x = torch.randn(v.n_points, generator=g, dtype=DT)
        ref = sol.AB_solve @ x
        got = solp.AB_solve @ x[perm]
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.shape[0])
        assert torch.allclose(got[inv], ref, rtol=1e-10, atol=1e-12)

    def test_mean_zero_constrained_solve(self):
        """D 1 = 0 and m^T D = 0 EXACTLY, and the solve is the
        constrained least squares (residual not worse than the
        projector sandwich on a non-symmetric fixture)."""
        # jittered ellipse-ish loop: breaks circle symmetry
        n = 96
        t = torch.arange(n, dtype=DT) * 2 * math.pi / n
        g = torch.Generator().manual_seed(1)
        r = 0.5 + 0.08 * torch.cos(2 * t) \
            + 0.01 * torch.randn(n, generator=g, dtype=DT)
        pos = torch.stack([r * t.cos(), r * t.sin()], 1)
        ang = torch.atan2(pos[:, 1], pos[:, 0])
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        tang = torch.stack([-v.normals[:, 1], v.normals[:, 0]], 1)
        w = torch.rand(n, generator=g, dtype=DT) * 0.01 + 0.02
        lbl = torch.zeros(n, dtype=torch.long)
        sol = solve_angle_map_blocks(v.positions, tang, w, lbl,
                                     SIG, "wendland_c2",
                                     consistency="mean_zero")
        D = sol.AB_solve
        one = torch.ones(n, dtype=DT)
        assert float((D @ one).abs().max()) < 1e-10
        assert float((w @ D).abs().max()) < 1e-10
        # residual optimality vs the (wrong) projector sandwich
        A, B = compute_angle_constraint_matrices_from_weights(
            v.positions, tang, w, SIG, "wendland_c2")
        P = torch.eye(n, dtype=DT) - torch.outer(one, w) / w.sum()
        naive = P @ solve_angle_map(A, B).AB_solve @ P
        s = torch.randn(n, generator=g, dtype=DT)
        r_cons = float((A @ (D @ s) - B @ (P @ s)).norm())
        r_naive = float((A @ (naive @ s) - B @ (P @ s)).norm())
        assert r_cons <= r_naive + 1e-12

    def test_cross_contamination_pin(self):
        """The decisive static pin: a pure radial motion of the hole
        sheet rotates the DISK normals under the union map, exactly
        zero under the direct sum."""
        v, tang, w, lbl = pair(R2=0.42)   # gap 0.07 < sigma 0.1
        A, B = compute_angle_constraint_matrices_from_weights(
            v.positions, tang, w, SIG, "wendland_c2")
        u = solve_angle_map(A, B)
        s = solve_angle_map_blocks(v.positions, tang, w, lbl,
                                   SIG, "wendland_c2")
        mo = torch.zeros(v.n_points, dtype=DT)
        mo[lbl == 1] = 1e-4
        disk = lbl == 0
        assert float((u.AB_solve @ mo)[disk].abs().max()) > 1e-8
        assert float((s.AB_solve @ mo)[disk].abs().max()) == 0.0

    def test_guards_and_defaults(self):
        cfg = MMConfig()
        assert cfg.angle_constraint_scope == "union"
        assert cfg.angle_constraint_measure == "visible_full"
        with pytest.raises(ValueError, match="scope"):
            MMStepper(MMConfig(
                angle_constraint_measure="raw_loopwise"))
        with pytest.raises(ValueError, match="loopwise mass"):
            MMStepper(MMConfig(
                angle_constraint_scope="loopwise",
                angle_constraint_measure="raw_loopwise"))
        with pytest.raises(ValueError, match="q at the OUTPUT|q_"):
            MMStepper(MMConfig(
                angle_constraint_scope="loopwise",
                angle_constraint_measure="raw_loopwise",
                mass_estimator="loopwise_oriented_kde",
                displacement_q_suppress=True))
        with pytest.raises(ValueError, match="mean_zero"):
            MMStepper(MMConfig(
                angle_constraint_consistency="mean_zero"))
