"""L0J-M pins: polygon first-moment rows + translation-balanced side
factors (reviewer 2026-08-21 corrected spec).

Moment rows: central-FD match with O(t^2) convergence (cubic
functional -> residual shrinks ~1/4 under t -> t/2), quadratic
residual along DM = 0 directions, rigid-translation identity
DM[a] = A a, rotation covariance (the (x, y) row pair transforms as a
vector), permutation/label equivariance, quotient aggregation
(Q (x) I_2), zero-row fail-closed. Solver: rank screening, bounds
semantics (unconstrained-out-of-bounds vs bounded-infeasible are
DIFFERENT statuses), synthetic feasible/infeasible cases.
"""

import math

import pytest
import torch

from src.torch.transport.loop_geometry import (
    LoopOrder,
    first_moment_variation_rows,
    polygon_first_moments,
)
from src.torch.perimeter.contact_complex import translation_balanced_r

DT = torch.float64


def circle_cloud(R=0.7, n=48, center=(0.2, -0.1)):
    t = torch.arange(n, dtype=DT) * 2 * math.pi / n
    pos = torch.stack([center[0] + R * t.cos(),
                       center[1] + R * t.sin()], 1)
    nor = torch.stack([t.cos(), t.sin()], 1)
    order = torch.arange(n)
    return pos, nor, order


class _LO:
    def __init__(self, order):
        self.order = order


def rows_of(pos, nor, order, n_bulk=1, bulk=None, loops=None):
    n = pos.shape[0]
    bulk = bulk if bulk is not None else torch.zeros(n,
                                                     dtype=torch.long)
    loops = loops if loops is not None else torch.zeros(
        n, dtype=torch.long)
    orders = {int(lb): _LO(order) for lb in loops.unique()}
    return first_moment_variation_rows(pos, nor, bulk, orders, loops,
                                       n_bulk)


class TestMomentRows:
    def test_fd_match_and_quadratic_shrink(self):
        pos, nor, order = circle_cloud()
        rows = rows_of(pos, nor, order)
        g = torch.Generator().manual_seed(3)
        s = torch.randn(pos.shape[0], generator=g, dtype=DT)
        DM = float(rows[0, :pos.shape[0]] @ s) * 1.0
        res = []
        for t in (1e-3, 5e-4):
            d = s.unsqueeze(1) * nor
            Mp = polygon_first_moments(pos + t * d, order)[0]
            Mm = polygon_first_moments(pos - t * d, order)[0]
            fd = float(Mp - Mm) / (2 * t)
            res.append(abs(fd - DM))
        assert res[0] < 1e-6
        assert res[1] < res[0] / 3.0        # ~1/4 (O(t^2))

    def test_null_direction_second_order(self):
        pos, nor, order = circle_cloud()
        rows = rows_of(pos, nor, order)
        g = torch.Generator().manual_seed(5)
        s = torch.randn(pos.shape[0], generator=g, dtype=DT)
        # project onto DM_x = DM_y = 0
        A = rows[:, :pos.shape[0]]
        s = s - A.T @ torch.linalg.lstsq(A.T, s.unsqueeze(1)
                                         ).solution.flatten()
        assert float((A @ s).abs().max()) < 1e-12
        M0 = polygon_first_moments(pos, order)
        for t in (1e-3, 5e-4):
            d = s.unsqueeze(1) * nor
            M1 = polygon_first_moments(pos + t * d, order)
            dM = abs(float(M1[0] - M0[0]))
            assert dM < 50.0 * t * t * float(s.norm()) ** 2

    def test_rigid_translation_identity(self):
        """dM[full rigid a] = A a EXACTLY (functional identity; for a
        POLYGON the tangential part of a rigid motion also carries
        first-order moment change, so this is checked on the full
        vector field, not through the normal-projected rows)."""
        pos, nor, order = circle_cloud()
        from src.torch.transport.loop_geometry import polygon_area
        A_poly = float(polygon_area(pos, order))
        a = torch.tensor([1.0, 0.0], dtype=DT)
        t = 1e-3
        Mp = polygon_first_moments(pos + t * a, order)
        Mm = polygon_first_moments(pos - t * a, order)
        assert abs(float(Mp[0] - Mm[0]) / (2 * t) - A_poly) \
            < 1e-10 * max(abs(A_poly), 1.0)
        assert abs(float(Mp[1] - Mm[1]) / (2 * t)) < 1e-10

    def test_rotation_covariance(self):
        pos, nor, order = circle_cloud()
        rows = rows_of(pos, nor, order)
        al = 0.41
        R = torch.tensor([[math.cos(al), -math.sin(al)],
                          [math.sin(al), math.cos(al)]], dtype=DT)
        rows_r = rows_of(pos @ R.T, nor @ R.T, order)
        n = pos.shape[0]
        # the (x, y) pair transforms as a vector: rows_r = R rows
        want = R @ rows[:, :n]
        assert torch.allclose(rows_r[:, :n], want, atol=1e-12,
                              rtol=0)

    def test_quotient_aggregation_and_equivariance(self):
        # two loops / two bulks; merged rows = sum of raw pairs
        p1, n1, o1 = circle_cloud(R=0.5, n=32, center=(-0.8, 0.0))
        p2, n2, o2 = circle_cloud(R=0.4, n=24, center=(0.8, 0.1))
        pos = torch.cat([p1, p2])
        nor = torch.cat([n1, n2])
        loops = torch.cat([torch.zeros(32, dtype=torch.long),
                           torch.ones(24, dtype=torch.long)])
        bulk = loops.clone()
        orders = {0: _LO(torch.arange(32)),
                  1: _LO(32 + torch.arange(24))}
        rows = first_moment_variation_rows(pos, nor, bulk, orders,
                                           loops, 2)
        assert rows.shape == (4, 2 * 56)
        merged = torch.stack([rows[0] + rows[2], rows[1] + rows[3]])
        # (Q (x) I2) with Q = [1 1]
        Q = torch.ones(1, 2, dtype=DT)
        M2 = torch.zeros(2, rows.shape[1], dtype=DT)
        for b in range(2):
            M2[0] += float(Q[0, b]) * rows[2 * b]
            M2[1] += float(Q[0, b]) * rows[2 * b + 1]
        assert torch.equal(M2, merged)
        # dtheta block exactly zero
        assert float(rows[:, 56:].abs().max()) == 0.0
        # label swap equivariance
        rows_sw = first_moment_variation_rows(
            pos, nor, 1 - bulk, orders, loops, 2)
        assert torch.equal(rows_sw[0], rows[2])
        assert torch.equal(rows_sw[2], rows[0])


class TestTranslationBalancedSolver:
    def _system(self, k=10):
        g = torch.Generator().manual_seed(11)
        e = torch.rand(k, generator=g, dtype=DT) + 0.5
        fx = torch.randn(k, generator=g, dtype=DT)
        A = torch.stack([e, fx])
        r_true = torch.full((k,), 0.8, dtype=DT)
        b = A @ r_true
        return A, b, r_true

    def test_ok_unconstrained(self):
        A, b, r_true = self._system()
        sol = translation_balanced_r(
            A, b, r0=torch.full_like(r_true, 0.7),
            w=torch.ones_like(r_true), hi=torch.full_like(r_true, 2.0))
        assert sol["status"] == "ok_unconstrained"
        assert float((A @ sol["r"] - b).abs().max()) < 1e-10

    def test_bounds_vs_infeasible_distinction(self):
        A, b, _ = self._system()
        # target reachable only outside a tight box -> unconstrained
        # projection leaves the box, and the bounded solve decides
        sol = translation_balanced_r(
            A, b * 2.4, r0=torch.full((10,), 0.7, dtype=DT),
            w=torch.ones(10, dtype=DT), hi=torch.full((10,), 1.0,
                                                      dtype=DT))
        assert sol["status"] in ("ok_bounded",
                                 "bounded_system_infeasible")
        if sol["status"] == "ok_bounded":
            assert sol["unconstrained_out_of_bounds"]
        # a plainly impossible target (energy row positive, bounds
        # cap the максимум) must be infeasible
        sol2 = translation_balanced_r(
            A, b * 10.0, r0=torch.full((10,), 0.7, dtype=DT),
            w=torch.ones(10, dtype=DT), hi=torch.full((10,), 1.0,
                                                      dtype=DT))
        assert sol2["status"] == "bounded_system_infeasible"

    def test_rank_screen(self):
        k = 8
        e = torch.ones(k, dtype=DT)
        A = torch.stack([e, e * 1e-14])       # second row ~ zero
        b = torch.tensor([4.0, 1.0], dtype=DT)   # target NOT within
        sol = translation_balanced_r(
            A, b, r0=torch.full((k,), 0.5, dtype=DT),
            w=torch.ones(k, dtype=DT), hi=torch.full((k,), 2.0,
                                                     dtype=DT))
        assert sol["status"] == "rank_deficient_residual"


class TestCurrentFormRows:
    """Order-free flux-family first-moment rows (user 2026-08-24):
    same zeta / r / endpoint machinery as the volume rows, consuming
    NO cyclic order."""

    def _flux(self, n=96, R=0.35, cx=0.0):
        import math
        import torch
        from src.torch.transport.boundary_flux_grid import (
            BoundaryFluxPolicies,
            BoundaryFluxToGrid,
        )
        from src.torch.transport.phase_grid import PhaseGridConfig
        t = torch.arange(n, dtype=torch.float64) * 2 * math.pi / n
        pos = torch.stack([cx + R * t.cos(), R * t.sin()], 1)
        nor = torch.stack([t.cos(), t.sin()], 1)
        m = torch.full((n,), 2 * math.pi * R / n, dtype=torch.float64)
        q = torch.ones(n, dtype=torch.float64)
        gc = PhaseGridConfig(grid_shape=(256, 256))
        fl = BoundaryFluxToGrid(pos, nor, m, q, gc,
                                BoundaryFluxPolicies())
        return fl, pos, nor, m, n

    def test_rigid_translation_identity_orderfree(self):
        # dM[V = a] = A a exactly in the continuum; the flux rows
        # reproduce it to O(h^2) DIRECTLY (unlike the polygon rows,
        # where the tangential part of a rigid motion carries
        # first-order moment change and the identity holds only at
        # the functional level)
        import math
        import torch
        fl, pos, nor, m, n = self._flux(cx=0.3)
        lbl = torch.zeros(n, dtype=torch.long)
        M = fl.first_moment_rows(lbl, 1)          # (2, 2N)
        A = math.pi * 0.35 ** 2
        for a in (torch.tensor([1.0, 0.0], dtype=torch.float64),
                  torch.tensor([0.0, 1.0], dtype=torch.float64),
                  torch.tensor([0.6, -0.8], dtype=torch.float64)):
            s = nor @ a                            # normal component
            dth = torch.zeros(n, dtype=torch.float64)
            v = torch.cat([s, dth])
            dM = M @ v
            ref = A * a
            assert torch.allclose(dM, ref, rtol=5e-3, atol=1e-12), \
                (dM, ref)

    def test_matches_polygon_rows_to_discretization(self):
        import torch
        from src.torch.transport.loop_geometry import (
            certified_cyclic_order,
            first_moment_variation_rows,
        )
        fl, pos, nor, m, n = self._flux(cx=0.3)
        lbl = torch.zeros(n, dtype=torch.long)
        Mc = fl.first_moment_rows(lbl, 1)
        orders = certified_cyclic_order(pos, lbl, nor, m,
                                        float(m.median()))
        Mp = first_moment_variation_rows(pos, nor, lbl, orders, lbl, 1)
        # s-blocks agree to O(h) relative; the dtheta blocks differ
        # by design (flux family vs positions-only functional)
        for r in (0, 1):
            a = Mc[r][:n]
            b = Mp[r][:n]
            rel = float((a - b).norm() / b.norm())
            assert rel < 0.05, rel

    def test_theta_block_analytic(self):
        import torch
        fl, pos, nor, m, n = self._flux()
        lbl = torch.zeros(n, dtype=torch.long)
        M = fl.first_moment_rows(lbl, 1)
        w = fl.zeta * fl.endpoint_positions[:, :, 0]
        ref = -(w * fl.r_physical).sum(dim=1)
        assert torch.equal(M[0][n:], ref)

    def test_stepper_default_bitwise_and_current_orderfree(self):
        # config default form == polygon (bitwise); the current form
        # must build WITHOUT the certified-order machinery
        from src.torch.solver.mm_step import MMConfig
        c = MMConfig()
        assert c.grid_bulk_first_moment_rows_form == "polygon"


class TestQuotientCenteredAggregation:
    """Reviewer 2026-08-24 correction: centered rows must aggregate
    FIRST, then center with the MERGED barycenter (eq 1); centering
    per raw bulk before aggregation (eq 2) wrongly retains
    per-component constraints in State III. ASYMMETRIC off-origin
    fixture -- a symmetric one hides the bug."""

    def _two_bulks(self):
        import math
        import torch
        from src.torch.transport.boundary_flux_grid import (
            BoundaryFluxPolicies,
            BoundaryFluxToGrid,
        )
        from src.torch.transport.phase_grid import PhaseGridConfig
        parts = [(0.30, 80, (-0.60, 0.10)), (0.50, 120, (0.70, -0.20))]
        pos, nor, m, lbl = [], [], [], []
        for b, (R, n, c) in enumerate(parts):
            t = torch.arange(n, dtype=torch.float64) * 2 * math.pi / n
            pos.append(torch.stack([c[0] + R * t.cos(),
                                    c[1] + R * t.sin()], 1))
            nor.append(torch.stack([t.cos(), t.sin()], 1))
            m.append(torch.full((n,), 2 * math.pi * R / n,
                                dtype=torch.float64))
            lbl.append(torch.full((n,), b, dtype=torch.long))
        pos, nor = torch.cat(pos), torch.cat(nor)
        m, lbl = torch.cat(m), torch.cat(lbl)
        q = torch.ones(pos.shape[0], dtype=torch.float64)
        fl = BoundaryFluxToGrid(pos, nor, m, q,
                                PhaseGridConfig(grid_shape=(256, 256)),
                                BoundaryFluxPolicies())
        return fl, pos, nor, m, lbl

    def _rows_scalars(self, fl, pos, nor, m, lbl):
        M = fl.first_moment_rows(lbl, 2)
        CA = fl.bulk_rows(lbl, 2)
        sc = []
        xn = (pos * nor).sum(dim=1)
        for b in (0, 1):
            sel = lbl == b
            A = float(0.5 * (m[sel] * xn[sel]).sum())
            sc.append((A,
                       float(0.5 * (m[sel] * pos[sel, 0] ** 2
                                    * nor[sel, 0]).sum()),
                       float(0.5 * (m[sel] * pos[sel, 1] ** 2
                                    * nor[sel, 1]).sum())))
        return M, CA, sc

    def test_eq1_vs_eq2_differ_and_eq1_realized(self):
        import torch
        from src.torch.transport.boundary_flux_grid import (
            quotient_centered_moment_rows,
        )
        fl, pos, nor, m, lbl = self._two_bulks()
        M, CA, sc = self._rows_scalars(fl, pos, nor, m, lbl)
        Q = torch.ones(1, 2, dtype=torch.float64)
        got = quotient_centered_moment_rows(M, CA, sc, Q)
        # eq (1) by hand
        A_q = sc[0][0] + sc[1][0]
        for k in (0, 1):
            Mq = sc[0][1 + k] + sc[1][1 + k]
            ref = (M[2 * 0 + k] + M[2 * 1 + k]) \
                - (Mq / A_q) * (CA[0] + CA[1])
            assert torch.allclose(got[k], ref, atol=1e-14)
        # eq (2) (the bug) differs, by exactly eq (3):
        # got - bug = sum_b (M_b/A_b - M_c/A_c) C_{A,b}
        bug = sum(M[2 * b + 0] - (sc[b][1] / sc[b][0]) * CA[b]
                  for b in (0, 1))
        diff = got[0] - bug
        Mc_over_Ac = (sc[0][1] + sc[1][1]) / A_q
        ref3 = sum((sc[b][1] / sc[b][0] - Mc_over_Ac) * CA[b]
                   for b in (0, 1))
        assert float(diff.norm()) > 1e-3 * float(got[0].norm())
        assert torch.allclose(diff, ref3, atol=1e-12)

    def test_translation_invariance_of_centered_row(self):
        import torch
        from src.torch.transport.boundary_flux_grid import (
            BoundaryFluxPolicies,
            BoundaryFluxToGrid,
            quotient_centered_moment_rows,
        )
        from src.torch.transport.phase_grid import PhaseGridConfig
        fl, pos, nor, m, lbl = self._two_bulks()
        M, CA, sc = self._rows_scalars(fl, pos, nor, m, lbl)
        Q = torch.ones(1, 2, dtype=torch.float64)
        base = quotient_centered_moment_rows(M, CA, sc, Q)
        T = torch.tensor([0.37, -0.81], dtype=torch.float64)
        q1 = torch.ones(pos.shape[0], dtype=torch.float64)
        fl2 = BoundaryFluxToGrid(pos + T, nor, m, q1,
                                 PhaseGridConfig(
                                     grid_shape=(256, 256)),
                                 BoundaryFluxPolicies())
        M2, CA2, sc2 = self._rows_scalars(fl2, pos + T, nor, m, lbl)
        shifted = quotient_centered_moment_rows(M2, CA2, sc2, Q)
        assert torch.allclose(shifted, base, atol=1e-10), \
            float((shifted - base).norm())

    def test_relative_volume_mode_coupled_but_not_pinned(self):
        # Physics (this pin originally asserted the OPPOSITE and the
        # numbers corrected it): transferring volume between the
        # separated bulks MOVES the merged barycenter
        # (dxbar_c = (xbar_2 - xbar_1) dV / A_c), so the correct
        # merged-centered row MUST couple to the relative-volume
        # mode (residual fraction < 1) while not containing it in
        # its span (residual fraction > 0) -- coupled, not pinned.
        # The eq (2) bug is orthogonal to per-bulk dilations by
        # construction (residual 0.95): an UNPHYSICAL decoupling --
        # it cannot see that a transfer moves the merged barycenter.
        import torch
        from src.torch.transport.boundary_flux_grid import (
            quotient_centered_moment_rows,
        )
        fl, pos, nor, m, lbl = self._two_bulks()
        M, CA, sc = self._rows_scalars(fl, pos, nor, m, lbl)
        Q = torch.ones(1, 2, dtype=torch.float64)
        cbar = quotient_centered_moment_rows(M, CA, sc, Q)
        bug = torch.stack([
            sum(M[2 * b + k] - (sc[b][1 + k] / sc[b][0]) * CA[b]
                for b in (0, 1)) for k in (0, 1)])

        def resid_frac(rows):
            span = torch.stack([CA[0] + CA[1], rows[0], rows[1]])
            r = CA[0] - CA[1]
            sol = torch.linalg.lstsq(span.T, r[:, None]).solution
            resid = r - (span.T @ sol).flatten()
            return float(resid.norm() / r.norm())
        frac_eq1 = resid_frac(cbar)
        frac_eq2 = resid_frac(bug)
        assert 0.1 < frac_eq1 < 0.9, frac_eq1     # coupled, not pinned
        assert frac_eq2 > frac_eq1 + 0.2, (frac_eq1, frac_eq2) \
            # the bug's unphysical decoupling, kept as its signature

    def test_state_ii_kernel_equivalence(self):
        # identity quotient: on {s : C_A,b s = 0} the centered and
        # uncentered rows act identically
        import torch
        from src.torch.transport.boundary_flux_grid import (
            quotient_centered_moment_rows,
        )
        fl, pos, nor, m, lbl = self._two_bulks()
        M, CA, sc = self._rows_scalars(fl, pos, nor, m, lbl)
        cen = quotient_centered_moment_rows(M, CA, sc, None)
        g = torch.Generator().manual_seed(3)
        v = torch.randn(M.shape[1], generator=g, dtype=torch.float64)
        # project v onto the kernel of both C_A rows
        for row in CA:
            v = v - (row @ v) / (row @ row) * row
        for b in (0, 1):
            for k in (0, 1):
                lhs = float(cen[2 * b + k] @ v)
                rhs = float(M[2 * b + k] @ v)
                assert abs(lhs - rhs) < 1e-10 * max(1.0, abs(rhs))

    def test_degenerate_class_volume_fail_closed(self):
        import pytest
        import torch
        from src.torch.transport.boundary_flux_grid import (
            quotient_centered_moment_rows,
        )
        fl, pos, nor, m, lbl = self._two_bulks()
        M, CA, sc = self._rows_scalars(fl, pos, nor, m, lbl)
        bad = [(0.0, sc[0][1], sc[0][2]), sc[1]]
        with pytest.raises(RuntimeError, match="degenerate"):
            quotient_centered_moment_rows(M, CA, bad, None)
