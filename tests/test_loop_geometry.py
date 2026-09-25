"""0L-A1 pins: certified cyclic order + exact polygon-area calculus.

- FD pin of the exact gradient g_i = 1/2 J_cw(x_{i+1}-x_{i-1}) on
  circle / ellipse / flower;
- exact discrete divergence identity A = 1/2 sum m^poly x . nu^poly
  (internal assert of polygon_area) + circle analytic values;
- EXACT quadratic identity A(X+dX) - A(X) = DA[dX] + Q_A(dX) to
  machine precision (the 0L-A2 gate quantity);
- certificate raises: center outside / self-intersection / long
  chord across a concavity / normal transversality;
- invariances: particle permutation, cyclic shift, reversal flips
  only the sign;
- annulus 2-loop orientation: hole loop signed NEGATIVE (aligned to
  its inward stored normals), so bulk sums need no manual signs;
- solver guards + default "current" non-regression.
"""

import math

import pytest
import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.shapes.generator import (
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_flower,
)
from src.torch.solver.mm_step import MMConfig, MMStepper
from src.torch.transport.loop_geometry import (
    LoopOrderError,
    area_quadratic_term,
    area_variation_rows,
    certified_cyclic_order,
    polygon_area,
    polygon_dual,
    polygon_gradient,
)

DT = torch.float64


def single_loop(v):
    lbl = torch.zeros(v.n_points, dtype=torch.long)
    m = torch.full((v.n_points,), 2 * math.pi / v.n_points, dtype=DT)
    spacing = float((v.positions.roll(-1, 0)
                     - v.positions).norm(dim=1).median())
    return lbl, m, spacing


def annulus_state(n_out=128, n_in=96, r_out=1.0, r_in=0.5):
    t1 = torch.arange(n_out, dtype=DT) * 2 * math.pi / n_out
    t2 = torch.arange(n_in, dtype=DT) * 2 * math.pi / n_in
    pos = torch.cat([
        r_out * torch.stack([t1.cos(), t1.sin()], 1),
        r_in * torch.stack([t2.cos(), t2.sin()], 1)])
    ang = torch.cat([t1, t2 + math.pi])       # inner sheet inward
    v = OrientedPointCloudVarifold(positions=pos, angles=ang)
    lbl = torch.cat([torch.zeros(n_out, dtype=torch.long),
                     torch.ones(n_in, dtype=torch.long)])
    m = torch.ones(v.n_points, dtype=DT) * 0.03
    return v, lbl, m


class TestPolygonCalculus:
    @pytest.mark.parametrize("maker", [
        lambda: generate_oriented_circle(96, 0.8, (0.0, 0.0), "cpu",
                                         DT),
        lambda: generate_oriented_ellipse(128, device="cpu", dtype=DT),
        lambda: generate_oriented_flower(160, device="cpu", dtype=DT),
    ])
    def test_gradient_fd_pin(self, maker):
        v = maker()
        lbl, m, spacing = single_loop(v)
        lo = certified_cyclic_order(v.positions, lbl, v.normals, m,
                                    spacing)[0]
        g = polygon_gradient(v.positions, lo.order)
        gen = torch.Generator().manual_seed(11)
        xi = torch.randn(v.positions.shape, generator=gen, dtype=DT)
        h = 1e-7
        fd = (float(polygon_area(v.positions + h * xi, lo.order))
              - float(polygon_area(v.positions - h * xi, lo.order))
              ) / (2 * h)
        assert float((g * xi).sum()) == pytest.approx(fd, rel=1e-7,
                                                      abs=1e-10)

    def test_circle_analytic_and_divergence_identity(self):
        R, n = 0.8, 256
        v = generate_oriented_circle(n, R, (0.0, 0.0), "cpu", DT)
        lbl, m, spacing = single_loop(v)
        lo = certified_cyclic_order(v.positions, lbl, v.normals, m,
                                    spacing)[0]
        A = float(polygon_area(v.positions, lo.order))
        assert A == pytest.approx(math.pi * R * R, rel=1e-3)
        g = polygon_gradient(v.positions, lo.order)
        # DA[nu] = perimeter (discrete)
        assert float((g * v.normals).sum()) == pytest.approx(
            2 * math.pi * R, rel=1e-3)
        m_poly, nu_poly = polygon_dual(v.positions, lo.order)
        # dual normal parallel to the radial stored normal
        cosphi = (nu_poly * v.normals[lo.order]).sum(1)
        assert float(cosphi.min()) > 1.0 - 1e-3

    def test_exact_quadratic_identity(self):
        v = generate_oriented_flower(160, device="cpu", dtype=DT)
        lbl, m, spacing = single_loop(v)
        lo = certified_cyclic_order(v.positions, lbl, v.normals, m,
                                    spacing)[0]
        g = polygon_gradient(v.positions, lo.order)
        gen = torch.Generator().manual_seed(5)
        dx = 0.02 * torch.randn(v.positions.shape, generator=gen,
                                dtype=DT)
        lhs = (float(polygon_area(v.positions + dx, lo.order))
               - float(polygon_area(v.positions, lo.order)))
        rhs = float((g * dx).sum()) + float(
            area_quadratic_term(dx, lo.order))
        assert lhs == pytest.approx(rhs, abs=1e-13)

    def test_invariances(self):
        v = generate_oriented_ellipse(96, device="cpu", dtype=DT)
        lbl, m, spacing = single_loop(v)
        lo = certified_cyclic_order(v.positions, lbl, v.normals, m,
                                    spacing)[0]
        A = float(polygon_area(v.positions, lo.order))
        # particle permutation: shuffle storage, same certified cycle
        gen = torch.Generator().manual_seed(2)
        perm = torch.randperm(v.n_points, generator=gen)
        v2 = OrientedPointCloudVarifold(positions=v.positions[perm],
                                        angles=v.angles[perm])
        lo2 = certified_cyclic_order(v2.positions, lbl, v2.normals,
                                     m, spacing)[0]
        assert float(polygon_area(v2.positions, lo2.order)) == \
            pytest.approx(A, rel=1e-14)
        # cyclic shift of the order tensor: identical area
        sh = torch.roll(lo.order, 7)
        assert float(polygon_area(v.positions, sh)) == \
            pytest.approx(A, rel=1e-14)
        # reversal: sign flip only
        assert float(polygon_area(v.positions, lo.order.flip(0))) == \
            pytest.approx(-A, rel=1e-14)

    def test_annulus_signed_orientation(self):
        v, lbl, m = annulus_state()
        spacing = 0.04
        orders = certified_cyclic_order(v.positions, lbl, v.normals,
                                        m, spacing)
        a_out = float(polygon_area(v.positions, orders[0].order))
        a_in = float(polygon_area(v.positions, orders[1].order))
        assert a_out == pytest.approx(math.pi, rel=1e-2)
        assert a_in == pytest.approx(-math.pi * 0.25, rel=1e-2)
        # bulk row: single annulus bulk sums outer + hole with signs
        rows = area_variation_rows(
            v.positions, v.normals, torch.zeros_like(lbl), orders,
            lbl, 1)
        assert rows.shape == (1, 2 * v.n_points)
        # dtheta block exactly zero
        assert float(rows[0, v.n_points:].abs().max()) == 0.0
        # uniform outward motion of the outer loop grows the bulk
        # area by ~perimeter; inward-normal motion of the hole sheet
        # (s > 0 along stored inward normals) SHRINKS the hole ->
        # also grows the annulus area
        s = torch.zeros(v.n_points, dtype=DT)
        s[:128] = 1.0
        assert float(rows[0, :v.n_points] @ s) == pytest.approx(
            2 * math.pi, rel=1e-2)
        s2 = torch.zeros(v.n_points, dtype=DT)
        s2[128:] = 1.0
        assert float(rows[0, :v.n_points] @ s2) == pytest.approx(
            2 * math.pi * 0.5, rel=1e-2)


class TestCertificates:
    def test_center_outside_raises(self):
        # crescent-like: points on an arc only -> centroid outside
        t = torch.linspace(0.0, 0.8, 40, dtype=DT)
        pos = torch.stack([t.cos(), t.sin()], 1)
        ang = t
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        lbl = torch.zeros(40, dtype=torch.long)
        m = torch.ones(40, dtype=DT)
        with pytest.raises(LoopOrderError):
            certified_cyclic_order(v.positions, lbl, v.normals, m,
                                   1.0)

    def test_long_chord_adjacency_raises(self):
        # remove a quarter of a circle: angular sort closes it with a
        # long chord -> adjacency envelope must reject
        n = 96
        t = torch.arange(n, dtype=DT) * 2 * math.pi / n
        keep = t < 1.5 * math.pi
        t = t[keep]
        pos = torch.stack([t.cos(), t.sin()], 1)
        v = OrientedPointCloudVarifold(positions=pos, angles=t)
        lbl = torch.zeros(pos.shape[0], dtype=torch.long)
        m = torch.ones(pos.shape[0], dtype=DT)
        spacing = 2 * math.pi / n
        with pytest.raises(LoopOrderError, match="adjacency|center"):
            certified_cyclic_order(v.positions, lbl, v.normals, m,
                                   spacing)

    def test_transversality_raises(self):
        # NON-uniform tilt (a uniform rotation is absorbed by the
        # orientation majority vote): rotate one quarter of the
        # stored normals by 120 degrees -> n . nu_poly < 0 there
        # while the majority stays aligned
        v = generate_oriented_circle(64, 1.0, (0.0, 0.0), "cpu", DT)
        lbl, m, spacing = single_loop(v)
        ang = v.angles.clone()
        ang[:16] = ang[:16] + math.radians(120)
        v2 = OrientedPointCloudVarifold(positions=v.positions,
                                        angles=ang)
        with pytest.raises(LoopOrderError, match="transversality"):
            certified_cyclic_order(v2.positions, lbl, v2.normals, m,
                                   spacing)

    def test_endgame_deep_state_certifies(self):
        """The certificate must HOLD on the real endgame-#2 deep
        state (three loops, pre-merger tilt)."""
        import sys
        from pathlib import Path
        p = Path("results/exact_merger/exact_merger_endgame2_states.pt")
        if not p.exists():
            pytest.skip("endgame2 states not present")
        sys.path.insert(0, "scripts/experiments")
        ck = torch.load(p, weights_only=True)
        st = ck[1350]
        v = OrientedPointCloudVarifold(positions=st["positions"],
                                       angles=st["angles"])
        from src.torch.oriented_varifold.loopwise_mass import (
            resolve_loopwise_oriented_mass_source,
        )
        from src.torch.oriented_varifold.mass import (
            compute_recommended_params,
        )
        d0, t0 = compute_recommended_params(v.positions)
        res = resolve_loopwise_oriented_mass_source(
            v.positions, v.normals, float(d0), float(t0))
        orders = certified_cyclic_order(
            v.positions, res.loop_labels_pre, v.normals, res.m_loop,
            float(res.m_loop.median()))
        assert len(orders) == 3
        for lo in orders.values():
            assert lo.min_transversality > 0.0


class TestSolverGuards:
    def test_geometric_requires_bulk_rows(self):
        with pytest.raises(ValueError, match="bulk_rows_mode"):
            MMStepper(MMConfig(
                grid_bulk_rows_functional="geometric_area"))

    def test_mixed_semantics_rejected(self):
        with pytest.raises(ValueError, match="mixed|rejected"):
            MMStepper(MMConfig(
                grid_bulk_rows_mode="augment",
                grid_volume_target_functional="geometric_area",
                grid_bulk_rows_functional="current"))

    def test_default_current_unchanged(self):
        cfg = MMConfig()
        assert cfg.grid_bulk_rows_functional == "current"
        assert cfg.grid_volume_target_functional == "current"
