"""Block-diagonal BIE Wasserstein metric (bie_wasserstein.py).

Gates (CPU, float64):
- disk Fourier modes against the exact interior-Neumann value
  (h/2) (R/k) pi R (V = cos k theta on the unit-speed scale);
- annulus radial mode against pi c^2 R+^2 log(R+/R-);
- two separated circles: exact block diagonality, additivity,
  componentwise compatibility (an exchange velocity is refused);
- analytic HVP == autograd HVP, symmetric dense Hessian, positive
  definite on the admissible subspace;
- merge + mask: two ellipses inside the bridging gap form ONE metric
  component whose cancelling pair is masked; concentric disk + annulus
  reduce to the outer loop alone; the bridging threshold is sharp.
"""

import math

import pytest
import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.shapes.generator import (
    generate_oriented_annulus,
    generate_oriented_circle,
    generate_oriented_two_circles,
    generate_oriented_two_ellipses,
)
from src.torch.transport.bem_wasserstein import IncompatibleVelocityError
from src.torch.transport.bie_wasserstein import (
    BIEMetricConfig,
    BIEWassersteinMetric,
    cancelling_pair_mask,
    loop_gap_matrix,
    merge_metric_labels,
)
from src.torch.transport.boundary_flux_grid import BoundaryFluxPolicies
from src.torch.transport.incidence import (
    oriented_graph_components,
    winding_bulk_labels,
)

DT = torch.float64
H = 1.0


def _partition(v, m):
    spacing = float(m.median())
    loops = oriented_graph_components(v.positions, v.normals, 1.5 * spacing)
    bulk_of_loop, margin = winding_bulk_labels(
        v.positions, v.normals, m, loops, spacing)
    assert bulk_of_loop is not None, f"winding margin {margin}"
    return dict(bulk_of_loop=bulk_of_loop, loops=loops,
                bulk_pp=bulk_of_loop[loops],
                n_bulk=int(bulk_of_loop.max()) + 1, margin=margin)


def _uniform_masses(v, loops):
    """arc-length masses per loop (closed polygon length / count)."""
    m = torch.zeros(v.n_points, dtype=DT)
    for lb in loops.unique():
        idx = torch.nonzero(loops == lb).flatten()
        p = v.positions[idx]
        L = float((p.roll(-1, 0) - p).norm(dim=1).sum())
        m[idx] = L / idx.numel()
    return m


def _metric(v, cfg=None):
    loops = oriented_graph_components(
        v.positions, v.normals,
        1.5 * float((v.positions.roll(-1, 0) - v.positions)
                    .norm(dim=1).median()))
    m = _uniform_masses(v, loops)
    part = _partition(v, m)
    met = BIEWassersteinMetric(cfg or BIEMetricConfig())
    met.setup_for_step(v, m, torch.ones_like(m), 0.0,
                       BoundaryFluxPolicies(), bulk_partition=part)
    return met, m, part


class TestExactModes:
    def test_disk_fourier(self):
        N, R = 256, 1.0
        v = generate_oriented_circle(N, R, (0.0, 0.0), "cpu", DT)
        met, _, _ = _metric(v)
        assert met.n_compat_components == 1
        th = torch.atan2(v.positions[:, 1], v.positions[:, 0])
        z = torch.zeros(N, dtype=DT)
        for k in (1, 2, 3, 5):
            W = float(met(torch.cos(k * th), z, H))
            exact = 0.5 * H * (R / k) * math.pi * R / H ** 2
            assert abs(W / exact - 1.0) < 0.02, (k, W, exact)

    def test_annulus_radial(self):
        Rp, Rm, c = 1.2, 0.5, 0.05
        v = generate_oriented_annulus(384, Rp, Rm, (0.0, 0.0), "cpu", DT)
        met, m, part = _metric(v)
        assert met.n_compat_components == 1 and part["n_bulk"] == 1
        r = v.positions.norm(dim=1)
        inner = r < 0.5 * (Rp + Rm)
        # inner speed from the DISCRETE mass balance (= c R+/R- up to
        # the polygon-length error), so the mode is exactly admissible
        c_in = c * float(m[~inner].sum()) / float(m[inner].sum())
        s = torch.where(inner, torch.full_like(r, -c_in),
                        torch.full_like(r, c))
        z = torch.zeros_like(s)
        rows = met.component_rows()
        flux = float(rows[0] @ torch.cat([s, z]))
        assert abs(flux) < 1e-12 * float(rows.abs().sum())
        W = float(met(s, z, H))
        exact = math.pi * c * c * Rp * Rp * math.log(Rp / Rm)
        assert abs(W / exact - 1.0) < 0.02, (W, exact)


class TestBlockStructure:
    def _pair(self):
        v = generate_oriented_two_circles(
            256, radius1=0.8, center1=(-1.0, 0.0),
            radius2=0.8, center2=(1.0, 0.0), device="cpu", dtype=DT)
        return v, *_metric(v)

    def test_two_circles_block_diagonal_and_additive(self):
        v, met, m, part = self._pair()
        assert met.n_compat_components == 2
        assert len(met.blocks) == 2 and not met.merged_pairs
        N = v.n_points
        Hd = met.hessian_dense(H)
        left = torch.nonzero(met.compat_labels == 0).flatten()
        right = torch.nonzero(met.compat_labels == 1).flatten()
        idx_l = torch.cat([left, N + left])
        idx_r = torch.cat([right, N + right])
        assert float(Hd[idx_l][:, idx_r].abs().max()) == 0.0
        th = torch.atan2(v.positions[:, 1] - 0.0,
                         v.positions[:, 0] - torch.where(
                             met.compat_labels == 0, -1.0, 1.0))
        z = torch.zeros(N, dtype=DT)
        s_l = torch.where(met.compat_labels == 0, torch.cos(2 * th), 0.0)
        s_r = torch.where(met.compat_labels == 1, torch.cos(3 * th), 0.0)
        W_l, W_r, W = (float(met(s_l, z, H)), float(met(s_r, z, H)),
                       float(met(s_l + s_r, z, H)))
        assert abs(W - W_l - W_r) < 1e-12 * (W_l + W_r)

    def test_exchange_velocity_refused(self):
        v, met, m, part = self._pair()
        s = torch.where(met.compat_labels == 0, 1.0, -1.0).to(DT)
        with pytest.raises(IncompatibleVelocityError):
            met(s, torch.zeros_like(s), H)


class TestHessian:
    def test_hessp_matches_autograd_and_spd(self):
        v = generate_oriented_annulus(192, 1.0, 0.5, (0.0, 0.0),
                                      "cpu", DT)
        met, m, part = _metric(v)
        N = v.n_points
        torch.manual_seed(3)
        rows = met.component_rows()
        # admissible random directions: project out the flux row
        def admissible(vec):
            r = rows[0]
            return vec - r * (float(r @ vec) / float(r @ r))
        d = admissible(torch.randn(2 * N, dtype=DT))
        x = admissible(torch.randn(2 * N, dtype=DT)).requires_grad_(True)
        W = met(x[:N], x[N:], H)
        g, = torch.autograd.grad(W, x, create_graph=True)
        hv, = torch.autograd.grad(g @ d, x)
        gs, gth = met.hessp(d[:N], d[N:], H)
        assert float((hv - torch.cat([gs, gth])).abs().max()) \
            < 1e-9 * float(hv.abs().max())
        Hd = met.hessian_dense(H)
        assert float((Hd - Hd.T).abs().max()) < 1e-12 * float(Hd.abs().max())
        # positive definite on the admissible subspace (complement of
        # the flux row)
        U, S, Vh = torch.linalg.svd(rows, full_matrices=True)
        Q = Vh[1:].T
        ev = torch.linalg.eigvalsh(Q.T @ Hd @ Q)
        assert float(ev[0]) > 0.0, float(ev[0])


class TestMergeAndMask:
    def _two_ellipses(self, gap_over_ell, n_el=128):
        # probe ell from a well separated configuration
        v0 = generate_oriented_two_ellipses(
            n_el, a1=0.4, b1=1.0, center1=(-1.0, 0.0),
            a2=0.4, b2=1.0, center2=(1.0, 0.0), device="cpu", dtype=DT)
        loops = oriented_graph_components(v0.positions, v0.normals, 0.06)
        ell = float(_uniform_masses(v0, loops).median())
        gap = gap_over_ell * ell
        v = generate_oriented_two_ellipses(
            n_el, a1=0.4, b1=1.0, center1=(-(0.4 + gap / 2), 0.0),
            a2=0.4, b2=1.0, center2=((0.4 + gap / 2), 0.0),
            device="cpu", dtype=DT)
        return v, ell

    def test_threshold_is_sharp(self):
        v, ell = self._two_ellipses(1.2)
        met, m, part = _metric(v)
        assert part["n_bulk"] == 2
        assert met.n_compat_components == 2 and not met.merged_pairs
        assert int(met.masked.sum()) == 0
        v, ell = self._two_ellipses(0.8)
        met, m, part = _metric(v)
        assert part["n_bulk"] == 2
        assert met.n_compat_components == 1
        assert len(met.merged_pairs) == 1
        assert len(met.blocks) == 1

    def test_mask_is_the_facing_pair(self):
        v, ell = self._two_ellipses(0.8)
        met, m, part = _metric(v)
        masked = met.masked
        assert 4 <= int(masked.sum()) <= 40
        # masked particles face the other loop within g_mask with
        # anti-parallel normals; no masked particle is far from the
        # other loop
        loops = part["loops"]
        for a in (0, 1):
            ia = torch.nonzero((loops == a) & masked).flatten()
            ib = torch.nonzero(loops == 1 - a).flatten()
            D = torch.cdist(v.positions[ia], v.positions[ib])
            assert float(D.min(dim=1).values.max()) <= met.g_mask + 1e-12
            dots = (v.normals[ia] @ v.normals[ib].T)
            assert bool(((D <= met.g_mask) & (dots <= -0.9)).any(dim=1)
                        .all())
        # block excludes the masked particles and is well conditioned
        blk = met.blocks[0]
        assert not bool(masked[blk.particle_index].any())
        assert blk.sigma_tail > 1e-3
        # merged compatibility row = sum of the two bulk rows
        rows = met.component_rows()
        bulk = met.flux.bulk_rows(part["bulk_pp"], 2)
        assert float((rows[0] - bulk.sum(dim=0)).abs().max()) < 1e-14

    def test_concentric_reduces_to_outer_loop(self):
        ell = 2 * math.pi / 384
        gap = 0.9 * ell
        va = generate_oriented_annulus(384, 1.0, 0.5, (0.0, 0.0),
                                       "cpu", DT)
        vd = generate_oriented_circle(int(round(128 * (0.5 - gap) / 0.5)),
                                      0.5 - gap, (0.0, 0.0), "cpu", DT)
        v = OrientedPointCloudVarifold(
            positions=torch.cat([va.positions, vd.positions]),
            angles=torch.cat([va.angles, vd.angles]))
        met, m, part = _metric(v)
        assert part["n_bulk"] == 2
        assert met.n_compat_components == 1 and len(met.blocks) == 1
        outer = v.positions.norm(dim=1) > 0.75
        assert torch.equal(met.masked, ~outer)
        blk = met.blocks[0]
        assert int(blk.particle_index.numel()) == int(outer.sum())
        assert blk.sigma_tail > 0.5


def test_merge_helpers_relabel_in_loop_order():
    gaps = torch.tensor([[math.inf, 0.5, 0.01],
                         [0.5, math.inf, 0.5],
                         [0.01, 0.5, math.inf]], dtype=DT)
    bulk_of_loop = torch.tensor([0, 1, 2])
    lab, pairs = merge_metric_labels(gaps, bulk_of_loop, 3, 0.02)
    assert lab.tolist() == [0, 1, 0]
    assert pairs == [(0, 2, 0.01)]
    lab, pairs = merge_metric_labels(gaps, bulk_of_loop, 3, 0.001)
    assert lab.tolist() == [0, 1, 2] and pairs == []
