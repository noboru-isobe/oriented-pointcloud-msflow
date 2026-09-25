"""0L-B0 pins: contact-layer row algebra and the diagnostic gauge hold.

- aggregation projector P_rel (idempotent, symmetric, kills A_gb^T,
  (G,B)=(1,2) -> span{C_d - C_a}, (G,B)=(2,4) rank B-G, permutation
  equivariance of the row space, containment in span{G, C_bulk})
- scale-invariant row-space discrepancy (0 for any scalar multiple,
  equals sqrt(1 - cos^2) for single rows, > 0 off-axis)
- bulk-seeded gauge labels on a two-circle fixture (partition of the
  active set, connected, non-empty, particle-consistent, permutation
  invariant) and gamma_cross recorded
- config defaults (stack / no gauge hold) so production is unchanged
"""

import math

import torch

from src.torch.solver.mm_step import MMConfig
from src.torch.transport.grid_wasserstein import (
    bulk_grid_incidence,
    reduce_constraint_rows,
    relative_bulk_rows,
    row_space_discrepancy,
)
from src.torch.transport.phase_grid import (
    PhaseGridConfig,
    bulk_seeded_component_labels,
    cross_label_conductivity,
)
from src.torch.transport.weighted_poisson import (
    WeightedPoissonConfig,
    WeightedPoissonOperator,
)

DT = torch.float64


def _rowspace(mat):
    return reduce_constraint_rows(mat)[0]


def _same_rowspace(a, b, tol=1e-10):
    ra, rb = _rowspace(a), _rowspace(b)
    if ra.shape[0] != rb.shape[0]:
        return False
    return float((ra - (ra @ rb.T) @ rb).abs().max()) < tol


class TestRelativeRows:
    def test_projector_and_two_bulk(self):
        g = torch.Generator().manual_seed(0)
        C = torch.randn(2, 40, generator=g, dtype=DT)
        A = torch.tensor([[1.0, 1.0]], dtype=DT)          # G=1, B=2
        rel = relative_bulk_rows(C, A)
        assert torch.allclose(_rowspace(rel),
                              _rowspace((C[0] - C[1]).unsqueeze(0))
                              * torch.sign(
                                  (_rowspace(rel) @ _rowspace(
                                      (C[0] - C[1]).unsqueeze(0)).T)))
        assert reduce_constraint_rows(rel)[1] == 1
        # projector properties
        P = torch.eye(2, dtype=DT) - A.T @ torch.linalg.pinv(A @ A.T) @ A
        assert torch.allclose(P @ P, P)
        assert torch.allclose(P, P.T)
        assert float((P @ A.T).abs().max()) < 1e-14

    def test_general_incidence_rank_and_permutation(self):
        g = torch.Generator().manual_seed(1)
        C = torch.randn(4, 50, generator=g, dtype=DT)
        A = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=DT)  # G=2,B=4
        rel = relative_bulk_rows(C, A)
        assert reduce_constraint_rows(rel)[1] == 2               # B - G
        perm = torch.tensor([2, 0, 3, 1])
        rel_p = relative_bulk_rows(C[perm], A[:, perm])
        assert _same_rowspace(rel, rel_p)
        # containment: span{G, rel} subset of span{G, C}
        G = torch.randn(2, 50, generator=g, dtype=DT)
        big = _rowspace(torch.cat([G, C]))
        small = _rowspace(torch.cat([G, rel]))
        assert float((small - (small @ big.T) @ big).abs().max()) < 1e-10

    def test_bulk_grid_incidence(self):
        grid_pp = torch.tensor([0, 0, 0, 1, 1])
        bulk_pp = torch.tensor([0, 1, 1, 2, 2])
        A = bulk_grid_incidence(grid_pp, bulk_pp, 2, 3)
        assert torch.equal(A, torch.tensor([[1., 1., 0.],
                                            [0., 0., 1.]], dtype=DT))


class TestRowDiscrepancy:
    def test_scale_invariance_and_angle(self):
        g = torch.Generator().manual_seed(2)
        G = torch.randn(1, 30, generator=g, dtype=DT)
        assert row_space_discrepancy(G, 2.0 * G) < 1e-12
        assert row_space_discrepancy(G, -0.3 * G) < 1e-12
        C = torch.randn(1, 30, generator=g, dtype=DT)
        cos = float((G @ C.T) / (G.norm() * C.norm()))
        expect = math.sqrt(1 - cos * cos)
        assert abs(row_space_discrepancy(G, C) - expect) < 1e-10
        assert row_space_discrepancy(G, C) > 0.1


class TestGaugeHold:
    def _fixture(self, H=96, eps=0.12):
        cfg = PhaseGridConfig(grid_shape=(H, H), fill_epsilon=eps,
                              support_threshold=1e-3)
        h = 4.0 / H
        c = torch.arange(H, dtype=DT) * h - 2.0 + 0.5 * h
        X, Y = torch.meshgrid(c, c, indexing="ij")
        r = torch.sqrt(X * X + Y * Y)
        # bridged two-blob rho: disk r<0.5 and annulus 0.6<r<1.0 with
        # a soft corridor at 0.5<r<0.6 above threshold
        rho = torch.where(r < 0.5, 1.0, torch.where(
            r < 0.6, 0.02, torch.where(r < 1.0, 1.0, 0.0)))
        rho = rho.to(DT)
        n1, n2 = 40, 64
        t1 = torch.arange(n1, dtype=DT) * 2 * math.pi / n1
        t2 = torch.arange(n2, dtype=DT) * 2 * math.pi / n2
        pos = torch.cat([torch.stack([0.5 * t1.cos(), 0.5 * t1.sin()], 1),
                         torch.stack([0.6 * t2.cos(), 0.6 * t2.sin()], 1)])
        bulk_pp = torch.cat([torch.zeros(n1, dtype=torch.long),
                             torch.ones(n2, dtype=torch.long)])
        return cfg, rho, pos, bulk_pp

    def test_labels_certificate_and_gamma(self):
        cfg, rho, pos, bulk_pp = self._fixture()
        active = rho > cfg.support_threshold
        labels, cert = bulk_seeded_component_labels(active, cfg, pos,
                                                    bulk_pp, 2)
        assert bool(((labels >= 0) == active).all())      # partition
        assert cert["n_labels"] == 2
        # permutation invariance (co-membership)
        labels_p, _ = bulk_seeded_component_labels(
            active, cfg, pos, 1 - bulk_pp, 2)
        assert bool(((labels == 0) == (labels_p == 1)).all())
        pcfg = WeightedPoissonConfig(grid_shape=cfg.grid_shape,
                                     support_threshold=cfg.support_threshold)
        op = WeightedPoissonOperator(rho, pcfg, component_labels=labels)
        assert op.n_components == 2
        gc = cross_label_conductivity(op)
        assert 0.0 < gc < 0.05        # weak corridor: small cross share

    def test_config_defaults(self):
        cfg = MMConfig()
        # 0L-B0 verdict: State II production mode = aligned prequotient
        assert cfg.grid_contact_rows_mode == "aligned_prequotient"
        assert cfg.grid_gauge_hold_bulk_seeded is False
