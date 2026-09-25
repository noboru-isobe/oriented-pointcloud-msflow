"""0F pins: winding-signature bulk incidence + physical bulk volume
rows + the augmentation branch semantics.

Reviewer-mandated pins:
- 4-geometry incidence (annulus 1 / two disks 2 / disk-two-holes 1 /
  two annuli 2) with graph-scale stability (ell factor 3.5/4.0/4.5);
- coincident antiparallel sheets stay separate loops;
- fail-closed certificate (integer margin);
- permutation invariance as CO-MEMBERSHIP (never label arrays);
- finite-difference volume pin for bulk_rows (disk expansion, annulus
  outer/inner radial modes) and angle-only first-order volume == 0;
- policy asserts fail closed (visible measure / q velocity);
- branch semantics: partitions equal -> augmentation does NOT fire and
  the step is BITWISE the legacy path; bridged grid + 2 winding bulks
  -> fires with stacked rank 3 and exactly satisfied bulk rows.
"""

import math

import pytest
import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import compute_recommended_params
from src.torch.shapes.generator import (
    generate_oriented_annulus,
    generate_oriented_circle,
    generate_oriented_two_circles,
)
from src.torch.solver.mm_step import MMStepper
from src.torch.transport.boundary_flux_grid import (
    BoundaryFluxPolicies,
    BoundaryFluxToGrid,
)
from src.torch.transport.incidence import (
    oriented_graph_components,
    partition_relation,
    winding_bulk_labels,
)
from src.torch.transport.phase_grid import PhaseGridConfig

DT = torch.float64


def _cat(*vs):
    return OrientedPointCloudVarifold(
        positions=torch.cat([v.positions for v in vs]),
        angles=torch.cat([v.angles for v in vs]))


def _hole(n, r, center):
    """Boundary loop of a hole: circle with material-outward normals
    pointing INTO the hole (angles + pi)."""
    c = generate_oriented_circle(n, r, center, "cpu", DT)
    return OrientedPointCloudVarifold(positions=c.positions,
                                      angles=c.angles + math.pi)


def _masses_oracle(counts_radii):
    return torch.cat([torch.full((n,), 2 * math.pi * r / n, dtype=DT)
                      for n, r in counts_radii])


def _incidence(v, m, ell_factor=4.0):
    spacing = float(m.median())
    loops = oriented_graph_components(v.positions, v.normals,
                                      ell_factor * spacing)
    bulk, margin = winding_bulk_labels(v.positions, v.normals, m,
                                       loops, spacing)
    return loops, bulk, margin


FIXTURES = {}


def _fixture(name):
    if name in FIXTURES:
        return FIXTURES[name]
    if name == "annulus":
        v = generate_oriented_annulus(128, R_outer=1.0, R_inner=0.5,
                                      device="cpu", dtype=DT)
        n_bulk = 1
    elif name == "two_disks":
        v = generate_oriented_two_circles(
            96, radius1=0.5, center1=(-1.0, 0.0),
            radius2=0.5, center2=(1.0, 0.0), device="cpu", dtype=DT)
        n_bulk = 2
    elif name == "disk_two_holes":
        v = _cat(generate_oriented_circle(160, 1.0, (0.0, 0.0),
                                          "cpu", DT),
                 _hole(40, 0.25, (-0.5, 0.0)),
                 _hole(40, 0.25, (0.5, 0.0)))
        n_bulk = 1
    elif name == "two_annuli":
        a1 = generate_oriented_annulus(96, R_outer=0.8, R_inner=0.4,
                                       center=(-1.5, 0.0),
                                       device="cpu", dtype=DT)
        a2 = generate_oriented_annulus(96, R_outer=0.8, R_inner=0.4,
                                       center=(1.5, 0.0),
                                       device="cpu", dtype=DT)
        v = _cat(a1, a2)
        n_bulk = 2
    else:
        raise ValueError(name)
    delta, tau = compute_recommended_params(v.positions)
    from src.torch.oriented_varifold.mass import compute_masses
    m = compute_masses(v.positions, float(delta), float(tau))
    FIXTURES[name] = (v, m, n_bulk)
    return FIXTURES[name]


class TestWindingIncidence:
    @pytest.mark.parametrize("name,expected", [
        ("annulus", 1), ("two_disks", 2),
        ("disk_two_holes", 1), ("two_annuli", 2)])
    def test_four_geometry_pin(self, name, expected):
        v, m, _ = _fixture(name)
        loops, bulk, margin = _incidence(v, m)
        assert bulk is not None, f"certificate failed, margin={margin}"
        assert int(bulk.max()) + 1 == expected
        assert margin < 0.05  # measured <= 0.0301; EPS_WIND = 0.15

    @pytest.mark.parametrize("name", ["annulus", "two_disks",
                                      "disk_two_holes", "two_annuli"])
    def test_graph_scale_stability(self, name):
        """Loop partition invariant over ell/median(m) in
        {3.5, 4.0, 4.5} -- probe-depth stability alone does not
        detect a wrong loop partition (reviewer item 6)."""
        v, m, _ = _fixture(name)
        parts = []
        for f in (3.5, 4.0, 4.5):
            loops, _, _ = _incidence(v, m, ell_factor=f)
            parts.append(loops)
        assert partition_relation(parts[0], parts[1]) == "equal"
        assert partition_relation(parts[0], parts[2]) == "equal"

    def test_antiparallel_coincident_sheets_separate_loops(self):
        rbar, n1, n2 = 0.394, 90, 141
        th1 = 2 * math.pi * torch.arange(n1, dtype=DT) / n1
        th2 = 2 * math.pi * torch.arange(n2, dtype=DT) / n2
        pos = torch.cat([
            rbar * torch.stack([th1.cos(), th1.sin()], 1),
            rbar * torch.stack([th2.cos(), th2.sin()], 1)])
        ang = torch.cat([th1, th2 + math.pi])
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        m = torch.full((n1 + n2,), 2 * math.pi * rbar / n1, dtype=DT)
        loops = oriented_graph_components(v.positions, v.normals,
                                          4.0 * float(m.median()))
        wall_out = (v.positions * v.normals).sum(1) > 0
        # exactly two loops, split exactly by orientation
        assert int(loops.max()) + 1 == 2
        assert loops[wall_out].unique().numel() == 1
        assert loops[~wall_out].unique().numel() == 1
        assert int(loops[wall_out][0]) != int(loops[~wall_out][0])

    def test_integer_margin_fail_closed(self):
        v, m, _ = _fixture("annulus")
        spacing = float(m.median())
        loops = oriented_graph_components(v.positions, v.normals,
                                          4.0 * spacing)
        bulk, margin = winding_bulk_labels(
            v.positions, v.normals, m, loops, spacing,
            eps_wind=1e-12)
        assert bulk is None
        assert margin > 1e-12

    def test_permutation_invariance_co_membership(self):
        v, m, _ = _fixture("two_annuli")
        torch.manual_seed(0)
        perm = torch.randperm(v.positions.shape[0])
        vp = OrientedPointCloudVarifold(positions=v.positions[perm],
                                        angles=v.angles[perm])
        loops, bulk, _ = _incidence(v, m)
        loops_p, bulk_p, _ = _incidence(vp, m[perm])
        assert partition_relation(loops_p, loops[perm]) == "equal"
        assert partition_relation(bulk_p[loops_p],
                                  bulk[loops][perm]) == "equal"

    def test_partition_relation_label_renaming_invariant(self):
        a = torch.tensor([0, 0, 1, 1, 2, 2])
        b = torch.tensor([5, 5, 3, 3, 9, 9])       # renamed a
        c = torch.tensor([0, 0, 0, 0, 1, 1])       # coarser
        assert partition_relation(a, b) == "equal"
        assert partition_relation(a, c) == "a_finer"
        assert partition_relation(c, a) == "b_finer"
        d = torch.tensor([0, 1, 0, 1, 0, 1])       # crossing
        assert partition_relation(a, d) == "incomparable"


class TestBulkRowsGeometry:
    def _flux(self, v, m, policies=None):
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.06)
        q = torch.ones_like(m)
        return BoundaryFluxToGrid(v.positions, v.normals, m, q, cfg,
                                  policies or BoundaryFluxPolicies())

    def test_finite_difference_volume_pin(self):
        """(V_b(X + t s n) - V_b(X)) / t -> C_bulk_b(s, 0) for the
        three radial modes, against ANALYTIC circle areas (oracle
        arclength masses -> rows are exact circumferences)."""
        disk = generate_oriented_circle(96, 0.3, (1.4, 0.0), "cpu", DT)
        outer = generate_oriented_circle(160, 0.9, (-0.7, 0.0),
                                         "cpu", DT)
        hole = _hole(80, 0.45, (-0.7, 0.0))
        v = _cat(disk, outer, hole)
        m = _masses_oracle([(96, 0.3), (160, 0.9), (80, 0.45)])
        loops, bulk, _ = _incidence(v, m)
        assert bulk is not None and int(bulk.max()) + 1 == 2
        bulk_pp = bulk[loops]
        flux = self._flux(v, m)
        rows = flux.bulk_rows(bulk_pp, 2)
        N = v.positions.shape[0]

        # angle-only first-order volume change is exactly zero
        # (symmetric endpoint quadrature)
        assert float(rows[:, N:].abs().max()) == 0.0

        is_disk = torch.zeros(N, dtype=torch.bool)
        is_disk[:96] = True
        b_disk = int(bulk_pp[0])          # disk particles come first
        t = 1e-6
        cases = [
            # (mask of moved particles, bulk id, dV/dt analytic)
            (is_disk, b_disk, 2 * math.pi * 0.3),
            (torch.arange(N).ge(96) & torch.arange(N).lt(96 + 160),
             1 - b_disk, 2 * math.pi * 0.9),
            (torch.arange(N).ge(96 + 160), 1 - b_disk,
             2 * math.pi * 0.45),
        ]
        for mask, b, dv in cases:
            s = mask.to(DT)
            val = float(rows[b] @ torch.cat([s, torch.zeros(N,
                                                            dtype=DT)]))
            assert abs(val - dv) < 1e-10          # oracle masses: exact
            fd = (dv + math.pi * t * (1 if dv > 0 else -1))
            # analytic finite difference of pi*(R + t)^2 etc.
            assert abs(fd - val) < 4 * t + 1e-10
            other = float(rows[1 - b] @ torch.cat(
                [s, torch.zeros(N, dtype=DT)]))
            assert abs(other) == 0.0              # disjoint support

    def test_policy_asserts_fail_closed(self):
        disk = generate_oriented_circle(64, 0.3, (0.0, 0.0), "cpu", DT)
        m = _masses_oracle([(64, 0.3)])
        lbl = torch.zeros(64, dtype=torch.long)
        f_vis = self._flux(disk, m, BoundaryFluxPolicies(
            operator_measure="visible"))
        with pytest.raises(RuntimeError, match="carrier"):
            f_vis.bulk_rows(lbl, 1)
        f_qv = self._flux(disk, m, BoundaryFluxPolicies(
            use_coherence_velocity=True))
        with pytest.raises(RuntimeError, match="coherence"):
            f_qv.bulk_rows(lbl, 1)


def _grid_cfg(delta, tau, n=256, eps=0.06, mode="off",
              contact_rows=None):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent
                           / "scripts" / "experiments"))
    from p1_production_comparison import make_cfg

    from src.torch.transport.grid_wasserstein import GridMetricConfig
    cfg = make_cfg("C3", delta, tau, 1)
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(n, n), fill_epsilon=eps))
    cfg.grid_bulk_rows_mode = mode
    if contact_rows is not None:
        cfg.grid_contact_rows_mode = contact_rows
    cfg.optimizer_max_iter = 25
    return cfg


class TestAugmentationBranch:
    def test_partitions_equal_is_bitwise_legacy(self):
        """Separated two circles: grid and winding both read 2
        components -> the augment branch must NOT fire and the step is
        bitwise the mode='off' step."""
        v = generate_oriented_two_circles(
            128, radius1=0.5, center1=(-1.0, 0.0),
            radius2=0.5, center2=(1.0, 0.0), device="cpu", dtype=DT)
        delta, tau = compute_recommended_params(v.positions)
        res = {}
        for mode in ("off", "augment"):
            st = MMStepper(_grid_cfg(delta, tau, mode=mode))
            res[mode] = st.step(v)
            sn = st._last_grid_snapshot
            if mode == "augment":
                assert sn.augmentation_fired is False
                assert sn.partition_relation_grid_bulk == "equal"
                assert sn.n_bulk_components == 2
        assert torch.equal(res["off"].displacements,
                           res["augment"].displacements)
        assert torch.equal(res["off"].delta_angles,
                           res["augment"].delta_angles)
        assert res["augment"].bulk_flux_residuals is None

    def test_bridged_grid_fires_augmentation(self):
        """Two circles with a gap below the mollification scale: the
        grid support bridges (1 component) while winding still reads 2
        bulks -> augmentation fires, stacked rank 3 (the grid row is
        generally independent of the bulk rows: w_act != 1), and the
        realized step satisfies the PHYSICAL bulk rows exactly."""
        v = generate_oriented_two_circles(
            128, radius1=0.5, center1=(-0.56, 0.0),
            radius2=0.5, center2=(0.56, 0.0), device="cpu", dtype=DT)
        delta, tau = compute_recommended_params(v.positions)
        st_off = MMStepper(_grid_cfg(delta, tau, eps=0.12, mode="off"))
        st_off._setup_step(v)
        assert st_off._last_grid_snapshot.n_components == 1, \
            "fixture must actually bridge at this fill_epsilon"

        # legacy rank-3 stack (0L-B0 cell A; diagnostic since B0)
        st = MMStepper(_grid_cfg(delta, tau, eps=0.12, mode="augment",
                                 contact_rows="stack"))
        res = st.step(v)
        sn = st._last_grid_snapshot
        assert sn.augmentation_fired is True
        assert sn.partition_relation_grid_bulk == "b_finer"
        assert sn.n_bulk_components == 2
        assert sn.r_grid == 1 and sn.r_bulk == 2
        assert sn.r_stack in (2, 3)
        assert sn.delta_compat is not None
        assert sn.n_params == v.positions.shape[0] - sn.r_stack
        g = torch.tensor(res.bulk_flux_residuals, dtype=DT)
        scale = float((st.fixed_masses
                       * res.displacements.abs()).sum()) + 1e-30
        assert float(g.abs().max()) < 1e-10 * max(scale, 1e-12)
        # 0L-B0 State II production default: aligned prequotient
        # rows {G, P_rel C_bulk} -- rank exactly 2, the RELATIVE bulk
        # row is satisfied exactly, the raw bulk rows only up to the
        # grid/raw common-mode quadrature discrepancy (shadow)
        st2 = MMStepper(_grid_cfg(delta, tau, eps=0.12, mode="augment"))
        assert st2.config.grid_contact_rows_mode == "aligned_prequotient"
        res2 = st2.step(v)
        sn2 = st2._last_grid_snapshot
        assert sn2.partition_relation_grid_bulk == "b_finer"
        assert sn2.r_stack == 2
        assert sn2.eps_row is not None and 0.0 <= sn2.eps_row < 1.0
        rel = torch.tensor(res2.bulk_flux_residuals, dtype=DT)
        assert float(rel.abs().max()) < 1e-10 * max(scale, 1e-12)
        assert res2.bulk_flux_residuals_shadow is not None
        assert res2.contact_rows_telemetry is not None
        assert "C_rel" in res2.contact_rows_telemetry
