"""MMStepper integration of metric_backend="bie".

- constructor guards (compile / spectral_bordered / grid-only arms /
  non-carrier measure);
- component-count pins from the winding components: disk 1 /
  annulus 1 / two circles 2 / annulus+disk 2, n_params == N - C;
- one-step optimizer parity vs the production BEM C3 variant on a
  smooth ellipse (< 2%, measured 0.09%) and vs the grid backend;
- realized step exactly flux-free per metric component;
- the merge step: two ellipses inside the bridging gap give the
  b_finer partition relation, a contact-pair context, the masked
  particles held fixed, and the definiteness check passing;
- concentric contact: the fully masked pair leaves the outer loop
  alone with ONE effective constraint row (the frozen-dependent
  relative area row is dropped, not counted as rank deficiency).
"""

import math
import sys
from pathlib import Path

import pytest
import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import compute_recommended_params
from src.torch.shapes.generator import (
    generate_oriented_annulus,
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_two_circles,
    generate_oriented_two_ellipses,
)
from src.torch.solver.mm_step import MMStepper
from src.torch.transport.bie_wasserstein import BIEMetricConfig
from src.torch.transport.grid_wasserstein import GridMetricConfig
from src.torch.transport.phase_grid import PhaseGridConfig

sys.path.insert(0, str(Path(__file__).parent.parent
                       / "scripts" / "experiments"))
from p1_production_comparison import make_cfg  # noqa: E402

DT = torch.float64


def _bie_cfg(delta, tau, **bie):
    cfg = make_cfg("C3", delta, tau, 1)
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "bie"
    cfg.grid_bulk_rows_mode = "augment"
    cfg.grid_contact_rows_mode = "aligned_prequotient"
    cfg.angle_constraint_scope = "loopwise"
    cfg.mass_estimator = "loopwise_oriented_kde"
    cfg.bie_metric = BIEMetricConfig(**bie)
    return cfg


def _grid_cfg(delta, tau, n=256, eps=0.06):
    cfg = make_cfg("C3", delta, tau, 1)
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(n, n), fill_epsilon=eps))
    return cfg


def _setup(v, **bie):
    delta, tau = compute_recommended_params(v.positions)
    st = MMStepper(_bie_cfg(delta, tau, **bie))
    st._setup_step(v)
    return st


class TestConstructorGuards:
    def _base(self):
        v = generate_oriented_circle(64, 1.0, (0.0, 0.0), "cpu", DT)
        delta, tau = compute_recommended_params(v.positions)
        return _bie_cfg(delta, tau)

    def test_rejects_compile(self):
        cfg = self._base()
        cfg.compile = True
        with pytest.raises(NotImplementedError):
            MMStepper(cfg)

    def test_rejects_spectral(self):
        cfg = self._base()
        cfg.bem_solver_mode = "spectral_bordered"
        with pytest.raises(NotImplementedError):
            MMStepper(cfg)

    @pytest.mark.parametrize("field,value", [
        ("grid_bulk_rows_mode", "off"),
        ("bem_operator_measure", "visible"),
        ("wlin_use_coherence_velocity", True),
        ("grid_vertical_dof", True),
        ("grid_fossil_advection", True),
        ("grid_carrier_amplitude_fade", True),
        ("grid_phase_support_projection", True),
    ])
    def test_rejects_grid_only_arms(self, field, value):
        cfg = self._base()
        setattr(cfg, field, value)
        with pytest.raises(ValueError, match="bie"):
            MMStepper(cfg)

    def test_metric_object_wiring(self):
        st = MMStepper(self._base())
        assert st.grid_wasserstein is None
        assert st.metric_object is st.bie_wasserstein
        assert st.wasserstein_metric is st.bie_wasserstein


class TestComponentCountPins:
    def _pin(self, v, C):
        st = _setup(v)
        sn = st._last_grid_snapshot
        assert sn.metric_backend == "bie"
        assert sn.n_components == C
        assert sn.constraint_rank == C
        assert sn.n_params == v.positions.shape[0] - C
        assert st.param.Q.shape == (v.positions.shape[0], sn.n_params)
        assert sn.bie_n_masked == 0 and not sn.bie_merged_pairs
        assert min(sn.bie_sigma_tail) > 1e-3
        assert sn.bie_lambda_min is not None and sn.bie_lambda_min > 0
        assert sn.fill_epsilon is None and sn.grid_shape is None

    def test_disk(self):
        self._pin(generate_oriented_circle(256, 1.0, (0.0, 0.0),
                                           "cpu", DT), 1)

    def test_annulus(self):
        self._pin(generate_oriented_annulus(
            256, R_outer=1.0, R_inner=0.5, device="cpu", dtype=DT), 1)

    def test_two_circles(self):
        self._pin(generate_oriented_two_circles(
            256, radius1=0.8, center1=(-1.0, 0.0),
            radius2=0.8, center2=(1.0, 0.0), device="cpu", dtype=DT), 2)

    def test_annulus_plus_disk(self):
        va = generate_oriented_annulus(
            256, R_outer=0.9, R_inner=0.45, center=(-0.7, 0.0),
            device="cpu", dtype=DT)
        vd = generate_oriented_circle(96, 0.3, (1.4, 0.0), "cpu", DT)
        v = OrientedPointCloudVarifold(
            positions=torch.cat([va.positions, vd.positions]),
            angles=torch.cat([va.angles, vd.angles]))
        self._pin(v, 2)


class TestOneStepParity:
    def test_ellipse_one_step(self):
        v = generate_oriented_ellipse(256, a=1.2, b=0.8, device="cpu",
                                      dtype=DT)
        delta, tau = compute_recommended_params(v.positions)
        r_bem = MMStepper(make_cfg("C3", delta, tau, 1)).step(v)
        cfg = _bie_cfg(delta, tau)
        cfg.mass_estimator = "kde"          # same estimator as C3
        cfg.angle_constraint_scope = "union"
        st = MMStepper(cfg)
        r = st.step(v)
        assert r_bem.objective_decreased and r.objective_decreased
        assert r.relative_gradient_norm < 1e-5
        rel_s = float((r.displacements - r_bem.displacements).norm()
                      / r_bem.displacements.norm())
        assert rel_s < 0.02, f"one-step velocity parity {rel_s:.4f}"
        r_grid = MMStepper(_grid_cfg(delta, tau)).step(v)
        rel_g = float((r.displacements - r_grid.displacements).norm()
                      / r_grid.displacements.norm())
        assert rel_g < 0.03, rel_g
        # realized step exactly flux-free per metric component
        rows = st.bie_wasserstein.component_rows()
        vec = torch.cat([r.displacements, r.delta_angles])
        flux = rows @ vec
        scale = rows.abs() @ vec.abs()
        assert float(flux.abs().max()) < 1e-13 * float(scale.max())
        assert r.grid_setup_snapshot is not None
        assert r.grid_step_stats is not None
        assert r.grid_step_stats.n_forward_solves > 0
        assert r.bem_setup_snapshot is None


def _two_ellipses_at(gap_over_ell, n_el=128):
    v0 = generate_oriented_two_ellipses(
        n_el, a1=0.4, b1=1.0, center1=(-1.0, 0.0),
        a2=0.4, b2=1.0, center2=(1.0, 0.0), device="cpu", dtype=DT)
    seg = (v0.positions[:n_el].roll(-1, 0) - v0.positions[:n_el]) \
        .norm(dim=1)
    ell = float(seg.median())
    gap = gap_over_ell * ell
    return generate_oriented_two_ellipses(
        n_el, a1=0.4, b1=1.0, center1=(-(0.4 + gap / 2), 0.0),
        a2=0.4, b2=1.0, center2=((0.4 + gap / 2), 0.0),
        device="cpu", dtype=DT), ell


class TestMergeStep:
    def test_separated_then_merged(self):
        v, ell = _two_ellipses_at(1.5)
        st = _setup(v)
        sn = st._last_grid_snapshot
        assert sn.n_components == 2 and sn.partition_relation_grid_bulk \
            == "equal"
        ctx = st._contact_pair_context
        assert ctx["relation"] == "equal"

        v, ell = _two_ellipses_at(0.8)
        delta, tau = compute_recommended_params(v.positions)
        st = MMStepper(_bie_cfg(delta, tau))
        r = st.step(v)
        sn = st._last_grid_snapshot
        assert sn.n_components == 1
        assert sn.partition_relation_grid_bulk == "b_finer"
        assert sn.augmentation_fired
        assert len(sn.bie_merged_pairs) == 1
        assert sn.bie_n_masked >= 4 and sn.n_frozen_dof == sn.bie_n_masked
        assert sn.bie_lambda_min is not None and sn.bie_lambda_min > 0
        ctx = st._contact_pair_context
        assert ctx["relation"] == "b_finer"
        assert set(ctx["loop_to_grid_component"].values()) == {0}
        assert set(ctx["loop_to_raw_bulk"].values()) == {0, 1}
        # the certificate machinery saw the candidate pair
        assert st._contact_certificates_source is not None
        # masked particles held fixed, the rest moved
        masked = st.bie_wasserstein.masked
        assert float(r.displacements[masked].abs().max()) \
            < 1e-15 * float(r.displacements.abs().max())
        assert float(r.displacements[~masked].abs().max()) > 0.0
        assert r.objective_decreased
        # both raw bulk rows (relative form) satisfied to rounding
        assert r.bulk_flux_residuals is not None
        assert max(abs(x) for x in r.bulk_flux_residuals) < 1e-12

    def test_concentric_full_mask(self):
        ell = 2 * math.pi / 384
        gap = 0.9 * ell
        va = generate_oriented_annulus(384, 1.0, 0.5, (0.0, 0.0),
                                       "cpu", DT)
        vd = generate_oriented_circle(int(round(128 * (0.5 - gap) / 0.5)),
                                      0.5 - gap, (0.0, 0.0), "cpu", DT)
        v = OrientedPointCloudVarifold(
            positions=torch.cat([va.positions, vd.positions]),
            angles=torch.cat([va.angles, vd.angles]))
        st = _setup(v)
        sn = st._last_grid_snapshot
        outer = v.positions.norm(dim=1) > 0.75
        assert sn.n_components == 1
        assert sn.bie_n_masked == int((~outer).sum())
        # G and the relative row coincide modulo the frozen unit rows
        assert sn.constraint_rank == 1
        assert sn.n_params == int(outer.sum()) - 1
        assert sn.bie_block_sizes == [3 * int(outer.sum())]
