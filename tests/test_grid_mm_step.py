"""Grid metric G2b: MMStepper integration with metric_backend="grid_poisson".

Self-verifying gates (user directive: the tests must catch wiring errors
without per-item review):
- constructor guards (compile / spectral_bordered / config type);
- the composed constraint basis (grid compatibility rows through the
  angle map and coherence scaling) produces EXACTLY component-flux-free
  directions, and the fail-closed forward accepts them;
- NEGATIVE CONTROL: the legacy q^2 m volume complement -- admissible for
  the BEM constraint -- is REJECTED by the grid forward solve (the two
  discretizations of the volume constraint differ at O(h^2), plan item
  "constraint double discretization": measured, not silently projected);
- one-step optimizer parity vs the production BEM C3 variant on a
  smooth resolved ellipse (pre-registered go/no-go: velocity difference
  < 2%), with the realized step exactly volume-conserving per component;
- component-count pins from the grid support: disk 1 / annulus 1 /
  two circles 2 / annulus+disk 2 -- the BIE rank machinery replacement;
- grid mode never touches the BEM cache (no hidden BIE dependency).
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
)
from src.torch.solver.mm_step import MMConfig, MMStepper
from src.torch.transport.bem_wasserstein import _orthogonal_complement
from src.torch.transport.grid_wasserstein import (
    GridMetricConfig,
    compose_constraint_basis,
)
from src.torch.transport.phase_grid import PhaseGridConfig
from src.torch.transport.weighted_poisson import (
    IncompatibleGridVelocityError,
)

sys.path.insert(0, str(Path(__file__).parent.parent
                       / "scripts" / "experiments"))
from p1_production_comparison import make_cfg  # noqa: E402

DT = torch.float64


def _grid_cfg(delta, tau, n=256, eps=0.06):
    """Production-comparison scales (C3 conventions) with the grid
    backend replacing both the BIE and the spectral constraint
    machinery. n/eps respect the calibrated laws for the fixtures
    below: h/eps <= 0.5 (porous wall), dx <= eps/2."""
    cfg = make_cfg("C3", delta, tau, 1)
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(n, n), fill_epsilon=eps))
    return cfg


def _setup(v, n=256, eps=0.06):
    delta, tau = compute_recommended_params(v.positions)
    st = MMStepper(_grid_cfg(delta, tau, n=n, eps=eps))
    st._setup_step(v)
    return st


class TestConstructorGuards:
    def test_compile_rejected(self):
        cfg = MMConfig(metric_backend="grid_poisson", compile=True)
        with pytest.raises(NotImplementedError, match="compile"):
            MMStepper(cfg)

    def test_spectral_bordered_rejected(self):
        cfg = MMConfig(metric_backend="grid_poisson",
                       bem_solver_mode="spectral_bordered")
        with pytest.raises(NotImplementedError, match="spectral|legacy"):
            MMStepper(cfg)

    def test_wrong_grid_config_type_rejected(self):
        cfg = MMConfig(metric_backend="grid_poisson",
                       grid_metric={"grid_shape": (64, 64)})
        with pytest.raises(TypeError, match="GridMetricConfig"):
            MMStepper(cfg)

    def test_default_backend_has_no_grid_object(self):
        st = MMStepper(MMConfig())
        assert st.grid_wasserstein is None
        assert st.wasserstein_metric is st.bem_wasserstein


class TestComposedConstraintBasis:
    def test_basis_admissible_orthonormal_and_accepted(self):
        v = generate_oriented_circle(256, 1.0, (0.0, 0.0), "cpu", DT)
        st = _setup(v)
        param, gm = st.param, st.grid_wasserstein
        N = 256

        # orthonormal, correct dimension N - C
        Q = param.Q
        assert Q.shape == (N, N - 1)
        eye = torch.eye(N - 1, dtype=DT)
        assert float((Q.T @ Q - eye).abs().max()) < 1e-12

        # every parameter direction gives exactly zero component flux
        rows = gm.flux.component_rows(gm.phase.component_labels,
                                      gm.phase.cell_area)
        torch.manual_seed(0)
        y = torch.randn(N - 1, dtype=DT)
        s, th = param.unpack_params(y)
        flux = rows @ torch.cat([s, th])
        scale = rows.abs() @ torch.cat([s, th]).abs()
        assert float(flux.abs().max()) < 1e-14 * float(scale.max())

        # ...and the fail-closed forward ACCEPTS it (finite objective)
        w = gm(s, th, st.config.time_step)
        assert torch.isfinite(w) and float(w) >= 0.0

    def test_negative_control_legacy_basis_rejected(self):
        """The q^2 m volume complement is BEM-admissible but NOT
        grid-admissible: the O(h^2) mismatch between the two volume
        discretizations must trip the componentwise compatibility
        rejection -- this is the gate that would catch a wrongly
        composed basis."""
        v = generate_oriented_circle(256, 1.0, (0.0, 0.0), "cpu", DT)
        st = _setup(v)
        param, gm = st.param, st.grid_wasserstein
        w_legacy = st.fixed_coherence ** 2 * st.fixed_masses
        Q_legacy = _orthogonal_complement(w_legacy)

        torch.manual_seed(1)
        y = torch.randn(Q_legacy.shape[1], dtype=DT)
        u = Q_legacy @ y
        q = st.fixed_coherence
        s = q * u if st.config.displacement_q_suppress else u
        th = param.AB_solve @ s
        if st.config.displacement_q_suppress:
            th = th * q
        with pytest.raises(IncompatibleGridVelocityError):
            gm(s, th, st.config.time_step)

    def test_rank_deficient_rows_refused(self):
        rows = torch.zeros(1, 8, dtype=DT)  # degenerate constraint
        AB = torch.zeros(4, 4, dtype=DT)
        q = torch.ones(4, dtype=DT)
        with pytest.raises(RuntimeError, match="rank"):
            compose_constraint_basis(rows, AB, q, q_suppress=False)


class TestComponentCountPins:
    """Grid support components replace the BIE rank machinery: the
    counts must read correctly from the phase, and the admissible
    space must lose exactly C dimensions."""

    def _pin(self, v, C, n=256, eps=0.06):
        st = _setup(v, n=n, eps=eps)
        sn = st._last_grid_snapshot
        assert sn.n_components == C
        assert sn.constraint_rank == C
        assert sn.n_params == v.positions.shape[0] - C
        assert st.param.Q.shape == (v.positions.shape[0], sn.n_params)

    def test_disk(self):
        self._pin(generate_oriented_circle(256, 1.0, (0.0, 0.0),
                                           "cpu", DT), 1)

    def test_annulus(self):
        # one phase component with two boundary loops -- the case the
        # BIE boundary-count heuristic gets structurally wrong
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


class TestGridModeIsolation:
    def test_bem_cache_untouched_and_snapshot_recorded(self):
        v = generate_oriented_circle(256, 1.0, (0.0, 0.0), "cpu", DT)
        st = _setup(v)
        # no hidden BIE dependency: the BEM object was never set up
        assert st.bem_wasserstein.setup_snapshot_fields() is None
        sn = st._last_grid_snapshot
        assert sn is not None
        assert sn.grid_shape == (256, 256)
        assert sn.operator_measure_policy == "carrier"
        assert sn.velocity_policy == "unit"
        # divergence-theorem volume target ~ pi (KDE mass quadrature)
        assert abs(sn.target_volume_current - math.pi) < 0.05
        assert sn.target_volume_used == sn.target_volume_current
        assert sn.reconstruction_volume_error < 1e-8


class TestQuadraticPathsAndBackends:
    """G3a: the analytic quadratic and the sparse_direct backend must
    reproduce the generic/native reference exactly (value, gradient,
    HVP), at the promised solve counts."""

    def _stepper(self, path, backend="native_pcg"):
        # backend passed EXPLICITLY: the GridMetricConfig default is
        # sparse_direct (G3a adoption), but this test compares the
        # PCG reference against it, so both must be pinned.
        from dataclasses import replace
        v = generate_oriented_circle(96, 1.0, (0.0, 0.0), "cpu", DT)
        delta, tau = compute_recommended_params(v.positions)
        cfg = make_cfg("C3", delta, tau, 1)
        cfg.bem_solver_mode = "legacy_global_projection"
        cfg.metric_backend = "grid_poisson"
        gm = GridMetricConfig(phase=PhaseGridConfig(
            grid_shape=(128, 128), fill_epsilon=0.13),
            quadratic_path=path)
        gm.poisson = replace(gm.poisson, solver_backend=backend)
        cfg.grid_metric = gm
        st = MMStepper(cfg)
        st._setup_step(v)
        return st

    def _wgh(self, st, y, p):
        yy = y.clone().requires_grad_(True)
        s, th = st.param.unpack_params(yy)
        w = st.wasserstein_metric(s, th, st.config.time_step)
        g = torch.autograd.grad(w, yy, create_graph=True)[0]
        hvp = torch.autograd.grad((g * p).sum(), yy)[0]
        return float(w), g.detach(), hvp.detach()

    def test_parity_and_solve_counts(self):
        torch.manual_seed(0)
        sa = self._stepper("analytic")
        sg = self._stepper("generic")
        sd = self._stepper("analytic", "sparse_direct")
        y = 1e-3 * torch.randn(95, dtype=DT)
        p = torch.randn(95, dtype=DT)
        wa, ga, ha = self._wgh(sa, y, p)
        wg, gg, hg = self._wgh(sg, y, p)
        wd, gd, hd = self._wgh(sd, y, p)

        assert abs(wa - wg) <= 1e-12 * abs(wg)
        assert float((ga - gg).norm()) <= 1e-9 * float(gg.norm())
        assert float((ha - hg).norm()) <= 1e-9 * float(hg.norm())
        assert abs(wd - wg) <= 1e-10 * abs(wg)
        assert float((gd - gg).norm()) <= 1e-8 * float(gg.norm())
        assert float((hd - hg).norm()) <= 1e-8 * float(hg.norm())

        # analytic: objective+gradient = ONE forward solve, the HVP =
        # ONE projected solve. generic re-solves in the first backward
        # (1 fwd + 3 proj for the same computation).
        st_a = sa.grid_wasserstein.poisson.stats
        assert (st_a.n_forward_solves, st_a.n_projected_solves) == (1, 1)
        st_g = sg.grid_wasserstein.poisson.stats
        assert (st_g.n_forward_solves, st_g.n_projected_solves) == (1, 3)
        # direct solves report 0 PCG iterations, residual at the same
        # acceptance criterion as PCG
        st_d = sd.grid_wasserstein.poisson.stats
        assert st_d.pcg_iterations == [0, 0]
        assert st_d.max_relative_residual < 1e-10


class TestVelocityScaling:
    """G3a: optimizer_variable='velocity' minimizes F(dt v)/dt -- the
    minimizer must be the SAME physical step (promotion rule
    |s - s_ref|/(1+|s_ref|) < 1e-5), with all reported diagnostics in
    y-space."""

    def test_velocity_matches_displacement(self):
        from dataclasses import replace
        v = generate_oriented_circle(96, 1.0, (0.0, 0.0), "cpu", DT)
        delta, tau = compute_recommended_params(v.positions)

        def run(variable):
            cfg = make_cfg("C3", delta, tau, 1)
            cfg.bem_solver_mode = "legacy_global_projection"
            cfg.metric_backend = "grid_poisson"
            cfg.optimizer_variable = variable
            gm = GridMetricConfig(phase=PhaseGridConfig(
                grid_shape=(128, 128), fill_epsilon=0.13))
            gm.poisson = replace(gm.poisson,
                                 solver_backend="sparse_direct")
            cfg.grid_metric = gm
            return MMStepper(cfg).step(v)

        rd = run("displacement")
        rv = run("velocity")
        ds = float((rv.displacements - rd.displacements).norm()) \
            / (1.0 + float(rd.displacements.norm()))
        assert ds < 1e-5, f"velocity/displacement minimizer drift {ds:.2e}"
        dF = abs(rv.objective - rd.objective) \
            / (1.0 + abs(rd.objective))
        assert dF < 1e-8
        assert rv.objective_decreased
        # diagnostics are y-space: same objective scale, comparable
        # step norm (not the 1/dt-scaled optimizer variable)
        assert abs(rv.step_norm - rd.step_norm) \
            < 1e-4 * max(1.0, rd.step_norm)


class TestOneStepParity:
    """Pre-registered go/no-go: smooth resolved case, one full
    trust-ncg step, velocity difference vs the production BEM C3
    variant < 2%. Measured 2026-08-06 (N=256 ellipse, 256^2 grid,
    eps 0.06): rel_s = 1.85%, both converge in 5 iters."""

    def test_ellipse_one_step(self):
        v = generate_oriented_ellipse(256, a=1.2, b=0.8, device="cpu",
                                      dtype=DT)
        delta, tau = compute_recommended_params(v.positions)

        r_bem = MMStepper(make_cfg("C3", delta, tau, 1)).step(v)
        st_g = MMStepper(_grid_cfg(delta, tau))
        r_grid = st_g.step(v)

        assert r_bem.objective_decreased and r_grid.objective_decreased
        assert r_grid.relative_gradient_norm < 1e-5

        rel_s = float((r_grid.displacements - r_bem.displacements).norm()
                      / r_bem.displacements.norm())
        assert rel_s < 0.02, f"one-step velocity parity {rel_s:.4f}"

        # realized step is EXACTLY component-flux-free
        gm = st_g.grid_wasserstein
        rows = gm.flux.component_rows(gm.phase.component_labels,
                                      gm.phase.cell_area)
        vec = torch.cat([r_grid.displacements, r_grid.delta_angles])
        flux = rows @ vec
        scale = rows.abs() @ vec.abs()
        assert float(flux.abs().max()) < 1e-14 * float(scale.max())

        # telemetry rides on the result
        assert r_grid.grid_setup_snapshot is not None
        assert r_grid.grid_setup_snapshot.n_components == 1
        assert r_grid.bem_setup_snapshot is None
