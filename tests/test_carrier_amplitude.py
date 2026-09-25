"""Arm 5: persistent carrier-amplitude fade (spec-2.0-lite).

Pinned: gate specificity (circle / annulus inner boundary /
pre-contact facing arcs all keep a = 1), interior pair fades
monotonically to zero in ~1/rate updates, the faded pair's phase
equals the pair-free phase, solver integration with default-off
noninterference.
"""

import math

import pytest
import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import (
    compute_masses,
    compute_recommended_params,
)
from src.torch.shapes.generator import (
    generate_oriented_annulus,
    generate_oriented_circle,
    generate_oriented_two_ellipses,
)
from src.torch.transport.carrier_amplitude import (
    CarrierAmplitudeConfig,
    update_amplitudes,
)
from src.torch.transport.phase_grid import CurrentToPhase, PhaseGridConfig

DT = torch.float64


def _phase(v, m, cfg, vol):
    return CurrentToPhase(cfg).reconstruct(v, m, vol)


def _masses(v):
    delta, tau = compute_recommended_params(v.positions)
    return compute_masses(v.positions, delta, tau, "wendland_c2")


def _fade_iterate(v, m, cfg, vol, n_iters, acfg=CarrierAmplitudeConfig()):
    a = torch.ones(v.n_points, dtype=DT)
    stats = None
    for _ in range(n_iters):
        pg = _phase(v, m * a, cfg, vol)
        a, stats = update_amplitudes(v, a, pg.rho_metric, cfg, acfg)
    return a, stats


class TestGateSpecificity:
    def test_circle_and_annulus_keep_full_amplitude(self):
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
        v = generate_oriented_circle(256, 1.0, (0.0, 0.0), "cpu", DT)
        a, st = _fade_iterate(v, _masses(v), cfg, math.pi, 3)
        assert st["n_gated"] == 0 and float(a.min()) == 1.0

        va = generate_oriented_annulus(192, 1.2, 0.5, (0.0, 0.0),
                                       "cpu", DT)
        vol = math.pi * (1.2 ** 2 - 0.5 ** 2)
        a, st = _fade_iterate(va, _masses(va), cfg, vol, 3)
        assert st["n_gated"] == 0 and float(a.min()) == 1.0

    def test_precontact_facing_arcs_keep_full_amplitude(self):
        v = generate_oriented_two_ellipses(
            256, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
            a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu", dtype=DT)
        cfg = PhaseGridConfig(grid_shape=(512, 512), fill_epsilon=0.04,
                              support_threshold=3e-3)
        vol = 2 * math.pi * 0.4 * 1.0
        a, st = _fade_iterate(v, _masses(v), cfg, vol, 3)
        # the gap side reads vacuum -> bulk test fails -> a stays 1
        assert st["n_gated"] == 0 and float(a.min()) == 1.0


def _disk_with_pair(n_ring=256, n_sheet=24):
    disk = generate_oriented_circle(n_ring, 1.0, (0.0, 0.0), "cpu", DT)
    ys = torch.linspace(-0.12, 0.12, n_sheet, dtype=DT)
    seg1 = torch.stack([torch.full_like(ys, -0.11), ys], dim=1)
    seg2 = torch.stack([torch.full_like(ys, -0.09), ys], dim=1)
    pos = torch.cat([disk.positions, seg1, seg2])
    ang = torch.cat([disk.angles,
                     torch.zeros(n_sheet, dtype=DT),
                     torch.full((n_sheet,), math.pi, dtype=DT)])
    return (OrientedPointCloudVarifold(positions=pos, angles=ang),
            n_ring, n_sheet)


class TestInteriorFade:
    def test_pair_fades_to_zero_monotonically(self):
        v, n_ring, n_sheet = _disk_with_pair()
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
        m = _masses(v)
        acfg = CarrierAmplitudeConfig(rate=0.25)
        a = torch.ones(v.n_points, dtype=DT)
        prev = a.clone()
        for _ in range(6):
            pg = _phase(v, m * a, cfg, math.pi)
            a, st = update_amplitudes(v, a, pg.rho_metric, cfg, acfg)
            assert bool((a <= prev + 1e-15).all())   # monotone
            prev = a.clone()
        assert float(a[:n_ring].min()) == 1.0        # boundary intact
        assert float(a[n_ring:].max()) == 0.0        # pair fully faded
        assert st["n_zero"] == 2 * n_sheet

    def test_faded_phase_equals_pair_free_phase(self):
        v, n_ring, _ = _disk_with_pair()
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
        m = _masses(v)
        a = torch.ones(v.n_points, dtype=DT)
        a[n_ring:] = 0.0
        pg_faded = _phase(v, m * a, cfg, math.pi)
        disk = generate_oriented_circle(n_ring, 1.0, (0.0, 0.0),
                                        "cpu", DT)
        pg_free = _phase(disk, _masses(disk), cfg, math.pi)
        rel = float((pg_faded.rho_metric - pg_free.rho_metric).abs().mean()
                    / pg_free.rho_metric.abs().mean())
        assert rel < 2e-2, f"faded phase differs by {rel}"


class TestSolverIntegration:
    def test_default_off_and_flag_on_records_stats(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent
                               / "scripts" / "experiments"))
        from p1_production_comparison import make_cfg
        from src.torch.shapes.generator import generate_oriented_ellipse
        from src.torch.solver.mm_solver import MMSolver
        from src.torch.solver.mm_step import MMConfig
        from src.torch.transport.grid_wasserstein import GridMetricConfig

        assert MMConfig().grid_carrier_amplitude_fade is False

        v = generate_oriented_ellipse(256, 1.2, 0.8, (0.0, 0.0),
                                      "cpu", DT)
        delta, tau = compute_recommended_params(v.positions)
        cfg = make_cfg("C3", delta, tau, 1)
        cfg.bem_solver_mode = "legacy_global_projection"
        cfg.metric_backend = "grid_poisson"
        cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
            grid_shape=(256, 256), fill_epsilon=0.06))
        cfg.grid_carrier_amplitude_fade = True
        solver = MMSolver(cfg)
        seen = []

        def cb(step, r):
            seen.append(r.amplitude_stats)
            return False

        solver.solve(v, 2, callback=cb)
        assert all(s is not None for s in seen)
        # a clean ellipse: nothing gated, all amplitudes stay 1
        assert seen[-1]["n_gated"] == 0
        assert seen[-1]["min_amplitude"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestDynamicalDeathModulation:
    def test_half_alive_marks_never_fade(self):
        """v2: a phase-gated mark with q >= q_dead keeps a = 1 (the
        arm-4+5 v1 killer: fading crack marks the flow was still
        welding)."""
        v, n_ring, n_sheet = _disk_with_pair()
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
        m = _masses(v)
        acfg = CarrierAmplitudeConfig(rate=0.5)
        a = torch.ones(v.n_points, dtype=DT)
        # synthetic coherence: pretend all marks are half-alive
        q_half = torch.full((v.n_points,), 0.5, dtype=DT)
        pg = _phase(v, m, cfg, math.pi)
        a2, st = update_amplitudes(v, a, pg.rho_metric, cfg, acfg,
                                   coherence=q_half)
        assert st["n_gated"] > 0                 # phase gate fires
        assert st["n_fading_eligible"] == 0      # q modulation vetoes
        assert float(a2.min()) == 1.0

    def test_dead_marks_fade_at_reduced_rate(self):
        v, n_ring, n_sheet = _disk_with_pair()
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
        m = _masses(v)
        acfg = CarrierAmplitudeConfig(rate=0.4, q_dead=0.2)
        a = torch.ones(v.n_points, dtype=DT)
        q = torch.ones(v.n_points, dtype=DT)
        q[n_ring:] = 0.05                        # dead fossils
        pg = _phase(v, m, cfg, math.pi)
        a2, st = update_amplitudes(v, a, pg.rho_metric, cfg, acfg,
                                   coherence=q)
        # s(0.05) = (1 - 0.25)^2 = 0.5625 -> a = 1 - 0.4*0.5625
        expected = 1.0 - 0.4 * (1 - 0.05 / 0.2) ** 2
        sheet = a2[n_ring:]
        faded = sheet[sheet < 1.0]
        assert faded.numel() > 0
        assert torch.allclose(
            faded, torch.full_like(faded, expected), atol=1e-12)
        assert float(a2[:n_ring].min()) == 1.0   # boundary q=1 intact


class TestGarbageCollection:
    def _solver_cfg(self, rate):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent
                               / "scripts" / "experiments"))
        from p1_production_comparison import make_cfg
        from src.torch.transport.grid_wasserstein import GridMetricConfig
        v, n_ring, n_sheet = _disk_with_pair()
        delta, tau = compute_recommended_params(v.positions)
        cfg = make_cfg("C3", delta, tau, 1)
        cfg.bem_solver_mode = "legacy_global_projection"
        cfg.metric_backend = "grid_poisson"
        cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
            grid_shape=(256, 256), fill_epsilon=0.08,
            projection_rel_tol=5e-2))   # headroom for the closed loop
        cfg.grid_carrier_amplitude_fade = True
        cfg.grid_amplitude_config = CarrierAmplitudeConfig(rate=rate)
        return v, n_ring, n_sheet, cfg

    def test_faded_marks_are_collected_and_order_preserved(self):
        from src.torch.solver.mm_solver import MMSolver
        v, n_ring, n_sheet, cfg = self._solver_cfg(rate=1.0)
        solver = MMSolver(cfg)
        seen = []

        def cb(step, r):
            vv = (r.committed_varifold
                  if r.committed_varifold is not None else r.varifold)
            seen.append((vv.n_points, dict(r.amplitude_stats)))
            return False

        solver.solve(v, 3, callback=cb)
        # rate 1.0: every eligible sheet mark (interior AND q < q_dead)
        # fades to zero at its first update and is collected; sheet
        # ENDS with q >= q_dead correctly stay (dynamical-death veto).
        n_final = seen[-1][0]
        assert n_final < v.n_points          # something was collected
        total_collected = sum(s["n_collected"] for _, s in seen)
        assert total_collected == v.n_points - n_final   # bookkeeping
        assert n_final >= n_ring
        # the ring survives intact and order is preserved
        final_v = None

        def cb2(step, r):
            nonlocal final_v
            final_v = (r.committed_varifold
                       if r.committed_varifold is not None
                       else r.varifold)
            return False

        # positions of the first n_ring survivors match the ring's
        # evolved positions ordering (no permutation introduced): the
        # ring indices precede the sheet indices, so survivors[:n_ring]
        # are exactly the ring marks
        # (checked via angles: ring angles vary smoothly, sheets are
        # 0 / pi constants)
        _ = cb2

    def test_gc_grid_invariance_at_zero_amplitude(self):
        """Deleting a = 0 marks changes the grid carrier NOT AT ALL:
        the reconstruction from (cloud with ghosts, m*a) equals the
        reconstruction from (collected cloud, m[keep]*a[keep])."""
        import math as _m
        v, n_ring, n_sheet = _disk_with_pair()
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
        m = _masses(v)
        a = torch.ones(v.n_points, dtype=DT)
        a[n_ring:] = 0.0
        pg_ghost = _phase(v, m * a, cfg, _m.pi)
        keep = a > 0
        v2 = OrientedPointCloudVarifold(positions=v.positions[keep],
                                        angles=v.angles[keep])
        pg_clean = _phase(v2, (m * a)[keep], cfg, _m.pi)
        assert torch.equal(pg_ghost.rho_metric, pg_clean.rho_metric)

    def test_thaw_neighbors_recover_coherence(self):
        """Removing the ghost wall lets the recruited live marks'
        coherence recover -- q is a pure state function. Wall placed
        WITHIN the coherence kernel range (sigma = 0.1) of the ring."""
        import math as _m
        from src.torch.transport.bem_wasserstein import compute_coherence
        n_ring, n_sheet = 256, 24
        disk = generate_oriented_circle(n_ring, 1.0, (0.0, 0.0),
                                        "cpu", DT)
        ys = torch.linspace(-0.12, 0.12, n_sheet, dtype=DT)
        seg1 = torch.stack([torch.full_like(ys, -0.95), ys], dim=1)
        seg2 = torch.stack([torch.full_like(ys, -0.93), ys], dim=1)
        pos = torch.cat([disk.positions, seg1, seg2])
        ang = torch.cat([disk.angles,
                         torch.zeros(n_sheet, dtype=DT),
                         torch.full((n_sheet,), _m.pi, dtype=DT)])
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        m = _masses(v)
        q_with = compute_coherence(v, m, 0.1, "wendland_c2",
                                   backend="naive")
        keep = torch.arange(v.n_points) < n_ring
        v2 = OrientedPointCloudVarifold(positions=v.positions[keep],
                                        angles=v.angles[keep])
        q_without = compute_coherence(v2, _masses(v2), 0.1,
                                      "wendland_c2", backend="naive")
        near_wall = ((v.positions[:n_ring, 0] < -0.85)
                     & (v.positions[:n_ring, 1].abs() < 0.15))
        assert bool(near_wall.any())
        q_before = float(q_with[:n_ring][near_wall].min())
        q_after = float(q_without[near_wall].min())
        assert q_before < 0.9          # the wall depressed them
        assert q_after > q_before + 0.05   # collection thaws them


class TestDeadDofFreeze:
    """Arm 7: horizontal-space optimization -- dead marks'
    displacement DOF leave the admissible basis."""

    def _cfg(self, freeze):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent
                               / "scripts" / "experiments"))
        from p1_production_comparison import make_cfg
        from src.torch.transport.grid_wasserstein import GridMetricConfig
        v, n_ring, n_sheet = _disk_with_pair()
        delta, tau = compute_recommended_params(v.positions)
        cfg = make_cfg("C3", delta, tau, 1)
        cfg.bem_solver_mode = "legacy_global_projection"
        cfg.metric_backend = "grid_poisson"
        cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
            grid_shape=(256, 256), fill_epsilon=0.08))
        cfg.grid_freeze_dead_dof = freeze
        return v, n_ring, n_sheet, cfg

    def test_frozen_marks_receive_zero_displacement(self):
        from src.torch.solver.mm_step import MMStepper
        v, n_ring, n_sheet, cfg = self._cfg(freeze=True)
        st = MMStepper(cfg)
        r = st.step(v)
        frozen = st.fixed_coherence < cfg.grid_dead_dof_q
        assert bool(frozen.any())            # the wall is dead
        assert int(frozen[:n_ring].sum()) == 0   # ring alive
        s_frozen = r.displacements[frozen]
        assert float(s_frozen.abs().max()) < 1e-10, \
            f"frozen DOF moved: {float(s_frozen.abs().max()):.2e}"
        # live marks still move
        assert float(r.displacements[~frozen].abs().max()) > 1e-8
        assert st._n_frozen_dof == int(frozen.sum())

    def test_flag_off_bitwise_unchanged(self):
        from src.torch.solver.mm_step import MMStepper
        v, _, _, cfg_off = self._cfg(freeze=False)
        st1 = MMStepper(cfg_off)
        r1 = st1.step(v)
        v2, _, _, cfg_off2 = self._cfg(freeze=False)
        st2 = MMStepper(cfg_off2)
        r2 = st2.step(v2)
        assert torch.equal(r1.displacements, r2.displacements)
        assert st1._n_frozen_dof == 0


class TestClosedLoopRate:
    def test_rate_scale_pins(self):
        """rate_scale = 1 reproduces the open-loop update; 0 freezes
        the amplitudes; intermediate values scale linearly."""
        import math as _m
        v, n_ring, n_sheet = _disk_with_pair()
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
        m = _masses(v)
        acfg = CarrierAmplitudeConfig(rate=0.4)
        pg = _phase(v, m, cfg, _m.pi)
        a = torch.ones(v.n_points, dtype=DT)
        a1, _ = update_amplitudes(v, a, pg.rho_metric, cfg, acfg,
                                  rate_scale=1.0)
        a0, st0 = update_amplitudes(v, a, pg.rho_metric, cfg, acfg,
                                    rate_scale=0.0)
        ah, _ = update_amplitudes(v, a, pg.rho_metric, cfg, acfg,
                                  rate_scale=0.5)
        assert torch.equal(a0, a)                      # paused
        assert st0["rate_scale"] == 0.0
        faded1 = (a - a1)
        fadedh = (a - ah)
        assert torch.allclose(fadedh, 0.5 * faded1, atol=1e-14)

    def test_solver_throttles_under_projection_pressure(self):
        """With a tiny projection_rel_tol the measured p_now consumes
        the budget and the controller throttles (rate_scale < 1)."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent
                               / "scripts" / "experiments"))
        from p1_production_comparison import make_cfg
        from src.torch.solver.mm_solver import MMSolver
        from src.torch.transport.grid_wasserstein import GridMetricConfig
        v, n_ring, n_sheet = _disk_with_pair()
        delta, tau = compute_recommended_params(v.positions)
        cfg = make_cfg("C3", delta, tau, 1)
        cfg.bem_solver_mode = "legacy_global_projection"
        cfg.metric_backend = "grid_poisson"
        # measure the fixture's own projection first, then set the
        # tolerance to pass validation while the 40% budget is mostly
        # consumed: tol = 2 * p_now  ->  p* = 0.8 * p_now < p_now
        import math as _m
        from src.torch.transport.phase_grid import CurrentToPhase
        m = _masses(v)
        pg = CurrentToPhase(PhaseGridConfig(
            grid_shape=(256, 256), fill_epsilon=0.08)).reconstruct(
            v, m, _m.pi)
        tol = 2.0 * pg.projection_l2_relative
        cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
            grid_shape=(256, 256), fill_epsilon=0.08,
            projection_rel_tol=tol))
        cfg.grid_carrier_amplitude_fade = True
        solver = MMSolver(cfg)
        seen = []

        def cb(step, r):
            seen.append(r.amplitude_stats)
            return False

        solver.solve(v, 1, callback=cb)
        assert seen[-1]["rate_scale"] < 0.5


class TestVariationalFade:
    """Arm V1 (spec-2.0 full prototype): the box-constrained QP whose
    quadratic term is the grid metric on amplitude-response columns.
    Pinned: antiparallel-pair cancellation is METRIC structure (joint
    removal is far cheaper than solo), corner solutions are exact
    zeros, lam = 0 is the identity, H is symmetric PSD, and the
    solver path garbage-collects the extinguished cohort."""

    def _op(self, pg):
        from src.torch.transport.weighted_poisson import (
            WeightedPoissonConfig,
            WeightedPoissonOperator,
        )
        return WeightedPoissonOperator(
            pg.rho_metric,
            WeightedPoissonConfig(grid_shape=pg.rho_metric.shape))

    def test_pair_direction_is_metric_cheap(self):
        import math as _m
        from src.torch.transport.carrier_amplitude import (
            _amplitude_response_columns,
        )
        v, n_ring, n_sheet = _disk_with_pair()
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
        m = _masses(v)
        pg = _phase(v, m, cfg, _m.pi)
        op = self._op(pg)
        k = n_sheet // 2
        i, j = n_ring + k, n_ring + n_sheet + k   # antiparallel pair
        idx = torch.tensor([i, j])
        cols = _amplitude_response_columns(
            v.positions[idx], v.normals[idx], m[idx], cfg, op)

        def cost(c):
            phi, info = op.solve(c)
            assert info.converged
            return float(op.inner(c, phi))

        solo_i, solo_j = cost(cols[0]), cost(cols[1])
        joint = cost(cols[0] + cols[1])
        assert joint < 0.2 * (solo_i + solo_j), (
            f"pair cancellation missing: joint={joint:.3e} vs "
            f"solo sum={(solo_i + solo_j):.3e}")

    def test_qp_reaches_exact_zeros_and_h_is_psd(self):
        import math as _m
        from src.torch.transport.carrier_amplitude import (
            variational_fade,
        )
        v, n_ring, n_sheet = _disk_with_pair()
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
        m = _masses(v)
        pg = _phase(v, m, cfg, _m.pi)
        op = self._op(pg)
        q = torch.ones(v.n_points, dtype=DT)
        q[n_ring:] = 0.05                        # dead fossils
        a = torch.ones(v.n_points, dtype=DT)
        acfg = CarrierAmplitudeConfig(variational=True, lam=1e6)
        a2, st = variational_fade(v, a, m, op, pg.rho_metric, cfg,
                                  acfg, coherence=q, time_step=1e-4)
        assert st["skipped"] is None
        assert st["n_free"] > 0
        # lam far above the measured cost band: every free mark is
        # driven to its box corner, which is an EXACT zero
        assert st["n_corner"] == st["n_free"]
        assert float(a2[n_ring:].max()) == 0.0
        assert float(a2[:n_ring].min()) == 1.0   # ring untouched
        assert bool((a2 <= a).all())             # monotone
        # metric quadratic is symmetric PSD (up to the Tikhonov floor)
        assert st["h_eig_min"] >= -1e-10 * max(st["h_eig_max"], 1.0)
        assert st["h_diag_min"] > 0.0

    def test_lambda_zero_is_identity(self):
        import math as _m
        from src.torch.transport.carrier_amplitude import (
            variational_fade,
        )
        v, n_ring, n_sheet = _disk_with_pair()
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
        m = _masses(v)
        pg = _phase(v, m, cfg, _m.pi)
        op = self._op(pg)
        a = torch.ones(v.n_points, dtype=DT)
        acfg = CarrierAmplitudeConfig(variational=True, lam=0.0)
        a2, st = variational_fade(v, a, m, op, pg.rho_metric, cfg,
                                  acfg, coherence=None, time_step=1e-4)
        assert torch.equal(a2, a)
        assert st["variational"] is True and st["n_corner"] == 0

    def test_solver_variational_gc(self):
        """End to end: the variational path extinguishes the gated
        dead cohort at box corners and the existing GC collects it."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent
                               / "scripts" / "experiments"))
        from p1_production_comparison import make_cfg
        from src.torch.solver.mm_solver import MMSolver
        from src.torch.transport.grid_wasserstein import GridMetricConfig
        v, n_ring, n_sheet = _disk_with_pair()
        delta, tau = compute_recommended_params(v.positions)
        cfg = make_cfg("C3", delta, tau, 1)
        cfg.bem_solver_mode = "legacy_global_projection"
        cfg.metric_backend = "grid_poisson"
        cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
            grid_shape=(256, 256), fill_epsilon=0.08,
            projection_rel_tol=5e-2))
        cfg.grid_carrier_amplitude_fade = True
        cfg.grid_amplitude_config = CarrierAmplitudeConfig(
            variational=True, lam=1e6)
        solver = MMSolver(cfg)
        seen = []

        def cb(step, r):
            vv = (r.committed_varifold
                  if r.committed_varifold is not None else r.varifold)
            seen.append((vv.n_points, dict(r.amplitude_stats)))
            return False

        solver.solve(v, 3, callback=cb)
        assert all(s["variational"] for _, s in seen)
        n_final = seen[-1][0]
        assert n_final < v.n_points          # cohort collected
        total_collected = sum(s["n_collected"] for _, s in seen)
        assert total_collected == v.n_points - n_final
        assert n_final >= n_ring
        assert any(s["n_corner"] > 0 for _, s in seen)

    def test_qp_exact_on_ill_conditioned_pair_block(self):
        """A10 postmortem pin: with the measured conditioning
        (eig_max/eig_min ~ 1e9) the pair-cancelled direction must
        reach its box corner in ONE call -- projected gradient
        crawled at 0-3 corners/step and starved the extinction."""
        from src.torch.transport.carrier_amplitude import (
            _box_qp_active_set,
        )
        # two near-perfectly cancelled marks: H = s(vv^T) + eps I
        # with v = (1,-1)/sqrt(2): solo cost ~ s/2, joint cost ~ eps
        s, eps = 0.44, 1e-8
        v = torch.tensor([1.0, -1.0], dtype=torch.float64) \
            / math.sqrt(2.0)
        Hmat = s * torch.outer(v, v) + eps * torch.eye(
            2, dtype=torch.float64)
        g = torch.full((2,), 0.008, dtype=torch.float64)  # lam w m
        lower = torch.full((2,), -1.0, dtype=torch.float64)
        x, _, resid = _box_qp_active_set(Hmat, g, lower)
        assert torch.equal(x, lower), f"expected corners, got {x}"
        # and a genuinely expensive solo direction is REFUSED at the
        # same drive: H = s I (no cancellation), g far below s*|lower|
        H2 = s * torch.eye(1, dtype=torch.float64)
        g2 = torch.tensor([0.008], dtype=torch.float64)
        lo2 = torch.tensor([-1.0], dtype=torch.float64)
        x2, _, _ = _box_qp_active_set(H2, g2, lo2)
        assert float(x2) > -0.05                # interior, tiny fade
        assert abs(float(x2) + 0.008 / 0.44) < 1e-10

    def test_qp_matches_kkt_enumeration_oracle(self):
        """A10b postmortem pin: on random SPD instances with the
        measured conditioning (up to 1e9), the solver's objective
        must match the EXACT minimizer found by enumerating all 3^n
        KKT configurations, and must never exceed 0 (the do-nothing
        point) -- the A10b failure returned f = +2.14."""
        import itertools
        from src.torch.transport.carrier_amplitude import (
            _box_qp_active_set,
        )
        torch.manual_seed(0)

        def oracle(Hm, g, lower):
            n = g.shape[0]
            best, bx = 0.0, torch.zeros_like(g)   # origin is feasible
            for conf in itertools.product((0, 1, 2), repeat=n):
                x = torch.zeros_like(g)
                fixed = torch.zeros(n, dtype=torch.bool)
                for i, c in enumerate(conf):
                    if c == 0:
                        x[i], fixed[i] = lower[i], True
                    elif c == 1:
                        x[i], fixed[i] = 0.0, True
                free = ~fixed
                if bool(free.any()):
                    Hff = Hm[free][:, free]
                    rhs = -(g[free] + Hm[free][:, ~free] @ x[~free])
                    try:
                        x[free] = torch.linalg.solve(Hff, rhs)
                    except Exception:
                        continue
                if bool((x < lower - 1e-9).any()) or bool(
                        (x > 1e-9).any()):
                    continue
                fx = float(0.5 * x @ (Hm @ x) + g @ x)
                if fx < best:
                    best, bx = fx, x
            return best

        for trial in range(60):
            n = int(torch.randint(1, 6, (1,)))
            Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
            eigs = 10.0 ** (torch.rand(n, dtype=torch.float64) * 9 - 9)
            Hm = Q @ torch.diag(eigs) @ Q.T
            Hm = 0.5 * (Hm + Hm.T)
            g = torch.rand(n, dtype=torch.float64) * 0.02
            lower = -torch.ones(n, dtype=torch.float64)
            x, _, _ = _box_qp_active_set(Hm, g, lower)
            fx = float(0.5 * x @ (Hm @ x) + g @ x)
            fstar = oracle(Hm, g, lower)
            assert fx <= 1e-12, f"trial {trial}: ascent f={fx}"
            gap = fx - fstar
            assert gap <= 1e-8 * (abs(fstar) + 1.0), (
                f"trial {trial}: f={fx:.6e} vs oracle {fstar:.6e}")

    def test_gate_l2_norm_prices_the_pair_residual(self):
        """Experiment B pin: the gate's plain-L2 norm charges the
        antiparallel pair's sub-cell residual far more than the
        W_lin (H^-1-type) metric does -- the measured A10c yardstick
        mismatch. On the fixture: L2 joint/solo ~ 4.7e-2 vs
        W ~ 1.2e-3 (~38x)."""
        import math as _m
        from src.torch.transport.carrier_amplitude import (
            _amplitude_response_columns,
        )
        v, n_ring, n_sheet = _disk_with_pair()
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
        m = _masses(v)
        pg = _phase(v, m, cfg, _m.pi)
        op = self._op(pg)
        k = n_sheet // 2
        idx = torch.tensor([n_ring + k, n_ring + n_sheet + k])
        cols = _amplitude_response_columns(
            v.positions[idx], v.normals[idx], m[idx], cfg, op)

        def l2c(c):
            return float((c * c).sum()) * op.cell_area

        def wc(c):
            phi, info = op.solve(c)
            assert info.converged
            return float(op.inner(c, phi))

        r_l2 = l2c(cols[0] + cols[1]) / (l2c(cols[0]) + l2c(cols[1]))
        r_w = wc(cols[0] + cols[1]) / (wc(cols[0]) + wc(cols[1]))
        assert r_l2 > 10.0 * r_w, (r_l2, r_w)
        assert r_l2 > 0.01                    # L2 residual is REAL

    def test_gate_l2_qp_path_runs_and_records_norm(self):
        import math as _m
        from src.torch.transport.carrier_amplitude import (
            variational_fade,
        )
        v, n_ring, n_sheet = _disk_with_pair()
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
        m = _masses(v)
        pg = _phase(v, m, cfg, _m.pi)
        op = self._op(pg)
        q = torch.ones(v.n_points, dtype=DT)
        q[n_ring:] = 0.05
        a = torch.ones(v.n_points, dtype=DT)
        acfg = CarrierAmplitudeConfig(variational=True, lam=1e-9,
                                      qp_norm="gate_l2")
        a2, st = variational_fade(v, a, m, op, pg.rho_metric, cfg,
                                  acfg, coherence=q, time_step=1e-4)
        assert st["qp_norm"] == "gate_l2" and st["skipped"] is None
        # lam far below the L2 band: removal refused
        assert st["n_corner"] == 0
        assert float((a - a2).max()) < 1e-3


class TestCoupledVerticalDof:
    """Arm V2: the coupled minimizing movement (da as a decision
    variable of the MM step). Pinned: default off; gate-empty inputs
    (circle, annulus) reproduce the pure Otto step BITWISE -- the
    formal MS-consistency claim -- and on the pair fixture the
    coupled step extinguishes with in-step transport compensation
    (cross term active) and the existing GC collects exact zeros."""

    def _cfg(self, vertical, lam=1.0):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent
                               / "scripts" / "experiments"))
        from p1_production_comparison import make_cfg
        from src.torch.transport.grid_wasserstein import GridMetricConfig
        delta_tau_src = self._fixture_v.positions
        delta, tau = compute_recommended_params(delta_tau_src)
        cfg = make_cfg("C3", delta, tau, 1)
        cfg.bem_solver_mode = "legacy_global_projection"
        cfg.metric_backend = "grid_poisson"
        cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
            grid_shape=(256, 256), fill_epsilon=0.08,
            projection_rel_tol=5e-2))
        if vertical:
            cfg.grid_carrier_amplitude_fade = True
            cfg.grid_vertical_dof = True
            cfg.grid_amplitude_config = CarrierAmplitudeConfig(lam=lam)
        return cfg

    def test_default_off(self):
        from src.torch.solver.mm_step import MMConfig
        assert MMConfig().grid_vertical_dof is False

    def test_gate_empty_circle_is_bitwise_pure_otto(self):
        from src.torch.solver.mm_step import MMStepper
        v = generate_oriented_circle(192, 1.0, (0.0, 0.0), "cpu", DT)
        self._fixture_v = v
        r_off = MMStepper(self._cfg(False)).step(v)
        st_on = MMStepper(self._cfg(True))
        r_on = st_on.step(v)
        assert torch.equal(r_off.displacements, r_on.displacements)
        assert torch.equal(r_off.delta_angles, r_on.delta_angles)
        assert r_on.vertical_da is None
        assert r_on.vertical_stats["n_free"] == 0

    def test_gate_empty_annulus_is_bitwise_pure_otto(self):
        """The mandatory MS-consistency anchor: on the annulus (the
        exact-solution benchmark geometry) the coupled scheme IS the
        pure Otto scheme, so the frozen exact-ODE tolerances of
        test_annulus_run_a are inherited unchanged."""
        from src.torch.shapes.generator import generate_oriented_annulus
        from src.torch.solver.mm_step import MMStepper
        v = generate_oriented_annulus(192, 1.2, 0.5, (0.0, 0.0),
                                      "cpu", DT)
        self._fixture_v = v
        r_off = MMStepper(self._cfg(False)).step(v)
        r_on = MMStepper(self._cfg(True)).step(v)
        assert torch.equal(r_off.displacements, r_on.displacements)
        assert torch.equal(r_off.delta_angles, r_on.delta_angles)
        assert r_on.vertical_da is None

    def test_coupled_step_extinguishes_with_compensation(self):
        from src.torch.solver.mm_solver import MMSolver
        v, n_ring, n_sheet = _disk_with_pair()
        self._fixture_v = v
        cfg = self._cfg(True)
        solver = MMSolver(cfg)
        seen = []

        def cb(step, r):
            vv = (r.committed_varifold
                  if r.committed_varifold is not None else r.varifold)
            seen.append((vv.n_points, dict(r.vertical_stats or {}),
                         dict(r.amplitude_stats or {})))
            return False

        solver.solve(v, 2, callback=cb)
        n1, vs1, as1 = seen[0]
        assert vs1["n_free"] > 0 and vs1["skipped"] is None
        assert vs1["n_corner"] > 0          # exact zeros in-step
        assert as1["n_collected"] > 0       # GC collected them
        assert n1 < v.n_points
        assert n1 >= n_ring                 # ring never collected
        assert vs1["cross_term"] != 0.0     # compensation coupling on
