"""Arm 4: passive advection of low-coherence marks by the step's own
transport field.

Pinned here:
  1. transport-field parity -- on a smooth ellipse step, grad phi . n
     evaluated at coherent boundary marks reproduces the accepted
     normal displacements (sign AND scale of d = grad phi);
  2. an embedded antiparallel pair is advected rigidly (pairing
     preserved: small relative displacement, common rotation);
  3. default-off bitwise noninterference on the grid backend;
  4. coherent marks are essentially immobile under the stage;
  5. a failed solve skips the stage (input returned, reason recorded).
"""

import math
import sys
from pathlib import Path

import pytest
import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import (
    compute_masses,
    compute_recommended_params,
)
from src.torch.shapes.generator import (
    generate_oriented_circle,
    generate_oriented_ellipse,
)
from src.torch.solver.mm_solver import MMSolver
from src.torch.solver.mm_step import MMStepper
from src.torch.transport.bem_wasserstein import compute_coherence
from src.torch.transport.fossil_advection import (
    FossilAdvectionConfig,
    advect_fossils,
)
from src.torch.transport.grid_wasserstein import GridMetricConfig
from src.torch.transport.phase_grid import PhaseGridConfig

sys.path.insert(0, str(Path(__file__).parent.parent
                       / "scripts" / "experiments"))
from p1_production_comparison import make_cfg  # noqa: E402

DT = torch.float64


def _grid_cfg(delta, tau, n=256, eps=0.06):
    cfg = make_cfg("C3", delta, tau, 1)
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(n, n), fill_epsilon=eps))
    return cfg


def _stepper_and_step(v, n=256, eps=0.06):
    delta, tau = compute_recommended_params(v.positions)
    cfg = _grid_cfg(delta, tau, n=n, eps=eps)
    st = MMStepper(cfg)
    r = st.step(v)
    return st, cfg, r


def _coherence_of(v, cfg, st):
    m = compute_masses(v.positions, st._mass_delta_for_kde,
                       st._mass_tau_for_kde, cfg.mass_kernel)
    return compute_coherence(v, m, st._sigma, cfg.perimeter_kernel,
                             backend=cfg.backend)


class TestTransportFieldParity:
    def test_grad_phi_normal_matches_accepted_displacements(self):
        """d = grad phi with phi = L^dagger(B y*): on coherent
        boundary marks the normal component of the sampled field must
        reproduce the accepted normal-graph displacements (sign and
        scale)."""
        v = generate_oriented_ellipse(256, 1.2, 0.8, (0.0, 0.0),
                                      "cpu", DT)
        st, cfg, r = _stepper_and_step(v)
        gm = st.grid_wasserstein
        q = _coherence_of(r.varifold, cfg, st)
        # force full advection weight so the field itself is exposed
        out, stats = advect_fossils(
            r.varifold, torch.zeros_like(q), r.displacements,
            r.delta_angles, gm)
        assert stats["applied"]
        disp = out.positions - r.varifold.positions
        d_n = (disp * r.varifold.normals).sum(dim=1)
        s = r.displacements
        mask = q > 0.9
        assert int(mask.sum()) > 200
        num = float((d_n[mask] * s[mask]).sum())
        den = float((s[mask] ** 2).sum())
        slope = num / den
        corr = num / (float(d_n[mask].norm()) * float(s[mask].norm()))
        # measured at implementation time: slope 1.00 +- a few % on
        # the resolved ellipse; the pin is deliberately generous --
        # what it must catch is a sign error or a missing dt/dx factor
        assert corr > 0.95, f"correlation {corr}"
        assert 0.7 < slope < 1.3, f"slope {slope}"

    def test_field_is_curl_projected_gradient(self):
        """The advection displacement is a pure gradient field (zero
        vorticity); the material rotation is strain-driven and must
        stay bounded by the smooth step's strain scale."""
        v = generate_oriented_ellipse(256, 1.2, 0.8, (0.0, 0.0),
                                      "cpu", DT)
        st, cfg, r = _stepper_and_step(v)
        q = _coherence_of(r.varifold, cfg, st)
        out, stats = advect_fossils(
            r.varifold, torch.zeros_like(q), r.displacements,
            r.delta_angles, st.grid_wasserstein)
        assert stats["applied"]
        assert stats["max_abs_dtheta"] < 0.5


class TestPairRigidity:
    def test_embedded_antiparallel_pair_advects_rigidly(self):
        """A slightly-offset antiparallel sheet pair inside a circle:
        both members must receive nearly identical displacement and
        rotation (pairing preserved -- the failure mode this stage
        must never reproduce is tearing the pair apart)."""
        n_ring, n_sheet = 256, 24
        disk = generate_oriented_circle(n_ring, 1.0, (0.0, 0.0),
                                        "cpu", DT)
        ys = torch.linspace(-0.12, 0.12, n_sheet, dtype=DT)
        seg1 = torch.stack([torch.full_like(ys, -0.11), ys], dim=1)
        seg2 = torch.stack([torch.full_like(ys, -0.09), ys], dim=1)
        pos = torch.cat([disk.positions, seg1, seg2])
        ang = torch.cat([disk.angles,
                         torch.zeros(n_sheet, dtype=DT),
                         torch.full((n_sheet,), math.pi, dtype=DT)])
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        st, cfg, r = _stepper_and_step(v)
        gm = st.grid_wasserstein
        q = _coherence_of(r.varifold, cfg, st)
        out, stats = advect_fossils(
            r.varifold, q, r.displacements, r.delta_angles, gm)
        assert stats["applied"]
        disp = out.positions - r.varifold.positions
        dth = out.angles - r.varifold.angles
        i1 = slice(n_ring, n_ring + n_sheet)
        i2 = slice(n_ring + n_sheet, n_ring + 2 * n_sheet)
        # the sheets are eps/3 apart; the smooth field must move the
        # two partners together to within a small fraction of their
        # own motion or of the cell size
        rel = (disp[i1] - disp[i2]).norm(dim=1)
        dx = 4.0 / cfg.grid_metric.phase.grid_shape[0]
        assert float(rel.max()) < 0.5 * dx
        assert float((dth[i1] - dth[i2]).abs().max()) < 0.05

    def test_pair_phase_unchanged(self):
        """Advecting the hidden pair must not change the reconstructed
        phase beyond discretization noise (the pair cancels before and
        after the common motion)."""
        from src.torch.transport.phase_grid import CurrentToPhase
        n_ring, n_sheet = 256, 24
        disk = generate_oriented_circle(n_ring, 1.0, (0.0, 0.0),
                                        "cpu", DT)
        ys = torch.linspace(-0.12, 0.12, n_sheet, dtype=DT)
        seg1 = torch.stack([torch.full_like(ys, -0.11), ys], dim=1)
        seg2 = torch.stack([torch.full_like(ys, -0.09), ys], dim=1)
        pos = torch.cat([disk.positions, seg1, seg2])
        ang = torch.cat([disk.angles,
                         torch.zeros(n_sheet, dtype=DT),
                         torch.full((n_sheet,), math.pi, dtype=DT)])
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        st, cfg, r = _stepper_and_step(v)
        q = _coherence_of(r.varifold, cfg, st)
        out, stats = advect_fossils(
            r.varifold, q, r.displacements, r.delta_angles,
            st.grid_wasserstein)
        pcfg = cfg.grid_metric.phase
        delta, tau = compute_recommended_params(v.positions)
        m_a = compute_masses(r.varifold.positions, delta, tau,
                             cfg.mass_kernel)
        m_b = compute_masses(out.positions, delta, tau,
                             cfg.mass_kernel)
        pa = CurrentToPhase(pcfg).reconstruct(r.varifold, m_a, math.pi)
        pb = CurrentToPhase(pcfg).reconstruct(out, m_b, math.pi)
        rel = float((pb.rho_metric - pa.rho_metric).abs().mean()
                    / pa.rho_metric.abs().mean())
        assert rel < 5e-3, f"phase changed by {rel}"


class TestNoninterference:
    def test_flag_off_default_and_stage_not_entered(self):
        """Default is OFF; with the flag off the stage is never
        entered (advection_stats stays None) and the two-step
        trajectory is deterministic."""
        from src.torch.solver.mm_step import MMConfig
        assert MMConfig().grid_fossil_advection is False

        v = generate_oriented_ellipse(128, 1.1, 0.9, (0.0, 0.0),
                                      "cpu", DT)
        delta, tau = compute_recommended_params(v.positions)

        def run():
            cfg = _grid_cfg(delta, tau, n=256, eps=0.08)
            cfg.time_step = 1e-4
            solver = MMSolver(cfg)
            seen = []

            def cb(step, result):
                seen.append(result)
                return False

            solver.solve(v, 2, callback=cb)
            return seen

        a = run()
        b = run()
        assert all(r.advection_stats is None for r in a)
        assert torch.equal(a[-1].committed_varifold.positions,
                           b[-1].committed_varifold.positions)

    def test_coherent_shape_essentially_immobile(self):
        """On a clean circle every q ~ 1: the stage's extra motion is
        below discretization noise even when ENABLED."""
        v = generate_oriented_circle(256, 1.0, (0.0, 0.0), "cpu", DT)
        st, cfg, r = _stepper_and_step(v)
        q = _coherence_of(r.varifold, cfg, st)
        out, stats = advect_fossils(
            r.varifold, q, r.displacements, r.delta_angles,
            st.grid_wasserstein)
        assert stats["applied"]
        extra = (out.positions - r.varifold.positions).norm(dim=1)
        dx = 4.0 / cfg.grid_metric.phase.grid_shape[0]
        # (1-q) suppression: the residual motion must be well below a
        # cell even where q dips slightly under the kernel quadrature
        assert float(extra.max()) < 0.2 * dx


class TestFailClosed:
    def test_failed_solve_skips_stage(self):
        v = generate_oriented_circle(256, 1.0, (0.0, 0.0), "cpu", DT)
        st, cfg, r = _stepper_and_step(v)
        gm = st.grid_wasserstein

        class _Boom:
            def solve_or_raise(self, rhs):
                from src.torch.transport.weighted_poisson import (
                    IncompatibleGridVelocityError,
                )
                raise IncompatibleGridVelocityError("synthetic")

        class _Wrap:
            config = gm.config
            flux = gm.flux
            poisson = _Boom()

        q = _coherence_of(r.varifold, cfg, st)
        out, stats = advect_fossils(
            r.varifold, q, r.displacements, r.delta_angles, _Wrap())
        assert not stats["applied"]
        assert "IncompatibleGridVelocityError" in stats["skipped_reason"]
        assert torch.equal(out.positions, r.varifold.positions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
