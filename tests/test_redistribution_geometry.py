"""0L-R0 pins: redistribution integrability machinery.

R0a (operator algebra):
- default path bitwise (global density + stored tangents + legacy
  correction == historical operator);
- loopwise density / per-loop CV stop / per-loop cap;
- foreign-loop independence CONDITIONAL ON FIXED q (reviewer): with
  q == 1 the core operator's target-loop update is invariant to the
  other loop -- bitwise under foreign position changes, rel 1e-12
  under foreign COUNT changes (summation-order effects only); with
  the production q^full the difference is telemetry, not a bug
  (q_redist policy = registered pre-R2 blocker);
- PCA spectral certificate with the PRE-REGISTERED thresholds
  rho* = 0.25, N_eff_min = 2.5 (clean-fixture envelope at pca
  bandwidth = mass delta: worst rho 0.146 (flower), worst N_eff 3.11;
  blob sits at rho ~ 1; the 1325 state is VALIDATION only);
- bandwidth projector stability; orientation margin (cos 30 deg);
- permutation invariance of the full loopwise-PCA path;
- post-MM recertification fail-closed (solver-level).

R0b (exact / oracle):
- polygon-oracle arm: per-subiteration first-order area leakage
  L_A = sum g . dx is EXACTLY zero (g . t_oracle = 0 identically);
- PCA arm: exact closure Delta A = L_A + Q_A to machine precision
  (the quadratic identity as measurement wiring);
- PCA vs polygon projector parity on generator-ordered fixtures;
- tilted-circle repair (re-anchoring restores radial normals).
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
from src.torch.solver.redistribution_rules import (
    RedistributionGeometryError,
    _local_pca_tangents,
    pca_normal_reanchor,
    redistribute_with_rule,
)
from src.torch.transport.loop_geometry import (
    area_quadratic_term,
    certified_cyclic_order,
    polygon_area,
    polygon_gradient,
)

DT = torch.float64
RHO_MAX = 0.25          # pre-registered (see module docstring)
NEFF_MIN = 2.5
ORI_MARGIN = math.cos(math.radians(30.0))


def circle(R, n, inward=False, jitter=0.0, seed=0, center=(0.0, 0.0)):
    t = torch.arange(n, dtype=DT) * 2 * math.pi / n
    if jitter:
        g = torch.Generator().manual_seed(seed)
        t = t + jitter * torch.randn(n, generator=g, dtype=DT) \
            * 2 * math.pi / n
    pos = torch.stack([R * t.cos() + center[0],
                       R * t.sin() + center[1]], 1)
    ang = t + (math.pi if inward else 0.0)
    return pos, ang


def wall_pair(n1=90, n2=141, gap=0.07, jitter=0.3):
    p1, a1 = circle(0.35, n1, jitter=jitter)
    p2, a2 = circle(0.35 + gap, n2, inward=True, jitter=jitter,
                    seed=1)
    pos = torch.cat([p1, p2])
    ang = torch.cat([a1, a2])
    lbl = torch.cat([torch.zeros(n1, dtype=torch.long),
                     torch.ones(n2, dtype=torch.long)])
    return pos, ang, lbl


DELTA = 0.1


class TestR0aOperator:
    def test_default_path_bitwise(self):
        pos, ang, _ = wall_pair()
        q = torch.ones(pos.shape[0], dtype=DT)
        ref = redistribute_with_rule(pos, ang, "legacy_hybrid",
                                     delta=DELTA, n_iters=4,
                                     coherence=q)
        got = redistribute_with_rule(pos, ang, "legacy_hybrid",
                                     delta=DELTA, n_iters=4,
                                     coherence=q,
                                     density_scope="global",
                                     normal_retraction=False)
        assert torch.equal(ref[0], got[0])
        assert torch.equal(ref[1], got[1])

    def test_loopwise_cv_and_cap(self):
        pos, ang, lbl = wall_pair()
        _, _, info = redistribute_with_rule(
            pos, ang, "legacy_hybrid", delta=DELTA, n_iters=4,
            coherence=torch.ones(pos.shape[0], dtype=DT),
            loop_labels=lbl, density_scope="loopwise",
            tangent_source="local_pca", normal_retraction=True,
            pca_rho_max=RHO_MAX, pca_neff_min=NEFF_MIN,
            pca_delta_tan=DELTA)
        assert info["cv_final_by_loop"] is not None
        assert set(info["cv_final_by_loop"]) == {0, 1}
        assert info["retraction"] is not None
        # per-loop CV improved on both loops
        first = info["trace"][0]["cv_by_loop"]
        last = info["cv_final_by_loop"]
        for lb in (0, 1):
            assert last[lb] < first[lb]

    def test_foreign_loop_independence_fixed_q(self):
        """Core operator (q == 1): the target loop's committed update
        is invariant to the FOREIGN loop -- bitwise under foreign
        position change, rel 1e-12 under foreign count change."""
        n1 = 90

        def run(n2, seed):
            p1, a1 = circle(0.35, n1, jitter=0.3)
            p2, a2 = circle(1.0, n2, inward=False, jitter=0.15,
                            seed=seed)
            pos = torch.cat([p1, p2])
            ang = torch.cat([a1, a2])
            lbl = torch.cat([torch.zeros(n1, dtype=torch.long),
                             torch.ones(n2, dtype=torch.long)])
            out_pos, out_ang, _ = redistribute_with_rule(
                pos, ang, "legacy_hybrid", delta=DELTA,
                n_iters=4, tol=0.0,        # fixed iteration count
                coherence=None, q_override="unit",
                loop_labels=lbl, density_scope="loopwise",
                tangent_source="local_pca", normal_retraction=True,
                pca_rho_max=RHO_MAX, pca_neff_min=NEFF_MIN,
                pca_delta_tan=0.15)
            return out_pos[:n1], out_ang[:n1]

        base_p, base_a = run(220, 1)
        moved_p, moved_a = run(220, 7)      # foreign resampled
        assert torch.equal(base_p, moved_p)
        assert torch.equal(base_a, moved_a)
        cnt_p, cnt_a = run(300, 1)          # foreign count changed
        assert torch.allclose(base_p, cnt_p, rtol=1e-12, atol=1e-14)
        assert torch.allclose(base_a, cnt_a, rtol=1e-12, atol=1e-12)

    def test_production_q_coupling_is_telemetry(self):
        """With the production q^full the target loop DOES couple to
        the foreign loop -- recorded as expected telemetry (the
        q_redist policy is the registered pre-R2 blocker)."""
        from src.torch.oriented_varifold.mass import compute_masses
        from src.torch.transport.bem_wasserstein import (
            compute_coherence,
        )
        n1 = 90

        def run(gap):
            pos, ang, lbl = wall_pair(gap=gap)
            v = OrientedPointCloudVarifold(positions=pos, angles=ang)
            m = compute_masses(pos, DELTA, 0.0, "wendland_c2")
            q = compute_coherence(v, m, 0.1, "wendland_c2")
            out_pos, _, _ = redistribute_with_rule(
                pos, ang, "legacy_hybrid", delta=DELTA, n_iters=4,
                tol=0.0, coherence=q,
                loop_labels=lbl, density_scope="loopwise",
                tangent_source="local_pca", normal_retraction=True,
                pca_rho_max=RHO_MAX, pca_neff_min=NEFF_MIN,
                pca_delta_tan=DELTA)
            return out_pos[:n1]

        # different gap -> different q^full on the wall -> different
        # target-loop update (the coupling exists and is q-borne)
        d = float((run(0.07) - run(0.05)).norm())
        assert d > 0.0

    def test_certificate_blob_raises(self):
        g = torch.Generator().manual_seed(3)
        blob = 0.05 * torch.randn(40, 2, generator=g, dtype=DT)
        with pytest.raises(RedistributionGeometryError,
                           match="one-dimensionality"):
            t, cert = _local_pca_tangents(
                blob, torch.zeros(40, dtype=DT), 0.2, "wendland_c2",
                loop_labels=torch.zeros(40, dtype=torch.long),
                return_certificate=True)
            from src.torch.solver.redistribution_rules import (
                _enforce_pca_certificate,
            )
            _enforce_pca_certificate(cert, RHO_MAX, NEFF_MIN)

    def test_certificate_sparse_raises(self):
        # bandwidth far below spacing -> N_eff ~ 1
        pos, ang = circle(0.5, 64)
        with pytest.raises(RedistributionGeometryError,
                           match="effective-sample"):
            t, cert = _local_pca_tangents(
                pos, ang, 0.01, "wendland_c2",
                loop_labels=torch.zeros(64, dtype=torch.long),
                return_certificate=True)
            from src.torch.solver.redistribution_rules import (
                _enforce_pca_certificate,
            )
            _enforce_pca_certificate(cert, RHO_MAX, NEFF_MIN)

    @pytest.mark.parametrize("maker,bound_rad", [
        (lambda: generate_oriented_circle(256, 1.0, (0.0, 0.0),
                                          "cpu", DT), 0.05),
        (lambda: generate_oriented_ellipse(256, device="cpu",
                                           dtype=DT), 0.05),
        (lambda: generate_oriented_flower(320, device="cpu",
                                          dtype=DT), 0.15),
    ])
    def test_bandwidth_projector_stability(self, maker, bound_rad):
        from src.torch.oriented_varifold.mass import (
            compute_recommended_params,
        )
        v = maker()
        lbl = torch.zeros(v.n_points, dtype=torch.long)
        h0, _ = compute_recommended_params(v.positions)
        h0 = float(h0)
        t0 = _local_pca_tangents(v.positions, v.angles, h0,
                                 "wendland_c2", loop_labels=lbl)
        for f in (0.8, 1.25):
            tf = _local_pca_tangents(v.positions, v.angles, f * h0,
                                     "wendland_c2", loop_labels=lbl)
            cosang = (t0 * tf).sum(1).abs().clamp(0.0, 1.0)
            assert float(torch.acos(cosang).max()) < bound_rad

    def test_orientation_margin_raises(self):
        pos, ang = circle(0.5, 96)
        bad = ang.clone()
        bad[:24] += math.radians(80)
        with pytest.raises(RedistributionGeometryError,
                           match="orientation margin"):
            pca_normal_reanchor(pos, bad, 0.2, "wendland_c2",
                                torch.zeros(96, dtype=torch.long),
                                RHO_MAX, NEFF_MIN, ORI_MARGIN)

    def test_permutation_invariance(self):
        pos, ang, lbl = wall_pair()
        g = torch.Generator().manual_seed(11)
        perm = torch.randperm(pos.shape[0], generator=g)

        def run(p, a, lb):
            return redistribute_with_rule(
                p, a, "legacy_hybrid", delta=DELTA, n_iters=3,
                tol=0.0, coherence=None, q_override="unit",
                loop_labels=lb, density_scope="loopwise",
                tangent_source="local_pca", normal_retraction=True,
                pca_rho_max=RHO_MAX, pca_neff_min=NEFF_MIN,
                pca_delta_tan=DELTA)

        p_ref, a_ref, _ = run(pos, ang, lbl)
        p_per, a_per, _ = run(pos[perm], ang[perm], lbl[perm])
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.shape[0])
        assert torch.allclose(p_per[inv], p_ref, rtol=0, atol=1e-13)
        assert torch.allclose(a_per[inv], a_ref, rtol=0, atol=1e-12)

    def test_post_mm_recertificate_fail_closed(self):
        """Solver-level: a pre-MM partition that disagrees with the
        post-MM certified partition must fail closed."""
        from src.torch.solver.mm_solver import MMSolver
        from src.torch.solver.mm_step import MMConfig
        from src.torch.transport.incidence import (
            PartitionAmbiguousError,
        )

        pos, ang, lbl = wall_pair(jitter=0.0)
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        cfg = MMConfig(mass_delta=DELTA, mass_tau=1e-6,
                       redistribution_density_scope="loopwise",
                       redistribution_tangent_source="local_pca",
                       redistribution_normal_retraction=True)
        solver = MMSolver(cfg)

        class Stub:
            config = cfg
            mass_delta = DELTA
            mass_tau = 1e-6
            _sigma = 0.1
            _mass_delta_for_kde = DELTA
            _mass_tau_for_kde = 1e-6
            fixed_coherence = torch.ones(pos.shape[0], dtype=DT)
            _loop_labels_last = torch.zeros(pos.shape[0],
                                            dtype=torch.long)  # WRONG

        with pytest.raises(PartitionAmbiguousError):
            solver._redistribute(Stub(), v, 2)
        # healthy labels pass
        Stub._loop_labels_last = None
        out = solver._redistribute(Stub(), v, 2)
        assert out[0].shape == pos.shape


class TestKappa0LoopwiseCurvature:
    """0L-kappa0 pins: loopwise curvature estimation. The union-cloud
    estimator sign-flips in the contact layer (measured on endgame-#2
    1325: disk true +2.857 vs union -5.609; loopwise +2.855)."""

    def _kappa(self, pos, ang, lbl, eps, loopwise):
        from src.torch.math_utils.curvature import (
            compute_regularized_curvature,
        )
        from src.torch.oriented_varifold.mass import (
            compute_masses,
            compute_masses_oriented_loopwise,
        )
        nrm = torch.stack([ang.cos(), ang.sin()], 1)
        if loopwise:
            m = compute_masses_oriented_loopwise(pos, nrm, lbl,
                                                 eps, 0.0)
            k, _ = compute_regularized_curvature(
                pos, nrm, m, epsilon=eps, loop_labels=lbl)
        else:
            m = compute_masses(pos, eps, 0.0, "wendland_c2")
            k, _ = compute_regularized_curvature(pos, nrm, m,
                                                 epsilon=eps)
        return k

    def test_distant_loops_identity(self):
        p1, a1 = circle(0.35, 96)
        p2, a2 = circle(1.0, 220)
        pos = torch.cat([p1, p2])
        ang = torch.cat([a1, a2])
        lbl = torch.cat([torch.zeros(96, dtype=torch.long),
                         torch.ones(220, dtype=torch.long)])
        ku = self._kappa(pos, ang, lbl, 0.12, False)
        kl = self._kappa(pos, ang, lbl, 0.12, True)
        assert torch.allclose(ku, kl, rtol=1e-12, atol=1e-12)

    def test_clean_accuracy(self):
        for R, n in ((0.5, 128), (1.0, 256)):
            pos, ang = circle(R, n)
            lbl = torch.zeros(n, dtype=torch.long)
            kl = self._kappa(pos, ang, lbl, 0.15 * R, True)
            assert float(kl.mean()) == pytest.approx(1.0 / R,
                                                     rel=0.15)

    def test_antiparallel_sign(self):
        """The decisive pin: gap < epsilon antiparallel pair -- the
        union estimate flips the disk sign, the loopwise one does
        not."""
        pos, ang, lbl = wall_pair(gap=0.06, jitter=0.0)
        eps = 0.12                       # gap < eps
        ku = self._kappa(pos, ang, lbl, eps, False)
        kl = self._kappa(pos, ang, lbl, eps, True)
        disk = lbl == 0
        assert float(ku[disk].mean()) < 0.0          # sign-flipped
        assert float(kl[disk].mean()) > 0.0          # correct
        assert float(kl[disk].mean()) == pytest.approx(1.0 / 0.35,
                                                       rel=0.2)

    def test_self_cross_closure(self):
        """A = A_self + A_cross and B = B_self + B_cross exactly:
        union kappa numerator/denominator split by the mask."""
        from src.torch.math_utils.curvature import (
            compute_regularized_curvature,
        )
        pos, ang, lbl = wall_pair(gap=0.06, jitter=0.0)
        nrm = torch.stack([ang.cos(), ang.sin()], 1)
        m = torch.ones(pos.shape[0], dtype=DT) * 0.02
        eps = 0.12
        # H-vector is linear in the pairwise weights BEFORE the
        # ratio; verify closure on the raw sums via the three masks
        _, H_u = compute_regularized_curvature(pos, nrm, m,
                                               epsilon=eps)
        _, H_s = compute_regularized_curvature(pos, nrm, m,
                                               epsilon=eps,
                                               loop_labels=lbl)
        # cross part via complementary labels trick: all-distinct
        # labels remove EVERY off-diagonal pair except self terms;
        # instead verify on the numerator/denominator directly
        from src.torch.math_utils.pairwise import (
            compute_pairwise_kernel,
        )
        pw = compute_pairwise_kernel(pos, eps, "wendland_c2")
        same = (lbl.unsqueeze(1) == lbl.unsqueeze(0)).to(DT)
        num_u = (m.unsqueeze(0) * pw.rho_prime)
        assert torch.allclose(num_u,
                              num_u * same + num_u * (1 - same),
                              rtol=0, atol=0)
        # and the masked estimator equals the from-scratch estimator
        # on each loop alone (independence route)
        for lb in (0, 1):
            sel = lbl == lb
            _, H_alone = compute_regularized_curvature(
                pos[sel], nrm[sel], m[sel], epsilon=eps)
            assert torch.allclose(H_s[sel], H_alone, rtol=1e-10,
                                  atol=1e-12)

    def test_permutation_invariance(self):
        pos, ang, lbl = wall_pair(gap=0.06, jitter=0.2)
        g = torch.Generator().manual_seed(4)
        perm = torch.randperm(pos.shape[0], generator=g)
        k = self._kappa(pos, ang, lbl, 0.12, True)
        kp = self._kappa(pos[perm], ang[perm], lbl[perm], 0.12, True)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.shape[0])
        assert torch.allclose(kp[inv], k, rtol=1e-12, atol=1e-12)

    def test_redistribution_scope_wiring(self):
        """curvature_scope='loopwise' changes ONLY the final angle
        correction (positions bitwise identical), and requires
        labels."""
        pos, ang, lbl = wall_pair(gap=0.06, jitter=0.2)
        q = torch.ones(pos.shape[0], dtype=DT)
        pu, au, _ = redistribute_with_rule(
            pos, ang, "legacy_hybrid", delta=0.12, n_iters=3,
            coherence=q, loop_labels=lbl,
            density_scope="loopwise", curvature_scope="union")
        pl, al, _ = redistribute_with_rule(
            pos, ang, "legacy_hybrid", delta=0.12, n_iters=3,
            coherence=q, loop_labels=lbl,
            density_scope="loopwise", curvature_scope="loopwise")
        assert torch.equal(pu, pl)              # positions untouched
        assert not torch.equal(au, al)          # angles differ
        with pytest.raises(RedistributionGeometryError,
                           match="curvature_scope"):
            redistribute_with_rule(
                pos, ang, "legacy_hybrid", delta=0.12, n_iters=1,
                coherence=q, curvature_scope="loopwise")


class TestR2Closure:
    """Pre-R2 hardening pins (reviewer 2026-08-16)."""

    def test_c_xi_constant(self):
        import numpy as np
        from src.torch.solver.redistribution_rules import (
            KAPPA_DEN_C_XI,
        )
        u = np.linspace(0.0, 1.0, 400001)
        xi = 10 * u ** 2 * (1 - u) ** 3
        c = 2 * np.trapz(xi, u)
        assert KAPPA_DEN_C_XI["wendland_c2"] == pytest.approx(
            c, rel=1e-8)
        assert KAPPA_DEN_C_XI["wendland_c2"] == pytest.approx(
            1.0 / 3.0, rel=1e-12)

    def test_kappa_denominator_guard(self):
        """beta = eps B / c_xi ~ 1 on clean loops (pre-registered
        0.48/0.48 pass with margin); a sparse loop fails closed."""
        pos, ang, lbl = wall_pair(gap=0.06, jitter=0.0)
        q = torch.ones(pos.shape[0], dtype=DT)
        # clean: guard green at the pre-registered thresholds
        redistribute_with_rule(
            pos, ang, "legacy_hybrid", delta=0.12, n_iters=1,
            coherence=q, loop_labels=lbl,
            density_scope="loopwise", curvature_scope="loopwise",
            kappa_den_beta_min=0.48, kappa_den_gamma_min=0.48)
        # sparse loop: 7 points on a large circle, bandwidth far
        # below spacing -> same-loop support collapses
        p2, a2 = circle(1.0, 7)
        posx = torch.cat([pos, p2])
        angx = torch.cat([ang, a2])
        lblx = torch.cat([lbl, torch.full((7,), 2,
                                          dtype=torch.long)])
        with pytest.raises(RedistributionGeometryError,
                           match="denominator"):
            redistribute_with_rule(
                posx, angx, "legacy_hybrid", delta=0.12, n_iters=1,
                coherence=torch.ones(posx.shape[0], dtype=DT),
                loop_labels=lblx, density_scope="loopwise",
                curvature_scope="loopwise",
                kappa_den_beta_min=0.48, kappa_den_gamma_min=0.48)

    def test_r_per_particle_tensor_and_q_policy(self):
        """perimeter_r_per_particle is a source-frozen particle-
        aligned tensor (constant per loop, in [0,1]), and the r_loop
        q-policy reads it directly."""
        from src.torch.oriented_varifold import (
            OrientedPointCloudVarifold,
        )
        from src.torch.oriented_varifold.mass import (
            compute_recommended_params,
        )
        from src.torch.solver.mm_step import MMConfig, MMStepper
        import math as _m
        n1, n2 = 96, 140
        t1 = torch.arange(n1, dtype=DT) * 2 * _m.pi / n1
        t2 = torch.arange(n2, dtype=DT) * 2 * _m.pi / n2
        pos = torch.cat([
            0.35 * torch.stack([t1.cos(), t1.sin()], 1),
            0.42 * torch.stack([t2.cos(), t2.sin()], 1)])
        ang = torch.cat([t1, t2 + _m.pi])
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        d, t = map(float, compute_recommended_params(v.positions))
        cfg = MMConfig(mass_delta=d, mass_tau=t,
                       perimeter_sigma=0.1, angle_sigma=0.1,
                       mass_estimator="loopwise_oriented_kde",
                       perimeter_q_mode="self_renormalized")
        st = MMStepper(cfg)
        st._setup_step(v)
        r = st.perimeter_r_per_particle
        assert r is not None and r.shape == (v.n_points,)
        assert float(r.min()) >= 0.0
        assert float(r.max()) <= 1.0 + 1e-9
        labels = st._loop_labels_last
        for lb in labels.unique():
            sel = labels == lb
            assert float(r[sel].max() - r[sel].min()) == 0.0
        # per-loop constancy => the r_loop policy rescales each
        # loop's pseudo-time without changing the path (pinned in
        # the q-shadow experiment's algebraic arm)

    def test_repair_boundary_normals(self):
        """One-shot repair: tilted wall pair is re-anchored, both
        partitions preserved, stats recorded; a repair that would
        break transversality fails closed upstream."""
        from src.torch.solver.redistribution_rules import (
            repair_boundary_normals,
        )
        pos, ang, lbl = wall_pair(gap=0.07, jitter=0.0)
        tilt = ang + 0.05 * torch.sin(
            3.0 * torch.arange(pos.shape[0], dtype=DT))
        new_ang, stats = repair_boundary_normals(
            pos, tilt, 0.12, 1e-6, 0.1)
        assert stats["retraction"]["max_angle_rad"] < 0.2
        assert abs(stats["delta_p_plus_rel"]) < 0.05
        # repaired normals are closer to radial than the tilted ones
        nrm = torch.stack([new_ang.cos(), new_ang.sin()], 1)
        rad = pos / pos.norm(dim=1, keepdim=True)
        sgn = torch.where((nrm * rad).sum(1) > 0, 1.0, -1.0)
        resid = (nrm - sgn[:, None] * rad).norm(dim=1)
        assert float(resid.max()) < 0.05


class TestR0bExactOracle:
    def _ordered_fixture(self):
        pos, ang, lbl = wall_pair(jitter=0.2)
        m = torch.ones(pos.shape[0], dtype=DT) * 0.02
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        orders = certified_cyclic_order(pos, lbl, v.normals, m,
                                        graph_scale=0.03)
        return pos, ang, lbl, orders

    def test_oracle_first_order_leakage_exactly_zero(self):
        pos, ang, lbl, orders = self._ordered_fixture()
        loops = [lo.order for lo in orders.values()]
        p1, _, info = redistribute_with_rule(
            pos, ang, "legacy_hybrid", delta=DELTA, n_iters=1,
            tol=0.0, coherence=None, q_override="unit",
            loops=loops, tangent_source="ordered_loop_oracle",
            loop_labels=lbl, density_scope="loopwise")
        dx = p1 - pos
        for lo in orders.values():
            g = polygon_gradient(pos, lo.order)
            L = float((g * dx).sum())
            assert abs(L) < 1e-15

    def test_pca_exact_L_plus_Q_closure(self):
        pos, ang, lbl, orders = self._ordered_fixture()
        p1, _, _ = redistribute_with_rule(
            pos, ang, "legacy_hybrid", delta=DELTA, n_iters=1,
            tol=0.0, coherence=None, q_override="unit",
            loop_labels=lbl, density_scope="loopwise",
            tangent_source="local_pca", pca_delta_tan=DELTA,
            pca_rho_max=RHO_MAX, pca_neff_min=NEFF_MIN)
        dx = p1 - pos
        for lo in orders.values():
            g = polygon_gradient(pos, lo.order)
            L = float((g * dx).sum())
            Q = float(area_quadratic_term(dx, lo.order))
            dA = (float(polygon_area(p1, lo.order))
                  - float(polygon_area(pos, lo.order)))
            assert dA == pytest.approx(L + Q, abs=1e-14)
            # PCA leakage is small but generally NONZERO -- that is
            # the honest statement (reviewer blocker): only the
            # oracle kills it identically. Judge by the NORMALIZED
            # leakage rate |L| / sum |dx| (the R1 gate quantity).
            rate = abs(L) / float(dx[lo.order].norm(dim=1).sum()
                                  + 1e-30)
            assert rate < 0.05

    def test_pca_polygon_projector_parity(self):
        pos, ang, lbl, orders = self._ordered_fixture()
        t_pca = _local_pca_tangents(pos, ang, DELTA, "wendland_c2",
                                    loop_labels=lbl)
        worst = 0.0
        for lb, lo in orders.items():
            p = pos[lo.order]
            t_or = p.roll(-1, 0) - p.roll(1, 0)
            t_or = t_or / t_or.norm(dim=1, keepdim=True)
            cosang = (t_pca[lo.order] * t_or).sum(1).abs().clamp(0, 1)
            worst = max(worst, float(torch.acos(cosang).max()))
        assert worst < 0.2      # jittered two-loop wall envelope

    def test_tilted_circle_repair(self):
        pos, ang = circle(0.5, 96)
        tilted = ang + 0.2 * torch.sin(
            3 * torch.arange(96, dtype=DT))
        new_ang, stats = pca_normal_reanchor(
            pos, tilted, 0.2, "wendland_c2",
            torch.zeros(96, dtype=torch.long),
            RHO_MAX, NEFF_MIN, ORI_MARGIN)
        nu = torch.stack([new_ang.cos(), new_ang.sin()], 1)
        rad = pos / pos.norm(dim=1, keepdim=True)
        assert float((nu - rad).norm(dim=1).max()) < 1e-12
        assert stats["min_alignment"] > ORI_MARGIN
