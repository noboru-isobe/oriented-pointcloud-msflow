"""0K-A pins: loopwise-masked oriented mass (reviewer-amended set).

Guarantee under test (exact wording): cross-loop independence +
sheetwise H^1 consistency -- each certified loop's raw mass
approximates ITS OWN arclength and is never attenuated by another
loop. NOT "mass preservation" (perimeters move under the flow).

Pins:
 1. regular identity: separated loops -> m_loop == m_oriented
    (float64) WITH cross-loop kernel overlap exactly zero asserted
    simultaneously;
 2. cross-loop independence: moving loop 2 inside delta_KDE leaves
    loop 1's theta^loop and m_loop bitwise unchanged;
 3. foreign-loop cardinality invariance: changing only loop 2's
    sample count (delta fixed, tau re-resolved by the standard rule)
    leaves loop 1's masses invariant -- the global-N cancellation;
 4. global-N pin: unequal-count loops each recover their own true
    perimeter, improving under refinement;
 5. theta^or = theta^self + theta^cross to rounding;
 6. source-energy identity of the 0K + 0J composition:
    P_WB(X|X) = P_full(X) on the loopwise masses;
 7. cutoff audit: the mask causes no new chi_tau firings;
 8. gradient check: D_Y m_loop with frozen labels vs central
    differences;
 9. bootstrap certificate: l_pre ~ l_post pin, graph-scale-unstable
    fixture raises, synthetic pre/post mismatch raises;
10. constructor guards + dispatch parity (default paths bitwise).
"""

import math

import pytest
import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.loopwise_mass import (
    LoopwiseMassResolution,
    certified_loop_labels,
    resolve_loopwise_oriented_mass_source,
)
from src.torch.oriented_varifold.mass import (
    KERNEL_CONSTANTS,
    chi_tau,
    compute_kde_density_oriented,
    compute_masses_oriented,
    compute_masses_oriented_loopwise,
    compute_recommended_params,
    compute_recommended_params_oriented,
    oriented_density_cross_split,
    wendland_c2,
)
from src.torch.solver.mm_step import MMConfig, MMStepper
from src.torch.transport.incidence import (
    PartitionAmbiguousError,
    partition_relation,
)

DT = torch.float64
SIGMA = 0.1


def circle(R, n, inward=False, center=(0.0, 0.0)):
    t = torch.arange(n, dtype=DT) * (2 * math.pi / n)
    pos = torch.stack([R * t.cos() + center[0],
                       R * t.sin() + center[1]], 1)
    ang = t + (math.pi if inward else 0.0)
    return OrientedPointCloudVarifold(positions=pos, angles=ang)


def two_loops(R1=0.35, n1=96, R2=1.0, n2=256, inward2=False):
    a, b = circle(R1, n1), circle(R2, n2, inward=inward2)
    return OrientedPointCloudVarifold(
        positions=torch.cat([a.positions, b.positions]),
        angles=torch.cat([a.angles, b.angles]))


def cross_kernel_mass(v, labels, delta):
    d = torch.cdist(v.positions, v.positions)
    ker = wendland_c2(d / delta)
    cross = labels.unsqueeze(1) != labels.unsqueeze(0)
    return float(ker[cross].abs().max()) if cross.any() else 0.0


class TestLoopwiseMassPins:
    def test_regular_identity_with_zero_cross_overlap(self):
        """Pin 1: value identity AND its structural reason together."""
        v = two_loops()
        delta, tau = compute_recommended_params(v.positions)
        res = resolve_loopwise_oriented_mass_source(
            v.positions, v.normals, delta, tau)
        assert cross_kernel_mass(v, res.loop_labels_pre, delta) == 0.0
        m_or = compute_masses_oriented(v.positions, v.normals, delta,
                                       tau)
        assert torch.allclose(res.m_loop, m_or, rtol=1e-14, atol=0.0)
        assert float(res.theta_cross.abs().max()) == 0.0

    def test_cross_loop_independence_bitwise(self):
        """Pin 2: hole sheet entering delta_KDE range must not touch
        the disk loop's density or mass AT ALL (bitwise)."""
        n1 = 96
        v_far = two_loops(R1=0.35, n1=n1, R2=1.0, n2=140, inward2=True)
        delta, tau = compute_recommended_params_oriented(
            v_far.positions, v_far.normals)
        res_far = resolve_loopwise_oriented_mass_source(
            v_far.positions, v_far.normals, delta, tau)
        # move loop 2 to gap = 0.4 delta (deep inside the mass kernel)
        v_near = two_loops(R1=0.35, n1=n1, R2=0.35 + 0.4 * delta,
                           n2=140, inward2=True)
        res_near = resolve_loopwise_oriented_mass_source(
            v_near.positions, v_near.normals, delta, tau)
        assert int(res_near.loop_labels_pre.max()) + 1 == 2
        assert torch.equal(res_far.theta_self[:n1],
                           res_near.theta_self[:n1])
        assert torch.equal(res_far.m_loop[:n1], res_near.m_loop[:n1])
        # and the unmasked estimator DOES move there (the 0K target)
        m_or = compute_masses_oriented(v_near.positions,
                                       v_near.normals, delta, tau)
        assert not torch.allclose(m_or[:n1], res_near.m_loop[:n1],
                                  rtol=1e-6)

    def test_foreign_loop_cardinality_invariance(self):
        """Pin 3 (reviewer section 1): with delta FIXED and tau
        re-resolved by the standard rule tau = 2 k_min/(N C_eta
        delta), changing only the far loop's count leaves loop 1's
        masses invariant -- the global N cancels exactly."""
        n1, R1 = 96, 0.35
        delta = 0.1  # resolves BOTH loop spacings at both counts
        masses = []
        for n2 in (160, 320):
            v = two_loops(R1=R1, n1=n1, R2=1.0, n2=n2)
            N = v.n_points
            tau = 2 * 1.0 / (N * KERNEL_CONSTANTS["wendland_c2"]
                             * delta)
            res = resolve_loopwise_oriented_mass_source(
                v.positions, v.normals, delta, tau)
            masses.append(res.m_loop[:n1])
        assert torch.allclose(masses[0], masses[1], rtol=1e-12,
                              atol=0.0)

    def test_global_n_perimeter_consistency(self):
        """Pin 4: unequal-count loops each recover their own H^1,
        improving under the KDE asymptotics delta -> 0, N_l delta ->
        infinity (delta ~ N^{-1/2} schedule; the production delta ~
        N^{-1} rule keeps delta/spacing fixed, which freezes the
        quadrature term instead of shrinking it)."""
        errs = []
        delta0 = 0.12
        for scale in (1, 2):
            v = two_loops(R1=0.35, n1=96 * scale, R2=1.0,
                          n2=256 * scale)
            delta = delta0 / math.sqrt(scale)
            tau = 2 * 1.0 / (v.n_points
                             * KERNEL_CONSTANTS["wendland_c2"] * delta)
            res = resolve_loopwise_oriented_mass_source(
                v.positions, v.normals, delta, tau)
            per = {s["label"]: s["M_l"] for s in res.loop_stats}
            lab1 = int(res.loop_labels_pre[0])
            lab2 = int(res.loop_labels_pre[-1])
            e1 = abs(per[lab1] - 2 * math.pi * 0.35) / (2 * math.pi
                                                        * 0.35)
            e2 = abs(per[lab2] - 2 * math.pi * 1.0) / (2 * math.pi)
            errs.append(max(e1, e2))
        assert errs[0] < 0.01
        assert errs[1] < errs[0]

    def test_theta_split_identity(self):
        """Pin 5: theta^or = theta^self + theta^cross to rounding on
        a NEAR fixture (nonzero cross)."""
        v = two_loops(R1=0.35, n1=96, R2=0.40, n2=140, inward2=True)
        delta, tau = compute_recommended_params_oriented(
            v.positions, v.normals)
        labels = certified_loop_labels(
            v.positions, v.normals,
            float(compute_masses_oriented(
                v.positions, v.normals, delta, tau).median()))
        ts, tc = oriented_density_cross_split(
            v.positions, v.normals, delta, "wendland_c2",
            loop_labels=labels)
        th = compute_kde_density_oriented(v.positions, v.normals,
                                          delta)
        assert float(tc.max()) > 0.0
        assert torch.allclose(ts + tc, th, rtol=1e-13, atol=1e-16)

    def test_source_energy_identity_with_0j(self):
        """Pin 6: the 0K + 0J composition keeps P_WB(X|X) =
        P_full(X) on the LOOPWISE masses (full stepper setup)."""
        v = two_loops(R1=0.35, n1=96, R2=0.42, n2=140, inward2=True)
        delta, tau = map(float,
                         compute_recommended_params(v.positions))
        cfg = MMConfig(mass_delta=delta, mass_tau=tau,
                       perimeter_sigma=SIGMA, angle_sigma=SIGMA,
                       mass_estimator="loopwise_oriented_kde",
                       perimeter_q_mode="self_renormalized")
        st = MMStepper(cfg)
        st._setup_step(v)
        res = st._last_mass_loop_snapshot
        assert isinstance(res, LoopwiseMassResolution)
        assert torch.equal(st.fixed_masses, res.m_loop)
        p_full = float((st.fixed_masses * st.fixed_coherence).sum())
        p_wb = float((st.fixed_masses
                      * st.perimeter_coherence).sum())
        assert p_wb == pytest.approx(p_full, rel=1e-12)

    def test_cutoff_audit_no_new_firings(self):
        """Pin 7: the mask must not push any point below chi_tau on
        the wall-approach fixtures."""
        for gap_factor in (2.0, 0.4):
            v0 = two_loops(R1=0.35, n1=96, R2=1.0, n2=140,
                           inward2=True)
            delta, tau = compute_recommended_params_oriented(
                v0.positions, v0.normals)
            v = two_loops(R1=0.35, n1=96,
                          R2=0.35 + gap_factor * delta, n2=140,
                          inward2=True)
            res = resolve_loopwise_oriented_mass_source(
                v.positions, v.normals, delta, tau)
            assert res.cutoff_count_oriented == 0
            assert res.cutoff_count_loop == 0

    def test_gradient_check_frozen_labels(self):
        """Pin 8: autograd D_Y m_loop (labels frozen) vs central
        differences along a fixed random direction."""
        v = two_loops(R1=0.35, n1=48, R2=0.40, n2=64, inward2=True)
        delta, tau = compute_recommended_params_oriented(
            v.positions, v.normals)
        labels = resolve_loopwise_oriented_mass_source(
            v.positions, v.normals, delta, tau).loop_labels_pre
        g = torch.Generator(device="cpu").manual_seed(7)
        xi = torch.randn(v.positions.shape, generator=g, dtype=DT)
        xi = xi / xi.norm()
        w = torch.randn(v.n_points, generator=g, dtype=DT)

        def f(eps):
            pos = v.positions + eps * xi
            m = compute_masses_oriented_loopwise(
                pos, v.normals, labels, delta, tau)
            return (w * m).sum()

        pos = v.positions.clone().requires_grad_(True)
        m = compute_masses_oriented_loopwise(pos, v.normals, labels,
                                             delta, tau)
        (grad,) = torch.autograd.grad((w * m).sum(), pos)
        got = float((grad * xi).sum())
        h = 1e-6
        want = float((f(h) - f(-h)) / (2 * h))
        assert got == pytest.approx(want, rel=1e-6, abs=1e-10)

    def test_bootstrap_certificate_pin_and_failures(self, monkeypatch):
        """Pin 9: l_pre ~ l_post on the wall fixture; a synthetic
        pre/post mismatch raises; a graph-scale-unstable geometry
        raises from certified_loop_labels itself."""
        v = two_loops(R1=0.35, n1=96, R2=0.42, n2=140, inward2=True)
        delta, tau = compute_recommended_params_oriented(
            v.positions, v.normals)
        res = resolve_loopwise_oriented_mass_source(
            v.positions, v.normals, delta, tau)
        assert res.certificate_equal
        assert partition_relation(res.loop_labels_pre,
                                  res.loop_labels_post) == "equal"

        # synthetic pre/post mismatch -> fail-closed
        import src.torch.oriented_varifold.loopwise_mass as lm
        real = lm.certified_loop_labels
        calls = {"n": 0}

        def flaky(positions, normals, spacing, *a, **k):
            calls["n"] += 1
            labels = real(positions, normals, spacing, *a, **k)
            if calls["n"] == 2:  # the l_post call: merge everything
                return torch.zeros_like(labels)
            return labels

        monkeypatch.setattr(lm, "certified_loop_labels", flaky)
        with pytest.raises(PartitionAmbiguousError):
            lm.resolve_loopwise_oriented_mass_source(
                v.positions, v.normals, delta, tau)
        monkeypatch.undo()

    def test_graph_scale_unstable_raises(self):
        """Two CONCENTRIC CO-ORIENTED circles with radial gap between
        the 3.5x and 4.5x graph radii (side-by-side circles never
        trigger this: their facing sheets are anti-oriented, so the
        co-orientation condition already excludes those edges): the
        3-scale certificate must fail closed."""
        v0 = circle(0.35, 96)
        delta, tau = compute_recommended_params(v0.positions)
        m = compute_masses_oriented(v0.positions, v0.normals, delta,
                                    tau)
        spacing = float(m.median())
        # both outward: radially adjacent points are co-oriented, gap
        # = 4.0 * spacing -> split at 3.5x, connected at 4.5x
        b = circle(0.35 + 4.0 * spacing, 96)
        v = OrientedPointCloudVarifold(
            positions=torch.cat([v0.positions, b.positions]),
            angles=torch.cat([v0.angles, b.angles]))
        with pytest.raises(PartitionAmbiguousError):
            certified_loop_labels(v.positions, v.normals, spacing)

    def test_constructor_guards_and_dispatch(self):
        """Pin 10: bandwidth guards, missing-labels guard, and
        dispatch parity for the loopwise path."""
        with pytest.raises(ValueError, match="global"):
            MMStepper(MMConfig(
                mass_estimator="loopwise_oriented_kde",
                mass_bandwidth_type="per_point_knn"))
        with pytest.raises(ValueError, match="adaptive"):
            MMStepper(MMConfig(
                mass_estimator="loopwise_oriented_kde",
                mass_delta_adaptive=True))
        v = two_loops()
        delta, tau = map(float,
                         compute_recommended_params(v.positions))
        cfg = MMConfig(mass_delta=delta, mass_tau=tau,
                       mass_estimator="loopwise_oriented_kde")
        st = MMStepper(cfg)
        st._mass_delta_for_kde = delta
        st._mass_tau_for_kde = tau
        with pytest.raises(RuntimeError, match="loop"):
            st._masses_for(v.positions, v.normals)
        res = resolve_loopwise_oriented_mass_source(
            v.positions, v.normals, delta, tau)
        got = st._masses_for(v.positions, v.normals,
                             loop_labels=res.loop_labels_pre)
        assert torch.equal(got, res.m_loop)

    def test_coincident_antiparallel_sheets_own_arclength(self):
        """M0 sibling: at distance ZERO, unequal-count antiparallel
        sheets are separate certified loops (proximity AND
        co-orientation), and each loopwise mass recovers its OWN
        arclength -- never attenuated by the opposite sheet, at any
        angular offset (the omega-residual channel is identically
        absent)."""
        n1, n2, rbar = 90, 141, 0.394
        L = 2 * math.pi * rbar
        t1 = torch.arange(n1, dtype=DT) * (2 * math.pi / n1)
        t2 = torch.arange(n2, dtype=DT) * (2 * math.pi / n2)
        pos = torch.cat([
            rbar * torch.stack([t1.cos(), t1.sin()], 1),
            rbar * torch.stack([t2.cos(), t2.sin()], 1)])
        ang = torch.cat([t1, t2 + math.pi])
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        delta, tau = compute_recommended_params_oriented(
            v.positions, v.normals)
        res = resolve_loopwise_oriented_mass_source(
            v.positions, v.normals, delta, tau)
        assert int(res.loop_labels_pre.max()) + 1 == 2
        for s in res.loop_stats:
            assert abs(s["M_l"] - L) / L < 0.02
        assert res.cutoff_count_loop == 0

    def test_scalar_bandwidth_required(self):
        v = two_loops()
        delta, tau = compute_recommended_params(v.positions)
        with pytest.raises(ValueError, match="scalar"):
            resolve_loopwise_oriented_mass_source(
                v.positions, v.normals,
                torch.full((v.n_points,), float(delta), dtype=DT),
                tau)
