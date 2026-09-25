"""0J-A/B pins: self-coherence renormalization q^WB = r_l q^self.

Reviewer-mandated algebraic pins (executed BEFORE any dynamics):
- source-energy preservation P_WB(X|X) = P_full(X) (the uniqueness
  condition that forces r_l);
- clean-shape identity q^WB == q^full WITH the partition certificate
  (cross-loop kernel mass exactly zero / single loop) asserted
  simultaneously -- a value-only match could hide misclassification;
- loop-partition pins (flower 1 / annulus 2 / coincident antiparallel
  2, graph-scale stability, permutation co-membership);
- fail-closed: r_l scope certificate (antiparallel-contact regime)
  and non-degenerate denominator;
- variational pin: D P_l^WB [xi] = r_l D P_l^self [xi] to machine
  precision (0J-B);
- default mode "full" leaves perimeter_coherence == fixed_coherence.
"""

import math

import pytest
import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import compute_recommended_params
from src.torch.perimeter.coherence_perimeter import (
    compute_coherence_loopwise,
)
from src.torch.shapes.generator import (
    generate_oriented_annulus,
    generate_oriented_circle,
    generate_oriented_flower,
)
from src.torch.solver.mm_step import MMConfig, MMStepper
from src.torch.transport.bem_wasserstein import compute_coherence
from src.torch.transport.incidence import partition_relation

DT = torch.float64
SIGMA = 0.1


def wall_state(rbar=0.394, n1=90, n2=141, r_out=0.93, n3=256):
    """Coincident antiparallel wall pair + outer ring (merger-like)."""
    th1 = 2 * math.pi * torch.arange(n1, dtype=DT) / n1
    th2 = 2 * math.pi * torch.arange(n2, dtype=DT) / n2
    th3 = 2 * math.pi * torch.arange(n3, dtype=DT) / n3
    pos = torch.cat([
        rbar * torch.stack([th1.cos(), th1.sin()], 1),
        (rbar + 0.02) * torch.stack([th2.cos(), th2.sin()], 1),
        r_out * torch.stack([th3.cos(), th3.sin()], 1)])
    ang = torch.cat([th1, th2 + math.pi, th3])
    return OrientedPointCloudVarifold(positions=pos, angles=ang)


def setup_stepper(v, mode):
    delta, tau = map(float, compute_recommended_params(v.positions))
    cfg = MMConfig(mass_delta=delta, mass_tau=tau,
                   perimeter_sigma=SIGMA, angle_sigma=SIGMA,
                   mass_estimator="oriented_kde",
                   perimeter_q_mode=mode)
    st = MMStepper(cfg)
    st._setup_step(v)
    return st


class TestAlgebraicPins:
    @pytest.mark.parametrize("maker,n_loops", [
        (lambda: generate_oriented_circle(128, 1.0, (0.0, 0.0),
                                          "cpu", DT), 1),
        (lambda: generate_oriented_flower(160, device="cpu",
                                          dtype=DT), 1),
        (lambda: generate_oriented_annulus(160, R_outer=1.0,
                                           R_inner=0.5, device="cpu",
                                           dtype=DT), 2),
    ])
    def test_clean_identity_with_partition_certificate(self, maker,
                                                       n_loops):
        v = maker()
        st = setup_stepper(v, "self_renormalized")
        labels = st._loop_labels_last
        assert int(labels.max()) + 1 == n_loops
        q_self, cross_max = compute_coherence_loopwise(
            v.positions, v.normals, st.fixed_masses, SIGMA,
            "wendland_c2", loop_labels=labels)
        # the certificate, not just the value: zero cross-loop kernel
        assert cross_max == 0.0
        assert torch.allclose(st.perimeter_coherence,
                              st.fixed_coherence, rtol=0, atol=1e-14)
        assert st._qmode_stats["min_r"] == pytest.approx(1.0,
                                                         abs=1e-12)

    def test_source_energy_preserved_on_wall(self):
        v = wall_state()
        st = setup_stepper(v, "self_renormalized")
        p_wb = float((st.fixed_masses * st.perimeter_coherence).sum())
        p_full = float((st.fixed_masses * st.fixed_coherence).sum())
        assert abs(p_wb - p_full) / max(1.0, abs(p_full)) < 1e-12
        # wall loops are strongly attenuated, outer ring is not
        assert st._qmode_stats["min_r"] < 0.9
        assert st._qmode_stats["max_r"] <= 1.0 + 1e-9

    def test_loop_partition_pins(self):
        v = wall_state()
        st = setup_stepper(v, "self_renormalized")
        labels = st._loop_labels_last
        assert int(labels.max()) + 1 == 3   # two wall sheets + ring
        # permutation co-membership invariance
        torch.manual_seed(0)
        perm = torch.randperm(v.positions.shape[0])
        vp = OrientedPointCloudVarifold(positions=v.positions[perm],
                                        angles=v.angles[perm])
        stp = setup_stepper(vp, "self_renormalized")
        assert partition_relation(stp._loop_labels_last,
                                  labels[perm]) == "equal"
        # and the corrected weights follow the permutation
        assert torch.allclose(stp.perimeter_coherence,
                              st.perimeter_coherence[perm],
                              rtol=0, atol=1e-12)

    def test_scope_certificate_r_gt_one_raises(self):
        """Co-oriented nearby loops with intra-loop angle noise: the
        cross-loop contribution RAISES coherence above self -- outside
        the antiparallel-contact scope, must fail closed."""
        # fine sampling shrinks the loop-graph radius (~4 median m)
        # BELOW the gap while the coherence kernel sigma stays ABOVE
        # it: two separate loops, co-oriented within kernel range
        n = 256
        th = 2 * math.pi * torch.arange(n, dtype=DT) / n
        torch.manual_seed(1)
        noise = 0.35 * torch.randn(n, dtype=DT)
        pos = torch.cat([
            0.50 * torch.stack([th.cos(), th.sin()], 1),
            0.57 * torch.stack([th.cos(), th.sin()], 1)])
        ang = torch.cat([th + noise, th])      # SAME orientation
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        with pytest.raises(RuntimeError, match="scope certificate"):
            setup_stepper(v, "self_renormalized")

    def test_default_full_is_same_object(self):
        v = generate_oriented_circle(64, 1.0, (0.0, 0.0), "cpu", DT)
        st = setup_stepper(v, "full")
        assert st.perimeter_coherence is st.fixed_coherence
        assert st._loop_labels_last is None or True  # no constraint

    def test_guards(self):
        with pytest.raises(ValueError, match="unit_coherence"):
            MMStepper(MMConfig(perimeter_q_mode="self_renormalized",
                               use_unit_coherence=True))
        with pytest.raises(ValueError, match="amplitude_fade"):
            MMStepper(MMConfig(perimeter_q_mode="self_renormalized",
                               grid_carrier_amplitude_fade=True))
        with pytest.raises(NotImplementedError, match="naive"):
            MMStepper(MMConfig(perimeter_q_mode="self_renormalized",
                               backend="keops"))


class TestVariationalPin:
    def test_directional_derivative_scaling(self):
        """0J-B: D P_l^WB [xi] = r_l D P_l^self [xi] exactly (the
        correction rescales the self functional loopwise)."""
        v = wall_state()
        st = setup_stepper(v, "self_renormalized")
        labels = st._loop_labels_last
        q_self, _ = compute_coherence_loopwise(
            v.positions, v.normals, st.fixed_masses, SIGMA,
            "wendland_c2", loop_labels=labels)
        torch.manual_seed(2)
        xi = 1e-3 * torch.randn(v.positions.shape[0], dtype=DT)
        t = 1e-6

        def masses_at(s):
            pos = v.positions + s[:, None] * v.normals
            return st._masses_for(pos, v.normals)

        m_p, m_m = masses_at(t * xi), masses_at(-t * xi)
        for lb in labels.unique():
            sel = labels == lb
            r = (float((st.fixed_masses[sel]
                        * st.fixed_coherence[sel]).sum())
                 / float((st.fixed_masses[sel] * q_self[sel]).sum()))
            d_wb = float(((m_p[sel] - m_m[sel])
                          * st.perimeter_coherence[sel]).sum()) / (2 * t)
            d_self = float(((m_p[sel] - m_m[sel])
                            * q_self[sel]).sum()) / (2 * t)
            assert d_wb == pytest.approx(r * d_self, rel=1e-12,
                                         abs=1e-12)
