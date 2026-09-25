"""P1-0 pins: Phase P dumbbell generators (Cassini + C^2 capsule).

Checks (plan + reviewer corrections): analytic area/neck width
agreement, outward normals (divergence-theorem area), spacing CV,
junction curvature CONTINUITY (kappa increments consistent with a
bounded d kappa/ds -- C^2, no jump), P_eff computed from the profile
with the fixed convention, area normalization, perimeter budget.
"""

import math

import pytest
import torch

from src.torch.shapes.dumbbell import (
    CapsuleProfile,
    build_capsule_dumbbell,
    build_cassini,
    p_eff_of_profile,
    perimeter_budget,
)

DT = torch.float64
H_REF = 2 * math.pi / 256
A0 = math.pi * 0.8


@pytest.fixture(scope="module")
def capsule():
    torch.set_default_dtype(DT)
    return build_capsule_dumbbell(R_l=0.55, R_r=0.55, L=2.0, w=0.29,
                                  h=H_REF, area_target=A0)


@pytest.fixture(scope="module")
def cassini():
    torch.set_default_dtype(DT)
    return build_cassini(a=0.98, b=1.0, h=H_REF, area_target=A0)


class TestCapsule:
    def test_area_normalized(self, capsule):
        assert abs(capsule["area"] - A0) / A0 < 2e-4

    def test_divergence_theorem_area(self, capsule):
        pos, ang = capsule["positions"], capsule["angles"]
        n = torch.stack([ang.cos(), ang.sin()], 1)
        m = (pos.roll(-1, 0) - pos.roll(1, 0)).norm(dim=1) / 2
        A_div = float(0.5 * (m * (pos * n).sum(-1)).sum())
        assert abs(A_div - capsule["area"]) / capsule["area"] < 1e-3

    def test_spacing_cv(self, capsule):
        seg = (capsule["positions"].roll(-1, 0)
               - capsule["positions"]).norm(dim=1)
        assert float(seg.max() / seg.min()) < 1.05

    def test_neck_width_exact(self, capsule):
        pr = capsule["profile"]
        assert abs(2 * pr.H(0.0) - capsule["w"]) < 1e-14
        # measured on the cloud: min |y| gap at the neck center strip
        pos = capsule["positions"]
        sel = pos[:, 0].abs() < capsule["L"] / 4
        top = pos[sel & (pos[:, 1] > 0)][:, 1].min()
        bot = pos[sel & (pos[:, 1] < 0)][:, 1].max()
        assert abs(float(top - bot) - capsule["w"]) < 1e-6

    def test_c2_junction_continuity(self, capsule):
        # kappa along the top graph is CONTINUOUS: adjacent-sample
        # increments must be explained by the bounded derivative
        # (a C^1-only fillet would show an O(1) jump independent of
        # the sampling density)
        n_samp = 20000
        pr = capsule["profile"]
        xl, xr = pr.graph_range()
        ds_max = (xr - xl) / n_samp * math.sqrt(2.0)
        assert capsule["kappa_jump_max"] <= \
            2.0 * capsule["dkappa_ds_max"] * ds_max

    def test_profile_c2_at_blend_ends(self):
        pr = CapsuleProfile(0.5, 0.5, 1.0, 0.3)
        for x0 in (pr.x_n, pr.r["x_m"]):
            for f in (pr.H, pr.dH, pr.ddH):
                lo = f(x0 - 1e-7)
                hi = f(x0 + 1e-7)
                assert abs(hi - lo) < 1e-4 * max(1.0, abs(hi)), \
                    (f.__name__, x0, lo, hi)

    def test_p_eff_convention_and_tunability(self, capsule):
        pr = capsule["profile"]
        # convention: H(ell) = 1.5 w/2 at the bisection point
        pe = capsule["p_eff_right"]
        assert pe > 4.0                      # this fixture: >4 class
        assert abs(capsule["p_eff_left"] - pe) < 1e-6   # symmetric
        # a long-blend short-neck fixture lands below 1
        low = build_capsule_dumbbell(R_l=0.3, R_r=0.3, L=0.3,
                                     w=0.29, h=H_REF, area_target=A0)
        assert low["p_eff_right"] < 1.0, low["p_eff_right"]

    def test_asymmetric_sides(self):
        c = build_capsule_dumbbell(R_l=0.35, R_r=0.7, L=2.0, w=0.29,
                                   h=H_REF, area_target=A0)
        assert c["p_eff_left"] != c["p_eff_right"]
        pos = c["positions"]
        # right lobe bigger
        assert float(pos[:, 0].max()) > -float(pos[:, 0].min()) * 0.8

    def test_perimeter_budget(self, capsule):
        assert capsule["perimeter"] >= perimeter_budget(A0)


class TestCassini:
    def test_area_and_width(self, cassini):
        assert abs(cassini["area"] - A0) / A0 < 2e-4
        a, b = cassini["a"], cassini["b"]
        assert abs(cassini["w"] - 2 * math.sqrt(b * b - a * a)) < 1e-12
        # cloud-measured neck width
        pos = cassini["positions"]
        sel = pos[:, 0].abs() < cassini["w"]
        top = pos[sel & (pos[:, 1] > 0)][:, 1].min()
        bot = pos[sel & (pos[:, 1] < 0)][:, 1].max()
        assert abs(float(top - bot) - cassini["w"]) < 2e-3

    def test_outward_normals(self, cassini):
        pos, ang = cassini["positions"], cassini["angles"]
        n = torch.stack([ang.cos(), ang.sin()], 1)
        m = (pos.roll(-1, 0) - pos.roll(1, 0)).norm(dim=1) / 2
        A_div = float(0.5 * (m * (pos * n).sum(-1)).sum())
        assert abs(A_div - cassini["area"]) / cassini["area"] < 1e-3

    def test_selected_cassini_is_below_budget_negative_control(
            self, cassini):
        # a/b = 0.98 sits below the symmetric-split budget: kept as
        # the NEGATIVE-energy-budget control (corrigendum 2026-08-26:
        # the earlier claim "near-pinch Cassini approaches two
        # tangent circles / sits at the budget" was WRONG)
        assert cassini["perimeter"] < perimeter_budget(A0)

    def test_lemniscate_limit_exceeds_budget(self):
        # b -> a limit of the Cassini family is the BERNOULLI
        # LEMNISCATE (not two tangent circles): P_lem/P_split =
        # sqrt(2 pi) Gamma(1/4)/Gamma(3/4) / (4 sqrt(pi)) ~ 1.04605
        # > 1, so a budget crossing exists inside the family and a
        # budget-positive near-pinch Cassini is a VALID pinch pilot
        a_lem = math.sqrt(A0 / 2)
        P_lem = math.sqrt(2 * math.pi) * a_lem \
            * math.gamma(0.25) / math.gamma(0.75)
        ratio = P_lem / perimeter_budget(A0)
        assert abs(ratio - 1.04605) < 1e-4
        # generator reproduces the limit trend: a/b = 0.999 is above
        c = build_cassini(a=0.999, b=1.0, h=H_REF, area_target=A0)
        assert c["perimeter"] > perimeter_budget(A0)

    def test_pilot_cassini_preselected(self):
        # PRE-REGISTERED pilot Cassini (static quadrature only, no
        # dynamics): a/b = 0.998 -- budget margin >= 1% AND
        # w0 >= 1.5 w* (w* = 0.075 admissibility window)
        c = build_cassini(a=0.998, b=1.0, h=H_REF, area_target=A0)
        assert c["perimeter"] >= 1.01 * perimeter_budget(A0)
        assert c["w"] >= 1.5 * 0.075


class TestPilotResolution:
    """Reviewer 2026-08-26 (third verdict, section 3): high-P_eff
    fixtures must be RESOLVED, not curvature-concentration artifacts.
    Pins on the pilot capsule class (moderate P_eff > 4, derived
    blend):
    simple curve, particles per blend, h -> h/2 stability of
    (P_eff, P0, A0, max kappa), cloud-vs-analytic curvature."""

    def _pilot(self, h):
        return build_capsule_dumbbell(R_l=0.55, R_r=0.55, L=1.0,
                                      w=0.29, h=h, area_target=A0)

    def test_simple_and_blend_particles(self):
        c = self._pilot(H_REF)
        from src.torch.transport.loop_geometry import (
            _has_self_intersection,
        )
        assert not _has_self_intersection(c["positions"])
        pr = c["profile"]
        pos = c["positions"]
        in_blend = ((pos[:, 0].abs() >= pr.x_n)
                    & (pos[:, 0].abs() <= abs(pr.r["x_m"]))
                    & (pos[:, 1] > 0))
        assert int(in_blend.sum()) >= 2 * 10   # >= 10 per blend side

    def test_h_refinement_stability(self):
        c1 = self._pilot(H_REF)
        c2 = self._pilot(H_REF / 2)
        for k in ("p_eff_right", "perimeter", "area"):
            assert abs(c1[k] - c2[k]) / abs(c1[k]) < 5e-3, (k, c1[k],
                                                            c2[k])
        k1 = float(c1["kappa"].abs().max())
        k2 = float(c2["kappa"].abs().max())
        assert abs(k1 - k2) / k1 < 0.05, (k1, k2)

    def test_cloud_vs_analytic_curvature(self):
        c = self._pilot(H_REF)
        pos, kap = c["positions"], c["kappa"]
        n = pos.shape[0]
        a, b, cc = pos.roll(1, 0), pos, pos.roll(-1, 0)
        cr = (b - a)[:, 0] * (cc - a)[:, 1] \
            - (b - a)[:, 1] * (cc - a)[:, 0]
        den = (b - a).norm(dim=1) * (cc - b).norm(dim=1) \
            * (cc - a).norm(dim=1)
        k_cloud = 2 * cr / den.clamp_min(1e-30)
        sel = kap.abs() > 0.2      # away from the flat neck
        rel = float(((k_cloud - kap)[sel]).abs().max()
                    / kap[sel].abs().max())
        assert rel < 0.08, rel
