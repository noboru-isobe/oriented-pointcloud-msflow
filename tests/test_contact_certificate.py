"""0L-B1 pins: contact/quotient certificate.

- eps_cur2 (squared-energy ratio): antiparallel coincident -> 0,
  uncorrelated far apart -> ~1, co-oriented coincident -> 2;
  denominator guard; PSD clamp
- directional metrics: unequal-count symmetry (anti max of both
  directions), gap90/h similarity invariance, local contact -> low
  coverage
- uniqueness: triple -> ambiguous; production requires bookkeeping
  maps (None -> raise); fixtures may opt in
- calibrated defaults: positive family certified, negative family
  rejected through the intended channel, sigma sweep agrees
- shared Gram helper == the former inline t_wall formula (bitwise)
"""

import json
import math

import pytest
import torch

from src.torch.transport.contact_certificate import (
    ContactCertificateError,
    ContactThresholds,
    directional_gap_metrics,
    evaluate_all_candidates,
    evaluate_contact_reference,
    find_contact_candidates,
    mollified_current_energy,
    pair_current_cancellation,
)

DT = torch.float64
SIG = 0.1


def circle(R, n, center=(0.0, 0.0), inward=False, phase=0.0,
           mass_scale=1.0):
    t = torch.arange(n, dtype=DT) * 2 * math.pi / n + phase
    pos = torch.stack([center[0] + R * t.cos(), center[1] + R * t.sin()], 1)
    ang = t + (math.pi if inward else 0.0)
    m = torch.full((n,), mass_scale * 2 * math.pi * R / n, dtype=DT)
    return pos, ang, m


def assemble(parts):
    pos = torch.cat([p for p, _, _ in parts])
    ang = torch.cat([a for _, a, _ in parts])
    m = torch.cat([mm for _, _, mm in parts])
    lbl = torch.cat([torch.full((p.shape[0],), i, dtype=torch.long)
                     for i, (p, _, _) in enumerate(parts)])
    nor = torch.stack([ang.cos(), ang.sin()], 1)
    return pos, nor, m, lbl


def idx(lbl, k):
    return (lbl == k).nonzero().flatten()


class TestCurrentCancellation:
    def test_three_regimes(self):
        pos, nor, m, lbl = assemble([circle(0.35, 141),
                                     circle(0.35, 141, inward=True)])
        c = pair_current_cancellation(pos, nor, m, SIG, idx(lbl, 0),
                                      idx(lbl, 1))
        assert c["eps_cur2"] < 1e-12
        pos, nor, m, lbl = assemble([circle(0.35, 141), circle(0.35, 141)])
        c = pair_current_cancellation(pos, nor, m, SIG, idx(lbl, 0),
                                      idx(lbl, 1))
        assert abs(c["eps_cur2"] - 2.0) < 1e-12
        pos, nor, m, lbl = assemble([circle(0.35, 141, center=(-1.5, 0)),
                                     circle(0.35, 141, center=(1.5, 0),
                                            inward=True)])
        c = pair_current_cancellation(pos, nor, m, SIG, idx(lbl, 0),
                                      idx(lbl, 1))
        assert abs(c["eps_cur2"] - 1.0) < 0.05

    def test_denominator_guard(self):
        pos, nor, m, lbl = assemble([circle(0.35, 141),
                                     circle(0.35, 141, inward=True)])
        c = pair_current_cancellation(pos, nor, m * 1e-9, SIG, idx(lbl, 0),
                                      idx(lbl, 1), energy_min=1.0)
        assert c["evaluable"] is False and c["eps_cur2"] is None

    def test_energy_matches_inline_formula(self):
        pos, nor, m, lbl = assemble([circle(0.35, 90),
                                     circle(0.4133, 141, inward=True)])
        e = mollified_current_energy(pos, nor, m, SIG,
                                     torch.arange(pos.shape[0]))
        d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
        ker = torch.exp(-d2 / (4 * SIG * SIG)) / (4 * math.pi * SIG * SIG)
        ref = float(((m[:, None] * m[None, :]) * (nor @ nor.T) * ker).sum())
        assert e == max(ref, 0.0)


class TestGeometry:
    def test_similarity_invariance_and_directionality(self):
        a = assemble([circle(0.35, 90), circle(0.35, 141, inward=True)])
        b = assemble([circle(0.21, 90), circle(0.21, 141, inward=True)])
        ma = directional_gap_metrics(a[0], a[2], idx(a[3], 0), idx(a[3], 1),
                                     float(a[2][idx(a[3], 0)].median()), 1.5)
        mb = directional_gap_metrics(b[0], b[2], idx(b[3], 0), idx(b[3], 1),
                                     float(b[2][idx(b[3], 0)].median()), 1.5)
        assert abs(ma["gap90_over_h"] - mb["gap90_over_h"]) < 1e-9
        assert ma["g90_ab"] != ma["g90_ba"]         # directional values kept

    def test_local_contact_low_coverage(self):
        pos, nor, m, lbl = assemble([circle(0.35, 90),
                                     circle(0.45, 141, center=(0.0999, 0.0),
                                            inward=True)])
        cs = evaluate_all_candidates(pos, nor, m, lbl, None, None, SIG,
                                     allow_unrestricted=True)
        assert len(cs) == 1
        assert cs[0].metrics["coverage"] < 0.4
        assert "coverage" in cs[0].failing and not cs[0].certified


class TestUniqueness:
    def test_triple_is_ambiguous(self):
        pos, nor, m, lbl = assemble([circle(0.35, 141),
                                     circle(0.35, 141, inward=True),
                                     circle(0.35, 141, inward=True,
                                            phase=math.pi / 141)])
        cs = find_contact_candidates(pos, m, lbl, None, None,
                                     ContactThresholds(),
                                     allow_unrestricted=True)
        assert cs and all(c.ambiguous for c in cs)
        certs = evaluate_all_candidates(pos, nor, m, lbl, None, None, SIG,
                                        allow_unrestricted=True)
        assert all(not c.certified and "uniqueness" in c.failing
                   for c in certs)

    def test_production_requires_bookkeeping(self):
        pos, nor, m, lbl = assemble([circle(0.35, 141),
                                     circle(0.35, 141, inward=True)])
        with pytest.raises(ContactCertificateError):
            find_contact_candidates(pos, m, lbl, None, None,
                                    ContactThresholds())
        # same grid component, different bulks -> candidate
        cs = find_contact_candidates(pos, m, lbl, {0: 0, 1: 0}, {0: 0, 1: 1},
                                     ContactThresholds())
        assert len(cs) == 1 and cs[0].bulk_pair == (0, 1)
        # same bulk -> excluded from State-II candidates
        assert find_contact_candidates(pos, m, lbl, {0: 0, 1: 0},
                                       {0: 0, 1: 0},
                                       ContactThresholds()) == []


class TestCalibratedDefaults:
    def _pos(self):
        R = 0.35
        h = 2 * math.pi * R / 90
        return [
            [circle(R, 141), circle(R, 141, inward=True)],
            [circle(R, 90), circle(R, 200, inward=True)],
            [circle(R, 141), circle(R, 141, inward=True, phase=math.pi / 141)],
            [circle(R, 90), circle(R + 0.5 * h, 141, inward=True)],
            [circle(R, 90), circle(R + 1.0 * h, 141, inward=True)],
            [circle(0.6 * R, 141), circle(0.6 * R, 141, inward=True)],
        ]

    def _neg(self):
        R = 0.35
        return [
            ("gap90_over_h", [circle(R, 90),
                              circle(R + 0.5 * SIG, 141, inward=True)]),
            ("anti", [circle(R, 141), circle(R, 141)]),
            ("mass", [circle(R, 141), circle(R, 141, inward=True,
                                              mass_scale=3.0)]),
            ("coverage", [circle(R, 90),
                          circle(0.45, 141, center=(0.0999, 0.0),
                                 inward=True)]),
            ("gap90_over_h", [circle(R, 90),
                              circle(R + 0.02, 141, center=(0.015, 0.0),
                                     inward=True)]),
        ]

    def test_positive_certified_negative_rejected(self):
        for parts in self._pos():
            pos, nor, m, lbl = assemble(parts)
            cs = evaluate_all_candidates(pos, nor, m, lbl, None, None, SIG,
                                         allow_unrestricted=True)
            assert len(cs) == 1 and cs[0].certified, cs[0].failing
            eps = [v["eps_cur2"] for v in cs[0].current.values()]
            assert max(eps) <= ContactThresholds().current_residual2_max
        for channel, parts in self._neg():
            pos, nor, m, lbl = assemble(parts)
            cs = evaluate_all_candidates(pos, nor, m, lbl, None, None, SIG,
                                         allow_unrestricted=True)
            assert len(cs) == 1 and not cs[0].certified
            assert channel in cs[0].failing, (channel, cs[0].failing)


class TestB2Hardening:
    def test_mapping_completeness_raises(self):
        pos, nor, m, lbl = assemble([circle(0.35, 141),
                                     circle(0.35, 141, inward=True)])
        with pytest.raises(ContactCertificateError, match="miss loops"):
            find_contact_candidates(pos, m, lbl, {0: 0}, {0: 0, 1: 1},
                                    ContactThresholds())

    def test_psd_guard_raises_beyond_rounding(self, monkeypatch):
        import src.torch.transport.contact_certificate as cc
        real = cc.mollified_current_gram

        def fake(pos, nor, m, sigma, ia, ib):
            if ia is ib or torch.equal(ia, ib):
                return 1.0
            return 5.0          # |E_ab| > sqrt(E_a E_b): not PSD
        monkeypatch.setattr(cc, "mollified_current_gram", fake)
        pos, nor, m, lbl = assemble([circle(0.35, 20),
                                     circle(0.35, 20, inward=True)])
        with pytest.raises(ContactCertificateError, match="not PSD"):
            cc.pair_current_cancellation(pos, nor, m, SIG, idx(lbl, 0),
                                         idx(lbl, 1))
        monkeypatch.setattr(cc, "mollified_current_gram", real)

    def test_transport_loop_maps(self):
        from src.torch.transport.contact_certificate import (
            transport_loop_maps,
        )
        src = torch.tensor([0, 0, 1, 1, 2, 2])
        dst = torch.tensor([5, 5, 3, 3, 9, 9])          # relabeled
        out = transport_loop_maps(src, dst, {"g": {0: 0, 1: 0, 2: 1},
                                             "b": {0: 0, 1: 1, 2: 1}})
        assert out["g"] == {5: 0, 3: 0, 9: 1} and out["b"] == {5: 0, 3: 1,
                                                                 9: 1}
        bad = torch.tensor([5, 5, 3, 9, 9, 9])          # loop 1 split
        with pytest.raises(ContactCertificateError):
            transport_loop_maps(src, bad, {"g": {0: 0, 1: 0, 2: 1}})

    def test_0k_mass_parity_on_key_fixtures(self):
        """Estimator parity (reviewer B2-0): the 0K loopwise masses,
        with the production bandwidth semantics (delta from the coarse
        loop, tau with the global N), classify the key positive /
        negative fixtures exactly like the oracle arclength masses."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"
                               / "experiments"))
        import contact_certificate_calibration as cal
        R = 0.35
        h = 2 * math.pi * R / 90
        cases = [
            (True, [cal.circle(R, 90), cal.circle(R, 141, inward=True)]),
            (True, [cal.circle(R, 90), cal.circle(R + h, 141, inward=True)]),
            (False, [cal.circle(R, 90),
                     cal.circle(R + 0.5 * SIG, 141, inward=True)]),
            (False, [cal.circle(R, 90),
                     cal.circle(0.45, 141, center=(0.0999, 0.0),
                                inward=True)]),
        ]
        for expect, parts in cases:
            for mode in ("oracle", "0k"):
                cal.MASS_MODE = mode
                r = cal.measure("x", parts)
                assert r["candidates"] == 1
                m = r["metrics"]
                T = ContactThresholds()
                ok = (m["gap90_over_h"] <= T.gap90_over_h_max
                      and m["coverage"] >= T.coverage_min
                      and m["anti"] <= T.anti_max
                      and m["mass_residual"] <= T.mass_max
                      and m["eps_cur2_max"] <= T.current_residual2_max)
                assert ok is expect, (mode, expect, m)
        cal.MASS_MODE = "oracle"


class TestMarginComparisonAmendment:
    def test_normalized_margins_and_rule(self):
        from src.torch.transport.contact_certificate import (
            MARGIN_COMPARE_TOL,
            margins_not_worse,
            normalized_margins,
        )
        pos, nor, m, lbl = assemble([circle(0.35, 90),
                                     circle(0.35 + 0.5 * 2 * math.pi * 0.35
                                            / 90, 141, inward=True)])
        c = evaluate_all_candidates(pos, nor, m, lbl, None, None, SIG,
                                    allow_unrestricted=True)[0].as_dict()
        nm = normalized_margins(c)
        assert set(nm) == {"gap90_over_h", "anti", "mass", "current",
                           "coverage"}
        assert all(0.0 <= v <= 1.0 for v in nm.values())
        # a drop copy with an anti-margin deficit of 1e-8 (relative
        # 2e-7 of the threshold) passes; a deficit of 1e-3 fails
        d = json.loads(json.dumps(c))
        d["margins"]["anti"] -= 1e-8
        ok, table = margins_not_worse(c, c, d)
        assert ok and table["anti"]["deficit"] < MARGIN_COMPARE_TOL
        d["margins"]["anti"] -= 1e-3
        ok, table = margins_not_worse(c, c, d)
        assert not ok and not table["anti"]["passed"]

    def test_recorded_768_failure_would_pass_under_amendment(self):
        """Audit (NOT a retroactive PASS): the archived two-arm record
        of the 768 switch at step 1921 fails only the original absolute
        1e-9 anti tolerance and satisfies eq. (2.1)."""
        from pathlib import Path
        from src.torch.transport.contact_certificate import (
            margins_not_worse,
        )
        p = Path("results/exact_merger/"
                 "exact_merger_b2_768c_quotient_failure.json")
        if not p.exists():
            pytest.skip("archived failure record not present")
        d = json.loads(p.read_text())
        assert d["gates"]["margins_not_worse"] is False
        ok, table = margins_not_worse(d["certificate_source"],
                                      d["keep"]["certificate"],
                                      d["drop"]["certificate"])
        assert ok, table
        assert abs(table["anti"]["deficit"]) < 1e-6

    def test_fixture_classification_invariant_under_adverse_tolerance(self):
        """Every B1 fixture keeps its classification when every
        threshold is moved adversely by the comparison tolerance."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"
                               / "experiments"))
        import contact_certificate_calibration as cal
        from src.torch.transport.contact_certificate import (
            MARGIN_COMPARE_TOL as tol,
        )
        T = ContactThresholds()
        P, N = cal.fixtures()
        for name, parts in P + [(n, p) for n, p, _ in N]:
            r = cal.measure(name, parts)
            if r.get("candidates", 0) == 0 or r["candidate"]["ambiguous"]:
                continue
            m = r["metrics"]

            def cls(f):
                return (m["gap90_over_h"] <= T.gap90_over_h_max * (1 - f)
                        and m["coverage"] >= T.coverage_min
                        + f * (1 - T.coverage_min)
                        and m["anti"] <= T.anti_max * (1 - f)
                        and m["mass_residual"] <= T.mass_max * (1 - f)
                        and m["eps_cur2_max"]
                        <= T.current_residual2_max * (1 - f))
            assert cls(0.0) == cls(tol) == cls(-tol), name


class TestContactReference:
    """0N-4 pins (reviewer 2026-08-18): the State III persistence
    invariant evaluates the FROZEN switch semantics (h_*, sigma_*,
    thresholds) on the stored particle partition -- distinct from the
    B1 entry certificate, whose moving live scale produced the 0N-3
    false positive on the shrinking fixed-N ghost pair."""

    def _reference(self):
        # switch-like certified state: coarse loop + antiparallel fine
        # loop at one-spacing offset (B1 positive family member)
        R = 0.35
        h = 2 * math.pi * R / 90
        pos, nor, m, lbl = assemble([circle(R, 90),
                                     circle(R + h, 141, inward=True)])
        cs = evaluate_all_candidates(pos, nor, m, lbl, None, None, SIG,
                                     allow_unrestricted=True)
        assert len(cs) == 1 and cs[0].certified
        c = cs[0]
        return (c.metrics["h_ab"], c.metrics["coverage_radius"],
                c.thresholds,
                max(c.metrics["g90_ab"], c.metrics["g90_ba"]))

    def test_differential_ghost_shrink_false_positive(self):
        """NOT a homothety (a true similarity leaves g/h invariant):
        the ghost radii and spacing shrink FASTER than the pair gap,
        so the live ratio crosses the threshold while the absolute gap
        decreases. The switch-scale hard invariant must PASS and the
        live ratio must be recorded as shadow telemetry."""
        h_star, cov_r, thr, g90_ref = self._reference()
        tau_g = thr["gap90_over_h_max"]
        g = 0.021                       # < g90_ref ~ h = 0.0244
        pos, nor, m, lbl = assemble([circle(0.23, 90),
                                     circle(0.23 + g, 141, inward=True)])
        cert = evaluate_contact_reference(
            pos, nor, m, lbl == 0, lbl == 1, h_star, cov_r, SIG, thr,
            (0, 1), (0, 1))
        g90_now = max(cert.metrics["g90_ab"], cert.metrics["g90_ba"])
        assert g90_now < g90_ref                     # absolute gap fell
        assert cert.metrics["h_ab_live"] < g90_now / tau_g < h_star
        assert cert.metrics["gap90_over_h_live"] > tau_g
        assert cert.metrics["gap90_over_h_switch"] < tau_g
        # fixed key semantics: the plain keys are the hard switch scale
        assert cert.metrics["gap90_over_h"] \
            == cert.metrics["gap90_over_h_switch"]
        assert cert.metrics["h_ab"] == h_star
        assert cert.metrics["coverage"] \
            == cert.metrics["coverage_switch_scale"]
        assert cert.certified, cert.failing

    def test_true_reseparation_fails(self):
        h_star, cov_r, thr, _ = self._reference()
        g = thr["gap90_over_h_max"] * h_star * 1.4
        pos, nor, m, lbl = assemble([circle(0.23, 90),
                                     circle(0.23 + g, 141, inward=True)])
        cert = evaluate_contact_reference(
            pos, nor, m, lbl == 0, lbl == 1, h_star, cov_r, SIG, thr,
            (0, 1), (0, 1))
        assert not cert.certified
        assert "gap90_over_h" in cert.failing

    def test_current_uncancellation_alone_fails(self):
        """Current-specific fixture (reviewer sec. 7): an m=2 mass
        modulation with preserved total leaves geometry, anti-alignment
        and the mass residual intact and degrades ONLY the mollified
        current cancellation."""
        h_star, cov_r, thr, _ = self._reference()
        R = 0.35
        pa, aa, ma_ = circle(R, 90)
        pb, ab, mb_ = circle(R, 141, inward=True)
        t = torch.arange(141, dtype=DT) * 2 * math.pi / 141
        mod = 1.0 + 0.9 * torch.cos(2.0 * t)
        mb_mod = mb_ * mod
        mb_mod = mb_mod * (mb_.sum() / mb_mod.sum())     # total preserved
        pos = torch.cat([pa, pb])
        ang = torch.cat([aa, ab])
        nor = torch.stack([ang.cos(), ang.sin()], 1)
        m = torch.cat([ma_, mb_mod])
        lbl = torch.cat([torch.zeros(90, dtype=torch.long),
                         torch.ones(141, dtype=torch.long)])
        cert = evaluate_contact_reference(
            pos, nor, m, lbl == 0, lbl == 1, h_star, cov_r, SIG, thr,
            (0, 1), (0, 1))
        assert cert.failing == ["current"], cert.failing
        assert cert.metrics["anti"] <= thr["anti_max"]
        assert cert.metrics["mass_residual"] <= thr["mass_max"]
        assert not cert.certified

    def test_reference_consistency_guard(self):
        h_star, cov_r, thr, _ = self._reference()
        pos, nor, m, lbl = assemble([circle(0.3, 90),
                                     circle(0.3, 141, inward=True)])
        with pytest.raises(ContactCertificateError, match="inconsistent"):
            evaluate_contact_reference(
                pos, nor, m, lbl == 0, lbl == 1, h_star, 2.0 * cov_r,
                SIG, thr, (0, 1), (0, 1))
