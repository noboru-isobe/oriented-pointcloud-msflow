"""L-A1 pins (reviewer 2026-08-22 hardened activation spec).

Probe three-way classification (hardening 1/2): benign statuses never
raise, fatal statuses are returned for the caller to fail-close on,
|I| >= 3 maturity boundary. Driver units: fingerprint mismatch
detection (3.1), barycenter-consistency identity (6.1), neutrality
decomposition identities (5.1-5.3). Transaction level (real M2
WB+rows mature state, production H=768 config): PASS record
completeness, artificial fingerprint mismatch -> pure fail, forced
keep-arm complex loss -> transient (not fail), shadow-replay parity
(8), volume-target seeding is a bitwise no-op when the seeded value
equals the derived one (default-path safety of the
_pending_target_volume hook).
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"
                       / "experiments"))

import two_ellipses_benchmark as teb  # noqa: E402
from src.torch.oriented_varifold import (  # noqa: E402
    OrientedPointCloudVarifold,
)
from src.torch.oriented_varifold.mass import (  # noqa: E402
    compute_recommended_params,
)
from src.torch.perimeter.coherence_perimeter import (  # noqa: E402
    compute_coherence_loopwise,
)
from src.torch.perimeter.contact_complex import (  # noqa: E402
    contact_complex_q,
    probe_contact_complex,
)
from src.torch.transport.bem_wasserstein import (  # noqa: E402
    compute_coherence,
)

DT = torch.float64
SIG = 0.1
STATES = Path("results/two_ellipses/two_ellipses_m2_WBrows_states.pt")
SRC_STEP = 1700          # M2 WB+rows checkpoint, sides (39, 39)


def circle(R, n, center=(0.0, 0.0), inward=False, phase=0.0):
    t = torch.arange(n, dtype=DT) * 2 * math.pi / n + phase
    pos = torch.stack([center[0] + R * t.cos(),
                       center[1] + R * t.sin()], 1)
    ang = t + (math.pi if inward else 0.0)
    m = torch.full((n,), 2 * math.pi * R / n, dtype=DT)
    return pos, ang, m


def assemble(parts):
    pos = torch.cat([p for p, _, _ in parts])
    ang = torch.cat([a for _, a, _ in parts])
    m = torch.cat([mm for _, _, mm in parts])
    lbl = torch.cat([torch.full((p.shape[0],), i, dtype=torch.long)
                     for i, (p, _, _) in enumerate(parts)])
    nor = torch.stack([ang.cos(), ang.sin()], 1)
    return pos, nor, m, lbl


def side_by_side(gap, R=0.35, n=90, y=0.0):
    return [circle(R, n, center=(-(R + gap / 2), y)),
            circle(R, n, center=(R + gap / 2, y))]


def qq(pos, nor, m, lbl):
    q_full = compute_coherence(teb._VV(pos, nor), m, SIG,
                               "wendland_c2")
    q_self, _ = compute_coherence_loopwise(pos, nor, m, SIG,
                                           "wendland_c2", lbl)
    return q_full, q_self


def probe(pos, nor, m, lbl, **kw):
    q_full, q_self = qq(pos, nor, m, lbl)
    return probe_contact_complex(pos, nor, m, lbl, SIG, "wendland_c2",
                                 q_full, q_self, **kw)


class TestProbeClassification:
    def test_inactive(self):
        pos, nor, m, lbl = assemble(side_by_side(0.5))
        pr = probe(pos, nor, m, lbl)
        assert pr["status"] == "inactive"
        assert pr["res"] is not None      # legal evaluation, 0 cx

    def test_mature_unique_and_maturity_boundary(self):
        pos, nor, m, lbl = assemble(side_by_side(0.03))
        pr = probe(pos, nor, m, lbl)
        assert pr["status"] == "mature_unique"
        sides = pr["res"]["cx"]["complexes"][0].sides
        ns = [s.n for s in sides]
        assert len(ns) == 2 and min(ns) >= 3
        # |I| boundary: threshold at exactly min(n) stays mature,
        # min(n) + 1 demotes to maturing (continue-WB, not fatal)
        assert probe(pos, nor, m, lbl,
                     maturity_min_side=min(ns))["status"] \
            == "mature_unique"
        pr2 = probe(pos, nor, m, lbl, maturity_min_side=min(ns) + 1)
        assert pr2["status"] == "maturing"
        assert "side sizes" in pr2["reason"]

    def test_satellite_is_maturing_not_a_raise(self):
        # the L0J onset-band single-pair satellite (fail-closed for
        # the ENERGY) is a benign continue-WB status for the probe
        pos, nor, m, lbl = assemble(side_by_side(0.0929))
        pr = probe(pos, nor, m, lbl)
        assert pr["status"] == "maturing"
        assert "< 2" in pr["reason"]

    def test_three_loop_chain_is_run_fatal(self):
        g, R = 0.03, 0.06
        d = 2 * R + g
        h = d * math.sqrt(3) / 2
        pos, nor, m, lbl = assemble([
            circle(R, 40, center=(-d / 2, 0.0)),
            circle(R, 40, center=(d / 2, 0.0)),
            circle(R, 40, center=(0.0, h))])
        pr = probe(pos, nor, m, lbl)
        assert pr["status"] in ("ambiguous", "geometry_error")
        assert pr["reason"]

    def test_self_contact_is_geometry_error(self):
        t = torch.linspace(0.15 * math.pi, 1.85 * math.pi, 120,
                           dtype=DT)
        outer = torch.stack([torch.cos(t), torch.sin(t)], 1)
        inner = torch.stack([0.93 * torch.cos(t.flip(0)),
                             0.93 * torch.sin(t.flip(0))], 1)
        loop = torch.cat([outer, inner])
        ang = torch.atan2(loop[:, 1], loop[:, 0])
        nor = torch.stack([ang.cos(), ang.sin()], 1)
        m = torch.full((loop.shape[0],), 0.02, dtype=DT)
        lbl = torch.zeros(loop.shape[0], dtype=torch.long)
        q_f, q_s = qq(loop, nor, m, lbl)
        pr = probe_contact_complex(loop, nor, m, lbl, SIG,
                                   "wendland_c2", q_f, q_s)
        assert pr["status"] == "geometry_error"

    def test_co_oriented_is_out_of_scope(self):
        pos, nor, m, lbl = assemble([circle(0.35, 90),
                                     circle(0.37, 141)])
        pr = probe(pos, nor, m, lbl)
        assert pr["status"] == "out_of_scope"

    def test_two_mature_complexes_are_ambiguous(self):
        pos, nor, m, lbl = assemble(
            side_by_side(0.03) + side_by_side(0.03, y=3.0))
        # relabel: four distinct loops (assemble already does)
        pr = probe(pos, nor, m, lbl)
        assert pr["status"] == "ambiguous"
        assert "mature complexes" in pr["reason"]
        # with a high maturity bar the SAME state is merely maturing
        pr2 = probe(pos, nor, m, lbl, maturity_min_side=1000)
        assert pr2["status"] == "maturing"


class TestDriverUnits:
    def test_fingerprints_match_detects_each_mismatch(self):
        base = dict(t=torch.arange(4, dtype=DT), x=1.25,
                    sv=[3.0, 1.0, 0.5], none_field=None)
        same = dict(t=base["t"].clone(), x=1.25,
                    sv=[3.0, 1.0, 0.5], none_field=None)
        assert teb.fingerprints_match(base, same) == []
        for key, bad_val in (("t", torch.arange(4, dtype=DT) + 1e-15),
                             ("x", 1.25 + 1e-12),
                             ("sv", [3.0, 1.0, 0.5 + 1e-14])):
            mut = dict(same)
            mut[key] = bad_val
            assert teb.fingerprints_match(base, mut) == [key]

    def test_barycenter_consistency_identity(self):
        # (6.1) is exact algebra: xbar1 - xbar0 == (dM - xbar0 dA)/A1
        g = torch.Generator().manual_seed(7)
        pos, _, _ = circle(0.35, 50)
        pert = pos + 1e-3 * torch.randn(pos.shape, generator=g,
                                        dtype=DT)
        A0 = teb.signed_area_order(pos)
        A1 = teb.signed_area_order(pert)
        M0 = teb.polygon_moments_order(pos)
        M1 = teb.polygon_moments_order(pert)
        b0 = teb.centroid_order(pos)
        b1 = teb.centroid_order(pert)
        for k in (0, 1):
            pred = ((M1[k] - M0[k]) - b0[k] * (A1 - A0)) / A1
            assert abs(pred - (b1[k] - b0[k])) < 1e-12

    def test_neutrality_decomposition_identities(self):
        # (5.1) per-loop WB, (5.2) per-side CC, (5.3) far exact
        pos, nor, m, lbl = assemble(side_by_side(0.03))
        q_full, q_self = qq(pos, nor, m, lbl)
        res = contact_complex_q(pos, nor, m, lbl, SIG, "wendland_c2",
                                q_full, q_self)
        q_cc = res["q_cc"]
        q_wb = torch.empty_like(q_full)
        for li in (0, 1):
            sel = lbl == li
            r_l = float((m[sel] * q_full[sel]).sum()
                        / (m[sel] * q_self[sel]).sum())
            q_wb[sel] = r_l * q_self[sel]
            assert abs(float((m[sel] * (q_wb - q_full)[sel]).sum())) \
                < 1e-13
        far = torch.ones_like(lbl, dtype=torch.bool)
        for comp in res["cx"]["complexes"]:
            for side in comp.sides:
                sel = side.mask
                assert abs(float((m[sel]
                                  * (q_cc - q_full)[sel]).sum())) \
                    < 1e-13
                far &= ~side.mask
        assert float((q_cc - q_full)[far].abs().max()) == 0.0
        assert abs(float((m * (q_cc - q_wb)).sum())) < 1e-13


# ---------------------------------------------------------------
# Transaction level: real mature state, production H=768 config.
# ---------------------------------------------------------------

needs_states = pytest.mark.skipif(
    not STATES.exists(), reason="M2 states file not present")


@pytest.fixture(scope="module")
def env():
    torch.set_default_dtype(DT)
    v0, m1, n_el, _ = teb.build_cloud(0.0)
    delta, tau = map(float, compute_recommended_params(v0.positions))

    def make_cfg(q_mode):
        c = teb.build_config(teb.production_args(
            grid=768, redist_monotone=True, quotient_mode="off",
            q_mode=q_mode), delta, tau)
        c.time_step = 1e-5
        c.grid_bulk_first_moment_rows = True
        return c

    # the run's frozen volume target, derived from the t=0 cloud
    # exactly as the production controller does
    from src.torch.solver.mm_step import MMStepper
    st = MMStepper(make_cfg("self_renormalized"))
    st._setup_step(OrientedPointCloudVarifold(
        positions=v0.positions.clone(), angles=v0.angles.clone()))
    target0 = float(st._grid_target_volume_initial)
    del st
    ck = torch.load(STATES, weights_only=True)
    src = ck[SRC_STEP]
    return dict(make_cfg=make_cfg, delta=delta, tau=tau,
                target0=target0, pos=src["positions"],
                ang=src["angles"])


@pytest.fixture(scope="module")
def passed(env):
    """The PASS transaction, run once and shared."""
    return teb.run_activation_transaction(
        env["pos"], env["ang"], env["make_cfg"], env["delta"],
        env["tau"], env["target0"], SRC_STEP)


@needs_states
class TestTransaction:
    def test_pass_record_complete(self, passed):
        status, rec, com = passed
        assert status == "pass", (rec.get("reason"),
                                  rec.get("exception"))
        assert com is not None
        assert rec["passed"] and all(rec["gates"].values())
        assert set(rec["gates"]) == {
            "fingerprint_parity", "neutrality", "conservation",
            "bar_consistency", "split_residual",
            "post_both_mature_same_pair", "objective_decrease",
            "iterations_under_cap"}
        assert rec["fingerprint_mismatch"] == []
        assert rec["neutrality"]["far_cc_max"] == 0.0
        assert [x[0] for x in rec["source_sides"]] == [0, 1]
        assert all(x[1] >= 3 for x in rec["source_sides"])
        assert rec["post_probe"] == {"WB": "mature_unique",
                                     "CC": "mature_unique"}
        for arm in ("WB", "CC"):
            d = rec["deltas"][arm]
            for li in (0, 1):
                assert abs(d[f"dA{li}"]) < 1e-4
                assert d[f"bar_consistency{li}"] < 1e-10
        assert rec["eps_A"] == teb.EPS_A_FLOOR
        assert rec["dX_arms_over_h"] < 0.5
        # L-A amendment audit fields (reviewer verdict)
        assert rec["activation_gap_threshold"] == 0.55
        assert "L0J-2b" in rec["activation_threshold_provenance"]
        assert rec["activation_gap_over_sigma"] \
            <= teb.ACT_GAP_OVER_SIGMA

    def test_shadow_replay_parity(self, env, passed):
        # hardening 8: regenerating X_CC^{n+1} from the recorded
        # source reproduces the committed shadow arm bitwise
        status, _, com = passed
        assert status == "pass"
        from src.torch.solver.mm_solver import MMSolver
        from src.torch.solver.mm_step import MMStepper
        cfg = env["make_cfg"]("contact_complex_renormalized")
        solver = MMSolver(cfg)
        st = MMStepper(cfg)
        st._grid_target_volume_initial = env["target0"]
        v = OrientedPointCloudVarifold(
            positions=env["pos"].clone(), angles=env["ang"].clone())
        _, replay = solver._advance_committed(st, v)
        assert torch.equal(replay.positions, com.positions)
        assert torch.equal(replay.angles, com.angles)

    def test_fingerprint_mismatch_fails_pure(self, env):
        # arms with different mass tau: fail, no committed state
        def bad_factory(q_mode):
            c = env["make_cfg"](q_mode)
            if q_mode == "contact_complex_renormalized":
                c.mass_tau = env["tau"] * 1.001
            return c
        status, rec, com = teb.run_activation_transaction(
            env["pos"], env["ang"], bad_factory, env["delta"],
            env["tau"], env["target0"], SRC_STEP)
        assert status == "fail"
        assert com is None
        assert "fingerprint_parity" in rec["reason"]
        assert rec["fingerprint_mismatch"]

    def test_keep_arm_complex_loss_is_transient(self, env,
                                                monkeypatch):
        # hardening 4: WB post-step probe not mature -> transient
        # (continue WB), never an activation and never a fail
        import src.torch.perimeter.contact_complex as ccm
        real = ccm.probe_contact_complex
        calls = {"n": 0}

        def wrapper(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 2:        # post-step probe, WB arm
                return dict(status="maturing", reason="forced by pin",
                            res=None)
            return real(*a, **kw)

        monkeypatch.setattr(ccm, "probe_contact_complex", wrapper)
        status, rec, com = teb.run_activation_transaction(
            env["pos"], env["ang"], env["make_cfg"], env["delta"],
            env["tau"], env["target0"], SRC_STEP)
        assert status == "transient"
        assert com is None
        assert rec.get("transient") is True
        assert rec["post_probe"]["WB"] == "maturing"

    def test_exception_path_fails_pure(self, env):
        def broken_factory(q_mode):
            raise RuntimeError("forced setup failure")
        status, rec, com = teb.run_activation_transaction(
            env["pos"], env["ang"], broken_factory, env["delta"],
            env["tau"], env["target0"], SRC_STEP)
        assert status == "fail"
        assert com is None
        assert "forced setup failure" in rec["exception"]


L0_STATES = Path("results/two_ellipses/two_ellipses_l0_768_states.pt")


class TestEligibilityMargin:
    """L-A amendment pins (reviewer 2026-08-22 verdict): eligibility
    eq (1) = mature_unique AND both |I| >= 3 AND g_min/sigma_X <=
    0.55, with sigma_X the source-frozen scale echoed by the complex
    evaluation and g_min the certified-pair cross-loop gap."""

    def _ratio(self, pos, ang, lbl):
        nor = torch.stack([ang.cos(), ang.sin()], 1)
        v0, _, _, _ = teb.build_cloud(0.0)
        delta, tau = map(float,
                         compute_recommended_params(v0.positions))
        m = teb.resolve_m(pos, nor, delta, tau)
        qf = compute_coherence(teb._VV(pos, nor), m, SIG,
                               "wendland_c2")
        qs, _ = compute_coherence_loopwise(pos, nor, m, SIG,
                                           "wendland_c2", lbl)
        pr = probe_contact_complex(pos, nor, m, lbl, SIG,
                                   "wendland_c2", qf, qs)
        assert pr["status"] == "mature_unique"
        assert pr["res"]["sigma"] == SIG   # source-frozen scale echo
        return teb.activation_gap_ratio(pos, lbl, pr["res"])

    @pytest.mark.skipif(not L0_STATES.exists(),
                        reason="L0 states file not present")
    def test_onset_band_mature_state_not_eligible(self):
        # first mature_unique of the L0 768 trajectory: g/sigma =
        # 0.912, sides 13/13 -- mature but NOT eligible (pin 1)
        ck = torch.load(L0_STATES, weights_only=True)
        st = ck[525]
        n = st["positions"].shape[0] // 2
        lbl = torch.zeros(st["positions"].shape[0], dtype=torch.long)
        lbl[n:] = 1
        ratio = self._ratio(st["positions"], st["angles"], lbl)
        assert abs(ratio - 0.912) < 5e-3
        assert ratio > teb.ACT_GAP_OVER_SIGMA        # not eligible

    @needs_states
    def test_inner_regime_mature_state_eligible(self):
        # M2 WBrows @1700 (g = 0.0403): mature AND inside the margin
        # -> eligible (pin 2)
        ck = torch.load(STATES, weights_only=True)
        st = ck[1700]
        n = st["positions"].shape[0] // 2
        lbl = torch.zeros(st["positions"].shape[0], dtype=torch.long)
        lbl[n:] = 1
        ratio = self._ratio(st["positions"], st["angles"], lbl)
        assert ratio <= teb.ACT_GAP_OVER_SIGMA
        assert ratio > 0.3          # sanity: a real gap, not zero

    def test_margin_does_not_soften_fail_closed_energy(self):
        # pin 3: the 0.55 sigma margin lives ONLY in eligibility --
        # the CC energy evaluation (and hence the post-activation
        # standing certificate) still fail-closes on a satellite
        from src.torch.perimeter.contact_complex import (
            ContactComplexNotMature,
        )
        pos, nor, m, lbl = assemble(side_by_side(0.0929))
        q_f, q_s = qq(pos, nor, m, lbl)
        with pytest.raises(ContactComplexNotMature):
            contact_complex_q(pos, nor, m, lbl, SIG, "wendland_c2",
                              q_f, q_s)

    def test_threshold_constants(self):
        assert teb.ACT_GAP_OVER_SIGMA == 0.55
        assert "L0J-2b" in teb.ACT_GAP_PROVENANCE


class TestTargetSeeding:
    def test_pending_target_seed_is_noop_when_equal(self):
        # the MMSolver._pending_target_volume hook: seeding the value
        # the stepper would derive anyway is bitwise inert, and the
        # unseeded path is the pre-L-A default
        torch.set_default_dtype(DT)
        from src.torch.solver.mm_solver import MMSolver
        pos, nor, m, lbl = assemble(side_by_side(0.5, n=120))
        ang = torch.atan2(nor[:, 1], nor[:, 0])
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        delta, tau = map(float, compute_recommended_params(pos))
        cfg = teb.build_config(teb.production_args(
            grid=512, redist_monotone=True, quotient_mode="off"),
            delta, tau)
        cfg.time_step = 1e-5

        def one(seed):
            sv = MMSolver(cfg)
            if seed is not None:
                sv._pending_target_volume = seed
            hist = sv.solve(OrientedPointCloudVarifold(
                positions=pos.clone(), angles=ang.clone()), 1)
            return hist._varifolds[-1]

        base = one(None)
        # derive the target the default path used
        from src.torch.solver.mm_step import MMStepper
        st = MMStepper(cfg)
        st._setup_step(OrientedPointCloudVarifold(
            positions=pos.clone(), angles=ang.clone()))
        seeded = one(float(st._grid_target_volume_initial))
        assert torch.equal(base.positions, seeded.positions)
        assert torch.equal(base.angles, seeded.angles)
