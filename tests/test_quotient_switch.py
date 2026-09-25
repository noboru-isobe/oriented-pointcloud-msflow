"""0L-B2 pins: quotient bookkeeping, committed-level transaction,
persistence, and the certificate-gated shadow switch.

- Q_quot rows: two raw bulks merged into one class inside one grid
  component -> rows == {G} (bitwise the grid_only diagnostic rows)
- raw bulk straddling a group -> fail-closed
- export/import round trip; fixed-N and disjointness guards
- mode 'off' -> MMSolver step bitwise the pre-B2 path
- _advance_committed == the solve-loop committed state (1 step)
- QuotientSwitchError carries the two-arm audit trail into
  EvolutionHistory.stop_details (forced fail: gates = 0)
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"
                       / "experiments"))

from src.torch.oriented_varifold import OrientedPointCloudVarifold  # noqa
from src.torch.oriented_varifold.mass import compute_recommended_params  # noqa
from src.torch.shapes.generator import generate_oriented_two_circles  # noqa
from src.torch.solver.mm_solver import MMSolver, QuotientSwitchError  # noqa
from src.torch.solver.mm_step import MMStepper  # noqa
from src.torch.transport.grid_wasserstein import GridMetricConfig  # noqa
from src.torch.transport.phase_grid import PhaseGridConfig  # noqa

DT = torch.float64


def _cfg(delta, tau, n=256, eps=0.12, contact_rows="aligned_prequotient",
         quotient="off", redistribute=False):
    from p1_production_comparison import make_cfg
    cfg = make_cfg("C3", delta, tau, 1)
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    cfg.grid_metric = GridMetricConfig(phase=PhaseGridConfig(
        grid_shape=(n, n), fill_epsilon=eps))
    cfg.grid_bulk_rows_mode = "augment"
    cfg.grid_contact_rows_mode = contact_rows
    cfg.grid_quotient_mode = quotient
    cfg.redistribute = redistribute
    cfg.remove_dead_points = False
    cfg.optimizer_max_iter = 25
    if quotient != "off":
        cfg.grid_quotient_eps_split = 1e-6
        cfg.grid_quotient_eps_switch_x = 1.0
        cfg.grid_quotient_eps_switch_theta = 1.0
        cfg.grid_quotient_eps_volume = 1e-3
    return cfg


def _bridged():
    v = generate_oriented_two_circles(
        128, radius1=0.5, center1=(-0.56, 0.0),
        radius2=0.5, center2=(0.56, 0.0), device="cpu", dtype=DT)
    delta, tau = compute_recommended_params(v.positions)
    return v, delta, tau


class TestQuotientRows:
    def test_merged_bulks_give_grid_only_rows(self):
        v, delta, tau = _bridged()
        st = MMStepper(_cfg(delta, tau))
        st._setup_step(v)
        sn = st._last_grid_snapshot
        assert sn.partition_relation_grid_bulk == "b_finer"
        assert sn.r_stack == 2 and sn.quotient_classes == [0, 1]
        # merge both raw bulks (whole cloud) -> rows == G
        st_q = MMStepper(_cfg(delta, tau))
        st_q.commit_quotient(torch.ones(v.n_points, dtype=torch.bool),
                             0, {"pin": True}, (0, 1))
        st_q._setup_step(v)
        sn_q = st_q._last_grid_snapshot
        assert sn_q.quotient_classes == [0, 0] and sn_q.r_stack == 1
        st_g = MMStepper(_cfg(delta, tau, contact_rows="grid_only"))
        st_g._setup_step(v)
        # same row SPACE (the SVD-reduced basis vector may flip sign
        # because the stacked input differs by an all-zero relative row)
        rq, rg = st_q._rows_used_last, st_g._rows_used_last
        assert rq.shape == rg.shape == (1, 2 * v.n_points)
        assert float((rq - (rq @ rg.T) @ rg).abs().max()) < 1e-12
        # raw rows are kept as shadow
        assert st_q._bulk_rows_shadow is not None

    def test_straddling_bulk_fails_closed(self):
        v, delta, tau = _bridged()
        st = MMStepper(_cfg(delta, tau))
        mask = torch.zeros(v.n_points, dtype=torch.bool)
        mask[:64] = True                       # half of circle 1 only
        st.commit_quotient(mask, 0, {}, (0, 1))
        with pytest.raises(RuntimeError, match="straddles"):
            st._setup_step(v)


class TestPersistence:
    def test_export_import_roundtrip_and_guards(self):
        v, delta, tau = _bridged()
        st = MMStepper(_cfg(delta, tau))
        mask = torch.ones(v.n_points, dtype=torch.bool)
        st.commit_quotient(mask, 7, {"c": 1}, (0, 1))
        state = st.export_quotient_state()
        assert state["quotient_at"] == 7 and state["particle_count"] == 256
        st2 = MMStepper(_cfg(delta, tau))
        st2.import_quotient_state(state, v.n_points)
        assert torch.equal(st2._quotient_bulk_groups[0], mask)
        with pytest.raises(ValueError, match="fixed-N"):
            MMStepper(_cfg(delta, tau)).import_quotient_state(state, 255)
        bad = dict(state)
        bad["quotient_bulk_groups"] = [mask, mask]
        with pytest.raises(ValueError, match="overlapping"):
            MMStepper(_cfg(delta, tau)).import_quotient_state(bad, 256)
        # union-find closure on commit
        st3 = MMStepper(_cfg(delta, tau))
        m1 = torch.zeros(256, dtype=torch.bool); m1[:100] = True
        m2 = torch.zeros(256, dtype=torch.bool); m2[50:200] = True
        st3.commit_quotient(m1, 1, {}, (0, 1))
        st3.commit_quotient(m2, 2, {}, (1, 2))
        assert len(st3._quotient_bulk_groups) == 1
        assert int(st3._quotient_bulk_groups[0].sum()) == 200


class TestTransaction:
    def test_advance_committed_matches_solve_loop(self):
        v, delta, tau = _bridged()
        cfg = _cfg(delta, tau, redistribute=True)
        cfg.redistribute_interval = 1
        solver = MMSolver(cfg)
        hist = solver.solve(v, 1)
        committed_loop = hist.get_final_varifold()
        solver2 = MMSolver(cfg)
        res, committed = solver2._advance_committed(MMStepper(cfg), v)
        assert torch.equal(committed.positions, committed_loop.positions)
        assert torch.equal(committed.angles, committed_loop.angles)

    def test_mode_off_is_bitwise(self):
        v, delta, tau = _bridged()
        s1 = MMSolver(_cfg(delta, tau)).solve(v, 1).get_final_varifold()
        st = MMStepper(_cfg(delta, tau))
        r = st.step(v)
        assert torch.equal(s1.positions, r.varifold.positions)

    def test_guard_requires_calibrated_gates(self):
        v, delta, tau = _bridged()
        cfg = _cfg(delta, tau, quotient="certificate_shadow")
        cfg.grid_quotient_eps_split = None
        with pytest.raises(ValueError, match="not calibrated"):
            MMStepper(cfg)
        cfg2 = _cfg(delta, tau, quotient="certificate_shadow",
                    contact_rows="stack")
        with pytest.raises(ValueError, match="aligned_prequotient"):
            MMStepper(cfg2)

    def test_forced_fail_records_details(self, monkeypatch):
        """Force a certified pair on the bridged fixture (monkeypatched
        certificate) with zero gates: the drop arm must be rejected
        fail-closed and BOTH arms must land in history.stop_details."""
        v, delta, tau = _bridged()
        cfg = _cfg(delta, tau, quotient="certificate_shadow",
                   redistribute=True)
        cfg.redistribute_interval = 1
        cfg.grid_quotient_eps_split = 0.0
        cfg.grid_quotient_eps_switch_x = 0.0
        cfg.grid_quotient_eps_switch_theta = 0.0
        cfg.grid_quotient_eps_volume = 0.0
        from src.torch.transport import contact_certificate as cc

        def certified(pos, nor, m, labels, g_map, b_map, sigma, *a, **k):
            # fabricate a certified pair on the two loops (bulks 0, 1)
            loops = sorted(int(x) for x in labels.unique())
            cand = cc.ContactCandidate(
                loop_pair=(loops[0], loops[1]),
                bulk_pair=(b_map[loops[0]], b_map[loops[1]]),
                grid_component=g_map[loops[0]], ambiguous=False,
                reason="forced", g_sym=0.0)
            from dataclasses import asdict
            thr = asdict(cc.ContactThresholds())
            return [cc.ContactCertificate(
                candidate=cand,
                metrics={"h_ab": 0.1,
                         "coverage_radius":
                             thr["coverage_radius_over_h"] * 0.1},
                current={},
                margins={"uniqueness": 1.0}, failing=[], certified=True,
                thresholds=thr)]
        monkeypatch.setattr(cc, "evaluate_all_candidates", certified)
        solver = MMSolver(cfg)
        hist = solver.solve(v, 2)
        assert hist.stop_exception_type == "QuotientSwitchError"
        d = hist.stop_details
        assert d is not None and "keep" in d and "drop" in d and "gates" in d
        assert d["gates"]["drop_ran"] is True
        assert d["passed"] is False
        # nothing committed on failure
        assert solver._stepper._quotient_bulk_groups == []


def _mk_cert(h_ab=0.1):
    from dataclasses import asdict
    from src.torch.transport import contact_certificate as cc
    thr = asdict(cc.ContactThresholds())
    return dict(
        candidate=dict(loop_pair=(0, 1), bulk_pair=(0, 1),
                       grid_component=0, ambiguous=False,
                       reason="pin", g_sym=0.0),
        metrics=dict(h_ab=h_ab,
                     coverage_radius=thr["coverage_radius_over_h"] * h_ab),
        current={}, margins={}, failing=[], certified=True,
        thresholds=thr)


def _two_loop_labels(n=256):
    return torch.cat([torch.zeros(n // 2, dtype=torch.long),
                      torch.ones(n // 2, dtype=torch.long)])


class TestContactReferencePersistence:
    """0N-4 pins: the switch semantics (h_*, sigma_*, thresholds_*,
    particle partition) are quotient metadata -- checkpoint parity,
    legacy reconstruction, mask validation, exact-match guard."""

    def test_roundtrip_bitwise_and_legacy_reconstruction(self):
        v, delta, tau = _bridged()
        st = MMStepper(_cfg(delta, tau))
        st._setup_step(v)
        labels = _two_loop_labels(v.n_points)
        cert = _mk_cert()
        st.commit_quotient(torch.ones(v.n_points, dtype=torch.bool), 7,
                           cert, (0, 1), labels_src=labels)
        ref = st._quotient_meta["contact_reference"]
        # switch commit freezes exactly the certificate scale
        assert ref["h_ab_at_switch"] == cert["metrics"]["h_ab"]
        assert ref["thresholds_at_switch"] == cert["thresholds"]
        assert ref["sigma_at_switch"] == float(st._sigma)
        assert torch.equal(ref["particles_a_mask"], labels == 0)
        state = st.export_quotient_state()
        st2 = MMStepper(_cfg(delta, tau))
        st2.import_quotient_state(state, v.n_points)
        r2 = st2._quotient_meta["contact_reference"]
        assert r2["h_ab_at_switch"] == ref["h_ab_at_switch"]
        assert r2["sigma_at_switch"] == ref["sigma_at_switch"]
        assert r2["thresholds_at_switch"] == ref["thresholds_at_switch"]
        assert torch.equal(r2["particles_a_mask"], ref["particles_a_mask"])
        assert torch.equal(r2["particles_b_mask"], ref["particles_b_mask"])
        # legacy checkpoint (pre-0N-4): no contact_reference key ->
        # reconstructed EXACTLY from the stored switch certificate,
        # sigma_* deferred, masks lazily resolved
        legacy = {k: vv for k, vv in state.items()
                  if k != "contact_reference"}
        st3 = MMStepper(_cfg(delta, tau))
        st3.import_quotient_state(legacy, v.n_points)
        r3 = st3._quotient_meta["contact_reference"]
        assert r3["legacy_reconstructed"] is True
        assert r3["h_ab_at_switch"] == cert["metrics"]["h_ab"]
        assert r3["thresholds_at_switch"] == cert["thresholds"]
        assert r3["sigma_at_switch"] is None
        assert r3["particles_a_mask"] is None
        # a stub certificate (no metrics) leaves the reference None
        stub = dict(legacy)
        stub["certificate_at_switch"] = {"c": 1}
        st4 = MMStepper(_cfg(delta, tau))
        st4.import_quotient_state(stub, v.n_points)
        assert st4._quotient_meta["contact_reference"] is None

    def test_mask_validation_on_import(self):
        v, delta, tau = _bridged()
        st = MMStepper(_cfg(delta, tau))
        st._setup_step(v)
        labels = _two_loop_labels(v.n_points)
        st.commit_quotient(torch.ones(v.n_points, dtype=torch.bool), 7,
                          _mk_cert(), (0, 1), labels_src=labels)
        state = st.export_quotient_state()
        for mutate, msg in (
                (lambda r: r.update(particles_a_mask=r["particles_a_mask"]
                                    .long()), "bool mask"),
                (lambda r: r.update(particles_a_mask=torch.ones(
                    7, dtype=torch.bool)), "length"),
                (lambda r: r.update(particles_a_mask=torch.zeros(
                    v.n_points, dtype=torch.bool)), "empty"),
                (lambda r: r.update(particles_a_mask=r["particles_b_mask"]
                                    .clone()), "overlap"),
        ):
            bad = dict(state)
            bad["contact_reference"] = dict(state["contact_reference"])
            mutate(bad["contact_reference"])
            with pytest.raises(ValueError, match=msg):
                MMStepper(_cfg(delta, tau)).import_quotient_state(
                    bad, v.n_points)

    def test_exact_match_guard(self, monkeypatch):
        """Reviewer eq. (3.1): the CURRENT loop of each stored mask
        must equal the stored particle set exactly; a single foreign
        particle joining the loop's label is fail-closed (otherwise
        the reference would grade a proper subset)."""
        from src.torch.transport.contact_certificate import (
            ContactCertificateError,
        )
        v, delta, tau = _bridged()
        st = MMStepper(_cfg(delta, tau))
        st._setup_step(v)
        labels = _two_loop_labels(v.n_points)
        st.commit_quotient(torch.ones(v.n_points, dtype=torch.bool), 3,
                           _mk_cert(), (0, 1), labels_src=labels)
        m = torch.full((v.n_points,), 0.01, dtype=DT)
        monkeypatch.setattr(
            MMStepper, "_resolve_state_masses_labels",
            lambda self, p, n: (m, labels))
        cert = st.contact_reference_on_state(v)
        assert cert is not None            # guard passes, evaluation runs
        assert cert.metrics["h_ab"] == 0.1  # frozen scale, not live
        # inflow: one particle of loop 1 re-labeled into loop 0
        bad = labels.clone()
        bad[200] = 0
        monkeypatch.setattr(
            MMStepper, "_resolve_state_masses_labels",
            lambda self, p, n: (m, bad))
        with pytest.raises(ContactCertificateError,
                           match="inflow/dropout"):
            st.contact_reference_on_state(v)
        # split: the stored mask spans two current labels
        split = labels.clone()
        split[10:20] = 5
        monkeypatch.setattr(
            MMStepper, "_resolve_state_masses_labels",
            lambda self, p, n: (m, split))
        with pytest.raises(ContactCertificateError, match="spans"):
            st.contact_reference_on_state(v)
