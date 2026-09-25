"""Comp-4 pins (reviewer 2026-08-19): conservative ghost compression.

Module-level (fast): analytic root + eps_alg + 0<c<=1, signed
combination V# = A_o + A_d - A_h, polygon-centroid scaling center,
transfer-map residual, homothety certificates (simple / orientation /
cyclic order / centroid), deletion-only rejection, mask completeness,
P#-gate arithmetic, driver-gate parity of the area functional.

Transaction/artifact level (skipped when the captured entry states or
the Comp-4b outputs are absent): frozen-target inheritance, atomic
transaction PASS on the real 768 entry state, failure purity,
event-index / shadow-recomputation-parity / driver-state-reset /
checkpoint-parity read off the Comp-4b run artifacts.
"""

import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"
                       / "experiments"))

from src.torch.solver.ghost_compression import (  # noqa: E402
    EPS_ALG,
    GhostCompressionError,
    compression_certificates,
    conservative_compression,
    current_distance,
    merged_geom_volume,
    phase_field_distance,
    polygon_centroid,
    polygon_is_simple,
    signed_shoelace,
    solve_transfer_scale,
    star_shoelace_area,
    transfer_map_residual,
)

DT = torch.float64
OUT = Path("results/exact_merger")


def circle(R, n, center=(0.0, 0.0), inward=False, phase=0.0):
    t = torch.arange(n, dtype=DT) * 2 * math.pi / n + phase
    pos = torch.stack([center[0] + R * t.cos(),
                       center[1] + R * t.sin()], 1)
    ang = t + (math.pi if inward else 0.0)
    return pos, ang


def triple(R_out=1.0, R_d=0.2, R_h=0.24, n_out=64, n_wall=48,
           center=(0.03, -0.02)):
    """outer (outward) + ghost disk (outward) + ghost hole (inward),
    off-center so centroid bookkeeping is exercised."""
    po, ao = circle(R_out, n_out)
    pd, ad = circle(R_d, n_wall, center=center)
    ph, ah = circle(R_h, n_wall, center=center, inward=True)
    pos = torch.cat([po, pd, ph])
    ang = torch.cat([ao, ad, ah])
    nor = torch.stack([ang.cos(), ang.sin()], 1)
    n = pos.shape[0]
    mo = torch.zeros(n, dtype=torch.bool); mo[:n_out] = True
    ma = torch.zeros(n, dtype=torch.bool); ma[n_out:n_out + n_wall] = True
    mb = torch.zeros(n, dtype=torch.bool); mb[n_out + n_wall:] = True
    return pos, ang, nor, mo, ma, mb


class TestMapAndRoot:
    def test_signed_combination_and_analytic_root(self):
        pos, ang, nor, mo, ma, mb = triple()
        A_o = star_shoelace_area(pos[mo])
        A_d = star_shoelace_area(pos[ma])
        A_h = star_shoelace_area(pos[mb])
        V = merged_geom_volume(pos, nor, [mo, ma, mb])
        assert abs(V - (A_o + A_d - A_h)) < 1e-14          # signed comb
        assert V < A_o                                     # V_pair < 0
        root = solve_transfer_scale(V, A_o)
        assert abs(root["c"] - math.sqrt(V / A_o)) < 1e-15
        assert abs(root["residual"]) <= EPS_ALG * V
        assert 0.0 < root["c"] <= 1.0
        comp = conservative_compression(pos, ang, nor, mo, ma, mb)
        assert abs(comp["V_comp_geom"] - V) <= 10 * EPS_ALG * V
        assert comp["angles"].shape[0] == int(mo.sum())
        # angles are untouched by the homothety
        assert torch.equal(comp["angles"], ang[mo])

    def test_generic_bisection_branch_and_bracket_failure(self):
        pos, ang, nor, mo, ma, mb = triple()
        A_o = star_shoelace_area(pos[mo])
        V = merged_geom_volume(pos, nor, [mo, ma, mb])
        root = solve_transfer_scale(V, A_o,
                                    F=lambda c: c * c * A_o - V)
        assert abs(root["c"] - math.sqrt(V / A_o)) < 1e-11
        assert root["branch"] == "bisection"
        with pytest.raises(GhostCompressionError, match="bracket"):
            solve_transfer_scale(V, A_o, F=lambda c: 1.0)

    def test_scale_outside_unit_interval_fails(self):
        # V_pair > 0 (hole smaller than disk) -> c > 1 -> structure
        # certificate rejects
        pos, ang, nor, mo, ma, mb = triple(R_d=0.3, R_h=0.2)
        with pytest.raises(GhostCompressionError, match="outside"):
            conservative_compression(pos, ang, nor, mo, ma, mb)

    def test_polygon_centroid_is_scaling_center(self):
        pos, ang, nor, mo, ma, mb = triple(center=(0.1, 0.05))
        z = polygon_centroid(pos[mo])
        comp = conservative_compression(pos, ang, nor, mo, ma, mb)
        zc = polygon_centroid(comp["positions"])
        assert float((zc - z).norm()) < 1e-12
        assert comp["certificates"]["centroid_preserved"]

    def test_gate_functional_matches_driver(self):
        from exact_merger_benchmark import shoelace_area
        pos, _, _, mo, _, _ = triple()
        assert star_shoelace_area(pos[mo]) == shoelace_area(pos[mo])


class TestResidualAndCertificates:
    def test_transfer_map_residual(self):
        pos, ang, nor, mo, ma, mb = triple()
        comp = conservative_compression(pos, ang, nor, mo, ma, mb)
        assert transfer_map_residual(comp["positions"],
                                     comp["positions"], 0.01) == 0.0
        pert = comp["positions"].clone()
        pert[0, 0] += 1e-6
        assert transfer_map_residual(pert, comp["positions"], 0.01) \
            > EPS_ALG * 1e3

    def test_certificates_negative_cases(self):
        pos, _ = circle(1.0, 32)
        assert polygon_is_simple(pos)
        # figure-eight ordering -> self-intersecting
        bad = pos.clone()
        bad[1], bad[17] = pos[17].clone(), pos[1].clone()
        assert not polygon_is_simple(bad)
        z = polygon_centroid(pos)
        certs = compression_certificates(pos, -pos.flip(0), z)
        assert not certs["all"]                    # orientation flipped

    def test_outer_not_outward_fails(self):
        pos, ang, nor, mo, ma, mb = triple()
        nor2 = nor.clone()
        nor2[mo] = -nor2[mo]
        with pytest.raises(GhostCompressionError, match="outward"):
            conservative_compression(pos, ang, nor2, mo, ma, mb)

    def test_mask_completeness_fail_closed(self):
        pos, ang, nor, mo, ma, mb = triple()
        mo2 = mo.clone()
        mo2[0] = False                            # uncovered particle
        with pytest.raises(GhostCompressionError, match="cover"):
            conservative_compression(pos, ang, nor, mo2, ma, mb)
        with pytest.raises(GhostCompressionError, match="overlap"):
            conservative_compression(pos, ang, nor, mo, ma, mb | ma)

    def test_deletion_only_rejected_by_volume(self):
        """Deletion-only (c = 1) leaves the outer area A_o, which
        misses the target V# by |V_pair| >> eps_alg -- the event
        volume gate rejects it (the +0.74% negative control)."""
        pos, ang, nor, mo, ma, mb = triple()
        A_o = star_shoelace_area(pos[mo])
        V = merged_geom_volume(pos, nor, [mo, ma, mb])
        assert abs(A_o - V) > 1e-3 * V            # far beyond eps_alg
        comp = conservative_compression(pos, ang, nor, mo, ma, mb)
        # and the unscaled candidate fails the transfer-map residual
        assert transfer_map_residual(pos[mo], comp["positions"],
                                     0.01) > EPS_ALG * 1e3

    def test_perimeter_non_increase_arithmetic(self):
        # a positive homothety with c <= 1 cannot increase the polygon
        # perimeter of the outer loop
        pos, ang, nor, mo, ma, mb = triple()
        comp = conservative_compression(pos, ang, nor, mo, ma, mb)
        def perim(p):
            return float((torch.roll(p, -1, 0) - p).norm(dim=1).sum())
        assert perim(comp["positions"]) <= perim(pos[mo]) + 1e-12


class TestDistances:
    def test_current_and_phase_distances(self):
        pos, _ = circle(1.0, 64)
        ang = torch.arange(64, dtype=DT) * 2 * math.pi / 64
        nor = torch.stack([ang.cos(), ang.sin()], 1)
        m = torch.full((64,), 2 * math.pi / 64, dtype=DT)
        d = current_distance(pos, nor, m, pos, nor, m, 0.1)
        assert abs(d) < 1e-12
        d2 = current_distance(pos, nor, m, pos * 0.99, nor, m, 0.1)
        assert d2 > 0.0
        r = torch.rand(16, 16, dtype=DT) + 0.5
        out = phase_field_distance(r, r * 1.01)
        assert abs(out["D_rho_1"] - 0.01) < 1e-12
        assert abs(out["D_rho_2"] - 0.01) < 1e-12


ENTRY = OUT / "exact_merger_0n4c_768_states.pt"


@pytest.mark.skipif(not ENTRY.exists(),
                    reason="captured 768 entry state not present")
class TestTransactionOnEntryState:
    def _setup(self):
        from exact_merger_benchmark import build_cloud, build_config
        from quotient_switch_stability import production_args
        from src.torch.oriented_varifold.mass import (
            compute_recommended_params,
        )
        torch.set_default_dtype(DT)
        v0, _ = build_cloud()
        delta, tau = map(float, compute_recommended_params(v0.positions))
        from exact_merger_benchmark import masses_for
        m0 = masses_for("loopwise_oriented_kde", v0.positions,
                        v0.normals, delta, tau)
        target0 = float(0.5 * (m0 * (v0.positions * v0.normals)
                               .sum(-1)).sum())
        cfg = build_config(production_args(grid=768,
                                           redist_monotone=True),
                           delta, tau)
        ck = torch.load(ENTRY, weights_only=True)
        st = ck[2294]
        return cfg, delta, tau, target0, st

    def test_frozen_target_inheritance(self):
        from exact_merger_benchmark import new_stepper_with_frozen_state
        cfg, delta, tau, target0, st = self._setup()
        stp = new_stepper_with_frozen_state(cfg, target0)
        assert stp._grid_target_volume_initial == target0

    def test_atomic_transaction_pass_and_failure_purity(self):
        from exact_merger_benchmark import run_compression_transaction
        cfg, delta, tau, target0, st = self._setup()
        pos, ang, quot = st["positions"], st["angles"], st["quotient"]
        pos0, ang0 = pos.clone(), ang.clone()
        ok, rec, vC, cC = run_compression_transaction(
            pos, ang, quot, cfg, delta, tau, target0, 2294)
        assert ok, rec
        assert rec["passed"] and all(rec["gates"].values())
        assert rec["n_before"] == 487 and rec["n_after"] == 256
        assert 0.0 < rec["scale"] <= 1.0
        assert abs(rec["V_comp_geom"] - rec["V_sharp_geom"]) \
            <= 10 * EPS_ALG * rec["V_sharp_geom"]
        assert rec["dP_sharp_event"] <= 1e-9      # energy non-increase
        assert rec["transfer_map_residual"] <= EPS_ALG * 1e3
        # failure purity: a broken quotient (no reference) is rejected
        # and the inputs are untouched
        quot_bad = dict(quot)
        quot_bad["contact_reference"] = None
        ok2, rec2, vC2, _ = run_compression_transaction(
            pos, ang, quot_bad, cfg, delta, tau, target0, 2294)
        assert not ok2 and vC2 is None
        assert torch.equal(pos, pos0) and torch.equal(ang, ang0)


COMP4B = OUT / "exact_merger_comp4b_768.json"


@pytest.mark.skipif(not COMP4B.exists(),
                    reason="Comp-4b 768 artifacts not present")
class TestComp4bArtifacts:
    def _load(self):
        return json.loads(COMP4B.read_text())

    def test_event_index_and_series_integrity(self):
        d = self._load()
        n_evt = d["meta"]["comp_committed_at"]
        assert n_evt == d["meta"]["comp_entry_at"]
        steps = [r["step"] for r in d["series"]]
        assert steps == sorted(set(steps))            # no dup / gap mix
        assert all(b - a == 1 for a, b in zip(steps, steps[1:]))
        row = [r for r in d["series"] if r["step"] == n_evt][0]
        assert row.get("compression_event", {}).get("passed") is True
        nxt = [r for r in d["series"] if r["step"] == n_evt + 1]
        assert len(nxt) == 1                          # exactly once

    def test_shadow_recomputation_parity(self):
        d = self._load()
        assert d["meta"]["shadow_parity_max_abs"] == 0.0

    def test_driver_state_reset(self):
        d = self._load()
        n_evt = d["meta"]["comp_committed_at"]
        post = [r for r in d["series"] if r["step"] > n_evt]
        assert post and d["meta"]["error"] is None
