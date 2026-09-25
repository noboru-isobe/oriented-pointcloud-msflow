"""Grid metric G1: current-to-phase filling (frozen rho0).

The reviewer's mandatory pins: sign/volume on the circle, annulus C=1
vs two-disk C=2 from the reconstructed support, permutation
invariance, hidden opposite sheets leaving the phase unchanged, box
doubling, and support-threshold stability.
"""

import math

import pytest
import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import compute_masses
from src.torch.shapes.generator import (
    generate_oriented_annulus,
    generate_oriented_circle,
    generate_oriented_two_ellipses,
)
from src.torch.transport.phase_grid import (
    CurrentToPhase,
    PhaseGridConfig,
)

DT = torch.float64


def _cfg(n=256, eps=0.08, **kw):
    return PhaseGridConfig(grid_shape=(n, n), fill_epsilon=eps, **kw)


def _masses(v):
    from src.torch.oriented_varifold.mass import (
        compute_recommended_params,
    )
    delta, tau = compute_recommended_params(v.positions)
    return compute_masses(v.positions, delta, tau, "wendland_c2")


def test_circle_sign_volume_and_l1():
    n_pts = 256
    v = generate_oriented_circle(n_pts, 1.0, (0.0, 0.0), "cpu", DT)
    m = _masses(v)
    pg = CurrentToPhase(_cfg()).reconstruct(v, m, math.pi)
    H = pg.rho_metric.shape[0]
    c = H // 2
    edge = int(0.25 * H)
    assert float(pg.rho_metric[c, c]) > 0.9          # inside ~ 1
    assert float(pg.rho_metric[5, 5]) < 0.05         # outside ~ 0
    assert pg.volume_error < 1e-6                    # projection fixes V
    assert pg.n_components == 1
    # L1 error vs sharp indicator is O(eps * perimeter)
    xs = torch.linspace(-2, 2, H + 1, dtype=DT)[:-1] + pg.dx / 2
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    chi = ((X * X + Y * Y).sqrt() <= 1.0).to(DT)
    l1 = float((pg.rho_metric - chi).abs().sum()) * pg.cell_area
    assert l1 < 4.0 * 0.08 * 2 * math.pi             # ~ eps * perim


def test_component_counts_annulus_vs_pair():
    n_out = 192
    v = generate_oriented_annulus(n_out, 1.2, 0.5, (0.0, 0.0),
                                  "cpu", DT)
    m = _masses(v)
    vol = math.pi * (1.2 ** 2 - 0.5 ** 2)
    pg = CurrentToPhase(_cfg()).reconstruct(v, m, vol)
    assert pg.n_components == 1                      # annulus: C=1
    # QUANTITATIVE filling check (G1.1): C=1 alone cannot distinguish
    # a correct annulus from a disk with the hole accidentally filled
    H = pg.rho_metric.shape[0]
    c = H // 2
    ring_ix = int((0.85 + 2.0) / pg.dx)
    assert float(pg.rho_metric[c, c]) < 0.02         # hole empty
    assert float(pg.rho_metric[c, ring_ix]) > 0.98   # bulk full
    xs = torch.linspace(-2, 2, H + 1, dtype=DT)[:-1] + pg.dx / 2
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    r = (X * X + Y * Y).sqrt()
    chi = ((r <= 1.2) & (r >= 0.5)).to(DT)
    l1 = float((pg.rho_metric - chi).abs().sum()) * pg.cell_area
    assert l1 < 4.0 * 0.08 * (2 * math.pi * (1.2 + 0.5))

    # pair needs eps << gap (the scale discipline): eps=0.05 < g/2
    # scale discipline, MEASURED: the scattered wall is porous unless
    # h/eps <~ 0.5 (gap-floor rho: 0.10 at h/eps~1, 0.005 at 0.6,
    # 0.0 at 0.45) -- n_per=256 gives h/eps=0.45 with eps=0.04 < g/2
    v2 = generate_oriented_two_ellipses(
        256, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
        a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu", dtype=DT)
    m2 = _masses(v2)
    vol2 = 2 * math.pi * 0.4 * 1.0
    pg2 = CurrentToPhase(_cfg(n=512, eps=0.04)).reconstruct(
        v2, m2, vol2)
    assert pg2.n_components == 2                     # pair: C=2


def test_permutation_invariance():
    v = generate_oriented_circle(128, 1.0, (0.0, 0.0), "cpu", DT)
    m = _masses(v)
    ctp = CurrentToPhase(_cfg(n=128))
    a = ctp.reconstruct(v, m, math.pi)
    perm = torch.randperm(v.n_points)
    v2 = OrientedPointCloudVarifold(positions=v.positions[perm],
                                    angles=v.angles[perm])
    b = ctp.reconstruct(v2, m[perm], math.pi)
    assert float((a.rho_metric - b.rho_metric).abs().max()) < 1e-12


def test_hidden_opposite_sheets_leave_phase_unchanged():
    """Adding a cancelling double sheet (same points, opposite
    orientations) must not change the reconstructed phase -- the
    filling map quotients hidden boundaries away."""
    v = generate_oriented_circle(128, 1.0, (0.0, 0.0), "cpu", DT)
    m = _masses(v)
    ctp = CurrentToPhase(_cfg(n=128))
    base = ctp.reconstruct(v, m, math.pi)

    ring = generate_oriented_circle(64, 0.5, (0.0, 0.0), "cpu", DT)
    ring_m = torch.full((64,), 2 * math.pi * 0.5 / 64, dtype=DT)
    pos = torch.cat([v.positions, ring.positions, ring.positions])
    ang = torch.cat([v.angles, ring.angles,
                     ring.angles + math.pi])       # opposite sheet
    vv = OrientedPointCloudVarifold(positions=pos, angles=ang)
    mm = torch.cat([m, ring_m, ring_m])
    both = ctp.reconstruct(vv, mm, math.pi)
    assert float((both.rho_metric - base.rho_metric).abs().max()) < 1e-10


def test_box_doubling_periodic_image_error():
    v = generate_oriented_circle(192, 1.0, (0.0, 0.0), "cpu", DT)
    m = _masses(v)
    small = CurrentToPhase(_cfg(n=256)).reconstruct(v, m, math.pi)
    big_cfg = PhaseGridConfig(grid_shape=(512, 512),
                              box_min=(-4.0, -4.0), box_max=(4.0, 4.0),
                              fill_epsilon=0.08)
    big = CurrentToPhase(big_cfg).reconstruct(v, m, math.pi)
    # compare on the common inner window [-2, 2)^2 (same dx)
    inner = big.rho_metric[128:384, 128:384]
    err = float((inner - small.rho_metric).abs().max())
    assert err < 5e-3      # periodic-image effect on rho is small
    # record: the METRIC-level image effect is audited in G2a


def test_support_threshold_stability():
    v2 = generate_oriented_two_ellipses(
        256, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
        a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu", dtype=DT)
    m2 = _masses(v2)
    vol2 = 2 * math.pi * 0.4 * 1.0
    counts, areas = [], []
    for thr in (1e-3, 3e-3, 1e-2):
        pg = CurrentToPhase(_cfg(n=512, eps=0.04,
                                 support_threshold=thr)).reconstruct(
            v2, m2, vol2)
        counts.append(pg.n_components)
        areas.append(float((pg.rho_metric > thr).sum())
                     * pg.cell_area)
    # stable ABOVE the reconstruction noise floor
    assert counts == [2, 2, 2]
    assert (max(areas) - min(areas)) / max(areas) < 0.15
    # the noise floor itself is a pinned measurement: at h/eps = 0.45
    # the porous-wall bridge sits between 1e-4 and 1e-3, so a
    # threshold below the floor reads the pair as ONE component --
    # exactly the instability the confidence sweep is built to detect
    pg_low = CurrentToPhase(_cfg(n=512, eps=0.04,
                                 support_threshold=1e-4)).reconstruct(
        v2, m2, vol2)
    # noise-floor pin: 1e-4 sits BELOW the reconstruction noise floor,
    # so the component count is WRONG there -- in which direction
    # depends on the background level (pre-G3c it merged everything
    # into 1; with the carrier renormalization the sub-noise tails
    # fragment instead). Either failure mode certifies the floor;
    # only reading the true C=2 would falsify the pin.
    assert pg_low.n_components != 2


def test_projection_gate_telemetry_and_validation():
    from src.torch.transport.phase_grid import (
        InvalidPhaseReconstructionError,
        validate_phase_reconstruction,
    )
    v = generate_oriented_circle(192, 1.0, (0.0, 0.0), "cpu", DT)
    m = _masses(v)
    cfg = _cfg()
    pg = CurrentToPhase(cfg).reconstruct(v, m, math.pi)
    assert pg.projection_l2_relative < cfg.projection_rel_tol
    assert pg.volume_error < 1e-8       # exact after final projection
    validate_phase_reconstruction(pg, cfg)   # healthy phase passes

    # a dropped speck must FAIL the scientific gate
    import dataclasses
    bad = dataclasses.replace(pg, n_speck_components_dropped=1,
                              speck_area_dropped=1e-4,
                              speck_mass_dropped=1e-4)
    with pytest.raises(InvalidPhaseReconstructionError):
        validate_phase_reconstruction(bad, cfg)


def test_carrier_renormalization_identity_and_fail_closed():
    """G3c: the filling renormalizes the carrier so the current's
    divergence-theorem volume equals the target EXACTLY. Pins:
    (i) factor recorded and == target/V_hat; (ii) a globally rescaled
    carrier with the same target reproduces the same phase (the drift
    channel is dead); (iii) target == own V_hat gives factor exactly
    1.0; (iv) >20% mismatch is refused."""
    from src.torch.transport.phase_grid import (
        InvalidPhaseReconstructionError,
    )
    v = generate_oriented_circle(256, 1.0, (0.0, 0.0), "cpu", DT)
    m = _masses(v)
    vhat = float(0.5 * (m * (v.positions * v.normals).sum(-1)).sum())
    ctp = CurrentToPhase(_cfg())

    pg = ctp.reconstruct(v, m, math.pi)
    assert pg.carrier_renormalization == pytest.approx(
        math.pi / vhat, rel=1e-14)

    # (ii) global carrier rescale is exactly compensated
    pg2 = ctp.reconstruct(v, m * 1.003, math.pi)
    assert float((pg2.rho_metric - pg.rho_metric).abs().max()) < 1e-12

    # (iii) own-V_hat target -> factor exactly 1
    pg3 = ctp.reconstruct(v, m, vhat)
    assert pg3.carrier_renormalization == 1.0

    # (iv) fail closed on a >20% mismatch
    with pytest.raises(InvalidPhaseReconstructionError,
                       match="renormalization"):
        ctp.reconstruct(v, m, 1.5 * math.pi)


def test_grid_config_consistency_assert():
    from src.torch.transport.phase_grid import (
        assert_grid_configs_consistent,
    )
    from src.torch.transport.weighted_poisson import (
        WeightedPoissonConfig,
    )
    p = _cfg(n=256)
    assert_grid_configs_consistent(
        p, WeightedPoissonConfig(grid_shape=(256, 256)))
    with pytest.raises(ValueError):
        assert_grid_configs_consistent(
            p, WeightedPoissonConfig(grid_shape=(512, 512)))
    with pytest.raises(ValueError):
        assert_grid_configs_consistent(
            p, WeightedPoissonConfig(grid_shape=(256, 256),
                                     support_threshold=1e-2))


def test_speck_gate_ambiguous_window_scoped_tolerance():
    """Otto pair v4 (2026-08-07): inside the threshold-UNSTABLE window
    (the same label-free window that activates the conservative-sweep
    deflation) single-cell crumbs of the healing interface flicker
    across the support threshold (measured: one cell, mass 1.8e-7 =
    7e-8 of the phase volume, at step 785 just after the merger).
    Pins: (i) in the ambiguous window, sub-resolution drops within the
    hard caps pass -- always recorded; (ii) outside the window the
    strict fail-closed gate is UNCHANGED; (iii) a drop beyond the caps
    raises even inside the window."""
    import dataclasses

    from src.torch.transport.phase_grid import (
        InvalidPhaseReconstructionError,
        validate_phase_reconstruction,
    )
    v = generate_oriented_circle(192, 1.0, (0.0, 0.0), "cpu", DT)
    m = _masses(v)
    cfg = _cfg()
    pg = CurrentToPhase(cfg).reconstruct(v, m, math.pi)

    # v9: the window-scoped budget governs SUPPORTED specks (matter);
    # the synthetic speck here is marked as supported to preserve this
    # test's original meaning
    tiny = dataclasses.replace(
        pg, n_speck_components_dropped=1,
        speck_area_dropped=float(pg.cell_area),
        speck_mass_dropped=1e-7,
        n_speck_supported_dropped=1,
        speck_supported_area_dropped=float(pg.cell_area),
        speck_supported_mass_dropped=1e-7)
    # (ii) outside the window: unchanged strict gate
    with pytest.raises(InvalidPhaseReconstructionError):
        validate_phase_reconstruction(tiny, cfg)
    # (i) inside the window: tolerated (recorded in the fields)
    validate_phase_reconstruction(tiny, cfg, ambiguous_window=True)
    # (iii) caps still bite inside the window
    big = dataclasses.replace(
        pg, n_speck_components_dropped=1,
        speck_area_dropped=float(pg.cell_area) * 100.0,
        speck_mass_dropped=1e-7,
        n_speck_supported_dropped=1,
        speck_supported_area_dropped=float(pg.cell_area) * 100.0,
        speck_supported_mass_dropped=1e-7)
    with pytest.raises(InvalidPhaseReconstructionError):
        validate_phase_reconstruction(big, cfg, ambiguous_window=True)


def test_projection_gate_ambiguous_window_cap():
    """v7 pin: inside the threshold-unstable window the projection
    gate uses projection_rel_tol_window (healing-zone cancellation
    disturbance, measured 1.33e-2 vs clean max 1.2e-3); outside it the
    strict tolerance is unchanged; the window cap still bites."""
    import dataclasses

    from src.torch.transport.phase_grid import (
        InvalidPhaseReconstructionError,
        validate_phase_reconstruction,
    )
    v = generate_oriented_circle(192, 1.0, (0.0, 0.0), "cpu", DT)
    m = _masses(v)
    cfg = _cfg()
    pg = CurrentToPhase(cfg).reconstruct(v, m, math.pi)
    mid = dataclasses.replace(pg, projection_l2_relative=2e-2)
    with pytest.raises(InvalidPhaseReconstructionError):
        validate_phase_reconstruction(mid, cfg)               # strict
    validate_phase_reconstruction(mid, cfg, ambiguous_window=True)
    big = dataclasses.replace(pg, projection_l2_relative=8e-2)
    with pytest.raises(InvalidPhaseReconstructionError):
        validate_phase_reconstruction(big, cfg,
                                      ambiguous_window=True)  # cap bites


class TestSpeckSupportSplit:
    """v9: budgets protect matter; matter is where the marks are."""

    def _pg_with_specks(self, n_sup, sup_mass, total, total_mass):
        import dataclasses
        import math as _m
        from src.torch.shapes.generator import generate_oriented_circle
        from src.torch.oriented_varifold.mass import (
            compute_masses, compute_recommended_params)
        from src.torch.transport.phase_grid import (
            CurrentToPhase, PhaseGridConfig)
        v = generate_oriented_circle(256, 1.0, (0.0, 0.0), "cpu",
                                     torch.float64)
        delta, tau = compute_recommended_params(v.positions)
        m = compute_masses(v.positions, delta, tau, "wendland_c2")
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
        pg = CurrentToPhase(cfg).reconstruct(v, m, _m.pi)
        return dataclasses.replace(
            pg, n_speck_components_dropped=total,
            speck_area_dropped=total * pg.cell_area,
            speck_mass_dropped=total_mass,
            n_speck_supported_dropped=n_sup,
            speck_supported_area_dropped=n_sup * pg.cell_area,
            speck_supported_mass_dropped=sup_mass), cfg

    def test_markless_ringing_within_mass_budget_passes(self):
        from src.torch.transport.phase_grid import (
            validate_phase_reconstruction)
        # 19 markless single cells at 1.8e-7 each (the measured
        # step-826 burst) -- window CLOSED, still tolerated
        pg, cfg = self._pg_with_specks(0, 0.0, 19, 19 * 1.8e-7)
        validate_phase_reconstruction(pg, cfg, ambiguous_window=False)

    def test_markless_ringing_over_mass_budget_fails(self):
        from src.torch.transport.phase_grid import (
            InvalidPhaseReconstructionError,
            validate_phase_reconstruction)
        pg, cfg = self._pg_with_specks(0, 0.0, 3, 1e-3)
        with pytest.raises(InvalidPhaseReconstructionError,
                           match="markless ringing"):
            validate_phase_reconstruction(pg, cfg,
                                          ambiguous_window=False)

    def test_supported_specks_keep_strict_budget(self):
        from src.torch.transport.phase_grid import (
            InvalidPhaseReconstructionError,
            validate_phase_reconstruction)
        # a single supported speck outside the window: strict fail
        pg, cfg = self._pg_with_specks(1, 1.8e-7, 1, 1.8e-7)
        with pytest.raises(InvalidPhaseReconstructionError,
                           match="supported"):
            validate_phase_reconstruction(pg, cfg,
                                          ambiguous_window=False)
        # inside the window and under caps: tolerated
        validate_phase_reconstruction(pg, cfg, ambiguous_window=True)


class TestFinalFragmentSupportSplit:
    """v9b: the fragmentation gate's subject is matter -- a markless
    sub-min-area FINAL component (threshold ringing surviving the
    metric labels) is counted, not gated."""

    def _pg(self):
        import math as _m
        from src.torch.shapes.generator import generate_oriented_circle
        from src.torch.oriented_varifold.mass import (
            compute_masses, compute_recommended_params)
        from src.torch.transport.phase_grid import (
            CurrentToPhase, PhaseGridConfig)
        v = generate_oriented_circle(256, 1.0, (0.0, 0.0), "cpu",
                                     torch.float64)
        delta, tau = compute_recommended_params(v.positions)
        m = compute_masses(v.positions, delta, tau, "wendland_c2")
        cfg = PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08)
        return CurrentToPhase(cfg).reconstruct(v, m, _m.pi), cfg

    def test_markless_final_debris_not_gated(self):
        import dataclasses
        from src.torch.transport.phase_grid import (
            validate_phase_reconstruction)
        pg, cfg = self._pg()
        # a 1-cell markless final component: unfiltered min is tiny,
        # supported min stays the disk -- passes
        mod = dataclasses.replace(
            pg, min_final_component_area=float(pg.cell_area),
            n_final_markless_debris=1)
        validate_phase_reconstruction(mod, cfg, ambiguous_window=False)

    def test_supported_final_fragment_still_gated(self):
        import dataclasses
        from src.torch.transport.phase_grid import (
            InvalidPhaseReconstructionError,
            validate_phase_reconstruction)
        pg, cfg = self._pg()
        mod = dataclasses.replace(
            pg, min_final_component_area=float(pg.cell_area),
            min_final_supported_component_area=float(pg.cell_area))
        with pytest.raises(InvalidPhaseReconstructionError,
                           match="post_projection_fragmentation"):
            validate_phase_reconstruction(mod, cfg,
                                          ambiguous_window=False)

    def test_reconstruct_fills_supported_min(self):
        pg, cfg = self._pg()
        assert pg.min_final_supported_component_area is not None
        assert pg.n_final_markless_debris == 0
        assert (pg.min_final_supported_component_area
                == pg.min_final_component_area)
