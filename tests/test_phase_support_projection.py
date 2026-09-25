"""Phase-support projection (review option B): label-free interior
fossil removal, decided by the reconstructed phase alone.

Mandatory fixtures (review): every boundary mark of a circle, an
annulus (BOTH boundaries -- the inner one has the hole on one side)
and a pre-contact pair is RETAINED; interior opposite sheets inside a
merged disk are removed while the outer boundary survives, with the
shadow deltas at discretization level; permutation equivariance; and
no perimeter jump on acceptance.
"""

import math

import pytest
import torch

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import (
    compute_masses,
    compute_recommended_params,
)
from src.torch.shapes.generator import (
    generate_oriented_annulus,
    generate_oriented_circle,
    generate_oriented_two_ellipses,
)
from src.torch.transport.phase_grid import PhaseGridConfig
from src.torch.transport.phase_support_projection import (
    PhaseSupportProjectionConfig,
    phase_support_projection,
)

DT = torch.float64


def _cfg(n=256, eps=0.08):
    return PhaseGridConfig(grid_shape=(n, n), fill_epsilon=eps)


def _masses(v):
    delta, tau = compute_recommended_params(v.positions)
    return compute_masses(v.positions, delta, tau, "wendland_c2"), delta


def _r_local(delta, eps, sigma=0.1):
    return max(eps, sigma, delta)


def test_circle_and_annulus_marks_all_retained():
    v = generate_oriented_circle(256, 1.0, (0.0, 0.0), "cpu", DT)
    m, delta = _masses(v)
    out = phase_support_projection(
        v, m, None, _cfg(), math.pi, _r_local(delta, 0.08))
    assert out.n_candidates == 0

    va = generate_oriented_annulus(192, 1.2, 0.5, (0.0, 0.0), "cpu", DT)
    ma, da = _masses(va)
    vol = math.pi * (1.2 ** 2 - 0.5 ** 2)
    out = phase_support_projection(
        va, ma, None, _cfg(), vol, _r_local(da, 0.08))
    # the annulus INNER boundary has the hole on one side -- retained
    assert out.n_candidates == 0


def test_precontact_pair_facing_arcs_retained():
    v = generate_oriented_two_ellipses(
        256, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
        a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu", dtype=DT)
    m, delta = _masses(v)
    vol = 2 * math.pi * 0.4 * 1.0
    out = phase_support_projection(
        v, m, None, PhaseGridConfig(grid_shape=(512, 512),
                                    fill_epsilon=0.04),
        vol, _r_local(delta, 0.04))
    # two components -> not stably single-component -> ineligible
    assert out.n_candidates == 0
    assert "single-component" in out.reason


def _merged_disk_with_interior_sheets(n_ring=256, n_sheet=48):
    """A disk boundary plus an exactly-cancelling opposite sheet pair
    deep inside -- the idealized fossil."""
    disk = generate_oriented_circle(n_ring, 1.0, (0.0, 0.0), "cpu", DT)
    ys = torch.linspace(-0.25, 0.25, n_sheet, dtype=DT)
    seg = torch.stack([torch.full_like(ys, -0.1), ys], dim=1)
    up = torch.zeros(n_sheet, dtype=DT)            # normal +x
    pos = torch.cat([disk.positions, seg, seg])
    ang = torch.cat([disk.angles, up, up + math.pi])
    return OrientedPointCloudVarifold(positions=pos, angles=ang), n_ring


def test_interior_sheets_removed_boundary_kept():
    v, n_ring = _merged_disk_with_interior_sheets()
    m, delta = _masses(v)
    out = phase_support_projection(
        v, m, None, _cfg(), math.pi, _r_local(delta, 0.08))
    assert out.n_candidates > 0
    assert out.accepted, out.reason
    keep = out.keep_mask
    # every ring mark kept, every sheet mark removed
    assert bool(keep[:n_ring].all())
    assert not bool(keep[n_ring:].any())
    # phase and current preserved at discretization level
    assert out.D_rho < 2e-3
    assert out.D_J < 2e-2
    assert out.vol_drift < 1e-4


def test_permutation_equivariance():
    v, n_ring = _merged_disk_with_interior_sheets()
    m, delta = _masses(v)
    out = phase_support_projection(
        v, m, None, _cfg(), math.pi, _r_local(delta, 0.08))
    perm = torch.randperm(v.n_points)
    v2 = OrientedPointCloudVarifold(positions=v.positions[perm],
                                    angles=v.angles[perm])
    out2 = phase_support_projection(
        v2, m[perm], None, _cfg(), math.pi, _r_local(delta, 0.08))
    assert out2.n_candidates == out.n_candidates
    assert bool((out2.keep_mask == out.keep_mask[perm]).all())


def test_no_perimeter_jump_on_acceptance():
    from src.torch.perimeter.coherence_perimeter import (
        compute_perimeter_coherence,
    )
    from src.torch.transport import compute_coherence

    v, _ = _merged_disk_with_interior_sheets()
    m, delta = _masses(v)
    sigma = 0.1

    def perim(vv, mm):
        q = compute_coherence(vv, mm, sigma, "wendland_c2",
                              backend="naive")
        return compute_perimeter_coherence(vv, mm, sigma,
                                           "wendland_c2",
                                           backend="naive")

    out = phase_support_projection(
        v, m, None, _cfg(), math.pi, _r_local(delta, 0.08),
        perimeter=lambda vv, mm: perim(vv, mm))
    assert out.accepted, out.reason
    assert out.D_P < 1e-3


def test_solver_integration_psp_commits_and_records():
    """Review blockers 1/2/4 integration pin: PSP driven through
    MMSolver on a merged-disk-with-interior-sheets state must (i)
    reduce the COMMITTED N, (ii) expose committed_varifold and the
    outcome payload (with shadow-recomputed D_V present) on the step
    result, (iii) leave result.varifold (the MM minimizer) at the
    pre-projection size -- the two must differ exactly by the
    removals."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent
                           / "scripts" / "experiments"))
    from p1_production_comparison import make_cfg

    from src.torch.solver.mm_solver import MMSolver
    from src.torch.transport.grid_wasserstein import GridMetricConfig

    # integration fixture uses an OFFSET pair (separation 0.02 <
    # eps), matching the real seam fossil: exactly-coincident pairs
    # have DEGENERATE coherence (0/0), the MM step then moves them by
    # O(0.5) and scatters the fixture before the projection can act --
    # measured; the module-level tests keep the coincident ideal.
    disk = generate_oriented_circle(256, 1.0, (0.0, 0.0), "cpu", DT)
    ys = torch.linspace(-0.25, 0.25, 48, dtype=DT)
    segA = torch.stack([torch.full_like(ys, -0.11), ys], dim=1)
    segB = torch.stack([torch.full_like(ys, -0.09), ys], dim=1)
    pos = torch.cat([disk.positions, segA, segB])
    ang = torch.cat([disk.angles,
                     torch.zeros(48, dtype=DT),           # +x
                     torch.full((48,), math.pi, dtype=DT)])  # -x
    v = OrientedPointCloudVarifold(positions=pos, angles=ang)
    n_ring = 256
    delta, tau = compute_recommended_params(v.positions)
    cfg = make_cfg("C3", delta, tau, 1)
    cfg.time_step = 1e-5
    cfg.bem_solver_mode = "legacy_global_projection"
    cfg.metric_backend = "grid_poisson"
    cfg.grid_metric = GridMetricConfig(
        phase=PhaseGridConfig(grid_shape=(256, 256), fill_epsilon=0.08),
        compatibility_components="conservative_sweep")
    cfg.redistribute = False
    cfg.remove_dead_points = False
    cfg.enforce_support_gates = False
    cfg.grid_phase_support_projection = True

    solver = MMSolver(cfg)
    seen = {}

    def cb(step, result):
        seen["result"] = result
        return True                       # single step suffices

    solver.solve(v, 1, callback=cb)
    r = seen["result"]
    assert r.committed_varifold is not None
    out = r.phase_support_projection_outcome
    assert out is not None and out["accepted"], out
    assert out["n_before"] == v.n_points
    assert r.committed_varifold.n_points == out["n_after"] \
        < v.n_points
    # the minimizer state is NOT the committed state
    assert r.varifold.n_points == out["n_before"]
    # shadow diagnostics present and sane (mass_fn path: D_V computed
    # from independently re-massed states)
    assert out["D_V"] < 1e-3
    assert out["D_P"] < 1e-3              # perimeter check non-vacuous
    assert len(out["removed_q"]) == out["n_before"] - out["n_after"]
