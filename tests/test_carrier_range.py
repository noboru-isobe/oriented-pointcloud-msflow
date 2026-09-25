"""0C-5d-1.2a pins: the carrier-overlap range mechanism, closed.

Measured (N = 128, paper configuration, explicit-delta sweep at fixed N):

    delta/g0 <= 1.0:  omega_cross = 0.0, m_global == m_cw EXACTLY
                      (Wendland support is [0, delta]), g_dot ~ -1
    delta/g0 = 1.25:  g_dot_early = -85   (NON-monotone: -183 at 1.5,
    delta/g0 = 1.5:   g_dot_early = -183   -97 at 2.0 -- threshold
                                           closed, force law not)

so delta = gap is the exact switch-on threshold of the finite-delta
carrier-overlap attraction/bias. The 2x2 call-site split identifies the
pathway: componentwise PERIMETER carrier alone kills the attraction
(-122.5 -> -1.6) while componentwise metric carrier alone leaves most of
it (-83.4) -- the frozen-perimeter energy gradient is the dominant
channel, with a subdominant metric-side contribution. The unit-q
trajectory's DIAGNOSTIC real q (current-mass, properly time-consistent)
falls 0.987 -> 0.064 as the gap closes: consequence, not cause. The
hard stop separates last-valid frames from REJECTED candidates: Q's
first violating candidate appears at gap +0.034 (and at every tested dt,
at gaps that do not vanish with dt: 0.034/0.017/0.015 for
dt = 1e-4/5e-5/2e-5, with checkpoint velocity dt-STABLE at ~80-95) while
M runs the cap under control. Q stays demoted to exploratory; a
continuous-time rejection remains unclaimed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from carrier_range_0c5d12a import (  # noqa: E402
    G0,
    carrier_mixing,
    hard_stop_run,
    pair_cloud,
)


def test_delta_equals_gap_is_the_exact_threshold():
    from src.torch.oriented_varifold.mass import compute_recommended_params

    v, n1 = pair_cloud()
    _, tau = compute_recommended_params(v.positions)
    for ratio in (0.9, 1.0):                    # delta = g0 itself included
        omega, mratio, diff, sat = carrier_mixing(v.positions, n1,
                                                  ratio * G0, tau)
        assert omega == 0.0
        assert diff == 0.0                      # machine equality, exact
        assert mratio == 1.0
        assert sat > 1.0                        # cutoff saturated
    omega_out, ratio_out, _, _ = carrier_mixing(v.positions, n1,
                                                1.5 * G0, tau)
    assert omega_out > 1e-2                     # measured 3.7e-2
    assert ratio_out < 0.999                    # measured 0.9977


def test_perimeter_energy_is_the_dominant_pathway():
    """2x2 call-site split (measured, vs baseline -1.5): the global
    perimeter carrier is necessary and dominant (-81.9 alone); the
    setup side is an amplification term (-39.1 on top, -0.1 without the
    global perimeter carrier), not an independent channel."""
    from carrier_range_0c5d12a import two_by_two_split

    split = two_by_two_split()
    assert split["perimeter_gl_metric_gl"] < -100.0    # measured -122.5
    assert split["perimeter_cw_metric_gl"] > -5.0      # measured -1.6
    assert split["perimeter_gl_metric_cw"] < -50.0     # measured -83.4
    assert split["perimeter_cw_metric_cw"] > -5.0      # measured -1.5


def test_projected_perimeter_gradient_pin():
    """Optimizer-free energy-gradient attribution (review; observable
    corrected after measurement): plain norms of the projected gradients
    differ only 2x, because BOTH carriers carry the O(1) compatible
    ROUNDING gradient. The attraction-specific object is the CROSS-CARRIER
    BIAS FORCE  b = grad P_gl(0) - grad P_cw(0), same Q, q == 1, BEM
    untouched. Measured: |b| = 1.41 with 100.00% of its squared norm on
    the facing arcs (dist to the other component < delta -- compact-kernel
    exactness again), uniformly NEGATIVE there (energy descends as the
    facing arcs advance = attraction), and 94% compatible
    (|Q^T b| = 1.32). This is '-grad P is the pathway' pinned without
    optimizer, dt, or coherence feedback."""
    import torch
    from src.torch.solver.mm_step import MMStepper
    from src.torch.oriented_varifold.mass import compute_masses
    from carrier_range_0c5d12a import pair_cloud
    from two_ellipses_paper_precontact import cfg_for

    v, n1 = pair_cloud()
    cfg = cfg_for(1e-4, "oracle", 2)
    cfg.use_unit_coherence = True
    stepper = MMStepper(cfg)
    stepper._setup_step(v)
    Q = stepper.param.Q
    delta, tau = stepper.mass_delta, stepper.mass_tau

    def grad(componentwise):
        s = torch.zeros(v.n_points, dtype=torch.float64,
                        requires_grad=True)
        pos = (stepper.param.prev_positions
               + s[:, None] * stepper.param.prev_normals)
        if componentwise:
            m = torch.cat([compute_masses(pos[:n1], delta, tau),
                           compute_masses(pos[n1:], delta, tau)])
        else:
            m = compute_masses(pos, delta, tau)
        (g,) = torch.autograd.grad(m.sum(), s)
        return g

    bias = grad(False) - grad(True)
    d = torch.cdist(v.positions[:n1], v.positions[n1:])
    facing = torch.zeros(v.n_points, dtype=torch.bool)
    facing[:n1] = d.min(1).values < delta
    facing[n1:] = d.min(0).values < delta

    assert bias.norm() > 1.0                               # measured 1.41
    frac = ((bias[facing] ** 2).sum() / (bias ** 2).sum()).item()
    assert frac > 0.999                                    # measured 1.0000
    assert (bias[facing] <= 0).all()                       # pure attraction
    assert (Q.T @ bias).norm() > 0.9 * bias.norm()         # compatible


def test_hard_stop_discriminates_m_and_q():
    q = hard_stop_run("Q", n_steps=10)
    m = hard_stop_run("M", n_steps=10)
    # Q's violating step is a REJECTED CANDIDATE, kept separate from the
    # last valid frame (production must roll back, not commit it)
    assert q["rejected_candidate"] is not None
    assert "max|s|/gap" in q["rejected_candidate"]["reason"]
    # bounded away from zero -- the claim is "Q hits the stop at a
    # FINITE gap", not the exact value. Measured 0.048 under the
    # pre-fix angle map (C_2D missing in the B-side kernel gradient:
    # normals under-rotated by pi/7); 0.0342 with the corrected map.
    assert q["last_valid_gap"] > 0.03
    assert m["rejected_candidate"] is None      # M stays controlled
    assert all(f["s_over_gap"] <= 0.25 for f in m["frames"])
