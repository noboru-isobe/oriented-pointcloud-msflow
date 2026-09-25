"""0C-5d-1.2 pins: causal attribution of the fast pair attraction.

Measured (N = 128, dt = 1e-4, paper configuration gap 0.1):

    A (baseline)              g_dot_early = -122.5
    B (q == 1)                -124.2   -> coherence contribution ~ 0
    C (block-diagonal BIE)    -131.1   -> BIE contribution small
    D (componentwise carrier)   -1.5   -> KDE carrier coupling IS the channel
    A at N = 256                -1.5   -> delta(256) ~ 0.09 < gap: decoupled

So the fast attraction is the finite-delta CARRIER-OVERLAP attraction
(two sheets within delta share density -> masses drop -> estimated
perimeter drops: the dynamic realisation of F8), NOT the coherent
perimeter -- the earlier q_min collapse was a consequence, not a cause.
It vanishes at fixed positive gap under delta -> 0 (a finite-bandwidth
contact-layer regularisation, outside the presently proved theory), and
the measured rates are causal-contrast data at fixed dt, not
time-converged contact velocities.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "experiments"))

from two_ellipses_causal_ablation_0c5d12 import run_mode  # noqa: E402

STEPS = 3


def test_attraction_is_kde_carrier_not_coherence():
    a = run_mode("A", 128, 1e-4, n_steps=STEPS)
    b = run_mode("B", 128, 1e-4, n_steps=STEPS)
    d = run_mode("D", 128, 1e-4, n_steps=STEPS)
    # fast in A, equally fast with unit coherence, gone with
    # componentwise carrier
    assert a["g_dot_early"] < -50.0
    assert abs(b["g_dot_early"] - a["g_dot_early"]) < 0.3 * abs(
        a["g_dot_early"])
    assert abs(d["g_dot_early"]) < 5.0
    assert d["q_min_final"] > 0.9        # no geometry-driven q collapse
    assert all(f["objective_decreased"] for f in a["frames"])


def test_attraction_decouples_at_fine_resolution():
    """delta(N=256) ~ 0.09 < gap 0.1: the kernels no longer reach across,
    and the pair follows the genuine independent-relaxation rate."""
    r = run_mode("A", 256, 1e-4, n_steps=STEPS)
    assert abs(r["g_dot_early"]) < 5.0   # measured -1.5
    assert r["q_min_final"] > 0.9
