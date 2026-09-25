"""Closure patch 2 (reviewer T): BEMSetupSnapshot telemetry.

The snapshot is OBSERVATIONAL: enabling it must leave the trajectory
bitwise identical, it must never construct solver state implicitly,
and it must record the ACTUALLY-consumed weights/policies separately
from the canonical pointwise quantities.
"""

import sys
from pathlib import Path

import torch

from src.torch.oriented_varifold.mass import compute_recommended_params
from src.torch.shapes.generator import generate_oriented_two_ellipses
from src.torch.solver.mm_step import MMStepper
from src.torch.transport.bem_wasserstein import BEMWasserstein

sys.path.insert(0, str(Path(__file__).parent.parent
                       / "scripts" / "experiments"))
from p1_production_comparison import make_cfg  # noqa: E402

DT = 1e-5


def _pair(n_per=32):
    return generate_oriented_two_ellipses(
        n_per, a1=0.4, b1=1.0, center1=(-0.45, 0.0),
        a2=0.4, b2=1.0, center2=(0.45, 0.0), device="cpu",
        dtype=torch.float64)


def _run(kind, telemetry, n_steps=3):
    v = _pair()
    delta, tau = compute_recommended_params(v.positions)
    cfg = make_cfg(kind, delta, tau, 2)
    cfg.time_step = DT
    cfg.bem_setup_telemetry = telemetry
    st = MMStepper(cfg)
    vv, snaps = v, []
    for _ in range(n_steps):
        r = st.step(vv)
        vv = r.varifold
        snaps.append(r.bem_setup_snapshot)
    return vv, snaps


def test_noninterference_bitwise():
    """Telemetry ON/OFF: trajectories bitwise identical (both modes)."""
    for kind in ("C3", "C1"):
        v_off, s_off = _run(kind, False)
        v_on, s_on = _run(kind, True)
        assert torch.equal(v_off.positions, v_on.positions), kind
        assert torch.equal(v_off.angles, v_on.angles), kind
        assert all(s is None for s in s_off)
        assert all(s is not None for s in s_on)


def test_no_implicit_setup():
    """setup_snapshot_fields on a fresh instance returns None -- it
    must never trigger BEM assembly as a side effect."""
    bem = BEMWasserstein()
    assert bem.setup_snapshot_fields() is None
    assert bem._K_star_cache is None  # still untouched


def test_actual_vs_canonical_and_conventions():
    """C3 (carrier operator measure, unit velocity): the ACTUAL operator
    weights equal m (not q*m); policies say so; singular values are
    ascending with the source labelled; scales match the BEM-used
    values; all tensors are detached."""
    _, snaps = _run("C3", True, n_steps=1)
    sn = snaps[0]
    assert sn.operator_measure_policy == "carrier"
    assert torch.equal(sn.operator_weights_used, sn.carrier_masses_m)
    assert not torch.equal(sn.operator_weights_used, sn.canonical_qm)
    assert sn.velocity_policy == "unit"        # C3: coherence velocity off
    assert sn.volume_constraint_policy.startswith("q2m")
    assert torch.allclose(
        sn.canonical_q2m,
        sn.coherence_q ** 2 * sn.carrier_masses_m)
    s = sn.singular_values_ascending
    assert s is not None and torch.all(s[:-1] <= s[1:])
    assert sn.singular_values_source == "full_svd"
    assert sn.epsilon_bem_used is not None and sn.epsilon_bem_used > 0
    for t in (sn.positions, sn.carrier_masses_m, sn.coherence_q,
              sn.operator_weights_used, sn.endpoint_positions):
        assert not t.requires_grad

    # legacy C1: visible measure, coherence velocity, no spectrum
    _, snaps1 = _run("C1", True, n_steps=1)
    sn1 = snaps1[0]
    assert sn1.operator_measure_policy == "visible"
    assert torch.equal(sn1.operator_weights_used, sn1.canonical_qm)
    assert sn1.velocity_policy == "coherence_q"
    assert sn1.singular_values_ascending is None
    assert sn1.singular_values_source == "unavailable"
    # endpoint/collocation telemetry available in LEGACY mode too
    # (hang-attribution requirement)
    assert sn1.endpoint_positions is not None
    assert sn1.endpoints_per_particle == 3
