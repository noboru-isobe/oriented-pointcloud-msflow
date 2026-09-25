"""Passive advection of low-coherence marks by the step's own
transport field (arm 4 of the merger campaign).

Post-merger, hidden-boundary pairs freeze: their coherence q collapses
(antiparallel neighbors), the normal-graph parametrization moves marks
along their own normals with velocity ~ q, and a mark whose normal has
lost its meaning must not move along it (moving an antiparallel pair
along its own normals tears the pair apart and injects net current).
The confirmatory suite measured that the FROZEN fossil is what kills
the runs -- through the interaction of a static mark cluster with a
moving field (trench feedback, KDE carrier drift, growing dipole).

This stage restores the Otto picture for those marks: a mark that the
phase declares interior is MATERIAL, not boundary, and material is
transported by the flow. The step's own optimal potential gives the
displacement field: with the committed step y* and the frozen filling
convention

    delta_rho = B y*,      L_rho phi = delta_rho,   L_rho = -div(rho grad .)

the continuity equation delta_rho + div(rho d) = 0 is solved by the
gradient field d = grad phi (the divergence-free component is
deliberately dropped -- it is exactly the lift-fiber direction the
metric does not see, and adding it would excite the redundancy this
stage exists to tame). Marks move by

    dx_i     = (1 - q_i)^p * grad phi(x_i)
    dtheta_i = (1 - q_i)^p * ( - tau_i^T Hess(phi)(x_i) n_i )

The angle update is the exact material-interface rotation under the
displacement field d = grad phi. A potential flow has NO vorticity
(the discrete curl of the central-difference gradient vanishes
identically -- measured 0.0 exactly), so the rotation is pure strain:
theta_dot = -tau^T (grad d) n, symmetric grad d = Hess(phi). For an
antiparallel pair (n2 = -n1, tau2 = -tau1) the two signs cancel and
both members receive the SAME rotation -- pairing is preserved by
construction.

So a coherent boundary mark (q ~ 1) is untouched, a dead fossil
(q ~ 0) rides the flow as a tracer with its sheet direction
transported consistently. No labels, no thresholds, no events: the
blend weight is a smooth function of the current state and vanishes
with the pathology it addresses.

On genuine boundary the field is consistent by construction: grad
phi . n IS the interface normal velocity of the step (pinned by the
transport-parity test), so the q-blend does not fight the primary
dynamics anywhere.

Default OFF (MMConfig.grid_fossil_advection); grid backend only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_T = torch.Tensor


@dataclass(frozen=True)
class FossilAdvectionConfig:
    exponent: float = 1.0            # blend weight (1 - q)^p
    max_displacement_cells: float = 5.0   # safety clip, telemetry on hit


def advect_fossils(varifold, coherence: _T, displacements: _T,
                   delta_angles: _T, gm,
                   cfg: FossilAdvectionConfig = FossilAdvectionConfig()):
    """Advect low-q marks by the committed step's transport field.

    varifold: the COMMITTED post-step varifold (positions already moved
    by the accepted normal-graph step).
    coherence: q recomputed on that committed state.
    displacements/delta_angles: the accepted step y* (pre-step frozen
    parametrization -- the same vectors the metric was evaluated on).
    gm: the stepper's GridWassersteinMetric, still holding the
    PRE-step frozen phase/poisson/flux.

    Returns (advected_varifold, stats). On an incompatible or
    non-converged solve the stage SKIPS (returns the input varifold)
    with the reason recorded -- never fails the step, never falls
    back to a different field.
    """
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    from .phase_support_projection import _bilinear_periodic
    from .weighted_poisson import (
        IncompatibleGridVelocityError,
        WeightedPoissonSolveError,
    )

    stats = {"applied": False, "skipped_reason": None,
             "effective_advected_fraction": 0.0,
             "max_disp_over_dx": 0.0, "n_clipped": 0,
             "max_abs_dtheta": 0.0}
    weight = (1.0 - coherence).clamp_min(0.0) ** cfg.exponent
    stats["effective_advected_fraction"] = float(weight.mean())

    drho = gm.flux(displacements, delta_angles)
    try:
        phi = gm.poisson.solve_or_raise(drho)
    except (IncompatibleGridVelocityError,
            WeightedPoissonSolveError) as exc:
        stats["skipped_reason"] = f"{type(exc).__name__}: {exc}"
        return varifold, stats

    phase_cfg = gm.config.phase
    H = phase_cfg.grid_shape[0]
    dx = (phase_cfg.box_max[0] - phase_cfg.box_min[0]) / H
    # displacement field d = grad phi (periodic central differences)
    d_x = (phi.roll(-1, 0) - phi.roll(1, 0)) / (2.0 * dx)
    d_y = (phi.roll(-1, 1) - phi.roll(1, 1)) / (2.0 * dx)
    # material rotation is pure strain (potential flow, zero
    # vorticity): sample the Hessian of phi
    h_xx = (d_x.roll(-1, 0) - d_x.roll(1, 0)) / (2.0 * dx)
    h_xy = (d_x.roll(-1, 1) - d_x.roll(1, 1)) / (2.0 * dx)
    h_yy = (d_y.roll(-1, 1) - d_y.roll(1, 1)) / (2.0 * dx)

    pos = varifold.positions
    ux = _bilinear_periodic(d_x, pos, phase_cfg.box_min,
                            phase_cfg.box_max)
    uy = _bilinear_periodic(d_y, pos, phase_cfg.box_min,
                            phase_cfg.box_max)
    hxx = _bilinear_periodic(h_xx, pos, phase_cfg.box_min,
                             phase_cfg.box_max)
    hxy = _bilinear_periodic(h_xy, pos, phase_cfg.box_min,
                             phase_cfg.box_max)
    hyy = _bilinear_periodic(h_yy, pos, phase_cfg.box_min,
                             phase_cfg.box_max)
    disp = torch.stack([ux, uy], dim=1) * weight.unsqueeze(1)
    # dtheta = -tau^T H n, tau = rot90(n)
    n = varifold.normals
    nx, ny = n[:, 0], n[:, 1]
    tx, ty = -ny, nx
    hn_x = hxx * nx + hxy * ny
    hn_y = hxy * nx + hyy * ny
    dtheta = -(tx * hn_x + ty * hn_y) * weight

    # safety clip (fail-loud telemetry; the cap is generous -- a hit
    # means the field sampling went somewhere it should not have)
    cap = cfg.max_displacement_cells * dx
    norms = disp.norm(dim=1)
    clip = norms > cap
    if bool(clip.any()):
        disp[clip] *= (cap / norms[clip]).unsqueeze(1)
        stats["n_clipped"] = int(clip.sum())

    stats["applied"] = True
    stats["max_disp_over_dx"] = float(norms.max() / dx)
    stats["max_abs_dtheta"] = float(dtheta.abs().max())
    out = OrientedPointCloudVarifold(
        positions=pos + disp, angles=varifold.angles + dtheta)
    return out, stats
