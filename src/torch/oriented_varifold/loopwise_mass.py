"""0K: loopwise-masked oriented mass -- shared source resolver.

The mass-loop bootstrap has an ordering circularity: loopwise masses
need certified loop labels, but the loop-graph spacing is derived from
the masses. This module is the ONE implementation of the two-stage
certificate that resolves it (reviewer 0K section 5):

    m_pre   oriented KDE (current estimator) bootstrap masses
    l_pre   3-scale certified partition at spacing median(m_pre)
    m_loop  loopwise-masked oriented masses on l_pre
    l_post  3-scale certified partition at spacing median(m_loop)
    require l_pre ~ l_post (co-membership) -- else fail-closed.

Consumers: MMStepper._setup_step, the exact-merger driver, the 0K-A/B
scripts, and telemetry -- all through resolve_loopwise_oriented_mass_
source, so the simulation and every diagnostic read the SAME masses.
The candidate objective alone calls compute_masses_oriented_loopwise
directly with the source-frozen labels (labels are never recomputed
per candidate -- that would make the objective non-smooth).

Guarantee wording (reviewer 0K section 9): cross-loop independence +
sheetwise H^1 consistency. NOT "mass preservation" -- loop perimeters
move, so sum m is not conserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import torch

from .mass import (
    chi_tau,
    compute_masses_oriented,
    compute_masses_oriented_loopwise,
    oriented_density_cross_split,
)


def certified_loop_labels(
    positions: torch.Tensor,
    normals: torch.Tensor,
    spacing: float,
    graph_scale_factors: Sequence[float] = (3.5, 4.0, 4.5),
    ell_factor: float = 4.0,
) -> torch.Tensor:
    """Boundary-loop labels from the co-oriented proximity graph at
    ell = ell_factor * spacing, certified co-membership-stable across
    graph_scale_factors (fail-closed on any disagreement). Single
    implementation shared by MMStepper and the 0K resolver."""
    from ..transport.incidence import (
        PartitionAmbiguousError,
        oriented_graph_components,
        partition_relation,
    )
    factors = tuple(graph_scale_factors)
    ref = factors[len(factors) // 2]
    parts = {f: oriented_graph_components(positions, normals,
                                          f * spacing)
             for f in factors}
    for f in factors:
        if f == ref:
            continue
        if partition_relation(parts[f], parts[ref]) != "equal":
            raise PartitionAmbiguousError(
                f"loop partition is graph-scale unstable "
                f"(ell factor {f} vs {ref}) -- fail-closed")
    if ell_factor in parts:
        return parts[ell_factor]
    return oriented_graph_components(positions, normals,
                                     ell_factor * spacing)


@dataclass(frozen=True)
class LoopwiseMassResolution:
    """Detached source-side snapshot of one 0K mass resolution
    (doubles as the MassLoopSetupSnapshot telemetry -- deliberately
    separate from GridMetricSetupSnapshot: 0K is a mass-layer
    construction, not a grid-metric one)."""
    m_pre: torch.Tensor
    loop_labels_pre: torch.Tensor
    m_loop: torch.Tensor
    loop_labels_post: torch.Tensor
    certificate_equal: bool
    theta_self: torch.Tensor
    theta_cross: torch.Tensor
    cutoff_count_loop: int      # chi_tau(theta^loop) < 1 firings
    cutoff_count_oriented: int  # chi_tau(theta^or) < 1 firings (audit)
    loop_stats: list = field(default_factory=list)
    # per loop: {label, n, M_l, zeta_cross}


def resolve_loopwise_oriented_mass_source(
    positions: torch.Tensor,
    normals: torch.Tensor,
    delta: float,
    tau: float,
    kernel: Literal["wendland_c2", "biweight",
                    "epanechnikov"] = "wendland_c2",
    graph_scale_factors: Sequence[float] = (3.5, 4.0, 4.5),
    ell_factor: float = 4.0,
) -> LoopwiseMassResolution:
    """Two-stage-certificated 0K source resolution (see module doc).

    Inputs must be detached source-state tensors and SCALAR (delta,
    tau) fixed for the trajectory (the loopwise estimator refuses
    adaptive bandwidths upstream: with the mask in place an adaptive
    bandwidth would re-open a channel for cross-loop geometry to
    enter the mass -- reviewer 0K section 2)."""
    from ..transport.incidence import (
        PartitionAmbiguousError,
        partition_relation,
    )
    if isinstance(delta, torch.Tensor) or isinstance(tau, torch.Tensor):
        raise ValueError(
            "0K loopwise mass requires scalar (delta, tau) frozen for "
            "the trajectory -- tensor bandwidths are rejected")
    pos = positions.detach()
    nor = normals.detach()
    m_pre = compute_masses_oriented(pos, nor, delta, tau, kernel)
    l_pre = certified_loop_labels(
        pos, nor, float(m_pre.median()), graph_scale_factors,
        ell_factor)
    m_loop = compute_masses_oriented_loopwise(
        pos, nor, l_pre, delta, tau, kernel)
    l_post = certified_loop_labels(
        pos, nor, float(m_loop.median()), graph_scale_factors,
        ell_factor)
    if partition_relation(l_pre, l_post) != "equal":
        raise PartitionAmbiguousError(
            "0K bootstrap certificate failed: the loop partition at "
            "spacing median(m_pre) and median(m_loop) disagree "
            "(co-membership) -- fail-closed")
    theta_self, theta_cross = oriented_density_cross_split(
        pos, nor, delta, kernel, loop_labels=l_pre)
    stats = []
    for lb in l_pre.unique():
        sel = l_pre == lb
        ts = float(theta_self[sel].sum())
        tc = float(theta_cross[sel].sum())
        stats.append(dict(
            label=int(lb), n=int(sel.sum()),
            M_l=float(m_loop[sel].sum()),
            zeta_cross=tc / ts if ts > 0 else None))
    return LoopwiseMassResolution(
        m_pre=m_pre,
        loop_labels_pre=l_pre,
        m_loop=m_loop,
        loop_labels_post=l_post,
        certificate_equal=True,
        theta_self=theta_self,
        theta_cross=theta_cross,
        cutoff_count_loop=int((chi_tau(theta_self, tau) < 1.0).sum()),
        cutoff_count_oriented=int(
            (chi_tau(theta_self + theta_cross, tau) < 1.0).sum()),
        loop_stats=stats,
    )
