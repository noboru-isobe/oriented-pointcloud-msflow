"""D1: the three dead-point time-level rules, one shared implementation.

Every consumer -- the D1a shadow audit, the exact-ODE-snapshot
calibration, the (future) active solver variants, and the timeline
logging -- calls the same pure function, so the audit can never drift
from the implementation it audits.

Rules (0D plan):

    legacy_lagged    e_i = m_i^{pre}          q_i^{pre}
    semi_implicit    e_i = m_i(x_post)        q_i^{pre}
    refreshed        e_i = m_i(x_post)        q_i(x_post)

with keep_i <=> e_i >= rho_dead * max_j e_j. The normalized threshold
margin

    margin_i = e_i / (rho_dead * max_j e_j) - 1

distinguishes essential rule differences from near-threshold numerical
ones (review D1 spec).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.torch.oriented_varifold.mass import compute_masses
from src.torch.transport import compute_coherence

_T = torch.Tensor

DEAD_POINT_RULES = ("legacy_lagged", "semi_implicit", "refreshed")


@dataclass(frozen=True)
class DeadPointDecision:
    rule: str
    score: _T
    threshold: float
    keep_mask: _T
    m_used: _T
    q_used: _T
    threshold_margin: _T

    @property
    def n_removed(self) -> int:
        return int((~self.keep_mask).sum().item())


def evaluate_dead_points(
    post_varifold,
    m_pre: _T,
    q_pre: _T,
    rule: str,
    rho_dead: float,
    *,
    delta_for_kde=None,
    tau_for_kde=None,
    mass_kernel: str = "wendland_c2",
    sigma: float = None,
    coherence_kernel: str = "wendland_c2",
    coherence_backend: str = "naive",
) -> DeadPointDecision:
    """Evaluate one dead-point rule on the post-step geometry.

    m_pre/q_pre are the frozen pre-step quantities (what legacy_lagged
    consumes); the current-geometry quantities are recomputed here with
    the ACTUAL production scales/kernels passed in (delta/tau required
    for semi_implicit and refreshed; sigma/kernel/backend additionally
    for refreshed).
    """
    if not (0.0 < rho_dead < 1.0):
        raise ValueError(f"rho_dead must lie in (0, 1); got {rho_dead}")
    if rule == "legacy_lagged":
        m_used, q_used = m_pre, q_pre
    elif rule in ("semi_implicit", "refreshed"):
        if delta_for_kde is None or tau_for_kde is None:
            raise ValueError(f"{rule} needs delta_for_kde and tau_for_kde")
        m_used = compute_masses(post_varifold.positions.detach(),
                                delta_for_kde, tau_for_kde, mass_kernel)
        if rule == "semi_implicit":
            q_used = q_pre
        else:
            if sigma is None:
                raise ValueError("refreshed needs sigma")
            q_used = compute_coherence(post_varifold, m_used, sigma,
                                       coherence_kernel,
                                       backend=coherence_backend)
    else:
        raise ValueError(f"unknown dead-point rule {rule!r}")

    n = post_varifold.n_points
    if m_used.shape[0] != n or q_used.shape[0] != n:
        raise ValueError(
            f"tensor length mismatch: geometry {n}, m {m_used.shape[0]},"
            f" q {q_used.shape[0]}")
    score = m_used * q_used
    if not torch.isfinite(score).all() or (score < 0).any():
        raise ValueError("scores must be finite and nonnegative")

    max_score = score.max().item()
    if max_score <= 0.0:
        # degenerate cloud: KEEP everything, and make the margin agree
        # with the mask (the previous code returned all-True keep with
        # all -1 margins -- self-contradictory)
        keep = torch.ones_like(score, dtype=torch.bool)
        return DeadPointDecision(
            rule=rule, score=score, threshold=0.0, keep_mask=keep,
            m_used=m_used, q_used=q_used,
            threshold_margin=torch.full_like(score, float("inf")))
    threshold = rho_dead * max_score
    keep = score >= threshold
    margin = score / threshold - 1.0
    return DeadPointDecision(rule=rule, score=score, threshold=threshold,
                             keep_mask=keep, m_used=m_used, q_used=q_used,
                             threshold_margin=margin)


def mask_symmetric_difference(a: DeadPointDecision,
                              b: DeadPointDecision) -> list:
    """Indices where the two rules decide differently."""
    return torch.nonzero(a.keep_mask != b.keep_mask).flatten().tolist()
