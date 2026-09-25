# src/transport/bem_wasserstein.py
"""Linearized Wasserstein distance via Boundary Element Method (BEM).

This module implements a BEM-based approach to computing linearized Wasserstein
distance for the Mullins-Sekerka flow. Instead of entropy-regularized Sinkhorn,
this computes transport cost directly from boundary points via solving the
interior Neumann problem.

Key advantages over Sinkhorn:
- No grid integration needed
- No phase field (no alpha parameter)
- No entropy regularization artifacts
- Direct boundary-to-boundary computation
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Tuple, Optional

import torch

_T = torch.Tensor


def _mass_weighted_mean(x: _T, w: _T) -> _T:
    """Return (w·x)/(w·1) for boundary quadrature weighting."""
    return (w * x).sum() / w.sum()


def _orthogonal_complement(weights: _T) -> _T:
    """Compute orthogonal complement basis for constraint ∫λ dw = 0.

    Given weight vector w, returns Q such that:
    - Q^T Q = I (orthonormal columns)
    - w^T Q = 0 (columns orthogonal to w)
    - Q has shape (N, N-1)

    Uses deterministic QR: place normalized w as first column of identity,
    then QR decomposition gives orthogonal basis with w as first column.

    Args:
        weights: (N,) weight vector (e.g., effective_masses)

    Returns:
        Q: (N, N-1) orthogonal complement basis
    """
    c = weights / weights.norm()  # unit vector
    Q, _ = torch.linalg.qr(c[:, None], mode="complete")
    return Q[:, 1:]  # (N, N-1) orthogonal to weights


TRACE_SIDES = ("legacy_exterior", "interior")


class IncompatibleVelocityError(ValueError):
    """A velocity with nonzero flux through some phase component was passed
    to the spectral solver. The one-phase tangent norm is +inf there; the
    solver must refuse rather than project the data onto a compatible
    velocity and return a finite energy for something else.
    """


class AmbiguousComponentRankError(RuntimeError):
    """The rank evidence is not clean (weak gap, criterion disagreement, or
    no null block). The plain auto solver must stop rather than guess: a
    wrong rank builds a self-consistent displacement space on which r_comp
    is machine zero, so the error would be silent downstream. Carries the
    structured `evidence` so hysteresis policies can distinguish the cases
    this exception folds together.
    """

    def __init__(self, message: str, evidence: "RankEvidence" = None):
        super().__init__(message)
        self.evidence = evidence


@dataclass(frozen=True)
class RankEvidence:
    """Structured output of the two complementary rank criteria (both read
    from the SAME spectrum of the weighted operator; they are not
    independent measurements).

    status:
        clean                   criteria agree and the gap is strong
        weak_gap                criteria AGREE on C but the winning ratio is
                                below gap_min (confidence, not count, is
                                what is lacking -- the hysteresis candidate)
        detector_disagreement   the two criteria return different C
        no_null_block           nothing below null_tol at all
    """
    c_abs: int                 # absolute-null criterion count
    c_gap: int                 # max-gap criterion split
    gap_ratio: float           # winning ratio s_(c_gap+1)/s_(c_gap), ascending
    abs_levels: tuple          # leading relative spectrum (ascending)
    status: str

    @property
    def rank(self) -> Optional[int]:
        """The agreed count, defined whenever the two criteria agree
        (clean AND weak_gap); None on genuine disagreement."""
        return self.c_gap if self.c_abs == self.c_gap else None


def rank_evidence(
    s_desc: "_T", rank_max: int, null_tol: float, gap_min: float,
) -> RankEvidence:
    """Read component-rank evidence from the weighted operator's singular
    values (descending torch.linalg.svd order)."""
    s_asc = s_desc.flip(0)
    s_max = s_desc[0].clamp_min(1e-300)
    rel = (s_asc / s_max).tolist()
    kmax = min(rank_max, len(rel) - 1)

    # absolute-null criterion: contiguous leading block below null_tol
    k_abs = 0
    while k_abs < kmax and rel[k_abs] < null_tol:
        k_abs += 1

    # max-gap criterion over candidate splits k = 1..kmax
    ratios = [rel[k] / max(rel[k - 1], 1e-300) for k in range(1, kmax + 1)]
    k_gap = max(range(1, kmax + 1), key=lambda k: ratios[k - 1])
    gap_ratio = ratios[k_gap - 1]

    if k_abs == 0:
        status = "no_null_block"
    elif k_gap != k_abs:
        status = "detector_disagreement"
    elif gap_ratio < gap_min:
        status = "weak_gap"
    else:
        status = "clean"
    return RankEvidence(k_abs, k_gap, gap_ratio, tuple(rel[: kmax + 2]),
                        status)


def _trace_signs(trace_side: str) -> Tuple[float, float]:
    """Jump-relation signs for the single-layer normal derivative.

    With G(x, y) = -(1/2pi) log|x - y| and nu the outward normal of E, the
    *interior* trace is d_nu^int S lam = (+1/2 I + K*) lam, so the interior
    Neumann problem d_nu phi = v reads (+1/2 I + K*) lam = +v.

    The original code used (-1/2 I + K*) lam = -v, which is the *exterior*
    trace. On a single circle the two sign flips cancel for mean-zero Fourier
    modes (K* annihilates them), so the error is invisible there; it shows up
    as soon as K* v != 0 -- any non-circular shape, and any boundary with more
    than one component.

    Returns:
        (identity_coefficient, rhs_sign)
    """
    if trace_side == "interior":
        return 0.5, 1.0
    if trace_side == "legacy_exterior":
        return -0.5, -1.0
    raise ValueError(f"trace_side must be one of {TRACE_SIDES}, got {trace_side!r}")


def _build_projected_system(
    K_star: _T,
    effective_masses: _T,
    reg: float = 0.0,
    trace_side: str = "legacy_exterior",
) -> Tuple[_T, _T, Tuple]:
    """Build projected BEM system A_proj = Q^T A Q and its LU factorization.

    The BEM operator A = (+-1/2)I + K* has a null space (constant functions).
    We project onto the constraint subspace ∫λ dm = 0.

    Args:
        K_star: Adjoint double layer matrix (N, N)
        effective_masses: masses * coherence (N,)
        reg: Optional regularization (default 0.0)
        trace_side: "interior" or "legacy_exterior" (see `_trace_signs`)

    Returns:
        (A, Q, A_proj_LU) where:
        - A: Full BEM operator (N, N)
        - Q: Orthogonal complement basis (N, N-1)
        - A_proj_LU: LU factorization of Q^T A Q
    """
    N = effective_masses.numel()
    device, dtype = effective_masses.device, effective_masses.dtype

    half, _ = _trace_signs(trace_side)
    I = torch.eye(N, device=device, dtype=dtype)
    A = half * I + K_star
    if reg > 0:
        A = A + reg * I

    Q = _orthogonal_complement(effective_masses)
    A_proj = Q.T @ A @ Q  # (N-1, N-1)
    A_proj_LU = torch.linalg.lu_factor(A_proj)

    return A, Q, A_proj_LU


def build_bem_matrices_point(
    positions: _T,   # (N, 2)
    normals: _T,     # (N, 2) outward normals
    masses: _T,      # (N,) quadrature weights from KDE
    epsilon: float,
) -> Tuple[_T, _T]:
    """Point-eval BEM matrices (cheap, needs epsilon).

    Builds Single Layer (S) and Adjoint Double Layer (K*) matrices using
    point evaluation with epsilon regularization for diagonal terms.

    Args:
        positions: Point positions (N, 2)
        normals: Unit outward normals (N, 2)
        masses: Quadrature weights from KDE (N,)
        epsilon: Regularization for diagonal (typically 0.5 * ℓ)

    Returns:
        S: Single layer matrix (N, N)
        K_star: Adjoint double layer matrix (N, N)
    """
    N = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]   # (N,N,2) x_i - x_j
    r2 = (diff * diff).sum(dim=-1) + epsilon * epsilon     # (N,N)

    # Single layer: G = -(1/(2π)) log|r| = -(1/(4π)) log(r²)
    S = -(1.0 / (4.0 * math.pi)) * torch.log(r2) * masses[None, :]

    # Adjoint double layer: ∂G/∂n_x = -(1/(2π)) ((x-y)·n_x)/|x-y|²
    # NOTE: Correct NEGATIVE sign!
    dot_n = (diff * normals[:, None, :]).sum(dim=-1)
    K_star = -(1.0 / (2.0 * math.pi)) * (dot_n / r2) * masses[None, :]

    # Principal value: diagonal = 0
    K_star = K_star.clone()
    K_star.fill_diagonal_(0.0)

    return S, K_star


def _F(u: _T, eta: _T, eps: float = 1e-12) -> _T:
    """Primitive for ∫ log(u² + η²) du.

    Used in analytical panel integration for the single layer potential.
    """
    r2 = torch.clamp(u * u + eta * eta, min=eps * eps)
    general = u * torch.log(r2) - 2.0 * u + 2.0 * eta * torch.atan2(u, eta)

    near = torch.abs(eta) < eps
    u_safe = torch.where(torch.abs(u) < eps, eps * torch.ones_like(u), u)
    eta0 = 2.0 * u * torch.log(torch.abs(u_safe)) - 2.0 * u

    return torch.where(near, eta0, general)


def _H_for_Kstar(u: _T, eta: _T, A: _T, B: _T, eps: float = 1e-12) -> _T:
    """Primitive for K* kernel integral.

    H(u) = -(A/2) log(u² + η²) + B arctan(u/η)

    For η=0: H(u) = -(A/2) log(u²)  (no jump term - handled by -1/2 I)

    Used in analytical panel integration for the adjoint double layer.
    """
    r2 = torch.clamp(u * u + eta * eta, min=eps * eps)
    general = -(A * 0.5) * torch.log(r2) + B * torch.atan2(u, eta)

    near = torch.abs(eta) < eps
    u2 = torch.clamp(u * u, min=eps * eps)
    eta0 = -(A * 0.5) * torch.log(u2)  # B-term dropped for η=0

    return torch.where(near, eta0, general)


def build_bem_matrices_panel(
    positions: _T,  # (N,2) point positions
    tangents: _T,   # (N,2) unit tangents (perpendicular to normals)
    normals: _T,    # (N,2) unit normals
    masses: _T,     # (N,) panel lengths (from KDE masses)
    eps: float = 1e-12,
) -> Tuple[_T, _T]:
    """Analytical panel integration (vectorized).

    Builds Single Layer (S) and Adjoint Double Layer (K*) matrices using
    analytical integration over panel segments. More accurate than point
    evaluation but more complex.

    Each point is treated as the center of a panel with:
    - Length: masses[j] (from KDE, represents boundary amount)
    - Direction: tangents[j] (perpendicular to normal)

    Args:
        positions: Point positions (N, 2)
        tangents: Unit tangent vectors (N, 2), perpendicular to normals
        normals: Unit outward normals (N, 2)
        masses: Panel lengths from KDE (N,)
        eps: Numerical epsilon for stability

    Returns:
        S: Single layer matrix (N, N)
        K_star: Adjoint double layer matrix (N, N)
    """
    N = positions.shape[0]
    half = (0.5 * masses)[:, None]
    a = positions - half * tangents  # panel start

    tau = tangents
    tau_perp = torch.stack([-tau[:, 1], tau[:, 0]], dim=-1)

    diff = positions[:, None, :] - a[None, :, :]  # (N,N,2)
    xi = (diff * tau[None, :, :]).sum(dim=-1)
    eta = (diff * tau_perp[None, :, :]).sum(dim=-1)

    L = masses[None, :].expand(N, N)
    u0, u1 = -xi, L - xi

    # Single layer
    S = -(1.0 / (4.0 * math.pi)) * (_F(u1, eta, eps) - _F(u0, eta, eps))

    # K* coefficients: (x-y)·n_i = -A*u + B*η
    A = (normals[:, None, :] * tau[None, :, :]).sum(dim=-1)
    B = (normals[:, None, :] * tau_perp[None, :, :]).sum(dim=-1)

    K_star = -(1.0 / (2.0 * math.pi)) * (
        _H_for_Kstar(u1, eta, A, B, eps) - _H_for_Kstar(u0, eta, A, B, eps)
    )

    # Principal value: diagonal = 0
    K_star = K_star.clone()
    K_star.fill_diagonal_(0.0)

    return S, K_star


def compute_tangents_from_normals(normals: _T) -> _T:
    """Compute unit tangent vectors from normals (90-degree rotation).

    tangent = rotate(normal, 90°) = (-n_y, n_x)

    Args:
        normals: Unit outward normals (N, 2)

    Returns:
        tangents: Unit tangent vectors (N, 2)
    """
    return torch.stack([-normals[:, 1], normals[:, 0]], dim=-1)


def solve_neumann_interior(
    S: _T,
    K_star: _T,
    V: _T,
    effective_masses: _T,
    reg: float = 0.0,
    Q: Optional[_T] = None,
    A_proj_LU: Optional[Tuple] = None,
    trace_side: str = "legacy_exterior",
) -> Tuple[_T, _T, Tuple]:
    """Solve interior Neumann problem with subspace projection.

    Jump formula selected by `trace_side` (see `_trace_signs`):
      "interior"        A = (+1/2)I + K*,  rhs = +V   (correct interior trace)
      "legacy_exterior" A = (-1/2)I + K*,  rhs = -V   (what the code shipped)
    Enforces ∫λ dm = 0 via projection (cleaner than KKT augmentation).

    Method (avoids saddle-point conditioning):
    1. c = effective_masses / ||effective_masses||  (unit vector)
    2. Q ∈ R^{N×(N-1)}: c^T Q = 0, Q^T Q = I (orthogonal complement)
    3. λ = Q y (constraint auto-satisfied)
    4. Solve (Q^T A Q) y = Q^T b, then λ = Q y

    Args:
        S: Single layer matrix (N, N)
        K_star: Adjoint double layer matrix (N, N)
        V: Neumann boundary data (N,)
        effective_masses: masses * coherence (N,) - weights visible boundary only
        reg: Regularization parameter (default 0.0)
        Q: Orthogonal complement basis (N, N-1). If None, compute.
        A_proj_LU: Cached LU of Q^T A Q. If None, compute.

    Returns:
        (phi, Q, A_proj_LU) - solution and caches for reuse
    """
    # Enforce compatibility: ∫ V dm = 0
    Vc = V - _mass_weighted_mean(V, effective_masses)

    # Build projection basis and projected system if no cache
    if Q is None or A_proj_LU is None:
        _, Q, A_proj_LU = _build_projected_system(
            K_star, effective_masses, reg, trace_side
        )

    # Project RHS: b_proj = Q^T (rhs_sign * Vc)
    _, rhs_sign = _trace_signs(trace_side)
    b_proj = Q.T @ (rhs_sign * Vc)  # (N-1,)

    # Solve projected system
    y = torch.linalg.lu_solve(*A_proj_LU, b_proj.unsqueeze(-1)).squeeze(-1)

    # Recover λ = Q y (automatically satisfies effective_masses^T λ = 0)
    lam = Q @ y

    # Compute potential
    phi = S @ lam

    # Gauge fix
    phi = phi - _mass_weighted_mean(phi, effective_masses)
    return phi, Q, A_proj_LU


@dataclass(frozen=True)
class EndpointOperator:
    """Endpoint-collocation BEM system, exposed for static audits.

    Attributes:
        positions: (NK, 2) collocation points p_{ik} = x_i + r_{ik} t_i
        weights: (NK,) quadrature weights zeta_{ik} = m_i / K. This is the
            W_end that defines the weighted inner product; nullspace audits
            must use it rather than the particle-level diag(m_i).
        single_layer: (NK, NK) S
        adjoint_double_layer: (NK, NK) K*
        trace_operator: (NK, NK) A = (+-1/2)I + K*, sign per `trace_side`
        constraint_basis: (NK, NK-1) orthogonal complement of the weight
            vector, i.e. the single global compatibility constraint the
            solver currently imposes
        epsilon: diagonal regularisation actually used (None for "panel")
        n_endpoints: K
        n_particles: N
        trace_side: which jump relation `trace_operator` encodes
    """
    positions: _T
    weights: _T
    single_layer: _T
    adjoint_double_layer: _T
    trace_operator: _T
    constraint_basis: _T
    epsilon: Optional[float]
    n_endpoints: int
    n_particles: int
    trace_side: str

    def weighted_trace_operator(self) -> _T:
        """W^{1/2} A W^{-1/2}, whose singular vectors give the weighted
        left/right nullspaces (see the 0C spectral audit)."""
        sw = self.weights.sqrt()
        return (sw[:, None] * self.trace_operator) / sw[None, :]


@dataclass
class BEMWasserstein:
    """Linearized Wasserstein via BEM with endpoint collocation (differentiable).

    KEY INSIGHT: S, K*, and A are built from PREVIOUS step's boundary
    (fixed during optimization). Only V changes with displacements s_i.
    → Cache LU factorization for massive speedup in MM optimization loop!

    ENDPOINT COLLOCATION (K=2 per segment):
    Each segment i has endpoints at r = ±m_i/2 along the tangent.
    Velocity at endpoint k: V_{ik} = (q_i/h) × (s_i - r_k × δθ_i)
    This captures angle change δθ_i in the transport cost (center-only loses it).

    Attributes:
        method: "panel" (analytical, recommended) or "point" (needs epsilon)
        epsilon_scale: Scale factor for epsilon in point method (ε = scale × ℓ)
        solver_reg: Regularization for the BEM linear system
        n_endpoints: Number of collocation points per segment (default 3)
    """
    method: str = "point"           # "point" (recommended) or "panel"
    epsilon_scale: float = 0.1      # only for "point"; must be > 0
    solver_reg: float = 0.0
    n_endpoints: int = 3            # K=3 collocation points per segment
    # Which single-layer jump relation to use; see `_trace_signs`. The default
    # reproduces every published artifact. "interior" is the correct trace for
    # the one-phase interior Neumann problem this solver is meant to discretise.
    trace_side: str = "legacy_exterior"
    # How the diagonal regularisation eps is chosen (0A scale API). "sigma"
    # reproduces every published artifact but couples eps to the coherence
    # bandwidth, so a sigma-refinement study silently changes the operator;
    # "carrier_segment_length" is q-independent and is what new benchmarks
    # should request. See `_resolve_epsilon`.
    epsilon_mode: str = "sigma"
    epsilon_explicit: Optional[float] = None
    # 0C-5a solver selection. "legacy_global_projection" is the production
    # path (single global compatibility, implicit weighted-mean subtraction
    # of the RHS). "spectral_bordered" solves the compatibility-constrained
    # bordered system in weighted coordinates with `component_rank` near-null
    # directions; it performs NO mean subtraction and REJECTS incompatible
    # data via `compat_reject_tol` (see IncompatibleVelocityError). Requires
    # the interior trace: the exterior trace's nullspace does not encode
    # phase components.
    solver_mode: str = "legacy_global_projection"
    # speedup Part 1 (flag-gated, default off): spectral W_lin through
    # the hand-differentiated quadratic-form Function
    analytic_gradient: bool = False
    _spec_quad: Optional[_T] = field(default=None, repr=False)
    # speedup Part 2 (flag-gated, default full): near-null extraction
    # method. "lobpcg" = warm-started Gram-free scipy LOBPCG with the
    # review's parity/fallback protocol; every fallback reason is
    # recorded in last_timings["svd_method_used"].
    svd_method: str = "full"
    _lobpcg_X: Optional[object] = field(default=None, repr=False)
    _lobpcg_calls: int = field(default=0, repr=False)
    _lobpcg_top: Optional[object] = field(default=None, repr=False)
    component_rank: Optional[int] = None
    compat_reject_tol: float = 0.1
    # 0C-5b: label-free rank selection. "oracle" uses component_rank as-is;
    # "auto" reads C from the weighted operator's singular values with TWO
    # independent detectors and REFUSES to guess when they disagree
    # (AmbiguousComponentRankError). NOTE r_comp is no rank validation: once
    # the displacement basis is built from the selected U0 it is machine
    # zero by construction even for a WRONG rank — hence the separate
    # confidence diagnostics below.
    rank_mode: str = "oracle"
    rank_max: int = 4
    # absolute-null criterion: s_(k)/s_max < rank_null_tol counts as null
    rank_null_tol: float = 0.05
    # max-gap criterion must win by at least this ratio, else ambiguous
    rank_gap_min: float = 30.0
    # 0C-5c: masses defining panel geometry / quadrature / K* / rank
    # detection. "visible" = q*m (legacy), "carrier" = m.
    operator_measure: str = "visible"
    _spec_U0: Optional[_T] = field(default=None, repr=False)
    _spec_V0: Optional[_T] = field(default=None, repr=False)
    _spec_LU: Optional[Tuple] = field(default=None, repr=False)
    _spec_svd: Optional[Tuple] = field(default=None, repr=False)
    _spec_A_weighted: Optional[_T] = field(default=None, repr=False)
    _spec_U0_pending_prev: Optional[_T] = field(default=None, repr=False)
    _last_r_comp: Optional[float] = field(default=None, repr=False)
    # rank-confidence diagnostics of the last setup (auto and oracle modes)
    detected_rank: Optional[int] = field(default=None, repr=False)
    rank_status: Optional[str] = field(default=None, repr=False)
    rank_evidence_last: Optional["RankEvidence"] = field(default=None,
                                                        repr=False)
    rank_spectrum_head: Optional[list] = field(default=None, repr=False)
    rank_gap_ratio: Optional[float] = field(default=None, repr=False)
    rank_abs_level: Optional[float] = field(default=None, repr=False)
    subspace_angle_deg: Optional[float] = field(default=None, repr=False)
    # wall-clock cost of the last setup: {assembly, svd, lu} seconds
    last_timings: Optional[dict] = field(default=None, repr=False)

    # Cache for projection-based solve (reused during optimization)
    _S_cache: Optional[_T] = field(default=None, repr=False)
    _K_star_cache: Optional[_T] = field(default=None, repr=False)
    _endpoint_positions_cache: Optional[_T] = field(default=None, repr=False)
    _epsilon_used: Optional[float] = field(default=None, repr=False)
    _endpoint_weights_cache: Optional[_T] = field(default=None, repr=False)
    _Q_cache: Optional[_T] = field(default=None, repr=False)
    _A_proj_LU_cache: Optional[Tuple] = field(default=None, repr=False)

    # Cache for endpoint velocity computation
    _N: int = field(default=0, repr=False)
    _K: int = field(default=3, repr=False)
    _r_physical: Optional[_T] = field(default=None, repr=False)  # (N, K) offset values
    _coherence_cache: Optional[_T] = field(default=None, repr=False)  # (N,) frozen coherence
    _use_coherence_velocity: bool = field(default=True, repr=False)  # q in V_{ik}?

    def _resolve_epsilon(self, effective_masses: _T,
                         carrier_masses: Optional[_T],
                         sigma: Optional[float], c_sigma: float) -> float:
        """Diagonal regularisation per `epsilon_mode`.

        "sigma"                  eps = scale * sigma / c_sigma (legacy; falls
                                 back to median(effective) when sigma is None)
        "visible_segment_length" eps = scale * median(q_i m_i)
        "carrier_segment_length" eps = scale * median(m_i)  -- q-independent
        "explicit"               eps = epsilon_explicit
        """
        if self.epsilon_mode == "sigma":
            # legacy branch, kept bit-identical to reproduce artifacts
            ell = (sigma / c_sigma) if sigma else effective_masses.median().item()
            return self.epsilon_scale * float(ell)
        if self.epsilon_mode == "visible_segment_length":
            return self.epsilon_scale * float(effective_masses.median().item())
        if self.epsilon_mode == "carrier_segment_length":
            if carrier_masses is None:
                raise ValueError(
                    "epsilon_mode='carrier_segment_length' needs "
                    "carrier_masses passed to setup_for_step")
            return self.epsilon_scale * float(carrier_masses.median().item())
        if self.epsilon_mode == "explicit":
            if self.epsilon_explicit is None:
                raise ValueError(
                    "epsilon_mode='explicit' needs epsilon_explicit set")
            return float(self.epsilon_explicit)
        raise ValueError(
            f"epsilon_mode must be one of 'sigma', 'visible_segment_length',"
            f" 'carrier_segment_length', 'explicit'; got {self.epsilon_mode!r}")

    def setup_for_step(
        self,
        positions: _T,           # Previous step's positions (FIXED)
        normals: _T,             # Previous step's normals (FIXED)
        effective_masses: _T,    # masses * coherence (FIXED)
        sigma: Optional[float] = None,
        c_sigma: float = 3.0,
        carrier_masses: Optional[_T] = None,   # m_i, for the carrier eps mode
    ):
        """Pre-compute BEM matrices and projection cache for an MM step.

        Uses K=2 endpoint collocation per segment:
        - Endpoint positions: p_{ik} = x_i + (r_k × m_i) × t_i
        - r_k = -0.5, +0.5 (normalized endpoints of segment)

        Args:
            positions: Point positions from previous step (N, 2) - FIXED
            normals: Unit outward normals from previous step (N, 2) - FIXED
            effective_masses: masses * coherence (N,) - segment lengths for quadrature
            sigma: Perimeter bandwidth (for epsilon computation in point method)
            c_sigma: Multiplier for sigma→ℓ conversion
        """
        device, dtype = positions.device, positions.dtype
        N = positions.shape[0]
        K = self.n_endpoints

        # Derive tangents from normals (90-degree rotation)
        tangents = compute_tangents_from_normals(normals)

        # Operator measure (0C-5c): which masses define the panel geometry,
        # the quadrature weights, K*, and the rank-detection operator.
        # "visible" (legacy) uses q*m; "carrier" uses the raw m, keeping the
        # topology sensor and the bordered solve q-independent -- 0C-2c
        # showed that q*m operator quadrature can destroy the Neumann
        # nullspace rank. The flux measure (1 vs q in the endpoint velocity)
        # is a SEPARATE choice; see setup_coherence.
        if self.operator_measure == "carrier":
            if carrier_masses is None:
                raise ValueError(
                    "operator_measure='carrier' needs carrier_masses")
            masses = carrier_masses
        elif self.operator_measure == "visible":
            masses = effective_masses
        else:
            raise ValueError(
                f"operator_measure must be 'visible' or 'carrier'; got "
                f"{self.operator_measure!r}")

        # Generate r values for endpoints
        if K == 1:
            # Center point only (r=0, no delta_theta contribution in V_ik formula)
            r_vals = torch.zeros(1, device=device, dtype=dtype)
        else:
            # K>1: linspace gives endpoints [-0.5, ..., 0.5] (normalized by m_i)
            r_vals = torch.linspace(-0.5, 0.5, K, device=device, dtype=dtype)  # (K,)

        # Physical offsets: r_physical[i, k] = r_vals[k] * m_i
        self._r_physical = r_vals[None, :] * masses[:, None]  # (N, K)

        # Endpoint positions: p_{ik} = x_i + r_physical_{ik} * t_i
        offsets = self._r_physical[:, :, None] * tangents[:, None, :]  # (N, K, 2)
        endpoint_positions = positions[:, None, :] + offsets  # (N, K, 2)
        endpoint_positions_flat = endpoint_positions.reshape(N * K, 2)  # (N*K, 2)

        # Endpoint normals: same as segment normal (each segment has uniform normal)
        endpoint_normals_flat = normals[:, None, :].expand(N, K, 2).reshape(N * K, 2)

        # Quadrature weights: w_{ik} = m_i / K (divide segment mass equally)
        endpoint_weights = (masses[:, None] / K).expand(N, K)  # (N, K)
        endpoint_weights_flat = endpoint_weights.reshape(N * K)  # (N*K,)

        # Build BEM matrices for N*K endpoint points
        _t_assembly0 = time.perf_counter()
        if self.method == "panel":
            endpoint_tangents_flat = tangents[:, None, :].expand(N, K, 2).reshape(N * K, 2)
            S, K_star = build_bem_matrices_panel(
                endpoint_positions_flat, endpoint_tangents_flat,
                endpoint_normals_flat, endpoint_weights_flat
            )
        else:
            eps = self._resolve_epsilon(masses, carrier_masses, sigma, c_sigma)
            S, K_star = build_bem_matrices_point(
                endpoint_positions_flat, endpoint_normals_flat,
                endpoint_weights_flat, eps
            )

        self._S_cache = S
        self._K_star_cache = K_star
        self._endpoint_positions_cache = endpoint_positions_flat
        self._epsilon_used = None if self.method == "panel" else float(eps)
        self._endpoint_weights_cache = endpoint_weights_flat
        # telemetry: the per-particle masses the OPERATOR actually consumed
        # (carrier m or visible q*m depending on operator_measure)
        self._operator_masses_cache = masses.detach()
        self._N = N
        self._K = K
        _t_assembly1 = time.perf_counter()
        self.last_timings = {"assembly": _t_assembly1 - _t_assembly0}

        if self.solver_mode == "spectral_bordered":
            # 0C-5a: compatibility-constrained bordered system in weighted
            # coordinates, `component_rank` near-null directions.
            if self.trace_side != "interior":
                raise ValueError(
                    "spectral_bordered requires trace_side='interior': the "
                    "exterior trace's nullspace does not encode phase "
                    "components")
            if self.rank_mode == "oracle":
                if self.component_rank is None:
                    raise ValueError(
                        "spectral_bordered with rank_mode='oracle' needs "
                        "component_rank")
            elif self.rank_mode not in ("auto", "deferred"):
                raise ValueError(
                    f"rank_mode must be 'oracle', 'auto', or 'deferred'; "
                    f"got {self.rank_mode!r}")
            if self.solver_reg != 0.0:
                # The legacy path adds solver_reg*I to its projected system;
                # the bordered solve has no such term. Forbid a nonzero value
                # rather than silently ignoring it (0C-5a.1).
                raise ValueError(
                    "spectral_bordered does not support solver_reg != 0 "
                    f"(got {self.solver_reg}); the bordered system is "
                    "unregularized by construction")
            half, _ = _trace_signs(self.trace_side)
            NK = endpoint_weights_flat.numel()
            I = torch.eye(NK, device=device, dtype=dtype)
            A = half * I + K_star
            sqrt_w = endpoint_weights_flat.sqrt()
            A_weighted = (sqrt_w[:, None] * A) / sqrt_w[None, :]
            _t_svd0 = time.perf_counter()
            U, _S, Vh = self._spectral_decompose(A_weighted)
            self.last_timings["svd"] = time.perf_counter() - _t_svd0
            self._spec_svd = (U, _S, Vh)
            self._spec_A_weighted = A_weighted
            ev = rank_evidence(_S, self.rank_max, self.rank_null_tol,
                               self.rank_gap_min)
            self.rank_evidence_last = ev
            if self.rank_mode == "auto":
                if ev.status != "clean":
                    raise AmbiguousComponentRankError(
                        f"rank evidence status={ev.status}: "
                        f"C_abs={ev.c_abs}, C_gap={ev.c_gap}, gap ratio "
                        f"{ev.gap_ratio:.1f} (min {self.rank_gap_min}), "
                        f"null tol {self.rank_null_tol}; leading relative "
                        f"spectrum (ascending) = "
                        f"{[f'{x:.2e}' for x in ev.abs_levels]}",
                        evidence=ev)
                self.finalize_with_rank(ev.c_gap)
            elif self.rank_mode == "oracle":
                self.finalize_with_rank(int(self.component_rank))
            else:
                # "deferred" (0C-5d-1.1): the caller's rank state machine
                # inspects the evidence, decides the WORKING rank (which may
                # be HELD through weak evidence), then calls
                # finalize_with_rank itself. Evolution does not stop on
                # same-rank weak evidence in this mode.
                self._spec_U0_pending_prev = self._spec_U0
                self._spec_LU = None
            self._Q_cache = None
            self._A_proj_LU_cache = None
        elif self.solver_mode == "legacy_global_projection":
            # Build projected system (use endpoint_weights, not mixing q)
            _, self._Q_cache, self._A_proj_LU_cache = _build_projected_system(
                K_star, endpoint_weights_flat, self.solver_reg, self.trace_side
            )
        else:
            raise ValueError(
                f"solver_mode must be 'legacy_global_projection' or "
                f"'spectral_bordered'; got {self.solver_mode!r}")

    def _spectral_decompose(self, A_weighted: _T):
        """Full SVD (baseline) or warm-started Gram-free LOBPCG for the
        near-null end of the spectrum (speedup Part 2, flag-gated).

        LOBPCG path: k = rank_max + 3 smallest left singular pairs of
        A via scipy lobpcg on A A^T (LinearOperator matvec = two dense
        mat-vecs; the Gram matrix is never formed), s_max via one
        largest pair, right vectors recovered as v = A^T u / s. Output
        keeps the descending torch.linalg.svd convention: last columns
        of U / last rows of Vh are the smallest pairs, and the
        synthetic spectrum [s_max, s_small_desc...] feeds the SAME
        rank_evidence function (it reads only the ascending head, k >=
        rank_max + 1 entries available).

        FALLBACKS to full SVD (reason recorded): scipy failure,
        residual above tolerance, non-finite output, size change
        (cold-start invalidation), and a periodic refresh every 50
        calls. Weak/ambiguous EVIDENCE is not hidden by the method --
        evidence is computed identically downstream and the rank
        machinery reacts to it exactly as with full SVD.
        """
        if self.svd_method != "lobpcg":
            return torch.linalg.svd(A_weighted)

        import numpy as np
        from scipy.sparse.linalg import LinearOperator, lobpcg

        NK = A_weighted.shape[0]
        k = int(self.rank_max) + 3
        reason = None
        try:
            An = A_weighted.detach().cpu().numpy()

            def mv(x):
                return An @ (An.T @ x)

            op = LinearOperator((NK, NK), matvec=mv,
                                matmat=lambda X: An @ (An.T @ X),
                                dtype=An.dtype)
            X0 = self._lobpcg_X
            self._lobpcg_calls += 1
            if X0 is None or X0.shape != (NK, k):
                rng = np.random.default_rng(0)
                X0 = rng.standard_normal((NK, k))
                reason_warm = "cold"
            else:
                reason_warm = "warm"
            if self._lobpcg_calls % 50 == 0:
                reason = "periodic_refresh"
            else:
                vals, vecs = lobpcg(op, X0, largest=False, tol=1e-10,
                                    maxiter=200)
                order = np.argsort(vals)
                vals, vecs = vals[order], vecs[:, order]
                res = np.linalg.norm(
                    mv(vecs) - vecs * vals[None, :], axis=0)
                s_small = np.sqrt(np.clip(vals, 0.0, None))
                top = getattr(self, "_lobpcg_top", None)
                if (top is None or top.shape[0] != NK
                        or self._lobpcg_calls % 10 == 0):
                    v1, u1 = lobpcg(op, np.random.default_rng(1)
                                    .standard_normal((NK, 1)),
                                    largest=True, tol=1e-8, maxiter=100)
                    top = u1[:, 0]
                else:
                    for _ in range(2):      # warm power iterations
                        top = mv(top)
                        top = top / np.linalg.norm(top)
                self._lobpcg_top = top
                s_max = float(np.sqrt(max(
                    float(top @ mv(top)), 0.0)))
                if (not np.isfinite(s_small).all()
                        or not np.isfinite(s_max)
                        or res.max() > 1e-8 * max(s_max ** 2, 1e-30)):
                    reason = "residual_or_nonfinite"
                else:
                    self._lobpcg_X = vecs.copy()
                    U_small = torch.from_numpy(vecs).to(A_weighted)
                    # descending convention: smallest LAST
                    U_eff = U_small.flip(1)
                    s_t = torch.from_numpy(s_small).to(A_weighted)
                    S_eff = torch.cat([s_t.new_tensor([s_max]),
                                       s_t.flip(0)])
                    # right vectors: v = A^T u / s
                    Vt = (A_weighted.T @ U_eff) / \
                        S_eff[1:].clamp_min(1e-300)[None, :]
                    Vt = Vt / Vt.norm(dim=0, keepdim=True).clamp_min(
                        1e-300)
                    Vh_eff = Vt.T
                    self.last_timings["svd_method_used"] = \
                        f"lobpcg_{reason_warm}"
                    return U_eff, S_eff, Vh_eff
        except Exception as e:                       # scipy failure
            reason = f"exception:{type(e).__name__}"
        self._lobpcg_X = None                        # cold restart next
        self.last_timings["svd_method_used"] = \
            f"full_fallback:{reason}"
        return torch.linalg.svd(A_weighted)

    def finalize_with_rank(self, C: int):
        """Build U0, V0 and the bordered LU at the given WORKING rank from
        the cached SVD of the current step's weighted operator.

        Called internally in 'oracle'/'auto' rank modes; in 'deferred' mode
        the caller's rank state machine decides C (possibly holding the
        previous rank through weak evidence) and calls this afterwards.
        """
        assert self._spec_svd is not None, \
            "spectral setup_for_step required first"
        U, _S, Vh = self._spec_svd
        A_weighted = self._spec_A_weighted
        NK = A_weighted.shape[0]
        ev = self.rank_evidence_last
        rel = ev.abs_levels
        self.detected_rank = int(C)
        self.rank_status = ev.status
        self.rank_gap_ratio = float(rel[C] / rel[C - 1]
                                    if C < len(rel) else float("inf"))
        self.rank_abs_level = float(rel[C - 1]) if C - 1 < len(rel) else None
        self.rank_spectrum_head = [float(x) for x in rel]
        # subspace continuation: principal angle to the previous step's
        # left null basis (same endpoint count only; None on first step
        # or after deletion). Computed BEFORE overwriting the cache.
        U0 = U[:, -C:].detach()
        prev_U0 = self._spec_U0
        if (prev_U0 is not None and prev_U0.shape[0] == U0.shape[0]
                and prev_U0.shape[1] == C):
            cosines = torch.linalg.svdvals(prev_U0.T @ U0)
            self.subspace_angle_deg = float(
                torch.rad2deg(torch.acos(cosines.min().clamp(-1, 1))))
        else:
            self.subspace_angle_deg = None
        V0 = Vh[-C:, :].T.detach()
        Msys = torch.zeros(NK + C, NK + C, device=U0.device, dtype=U0.dtype)
        Msys[:NK, :NK] = A_weighted
        Msys[:NK, NK:] = U0
        Msys[NK:, :NK] = V0.T
        self._spec_U0 = U0
        self._spec_V0 = V0
        _t_lu0 = time.perf_counter()
        self._spec_LU = torch.linalg.lu_factor(Msys)
        self.last_timings["lu"] = time.perf_counter() - _t_lu0
        # speedup Part 1 (flag-gated): W_lin is an EXACT quadratic form
        # in V within the step, so assemble its symmetric matrix ONCE
        # via a single batched solve; the inner optimization loop then
        # evaluates W = (h/2) V^T (Mq V) -- two graph nodes, closed
        # gradient Mq V and HVP Mq v through plain autograd, and NO
        # lu_solve inside the loop.
        self._spec_quad = None
        if self.analytic_gradient:
            _t_q0 = time.perf_counter()
            w_flat = self._endpoint_weights_cache
            sqrt_w = w_flat.sqrt()
            RHS = torch.zeros(NK + C, NK, device=U0.device,
                              dtype=U0.dtype)
            RHS[:NK, :] = torch.diag(sqrt_w)
            SOL = torch.linalg.lu_solve(*self._spec_LU, RHS)
            LAM = SOL[:NK] / sqrt_w[:, None]
            PHI = self._S_cache @ LAM
            A_q = w_flat[:, None] * PHI
            self._spec_quad = 0.5 * (A_q + A_q.T)
            self.last_timings["quad_assembly"] = \
                time.perf_counter() - _t_q0

    def endpoint_operator(self) -> "EndpointOperator":
        """Assembled endpoint-collocation system, for static diagnostics.

        Everything is at endpoint-collocation level (N*K), not particle level:
        the quadrature weights are zeta_{ik} = m_i / K, and the weighted
        left/right nullspaces that carry the compatibility and gauge structure
        live in this space. Diagnostics should never have to reach for the
        private caches.
        """
        assert self._K_star_cache is not None, "Call setup_for_step() first!"
        w = self._endpoint_weights_cache
        half, _ = _trace_signs(self.trace_side)
        I = torch.eye(w.numel(), device=w.device, dtype=w.dtype)
        return EndpointOperator(
            positions=self._endpoint_positions_cache,
            weights=w,
            single_layer=self._S_cache,
            adjoint_double_layer=self._K_star_cache,
            trace_operator=half * I + self._K_star_cache,
            constraint_basis=self._Q_cache,
            epsilon=self._epsilon_used,
            n_endpoints=self._K,
            n_particles=self._N,
            trace_side=self.trace_side,
        )

    def setup_snapshot_fields(self) -> Optional[dict]:
        """Read-only BEM-side telemetry of the CURRENT setup (closure
        patch 2, reviewer T). Returns None if setup_for_step has not run
        -- never constructs anything implicitly. All tensors are
        detached clones. Records what the solve ACTUALLY consumed
        (operator masses under operator_measure, endpoint weights,
        velocity scale under the flux policy) separately from any
        canonical quantity the caller may also log."""
        if self._K_star_cache is None:
            return None
        def _c(t):
            return None if t is None else \
                t.detach().clone().requires_grad_(False)
        if self.solver_mode == "spectral_bordered" \
                and self._spec_svd is not None:
            s_asc = torch.sort(self._spec_svd[1].detach()).values.clone()
            s_source = self.last_timings.get("svd_method_used",
                                             self.svd_method)
            if s_source == "lobpcg":
                # lobpcg computes only the extremal pairs; the sorted
                # vector is NOT a full spectrum
                s_source = "lobpcg_partial"
            elif s_source == "full":
                s_source = "full_svd"
        else:
            s_asc, s_source = None, "unavailable"
        if self._coherence_cache is not None:
            vel_scale = _c(self.flux_scale())
            vel_policy = ("coherence_q" if self._use_coherence_velocity
                          else "unit")
        else:
            vel_scale, vel_policy = None, "coherence_not_set"
        return dict(
            operator_weights_used=_c(self._operator_masses_cache),
            # signed physical endpoint offsets r_ik (N, K): PUBLIC so
            # audits never read the private cache (G2a-0 reviewer)
            physical_offsets=_c(self._r_physical),
            operator_measure_policy=self.operator_measure,
            endpoint_weights_used=_c(self._endpoint_weights_cache),
            endpoint_weights_policy=(
                f"{self.operator_measure}_masses/K"),
            velocity_scale_used=vel_scale,
            velocity_policy=vel_policy,
            endpoint_positions=_c(self._endpoint_positions_cache),
            endpoints_per_particle=self._K,
            epsilon_bem_used=self._epsilon_used,
            epsilon_mode=self.epsilon_mode,
            solver_mode=self.solver_mode,
            singular_values_ascending=s_asc,
            singular_values_source=s_source,
        )

    def spectral_pieces(self) -> Tuple[_T, _T, _T, int]:
        """(U0, endpoint weights, endpoint offsets, K) for building the
        spectral compatibility constraint. Only valid after setup_for_step
        in spectral_bordered mode."""
        assert self._spec_U0 is not None, \
            "spectral_bordered setup_for_step required first"
        return (self._spec_U0, self._endpoint_weights_cache,
                self._r_physical, self._K)

    def expand_to_endpoints(self, per_particle: _T) -> _T:
        """Replicate a per-particle field to the (N*K,) endpoint layout."""
        assert self._K_star_cache is not None, "Call setup_for_step() first!"
        return per_particle[:, None].expand(-1, self._K).reshape(-1)

    def setup_coherence(self, coherence: _T, use_coherence_velocity: bool = True):
        """Store frozen coherence for use during optimization.

        Args:
            coherence: Coherence values (N,) from previous step - FIXED
            use_coherence_velocity: If True, V_{ik} = (q_i/h)(...). If False, V_{ik} = (1/h)(...).
        """
        self._coherence_cache = coherence.detach()
        self._use_coherence_velocity = use_coherence_velocity

    def flux_scale(self) -> _T:
        """THE per-particle scale of the endpoint Neumann data: q if
        coherence velocity is on, else ones. Hard invariant (0C-5c): the
        spectral compatibility constraint MUST be built from this same
        tensor -- a constraint enforcing a different flux measure than the
        bordered BIE's RHS would make the solver enforce one physics and
        solve another.
        """
        assert self._coherence_cache is not None, "Call setup_coherence() first!"
        if self._use_coherence_velocity:
            return self._coherence_cache
        return torch.ones_like(self._coherence_cache)

    def __call__(
        self,
        displacements: _T,  # (N,) s_i - VARIABLE during optimization
        delta_angles: _T,   # (N,) δθ_i - VARIABLE during optimization
        time_step: float,   # h
    ) -> _T:
        """Compute linearized Wasserstein cost with endpoint collocation.

        Endpoint velocity formula (captures angle change):
            V_{ik} = (q_i/h) × (s_i - r_{ik} × δθ_i)

        where r_{ik} = r_k × m_i is the physical offset from segment center.

        W_lin = (h/2) × Σ_{i,k} w_{ik} × φ_{ik} × V_{ik}

        NOTE: Coherence q_i is applied exactly ONCE (in V_{ik}).
        The final W_lin ∝ q² is expected (energy is quadratic form).

        Args:
            displacements: Scalar displacements along normal (N,) - VARIABLE
            delta_angles: Angle changes from previous step (N,) - VARIABLE
            time_step: Time step h for MM scheme

        Returns:
            W_lin: Linearized Wasserstein cost (scalar tensor)
        """
        assert self._S_cache is not None, "Call setup_for_step() first!"
        assert self._coherence_cache is not None, "Call setup_coherence() first!"

        w_flat = self._endpoint_weights_cache  # (N*K,)
        S = self._S_cache
        Q = self._Q_cache
        h = time_step
        N, K = self._N, self._K

        # Endpoint velocities: V_{ik} = (scale_i / h) * (s_i - r_{ik} * δθ_i)
        # scale_i comes from flux_scale() -- the same tensor the spectral
        # compatibility constraint is built from (hard invariant, 0C-5c).
        # r_physical[i, k] = r_vals[k] × m_i
        scale = self.flux_scale()[:, None]
        V_ik = (scale / h) * (
            displacements[:, None] - self._r_physical * delta_angles[:, None]
        )  # (N, K)
        V_flat = V_ik.reshape(N * K)

        if self.solver_mode == "spectral_bordered":
            # NO mean subtraction: the legacy path silently projects an
            # incompatible RHS onto compatibility, returning the energy of a
            # DIFFERENT velocity. Here incompatibility is measured and, above
            # tolerance, refused.
            sqrt_w = w_flat.sqrt()
            vt = sqrt_w * V_flat
            U0 = self._spec_U0
            with torch.no_grad():
                r_comp = (U0.T @ vt).norm() / vt.norm().clamp_min(1e-300)
                self._last_r_comp = float(r_comp)
            if self._last_r_comp > self.compat_reject_tol:
                raise IncompatibleVelocityError(
                    f"componentwise Neumann compatibility violated: "
                    f"||U0^T v~||/||v~|| = {self._last_r_comp:.3f} > "
                    f"{self.compat_reject_tol}")
            C = U0.shape[1]
            if self.analytic_gradient and self._spec_quad is not None:
                return (h / 2) * (V_flat @ (self._spec_quad @ V_flat))
            rhs = torch.cat([vt, vt.new_zeros(C)])
            sol = torch.linalg.lu_solve(
                *self._spec_LU, rhs.unsqueeze(-1)).squeeze(-1)
            lam = sol[:N * K] / sqrt_w
            phi = S @ lam
            W_lin = (h / 2) * (w_flat * phi * V_flat).sum()
            return W_lin

        # Compatibility condition: remove endpoint_weights mean from V
        V_flat = V_flat - (w_flat * V_flat).sum() / w_flat.sum()

        # Solve using projection method (avoids KKT saddle-point)
        # Project RHS: b_proj = Q^T (rhs_sign * V_flat); sign matches the jump
        # relation chosen by self.trace_side (see `_trace_signs`).
        _, rhs_sign = _trace_signs(self.trace_side)
        b_proj = Q.T @ (rhs_sign * V_flat)

        # Solve projected system: (Q^T A Q) y = b_proj
        y = torch.linalg.lu_solve(*self._A_proj_LU_cache, b_proj.unsqueeze(-1)).squeeze(-1)

        # Recover λ = Q y (automatically satisfies w_flat^T λ = 0)
        lam = Q @ y
        phi = S @ lam
        phi = phi - (w_flat * phi).sum() / w_flat.sum()  # Gauge fix

        # W_lin = (h/2) × Σ w_{ik} φ_{ik} V_{ik} (use projected V_flat)
        W_lin = (h / 2) * (w_flat * phi * V_flat).sum()
        return W_lin


def compute_coherence(
    varifold,
    masses: _T,
    sigma: float,
    kernel: str = "wendland_c2",
    backend: str = "naive",
) -> _T:
    """Compute coherence q_i = |V_σ(x_i)| / U_σ(x_i) for each point.

    The coherence measures how aligned nearby normals are. It's used to
    weight the normal velocity in the BEM Wasserstein computation.

    This is computed from the varifold using the same kernel as perimeter.

    Args:
        varifold: OrientedPointCloudVarifold
        masses: Point masses (N,)
        sigma: Kernel bandwidth
        kernel: Kernel function name

    Returns:
        coherence: Coherence values (N,) in [0, 1]
    """
    from src.torch.perimeter.coherence_perimeter import (
        compute_scalar_density,
        compute_vector_field,
        compute_coherence as compute_coherence_ratio,
    )

    positions = varifold.positions
    normals = varifold.normals

    # Compute scalar density and vector field
    U = compute_scalar_density(positions, masses, sigma, kernel, backend=backend)
    V_vec = compute_vector_field(positions, normals, masses, sigma, kernel, backend=backend)

    # Coherence: q_i = |V_i| / U_i
    coherence = compute_coherence_ratio(U, V_vec)

    # Clamp to [0, 1]
    coherence = torch.clamp(coherence, 0.0, 1.0)

    return coherence
