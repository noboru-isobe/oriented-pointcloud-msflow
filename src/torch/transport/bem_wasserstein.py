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


def _build_projected_system(
    K_star: _T,
    effective_masses: _T,
    reg: float = 0.0,
) -> Tuple[_T, _T, Tuple]:
    """Build projected BEM system A_proj = Q^T A Q and its LU factorization.

    The BEM operator A = (-1/2)I + K* has a null space (constant functions).
    We project onto the constraint subspace ∫λ dm = 0.

    Args:
        K_star: Adjoint double layer matrix (N, N)
        effective_masses: masses * coherence (N,)
        reg: Optional regularization (default 0.0)

    Returns:
        (A, Q, A_proj_LU) where:
        - A: Full BEM operator (N, N)
        - Q: Orthogonal complement basis (N, N-1)
        - A_proj_LU: LU factorization of Q^T A Q
    """
    N = effective_masses.numel()
    device, dtype = effective_masses.device, effective_masses.dtype

    I = torch.eye(N, device=device, dtype=dtype)
    A = (-0.5) * I + K_star
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
) -> Tuple[_T, _T, Tuple]:
    """Solve interior Neumann problem with subspace projection.

    Uses interior jump formula: A = (-1/2)I + K*
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
        _, Q, A_proj_LU = _build_projected_system(K_star, effective_masses, reg)

    # Project RHS: b_proj = Q^T (-Vc)
    b_proj = Q.T @ (-Vc)  # (N-1,)

    # Solve projected system
    y = torch.linalg.lu_solve(*A_proj_LU, b_proj.unsqueeze(-1)).squeeze(-1)

    # Recover λ = Q y (automatically satisfies effective_masses^T λ = 0)
    lam = Q @ y

    # Compute potential
    phi = S @ lam

    # Gauge fix
    phi = phi - _mass_weighted_mean(phi, effective_masses)
    return phi, Q, A_proj_LU


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

    # Cache for projection-based solve (reused during optimization)
    _S_cache: Optional[_T] = field(default=None, repr=False)
    _endpoint_weights_cache: Optional[_T] = field(default=None, repr=False)
    _Q_cache: Optional[_T] = field(default=None, repr=False)
    _A_proj_LU_cache: Optional[Tuple] = field(default=None, repr=False)

    # Cache for endpoint velocity computation
    _N: int = field(default=0, repr=False)
    _K: int = field(default=3, repr=False)
    _r_physical: Optional[_T] = field(default=None, repr=False)  # (N, K) offset values
    _coherence_cache: Optional[_T] = field(default=None, repr=False)  # (N,) frozen coherence
    _use_coherence_velocity: bool = field(default=True, repr=False)  # q in V_{ik}?

    def setup_for_step(
        self,
        positions: _T,           # Previous step's positions (FIXED)
        normals: _T,             # Previous step's normals (FIXED)
        effective_masses: _T,    # masses * coherence (FIXED)
        sigma: Optional[float] = None,
        c_sigma: float = 3.0,
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

        # Use effective_masses as segment lengths
        masses = effective_masses

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
        if self.method == "panel":
            endpoint_tangents_flat = tangents[:, None, :].expand(N, K, 2).reshape(N * K, 2)
            S, K_star = build_bem_matrices_panel(
                endpoint_positions_flat, endpoint_tangents_flat,
                endpoint_normals_flat, endpoint_weights_flat
            )
        else:
            ell = (sigma / c_sigma) if sigma else masses.median().item()
            eps = self.epsilon_scale * float(ell)
            S, K_star = build_bem_matrices_point(
                endpoint_positions_flat, endpoint_normals_flat,
                endpoint_weights_flat, eps
            )

        self._S_cache = S
        self._endpoint_weights_cache = endpoint_weights_flat
        self._N = N
        self._K = K

        # Build projected system (use endpoint_weights, not mixing q)
        _, self._Q_cache, self._A_proj_LU_cache = _build_projected_system(
            K_star, endpoint_weights_flat, self.solver_reg
        )

    def setup_coherence(self, coherence: _T, use_coherence_velocity: bool = True):
        """Store frozen coherence for use during optimization.

        Args:
            coherence: Coherence values (N,) from previous step - FIXED
            use_coherence_velocity: If True, V_{ik} = (q_i/h)(...). If False, V_{ik} = (1/h)(...).
        """
        self._coherence_cache = coherence.detach()
        self._use_coherence_velocity = use_coherence_velocity

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
        q = self._coherence_cache  # (N,) coherence

        # Endpoint velocities: V_{ik} = (scale_i / h) * (s_i - r_{ik} * δθ_i)
        # scale_i = q_i (coherence) or 1.0 depending on _use_coherence_velocity
        # r_physical[i, k] = r_vals[k] × m_i
        scale = q[:, None] if self._use_coherence_velocity else 1.0
        V_ik = (scale / h) * (
            displacements[:, None] - self._r_physical * delta_angles[:, None]
        )  # (N, K)
        V_flat = V_ik.reshape(N * K)

        # Compatibility condition: remove endpoint_weights mean from V
        V_flat = V_flat - (w_flat * V_flat).sum() / w_flat.sum()

        # Solve using projection method (avoids KKT saddle-point)
        # Project RHS: b_proj = Q^T (-V_flat)
        b_proj = Q.T @ (-V_flat)

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
