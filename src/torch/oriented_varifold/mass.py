"""Mass computation for oriented point cloud varifolds.

Implements Buet-Leonardi-Masnou type mass computation using KDE density estimation
with smooth cutoff functions.
"""

import torch
from typing import Literal


# =============================================================================
# Kernel functions (compact support on [0, 1])
# =============================================================================

def wendland_c2(u: torch.Tensor) -> torch.Tensor:
    """
    Wendland C² kernel: η(u) = (1-u)₊⁴(4u+1)

    Smooth kernel with compact support [0, 1].
    C² continuous, ideal for AD optimization.

    Args:
        u: (N,) or (N, M) tensor of distances normalized by delta

    Returns:
        Kernel values, same shape as input
    """
    v = torch.relu(1.0 - u)  # (1-u)₊
    return (v ** 4) * (4 * u + 1)


def biweight(u: torch.Tensor) -> torch.Tensor:
    """
    Biweight kernel: η(u) = (1-u²)₊²

    Simple smooth kernel with compact support [0, 1].

    Args:
        u: (N,) or (N, M) tensor of distances normalized by delta

    Returns:
        Kernel values, same shape as input
    """
    v = torch.relu(1.0 - u * u)  # (1-u²)₊
    return v ** 2


def epanechnikov(u: torch.Tensor) -> torch.Tensor:
    """
    Epanechnikov kernel: η(u) = (1-u²)₊

    Classic kernel, but only C⁰ at boundary.

    Args:
        u: (N,) or (N, M) tensor of distances normalized by delta

    Returns:
        Kernel values, same shape as input
    """
    return torch.relu(1.0 - u * u)


def buet_rumpf_exp(u: torch.Tensor) -> torch.Tensor:
    """
    Buet-Rumpf exponential kernel: ρ(u) = exp(1/(u²-1)) for |u| < 1.

    C^∞ kernel with compact support [0, 1].
    Used in Buet, Leonardi, Masnou (2017) for regularized mean curvature.

    Args:
        u: (N,) or (N, M) tensor of distances normalized by delta

    Returns:
        Kernel values, same shape as input
    """
    _EPS = 1e-12
    den = u * u - 1.0
    den = torch.minimum(den, torch.full_like(den, -_EPS))
    exp_term = torch.exp(1.0 / den)
    mask = u.abs() < 1.0 - _EPS
    return torch.where(mask, exp_term, torch.zeros_like(u))


def _buet_rumpf_exp_grad_ratio(u: torch.Tensor) -> torch.Tensor:
    """ψ(u) = ρ'(u)/u = -2/(u²-1)² · exp(1/(u²-1)) for |u| < 1."""
    _EPS = 1e-12
    den = u * u - 1.0
    den = torch.minimum(den, torch.full_like(den, -_EPS))
    val = -2.0 / (den * den) * torch.exp(1.0 / den)
    mask = u.abs() < 1.0 - _EPS
    return torch.where(mask, val, torch.zeros_like(u))


# Normalization constants C_η for each kernel
# These ensure ∫₀¹ η(u) du = C_η
KERNEL_CONSTANTS = {
    "wendland_c2": 2.0 / 3.0,
    "biweight": 16.0 / 15.0,
    "epanechnikov": 4.0 / 3.0,
    "buet_rumpf_exp": 0.221996908084039,  # numerical integration
}

KERNEL_FUNCTIONS = {
    "wendland_c2": wendland_c2,
    "biweight": biweight,
    "epanechnikov": epanechnikov,
    "buet_rumpf_exp": buet_rumpf_exp,
}

# ψ(u) = η'(u)/u for each kernel (avoids z/|z| singularity in backward)
KERNEL_GRAD_RATIO = {
    "wendland_c2": lambda u: -20.0 * torch.relu(1.0 - u) ** 3,
    "biweight": lambda u: -4.0 * torch.relu(1.0 - u * u),
    "epanechnikov": lambda u: torch.where(
        u < 1.0, u.new_full((), -2.0), u.new_zeros(())
    ),
    "buet_rumpf_exp": _buet_rumpf_exp_grad_ratio,
}

_SAFE_SQRT_EPS = 1e-24


class _SafeKernelEval(torch.autograd.Function):
    """Evaluate η(|z|/h) with hand-computed backward that avoids z/|z| at z=0.

    Forward:  η(u) computed normally (no grad tracking inside Function).
    Backward: ∂η/∂z_k = ψ(u) · z_k / h²  where ψ(u) = η'(u)/u.
              This avoids the z/|z| singularity because η'(u)/u is finite at u=0.
    """

    @staticmethod
    def forward(ctx, z, h, kernel_name):
        eta = KERNEL_FUNCTIONS[kernel_name]
        r = z.norm(dim=-1)
        u = r / h
        result = eta(u)
        ctx.save_for_backward(z)
        ctx.h = h
        ctx.kernel_name = kernel_name
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        h = ctx.h
        r_sq = (z * z).sum(dim=-1)
        u = (r_sq + _SAFE_SQRT_EPS).sqrt() / h
        psi = KERNEL_GRAD_RATIO[ctx.kernel_name](u)
        # ∂η/∂z_k = ψ(u) · z_k / h²
        grad_z = (grad_output * psi / (h * h)).unsqueeze(-1) * z
        return grad_z, None, None


class _SafeKernelEvalAdaptive(torch.autograd.Function):
    """Per-point bandwidth version of _SafeKernelEval.

    h is (N,) tensor: row i of the (N, N) distance matrix uses bandwidth h_i.

    Forward:  u_{ij} = |z_{ij}| / h_i, result = η(u_{ij})
    Backward: ∂η/∂z_k = ψ(u) · z_k / h_i²
    """

    @staticmethod
    def forward(ctx, z, h, kernel_name):
        # z: (N, N, 2), h: (N,)
        eta = KERNEL_FUNCTIONS[kernel_name]
        r = z.norm(dim=-1)          # (N, N)
        u = r / h[:, None]          # (N, N) — row i uses h_i
        result = eta(u)
        ctx.save_for_backward(z, h)
        ctx.kernel_name = kernel_name
        return result

    @staticmethod
    def backward(ctx, grad_output):
        z, h = ctx.saved_tensors
        r_sq = (z * z).sum(dim=-1)
        u = (r_sq + _SAFE_SQRT_EPS).sqrt() / h[:, None]
        psi = KERNEL_GRAD_RATIO[ctx.kernel_name](u)
        h_sq = (h * h)[:, None]     # (N, 1)
        grad_z = (grad_output * psi / h_sq).unsqueeze(-1) * z
        return grad_z, None, None


# =============================================================================
# Cutoff function χ_τ
# =============================================================================

def chi_tau(t: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
    """
    Smooth cutoff function χ_τ(t).

    χ_τ(t) = ReLU(2t/τ - 1) - ReLU(2t/τ - 2)

    This function:
    - Is 0 for t < τ/2
    - Linearly increases from 0 to 1 for τ/2 < t < τ
    - Is 1 for t > τ

    Used to filter out points with low density (likely outliers or noise).

    Args:
        t: (N,) tensor of density values
        tau: Cutoff threshold — scalar or (N,) per-point tensor

    Returns:
        (N,) tensor of cutoff values in [0, 1]
    """
    if isinstance(tau, (int, float)) and tau == 0:
        return torch.ones_like(t)
    scaled = 2.0 * t / tau
    return torch.relu(scaled - 1.0) - torch.relu(scaled - 2.0)


# =============================================================================
# Parameter selection
# =============================================================================

def compute_knn_distances(points: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute k-th nearest neighbor distance for each point.

    Args:
        points: (N, 2) tensor of point positions
        k: Which nearest neighbor (1 = nearest, excluding self)

    Returns:
        (N,) tensor of k-th nearest neighbor distances
    """
    # Pairwise distances
    diff = points.unsqueeze(1) - points.unsqueeze(0)  # (N, N, 2)
    dist = diff.norm(dim=-1)  # (N, N)
    # Sort and get k-th nearest (index 0 is self with distance 0)
    sorted_dist, _ = dist.sort(dim=1)
    return sorted_dist[:, k]


def compute_recommended_params(
    points: torch.Tensor,
    k0: int = 10,
    k_min: float = 1.0,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
) -> tuple[float, float]:
    """
    Compute recommended delta and tau for mass computation.

    Uses kNN-based method for delta and minimum effective neighbor count for tau.

    Method:
        δ = median_i r_i^(k0)     (k0-th nearest neighbor distance)
        τ = 2 * k_min / (N * C_η * δ)

    Recommended parameters:
        - k0=10: Works well for both simple and multi-component shapes
        - k_min=1.0: Minimal filtering, good for clean point clouds
        - For noisy data, increase k_min to 2-3

    Args:
        points: (N, 2) tensor of point positions
        k0: Which nearest neighbor to use for delta (default: 10)
        k_min: Minimum effective neighbor count for tau (default: 1.0)
        kernel: Kernel function to use

    Returns:
        (delta, tau) tuple of recommended parameters
    """
    N = points.shape[0]
    C_eta = KERNEL_CONSTANTS[kernel]

    # Compute delta from k0-th nearest neighbor distances
    r_k0 = compute_knn_distances(points, k0)
    delta = r_k0.median().item()

    # Compute tau from minimum effective neighbor count
    tau = 2 * k_min / (N * C_eta * delta)

    return delta, tau


def compute_recommended_params_oriented(
    points: torch.Tensor,
    normals: torch.Tensor,
    k0: int = 10,
    k_min: float = 1.0,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
) -> tuple[float, float]:
    """The SAME (delta, tau) recommendation rule as
    compute_recommended_params, applied to the geometry the oriented
    estimator actually reads: delta = median of the k0-th
    CO-ORIENTED (n_i . n_j > 0) neighbor distance, tau = 2 k_min /
    (N C_eta delta). On a clean boundary co-oriented kNN equals
    ordinary kNN, so this reproduces the position-only rule; on an
    antiparallel wall it reads the own-sheet scale instead of the
    doubled merged density (whose k0-NN distance under-sizes delta
    and makes chi_tau fire on the sparse sheet -- the IIIc-0 cutoff
    audit finding)."""
    N = points.shape[0]
    C_eta = KERNEL_CONSTANTS[kernel]
    d = torch.cdist(points, points)
    co = (normals @ normals.T) > 0
    d = torch.where(co, d, torch.full_like(d, float("inf")))
    d.fill_diagonal_(float("inf"))
    k = min(k0, N - 1)
    r_k0 = d.topk(k, dim=1, largest=False).values[:, -1]
    finite = torch.isfinite(r_k0)
    delta = float(r_k0[finite].median())
    tau = 2 * k_min / (N * C_eta * delta)
    return delta, tau


def compute_per_point_bandwidths_knn(
    points: torch.Tensor,
    k: int = 10,
    k_min: float = 1.0,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-point adaptive bandwidths using k-th nearest neighbor distance.

    δ_i = r_i^(k)   (k-th nearest neighbor distance for point i)
    τ_i = 2 k_min / (N C_η δ_i)

    Args:
        points: (N, 2) tensor of point positions
        k: Which nearest neighbor (default: 10)
        k_min: Minimum effective neighbor count for τ
        kernel: Kernel function to use

    Returns:
        (delta_pp, tau_pp) — both (N,) tensors
    """
    N = points.shape[0]
    C_eta = KERNEL_CONSTANTS[kernel]

    r_k = compute_knn_distances(points, k)  # (N,)
    delta_pp = r_k
    tau_pp = 2 * k_min / (N * C_eta * delta_pp)

    return delta_pp, tau_pp


def compute_per_point_bandwidths_abramson(
    points: torch.Tensor,
    k0: int = 10,
    k_min: float = 1.0,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-point adaptive bandwidths using Abramson's square root law.

    Step 1: Compute pilot density f̃ using global δ₀ (from compute_recommended_params).
    Step 2: δ_i = δ₀ × f̃(x_i)^{-1/2} / γ,  where γ = geometric mean of f̃^{-1/2}.

    Reference: Abramson (1982) "On Bandwidth Variation in Kernel Estimates—A Square Root Law"

    Args:
        points: (N, 2) tensor of point positions
        k0: Which nearest neighbor for pilot δ₀ (default: 10)
        k_min: Minimum effective neighbor count for τ
        kernel: Kernel function to use

    Returns:
        (delta_pp, tau_pp) — both (N,) tensors
    """
    N = points.shape[0]
    C_eta = KERNEL_CONSTANTS[kernel]

    # Step 1: pilot — global δ₀ and pilot density
    delta0, _ = compute_recommended_params(points, k0=k0, kernel=kernel, k_min=k_min)
    pilot_density = compute_kde_density(points, delta0, kernel)  # (N,)

    # Step 2: per-point bandwidth via Abramson square root law
    # δ_i = δ₀ × f̃(x_i)^{-1/2} / γ
    # γ = exp(mean(log(f̃^{-1/2}))) = geometric mean of f̃^{-1/2}
    inv_sqrt_f = pilot_density.clamp(min=1e-30).pow(-0.5)
    gamma = inv_sqrt_f.log().mean().exp()  # geometric mean
    delta_pp = delta0 * inv_sqrt_f / gamma

    tau_pp = 2 * k_min / (N * C_eta * delta_pp)

    return delta_pp, tau_pp


# =============================================================================
# KDE density estimation
# =============================================================================

def compute_kde_density(
    points: torch.Tensor,
    delta: float,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
    backend: Literal["naive", "keops"] = "naive",
) -> torch.Tensor:
    """
    Compute KDE density at each point.

    θ_{δ,N}(x_i) = (1 / N C_η δ) Σ_j η(|x_i - x_j| / δ)

    Args:
        points: (N, 2) tensor of point positions
        delta: Bandwidth parameter for KDE
        kernel: Kernel function to use
        backend: ``"naive"`` (default; full N×N pairwise via
            ``_SafeKernelEval`` — preserves the existing custom backward
            and the gradients tests rely on) or ``"keops"`` (lazy
            pairwise via ``pykeops``; O(N) memory, one JIT compile
            per kernel formula per session). See
            ``src/torch/math_utils/keops_pairwise.py``.

    Returns:
        (N,) tensor of density values at each point
    """
    if backend == "keops":
        from ..math_utils.keops_pairwise import kde_density_keops
        return kde_density_keops(points, delta, kernel)

    N = points.shape[0]
    C_eta = KERNEL_CONSTANTS[kernel]

    diff = points.unsqueeze(1) - points.unsqueeze(0)  # (N, N, 2)
    kernel_vals = _SafeKernelEval.apply(diff, delta, kernel)  # (N, N)

    # Sum over all points and normalize
    density = kernel_vals.sum(dim=1) / (N * C_eta * delta)  # (N,)

    return density


def compute_kde_density_adaptive(
    points: torch.Tensor,
    delta_per_point: torch.Tensor,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
) -> torch.Tensor:
    """
    Compute KDE density with per-point adaptive bandwidth (balloon estimator).

    θ_i = (1 / N C_η δ_i) Σ_j η(|x_i - x_j| / δ_i)

    Each point i uses its own bandwidth δ_i for the kernel evaluation.

    Args:
        points: (N, 2) tensor of point positions
        delta_per_point: (N,) tensor of per-point bandwidths
        kernel: Kernel function to use

    Returns:
        (N,) tensor of density values at each point
    """
    N = points.shape[0]
    C_eta = KERNEL_CONSTANTS[kernel]

    diff = points.unsqueeze(1) - points.unsqueeze(0)  # (N, N, 2)
    kernel_vals = _SafeKernelEvalAdaptive.apply(diff, delta_per_point, kernel)  # (N, N)

    # Per-point normalization: θ_i = Σ_j η(...) / (N C_η δ_i)
    density = kernel_vals.sum(dim=1) / (N * C_eta * delta_per_point)  # (N,)

    return density


# =============================================================================
# Mass computation
# =============================================================================

def compute_masses(
    points: torch.Tensor,
    delta: float | torch.Tensor,
    tau: float | torch.Tensor,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
) -> torch.Tensor:
    """
    Compute masses for each point using Buet-Leonardi-Masnou formula.

    m_i = (1/N) χ_τ(θ_i) / θ_i

    where θ_i is the KDE density at point i.

    The χ_τ cutoff filters out low-density points (potential outliers),
    and dividing by θ_i normalizes by local density so that points in
    sparse regions get higher mass.

    Args:
        points: (N, 2) tensor of point positions
        delta: Bandwidth — scalar (global) or (N,) tensor (per-point adaptive)
        tau: Cutoff threshold — scalar (global) or (N,) tensor (per-point)
        kernel: Kernel function to use

    Returns:
        (N,) tensor of masses for each point
    """
    N = points.shape[0]

    # Compute KDE density (dispatch on delta type)
    if isinstance(delta, torch.Tensor) and delta.dim() > 0:
        theta = compute_kde_density_adaptive(points, delta, kernel)
    else:
        theta = compute_kde_density(points, delta, kernel)

    # Apply cutoff and compute mass
    # m_i = (1/N) χ_τ(θ_i) / θ_i
    # When θ → 0, χ_τ(θ) → 0, and we define χ_τ(θ)/θ → 0 (isolated points don't contribute)
    # Note: chi > 0 implies θ ≥ τ/2 > 0, so division is safe when chi > 0
    chi = chi_tau(theta, tau)  # (N,)
    masses = torch.where(chi > 0, chi / theta, torch.zeros_like(theta)) / N  # (N,)

    return masses


def compute_kde_density_oriented(
    points: torch.Tensor,
    normals: torch.Tensor,
    delta: float,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
) -> torch.Tensor:
    """Antipodal-sheet-aware KDE density (Phase IIIc-0).

    theta^or_i = (1/(N C_eta delta)) sum_j eta(|x_i-x_j|/delta)
                 * omega(n_i . n_j),   omega(s) = (1+s)/2.

    SCOPE (stated, not incidental): this separates ANTIPARALLEL
    coincident sheets -- the topology-change wall whose position-only
    KDE left a count-ratio residual |N- - N+|/(N- + N+) in the
    coherence (measured 0.225 vs the analytic 51/231 = 0.2208). It is
    NOT a general oriented-multiplicity estimator: co-oriented
    coincident sheets (omega(1) = 1) are out of scope. On a smooth
    single sheet omega = 1 - O(kappa^2 delta^2), so the difference
    from the position-only estimator is higher order. omega is
    differentiable, threshold-free, and uses no labels or history.

    Global scalar bandwidth only: per-point / adaptive modes are
    rejected explicitly (no silent fallback).
    """
    N = points.shape[0]
    C_eta = KERNEL_CONSTANTS[kernel]
    kw = _oriented_kernel_omega(points, normals, delta, kernel)
    return kw.sum(dim=1) / (N * C_eta * delta)


def compute_masses_oriented(
    points: torch.Tensor,
    normals: torch.Tensor,
    delta: float,
    tau: float | torch.Tensor,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
) -> torch.Tensor:
    """Buet-Leonardi-Masnou masses with the antipodal-sheet-aware
    density: m_i = (1/N) chi_tau(theta^or_i) / theta^or_i. See
    compute_kde_density_oriented for scope and restrictions."""
    N = points.shape[0]
    theta = compute_kde_density_oriented(points, normals, delta,
                                         kernel)
    chi = chi_tau(theta, tau)
    return torch.where(chi > 0, chi / theta,
                       torch.zeros_like(theta)) / N


def _oriented_kernel_omega(
    points: torch.Tensor,
    normals: torch.Tensor,
    delta: float,
    kernel: str,
) -> torch.Tensor:
    """(N, N) matrix eta(|x_i-x_j|/delta) * omega(n_i . n_j) shared by
    the oriented density, its loopwise-masked variant, and the
    self/cross split -- one implementation so the identity
    theta^or = theta^self + theta^cross holds by construction."""
    if isinstance(delta, torch.Tensor) and delta.dim() > 0:
        raise NotImplementedError(
            "oriented KDE supports a global scalar bandwidth only; "
            "per-point/adaptive bandwidth paths must not silently "
            "fall back")
    diff = points.unsqueeze(1) - points.unsqueeze(0)
    kernel_vals = _SafeKernelEval.apply(diff, delta, kernel)
    omega = 0.5 * (1.0 + normals @ normals.T)
    return kernel_vals * omega


def compute_kde_density_oriented_loopwise(
    points: torch.Tensor,
    normals: torch.Tensor,
    delta: float,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
    loop_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """0K: loopwise-masked oriented density.

    theta^loop_i = (1/(N C_eta delta)) sum_{j: l_j = l_i}
                   eta(|x_i-x_j|/delta) omega(n_i . n_j)

    with the GLOBAL N normalization (not N_l): under the standard
    cutoff rule tau = 2 k_min / (N C_eta delta) the global N cancels
    exactly in both chi_tau firing and the mass 1/(N theta), so each
    loop's mass is invariant to the cardinality of the other loops
    (reviewer 0K section 1). loop_labels are the SOURCE-frozen
    certified boundary-loop labels; they are consumed as constants
    (no gradient path through the mask), so within one MM step the
    objective stays smooth. Scope: distinct certified loops --
    same-loop self-contact is NOT separated by this mask (future
    contact-complex work)."""
    if loop_labels is None:
        raise ValueError(
            "loopwise oriented KDE requires explicit loop_labels "
            "(source-frozen certified partition) -- no silent fallback")
    N = points.shape[0]
    C_eta = KERNEL_CONSTANTS[kernel]
    kw = _oriented_kernel_omega(points, normals, delta, kernel)
    same = (loop_labels.unsqueeze(1)
            == loop_labels.unsqueeze(0)).to(kw.dtype)
    return (kw * same).sum(dim=1) / (N * C_eta * delta)


def compute_masses_oriented_loopwise(
    points: torch.Tensor,
    normals: torch.Tensor,
    loop_labels: torch.Tensor,
    delta: float,
    tau: float | torch.Tensor,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
) -> torch.Tensor:
    """0K masses: m_i = (1/N) chi_tau(theta^loop_i)/theta^loop_i.

    Guarantee (exact wording, reviewer 0K section 9): cross-loop
    independence + sheetwise H^1 consistency -- each loop's raw mass
    approximates ITS OWN arclength and is never attenuated by an
    approaching antiparallel loop. sum_i m_i is NOT a conserved
    quantity (loop perimeters move under the flow)."""
    N = points.shape[0]
    theta = compute_kde_density_oriented_loopwise(
        points, normals, delta, kernel, loop_labels=loop_labels)
    chi = chi_tau(theta, tau)
    return torch.where(chi > 0, chi / theta,
                       torch.zeros_like(theta)) / N


def oriented_density_cross_split(
    points: torch.Tensor,
    normals: torch.Tensor,
    delta: float,
    kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2",
    loop_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(theta^self, theta^cross) from ONE kernel-omega matrix, so
    theta^or = theta^self + theta^cross holds to rounding. Telemetry /
    0K-B decomposition helper (eq 4.1); callers pass detached inputs."""
    if loop_labels is None:
        raise ValueError("cross split requires loop_labels")
    N = points.shape[0]
    C_eta = KERNEL_CONSTANTS[kernel]
    kw = _oriented_kernel_omega(points, normals, delta, kernel)
    same = (loop_labels.unsqueeze(1)
            == loop_labels.unsqueeze(0)).to(kw.dtype)
    norm = N * C_eta * delta
    theta_self = (kw * same).sum(dim=1) / norm
    theta_cross = (kw * (1.0 - same)).sum(dim=1) / norm
    return theta_self, theta_cross


def compute_masses_uniform(
    n_points: int,
    total_mass: float,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute uniform masses (simple baseline).

    Each point gets equal mass: m_i = total_mass / N

    Args:
        n_points: Number of points
        total_mass: Total mass (e.g., perimeter of curve)
        device: Torch device
        dtype: Torch dtype

    Returns:
        (N,) tensor of uniform masses
    """
    return torch.full(
        (n_points,),
        total_mass / n_points,
        device=device,
        dtype=dtype,
    )
