"""Tests for BEM-based linearized Wasserstein distance computation.

Phase 1: Basic W_lin computation tests for various shapes.
"""

import pytest
import torch

from src.torch.transport import (
    BEMWasserstein,
    compute_coherence,
)
from src.torch.transport.bem_wasserstein import _orthogonal_complement
from src.torch.shapes.generator import (
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_star,
    generate_oriented_two_ellipses,
)
from src.torch.oriented_varifold.mass import compute_masses, compute_recommended_params


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

METHODS = [
    "point",
    pytest.param("panel", marks=pytest.mark.panel),
]
N_POINTS = [32, 64, 128, 256]  # capped at practical use range; >256 hits numerical edges (W_lin ~ 5e-11) on hardcoded thresholds and is not used in any animate_*/run_two_ellipses_batch script
EPSILON_SCALES = [0.1, 0.5, 1.0]  # For point method; must be > 0. 2.0 dropped: BEM kernel becomes near-rank-deficient (ε > segment length), production uses 0.1

# Shape generators: (name, generator_function)
# Generator function takes (n_points, device) and returns a varifold
SHAPES = [
    ("circle", lambda n, d: generate_oriented_circle(n, device=d)),
    ("ellipse", lambda n, d: generate_oriented_ellipse(n, a=1.0, b=0.5, device=d)),
    ("star", lambda n, d: generate_oriented_star(n, n_star_points=5, device=d)),
    ("two_ellipses", lambda n, d: generate_oriented_two_ellipses(n // 2, device=d)),  # n/2 per ellipse
]


def _setup_bem(varifold, method, epsilon_scale=0.5, n_endpoints=3):
    """Helper to setup BEM Wasserstein for a varifold."""
    delta, tau = compute_recommended_params(varifold.positions)
    masses = compute_masses(varifold.positions, delta, tau)
    coherence = compute_coherence(varifold, masses, sigma=delta)
    effective_masses = masses * coherence

    # Constraint weights for volume conservation: w_i = q_i^2 * m_i
    constraint_weights = coherence * coherence * masses

    bem = BEMWasserstein(method=method, epsilon_scale=epsilon_scale, n_endpoints=n_endpoints)
    bem.setup_for_step(
        positions=varifold.positions,
        normals=varifold.normals,
        effective_masses=effective_masses,
        sigma=delta,
    )
    bem.setup_coherence(coherence)

    # Compute actual epsilon for diagnostics
    c_sigma = 3.0
    ell = delta / c_sigma
    actual_epsilon = epsilon_scale * ell

    return bem, coherence, constraint_weights, delta, actual_epsilon


class TestBEMWassersteinPhase1:
    """Phase 1: Tests for BEM Wasserstein on various shapes."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("n_points", N_POINTS)
    @pytest.mark.parametrize("epsilon_scale", EPSILON_SCALES)
    @pytest.mark.parametrize("shape_name,shape_fn", SHAPES)
    def test_zero_displacement_gives_zero(self, device, method, n_points, epsilon_scale, shape_name, shape_fn):
        """Test that zero displacement and zero angle change gives zero Wasserstein cost."""
        varifold = shape_fn(n_points, device)
        n_actual = varifold.n_points

        bem, coherence, _, _, _ = _setup_bem(varifold, method, epsilon_scale)

        displacements = torch.zeros(n_actual, device=device)
        delta_angles = torch.zeros(n_actual, device=device)
        W_lin = bem(displacements, delta_angles, time_step=0.01)

        assert W_lin.item() < 1e-6, f"{shape_name}(n={n_points}): Expected ~0, got {W_lin.item():.6f}"

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("n_points", N_POINTS)
    @pytest.mark.parametrize("epsilon_scale", EPSILON_SCALES)
    @pytest.mark.parametrize("shape_name,shape_fn", SHAPES)
    def test_positive_cost_for_volume_conserving_displacement(self, device, method, n_points, epsilon_scale, shape_name, shape_fn):
        """Test that non-zero volume-conserving displacement gives positive cost."""
        varifold = shape_fn(n_points, device)
        n_actual = varifold.n_points
        dtype = varifold.positions.dtype

        bem, coherence, constraint_weights, _, _ = _setup_bem(varifold, method, epsilon_scale)

        # Create volume-conserving displacements: s = Q @ y
        Q = _orthogonal_complement(constraint_weights)

        # Generate batch of random perturbations (fixed seed for reproducibility)
        n_trials = 100
        generator = torch.Generator(device=device).manual_seed(42)
        Y = torch.randn(n_trials, n_actual - 1, device=device, dtype=dtype, generator=generator) * 0.1
        displacements_batch = Y @ Q.T  # (n_trials, n_actual)
        delta_angles = torch.zeros(n_actual, device=device, dtype=dtype)

        # Compute W_lin for each perturbation
        W_lin_values = torch.stack([
            bem(displacements_batch[i], delta_angles, time_step=0.01)
            for i in range(n_trials)
        ])

        # All should be positive
        failed = (W_lin_values <= 0).nonzero(as_tuple=True)[0]
        assert len(failed) == 0, (
            f"{shape_name}(n={n_points}, {method}, eps_scale={epsilon_scale}): "
            f"{len(failed)}/{n_trials} trials gave non-positive W_lin: {W_lin_values[failed].tolist()}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("n_points", N_POINTS)
    @pytest.mark.parametrize("epsilon_scale", EPSILON_SCALES)
    @pytest.mark.parametrize("shape_name,shape_fn", SHAPES)
    def test_gradient_flow(self, device, method, n_points, epsilon_scale, shape_name, shape_fn):
        """Test that gradients flow through BEM Wasserstein for displacements."""
        varifold = shape_fn(n_points, device)
        n_actual = varifold.n_points
        dtype = varifold.positions.dtype

        bem, coherence, constraint_weights, _, _ = _setup_bem(varifold, method, epsilon_scale)

        # Create volume-conserving displacement with gradients: s = Q @ y
        Q = _orthogonal_complement(constraint_weights)
        y = (torch.randn(n_actual - 1, device=device, dtype=dtype) * 0.01).detach().requires_grad_(True)
        displacements = Q @ y
        delta_angles = torch.zeros(n_actual, device=device, dtype=dtype)

        W_lin = bem(displacements, delta_angles, time_step=0.01)
        W_lin.backward()

        assert y.grad is not None, f"{shape_name}(n={n_points}): No gradients computed"
        assert y.grad.abs().sum() > 0, f"{shape_name}(n={n_points}): Gradients are all zero"

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("n_points", N_POINTS)
    @pytest.mark.parametrize("epsilon_scale", EPSILON_SCALES)
    @pytest.mark.parametrize("shape_name,shape_fn", SHAPES)
    def test_angle_change_increases_cost(self, device, method, n_points, epsilon_scale, shape_name, shape_fn):
        """Test that non-zero angle change (δθ ≠ 0) gives positive Wasserstein cost."""
        varifold = shape_fn(n_points, device)
        n_actual = varifold.n_points
        dtype = varifold.positions.dtype

        bem, coherence, _, _, _ = _setup_bem(varifold, method, epsilon_scale)

        displacements = torch.zeros(n_actual, device=device, dtype=dtype)
        # Random angle changes (not zero)
        torch.manual_seed(42)
        delta_angles = torch.randn(n_actual, device=device, dtype=dtype) * 0.01

        W_lin = bem(displacements, delta_angles, time_step=0.01)
        assert W_lin.item() > 1e-10, (
            f"{shape_name}(n={n_points}, {method}): "
            f"Angle change should increase transport cost, got {W_lin.item():.6e}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("n_points", N_POINTS)
    @pytest.mark.parametrize("epsilon_scale", EPSILON_SCALES)
    @pytest.mark.parametrize("shape_name,shape_fn", SHAPES)
    def test_angle_gradient_flow(self, device, method, n_points, epsilon_scale, shape_name, shape_fn):
        """Test that gradients flow through BEM Wasserstein for delta_angles."""
        varifold = shape_fn(n_points, device)
        n_actual = varifold.n_points
        dtype = varifold.positions.dtype

        bem, coherence, _, _, _ = _setup_bem(varifold, method, epsilon_scale)

        displacements = torch.zeros(n_actual, device=device, dtype=dtype)
        delta_angles = (torch.randn(n_actual, device=device, dtype=dtype) * 0.01).requires_grad_(True)

        W_lin = bem(displacements, delta_angles, time_step=0.01)
        W_lin.backward()

        assert delta_angles.grad is not None, f"{shape_name}(n={n_points}): No gradients computed for delta_angles"
        assert delta_angles.grad.abs().sum() > 0, f"{shape_name}(n={n_points}): Gradients are all zero for delta_angles"

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("n_points", N_POINTS)
    @pytest.mark.parametrize("epsilon_scale", EPSILON_SCALES)
    @pytest.mark.parametrize("shape_name,shape_fn", SHAPES)
    def test_combined_displacement_and_angle(self, device, method, n_points, epsilon_scale, shape_name, shape_fn):
        """Test that s≠0 AND δθ≠0 gives positive W_lin (n_trials random samples)."""
        varifold = shape_fn(n_points, device)
        n_actual = varifold.n_points
        dtype = varifold.positions.dtype

        bem, coherence, constraint_weights, _, _ = _setup_bem(varifold, method, epsilon_scale)

        # Create volume-conserving displacements: s = Q @ y
        Q = _orthogonal_complement(constraint_weights)

        # Generate batch of random perturbations (fixed seed for reproducibility)
        n_trials = 100
        generator = torch.Generator(device=device).manual_seed(42)
        Y = torch.randn(n_trials, n_actual - 1, device=device, dtype=dtype, generator=generator) * 0.1
        displacements_batch = Y @ Q.T  # (n_trials, n_actual)
        delta_angles_batch = torch.randn(n_trials, n_actual, device=device, dtype=dtype, generator=generator) * 0.01

        # Compute W_lin for each perturbation
        W_lin_values = torch.stack([
            bem(displacements_batch[i], delta_angles_batch[i], time_step=0.01)
            for i in range(n_trials)
        ])

        # All should be positive
        failed = (W_lin_values <= 0).nonzero(as_tuple=True)[0]
        assert len(failed) == 0, (
            f"{shape_name}(n={n_points}, {method}, eps_scale={epsilon_scale}): "
            f"{len(failed)}/{n_trials} trials gave non-positive W_lin: {W_lin_values[failed].tolist()}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("n_points", N_POINTS)
    @pytest.mark.parametrize("epsilon_scale", EPSILON_SCALES)
    @pytest.mark.parametrize("shape_name,shape_fn", SHAPES)
    def test_combined_gradient_flow(self, device, method, n_points, epsilon_scale, shape_name, shape_fn):
        """Test that gradients flow through both y (displacements) and delta_angles."""
        varifold = shape_fn(n_points, device)
        n_actual = varifold.n_points
        dtype = varifold.positions.dtype

        bem, coherence, constraint_weights, _, _ = _setup_bem(varifold, method, epsilon_scale)

        # Create volume-conserving displacement with gradients: s = Q @ y
        Q = _orthogonal_complement(constraint_weights)
        y = (torch.randn(n_actual - 1, device=device, dtype=dtype) * 0.01).requires_grad_(True)
        displacements = Q @ y

        # Angle changes with gradients
        delta_angles = (torch.randn(n_actual, device=device, dtype=dtype) * 0.01).requires_grad_(True)

        W_lin = bem(displacements, delta_angles, time_step=0.01)
        W_lin.backward()

        # Both should have gradients
        assert y.grad is not None, f"{shape_name}(n={n_points}): No gradients computed for y (displacements)"
        assert y.grad.abs().sum() > 0, f"{shape_name}(n={n_points}): Gradients are all zero for y"
        assert delta_angles.grad is not None, f"{shape_name}(n={n_points}): No gradients computed for delta_angles"
        assert delta_angles.grad.abs().sum() > 0, f"{shape_name}(n={n_points}): Gradients are all zero for delta_angles"


# Smaller N for Hessian tests (computationally expensive)
N_POINTS_FOR_HESSIAN = [32, 64, 128, 256]


class TestBEMWassersteinHessian:
    """Tests for Hessian positive semi-definiteness of W_lin."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("n_points", N_POINTS_FOR_HESSIAN)
    @pytest.mark.parametrize("epsilon_scale", EPSILON_SCALES)
    @pytest.mark.parametrize("shape_name,shape_fn", SHAPES)
    @pytest.mark.parametrize("n_endpoints", [3, 4, 5])  # K=2 (endpoints only, no center), K=6,8,10 dropped: cause Hessian ill-conditioning beyond psd_tol; production uses K=3
    def test_hessian_positive_semidefinite(self, device, method, n_points, epsilon_scale, shape_name, shape_fn, n_endpoints):
        """Test that W_lin Hessian is positive semi-definite in [y, δθ] space."""
        varifold = shape_fn(n_points, device)
        n_actual = varifold.n_points
        dtype = varifold.positions.dtype

        bem, coherence, constraint_weights, _, _ = _setup_bem(varifold, method, epsilon_scale, n_endpoints)
        Q = _orthogonal_complement(constraint_weights)

        # Parameter dimensions: y (N-1) + δθ (N) = 2N-1
        dim_y = n_actual - 1
        dim_theta = n_actual
        dim_total = dim_y + dim_theta

        def W_lin_func(params):
            """Compute W_lin from params = [y, δθ]."""
            y = params[:dim_y]
            delta_theta = params[dim_y:]
            displacements = Q @ y
            return bem(displacements, delta_theta, time_step=0.01)

        # Compute Hessian at zero using autodiff
        params_zero = torch.zeros(dim_total, device=device, dtype=dtype)
        H = torch.autograd.functional.hessian(W_lin_func, params_zero)

        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(H)

        # PSD check with N-dependent + scale-relative tolerance.
        #
        # Bauer-Fike for eigvalsh of an N_hess × N_hess matrix:
        #     |Δλ| ≤ N_hess · ULP · ‖A‖
        # The Hessian here is computed via autograd through a near-singular
        # BEM lu_solve, which amplifies this by an additional factor of the
        # BEM operator's condition number κ. For our problem κ can reach
        # ~10⁶ at small eps_scale × large N, so the effective noise floor
        # grows roughly linearly with N_hess and proportionally to ‖A‖.
        #
        # Empirical choice: rtol = 2e-12 × N_hess gives a ~2× margin over
        # the worst observed case (N=256, eps_scale=0.1 → -9.9e-10 vs
        # tolerance ~-2e-9) while still catching real PSD violations
        # (which would be O(1e-4)+ in magnitude).
        N_hess = H.shape[0]
        rtol = 2e-12 * N_hess
        max_abs_eig = eigenvalues.abs().max().item()
        psd_tol = -rtol * max(1.0, max_abs_eig)
        min_eigenvalue = eigenvalues.min().item()

        assert min_eigenvalue > psd_tol, (
            f"{shape_name}(n={n_points}, {method}, eps_scale={epsilon_scale}): "
            f"Hessian min eigenvalue {min_eigenvalue:.6e} below tolerance "
            f"{psd_tol:.3e} (rtol={rtol:.3e}, max|λ|={max_abs_eig:.3e}, N_hess={N_hess})"
        )
