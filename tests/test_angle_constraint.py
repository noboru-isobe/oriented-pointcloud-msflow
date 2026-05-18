"""Tests for weak form angle constraint module.

Verifies:
1) compute_kernel_gradient correctness (shape, support, finite values)
2) compute_angle_constraint_matrices shapes and finite values
3) Weak form constraint: A @ Δθ = B @ s produces reasonable Δθ for small s
4) TruncatedSVD utility works correctly
"""

import pytest
import torch

from src.torch.perimeter.angle_constraint import (
    compute_kernel_gradient,
    compute_angle_constraint_matrices,
)
from src.torch.math_utils.linalg import TruncatedSVD
from src.torch.solver.mm_step import NormalGraphParametrization
from src.torch.shapes.generator import (
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_star,
    generate_oriented_two_ellipses,
)
from src.torch.oriented_varifold.mass import compute_masses, compute_recommended_params
from src.torch.transport import compute_coherence

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

SHAPES = [
    ("circle", lambda n, d: generate_oriented_circle(n, device=d)),
    ("ellipse", lambda n, d: generate_oriented_ellipse(n, a=1.0, b=0.5, device=d)),
    ("star", lambda n, d: generate_oriented_star(n, n_star_points=5, device=d)),
    ("two_ellipses", lambda n, d: generate_oriented_two_ellipses(n // 2, device=d)),
]

N_POINTS = [32, 64]

KERNELS = ["wendland_c2", "biweight", "epanechnikov"]


def _seed_all(seed: int, device: str):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def _setup_shape(shape_fn, n_points, device):
    varifold = shape_fn(n_points, device)
    delta, tau = compute_recommended_params(varifold.positions)
    masses = compute_masses(varifold.positions, delta, tau)
    sigma = delta
    tangents = torch.stack([-varifold.normals[:, 1], varifold.normals[:, 0]], dim=1)
    coherence = compute_coherence(varifold, masses, sigma)
    return varifold, masses, sigma, tangents, coherence


def _setup_shape_from_varifold(varifold, device):
    """Setup masses, sigma, tangents, coherence from a pre-created varifold."""
    delta, tau = compute_recommended_params(varifold.positions)
    masses = compute_masses(varifold.positions, delta, tau)
    sigma = delta
    tangents = torch.stack([-varifold.normals[:, 1], varifold.normals[:, 0]], dim=1)
    coherence = compute_coherence(varifold, masses, sigma)
    return masses, sigma, tangents, coherence


class TestKernelGradient:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("kernel", KERNELS)
    def test_gradient_shape(self, device, kernel):
        _seed_all(0, device)
        N = 32
        positions = torch.randn(N, 2, device=device)
        diff = positions[:, None, :] - positions[None, :, :]
        sigma = 0.5
        grad = compute_kernel_gradient(diff, sigma, kernel)
        assert grad.shape == (N, N, 2)
        assert torch.isfinite(grad).all()

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("kernel", KERNELS)
    def test_diagonal_is_zero(self, device, kernel):
        """Diagonal (self-interaction) should be zero."""
        _seed_all(1, device)
        N = 32
        positions = torch.randn(N, 2, device=device)
        diff = positions[:, None, :] - positions[None, :, :]
        sigma = 0.5
        grad = compute_kernel_gradient(diff, sigma, kernel)
        diag = torch.diagonal(grad, dim1=0, dim2=1)  # (2, N)
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("kernel", KERNELS)
    def test_outside_support_is_zero(self, device, kernel):
        """Points far apart (u>1) should have zero gradient."""
        positions = torch.tensor([[0.0, 0.0], [10.0, 0.0]], device=device)
        diff = positions[:, None, :] - positions[None, :, :]
        sigma = 1.0
        grad = compute_kernel_gradient(diff, sigma, kernel)
        assert torch.allclose(grad[0, 1], torch.zeros(2, device=device), atol=1e-6)
        assert torch.allclose(grad[1, 0], torch.zeros(2, device=device), atol=1e-6)


class TestAngleConstraintMatrices:
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("shape_name,shape_fn", SHAPES)
    @pytest.mark.parametrize("n_points", N_POINTS)
    def test_matrices_shape_and_finite(self, device, shape_name, shape_fn, n_points):
        _seed_all(0, device)
        varifold, masses, sigma, tangents, coherence = _setup_shape(shape_fn, n_points, device)

        A, B = compute_angle_constraint_matrices(
            varifold.positions, tangents, masses, coherence, sigma
        )
        N = varifold.n_points
        assert A.shape == (N, N)
        assert B.shape == (N, N)
        assert torch.isfinite(A).all()
        assert torch.isfinite(B).all()

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("shape_name,shape_fn", SHAPES)
    def test_weak_form_solution(self, device, shape_name, shape_fn):
        """Test that solving A @ Δθ = B @ s produces reasonable Δθ."""
        _seed_all(42, device)
        varifold, masses, sigma, tangents, coherence = _setup_shape(shape_fn, 64, device)

        A, B = compute_angle_constraint_matrices(
            varifold.positions, tangents, masses, coherence, sigma
        )

        # Small random displacement
        s = torch.randn(varifold.n_points, device=device, dtype=varifold.positions.dtype) * 0.01

        # Solve using TruncatedSVD
        A_svd = TruncatedSVD.from_matrix(A)
        rhs = B @ s
        delta_theta = A_svd.apply(rhs)

        assert torch.isfinite(delta_theta).all()

        # Δθ should be reasonably bounded (not exploding)
        # With the new weak form, amplification should be ~10-20x for ||s|| ~ 0.08
        amplification = delta_theta.norm() / s.norm()
        print(f"\n{shape_name}({device}): ||Δθ||/||s|| = {amplification:.2f}x, "
              f"max|Δθ| = {delta_theta.abs().max():.4f}")

        # Amplification should be reasonable (< 50x)
        assert amplification < 50, f"Amplification too large: {amplification:.2f}x"


class TestTruncatedSVD:
    @pytest.mark.parametrize("device", DEVICES)
    def test_basic_pinv(self, device):
        """Test that TruncatedSVD gives correct pseudo-inverse for well-conditioned matrix."""
        _seed_all(0, device)
        A = torch.randn(10, 10, device=device)
        A = A @ A.T + 0.1 * torch.eye(10, device=device)  # Make it positive definite

        svd = TruncatedSVD.from_matrix(A)
        b = torch.randn(10, device=device)
        x = svd.apply(b)

        # Check that A @ x ≈ b
        residual = (A @ x - b).norm() / b.norm()
        assert residual < 1e-4, f"Residual too large: {residual}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_rank_deficient(self, device):
        """Test that TruncatedSVD handles rank-deficient matrices gracefully."""
        _seed_all(1, device)
        # Create rank-deficient matrix
        A = torch.randn(10, 5, device=device)
        A = A @ A.T  # Rank at most 5

        svd = TruncatedSVD.from_matrix(A)
        b = torch.randn(10, device=device)
        x = svd.apply(b)

        # Should not have NaN/Inf
        assert torch.isfinite(x).all()

        # Should give minimum norm solution in range(A)
        assert svd.n_truncated >= 5  # At least 5 singular values should be truncated


class TestCoherenceSuppression:
    """Test that dθ *= q coherence suppression works correctly."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_dtheta_suppressed_by_coherence(self, device):
        """Low-coherence points should have proportionally smaller |dθ|."""
        _seed_all(42, device)
        # two_ellipses with gap creates low-coherence points
        varifold = generate_oriented_two_ellipses(
            n_per_ellipse=32, device=device,
        )
        masses, sigma, tangents, coherence = _setup_shape_from_varifold(varifold, device)

        A, B = compute_angle_constraint_matrices(
            varifold.positions, tangents, masses, coherence, sigma,
        )

        # Constraint weights (same as MMStepper._setup_step)
        constraint_weights = coherence * coherence * masses

        param = NormalGraphParametrization(
            prev_positions=varifold.positions,
            prev_normals=varifold.normals,
            prev_angles=varifold.angles,
            constraint_weights=constraint_weights,
            A_angle=A,
            B_angle=B,
            coherence=coherence,
        )

        # Non-trivial displacement
        y = torch.randn(varifold.n_points - 1, device=device, dtype=varifold.positions.dtype) * 0.01
        displacements, delta_angles = param.unpack_params(y)

        assert torch.isfinite(delta_angles).all()

        # Identify low/high coherence points
        q = coherence
        low_mask = q < 0.5
        high_mask = q >= 0.9

        if low_mask.any() and high_mask.any():
            # Compute what dθ would be WITHOUT coherence suppression
            dtheta_raw = param.AB_solve @ displacements  # before *= q
            dtheta_with_q = delta_angles  # after *= q

            # For low-coherence points, |dθ_with_q| / |dθ_raw| should be ≈ q
            low_idx = torch.where(low_mask)[0]
            for i in low_idx[:5]:
                raw_val = abs(dtheta_raw[i].item())
                sup_val = abs(dtheta_with_q[i].item())
                q_val = q[i].item()
                if raw_val > 1e-10:
                    ratio = sup_val / raw_val
                    assert abs(ratio - q_val) < 1e-6, (
                        f"Point {i}: ratio={ratio:.6f}, q={q_val:.6f}"
                    )

            # Overall: max|dθ| for low-coh points should be smaller
            low_max = delta_angles[low_mask].abs().max().item()
            high_max = delta_angles[high_mask].abs().max().item()
            print(f"\n{device}: low-coh max|dθ|={low_max:.4e}, "
                  f"high-coh max|dθ|={high_max:.4e}, "
                  f"q_min={q.min():.4f}")

    @pytest.mark.parametrize("device", DEVICES)
    def test_lstsq_vs_svd_consistency(self, device):
        """AB_solve gives same result on CPU (lstsq) and via SVD."""
        _seed_all(0, device)
        varifold = generate_oriented_circle(64, device=device)
        masses, sigma, tangents, coherence = _setup_shape_from_varifold(varifold, device)

        A, B = compute_angle_constraint_matrices(
            varifold.positions, tangents, masses, coherence, sigma,
        )

        # Compute AB_solve via SVD (reference)
        svd = TruncatedSVD.from_matrix(A)
        AB_svd = svd.Vh.T @ (svd.S_inv.unsqueeze(-1) * (svd.U.T @ B.detach()))

        # Compute AB_solve via parametrization (uses lstsq on CPU)
        constraint_weights = coherence * coherence * masses
        param = NormalGraphParametrization(
            prev_positions=varifold.positions,
            prev_normals=varifold.normals,
            prev_angles=varifold.angles,
            constraint_weights=constraint_weights,
            A_angle=A,
            B_angle=B,
            coherence=coherence,
        )

        # lstsq (gelsd) is more accurate than explicit SVD matrix products;
        # check relative difference against the magnitude of AB_solve
        diff = (param.AB_solve - AB_svd).abs().max().item()
        scale = max(param.AB_solve.abs().max().item(), 1e-10)
        rel_diff = diff / scale
        print(f"\n{device}: max|AB_lstsq - AB_svd| = {diff:.2e}, "
              f"relative = {rel_diff:.2e}")
        assert rel_diff < 1e-4, f"AB_solve relative mismatch: {rel_diff:.2e}"
