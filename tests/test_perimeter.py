"""Tests for coherence-based perimeter computation."""

import pytest
import torch
import math
from pathlib import Path

import matplotlib.pyplot as plt

from src.torch.perimeter import (
    mollifier_2d,
    compute_scalar_density,
    compute_vector_field,
    compute_coherence,
    compute_recommended_sigma,
    compute_perimeter_coherence,
    compute_perimeter_coherence_with_details,
)
from src.torch.oriented_varifold import (
    OrientedPointCloudVarifold,
    compute_recommended_params,
    compute_masses,
)
from src.torch.shapes import generate_oriented_circle, generate_oriented_two_ellipses


OUTPUT_DIR = Path(__file__).parent / "outputs" / "perimeter"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def ellipse_perimeter(a: float, b: float) -> float:
    """Approximate perimeter of ellipse (Ramanujan approximation)."""
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


class TestMollifier2D:
    """Tests for 2D mollifier functions."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("kernel", ["wendland_c2", "biweight", "epanechnikov"])
    def test_mollifier_normalization(self, device, kernel):
        """Test that 2D mollifier integrates to 1."""
        sigma = 1.0
        n_grid = 100
        x = torch.linspace(-sigma * 1.5, sigma * 1.5, n_grid, device=device)
        y = torch.linspace(-sigma * 1.5, sigma * 1.5, n_grid, device=device)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        xx, yy = torch.meshgrid(x, y, indexing="ij")
        z = torch.stack([xx, yy], dim=-1)

        rho = mollifier_2d(z, sigma, kernel)
        integral = (rho * dx * dy).sum().item()

        assert abs(integral - 1.0) < 0.05, f"Integral = {integral:.4f}, expected 1.0"

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("kernel", ["wendland_c2", "biweight", "epanechnikov"])
    def test_mollifier_compact_support(self, device, kernel):
        """Test that mollifier is zero outside support."""
        sigma = 1.0
        z_outside = torch.tensor([[1.5, 0.0], [0.0, 1.5], [1.1, 1.1]], device=device)
        rho = mollifier_2d(z_outside, sigma, kernel)
        assert (rho == 0).all(), "Mollifier should be 0 outside support"


class TestCoherence:
    """Tests for coherence computation."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_coherence_aligned_normals(self, device):
        """Test that aligned normals give coherence ≈ 1."""
        n_points = 10
        positions = torch.randn(n_points, 2, device=device)
        normals = torch.tensor([[1.0, 0.0]], device=device).expand(n_points, 2).contiguous()
        masses = torch.ones(n_points, device=device) / n_points

        sigma = 0.5
        U = compute_scalar_density(positions, masses, sigma)
        V = compute_vector_field(positions, normals, masses, sigma)
        q = compute_coherence(U, V)

        assert (q > 0.9).all(), f"Coherence should be ~1 for aligned normals, got {q}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_coherence_opposing_normals(self, device):
        """Test that opposing normals give coherence ≈ 0."""
        positions = torch.tensor([[0.0, 0.0], [0.0, 0.0]], device=device)
        normals = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], device=device)
        masses = torch.tensor([0.5, 0.5], device=device)

        sigma = 0.5
        U = compute_scalar_density(positions, masses, sigma)
        V = compute_vector_field(positions, normals, masses, sigma)
        q = compute_coherence(U, V)

        assert (q < 0.1).all(), f"Coherence should be ~0 for opposing normals, got {q}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_coherence_range(self, device):
        """Test that coherence is in [0, 1]."""
        n_points = 50
        positions = torch.randn(n_points, 2, device=device)
        angles = torch.rand(n_points, device=device) * 2 * math.pi
        normals = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        masses = torch.ones(n_points, device=device) / n_points

        sigma = 0.5
        U = compute_scalar_density(positions, masses, sigma)
        V = compute_vector_field(positions, normals, masses, sigma)
        q = compute_coherence(U, V)

        assert (q >= 0).all() and (q <= 1.0 + 1e-6).all()


class TestPerimeter:
    """Tests for coherence-based perimeter estimation."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_circle_perimeter_accuracy(self, device):
        """Test perimeter estimation on a circle."""
        n_points = 256
        radius = 1.0
        v = generate_oriented_circle(n_points, radius=radius, device=device)

        delta, tau = compute_recommended_params(v.positions, k0=10, k_min=1.0)
        masses = compute_masses(v.positions, delta, tau)

        perimeter = compute_perimeter_coherence(v, masses)
        expected = 2 * math.pi * radius

        rel_error = abs(perimeter.item() - expected) / expected
        assert rel_error < 0.05, f"Error {rel_error:.1%}: got {perimeter.item():.4f}, expected {expected:.4f}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_perimeter_scales_with_radius(self, device):
        """Test that perimeter scales linearly with radius."""
        n_points = 128
        perimeters = []

        for radius in [0.5, 1.0, 2.0]:
            v = generate_oriented_circle(n_points, radius=radius, device=device)
            delta, tau = compute_recommended_params(v.positions, k0=10, k_min=1.0)
            masses = compute_masses(v.positions, delta, tau)
            p = compute_perimeter_coherence(v, masses)
            perimeters.append(p.item())

        ratio_1_2 = perimeters[1] / perimeters[0]
        ratio_2_3 = perimeters[2] / perimeters[1]

        assert abs(ratio_1_2 - 2.0) < 0.2, f"Ratio should be 2.0, got {ratio_1_2:.2f}"
        assert abs(ratio_2_3 - 2.0) < 0.2, f"Ratio should be 2.0, got {ratio_2_3:.2f}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_separated_ellipses(self, device):
        """Test perimeter of two separated ellipses."""
        n_per_ellipse = 192
        v = generate_oriented_two_ellipses(n_per_ellipse, device=device)

        delta, tau = compute_recommended_params(v.positions, k0=10, k_min=1.0)
        masses = compute_masses(v.positions, delta, tau)
        perimeter = compute_perimeter_coherence(v, masses)

        expected = 2 * ellipse_perimeter(0.4, 1.0)
        rel_error = abs(perimeter.item() - expected) / expected

        assert rel_error < 0.1, f"Error {rel_error:.1%}: got {perimeter.item():.4f}, expected {expected:.4f}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_touching_ellipses_hidden_boundary(self, device):
        """Test that touching ellipses have reduced perimeter (hidden boundary)."""
        n_per_ellipse = 192

        # Separated
        v_sep = generate_oriented_two_ellipses(
            n_per_ellipse,
            a1=0.3, b1=0.8, center1=(-0.5, 0.0),
            a2=0.3, b2=0.8, center2=(0.5, 0.0),
            device=device,
        )
        delta_sep, tau_sep = compute_recommended_params(v_sep.positions, k0=10, k_min=1.0)
        masses_sep = compute_masses(v_sep.positions, delta_sep, tau_sep)
        p_sep = compute_perimeter_coherence(v_sep, masses_sep)

        # Touching
        v_touch = generate_oriented_two_ellipses(
            n_per_ellipse,
            a1=0.3, b1=0.8, center1=(-0.3, 0.0),
            a2=0.3, b2=0.8, center2=(0.3, 0.0),
            device=device,
        )
        delta_touch, tau_touch = compute_recommended_params(v_touch.positions, k0=10, k_min=1.0)
        masses_touch = compute_masses(v_touch.positions, delta_touch, tau_touch)
        p_touch = compute_perimeter_coherence(v_touch, masses_touch)

        assert p_touch < p_sep, f"Touching ({p_touch.item():.4f}) should be < separated ({p_sep.item():.4f})"


class TestGradientSafety:
    """Tests for NaN-free gradients through mollifier at z=0."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("kernel", ["wendland_c2", "biweight", "epanechnikov"])
    def test_mollifier_2d_grad_no_nan(self, device, kernel):
        """Gradient through mollifier_2d must be NaN-free even at z=0 (diagonal)."""
        pos = torch.tensor(
            [[0.0, 0.0], [0.1, 0.0]], dtype=torch.float64, device=device,
        )
        pos.requires_grad_(True)
        z = pos.unsqueeze(1) - pos.unsqueeze(0)  # (2, 2, 2), diagonal has z=0
        result = mollifier_2d(z, sigma=0.5, kernel=kernel)
        grad = torch.autograd.grad(result.sum(), pos)[0]
        assert not torch.isnan(grad).any(), f"NaN in grad for kernel={kernel}: {grad}"

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("kernel", ["wendland_c2", "biweight", "epanechnikov"])
    def test_mollifier_2d_hessian_no_nan(self, device, kernel):
        """Hessian through mollifier_2d must be NaN-free even at z=0."""
        pos = torch.tensor(
            [[0.0, 0.0], [0.1, 0.0]], dtype=torch.float64, device=device,
        )

        def f(p):
            z = p.unsqueeze(1) - p.unsqueeze(0)
            return mollifier_2d(z, sigma=0.5, kernel=kernel).sum()

        H = torch.autograd.functional.hessian(f, pos)
        assert not torch.isnan(H).any(), f"NaN in Hessian for kernel={kernel}"

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("kernel", ["wendland_c2", "biweight", "epanechnikov"])
    def test_mollifier_2d_grad_vs_finite_diff(self, device, kernel):
        """Hand-computed backward must match finite-difference gradient, including near z=0."""
        # Points include a near-zero pair (0.0, 0.0) and (1e-8, 0.0).
        # All pairwise distances are well inside support (u << 1) to avoid
        # kink at support boundary of C⁰ kernels like Epanechnikov.
        pos = torch.tensor(
            [[0.0, 0.0], [1e-8, 0.0], [0.2, 0.0], [0.0, 0.15]],
            dtype=torch.float64, device=device,
        )
        sigma = 0.5

        def f(p):
            z = p.unsqueeze(1) - p.unsqueeze(0)
            return mollifier_2d(z, sigma, kernel=kernel).sum()

        # Autograd gradient
        pos_ad = pos.clone().requires_grad_(True)
        f(pos_ad).backward()
        grad_ad = pos_ad.grad.clone()

        # Finite-difference gradient
        eps = 1e-6
        grad_fd = torch.zeros_like(pos)
        for i in range(pos.shape[0]):
            for j in range(pos.shape[1]):
                p_plus = pos.clone()
                p_plus[i, j] += eps
                p_minus = pos.clone()
                p_minus[i, j] -= eps
                grad_fd[i, j] = (f(p_plus) - f(p_minus)) / (2 * eps)

        assert not torch.isnan(grad_ad).any(), f"NaN in AD grad: {grad_ad}"
        assert torch.allclose(grad_ad, grad_fd, atol=1e-4, rtol=1e-3), (
            f"kernel={kernel}: AD vs FD mismatch\n"
            f"  AD:  {grad_ad}\n"
            f"  FD:  {grad_fd}\n"
            f"  diff: {(grad_ad - grad_fd).abs()}"
        )


class TestDifferentiability:
    """Tests for AD gradient flow in perimeter computation."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradient_to_positions(self, device):
        """Test that gradients flow to positions."""
        n_points = 64
        v = generate_oriented_circle(n_points, radius=1.0, device=device)
        v.positions.requires_grad_(True)

        delta, tau = compute_recommended_params(v.positions.detach(), k0=10, k_min=1.0)
        masses = compute_masses(v.positions.detach(), delta, tau)

        perimeter = compute_perimeter_coherence(v, masses)
        perimeter.backward()

        assert v.positions.grad is not None
        assert not torch.isnan(v.positions.grad).any()

    @pytest.mark.parametrize("device", DEVICES)
    def test_gradient_to_angles(self, device):
        """Test that gradients flow to angles."""
        n_points = 64
        v = generate_oriented_circle(n_points, radius=1.0, device=device)
        v.angles.requires_grad_(True)

        delta, tau = compute_recommended_params(v.positions, k0=10, k_min=1.0)
        masses = compute_masses(v.positions, delta, tau)

        perimeter = compute_perimeter_coherence(v, masses)
        perimeter.backward()

        assert v.angles.grad is not None
        assert not torch.isnan(v.angles.grad).any()


class TestVisualization:
    """Visualization tests for perimeter computation."""

    def test_visualization_circle_coherence(self):
        """Visualize coherence on a circle."""
        n_points = 256
        radius = 1.0
        v = generate_oriented_circle(n_points, radius=radius, device="cpu")

        delta, tau = compute_recommended_params(v.positions, k0=10, k_min=1.0)
        masses = compute_masses(v.positions, delta, tau)

        perimeter, U, V, q, sigma = compute_perimeter_coherence_with_details(v, masses)
        expected = 2 * math.pi * radius

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        pos = v.positions.numpy()

        ax = axes[0]
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=q.numpy(), cmap="viridis", s=30, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label="Coherence q")
        ax.set_title(f"Coherence (σ={sigma:.3f})\nMean q = {q.mean():.4f}")
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        ax = axes[1]
        mq = (masses * q).numpy()
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=mq, cmap="plasma", s=30)
        plt.colorbar(sc, ax=ax, label="m_i × q_i")
        ax.set_title(f"Perimeter Contribution\nP̂ = {perimeter.item():.3f} (expected: {expected:.3f})")
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "coherence_circle.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()

    def test_visualization_two_ellipses_touching(self):
        """Visualize coherence on two touching ellipses."""
        n_per_ellipse = 192
        v = generate_oriented_two_ellipses(
            n_per_ellipse,
            a1=0.3, b1=0.8, center1=(-0.3, 0.0),
            a2=0.3, b2=0.8, center2=(0.3, 0.0),
            device="cpu",
        )

        delta, tau = compute_recommended_params(v.positions, k0=10, k_min=1.0)
        masses = compute_masses(v.positions, delta, tau)

        perimeter, U, V, q, sigma = compute_perimeter_coherence_with_details(v, masses)
        expected_full = 2 * ellipse_perimeter(0.3, 0.8)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        pos = v.positions.numpy()
        V_np = V.detach().numpy()

        # Plot 1: Coherence q
        ax = axes[0]
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=q.numpy(), cmap="viridis", s=30, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label="Coherence q")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Contact")
        ax.set_title(f"Coherence q = |V|/U (σ={sigma:.3f})")
        ax.set_aspect("equal")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.legend()

        # Plot 2: Vector field V_σ (quiver)
        ax = axes[1]
        V_norm = V.norm(dim=-1).numpy()
        V_max = V_norm.max()

        # Background scatter colored by |V|
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=V_norm, cmap="viridis", s=15, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="|V_σ|")

        # Quiver plot with normalized arrows (color shows magnitude)
        step = 3
        Q = ax.quiver(
            pos[::step, 0], pos[::step, 1],
            V_np[::step, 0], V_np[::step, 1],
            V_norm[::step],  # color by magnitude
            cmap="Reds", alpha=0.9,
            scale=V_max * 15, width=0.008
        )
        # Add reference arrow
        ax.quiverkey(Q, 0.85, 0.95, V_max, f"|V|={V_max:.2f}", labelpos="W", coordinates="axes")

        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Vector Field V_σ (σ={sigma:.3f})")
        ax.set_aspect("equal")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

        # Plot 3: Perimeter contribution
        ax = axes[2]
        mq = (masses * q).numpy()
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=mq, cmap="plasma", s=30)
        plt.colorbar(sc, ax=ax, label="m_i × q_i")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Contact")
        ax.set_title(f"Perimeter Contribution\nP̂ = {perimeter.item():.3f}, Full: {expected_full:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.legend()

        plt.tight_layout()
        output_path = OUTPUT_DIR / "coherence_touching_ellipses.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()

    def test_visualization_sigma_sensitivity(self):
        """Visualize perimeter vs sigma."""
        n_per_ellipse = 192
        v = generate_oriented_two_ellipses(
            n_per_ellipse,
            a1=0.3, b1=0.8, center1=(-0.3, 0.0),
            a2=0.3, b2=0.8, center2=(0.3, 0.0),
            device="cpu",
        )

        delta, tau = compute_recommended_params(v.positions, k0=10, k_min=1.0)
        masses = compute_masses(v.positions, delta, tau)

        sigma_base = compute_recommended_sigma(v.positions, c_sigma=1.0)
        sigma_multipliers = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        sigmas = [sigma_base * m for m in sigma_multipliers]
        perimeters = []

        for sigma in sigmas:
            p = compute_perimeter_coherence(v, masses, sigma=sigma)
            perimeters.append(p.item())

        expected_full = 2 * ellipse_perimeter(0.3, 0.8)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sigma_multipliers, perimeters, "o-", linewidth=2, markersize=8)
        ax.axhline(y=expected_full, color="red", linestyle="--", label=f"Full: {expected_full:.3f}")
        ax.set_xlabel("c_σ (σ = c_σ × median NN distance)")
        ax.set_ylabel("Estimated Perimeter")
        ax.set_title("Perimeter vs σ (Touching Ellipses)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = OUTPUT_DIR / "coherence_perimeter_vs_sigma.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()
