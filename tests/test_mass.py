"""Tests for mass computation module."""

import pytest
import torch
import math
from pathlib import Path

import matplotlib.pyplot as plt

from src.torch.oriented_varifold import (
    wendland_c2,
    biweight,
    epanechnikov,
    chi_tau,
    compute_knn_distances,
    compute_recommended_params,
    compute_kde_density,
    compute_masses,
    compute_masses_uniform,
)
from src.torch.shapes import generate_oriented_circle, generate_oriented_two_ellipses


def ellipse_perimeter(a: float, b: float) -> float:
    """Approximate perimeter of ellipse (Ramanujan approximation)."""
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


OUTPUT_DIR = Path(__file__).parent / "outputs" / "mass"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


class TestKernelFunctions:
    """Tests for kernel functions."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_wendland_c2_support(self, device):
        """Test Wendland C2 has compact support [0, 1]."""
        u = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], device=device)
        k = wendland_c2(u)

        # Inside support: positive
        assert k[0] > 0
        assert k[1] > 0

        # At boundary: zero
        assert k[2] == 0

        # Outside support: zero
        assert k[3] == 0
        assert k[4] == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_wendland_c2_at_zero(self, device):
        """Test Wendland C2 value at u=0."""
        u = torch.tensor([0.0], device=device)
        k = wendland_c2(u)
        # η(0) = (1-0)⁴(4*0+1) = 1
        assert torch.isclose(k[0], torch.tensor(1.0, device=device))

    @pytest.mark.parametrize("device", DEVICES)
    def test_biweight_support(self, device):
        """Test biweight has compact support [0, 1]."""
        u = torch.tensor([0.0, 0.5, 1.0, 1.5], device=device)
        k = biweight(u)

        assert k[0] > 0
        assert k[1] > 0
        assert k[2] == 0
        assert k[3] == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_biweight_at_zero(self, device):
        """Test biweight value at u=0."""
        u = torch.tensor([0.0], device=device)
        k = biweight(u)
        # η(0) = (1-0)² = 1
        assert torch.isclose(k[0], torch.tensor(1.0, device=device))

    @pytest.mark.parametrize("device", DEVICES)
    def test_epanechnikov_support(self, device):
        """Test Epanechnikov has compact support [0, 1]."""
        u = torch.tensor([0.0, 0.5, 1.0, 1.5], device=device)
        k = epanechnikov(u)

        assert k[0] > 0
        assert k[1] > 0
        assert k[2] == 0
        assert k[3] == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_kernels_non_negative(self, device):
        """Test all kernels are non-negative."""
        u = torch.linspace(0, 2, 100, device=device)

        assert (wendland_c2(u) >= 0).all()
        assert (biweight(u) >= 0).all()
        assert (epanechnikov(u) >= 0).all()


class TestChiTau:
    """Tests for chi_tau cutoff function."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_chi_tau_regions(self, device):
        """Test chi_tau behavior in different regions."""
        tau = 1.0
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.5], device=device)
        chi = chi_tau(t, tau)

        # t < tau/2: chi = 0
        assert chi[0] == 0  # t=0
        assert chi[1] == 0  # t=0.25

        # t = tau/2: chi = 0 (at boundary)
        assert chi[2] == 0  # t=0.5

        # tau/2 < t < tau: chi in (0, 1)
        assert 0 < chi[3] < 1  # t=0.75

        # t >= tau: chi = 1
        assert chi[4] == 1  # t=1.0
        assert chi[5] == 1  # t=1.5

    @pytest.mark.parametrize("device", DEVICES)
    def test_chi_tau_linear(self, device):
        """Test chi_tau is linear in transition region."""
        tau = 2.0
        # t in (tau/2, tau) = (1, 2)
        t = torch.tensor([1.25, 1.5, 1.75], device=device)
        chi = chi_tau(t, tau)

        # Expected: (2t/tau - 1) = t - 1
        expected = t - 1.0
        assert torch.allclose(chi, expected)

    @pytest.mark.parametrize("device", DEVICES)
    def test_chi_tau_range(self, device):
        """Test chi_tau output is in [0, 1]."""
        tau = 1.0
        t = torch.rand(100, device=device) * 3  # Random values in [0, 3]
        chi = chi_tau(t, tau)

        assert (chi >= 0).all()
        assert (chi <= 1).all()


class TestRecommendedParams:
    """Tests for automatic parameter selection."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_knn_distances_circle(self, device):
        """Test kNN distances on uniformly sampled circle."""
        n_points = 64
        radius = 1.0
        v = generate_oriented_circle(n_points, radius=radius, device=device)

        # For uniform circle, all k-th neighbor distances should be equal
        for k in [1, 5, 10]:
            r_k = compute_knn_distances(v.positions, k)
            # All distances should be nearly identical
            cv = r_k.std() / r_k.mean()
            assert cv < 0.01, f"kNN distances not uniform: cv={cv:.4f}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_recommended_params_circle(self, device):
        """Test recommended params give good mass estimation for circle."""
        n_points = 64
        radius = 1.0
        v = generate_oriented_circle(n_points, radius=radius, device=device)

        delta, tau = compute_recommended_params(v.positions, k0=10, k_min=1.0)

        # delta should be reasonable (not too small or large)
        arc_length = 2 * math.pi * radius / n_points
        assert delta > arc_length, f"delta too small: {delta:.4f} < {arc_length:.4f}"
        assert delta < radius, f"delta too large: {delta:.4f} > {radius:.4f}"

        # Compute masses and check sum
        masses = compute_masses(v.positions, delta, tau)
        total_mass = masses.sum().item()
        expected = 2 * math.pi * radius
        rel_error = abs(total_mass - expected) / expected
        assert rel_error < 0.05, f"Mass error too large: {rel_error:.1%}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_recommended_params_two_ellipses(self, device):
        """Test recommended params give good mass estimation for two ellipses."""
        n_per_ellipse = 48
        v = generate_oriented_two_ellipses(n_per_ellipse, device=device)

        delta, tau = compute_recommended_params(v.positions, k0=10, k_min=1.0)

        # Compute masses and check sum
        masses = compute_masses(v.positions, delta, tau)
        total_mass = masses.sum().item()
        expected = 2 * ellipse_perimeter(0.4, 1.0)
        rel_error = abs(total_mass - expected) / expected
        assert rel_error < 0.05, f"Mass error too large: {rel_error:.1%}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_recommended_params_scales_with_n(self, device):
        """Test that delta decreases as N increases."""
        radius = 1.0
        deltas = []

        for n_points in [32, 64, 128]:
            v = generate_oriented_circle(n_points, radius=radius, device=device)
            delta, tau = compute_recommended_params(v.positions, k0=10, k_min=1.0)
            deltas.append(delta)

        # Delta should decrease with more points
        assert deltas[0] > deltas[1] > deltas[2], f"Delta should decrease: {deltas}"


class TestKDEDensity:
    """Tests for KDE density estimation."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("kernel", ["wendland_c2", "biweight", "epanechnikov"])
    def test_density_positive(self, device, kernel):
        """Test density is always positive."""
        points = torch.randn(50, 2, device=device)
        delta = 0.5

        density = compute_kde_density(points, delta, kernel)

        assert (density > 0).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_density_uniform_grid(self, device):
        """Test density is approximately uniform on uniform grid."""
        # Create uniform grid
        x = torch.linspace(-1, 1, 10, device=device)
        y = torch.linspace(-1, 1, 10, device=device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

        delta = 0.5
        density = compute_kde_density(points, delta, "wendland_c2")

        # Interior points should have similar density
        interior_mask = (points.abs() < 0.5).all(dim=1)
        if interior_mask.sum() > 1:
            interior_density = density[interior_mask]
            cv = interior_density.std() / interior_density.mean()  # coefficient of variation
            assert cv < 0.3, f"Interior density not uniform: cv={cv:.3f}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_density_circle_points(self, device):
        """Test density on circle points."""
        n_points = 64
        v = generate_oriented_circle(n_points, radius=1.0, device=device)
        points = v.positions

        delta = 0.3
        density = compute_kde_density(points, delta, "wendland_c2")

        # All points on circle should have similar density
        cv = density.std() / density.mean()
        assert cv < 0.2, f"Circle density not uniform: cv={cv:.3f}"


class TestComputeMasses:
    """Tests for mass computation."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_masses_sum_to_perimeter(self, device):
        """Test that masses sum to approximately the curve perimeter."""
        n_points = 64
        radius = 1.0
        v = generate_oriented_circle(n_points, radius=radius, device=device)

        # Use parameters that ensure chi_tau doesn't filter all points
        delta = 0.3
        tau = 0.1  # Small tau so chi_tau(density, tau) ≈ 1

        masses = compute_masses(v.positions, delta, tau)

        # Masses should sum to approximately the perimeter (2πr)
        total_mass = masses.sum().item()
        expected_perimeter = 2 * math.pi * radius
        rel_error = abs(total_mass - expected_perimeter) / expected_perimeter
        assert rel_error < 0.1, f"Total mass {total_mass:.3f}, expected perimeter {expected_perimeter:.3f}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_masses_non_negative(self, device):
        """Test that masses are non-negative."""
        points = torch.randn(50, 2, device=device)
        delta = 0.5
        tau = 0.5

        masses = compute_masses(points, delta, tau)

        assert (masses >= 0).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_masses_sparse_points_filtered(self, device):
        """Test that sparse outlier points get low mass."""
        # Dense cluster + sparse outlier
        cluster = torch.randn(50, 2, device=device) * 0.2
        outlier = torch.tensor([[5.0, 5.0]], device=device)
        points = torch.cat([cluster, outlier], dim=0)

        delta = 0.5
        tau = 0.5

        masses = compute_masses(points, delta, tau)

        # Outlier should have lower mass than cluster points
        outlier_mass = masses[-1]
        cluster_masses = masses[:-1]

        assert outlier_mass < cluster_masses.mean(), "Outlier mass should be lower"


class TestUniformMasses:
    """Tests for uniform mass computation."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_uniform_masses_sum(self, device):
        """Test uniform masses sum to total_mass."""
        n_points = 100
        total_mass = 2 * math.pi  # perimeter of unit circle

        masses = compute_masses_uniform(n_points, total_mass, device=device)

        assert torch.isclose(masses.sum(), torch.tensor(total_mass, device=device))

    @pytest.mark.parametrize("device", DEVICES)
    def test_uniform_masses_equal(self, device):
        """Test uniform masses are all equal."""
        n_points = 50
        total_mass = 5.0

        masses = compute_masses_uniform(n_points, total_mass, device=device)

        expected = total_mass / n_points
        assert torch.allclose(masses, torch.full_like(masses, expected))


class TestMassVisualization:
    """Visualization tests for mass computation."""

    def test_visualization_kernel_functions(self):
        """Visualize all kernel functions."""
        u = torch.linspace(0, 1.5, 100)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(u.numpy(), wendland_c2(u).numpy(), label="Wendland C²", linewidth=2)
        ax.plot(u.numpy(), biweight(u).numpy(), label="Biweight", linewidth=2)
        ax.plot(u.numpy(), epanechnikov(u).numpy(), label="Epanechnikov", linewidth=2)

        ax.axvline(x=1.0, color="gray", linestyle="--", label="Support boundary")
        ax.set_xlabel("u = |x|/δ")
        ax.set_ylabel("η(u)")
        ax.set_title("Kernel Functions")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 1.1)

        output_path = OUTPUT_DIR / "kernel_functions.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()

    def test_visualization_chi_tau(self):
        """Visualize chi_tau function for different tau values."""
        t = torch.linspace(0, 2, 100)

        fig, ax = plt.subplots(figsize=(8, 6))

        for tau in [0.5, 1.0, 1.5]:
            chi = chi_tau(t, tau)
            ax.plot(t.numpy(), chi.numpy(), label=f"τ={tau}", linewidth=2)

        ax.set_xlabel("t (density)")
        ax.set_ylabel("χ_τ(t)")
        ax.set_title("Cutoff Function χ_τ")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2)
        ax.set_ylim(-0.1, 1.1)

        output_path = OUTPUT_DIR / "chi_tau.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()

    def test_visualization_circle_masses(self):
        """Visualize mass distribution on a circle."""
        n_points = 256
        radius = 1.0
        v = generate_oriented_circle(n_points, radius=radius, device="cpu")

        # Use recommended parameters
        delta, tau = compute_recommended_params(v.positions, k0=10, k_min=1.0)

        # KDE-based masses
        masses_kde = compute_masses(v.positions, delta, tau)

        # Uniform masses
        masses_uniform = compute_masses_uniform(n_points, 2 * math.pi * radius, device="cpu")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        pos = v.positions.numpy()

        # KDE masses
        ax = axes[0]
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=masses_kde.numpy(), cmap="viridis", s=50)
        plt.colorbar(sc, ax=ax, label="Mass")
        ax.set_title(f"KDE Masses (δ={delta:.3f}, τ={tau:.3f})\nTotal: {masses_kde.sum():.3f}, Expected: {2*math.pi*radius:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        # Uniform masses
        ax = axes[1]
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=masses_uniform.numpy(), cmap="viridis", s=50)
        plt.colorbar(sc, ax=ax, label="Mass")
        ax.set_title(f"Uniform Masses\nTotal: {masses_uniform.sum():.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "circle_masses.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()

    def test_visualization_two_ellipses_masses(self):
        """Visualize mass distribution on two ellipses."""
        n_per_ellipse = 192
        v = generate_oriented_two_ellipses(n_per_ellipse, device="cpu")

        # Use recommended parameters
        delta, tau = compute_recommended_params(v.positions, k0=10, k_min=1.0)

        masses = compute_masses(v.positions, delta, tau)

        # Expected perimeter
        a, b = 0.4, 1.0
        expected = 2 * ellipse_perimeter(a, b)

        fig, ax = plt.subplots(figsize=(8, 6))
        pos = v.positions.numpy()

        sc = ax.scatter(pos[:, 0], pos[:, 1], c=masses.numpy(), cmap="viridis", s=30)
        plt.colorbar(sc, ax=ax, label="Mass")
        ax.set_title(f"Two Ellipses Masses (δ={delta:.3f}, τ={tau:.3f})\nTotal: {masses.sum():.3f}, Expected: {expected:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        output_path = OUTPUT_DIR / "two_ellipses_masses.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()

    def test_visualization_two_ellipses_touching_masses(self):
        """Visualize mass distribution on two touching ellipses."""
        from src.torch.shapes import generate_oriented_two_ellipses

        n_per_ellipse = 192
        # Touching configuration: centers at (-0.3, 0) and (0.3, 0), a=0.3 -> edges meet at x=0
        v = generate_oriented_two_ellipses(
            n_per_ellipse,
            a1=0.3, b1=0.8, center1=(-0.3, 0.0),
            a2=0.3, b2=0.8, center2=(0.3, 0.0),
            device="cpu",
        )

        # Use recommended parameters
        delta, tau = compute_recommended_params(v.positions, k0=10, k_min=1.0)

        masses = compute_masses(v.positions, delta, tau)

        # Expected perimeter
        a, b = 0.3, 0.8
        expected = 2 * ellipse_perimeter(a, b)

        fig, ax = plt.subplots(figsize=(8, 6))
        pos = v.positions.numpy()

        sc = ax.scatter(pos[:, 0], pos[:, 1], c=masses.numpy(), cmap="viridis", s=30)
        plt.colorbar(sc, ax=ax, label="Mass")
        ax.set_title(f"Two Touching Ellipses (δ={delta:.3f}, τ={tau:.3f})\nTotal: {masses.sum():.3f}, Expected: {expected:.3f}")
        ax.set_aspect("equal")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Contact line")
        ax.legend()

        output_path = OUTPUT_DIR / "two_ellipses_touching_masses.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()
