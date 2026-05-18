"""Tests for arc length parametrization."""

import pytest
import torch
import numpy as np
import math
from pathlib import Path

from src.torch.shapes import (
    circle,
    ellipse,
    flower,
    star,
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_flower,
    generate_oriented_star,
)
from src.torch.shapes.arclength import (
    compute_arc_length_cumulative,
    verify_arc_length_with_torchquad,
    find_parameters_for_arc_lengths,
    sample_curve_arc_length,
)


class TestSpeedFunctions:
    """Test speed_fn for each curve type."""

    def test_circle_speed_fn(self):
        """Circle speed_fn should return constant radius."""
        radius = 1.5
        c = circle(radius=radius)

        t = torch.linspace(0, 2 * math.pi, 100)
        speeds = c.speed_fn(t)

        assert torch.allclose(speeds, torch.full_like(speeds, radius))
        assert speeds.std() < 1e-6

    def test_ellipse_speed_fn(self):
        """Ellipse speed_fn should match analytical formula."""
        a, b = 1.0, 0.5
        e = ellipse(a=a, b=b)

        t = torch.linspace(0, 2 * math.pi, 100)
        speeds = e.speed_fn(t)

        # Analytical: |γ'(t)| = sqrt(a²sin²t + b²cos²t)
        expected = torch.sqrt(a**2 * torch.sin(t)**2 + b**2 * torch.cos(t)**2)
        assert torch.allclose(speeds, expected, rtol=1e-5)

    def test_flower_speed_fn(self):
        """Flower speed_fn should be positive and vary with position."""
        f = flower(n_petals=5, inner_radius=0.5, outer_radius=1.0)

        t = torch.linspace(0, 2 * math.pi, 100)
        speeds = f.speed_fn(t)

        # Speed should always be positive
        assert (speeds > 0).all()
        # Speed should vary (not constant like circle)
        assert speeds.std() > 0.01

    def test_star_speed_fn(self):
        """Star speed_fn should be positive and vary with position."""
        s = star(n_points=5, inner_radius=0.4, outer_radius=1.0)

        t = torch.linspace(0, 2 * math.pi, 100)
        speeds = s.speed_fn(t)

        # Speed should always be positive
        assert (speeds > 0).all()
        # Speed should vary (not constant like circle)
        assert speeds.std() > 0.01


class TestArcLength:
    """Test arc length computation."""

    def test_circle_arc_length_matches_analytical(self):
        """Circle arc length should be 2πr."""
        radius = 1.5
        c = circle(radius=radius)

        _, _, total_length = compute_arc_length_cumulative(c.speed_fn)
        expected = 2 * math.pi * radius

        assert abs(total_length - expected) < 0.001  # < 0.1% error

    def test_ellipse_arc_length_reasonable(self):
        """Ellipse arc length should be between 2π*min(a,b) and 2π*max(a,b)."""
        a, b = 1.0, 0.5
        e = ellipse(a=a, b=b)

        _, _, total_length = compute_arc_length_cumulative(e.speed_fn)

        # Arc length should be between circle perimeters
        assert total_length > 2 * math.pi * min(a, b)
        assert total_length < 2 * math.pi * max(a, b)

    def test_torchquad_simpson_matches_cumulative(self):
        """Torchquad Simpson and trapezoidal cumulative should agree."""
        f = flower(n_petals=5, inner_radius=0.5, outer_radius=1.0)

        _, _, trap_length = compute_arc_length_cumulative(f.speed_fn, n_integration_points=4096)
        simpson_length = verify_arc_length_with_torchquad(f.speed_fn, n_points=1001)

        relative_error = abs(trap_length - simpson_length) / simpson_length
        assert relative_error < 0.01  # < 1% difference


class TestParameterInversion:
    """Test arc length to parameter inversion."""

    def test_find_parameters_endpoints(self):
        """Inversion should work for s=0 and s=L."""
        c = circle(radius=1.0)
        t_grid, s_cumulative, total_length = compute_arc_length_cumulative(c.speed_fn)

        target_s = torch.tensor([0.0, total_length])
        t_values = find_parameters_for_arc_lengths(t_grid, s_cumulative, target_s)

        assert abs(t_values[0].item()) < 0.01  # t=0 for s=0
        assert abs(t_values[1].item() - 2 * math.pi) < 0.01  # t=2π for s=L

    def test_find_parameters_monotonic(self):
        """Inverted parameters should be monotonically increasing."""
        f = flower(n_petals=5)
        t_grid, s_cumulative, total_length = compute_arc_length_cumulative(f.speed_fn)

        target_s = torch.linspace(0, total_length * 0.99, 50)
        t_values = find_parameters_for_arc_lengths(t_grid, s_cumulative, target_s)

        # Should be monotonically increasing
        assert (t_values[1:] >= t_values[:-1]).all()


class TestUniformity:
    """Test uniformity of arc length sampling."""

    def test_flower_spacing_uniformity(self):
        """Flower arc length sampling should have uniform point spacing (CV < 0.05)."""
        f = flower(n_petals=5, inner_radius=0.5, outer_radius=1.0)
        varifold = sample_curve_arc_length(f, n_points=64, device='cpu')

        positions = varifold.positions.numpy()

        # Compute distances between consecutive points (closed curve)
        # Use roll for proper wrap-around
        next_pos = np.roll(positions, -1, axis=0)
        distances = np.linalg.norm(next_pos - positions, axis=1)

        # Coefficient of variation should be small for uniform spacing
        cv = distances.std() / distances.mean()
        assert cv < 0.05, f"CV={cv:.3f} > 0.05 (spacing not uniform)"

    def test_ellipse_spacing_uniformity(self):
        """Ellipse arc length sampling should have uniform point spacing (CV < 0.05)."""
        e = ellipse(a=1.0, b=0.5)
        varifold = sample_curve_arc_length(e, n_points=64, device='cpu')

        positions = varifold.positions.numpy()
        next_pos = np.roll(positions, -1, axis=0)
        distances = np.linalg.norm(next_pos - positions, axis=1)

        cv = distances.std() / distances.mean()
        assert cv < 0.05, f"CV={cv:.3f} > 0.05 (spacing not uniform)"

    def test_star_spacing_uniformity(self):
        """Star arc length sampling should have uniform point spacing (CV < 0.05)."""
        s = star(n_points=5, inner_radius=0.4, outer_radius=1.0)
        varifold = sample_curve_arc_length(s, n_points=64, device='cpu')

        positions = varifold.positions.numpy()
        next_pos = np.roll(positions, -1, axis=0)
        distances = np.linalg.norm(next_pos - positions, axis=1)

        cv = distances.std() / distances.mean()
        assert cv < 0.05, f"CV={cv:.3f} > 0.05 (spacing not uniform)"

    def test_mass_distribution_uniformity(self):
        """Arc length sampling should produce uniform mass distribution (CV < 0.1)."""
        from src.torch.oriented_varifold.mass import compute_masses, compute_recommended_params

        f = flower(n_petals=5, inner_radius=0.5, outer_radius=1.0)
        varifold = sample_curve_arc_length(f, n_points=64, device='cpu')

        delta, tau = compute_recommended_params(varifold.positions)
        masses = compute_masses(varifold.positions, delta, tau)

        cv = masses.std().item() / masses.mean().item()
        assert cv < 0.1, f"Mass CV={cv:.3f} > 0.1 (mass not uniform)"


class TestGeneratorIntegration:
    """Test generator functions with initial_sampling parameter."""

    def test_generate_oriented_flower_arc_length(self):
        """generate_oriented_flower with initial_sampling="arc_length" should work."""
        v = generate_oriented_flower(n_points=32, initial_sampling="arc_length", device='cpu')
        assert v.positions.shape == (32, 2)
        assert v.angles.shape == (32,)

    def test_generate_oriented_ellipse_arc_length(self):
        """generate_oriented_ellipse with initial_sampling="arc_length" should work."""
        v = generate_oriented_ellipse(n_points=32, a=1.0, b=0.5, initial_sampling="arc_length", device='cpu')
        assert v.positions.shape == (32, 2)
        assert v.angles.shape == (32,)

    def test_generate_oriented_star_arc_length(self):
        """generate_oriented_star with initial_sampling="arc_length" should work."""
        v = generate_oriented_star(n_points=32, initial_sampling="arc_length", device='cpu')
        assert v.positions.shape == (32, 2)
        assert v.angles.shape == (32,)

    def test_backward_compatibility(self):
        """Default initial_sampling="parameter" should preserve existing behavior."""
        # Generate with default (equal parameter)
        v1 = generate_oriented_flower(n_points=32, device='cpu')
        v2 = generate_oriented_flower(n_points=32, initial_sampling="parameter", device='cpu')

        assert torch.allclose(v1.positions, v2.positions)
        assert torch.allclose(v1.angles, v2.angles)


class TestVisualization:
    """Visualization tests (generate comparison figures)."""

    @pytest.mark.slow
    def test_visualize_mass_comparison(self):
        """Generate mass comparison figure: Equal Angle vs Arc Length.

        Outputs to tests/outputs/arclength/mass_comparison.png
        """
        import matplotlib.pyplot as plt
        from src.torch.oriented_varifold.mass import compute_masses, compute_recommended_params
        from src.torch.visualization.static import plot_shape_grid

        output_dir = Path('tests/outputs/arclength')
        output_dir.mkdir(parents=True, exist_ok=True)

        shapes = ['flower', 'ellipse', 'star']
        positions_grid = []
        masses_grid = []
        angles_grid = []

        for shape in shapes:
            row_pos, row_mass, row_angles = [], [], []
            for sampling in ["parameter", "arc_length"]:
                if shape == 'flower':
                    v = generate_oriented_flower(n_points=64, initial_sampling=sampling, device='cpu')
                elif shape == 'ellipse':
                    v = generate_oriented_ellipse(n_points=64, a=1.0, b=0.5, initial_sampling=sampling, device='cpu')
                else:
                    v = generate_oriented_star(n_points=64, initial_sampling=sampling, device='cpu')

                delta, tau = compute_recommended_params(v.positions)
                masses = compute_masses(v.positions, delta, tau)

                row_pos.append(v.positions.numpy())
                row_mass.append(masses.numpy())
                row_angles.append(v.angles.numpy())

            positions_grid.append(row_pos)
            masses_grid.append(row_mass)
            angles_grid.append(row_angles)

        fig = plot_shape_grid(
            positions_grid,
            row_labels=['Flower', 'Ellipse', 'Star'],
            col_labels=['Equal Angle', 'Arc Length'],
            angles_grid=angles_grid,
            masses_grid=masses_grid,
            save_path=output_dir / 'mass_comparison.png',
            figsize=(10, 12),
            show_normals=False,
        )
        plt.close(fig)

        assert (output_dir / 'mass_comparison.png').exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
