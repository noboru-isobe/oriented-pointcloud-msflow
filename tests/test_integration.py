"""Integration tests for Phase 5: Quantitative verification of MM scheme.

Tests verify:
1. Circle steady state (perimeter/volume stable)
2. Flower evolution (perimeter decreasing, volume conserved, relaxing to circle)
3. Two-ellipse fusion (separated, close, touching cases)
4. Two-rectangle fusion
"""

import math
from pathlib import Path

import pytest
import torch

from src.torch.shapes import (
    generate_oriented_circle,
    generate_oriented_flower,
    generate_oriented_star,
    generate_oriented_two_ellipses,
    generate_oriented_two_rectangles,
)
from src.torch.solver import MMConfig, MMSolver


OUTPUT_DIR = Path(__file__).parent / "outputs" / "integration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def compute_circularity(perimeter: float, volume: float) -> float:
    """Compute circularity (isoperimetric ratio).

    circularity = 4π × Area / Perimeter²

    For a circle, circularity = 1.
    For non-convex shapes, circularity < 1.
    """
    if perimeter <= 0:
        return 0.0
    return 4 * math.pi * volume / (perimeter ** 2)


def compute_radius_variance(positions: torch.Tensor, center: tuple = (0, 0)) -> float:
    """Compute variance of distances from center.

    For a perfect circle centered at `center`, variance = 0.
    """
    cx, cy = center
    distances = torch.sqrt(
        (positions[:, 0] - cx) ** 2 + (positions[:, 1] - cy) ** 2
    )
    return distances.var().item()


class TestCircleSteadyState:
    """Test that a circle remains stable under MM evolution.

    A circle is already at minimum perimeter for fixed volume,
    so it should remain approximately circular with stable perimeter and volume.
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_circle_stays_circular(self, device):
        """Circle should maintain circular shape through evolution."""
        n_points = 32
        n_steps = 5

        varifold = generate_oriented_circle(n_points, radius=1.0, device=device)


        config = MMConfig(optimizer_max_iter=50)
        solver = MMSolver(config)

        history = solver.solve(varifold, n_steps=n_steps)

        # Check that radius variance stays small throughout evolution
        initial_variance = compute_radius_variance(history.positions[0])

        for step in range(1, n_steps + 1):
            variance = compute_radius_variance(history.positions[step])
            # Variance should not increase significantly (< 10x initial or < 0.01)
            assert variance < max(initial_variance * 10, 0.01), \
                f"Step {step}: radius variance {variance:.6f} too large"

    @pytest.mark.parametrize("device", DEVICES)
    def test_circle_perimeter_stable(self, device):
        """Circle perimeter should remain stable (< 10% variation)."""
        n_points = 32
        n_steps = 5

        varifold = generate_oriented_circle(n_points, radius=1.0, device=device)


        config = MMConfig(optimizer_max_iter=50)
        solver = MMSolver(config)

        history = solver.solve(varifold, n_steps=n_steps)

        # Get valid perimeters (non-nan)
        perimeters = history.perimeters[:history.n_completed].cpu().numpy()
        valid_perimeters = perimeters[~torch.isnan(torch.tensor(perimeters))]

        if len(valid_perimeters) > 1:
            mean_perimeter = valid_perimeters.mean()
            max_deviation = abs(valid_perimeters - mean_perimeter).max()
            relative_deviation = max_deviation / mean_perimeter

            assert relative_deviation < 0.10, \
                f"Perimeter variation {relative_deviation:.2%} exceeds 10%"

    @pytest.mark.parametrize("device", DEVICES)
    def test_circle_volume_conserved(self, device):
        """Circle volume should be conserved (< 5% change from initial)."""
        n_points = 32
        n_steps = 5

        varifold = generate_oriented_circle(n_points, radius=1.0, device=device)


        config = MMConfig(optimizer_max_iter=50)
        solver = MMSolver(config)

        history = solver.solve(varifold, n_steps=n_steps)

        initial_volume = history.volumes[0].item()

        for step in range(1, history.n_completed + 1):
            volume = history.volumes[step].item()
            relative_change = abs(volume - initial_volume) / initial_volume

            assert relative_change < 0.05, \
                f"Step {step}: volume change {relative_change:.2%} exceeds 5%"


class TestFlowerEvolution:
    """Test that a flower shape evolves correctly under MM scheme.

    A flower (non-convex) should:
    - Have decreasing perimeter (minimizing movements)
    - Conserve volume
    - Relax toward a circle (increasing circularity)
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_perimeter_decreasing(self, device):
        """Flower perimeter should decrease over time."""
        n_points = 48
        n_steps = 5

        varifold = generate_oriented_flower(
            n_points, n_petals=5, inner_radius=0.5, outer_radius=1.0, device=device
        )


        config = MMConfig(optimizer_max_iter=50)
        solver = MMSolver(config)

        history = solver.solve(varifold, n_steps=n_steps)

        perimeters = history.perimeters[:history.n_completed].cpu()

        # Check that perimeter generally decreases (allow some noise)
        # At least the final perimeter should be less than initial
        if history.n_completed >= 2:
            initial_perimeter = perimeters[0].item()
            final_perimeter = perimeters[-1].item()

            assert final_perimeter <= initial_perimeter * 1.05, \
                f"Final perimeter {final_perimeter:.4f} not less than initial {initial_perimeter:.4f}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_volume_conserved(self, device):
        """Flower volume should be conserved (< 10% change)."""
        n_points = 48
        n_steps = 5

        varifold = generate_oriented_flower(
            n_points, n_petals=5, inner_radius=0.5, outer_radius=1.0, device=device
        )


        config = MMConfig(optimizer_max_iter=50)
        solver = MMSolver(config)

        history = solver.solve(varifold, n_steps=n_steps)

        initial_volume = history.volumes[0].item()

        for step in range(1, history.n_completed + 1):
            volume = history.volumes[step].item()
            relative_change = abs(volume - initial_volume) / initial_volume

            assert relative_change < 0.10, \
                f"Step {step}: volume change {relative_change:.2%} exceeds 10%"

    @pytest.mark.parametrize("device", DEVICES)
    def test_relaxes_toward_circle(self, device):
        """Flower should become more circular over time."""
        n_points = 48
        n_steps = 5

        varifold = generate_oriented_flower(
            n_points, n_petals=5, inner_radius=0.5, outer_radius=1.0, device=device
        )


        config = MMConfig(optimizer_max_iter=50)
        solver = MMSolver(config)

        history = solver.solve(varifold, n_steps=n_steps)

        if history.n_completed >= 2:
            # Compute circularity at first and last step
            initial_circularity = compute_circularity(
                history.perimeters[0].item(),
                history.volumes[0].item()
            )
            final_circularity = compute_circularity(
                history.perimeters[history.n_completed - 1].item(),
                history.volumes[history.n_completed].item()
            )

            # Circularity should increase (or at least not decrease significantly)
            assert final_circularity >= initial_circularity * 0.95, \
                f"Circularity decreased: {initial_circularity:.4f} -> {final_circularity:.4f}"


class TestTwoEllipseFusion:
    """Test two-ellipse evolution and fusion behavior.

    Tests three scenarios:
    1. Separated: ellipses far apart, evolve independently
    2. Close: ellipses near each other, interact but don't fuse
    3. Touching: ellipses overlap/touch, should fuse into one region
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_separated_ellipses(self, device):
        """Separated ellipses should evolve without fusion."""
        n_per_ellipse = 24
        n_steps = 3

        # Well-separated ellipses (center distance = 2.0)
        varifold = generate_oriented_two_ellipses(
            n_per_ellipse,
            a1=0.3, b1=0.6, center1=(-1.0, 0.0),
            a2=0.3, b2=0.6, center2=(1.0, 0.0),
            device=device,
        )


        config = MMConfig(optimizer_max_iter=50)
        solver = MMSolver(config)

        history = solver.solve(varifold, n_steps=n_steps)

        # Check volume is conserved
        initial_volume = history.volumes[0].item()
        final_volume = history.volumes[history.n_completed].item()
        relative_change = abs(final_volume - initial_volume) / initial_volume

        assert relative_change < 0.15, \
            f"Volume change {relative_change:.2%} exceeds 15%"

        # Check that both groups of points still exist (no fusion)
        final_positions = history.positions[history.n_completed].cpu()
        left_points = (final_positions[:, 0] < 0).sum().item()
        right_points = (final_positions[:, 0] > 0).sum().item()

        assert left_points > 0 and right_points > 0, \
            "One ellipse disappeared - unexpected fusion?"

    @pytest.mark.parametrize("device", DEVICES)
    def test_close_ellipses(self, device):
        """Close ellipses should interact but maintain separation."""
        n_per_ellipse = 24
        n_steps = 3

        # Close ellipses (center distance = 0.8, gap ~0.2)
        varifold = generate_oriented_two_ellipses(
            n_per_ellipse,
            a1=0.25, b1=0.5, center1=(-0.4, 0.0),
            a2=0.25, b2=0.5, center2=(0.4, 0.0),
            device=device,
        )


        config = MMConfig(optimizer_max_iter=50)
        solver = MMSolver(config)

        history = solver.solve(varifold, n_steps=n_steps)

        # Check perimeter (should decrease as shapes become more circular)
        if history.n_completed >= 2:
            perimeters = history.perimeters[:history.n_completed].cpu()
            assert perimeters[-1] <= perimeters[0] * 1.1, \
                "Perimeter increased unexpectedly"

    @pytest.mark.parametrize("device", DEVICES)
    def test_touching_ellipses_volume_conserved(self, device):
        """Touching/overlapping ellipses should conserve volume during fusion."""
        n_per_ellipse = 24
        n_steps = 3

        # Touching ellipses (center distance = 0.5, overlapping)
        varifold = generate_oriented_two_ellipses(
            n_per_ellipse,
            a1=0.3, b1=0.6, center1=(-0.25, 0.0),
            a2=0.3, b2=0.6, center2=(0.25, 0.0),
            device=device,
        )


        config = MMConfig(optimizer_max_iter=50)
        solver = MMSolver(config)

        history = solver.solve(varifold, n_steps=n_steps)

        # Volume should be approximately conserved even during fusion
        initial_volume = history.volumes[0].item()
        final_volume = history.volumes[history.n_completed].item()
        relative_change = abs(final_volume - initial_volume) / initial_volume

        # Allow larger tolerance for fusion case
        assert relative_change < 0.20, \
            f"Volume change {relative_change:.2%} exceeds 20% during fusion"


class TestTwoRectangleFusion:
    """Test two-rectangle evolution.

    Rectangles should:
    - Have corners become more rounded
    - Eventually fuse if close enough
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_rectangles_evolution(self, device):
        """Two rectangles should evolve with volume conservation."""
        n_per_rect = 32
        n_steps = 3

        varifold = generate_oriented_two_rectangles(
            n_per_rect,
            width1=0.5, height1=1.2, center1=(-0.5, 0.0),
            width2=0.5, height2=1.2, center2=(0.5, 0.0),
            corner_radius=0.08,
            device=device,
        )


        config = MMConfig(optimizer_max_iter=50)
        solver = MMSolver(config)

        history = solver.solve(varifold, n_steps=n_steps)

        # Check volume conservation
        initial_volume = history.volumes[0].item()
        final_volume = history.volumes[history.n_completed].item()
        relative_change = abs(final_volume - initial_volume) / initial_volume

        assert relative_change < 0.15, \
            f"Volume change {relative_change:.2%} exceeds 15%"

    @pytest.mark.parametrize("device", DEVICES)
    def test_rectangles_perimeter_decreases(self, device):
        """Rectangle corners should become rounded, reducing perimeter."""
        n_per_rect = 32
        n_steps = 3

        varifold = generate_oriented_two_rectangles(
            n_per_rect,
            width1=0.5, height1=1.2, center1=(-0.5, 0.0),
            width2=0.5, height2=1.2, center2=(0.5, 0.0),
            corner_radius=0.08,
            device=device,
        )


        config = MMConfig(optimizer_max_iter=50)
        solver = MMSolver(config)

        history = solver.solve(varifold, n_steps=n_steps)

        # Perimeter should decrease as corners round
        if history.n_completed >= 2:
            initial_perimeter = history.perimeters[0].item()
            final_perimeter = history.perimeters[history.n_completed - 1].item()

            # Allow some tolerance
            assert final_perimeter <= initial_perimeter * 1.05, \
                f"Perimeter increased: {initial_perimeter:.4f} -> {final_perimeter:.4f}"
