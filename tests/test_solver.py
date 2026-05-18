"""Tests for MM solver module."""

import pytest
import torch
import math
from pathlib import Path

import matplotlib.pyplot as plt

from src.torch.solver import (
    MMConfig,
    MMStepResult,
    EvolutionHistory,
    mm_step,
    MMSolver,
    compute_volume,
    compute_volume_divergence,
    create_history,
)
from src.torch.oriented_varifold.mass import compute_masses, compute_recommended_params
from src.torch.shapes import (
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_flower,
    generate_oriented_star,
    generate_oriented_two_ellipses,
    generate_oriented_rectangle,
)


OUTPUT_DIR = Path(__file__).parent / "outputs" / "solver"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


class TestMMConfig:
    """Tests for MMConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MMConfig()
        assert config.time_step == 1e-4
        assert config.optimizer_method == "bfgs"
        assert config.bem_method == "point"

    def test_custom_config(self):
        """Test custom configuration."""
        config = MMConfig(
            time_step=0.005,
            perimeter_c_sigma=4.0,
        )
        assert config.time_step == 0.005
        assert config.perimeter_c_sigma == 4.0


class TestComputeVolume:
    """Tests for volume computation using shoelace formula."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_volume_circle(self, device):
        """Test volume computation for a circle."""
        n_points = 64
        radius = 1.0

        varifold = generate_oriented_circle(n_points, radius=radius, device=device)

        volume = compute_volume(varifold)

        expected = math.pi * radius ** 2
        rel_error = abs(volume - expected) / expected

        assert rel_error < 0.05, f"Volume {volume} far from expected {expected}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_volume_ellipse(self, device):
        """Test volume computation for an ellipse."""
        n_points = 64
        a, b = 1.5, 0.8

        varifold = generate_oriented_ellipse(n_points, a=a, b=b, device=device)

        volume = compute_volume(varifold)

        expected = math.pi * a * b
        rel_error = abs(volume - expected) / expected

        assert rel_error < 0.1, f"Volume {volume} far from expected {expected}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_volume_two_ellipses(self, device):
        """Test volume computation for two ellipses (single connected component only)."""
        n_points = 48

        # Use single ellipse since shoelace assumes single connected boundary
        varifold = generate_oriented_ellipse(n_points, a=1.0, b=0.4, device=device)

        volume = compute_volume(varifold)

        expected = math.pi * 1.0 * 0.4
        rel_error = abs(volume - expected) / expected

        assert rel_error < 0.15, f"Volume {volume} far from expected {expected}"


class TestComputeVolumeDivergence:
    """Tests for volume computation using divergence theorem."""

    def _compute_masses(self, varifold):
        delta, tau = compute_recommended_params(varifold.positions)
        return compute_masses(varifold.positions, delta, tau)

    @pytest.mark.parametrize("device", DEVICES)
    def test_volume_divergence_circle(self, device):
        """Test divergence theorem volume for a circle."""
        n_points = 64
        radius = 1.0
        varifold = generate_oriented_circle(n_points, radius=radius, device=device)
        masses = self._compute_masses(varifold)

        volume = compute_volume_divergence(varifold, masses)
        expected = math.pi * radius ** 2
        rel_error = abs(volume - expected) / expected

        assert rel_error < 0.05, f"Volume {volume} far from expected {expected}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_volume_divergence_ellipse(self, device):
        """Test divergence theorem volume for an ellipse."""
        n_points = 64
        a, b = 1.5, 0.8
        varifold = generate_oriented_ellipse(n_points, a=a, b=b, device=device)
        masses = self._compute_masses(varifold)

        volume = compute_volume_divergence(varifold, masses)
        expected = math.pi * a * b
        rel_error = abs(volume - expected) / expected

        assert rel_error < 0.1, f"Volume {volume} far from expected {expected}"

    @pytest.mark.parametrize("device", DEVICES)
    def test_volume_divergence_two_ellipses(self, device):
        """Test divergence theorem volume for two separate ellipses.

        Unlike shoelace, divergence theorem handles multi-component
        boundaries without splitting.
        """
        varifold = generate_oriented_two_ellipses(
            n_per_ellipse=48, a1=0.4, b1=1.0, a2=0.4, b2=1.0,
            center1=(-0.5, 0.0), center2=(0.5, 0.0), device=device,
        )
        masses = self._compute_masses(varifold)

        volume = compute_volume_divergence(varifold, masses)
        expected = math.pi * 0.4 * 1.0 + math.pi * 0.4 * 1.0
        rel_error = abs(volume - expected) / expected

        assert rel_error < 0.15, f"Volume {volume} far from expected {expected}"


class TestMMStep:
    """Tests for single MM step."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_mm_step_circle(self, device):
        """Test that mm_step runs on circle."""
        n_points = 32
        radius = 1.0

        varifold = generate_oriented_circle(n_points, radius=radius, device=device)

        config = MMConfig(
            time_step=0.01,
            optimizer_max_iter=10,
        )

        result = mm_step(varifold, config)

        assert isinstance(result, MMStepResult)
        assert result.varifold.n_points == n_points
        assert result.perimeter > 0
        assert result.objective > 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_mm_step_ellipse(self, device):
        """Test mm_step on ellipse."""
        n_points = 32
        a, b = 1.5, 0.8

        varifold = generate_oriented_ellipse(n_points, a=a, b=b, device=device)

        config = MMConfig(
            time_step=0.01,
            optimizer_max_iter=10,
        )

        result = mm_step(varifold, config)

        assert result.perimeter > 0
        assert result.objective > 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_mm_step_flower(self, device):
        """Test mm_step on flower shape."""
        n_points = 48

        varifold = generate_oriented_flower(n_points, device=device)

        config = MMConfig(
            time_step=0.01,
            optimizer_max_iter=10,
        )

        result = mm_step(varifold, config)

        assert result.perimeter > 0
        assert result.objective > 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_mm_step_star(self, device):
        """Test mm_step on star shape."""
        n_points = 48

        varifold = generate_oriented_star(n_points, device=device)

        config = MMConfig(
            time_step=0.01,
            optimizer_max_iter=10,
        )

        result = mm_step(varifold, config)

        assert result.perimeter > 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_mm_step_rectangle(self, device):
        """Test mm_step on rounded rectangle."""
        n_points = 32

        varifold = generate_oriented_rectangle(
            n_points, width=2.0, height=1.0, corner_radius=0.1, device=device
        )

        config = MMConfig(
            time_step=0.01,
            optimizer_max_iter=10,
        )

        result = mm_step(varifold, config)

        assert result.perimeter > 0


class TestCreateHistory:
    """Tests for history creation."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_create_history_shapes(self, device):
        """Test that history scalar tensors have correct shapes and varifold list starts empty."""
        n_steps = 10
        dtype = torch.float32

        history = create_history(n_steps, device, dtype)

        # _varifolds is a list (supports variable N after dead point removal); empty at creation
        assert history._varifolds == []
        # Scalar histories pre-allocated
        assert history.perimeters.shape == (n_steps,)
        assert history.wassersteins.shape == (n_steps,)
        assert history.objectives.shape == (n_steps,)
        assert history.volumes.shape == (n_steps + 1,)
        assert history.converged.shape == (n_steps,)
        assert history.converged.dtype == torch.bool
        assert history.n_completed == 0
        assert history.perimeters.device.type == device


class TestMMSolver:
    """Tests for full MM solver."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_solver_circle(self, device):
        """Test solver on circle."""
        n_points = 32
        radius = 1.0

        varifold = generate_oriented_circle(n_points, radius=radius, device=device)

        config = MMConfig(
            time_step=0.01,
            optimizer_max_iter=10,
        )

        solver = MMSolver(config)
        history = solver.solve(varifold, n_steps=3)

        assert history.n_completed == 3
        assert len(history) == 4

        assert (history.perimeters[:3] > 0).all()

        vol_init = history.volumes[0].item()
        vol_final = history.volumes[3].item()
        vol_change = abs(vol_final - vol_init) / vol_init
        assert vol_change < 0.2, f"Volume changed by {vol_change*100:.1f}%"

    @pytest.mark.parametrize("device", DEVICES)
    def test_solver_ellipse(self, device):
        """Test solver on ellipse."""
        n_points = 32
        a, b = 1.5, 0.8

        varifold = generate_oriented_ellipse(n_points, a=a, b=b, device=device)

        config = MMConfig(
            time_step=0.01,
            optimizer_max_iter=10,
        )

        solver = MMSolver(config)
        history = solver.solve(varifold, n_steps=3)

        assert history.n_completed == 3
        assert (history.perimeters[:3] > 0).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_solver_flower(self, device):
        """Test solver on flower shape (should become more circular)."""
        n_points = 48

        varifold = generate_oriented_flower(n_points, device=device)

        config = MMConfig(
            time_step=0.01,
            optimizer_max_iter=10,
        )

        solver = MMSolver(config)
        history = solver.solve(varifold, n_steps=3)

        assert history.n_completed == 3
        assert (history.perimeters[:3] > 0).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_solver_star(self, device):
        """Test solver on star shape."""
        n_points = 48

        varifold = generate_oriented_star(n_points, device=device)

        config = MMConfig(
            time_step=0.01,
            optimizer_max_iter=10,
        )

        solver = MMSolver(config)
        history = solver.solve(varifold, n_steps=3)

        assert history.n_completed == 3
        assert (history.perimeters[:3] > 0).all()

    @pytest.mark.parametrize("device", DEVICES)
    def test_solver_with_callback(self, device):
        """Test solver with callback function."""
        n_points = 32
        radius = 1.0

        varifold = generate_oriented_circle(n_points, radius=radius, device=device)

        config = MMConfig(
            time_step=0.01,
            optimizer_max_iter=5,
        )

        callback_count = [0]

        def callback(step, _result):
            callback_count[0] += 1
            return step >= 1  # Stop after step 1

        solver = MMSolver(config)
        history = solver.solve(varifold, n_steps=10, callback=callback)

        assert history.n_completed == 2
        assert callback_count[0] == 2


class TestVisualization:
    """Visualization tests for solver."""

    def test_visualize_circle_evolution(self):
        """Visualize circle evolution."""
        device = "cpu"
        n_points = 48
        radius = 1.0

        varifold = generate_oriented_circle(n_points, radius=radius, device=device)

        config = MMConfig(
            time_step=0.01,
            optimizer_max_iter=20,
        )

        solver = MMSolver(config)
        history = solver.solve(varifold, n_steps=3)

        _, axes = plt.subplots(1, 4, figsize=(16, 4))

        for i, ax in enumerate(axes):
            v = history.get_varifold(i)
            pos = v.positions.cpu().numpy()
            normals = v.normals.cpu().numpy()

            ax.scatter(pos[:, 0], pos[:, 1], c="blue", s=20)
            ax.quiver(pos[:, 0], pos[:, 1], normals[:, 0], normals[:, 1],
                      color="red", scale=20, width=0.005)

            if i == 0:
                ax.set_title("Initial")
            else:
                P = history.perimeters[i-1].item()
                ax.set_title(f"Step {i}, P={P:.3f}")

            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "circle_evolution.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()

    def test_visualize_flower_evolution(self):
        """Visualize flower evolution (should become more circular)."""
        device = "cpu"
        n_points = 64

        varifold = generate_oriented_flower(n_points, device=device)

        config = MMConfig(
            time_step=0.01,
            optimizer_max_iter=30,
        )

        solver = MMSolver(config)
        history = solver.solve(varifold, n_steps=3)

        _, axes = plt.subplots(1, 4, figsize=(16, 4))

        for i, ax in enumerate(axes):
            v = history.get_varifold(i)
            pos = v.positions.cpu().numpy()
            normals = v.normals.cpu().numpy()

            ax.scatter(pos[:, 0], pos[:, 1], c="blue", s=15)
            ax.quiver(pos[:, 0], pos[:, 1], normals[:, 0], normals[:, 1],
                      color="red", scale=25, width=0.004)

            if i == 0:
                ax.set_title("Initial (Flower)")
            else:
                P = history.perimeters[i-1].item()
                ax.set_title(f"Step {i}, P={P:.3f}")

            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-2.0, 2.0)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "flower_evolution.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()

    def test_visualize_star_evolution(self):
        """Visualize star evolution."""
        device = "cpu"
        n_points = 64

        varifold = generate_oriented_star(n_points, device=device)

        config = MMConfig(
            time_step=0.01,
            optimizer_max_iter=30,
        )

        solver = MMSolver(config)
        history = solver.solve(varifold, n_steps=3)

        _, axes = plt.subplots(1, 4, figsize=(16, 4))

        for i, ax in enumerate(axes):
            v = history.get_varifold(i)
            pos = v.positions.cpu().numpy()
            normals = v.normals.cpu().numpy()

            ax.scatter(pos[:, 0], pos[:, 1], c="blue", s=15)
            ax.quiver(pos[:, 0], pos[:, 1], normals[:, 0], normals[:, 1],
                      color="red", scale=25, width=0.004)

            if i == 0:
                ax.set_title("Initial (Star)")
            else:
                P = history.perimeters[i-1].item()
                ax.set_title(f"Step {i}, P={P:.3f}")

            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-2.0, 2.0)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "star_evolution.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()

    def test_visualize_perimeter_volume(self):
        """Visualize perimeter and volume over time for flower."""
        device = "cpu"
        n_points = 64

        varifold = generate_oriented_flower(n_points, device=device)

        config = MMConfig(
            time_step=0.01,
            optimizer_max_iter=30,
        )

        solver = MMSolver(config)
        history = solver.solve(varifold, n_steps=5)

        _, axes = plt.subplots(1, 2, figsize=(10, 4))

        steps = range(history.n_completed)
        perimeters = history.perimeters[:history.n_completed].cpu().numpy()
        volumes = history.volumes[:history.n_completed + 1].cpu().numpy()

        axes[0].plot(steps, perimeters, "b.-")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Perimeter")
        axes[0].set_title("Perimeter Evolution (Flower)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(range(len(volumes)), volumes, "g.-")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Volume")
        axes[1].set_title("Volume Evolution (Flower)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "flower_perimeter_volume.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()
