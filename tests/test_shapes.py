"""Tests for shapes module."""

import pytest
import torch
import math
from pathlib import Path

import matplotlib.pyplot as plt

from src.torch.shapes import (
    generate_oriented_circle,
    generate_oriented_ellipse,
    generate_oriented_flower,
    generate_oriented_star,
    generate_oriented_two_ellipses,
    generate_oriented_two_circles,
    generate_oriented_rectangle,
    generate_oriented_two_rectangles,
)


OUTPUT_DIR = Path(__file__).parent / "outputs" / "shapes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def plot_varifold(ax, varifold, title, scale=0.15):
    """Helper to plot varifold with normals."""
    pos = varifold.positions.cpu()
    normals = varifold.normals.cpu()

    ax.set_aspect("equal")
    ax.scatter(pos[:, 0], pos[:, 1], c="blue", s=20)

    for i in range(varifold.n_points):
        ax.arrow(
            pos[i, 0].item(),
            pos[i, 1].item(),
            scale * normals[i, 0].item(),
            scale * normals[i, 1].item(),
            head_width=0.03,
            head_length=0.015,
            fc="red",
            ec="red",
        )
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


class TestShapeGeneration:
    """Tests for shape generation functions."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_circle_shape(self, device):
        """Test circle generation."""
        v = generate_oriented_circle(32, radius=1.0, center=(0, 0), device=device)
        assert v.n_points == 32
        assert v.device.type == device

        # Check points are on circle
        radii = v.positions.norm(dim=1)
        assert torch.allclose(radii, torch.ones_like(radii), atol=1e-5)

        # Check normals point outward (same as position for unit circle)
        assert torch.allclose(v.normals, v.positions, atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_ellipse_shape(self, device):
        """Test ellipse generation."""
        a, b = 2.0, 1.0
        v = generate_oriented_ellipse(32, a=a, b=b, center=(0, 0), device=device)
        assert v.n_points == 32

        # Check points are on ellipse: (x/a)^2 + (y/b)^2 = 1
        check = (v.positions[:, 0] / a) ** 2 + (v.positions[:, 1] / b) ** 2
        assert torch.allclose(check, torch.ones_like(check), atol=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_ellipses_shape(self, device):
        """Test two ellipses generation."""
        v = generate_oriented_two_ellipses(24, device=device)
        assert v.n_points == 48  # 24 * 2

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_circles_shape(self, device):
        """Test two circles generation."""
        v = generate_oriented_two_circles(24, device=device)
        assert v.n_points == 48

    @pytest.mark.parametrize("device", DEVICES)
    def test_flower_shape(self, device):
        """Test flower generation."""
        v = generate_oriented_flower(64, n_petals=5, device=device)
        assert v.n_points == 64

    @pytest.mark.parametrize("device", DEVICES)
    def test_star_shape(self, device):
        """Test star generation."""
        v = generate_oriented_star(64, n_star_points=5, device=device)
        assert v.n_points == 64

    @pytest.mark.parametrize("device", DEVICES)
    def test_rectangle_shape(self, device):
        """Test rectangle generation."""
        v = generate_oriented_rectangle(48, width=2.0, height=1.0, device=device)
        assert v.n_points > 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_two_rectangles_shape(self, device):
        """Test two rectangles generation."""
        v = generate_oriented_two_rectangles(48, device=device)
        assert v.n_points > 0


class TestShapeVisualization:
    """Visualization tests for shapes."""

    def test_visualization_all_shapes(self):
        """Visualize all basic shapes."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Circle
        v = generate_oriented_circle(32, device="cpu")
        plot_varifold(axes[0, 0], v, "Circle", scale=0.15)
        axes[0, 0].set_xlim(-1.5, 1.5)
        axes[0, 0].set_ylim(-1.5, 1.5)

        # Ellipse
        v = generate_oriented_ellipse(32, a=1.5, b=0.8, device="cpu")
        plot_varifold(axes[0, 1], v, "Ellipse (a=1.5, b=0.8)", scale=0.2)
        axes[0, 1].set_xlim(-2, 2)
        axes[0, 1].set_ylim(-1.5, 1.5)

        # Flower
        v = generate_oriented_flower(64, n_petals=5, device="cpu")
        plot_varifold(axes[0, 2], v, "Flower (5 petals)", scale=0.12)
        axes[0, 2].set_xlim(-1.5, 1.5)
        axes[0, 2].set_ylim(-1.5, 1.5)

        # Star
        v = generate_oriented_star(64, n_star_points=5, device="cpu")
        plot_varifold(axes[1, 0], v, "Star (5 points)", scale=0.1)
        axes[1, 0].set_xlim(-1.5, 1.5)
        axes[1, 0].set_ylim(-1.5, 1.5)

        # Rectangle
        v = generate_oriented_rectangle(48, width=2.0, height=1.2, corner_radius=0.15, device="cpu")
        plot_varifold(axes[1, 1], v, "Rounded Rectangle", scale=0.15)
        axes[1, 1].set_xlim(-1.5, 1.5)
        axes[1, 1].set_ylim(-1, 1)

        # Two Ellipses
        v = generate_oriented_two_ellipses(24, device="cpu")
        plot_varifold(axes[1, 2], v, "Two Ellipses", scale=0.15)
        axes[1, 2].set_xlim(-1.5, 1.5)
        axes[1, 2].set_ylim(-1.5, 1.5)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "shapes_all.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()

    def test_visualization_two_rectangles(self):
        """Visualize two rectangles."""
        v = generate_oriented_two_rectangles(
            48,
            width1=0.6, height1=1.8, center1=(-0.5, 0.0),
            width2=0.6, height2=1.8, center2=(0.5, 0.0),
            corner_radius=0.1,
            device="cpu",
        )

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_varifold(ax, v, "Two Rectangles", scale=0.12)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "shapes_two_rectangles.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()
