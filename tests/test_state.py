"""Tests for OrientedPointCloudVarifold."""

import pytest
import torch
import math
from pathlib import Path

import matplotlib.pyplot as plt

from src.torch.oriented_varifold import OrientedPointCloudVarifold


# Output directory for test visualizations
OUTPUT_DIR = Path(__file__).parent / "outputs" / "state"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


class TestOrientedPointCloudVarifold:
    """Tests for OrientedPointCloudVarifold."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_creation(self, device):
        """Test basic varifold creation."""
        n = 10
        positions = torch.randn(n, 2, device=device)
        angles = torch.randn(n, device=device)

        varifold = OrientedPointCloudVarifold(positions=positions, angles=angles)

        assert varifold.n_points == n
        assert varifold.device.type == device
        assert varifold.positions.shape == (n, 2)
        assert varifold.angles.shape == (n,)

    @pytest.mark.parametrize("device", DEVICES)
    def test_normals_unit_circle(self, device):
        """Test that normals are correctly computed for a unit circle."""
        n = 4
        t = torch.linspace(0, 2 * math.pi, n + 1, device=device)[:-1]
        positions = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
        angles = t.clone()

        varifold = OrientedPointCloudVarifold(positions=positions, angles=angles)
        normals = varifold.normals

        # For unit circle, normals should equal positions
        assert torch.allclose(normals, positions, atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_tangents_perpendicular_to_normals(self, device):
        """Test that tangents are perpendicular to normals."""
        n = 100
        angles = torch.linspace(0, 2 * math.pi, n, device=device)
        positions = torch.randn(n, 2, device=device)

        varifold = OrientedPointCloudVarifold(positions=positions, angles=angles)

        dot_products = (varifold.normals * varifold.tangents).sum(dim=1)
        assert torch.allclose(dot_products, torch.zeros_like(dot_products), atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_normals_are_unit_vectors(self, device):
        """Test that normals have unit length."""
        n = 100
        angles = torch.randn(n, device=device)
        positions = torch.randn(n, 2, device=device)

        varifold = OrientedPointCloudVarifold(positions=positions, angles=angles)
        norms = varifold.normals.norm(dim=1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_tangents_are_unit_vectors(self, device):
        """Test that tangents have unit length."""
        n = 100
        angles = torch.randn(n, device=device)
        positions = torch.randn(n, 2, device=device)

        varifold = OrientedPointCloudVarifold(positions=positions, angles=angles)
        norms = varifold.tangents.norm(dim=1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_flat_params_roundtrip(self, device):
        """Test flattening and reconstruction."""
        n = 10
        positions = torch.randn(n, 2, device=device)
        angles = torch.randn(n, device=device)

        varifold = OrientedPointCloudVarifold(positions=positions, angles=angles)
        flat = varifold.get_flat_params()

        assert flat.shape == (3 * n,)

        reconstructed = OrientedPointCloudVarifold.from_flat_params(flat, n)

        assert torch.allclose(reconstructed.positions, positions)
        assert torch.allclose(reconstructed.angles, angles)

    @pytest.mark.parametrize("device", DEVICES)
    def test_clone(self, device):
        """Test clone creates independent copy."""
        positions = torch.randn(5, 2, device=device)
        angles = torch.randn(5, device=device)

        varifold = OrientedPointCloudVarifold(positions=positions, angles=angles)
        cloned = varifold.clone()

        # Modify original
        varifold.positions[0, 0] = 999.0

        # Clone should be unchanged
        assert cloned.positions[0, 0] != 999.0

    @pytest.mark.parametrize("device", DEVICES)
    def test_requires_grad(self, device):
        """Test requires_grad_ method."""
        positions = torch.randn(5, 2, device=device)
        angles = torch.randn(5, device=device)

        varifold = OrientedPointCloudVarifold(positions=positions, angles=angles)
        varifold.requires_grad_(True)

        assert varifold.positions.requires_grad
        assert varifold.angles.requires_grad

    @pytest.mark.parametrize("device", DEVICES)
    def test_to_device(self, device):
        """Test moving to different device."""
        positions = torch.randn(5, 2)
        angles = torch.randn(5)

        varifold = OrientedPointCloudVarifold(positions=positions, angles=angles)
        moved = varifold.to(device=device)

        assert moved.device.type == device

    def test_validation_shape_mismatch(self):
        """Test that mismatched shapes raise error."""
        positions = torch.randn(5, 2)
        angles = torch.randn(3)  # Wrong size

        with pytest.raises(AssertionError):
            OrientedPointCloudVarifold(positions=positions, angles=angles)

    def test_validation_wrong_position_dim(self):
        """Test that wrong position dimensions raise error."""
        positions = torch.randn(5, 3)  # Should be (N, 2)
        angles = torch.randn(5)

        with pytest.raises(AssertionError):
            OrientedPointCloudVarifold(positions=positions, angles=angles)

    def test_visualization_unit_circle(self):
        """Visualize unit circle with normals and tangents."""
        n = 32
        t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        positions = torch.stack([torch.cos(t), torch.sin(t)], dim=1)
        angles = t.clone()

        varifold = OrientedPointCloudVarifold(positions=positions, angles=angles)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Points with normals
        ax1 = axes[0]
        ax1.set_aspect("equal")
        ax1.scatter(positions[:, 0], positions[:, 1], c="blue", s=30)
        scale = 0.2
        for i in range(n):
            ax1.arrow(
                positions[i, 0].item(),
                positions[i, 1].item(),
                scale * varifold.normals[i, 0].item(),
                scale * varifold.normals[i, 1].item(),
                head_width=0.05,
                head_length=0.02,
                fc="red",
                ec="red",
            )
        ax1.set_title("Unit Circle: Points + Normals (red)")
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Points with tangents
        ax2 = axes[1]
        ax2.set_aspect("equal")
        ax2.scatter(positions[:, 0], positions[:, 1], c="blue", s=30)
        for i in range(n):
            ax2.arrow(
                positions[i, 0].item(),
                positions[i, 1].item(),
                scale * varifold.tangents[i, 0].item(),
                scale * varifold.tangents[i, 1].item(),
                head_width=0.05,
                head_length=0.02,
                fc="green",
                ec="green",
            )
        ax2.set_title("Unit Circle: Points + Tangents (green)")
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "varifold_unit_circle.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()

    def test_visualization_ellipse(self):
        """Visualize ellipse with normals and tangents."""
        n = 32
        a, b = 2.0, 1.0  # semi-major and semi-minor axes
        t = torch.linspace(0, 2 * math.pi, n + 1)[:-1]

        # Ellipse parametrization: (a*cos(t), b*sin(t))
        positions = torch.stack([a * torch.cos(t), b * torch.sin(t)], dim=1)

        # Normal angle for ellipse: atan2(a*sin(t), b*cos(t))
        angles = torch.atan2(a * torch.sin(t), b * torch.cos(t))

        varifold = OrientedPointCloudVarifold(positions=positions, angles=angles)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.set_aspect("equal")
        ax.scatter(positions[:, 0], positions[:, 1], c="blue", s=30)

        scale = 0.3
        for i in range(n):
            ax.arrow(
                positions[i, 0].item(),
                positions[i, 1].item(),
                scale * varifold.normals[i, 0].item(),
                scale * varifold.normals[i, 1].item(),
                head_width=0.08,
                head_length=0.03,
                fc="red",
                ec="red",
            )
        ax.set_title(f"Ellipse (a={a}, b={b}): Points + Normals")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "varifold_ellipse.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()

    def test_visualization_two_ellipses(self):
        """Visualize two ellipses with normals (multi-component test)."""
        n_per_ellipse = 24

        # Ellipse 1: center (-0.6, 0), semi-axes (0.4, 1.0) - 縦長
        t1 = torch.linspace(0, 2 * math.pi, n_per_ellipse + 1)[:-1]
        c1 = torch.tensor([-0.6, 0.0])
        a1, b1 = 0.4, 1.0  # a=横, b=縦
        pos1 = torch.stack([c1[0] + a1 * torch.cos(t1), c1[1] + b1 * torch.sin(t1)], dim=1)
        angles1 = torch.atan2(a1 * torch.sin(t1), b1 * torch.cos(t1))

        # Ellipse 2: center (0.6, 0), semi-axes (0.4, 1.0) - 縦長
        t2 = torch.linspace(0, 2 * math.pi, n_per_ellipse + 1)[:-1]
        c2 = torch.tensor([0.6, 0.0])
        a2, b2 = 0.4, 1.0
        pos2 = torch.stack([c2[0] + a2 * torch.cos(t2), c2[1] + b2 * torch.sin(t2)], dim=1)
        angles2 = torch.atan2(a2 * torch.sin(t2), b2 * torch.cos(t2))

        # Concatenate
        positions = torch.cat([pos1, pos2], dim=0)
        angles = torch.cat([angles1, angles2], dim=0)

        varifold = OrientedPointCloudVarifold(positions=positions, angles=angles)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_aspect("equal")
        ax.scatter(positions[:, 0], positions[:, 1], c="blue", s=30)

        scale = 0.2
        for i in range(varifold.n_points):
            ax.arrow(
                positions[i, 0].item(),
                positions[i, 1].item(),
                scale * varifold.normals[i, 0].item(),
                scale * varifold.normals[i, 1].item(),
                head_width=0.04,
                head_length=0.02,
                fc="red",
                ec="red",
            )
        ax.set_title("Two Ellipses (vertical): Points + Normals")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "varifold_two_ellipses.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        assert output_path.exists()
