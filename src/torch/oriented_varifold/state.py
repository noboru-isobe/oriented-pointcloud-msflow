"""State representation for oriented point cloud varifolds."""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class OrientedPointCloudVarifold:
    """
    Oriented point cloud varifold representation.

    Unlike unoriented varifolds where normals are estimated from local geometry,
    this representation explicitly stores the normal angle θ_i as an optimization
    variable, enabling correct winding number computation via line segment angle sum.

    Attributes:
        positions: (N, 2) tensor of point positions, requires_grad for optimization
        angles: (N,) tensor of normal angles in radians, requires_grad for optimization
    """
    positions: torch.Tensor  # (N, 2)
    angles: torch.Tensor     # (N,)

    def __post_init__(self):
        """Validate tensor shapes."""
        assert self.positions.dim() == 2, f"positions must be 2D, got {self.positions.dim()}D"
        assert self.positions.shape[1] == 2, f"positions must have shape (N, 2), got {self.positions.shape}"
        assert self.angles.dim() == 1, f"angles must be 1D, got {self.angles.dim()}D"
        assert self.positions.shape[0] == self.angles.shape[0], (
            f"Number of positions ({self.positions.shape[0]}) must match "
            f"number of angles ({self.angles.shape[0]})"
        )

    @property
    def n_points(self) -> int:
        """Number of points in the varifold."""
        return self.positions.shape[0]

    @property
    def device(self) -> torch.device:
        """Device of the tensors."""
        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the tensors."""
        return self.positions.dtype

    @property
    def normals(self) -> torch.Tensor:
        """
        Unit normal vectors computed from angles.

        Returns:
            (N, 2) tensor of unit normals: n_i = (cos θ_i, sin θ_i)
        """
        return torch.stack([torch.cos(self.angles), torch.sin(self.angles)], dim=1)

    @property
    def tangents(self) -> torch.Tensor:
        """
        Unit tangent vectors computed from angles.

        The tangent is obtained by rotating the normal by 90 degrees:
        t = R @ n where R = [[0, -1], [1, 0]]

        Returns:
            (N, 2) tensor of unit tangents: t_i = (-sin θ_i, cos θ_i)
        """
        return torch.stack([-torch.sin(self.angles), torch.cos(self.angles)], dim=1)

    def clone(self) -> "OrientedPointCloudVarifold":
        """Create a deep copy of this varifold."""
        return OrientedPointCloudVarifold(
            positions=self.positions.clone(),
            angles=self.angles.clone(),
        )

    def detach(self) -> "OrientedPointCloudVarifold":
        """Create a detached copy (no gradient tracking)."""
        return OrientedPointCloudVarifold(
            positions=self.positions.detach(),
            angles=self.angles.detach(),
        )

    def requires_grad_(self, requires_grad: bool = True) -> "OrientedPointCloudVarifold":
        """Set requires_grad for both positions and angles."""
        self.positions.requires_grad_(requires_grad)
        self.angles.requires_grad_(requires_grad)
        return self

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "OrientedPointCloudVarifold":
        """Move varifold to specified device and/or dtype."""
        return OrientedPointCloudVarifold(
            positions=self.positions.to(device=device, dtype=dtype),
            angles=self.angles.to(device=device, dtype=dtype),
        )

    def get_flat_params(self) -> torch.Tensor:
        """
        Flatten positions and angles into a single parameter vector.

        Useful for optimization with L-BFGS or other optimizers.

        Returns:
            (3N,) tensor: [x_0, y_0, x_1, y_1, ..., x_{N-1}, y_{N-1}, θ_0, ..., θ_{N-1}]
        """
        return torch.cat([self.positions.flatten(), self.angles])

    @staticmethod
    def from_flat_params(
        params: torch.Tensor,
        n_points: int,
    ) -> "OrientedPointCloudVarifold":
        """
        Reconstruct varifold from flattened parameter vector.

        Args:
            params: (3N,) tensor from get_flat_params()
            n_points: number of points N

        Returns:
            Reconstructed OrientedPointCloudVarifold
        """
        positions = params[:2 * n_points].reshape(n_points, 2)
        angles = params[2 * n_points:]
        return OrientedPointCloudVarifold(positions=positions, angles=angles)
