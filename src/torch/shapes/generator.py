"""Shape generators for OrientedPointCloudVarifold."""

import torch
import math
from typing import Tuple, Optional

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from .curves import ParametricCurve, circle, ellipse, flower, star


def sample_curve(
    curve: ParametricCurve,
    n_points: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    initial_sampling: str = "parameter",
) -> OrientedPointCloudVarifold:
    """
    Sample points from a parametric curve.

    Args:
        curve: ParametricCurve to sample
        n_points: number of points to sample
        device: torch device
        dtype: torch dtype
        initial_sampling: sampling strategy
            - "parameter": equal parameter spacing (default)
            - "arc_length": arc length parametrization for uniform spacing
            - "mass_uniform": KDE mass-uniform parametrization

    Returns:
        OrientedPointCloudVarifold with sampled points and angles
    """
    if initial_sampling == "mass_uniform":
        if curve.speed_fn is None:
            raise ValueError(f"Curve '{curve.name}' does not support mass_uniform sampling (no speed_fn)")
        from .arclength import sample_curve_mass_uniform
        return sample_curve_mass_uniform(curve, n_points, device, dtype)

    if initial_sampling == "arc_length":
        if curve.speed_fn is None:
            raise ValueError(f"Curve '{curve.name}' does not support arc_length sampling (no speed_fn)")
        from .arclength import sample_curve_arc_length
        return sample_curve_arc_length(curve, n_points, device, dtype)

    # Default: equal angle parametrization
    t = torch.linspace(0, 2 * math.pi, n_points + 1, device=device, dtype=dtype)[:-1]
    x, y = curve.position_fn(t)
    positions = torch.stack([x, y], dim=1)
    angles = curve.normal_angle_fn(t)
    return OrientedPointCloudVarifold(positions=positions, angles=angles)


def generate_oriented_circle(
    n_points: int,
    radius: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> OrientedPointCloudVarifold:
    """Generate oriented point cloud for a circle."""
    return sample_curve(circle(radius, center), n_points, device, dtype)


def generate_oriented_ellipse(
    n_points: int,
    a: float = 1.0,
    b: float = 0.5,
    center: Tuple[float, float] = (0.0, 0.0),
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    initial_sampling: str = "parameter",
) -> OrientedPointCloudVarifold:
    """Generate oriented point cloud for an ellipse."""
    return sample_curve(ellipse(a, b, center), n_points, device, dtype, initial_sampling)


def generate_oriented_flower(
    n_points: int,
    n_petals: int = 5,
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    initial_sampling: str = "parameter",
) -> OrientedPointCloudVarifold:
    """Generate oriented point cloud for a flower shape."""
    return sample_curve(
        flower(n_petals, inner_radius, outer_radius, center),
        n_points,
        device,
        dtype,
        initial_sampling,
    )


def generate_oriented_star(
    n_points: int,
    n_star_points: int = 5,
    inner_radius: float = 0.4,
    outer_radius: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    initial_sampling: str = "parameter",
) -> OrientedPointCloudVarifold:
    """Generate oriented point cloud for a star shape."""
    return sample_curve(
        star(n_star_points, inner_radius, outer_radius, center),
        n_points,
        device,
        dtype,
        initial_sampling,
    )


def generate_oriented_two_ellipses(
    n_per_ellipse: int,
    a1: float = 0.4,
    b1: float = 1.0,
    center1: Tuple[float, float] = (-0.6, 0.0),
    a2: float = 0.4,
    b2: float = 1.0,
    center2: Tuple[float, float] = (0.6, 0.0),
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> OrientedPointCloudVarifold:
    """
    Generate oriented point cloud for two ellipses.

    This is a key test case for multi-component domains.
    Default parameters create two vertical ellipses close together.

    Args:
        n_per_ellipse: points per ellipse
        a1, b1: semi-axes of ellipse 1 (a=x, b=y)
        center1: center of ellipse 1
        a2, b2: semi-axes of ellipse 2
        center2: center of ellipse 2
        device: torch device
        dtype: torch dtype

    Returns:
        OrientedPointCloudVarifold with 2*n_per_ellipse points
    """
    v1 = generate_oriented_ellipse(n_per_ellipse, a1, b1, center1, device, dtype)
    v2 = generate_oriented_ellipse(n_per_ellipse, a2, b2, center2, device, dtype)

    positions = torch.cat([v1.positions, v2.positions], dim=0)
    angles = torch.cat([v1.angles, v2.angles], dim=0)

    return OrientedPointCloudVarifold(positions=positions, angles=angles)


def generate_oriented_two_circles(
    n_per_circle: int,
    radius1: float = 0.8,
    center1: Tuple[float, float] = (-1.0, 0.0),
    radius2: float = 0.8,
    center2: Tuple[float, float] = (1.0, 0.0),
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> OrientedPointCloudVarifold:
    """
    Generate oriented point cloud for two circles.

    Args:
        n_per_circle: points per circle
        radius1, center1: first circle parameters
        radius2, center2: second circle parameters
        device: torch device
        dtype: torch dtype

    Returns:
        OrientedPointCloudVarifold with 2*n_per_circle points
    """
    v1 = generate_oriented_circle(n_per_circle, radius1, center1, device, dtype)
    v2 = generate_oriented_circle(n_per_circle, radius2, center2, device, dtype)

    positions = torch.cat([v1.positions, v2.positions], dim=0)
    angles = torch.cat([v1.angles, v2.angles], dim=0)

    return OrientedPointCloudVarifold(positions=positions, angles=angles)


def generate_oriented_rectangle(
    n_points: int,
    width: float = 2.0,
    height: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
    corner_radius: float = 0.1,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> OrientedPointCloudVarifold:
    """
    Generate oriented point cloud for a rounded rectangle.

    Points are distributed uniformly along the perimeter.
    Traversal order: right edge → top-right corner → top edge → top-left corner →
                     left edge → bottom-left corner → bottom edge → bottom-right corner

    Args:
        n_points: total number of points
        width: rectangle width
        height: rectangle height
        center: center of rectangle
        corner_radius: radius of rounded corners (0 for sharp corners)
        device: torch device
        dtype: torch dtype

    Returns:
        OrientedPointCloudVarifold
    """
    cx, cy = center
    r = corner_radius
    hw = width / 2 - r  # half-width of straight part
    hh = height / 2 - r  # half-height of straight part

    # Perimeter segment lengths
    len_straight_v = 2 * hh  # right and left edges
    len_straight_h = 2 * hw  # top and bottom edges
    len_corner = (math.pi / 2) * r if r > 0 else 0  # each quarter circle
    total_len = 2 * len_straight_v + 2 * len_straight_h + 4 * len_corner

    # Parametrize by arc length s ∈ [0, total_len)
    s_vals = torch.linspace(0, total_len, n_points + 1, device=device, dtype=dtype)[:-1]

    positions_x = torch.zeros(n_points, device=device, dtype=dtype)
    positions_y = torch.zeros(n_points, device=device, dtype=dtype)
    angles = torch.zeros(n_points, device=device, dtype=dtype)

    # Cumulative lengths for segment boundaries
    s1 = len_straight_v                          # end of right edge
    s2 = s1 + len_corner                         # end of top-right corner
    s3 = s2 + len_straight_h                     # end of top edge
    s4 = s3 + len_corner                         # end of top-left corner
    s5 = s4 + len_straight_v                     # end of left edge
    s6 = s5 + len_corner                         # end of bottom-left corner
    s7 = s6 + len_straight_h                     # end of bottom edge
    # s8 = total_len                             # end of bottom-right corner

    for i, s in enumerate(s_vals):
        s = s.item()

        if s < s1:
            # Right edge: x = cx + hw + r, y goes from cy - hh to cy + hh
            t = s / len_straight_v if len_straight_v > 0 else 0
            positions_x[i] = cx + hw + r
            positions_y[i] = cy - hh + t * 2 * hh
            angles[i] = 0.0  # normal points right

        elif s < s2:
            # Top-right corner: center at (cx + hw, cy + hh)
            # Arc from (cx + hw + r, cy + hh) to (cx + hw, cy + hh + r)
            # theta: 0 → π/2
            t = (s - s1) / len_corner if len_corner > 0 else 0
            theta = t * (math.pi / 2)
            positions_x[i] = cx + hw + r * math.cos(theta)
            positions_y[i] = cy + hh + r * math.sin(theta)
            angles[i] = theta

        elif s < s3:
            # Top edge: y = cy + hh + r, x goes from cx + hw to cx - hw
            t = (s - s2) / len_straight_h if len_straight_h > 0 else 0
            positions_x[i] = cx + hw - t * 2 * hw
            positions_y[i] = cy + hh + r
            angles[i] = math.pi / 2  # normal points up

        elif s < s4:
            # Top-left corner: center at (cx - hw, cy + hh)
            # Arc from (cx - hw, cy + hh + r) to (cx - hw - r, cy + hh)
            # theta: π/2 → π
            t = (s - s3) / len_corner if len_corner > 0 else 0
            theta = math.pi / 2 + t * (math.pi / 2)
            positions_x[i] = cx - hw + r * math.cos(theta)
            positions_y[i] = cy + hh + r * math.sin(theta)
            angles[i] = theta

        elif s < s5:
            # Left edge: x = cx - hw - r, y goes from cy + hh to cy - hh
            t = (s - s4) / len_straight_v if len_straight_v > 0 else 0
            positions_x[i] = cx - hw - r
            positions_y[i] = cy + hh - t * 2 * hh
            angles[i] = math.pi  # normal points left

        elif s < s6:
            # Bottom-left corner: center at (cx - hw, cy - hh)
            # Arc from (cx - hw - r, cy - hh) to (cx - hw, cy - hh - r)
            # theta: π → 3π/2 (or -π/2)
            t = (s - s5) / len_corner if len_corner > 0 else 0
            theta = math.pi + t * (math.pi / 2)
            positions_x[i] = cx - hw + r * math.cos(theta)
            positions_y[i] = cy - hh + r * math.sin(theta)
            angles[i] = theta

        elif s < s7:
            # Bottom edge: y = cy - hh - r, x goes from cx - hw to cx + hw
            t = (s - s6) / len_straight_h if len_straight_h > 0 else 0
            positions_x[i] = cx - hw + t * 2 * hw
            positions_y[i] = cy - hh - r
            angles[i] = -math.pi / 2  # normal points down

        else:
            # Bottom-right corner: center at (cx + hw, cy - hh)
            # Arc from (cx + hw, cy - hh - r) to (cx + hw + r, cy - hh)
            # theta: -π/2 → 0
            t = (s - s7) / len_corner if len_corner > 0 else 0
            theta = -math.pi / 2 + t * (math.pi / 2)
            positions_x[i] = cx + hw + r * math.cos(theta)
            positions_y[i] = cy - hh + r * math.sin(theta)
            angles[i] = theta

    positions = torch.stack([positions_x, positions_y], dim=1)
    return OrientedPointCloudVarifold(positions=positions, angles=angles)


def generate_oriented_two_rectangles(
    n_per_rect: int,
    width1: float = 0.8,
    height1: float = 2.0,
    center1: Tuple[float, float] = (-0.6, 0.0),
    width2: float = 0.8,
    height2: float = 2.0,
    center2: Tuple[float, float] = (0.6, 0.0),
    corner_radius: float = 0.1,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> OrientedPointCloudVarifold:
    """
    Generate oriented point cloud for two rounded rectangles.

    Default parameters create two vertical rectangles close together.

    Args:
        n_per_rect: points per rectangle
        width1, height1, center1: first rectangle parameters
        width2, height2, center2: second rectangle parameters
        corner_radius: corner radius for both rectangles
        device: torch device
        dtype: torch dtype

    Returns:
        OrientedPointCloudVarifold with ~2*n_per_rect points
    """
    v1 = generate_oriented_rectangle(n_per_rect, width1, height1, center1, corner_radius, device, dtype)
    v2 = generate_oriented_rectangle(n_per_rect, width2, height2, center2, corner_radius, device, dtype)

    positions = torch.cat([v1.positions, v2.positions], dim=0)
    angles = torch.cat([v1.angles, v2.angles], dim=0)

    return OrientedPointCloudVarifold(positions=positions, angles=angles)
