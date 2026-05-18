"""Parametric curve definitions for shape generation."""

import torch
import math
from typing import Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class ParametricCurve:
    """
    Parametric curve representation.

    A curve γ(t) = (x(t), y(t)) for t ∈ [0, 2π).
    The normal angle θ(t) is computed from the tangent.

    Attributes:
        position_fn: (x(t), y(t)) position at parameter t
        normal_angle_fn: outward normal angle θ(t)
        speed_fn: |γ'(t)| = sqrt(dx/dt² + dy/dt²) for arc length computation
        name: curve identifier
    """
    position_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    normal_angle_fn: Callable[[torch.Tensor], torch.Tensor]
    speed_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    name: str = "curve"


def circle(radius: float = 1.0, center: Tuple[float, float] = (0.0, 0.0)) -> ParametricCurve:
    """
    Circle parametrization.

    γ(t) = center + radius * (cos(t), sin(t))
    Normal angle: θ = t (outward normal)
    Speed: |γ'(t)| = radius (constant)
    """
    cx, cy = center

    def position_fn(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = cx + radius * torch.cos(t)
        y = cy + radius * torch.sin(t)
        return x, y

    def normal_angle_fn(t: torch.Tensor) -> torch.Tensor:
        return t

    def speed_fn(t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, radius)

    return ParametricCurve(position_fn, normal_angle_fn, speed_fn, name="circle")


def ellipse(
    a: float = 1.0,
    b: float = 0.5,
    center: Tuple[float, float] = (0.0, 0.0),
) -> ParametricCurve:
    """
    Ellipse parametrization.

    γ(t) = center + (a*cos(t), b*sin(t))
    Normal angle: θ = atan2(a*sin(t), b*cos(t))
    Speed: |γ'(t)| = sqrt(a²sin²t + b²cos²t)

    Args:
        a: semi-axis in x direction
        b: semi-axis in y direction
        center: center of ellipse
    """
    cx, cy = center

    def position_fn(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = cx + a * torch.cos(t)
        y = cy + b * torch.sin(t)
        return x, y

    def normal_angle_fn(t: torch.Tensor) -> torch.Tensor:
        return torch.atan2(a * torch.sin(t), b * torch.cos(t))

    def speed_fn(t: torch.Tensor) -> torch.Tensor:
        # |γ'(t)| = |(-a sin t, b cos t)| = sqrt(a² sin²t + b² cos²t)
        return torch.sqrt(a**2 * torch.sin(t)**2 + b**2 * torch.cos(t)**2)

    return ParametricCurve(position_fn, normal_angle_fn, speed_fn, name="ellipse")


def flower(
    n_petals: int = 5,
    inner_radius: float = 0.5,
    outer_radius: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
) -> ParametricCurve:
    """
    Flower shape (rose curve variant).

    r(t) = base + amp * cos(n*t)
    γ(t) = center + r(t) * (cos(t), sin(t))
    Speed: |γ'(t)| = sqrt(tx² + ty²) where (tx, ty) is the tangent vector
    """
    cx, cy = center
    amp = (outer_radius - inner_radius) / 2
    base = inner_radius + amp

    def position_fn(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = base + amp * torch.cos(n_petals * t)
        x = cx + r * torch.cos(t)
        y = cy + r * torch.sin(t)
        return x, y

    def normal_angle_fn(t: torch.Tensor) -> torch.Tensor:
        r = base + amp * torch.cos(n_petals * t)
        dr = -n_petals * amp * torch.sin(n_petals * t)
        # Tangent: (dx/dt, dy/dt)
        tx = dr * torch.cos(t) - r * torch.sin(t)
        ty = dr * torch.sin(t) + r * torch.cos(t)
        # Outward normal: rotate tangent clockwise by 90° = (ty, -tx)
        # Normal angle = atan2(-tx, ty)
        return torch.atan2(-tx, ty)

    def speed_fn(t: torch.Tensor) -> torch.Tensor:
        r = base + amp * torch.cos(n_petals * t)
        dr = -n_petals * amp * torch.sin(n_petals * t)
        tx = dr * torch.cos(t) - r * torch.sin(t)
        ty = dr * torch.sin(t) + r * torch.cos(t)
        return torch.sqrt(tx**2 + ty**2)

    return ParametricCurve(position_fn, normal_angle_fn, speed_fn, name="flower")


def star(
    n_points: int = 5,
    inner_radius: float = 0.4,
    outer_radius: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
) -> ParametricCurve:
    """
    Star shape with n_points peaks.

    r(t) = base + amp * cos(n*t)
    Creates a star with n_points peaks (outer_radius) and n_points valleys (inner_radius).
    Speed: |γ'(t)| = sqrt(tx² + ty²) where (tx, ty) is the tangent vector
    """
    cx, cy = center
    amp = (outer_radius - inner_radius) / 2
    base = (outer_radius + inner_radius) / 2

    def position_fn(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = base + amp * torch.cos(n_points * t)
        x = cx + r * torch.cos(t)
        y = cy + r * torch.sin(t)
        return x, y

    def normal_angle_fn(t: torch.Tensor) -> torch.Tensor:
        r = base + amp * torch.cos(n_points * t)
        dr = -n_points * amp * torch.sin(n_points * t)
        tx = dr * torch.cos(t) - r * torch.sin(t)
        ty = dr * torch.sin(t) + r * torch.cos(t)
        # Outward normal: rotate tangent clockwise by 90°
        return torch.atan2(-tx, ty)

    def speed_fn(t: torch.Tensor) -> torch.Tensor:
        r = base + amp * torch.cos(n_points * t)
        dr = -n_points * amp * torch.sin(n_points * t)
        tx = dr * torch.cos(t) - r * torch.sin(t)
        ty = dr * torch.sin(t) + r * torch.cos(t)
        return torch.sqrt(tx**2 + ty**2)

    return ParametricCurve(position_fn, normal_angle_fn, speed_fn, name="star")
