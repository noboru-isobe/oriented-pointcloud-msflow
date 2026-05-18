"""Core utilities for visualization of oriented point cloud varifolds."""

import numpy as np
import torch
from typing import Tuple, Optional
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from ..oriented_varifold import OrientedPointCloudVarifold
from ..oriented_varifold.mass import compute_masses
from ..perimeter.coherence_perimeter import compute_coherence


def compute_mass_colors(masses: torch.Tensor, 
                       colormap: str = "viridis") -> np.ndarray:
    """Compute colors for mass visualization.
    
    Args:
        masses: (N,) tensor of point masses
        colormap: Matplotlib colormap name
        
    Returns:
        (N, 4) RGBA color array for PyQtGraph/Matplotlib
    """
    masses_np = masses.cpu().numpy()
    
    # Normalize masses to [0, 1]
    if masses_np.max() > masses_np.min():
        norm_masses = (masses_np - masses_np.min()) / (masses_np.max() - masses_np.min())
    else:
        norm_masses = np.ones_like(masses_np)
    
    # Get colormap and convert to RGBA
    cmap = cm.get_cmap(colormap)
    colors = cmap(norm_masses)  # (N, 4) RGBA
    
    return colors


def compute_coherence_colors(varifold: OrientedPointCloudVarifold,
                           masses: torch.Tensor,
                           sigma: Optional[float] = None,
                           colormap: str = "plasma") -> np.ndarray:
    """Compute colors for coherence visualization.
    
    Args:
        varifold: Oriented point cloud varifold
        masses: Point masses
        sigma: Mollifier bandwidth (auto if None)
        colormap: Matplotlib colormap name
        
    Returns:
        (N, 4) RGBA color array
    """
    # Compute coherence field q_i = |V_σ(x_i)| / U_σ(x_i)
    coherence = compute_coherence(varifold, masses, sigma)
    coherence_np = coherence.cpu().numpy()
    
    # Coherence is already in [0, 1] range
    cmap = cm.get_cmap(colormap)
    colors = cmap(coherence_np)  # (N, 4) RGBA
    
    return colors


def prepare_boundary_data(varifold: OrientedPointCloudVarifold,
                         masses: torch.Tensor,
                         arrow_scale: float = 0.05) -> dict:
    """Prepare boundary visualization data.
    
    Args:
        varifold: Oriented point cloud varifold
        masses: Point masses
        arrow_scale: Scale factor for normal arrows
        
    Returns:
        Dictionary with positions, colors, arrows
    """
    positions = varifold.positions.cpu().numpy()  # (N, 2)
    normals = varifold.normals.cpu().numpy()      # (N, 2)
    
    # Colors based on mass
    colors = compute_mass_colors(masses)
    
    # Arrow endpoints for normal vectors
    arrow_ends = positions + arrow_scale * normals  # (N, 2)
    
    return {
        'positions': positions,
        'colors': colors,
        'normals': normals,
        'arrow_ends': arrow_ends,
        'arrow_scale': arrow_scale,
    }


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy safely."""
    return tensor.detach().cpu().numpy()


def safe_normalize_colors(values: np.ndarray, 
                         vmin: Optional[float] = None,
                         vmax: Optional[float] = None) -> np.ndarray:
    """Safely normalize values to [0, 1] for colormap.
    
    Args:
        values: Input values
        vmin, vmax: Optional bounds (auto-computed if None)
        
    Returns:
        Normalized values in [0, 1]
    """
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
        
    if vmax > vmin:
        return (values - vmin) / (vmax - vmin)
    else:
        return np.ones_like(values)


def create_colorbar_data(values: np.ndarray, 
                        colormap: str = "viridis",
                        n_levels: int = 256) -> dict:
    """Create colorbar data for PyQtGraph.
    
    Args:
        values: Data values for colorbar range
        colormap: Matplotlib colormap name
        n_levels: Number of colorbar levels
        
    Returns:
        Dictionary with colorbar info
    """
    vmin, vmax = values.min(), values.max()
    
    # Create color array
    cmap = cm.get_cmap(colormap)
    levels = np.linspace(0, 1, n_levels)
    colors = cmap(levels)  # (n_levels, 4) RGBA
    
    return {
        'colors': colors,
        'levels': np.linspace(vmin, vmax, n_levels),
        'vmin': vmin,
        'vmax': vmax,
        'colormap': colormap,
    }