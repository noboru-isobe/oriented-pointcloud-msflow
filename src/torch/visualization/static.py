"""Static visualization using Matplotlib for publication-quality plots."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from ..oriented_varifold import OrientedPointCloudVarifold
from .core import (
    prepare_boundary_data,
    compute_coherence_colors,
    compute_mass_colors,
)


def plot_boundary_evolution(varifolds: List[OrientedPointCloudVarifold],
                          masses_list: List[torch.Tensor],
                          time_points: Optional[List[float]] = None,
                          save_path: Optional[Path] = None,
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """Plot boundary evolution over time.
    
    Args:
        varifolds: List of varifolds at different time steps
        masses_list: List of corresponding masses
        time_points: Time values (or step indices if None)
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_steps = len(varifolds)
    if time_points is None:
        time_points = list(range(n_steps))
        
    # Create subplots - show evolution snapshots
    n_cols = min(4, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_steps == 1:
        axes = [axes]  # Single axes object -> list
    elif n_rows == 1:
        pass  # Already 1D array, can be indexed directly
    else:
        axes = axes.flatten()  # 2D array -> 1D
    
    for i in range(n_steps):
        ax = axes[i] if i < len(axes) else axes[-1]
        
        varifold = varifolds[i]
        masses = masses_list[i]
        
        # Prepare boundary data
        boundary_data = prepare_boundary_data(varifold, masses)
        positions = boundary_data['positions']
        colors = boundary_data['colors']
        normals = boundary_data['normals']
        
        # Plot boundary points with mass colors
        ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=30, alpha=0.8)
        
        # Plot normal vectors (subsample for clarity)
        n_arrows = min(20, len(positions))
        indices = np.linspace(0, len(positions)-1, n_arrows, dtype=int)
        
        for idx in indices:
            pos = positions[idx]
            normal = normals[idx] * 0.05  # Scale arrow
            ax.arrow(pos[0], pos[1], normal[0], normal[1],
                    head_width=0.01, head_length=0.01, 
                    fc='white', ec='white', alpha=0.7)
        
        ax.set_aspect('equal')
        ax.set_title(f't = {time_points[i]:.3f}')
        ax.grid(True, alpha=0.3)
        
    # Hide unused subplots
    for i in range(n_steps, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def plot_coherence_field(varifold: OrientedPointCloudVarifold,
                        masses: torch.Tensor,
                        sigma: Optional[float] = None,
                        save_path: Optional[Path] = None,
                        figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """Plot coherence field for hidden boundary detection.
    
    Args:
        varifold: Oriented point cloud varifold
        masses: Point masses
        sigma: Mollifier bandwidth
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Prepare data
    boundary_data = prepare_boundary_data(varifold, masses)
    positions = boundary_data['positions']
    
    # Mass colors
    mass_colors = compute_mass_colors(masses, 'viridis')
    
    # Coherence colors  
    coherence_colors = compute_coherence_colors(varifold, masses, sigma, 'plasma')
    
    # Plot 1: Mass field
    ax1.scatter(positions[:, 0], positions[:, 1], c=mass_colors, s=50, alpha=0.8)
    ax1.set_title('Mass Distribution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coherence field
    scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], 
                          c=coherence_colors, s=50, alpha=0.8)
    ax2.set_title('Coherence Field (Hidden Boundary Detection)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Add coherence colorbar
    cbar = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar.set_label('Coherence q_i')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def plot_diagnostics(history: Dict[str, List[float]],
                    save_path: Optional[Path] = None,
                    figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
    """Plot diagnostic curves (perimeter, volume, circularity).
    
    Args:
        history: Dictionary with 'perimeters', 'volumes', 'times' lists
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    steps = np.arange(len(history['perimeters']))
    perimeters = np.array(history['perimeters'])
    volumes = np.array(history['volumes'])
    
    # Plot 1: Perimeter evolution
    ax1.plot(steps, perimeters, 'b-', linewidth=2, label='Perimeter')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Perimeter')
    ax1.set_title('Perimeter Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Shoelace area conservation
    initial_volume = volumes[0] if len(volumes) > 0 else 1.0
    volume_error = np.abs(volumes - initial_volume) / initial_volume * 100

    ax2.plot(steps, volume_error, 'r-', linewidth=2, label='Area Error')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Area Error (%)')
    ax2.set_title('Area Conservation (div. thm.)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Circularity evolution
    if len(perimeters) > 0 and len(volumes) > 0:
        circularity = 4 * np.pi * volumes / (perimeters ** 2)
        ax3.plot(steps, circularity, 'g-', linewidth=2, label='Circularity')
        ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Circle')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Circularity')
        ax3.set_title('Shape Evolution')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def create_publication_figure(varifolds: List[OrientedPointCloudVarifold],
                             masses_list: List[torch.Tensor],
                             history: Dict[str, List[float]],
                             key_steps: List[int] = None,
                             save_path: Optional[Path] = None,
                             figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
    """Create comprehensive publication-quality figure.
    
    Args:
        varifolds: List of varifolds at different time steps
        masses_list: Corresponding masses
        history: Diagnostic history
        key_steps: Indices of key time steps to show
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if key_steps is None:
        # Show initial, middle, and final states
        n_steps = len(varifolds)
        key_steps = [0, n_steps//2, n_steps-1] if n_steps >= 3 else list(range(n_steps))
    
    fig = plt.figure(figsize=figsize)
    
    # Create custom layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Top row: Evolution snapshots
    for i, step_idx in enumerate(key_steps):
        if i >= 3:  # Maximum 3 snapshots
            break
            
        ax = fig.add_subplot(gs[0, i])
        
        varifold = varifolds[step_idx]
        masses = masses_list[step_idx]
        
        boundary_data = prepare_boundary_data(varifold, masses)
        positions = boundary_data['positions']
        colors = boundary_data['colors']
        
        ax.scatter(positions[:, 0], positions[:, 1], 
                  c=colors, s=25, alpha=0.8)
        ax.set_aspect('equal')
        ax.set_title(f'Step {step_idx}')
        ax.grid(True, alpha=0.3)
    
    # Bottom row: Diagnostics
    ax_diag = fig.add_subplot(gs[1, :])
    
    steps = np.arange(len(history['perimeters']))
    perimeters = np.array(history['perimeters'])
    volumes = np.array(history['volumes'])
    
    # Dual y-axis for perimeter and volume
    ax_p = ax_diag
    ax_v = ax_diag.twinx()
    
    ax_p.plot(steps, perimeters, 'b-', linewidth=2, label='Perimeter')
    ax_p.set_ylabel('Perimeter', color='b')
    ax_p.tick_params(axis='y', labelcolor='b')

    ax_v.plot(steps, volumes, 'r-', linewidth=2, label='Area (div. thm.)')
    ax_v.set_ylabel('Area (div. thm.)', color='r')
    ax_v.tick_params(axis='y', labelcolor='r')
    
    ax_p.set_xlabel('Step')
    ax_p.set_title('Evolution Diagnostics')
    ax_p.grid(True, alpha=0.3)
    
    # Mark key steps
    for step_idx in key_steps:
        ax_p.axvline(x=step_idx, color='gray', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def setup_matplotlib_style():
    """Setup matplotlib style for publication-quality plots."""
    plt.style.use('default')

    # Set publication parameters
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.0,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        # LaTeX rendering
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath,amssymb}',
    })


def plot_shape_grid(
    positions_grid: List[List[np.ndarray]],
    row_labels: List[str],
    col_labels: List[str],
    titles: Optional[List[List[str]]] = None,
    angles_grid: Optional[List[List[np.ndarray]]] = None,
    masses_grid: Optional[List[List[np.ndarray]]] = None,
    save_path: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None,
    xlim: Tuple[float, float] = (-1.5, 1.5),
    ylim: Tuple[float, float] = (-1.5, 1.5),
    show_normals: bool = True,
    show_colorbar: bool = True,
    arrow_scale: float = 0.25,
) -> plt.Figure:
    """Plot shapes in a grid layout for comparison.

    Args:
        positions_grid: 2D list of position arrays, [row][col] -> (N, 2) array.
            None entries will show "No data".
        row_labels: Labels for each row (e.g., ["n=32", "n=64", "n=128"]).
        col_labels: Labels for each column (e.g., ["K=1", "K=3"]).
        titles: Optional 2D list of titles for each panel (e.g., "C=0.98").
        angles_grid: Optional 2D list of angle arrays for normal computation.
        masses_grid: Optional 2D list of mass arrays for coloring.
        save_path: Path to save the figure.
        figsize: Figure size. If None, computed from grid dimensions.
        xlim: X-axis limits for all panels.
        ylim: Y-axis limits for all panels.
        show_normals: Whether to show normal arrows (requires angles_grid).
        show_colorbar: Whether to show colorbar for mass.
        arrow_scale: Scale factor for normal arrows.

    Returns:
        Matplotlib figure.
    """
    n_rows = len(row_labels)
    n_cols = len(col_labels)

    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    # Track scatter for colorbar
    last_scatter = None

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]

            # Get data (may be None)
            positions = None
            if i < len(positions_grid) and j < len(positions_grid[i]):
                positions = positions_grid[i][j]

            angles = None
            if angles_grid is not None and i < len(angles_grid) and j < len(angles_grid[i]):
                angles = angles_grid[i][j]

            masses = None
            if masses_grid is not None and i < len(masses_grid) and j < len(masses_grid[i]):
                masses = masses_grid[i][j]

            if positions is not None:
                # Determine colors
                if masses is not None:
                    colors = masses
                    cmap = 'viridis'
                else:
                    colors = 'blue'
                    cmap = None

                # Plot as scatter (point cloud, no lines)
                scatter = ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    s=30,
                    c=colors,
                    cmap=cmap,
                    alpha=0.8,
                )
                if masses is not None:
                    last_scatter = scatter

                # Draw normal arrows if angles provided
                if show_normals and angles is not None:
                    normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
                    n_points = len(positions)
                    # Subsample arrows for clarity
                    if n_points <= 32:
                        indices = np.arange(n_points)
                    elif n_points <= 64:
                        indices = np.arange(0, n_points, 2)
                    else:
                        indices = np.arange(0, n_points, 4)

                    for idx in indices:
                        pos = positions[idx]
                        normal = normals[idx] * arrow_scale
                        ax.annotate(
                            '', xy=pos + normal, xytext=pos,
                            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7, lw=1.0)
                        )

                # Add title if provided
                if titles is not None and i < len(titles) and j < len(titles[i]):
                    ax.set_title(titles[i][j], fontsize=10)
            else:
                ax.text(
                    0.5, 0.5, "No data",
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=12, color='gray',
                )

            ax.set_aspect('equal')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)

            # Row labels on left
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=12)

    # Column labels on top
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(
            f"{label}\n{titles[0][j] if titles else ''}",
            fontsize=10,
        ) if titles else axes[0, j].set_xlabel(label, fontsize=12)
        axes[0, j].xaxis.set_label_position('top')

    # Add colorbar if masses were provided
    if show_colorbar and last_scatter is not None and masses_grid is not None:
        # Make space on the right for colorbar
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        fig.colorbar(last_scatter, cax=cbar_ax, label=r'Mass $m_i$')
    else:
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_perimeter_comparison(
    perimeters: Dict[str, np.ndarray],
    total_time: float = None,
    time_step: float = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """Plot perimeter evolution curves overlaid for comparison.

    Args:
        perimeters: Dictionary mapping label -> (n_steps,) array of perimeter values.
            Example: {"n=32, K=1": array([...]), "n=64, K=3": array([...])}
        total_time: Total simulation time. If provided, x-axis is scaled to [0, total_time].
        time_step: Time step size for x-axis scaling (used if total_time not provided).
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color cycle for different curves
    colors = plt.cm.tab10.colors

    for i, (label, data) in enumerate(perimeters.items()):
        if total_time is not None:
            # All curves share the same total time but may have different n_steps
            time_axis = np.linspace(0, total_time, len(data))
        elif time_step is not None:
            time_axis = np.arange(len(data)) * time_step
        else:
            time_axis = np.arange(len(data))
        color = colors[i % len(colors)]
        ax.plot(time_axis, data, linewidth=1.5, label=label, color=color)

    ax.set_xlabel('Time')
    ax.set_ylabel('Perimeter')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig