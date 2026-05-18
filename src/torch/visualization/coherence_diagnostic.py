"""Coherence diagnostic visualization for oriented point cloud varifolds.

Provides functions to compute and visualize coherence-related quantities
for diagnosing hidden boundary suppression in multi-component flows.

Usage:
    from src.torch.visualization.coherence_diagnostic import (
        compute_coherence_diagnostics,
        plot_coherence_snapshot,
    )

    diag = compute_coherence_diagnostics(varifold, mass_delta, mass_tau,
                                          mass_kernel, perimeter_sigma, perimeter_kernel)
    plot_coherence_snapshot(diag, step=10, n_per_ellipse=64, save_path="snapshot.png")
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ..oriented_varifold import OrientedPointCloudVarifold
from ..oriented_varifold.mass import compute_masses, compute_recommended_params
from ..perimeter.coherence_perimeter import (
    compute_scalar_density,
    compute_vector_field,
    compute_recommended_sigma,
)
from ..transport.bem_wasserstein import compute_coherence as compute_coherence_full
from ..math_utils.curvature import compute_regularized_curvature
from .core import safe_normalize_colors, tensor_to_numpy


def compute_coherence_diagnostics(
    varifold: OrientedPointCloudVarifold,
    mass_delta: float,
    mass_tau: float,
    mass_kernel: str = "wendland_c2",
    perimeter_sigma: float | None = None,
    perimeter_kernel: str = "wendland_c2",
    use_unit_coherence: bool = False,
) -> dict:
    """Compute coherence-related quantities from a single varifold.

    Args:
        varifold: Oriented point cloud varifold.
        mass_delta: Mass KDE bandwidth.
        mass_tau: Mass cutoff threshold.
        mass_kernel: Kernel for mass computation.
        perimeter_sigma: Mollifier bandwidth for perimeter (auto if None).
        perimeter_kernel: Kernel for perimeter computation.
        use_unit_coherence: If True, set coherence to 1.0 for all points
            (matches solver behaviour when use_unit_coherence=True).

    Returns:
        Dictionary with:
            positions: (N, 2) numpy array
            normals: (N, 2) numpy array
            masses: (N,) numpy array
            coherence: (N,) numpy array in [0, 1]
            perimeter_contrib: (N,) numpy array — m_i * q_i
            scalar_density: (N,) numpy array — U_σ(x_i)
            vector_field: (N, 2) numpy array — V_σ(x_i)
            sigma_used: float — perimeter sigma used
            use_unit_coherence: bool — whether unit coherence was used
    """
    positions = varifold.positions
    normals = varifold.normals

    # Compute masses
    masses = compute_masses(positions, mass_delta, mass_tau, mass_kernel)

    # Auto-compute perimeter sigma if needed
    if perimeter_sigma is None:
        perimeter_sigma = compute_recommended_sigma(positions)

    # Compute scalar density and vector field
    U = compute_scalar_density(positions, masses, perimeter_sigma, perimeter_kernel)
    V = compute_vector_field(positions, normals, masses, perimeter_sigma, perimeter_kernel)

    # Compute coherence (same as solver)
    if use_unit_coherence:
        q = torch.ones(positions.shape[0], device=positions.device, dtype=positions.dtype)
    else:
        q = compute_coherence_full(
            varifold, masses, perimeter_sigma, perimeter_kernel,
        )

    # Perimeter contribution per point
    perimeter_contrib = masses * q

    # Curvature (BLM regularized)
    kappa, _ = compute_regularized_curvature(
        positions, normals, masses, epsilon=mass_delta, kernel=mass_kernel,
    )

    return {
        "positions": tensor_to_numpy(positions),
        "normals": tensor_to_numpy(normals),
        "masses": tensor_to_numpy(masses),
        "coherence": tensor_to_numpy(q),
        "perimeter_contrib": tensor_to_numpy(perimeter_contrib),
        "scalar_density": tensor_to_numpy(U),
        "vector_field": tensor_to_numpy(V),
        "curvature": tensor_to_numpy(kappa),
        "sigma_used": perimeter_sigma,
        "use_unit_coherence": use_unit_coherence,
    }


def plot_coherence_snapshot(
    diagnostics: dict,
    step: int | None = None,
    n_per_ellipse: int | None = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 15),
) -> plt.Figure:
    """Plot 3x2 coherence diagnostic panel.

    Panels:
        (0,0) Coherence heatmap — scatter colored by q_i (plasma, [0, 1])
        (0,1) Curvature heatmap — scatter colored by κ_i (coolwarm, symmetric)
        (1,0) Perimeter contribution — scatter colored by m_i*q_i (viridis)
        (1,1) Effective mass with normal vectors
        (2,0) Coherence profile — q_i vs point index, gap region highlighted
        (2,1) Curvature profile — κ_i vs point index, gap region highlighted

    Args:
        diagnostics: Output of compute_coherence_diagnostics.
        step: Step number for title annotation.
        n_per_ellipse: Points per ellipse (for gap region highlight).
        save_path: Path to save figure (None to skip saving).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    pos = diagnostics["positions"]
    normals = diagnostics["normals"]
    masses = diagnostics["masses"]
    coherence = diagnostics["coherence"]
    perimeter_contrib = diagnostics["perimeter_contrib"]
    curvature = diagnostics.get("curvature")

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    step_str = f" (step {step})" if step is not None else ""

    # --- (0,0) Coherence heatmap ---
    ax = axes[0, 0]
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=coherence, cmap="plasma",
                    vmin=0, vmax=1, s=40, edgecolors="k", linewidths=0.3)
    fig.colorbar(sc, ax=ax, shrink=0.8, label=r"$q_i$")
    ax.set_aspect("equal")
    ax.set_title(f"Coherence $q_i${step_str}")
    ax.grid(True, alpha=0.3)

    # --- (0,1) Curvature heatmap ---
    ax = axes[0, 1]
    if curvature is not None:
        kappa_abs_max = max(np.abs(curvature).max(), 1e-10)
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=curvature, cmap="coolwarm",
                        vmin=-kappa_abs_max, vmax=kappa_abs_max,
                        s=40, edgecolors="k", linewidths=0.3)
        fig.colorbar(sc, ax=ax, shrink=0.8, label=r"$\kappa_i$")
        ax.set_title(f"Curvature $\\kappa_i${step_str}")
    else:
        ax.set_title(f"Curvature (N/A){step_str}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # --- (1,0) Perimeter contribution m_i * q_i ---
    ax = axes[1, 0]
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=perimeter_contrib, cmap="viridis",
                    s=40, edgecolors="k", linewidths=0.3)
    fig.colorbar(sc, ax=ax, shrink=0.8, label=r"$m_i q_i$")
    ax.set_aspect("equal")
    ax.set_title(f"Perimeter contrib $m_i q_i${step_str}")
    ax.grid(True, alpha=0.3)

    # --- (1,1) Effective mass + normal vectors ---
    ax = axes[1, 1]
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=masses, cmap="viridis",
                    s=40, edgecolors="k", linewidths=0.3)
    fig.colorbar(sc, ax=ax, shrink=0.8, label=r"$m_i$")

    # Draw normal arrows (subsample for clarity)
    n_pts = len(pos)
    stride = 1
    arrow_scale = 0.08
    for i in range(0, n_pts, stride):
        ax.quiver(pos[i, 0], pos[i, 1],
                  normals[i, 0] * arrow_scale, normals[i, 1] * arrow_scale,
                  color="red", alpha=0.7, scale=1, scale_units="xy",
                  width=0.004, headwidth=3, headlength=4)
    ax.set_aspect("equal")
    ax.set_title(f"Mass $m_i$ + normals{step_str}")
    ax.grid(True, alpha=0.3)

    # --- Helper for gap region highlighting ---
    def _add_gap_highlights(ax, n_per_ellipse, coherence, indices):
        if n_per_ellipse is not None:
            boundary_idx = n_per_ellipse
            ax.axvline(x=boundary_idx, color="gray", ls="--", alpha=0.6,
                        label=f"Component boundary ({boundary_idx})")
            gap_mask = coherence < 0.5
            if gap_mask.any():
                ylim = ax.get_ylim()
                ax.fill_between(indices, ylim[0], ylim[1], where=gap_mask,
                                alpha=0.15, color="red",
                                label="Low coherence ($q < 0.5$)")
                ax.set_ylim(ylim)
            ax.legend(fontsize=8, loc="lower left")

    indices = np.arange(len(coherence))

    # --- (2,0) Coherence profile (point index order) ---
    ax = axes[2, 0]
    ax.scatter(indices, coherence, s=8, color="C0", zorder=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Point index")
    ax.set_ylabel(r"Coherence $q_i$")
    ax.set_title(f"Coherence profile{step_str}")
    ax.grid(True, alpha=0.3)
    _add_gap_highlights(ax, n_per_ellipse, coherence, indices)

    # --- (2,1) Curvature profile (point index order) ---
    ax = axes[2, 1]
    if curvature is not None:
        ax.scatter(indices, curvature, s=8, color="C1", zorder=2)
        ax.set_xlabel("Point index")
        ax.set_ylabel(r"Curvature $\kappa_i$")
        ax.set_title(f"Curvature profile{step_str}")
        ax.grid(True, alpha=0.3)
        _add_gap_highlights(ax, n_per_ellipse, coherence, indices)
    else:
        ax.set_title(f"Curvature profile (N/A){step_str}")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
