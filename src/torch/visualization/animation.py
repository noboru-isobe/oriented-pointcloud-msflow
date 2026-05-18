"""Animation utilities for oriented point cloud varifolds using matplotlib.

This module provides functions to create animated visualizations of varifold evolution,
including MP4 video generation and frame-by-frame analysis.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable
import time

from ..oriented_varifold import OrientedPointCloudVarifold
from ..solver import MMSolver, MMConfig
from ..solver.mm_solver import compute_volume_divergence
from ..oriented_varifold.mass import compute_masses, compute_recommended_params
from .core import prepare_boundary_data, compute_mass_colors, compute_coherence_colors
from .static import setup_matplotlib_style


class VarifoldAnimator:
    """Base class for creating animated visualizations of varifold evolution."""
    
    def __init__(self,
                 varifold: OrientedPointCloudVarifold,
                 config: Optional[MMConfig] = None,
                 figsize: Tuple[int, int] = (12, 8)):
        """Initialize animator.

        Args:
            varifold: Initial oriented point cloud varifold
            config: MM solver configuration
            figsize: Figure size for animation
        """
        self.initial_varifold = varifold
        self.config = config or MMConfig()
        self.solver = MMSolver(self.config)
        self.figsize = figsize
        
        # Animation data
        self.varifolds = [varifold]
        self.history = {
            'perimeters': [],
            'volumes': [],
            'times': [],
        }
        
        # Figure setup
        self.fig = None
        self.axes = None
        self.artists = {}
        
    def setup_figure(self):
        """Setup matplotlib figure and axes. Override in subclasses."""
        setup_matplotlib_style()
        self.fig, self.axes = plt.subplots(2, 2, figsize=self.figsize)
        self.axes = self.axes.flatten()

        # axes[0]: Boundary evolution (mass colorbar + normals)
        self.axes[0].set_aspect('equal')
        self.axes[0].set_title('Boundary Evolution')
        self.axes[0].grid(True, alpha=0.3)
        if hasattr(self, 'plot_limits') and 'boundary' in self.plot_limits:
            (xmin, xmax), (ymin, ymax) = self.plot_limits['boundary']
            self.axes[0].set_xlim(xmin, xmax)
            self.axes[0].set_ylim(ymin, ymax)

        # axes[1]: Perimeter evolution
        self.axes[1].set_title('Perimeter Evolution')
        self.axes[1].set_xlabel(r'$\text{time} = n\tau$')
        self.axes[1].set_ylabel('Perimeter')
        self.axes[1].grid(True, alpha=0.3)

        # axes[2]: Area conservation (divergence theorem, relative error %)
        self.axes[2].set_title('Area Conservation (div. thm.)')
        self.axes[2].set_xlabel(r'$\text{time} = n\tau$')
        self.axes[2].set_ylabel(r'$(A - A_0) / A_0$ [\%]')
        self.axes[2].grid(True, alpha=0.3)

        # axes[3]: Circularity metric
        self.axes[3].set_title('Circularity Metric')
        self.axes[3].set_xlabel(r'$\text{time} = n\tau$')
        self.axes[3].set_ylabel(r'$4\pi \cdot \text{Area}/\text{Perimeter}^2$')
        self.axes[3].grid(True, alpha=0.3)
        
    def evolve_varifold(self, n_steps: int, 
                       callback: Optional[Callable] = None) -> List[OrientedPointCloudVarifold]:
        """Evolve varifold for n_steps and record history.
        
        Args:
            n_steps: Number of evolution steps
            callback: Optional callback function called after each step
            
        Returns:
            List of varifolds at each time step
        """
        print(f"Evolving varifold for {n_steps} steps...")

        try:
            # Record initial state perimeter and volume
            from ..perimeter import compute_perimeter_coherence
            from ..oriented_varifold.mass import compute_masses

            # Compute mass parameters locally (do NOT mutate config)
            cfg = self.solver.config
            if cfg.mass_delta is None or cfg.mass_tau is None:
                delta, tau = compute_recommended_params(
                    self.initial_varifold.positions,
                    kernel=cfg.mass_kernel,
                    k_min=cfg.mass_k_min,
                )
                mass_delta = cfg.mass_delta if cfg.mass_delta is not None else delta
                mass_tau = cfg.mass_tau if cfg.mass_tau is not None else tau
            else:
                mass_delta = cfg.mass_delta
                mass_tau = cfg.mass_tau

            # Compute initial masses, perimeter and volume
            initial_masses = compute_masses(
                self.initial_varifold.positions,
                mass_delta,
                mass_tau,
                cfg.mass_kernel,
            )
            initial_perimeter = compute_perimeter_coherence(
                self.initial_varifold, initial_masses,
                sigma=cfg.perimeter_sigma,
                kernel=cfg.perimeter_kernel,
                c_sigma=cfg.perimeter_c_sigma,
            )
            # Store initial volume as target for conservation (divergence theorem)
            self.target_volume = compute_volume_divergence(self.initial_varifold, initial_masses)
            initial_volume = self.target_volume

            # Clear and initialize history
            self.history = {
                'perimeters': [initial_perimeter.item()],  # torch.Tensor -> .item()
                'volumes': [initial_volume],               # float -> そのまま
                'times': [time.time()]
            }

            # Use solver.solve() to get full evolution history
            # solve() catches internal exceptions and returns partial history
            evolution_history = self.solver.solve(self.initial_varifold, n_steps)

            # Extract varifolds from history (works for both full and partial results)
            varifolds = [self.initial_varifold]  # Initial state
            device = self.initial_varifold.positions.device

            for step in range(evolution_history.n_completed):
                # Use get_varifold (list-based) to support variable N
                varifold = evolution_history.get_varifold(step + 1)  # +1 because [0] is initial
                varifolds.append(varifold)

                # Record history for animation - compute directly from current varifold
                from ..perimeter import compute_perimeter_coherence
                from ..oriented_varifold.mass import compute_masses, compute_recommended_params as _crp

                # Compute current masses and perimeter (adaptive δ/τ per frame)
                if cfg.mass_delta is None or cfg.mass_tau is None:
                    _d, _t = _crp(varifold.positions, kernel=cfg.mass_kernel, k_min=cfg.mass_k_min)
                    _md = cfg.mass_delta if cfg.mass_delta is not None else _d
                    _mt = cfg.mass_tau if cfg.mass_tau is not None else _t
                else:
                    _md, _mt = cfg.mass_delta, cfg.mass_tau
                masses = compute_masses(varifold.positions, _md, _mt, cfg.mass_kernel)
                current_perimeter = compute_perimeter_coherence(
                    varifold, masses,
                    sigma=cfg.perimeter_sigma,
                    kernel=cfg.perimeter_kernel,
                    c_sigma=cfg.perimeter_c_sigma,
                )
                # Compute volume using divergence theorem
                current_volume = compute_volume_divergence(varifold, masses)

                self.history['perimeters'].append(current_perimeter.item())  # torch.Tensor -> .item()
                self.history['volumes'].append(current_volume)            # float -> そのまま
                self.history['times'].append(time.time())

                # Optional callback
                if callback:
                    callback(step, varifold, None)

                # Progress report
                if (step + 1) % max(1, n_steps // 10) == 0:
                    print(f"  Step {step + 1}/{n_steps} completed")

            print(f"  Evolution completed: {evolution_history.n_completed}/{n_steps} steps")

        except Exception as e:
            print(f"Evolution error: {e}")
            # Fallback to just initial varifold
            varifolds = [self.initial_varifold]

        self.varifolds = varifolds
        
        # Pre-compute axis limits for consistent scaling
        self._compute_plot_limits()
        
        return varifolds
        
    def _compute_plot_limits(self):
        """Pre-compute axis limits for consistent scaling across animation.

        Uses initial values as reference: perimeter [0, 2*P0], volume [0, 2*V0].
        """
        self.plot_limits = {}

        if len(self.history['perimeters']) > 0:
            p0 = self.history['perimeters'][0]
            self.plot_limits['perimeter'] = (0, 2 * p0)

        if len(self.history['volumes']) > 0:
            v0 = self.history['volumes'][0]
            self.plot_limits['volume'] = (0, 2 * v0)

        # Boundary extent and mass limits from initial frame only
        if len(self.varifolds) > 0:
            pos0 = self.varifolds[0].positions.detach().cpu().numpy()
            x_range = pos0[:, 0].max() - pos0[:, 0].min()
            y_range = pos0[:, 1].max() - pos0[:, 1].min()
            margin = 0.3 * max(x_range, y_range, 0.1)
            self.plot_limits['boundary'] = (
                (pos0[:, 0].min() - margin, pos0[:, 0].max() + margin),
                (pos0[:, 1].min() - margin, pos0[:, 1].max() + margin),
            )

            delta, tau = compute_recommended_params(self.varifolds[0].positions)
            masses0 = compute_masses(self.varifolds[0].positions, delta, tau).cpu().numpy()
            self.plot_limits['mass'] = (0.0, masses0.max() * 2)
        
    def animate_frame(self, frame: int):
        """Animate single frame. Override in subclasses.
        
        Args:
            frame: Frame number
            
        Returns:
            List of artists to update
        """
        if frame >= len(self.varifolds):
            return []
            
        varifold = self.varifolds[frame]
        
        # Update boundary plot
        self._update_boundary_plot(varifold, frame)
        
        # Update diagnostic plot
        self._update_diagnostic_plot(frame)
        
        return list(self.artists.values())
        
    def _update_boundary_plot(self, varifold: OrientedPointCloudVarifold, frame: int):
        """Update boundary visualization.

        Points are coloured by the **effective mass** ``m_i · q_i`` — the
        per-point perimeter contribution (since P̂ = Σ m_i q_i). This is
        ≈ m_i for pure boundaries (q ≈ 1) and drops where opposing
        normals cancel (q ≈ 0), e.g. at the touching point of two
        ellipses just before fusion.
        """
        # Local import to avoid a circular dependency with bem_wasserstein.
        from ..transport.bem_wasserstein import compute_coherence

        # Compute masses
        delta, tau = compute_recommended_params(varifold.positions)
        masses = compute_masses(varifold.positions, delta, tau)

        # Coherence q_i. perimeter_sigma may be set explicitly on the
        # config; otherwise fall back to the recommended bandwidth from
        # the nearest-neighbour scale.
        cfg = self.config
        if cfg.perimeter_sigma is not None:
            sigma = cfg.perimeter_sigma
        else:
            from ..perimeter.coherence_perimeter import compute_recommended_sigma
            sigma = compute_recommended_sigma(varifold.positions,
                                              c_sigma=cfg.perimeter_c_sigma)
        coherence = compute_coherence(
            varifold, masses, sigma, cfg.perimeter_kernel,
        ).clamp(0.0, 1.0)
        effective_masses_np = (masses * coherence).cpu().numpy()

        # Prepare boundary data
        boundary_data = prepare_boundary_data(varifold, masses)
        positions = boundary_data['positions']

        # Determine colorbar limits (use pre-computed if available, otherwise current frame)
        if 'mass' in self.plot_limits:
            vmin, vmax = self.plot_limits['mass']
        else:
            vmin, vmax = float(effective_masses_np.min()), float(effective_masses_np.max())

        # Update scatter plot with colorbar
        if 'boundary_scatter' not in self.artists:
            self.artists['boundary_scatter'] = self.axes[0].scatter(
                positions[:, 0], positions[:, 1], c=effective_masses_np,
                s=20, alpha=0.8, cmap='viridis', vmin=vmin, vmax=vmax,
            )
            # Add colorbar
            if 'colorbar' not in self.artists:
                self.artists['colorbar'] = self.fig.colorbar(
                    self.artists['boundary_scatter'], ax=self.axes[0],
                    shrink=0.8, label=r'Effective mass $m_i\,q_i$'
                )
        else:
            # Update scatter data and colors
            self.artists['boundary_scatter'].set_offsets(positions)
            self.artists['boundary_scatter'].set_array(effective_masses_np)
            # Use fixed colorbar limits for consistent scaling
            self.artists['boundary_scatter'].set_clim(vmin=vmin, vmax=vmax)
            
        # Update normal arrows (subsample)
        self._update_normal_arrows(boundary_data)

        # Apply fixed axis limits
        if hasattr(self, 'plot_limits') and 'boundary' in self.plot_limits:
            (xmin, xmax), (ymin, ymax) = self.plot_limits['boundary']
            self.axes[0].set_xlim(xmin, xmax)
            self.axes[0].set_ylim(ymin, ymax)

        # Update title with frame info
        h = self.config.time_step
        self.axes[0].set_title(rf'Boundary Evolution  ($\text{{time}} = {frame * h:.4f}$)')
        
    def _update_normal_arrows(self, boundary_data):
        """Update normal vector arrows."""
        positions = boundary_data['positions']
        normals = boundary_data['normals']
        
        # Remove old arrows
        for key in list(self.artists.keys()):
            if key.startswith('arrow_'):
                self.artists[key].remove()
                del self.artists[key]
                
        # Add new arrows (subsample for performance)
        n_points = len(positions)
        if n_points <= 32:
            # Show all arrows for small point sets
            indices = np.arange(n_points)
        elif n_points <= 128:
            # Show every 2nd arrow for medium point sets
            indices = np.arange(0, n_points, 2)
        else:
            # Show every 4th arrow for large point sets  
            indices = np.arange(0, n_points, 4)
        
        # Dynamic arrow scaling based on average inter-point distance
        if len(positions) > 1:
            # Estimate characteristic length scale
            avg_spacing = np.sqrt(2.0 / len(positions))  # Rough estimate for unit circle
            arrow_scale = max(0.20, min(0.35, avg_spacing * 2.0))
        else:
            arrow_scale = 0.30
        
        for i, idx in enumerate(indices):
            pos = positions[idx]
            normal = normals[idx] * arrow_scale
            
            arrow_key = f'arrow_{i}'
            self.artists[arrow_key] = self.axes[0].annotate(
                '', xy=pos + normal, xytext=pos,
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.8, lw=1.5)
            )
            
    def _update_diagnostic_plot(self, frame: int):
        """Update diagnostic plots: perimeter, volume error %, circularity.

        x-axis is physical time ``t = n h`` (step index × MM time step),
        not the step index itself.
        """
        if frame == 0 or len(self.history['perimeters']) == 0:
            return

        h = self.config.time_step
        steps = np.arange(min(frame + 1, len(self.history['perimeters'])))
        t = steps * h
        perimeters = np.array(self.history['perimeters'][:len(steps)])
        volumes = np.array(self.history['volumes'][:len(steps)])

        # --- axes[1]: Perimeter ---
        if 'perimeter_line' not in self.artists:
            self.artists['perimeter_line'], = self.axes[1].plot(
                t, perimeters, 'b-', linewidth=2, label='Perimeter'
            )
            self.axes[1].legend()
        else:
            self.artists['perimeter_line'].set_data(t, perimeters)

        # --- axes[2]: Volume relative error (V - V₀)/V₀ [%] ---
        if hasattr(self, 'target_volume') and self.target_volume != 0:
            volume_error_pct = (volumes - self.target_volume) / self.target_volume * 100
        else:
            # Fallback: use initial volume
            v0 = volumes[0] if len(volumes) > 0 else 1.0
            volume_error_pct = (volumes - v0) / v0 * 100 if v0 != 0 else volumes * 0

        if 'volume_line' not in self.artists:
            self.artists['volume_line'], = self.axes[2].plot(
                t, volume_error_pct, 'r-', linewidth=2, label='Relative error'
            )
            self.axes[2].axhline(y=0.0, color='r',
                                linestyle='--', alpha=0.7, label='Exact conservation')
            self.axes[2].legend()
        else:
            self.artists['volume_line'].set_data(t, volume_error_pct)

        # --- axes[3]: Circularity 4π V / P² ---
        circularity = 4 * np.pi * volumes / (perimeters ** 2)

        if 'circularity_line' not in self.artists:
            self.artists['circularity_line'], = self.axes[3].plot(
                t, circularity, 'g-', linewidth=2, label='Circularity'
            )
            self.axes[3].axhline(y=1.0, color='g',
                                linestyle='--', alpha=0.7, label='Perfect Circle')
            self.axes[3].legend()
        else:
            self.artists['circularity_line'].set_data(t, circularity)

        # --- Axis limits ---
        total_steps = len(self.history['perimeters'])
        t_total = total_steps * h
        for ax in [self.axes[1], self.axes[2], self.axes[3]]:
            ax.set_xlim(0, t_total)

        if 'perimeter' in self.plot_limits:
            self.axes[1].set_ylim(*self.plot_limits['perimeter'])

        if 'volume' in self.plot_limits:
            all_volumes = np.array(self.history['volumes'])
            v0 = self.target_volume if hasattr(self, 'target_volume') and self.target_volume != 0 else all_volumes[0]
            err_min = (all_volumes.min() - v0) / v0 * 100
            err_max = (all_volumes.max() - v0) / v0 * 100
            # Narrow y-limits: 5% pad around the actual data range; ensure
            # 0 (the exact-conservation reference) stays visible.
            pad = max((err_max - err_min) * 0.05, 0.02)
            self.axes[2].set_ylim(min(err_min, 0.0) - pad, max(err_max, 0.0) + pad)

        # Circularity limits (handle mismatched history lengths and zero perimeters)
        n_common = min(len(self.history['perimeters']), len(self.history['volumes']))
        if n_common > 0:
            all_perimeters = np.array(self.history['perimeters'][:n_common])
            all_volumes = np.array(self.history['volumes'][:n_common])
            valid = all_perimeters > 0
            if valid.any():
                all_circularity = 4 * np.pi * all_volumes[valid] / (all_perimeters[valid] ** 2)
                cmin = min(all_circularity.min(), 0.95)
                cmax = max(all_circularity.max(), 1.05)
                self.axes[3].set_ylim(cmin, cmax)
        
    def create_animation(self,
                        n_steps: int = 50,
                        interval: int = 200,
                        save_path: Optional[Path] = None,
                        skip_frames: int = 1) -> animation.FuncAnimation:
        """Create matplotlib animation.

        Args:
            n_steps: Number of evolution steps
            interval: Time between frames in milliseconds
            save_path: Optional path to save MP4 video
            skip_frames: Only render every N-th frame for faster video generation

        Returns:
            FuncAnimation object
        """
        # Evolve varifold if not already done (allows precomputed history)
        if len(self.varifolds) <= 1:
            self.evolve_varifold(n_steps)

        # Setup figure
        self.setup_figure()

        # Set initial axis limits before animation starts
        self._set_initial_axis_limits()

        # Determine frame indices with skip_frames
        total_frames = len(self.varifolds)
        if skip_frames > 1:
            frame_indices = list(range(0, total_frames, skip_frames))
            # Always include the last frame
            if frame_indices[-1] != total_frames - 1:
                frame_indices.append(total_frames - 1)
            print(f"Rendering {len(frame_indices)} frames (skip_frames={skip_frames})")
        else:
            frame_indices = list(range(total_frames))

        # Create FuncAnimation for callers that may use it interactively
        anim = animation.FuncAnimation(
            self.fig,
            self.animate_frame,
            frames=frame_indices,
            interval=interval,
            blit=False,  # Set to False for compatibility
            repeat=True
        )

        # Save video via PyAV (libav in-process; ~5-10× faster than FFMpegWriter)
        if save_path:
            print(f"Saving animation to {save_path}...")
            fps = max(3, min(10, 5 * skip_frames))
            self._save_video_pyav(save_path, frame_indices, fps)
            print(f"Animation saved to {save_path}")

        return anim

    def _save_video_pyav(self, save_path, frame_indices, fps, codec='libx264'):
        """Encode frames to MP4 using PyAV (libav in-process).

        Avoids matplotlib.animation.FFMpegWriter, which spawns ffmpeg as a
        subprocess and pipes per-frame PNG bytes through stdin — both expensive.
        """
        import av

        self.fig.canvas.draw()  # initialize canvas
        w, h = self.fig.canvas.get_width_height()
        out_w, out_h = w - w % 2, h - h % 2  # yuv420p needs even dims

        container = av.open(str(save_path), mode='w')
        stream = container.add_stream(codec, rate=fps)
        stream.width, stream.height = out_w, out_h
        stream.pix_fmt = 'yuv420p'
        if codec == 'libx264':
            stream.options = {'crf': '23', 'preset': 'medium'}

        n = len(frame_indices)
        report_every = max(1, n // 10)
        for i, frame in enumerate(frame_indices):
            self.animate_frame(frame)
            self.fig.canvas.draw()
            rgba = np.asarray(self.fig.canvas.buffer_rgba())
            rgb = np.ascontiguousarray(rgba[:out_h, :out_w, :3])
            av_frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            for pkt in stream.encode(av_frame):
                container.mux(pkt)
            if (i + 1) % report_every == 0:
                print(f"  Encoded {i + 1}/{n} frames")

        for pkt in stream.encode():  # flush
            container.mux(pkt)
        container.close()
        
    def _set_initial_axis_limits(self):
        """Set initial axis limits for consistent scaling from frame 0."""
        if hasattr(self, 'plot_limits'):
            # Boundary limits
            if 'boundary' in self.plot_limits:
                (xmin, xmax), (ymin, ymax) = self.plot_limits['boundary']
                self.axes[0].set_xlim(xmin, xmax)
                self.axes[0].set_ylim(ymin, ymax)

            # Set x-axis limits for all diagnostic plots
            total_steps = len(self.history['perimeters'])
            for ax in [self.axes[1], self.axes[2], self.axes[3]]:
                ax.set_xlim(0, total_steps)

            # Perimeter y-limits
            if 'perimeter' in self.plot_limits:
                self.axes[1].set_ylim(*self.plot_limits['perimeter'])

            # Volume error y-limits
            if 'volume' in self.plot_limits and len(self.history['volumes']) > 0:
                all_volumes = np.array(self.history['volumes'])
                v0 = self.target_volume if hasattr(self, 'target_volume') and self.target_volume != 0 else all_volumes[0]
                err_min = (all_volumes.min() - v0) / v0 * 100
                err_max = (all_volumes.max() - v0) / v0 * 100
                margin = max(abs(err_min), abs(err_max)) * 0.1
                self.axes[2].set_ylim(min(err_min, -margin), max(err_max, margin))

            # Circularity y-limits
            n_perim = len(self.history['perimeters'])
            n_vol = len(self.history['volumes'])
            if n_perim > 0 and n_vol > 0:
                n_common = min(n_perim, n_vol)
                perims = np.array(self.history['perimeters'][:n_common])
                vols = np.array(self.history['volumes'][:n_common])
                valid = perims > 0
                if valid.any():
                    all_circularity = 4 * np.pi * vols[valid] / (perims[valid] ** 2)
                    cmin = min(all_circularity.min(), 0.95)
                    cmax = max(all_circularity.max(), 1.05)
                    self.axes[3].set_ylim(cmin, cmax)

    def save_history(self, save_path):
        """Save evolution history to .npz for replay via --from-history /
        post-hoc snapshot rendering.

        - Constant N across frames: positions has shape ``(N_frames, N, 2)``,
          angles ``(N_frames, N)`` — np.stack path (default).
        - Variable N (e.g. dead-point removal active): positions / angles are
          object arrays of length N_frames where element i is a ``(Nᵢ, 2)`` /
          ``(Nᵢ,)`` ndarray. Replay paths that need constant N should check
          ``positions.dtype == object`` and either pad or sample frames where
          N is fixed.
        """
        if not self.varifolds:
            raise RuntimeError("No varifolds to save; call evolve_varifold() first")

        pos_list = [v.positions.detach().cpu().numpy() for v in self.varifolds]
        ang_list = [v.angles.detach().cpu().numpy() for v in self.varifolds]
        ns = [p.shape[0] for p in pos_list]

        if len(set(ns)) > 1:
            positions = np.empty(len(pos_list), dtype=object)
            angles = np.empty(len(ang_list), dtype=object)
            for i, (p, a) in enumerate(zip(pos_list, ang_list)):
                positions[i] = p
                angles[i] = a
            n_repr = f"variable N ({min(ns)}..{max(ns)})"
        else:
            positions = np.stack(pos_list)
            angles = np.stack(ang_list)
            n_repr = f"{positions.shape[1]} points"

        perimeters = np.array(self.history.get('perimeters', []))
        volumes = np.array(self.history.get('volumes', []))

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # allow_pickle is required for the object-array path; harmless when
        # positions is a plain ndarray (constant-N case).
        np.savez(save_path,
                 positions=positions, angles=angles,
                 perimeters=perimeters, volumes=volumes)
        print(f"History saved: {save_path} ({len(pos_list)} frames, {n_repr})")

    def save_frames(self,
                   output_dir: Path,
                   n_steps: int = 50,
                   frame_format: str = 'png') -> List[Path]:
        """Save individual frames as images.
        
        Args:
            output_dir: Directory to save frames
            n_steps: Number of evolution steps
            frame_format: Image format (png, jpg, etc.)
            
        Returns:
            List of paths to saved frames
        """
        # Evolve varifold
        self.evolve_varifold(n_steps)
        
        # Setup figure
        self.setup_figure()
        
        # Save frames
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        
        print(f"Saving frames to {output_dir}...")
        
        for frame in range(len(self.varifolds)):
            self.animate_frame(frame)
            
            frame_path = output_dir / f"frame_{frame:04d}.{frame_format}"
            self.fig.savefig(frame_path, dpi=150, bbox_inches='tight')
            frame_paths.append(frame_path)
            
            if (frame + 1) % max(1, len(self.varifolds) // 10) == 0:
                print(f"  Saved frame {frame + 1}/{len(self.varifolds)}")
                
        plt.close(self.fig)
        return frame_paths


class CircleAnimator(VarifoldAnimator):
    """Specialized animator for circle steady state verification."""
    
    def __init__(self, radius: float = 1.0, n_points: int = 32, **kwargs):
        from ..shapes import generate_oriented_circle

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        varifold = generate_oriented_circle(
            n_points=n_points, radius=radius, device=device
        )

        super().__init__(varifold, **kwargs)
        
        # Theoretical values
        self.theoretical_perimeter = 2 * np.pi * radius
        self.theoretical_volume = np.pi * radius ** 2
        
    def setup_figure(self):
        """Setup figure with circle-specific layout."""
        setup_matplotlib_style()
        self.fig, self.axes = plt.subplots(2, 2, figsize=self.figsize)
        self.axes = self.axes.flatten()
        
        # Boundary plot
        self.axes[0].set_aspect('equal')
        self.axes[0].set_title('Circle Boundary')
        self.axes[0].grid(True, alpha=0.3)
        self.axes[0].set_xlim(-2.5, 2.5)
        self.axes[0].set_ylim(-2.5, 2.5)
        
        # Perimeter evolution
        self.axes[1].set_title('Perimeter Evolution')
        self.axes[1].set_xlabel('Step')
        self.axes[1].set_ylabel('Perimeter')
        self.axes[1].grid(True, alpha=0.3)
        
        # Area conservation (divergence theorem)
        self.axes[2].set_title('Area Conservation (div. thm.)')
        self.axes[2].set_xlabel('Step')
        self.axes[2].set_ylabel('Area')
        self.axes[2].grid(True, alpha=0.3)

        # Error plot
        self.axes[3].set_title('Relative Errors')
        self.axes[3].set_xlabel('Step')
        self.axes[3].set_ylabel('Error (%)')
        self.axes[3].grid(True, alpha=0.3)
        
    def _update_diagnostic_plot(self, frame: int):
        """Update diagnostic plots for circle."""
        if frame == 0 or len(self.history['perimeters']) == 0:
            return
            
        steps = np.arange(min(frame + 1, len(self.history['perimeters'])))
        perimeters = np.array(self.history['perimeters'][:len(steps)])
        volumes = np.array(self.history['volumes'][:len(steps)])
        
        # Perimeter plot
        if 'perimeter_line' not in self.artists:
            self.artists['perimeter_line'], = self.axes[1].plot(
                steps, perimeters, 'b-', linewidth=2, label='Current'
            )
            self.axes[1].axhline(y=self.theoretical_perimeter, color='b', 
                                linestyle='--', alpha=0.7, label='Theoretical')
            self.axes[1].legend()
        else:
            self.artists['perimeter_line'].set_data(steps, perimeters)
            
        # Volume plot
        if 'volume_line' not in self.artists:
            self.artists['volume_line'], = self.axes[2].plot(
                steps, volumes, 'r-', linewidth=2, label='Current'
            )
            self.axes[2].axhline(y=self.theoretical_volume, color='r',
                                linestyle='--', alpha=0.7, label='Theoretical')
            self.axes[2].legend()
        else:
            self.artists['volume_line'].set_data(steps, volumes)
            
        # Error plot
        p_error = np.abs(perimeters - self.theoretical_perimeter) / self.theoretical_perimeter * 100
        v_error = np.abs(volumes - self.theoretical_volume) / self.theoretical_volume * 100
        
        if 'p_error_line' not in self.artists:
            self.artists['p_error_line'], = self.axes[3].plot(
                steps, p_error, 'b-', label='Perimeter Error', linewidth=2
            )
            self.artists['v_error_line'], = self.axes[3].plot(
                steps, v_error, 'r-', label='Shoelace Area Error', linewidth=2
            )
            self.axes[3].legend()
        else:
            self.artists['p_error_line'].set_data(steps, p_error)
            self.artists['v_error_line'].set_data(steps, v_error)
            
        # Use pre-computed axis limits for consistent scaling
        total_steps = len(self.history['perimeters'])
        
        for ax in [self.axes[1], self.axes[2], self.axes[3]]:
            ax.set_xlim(0, total_steps)
        
        # Set y-limits using pre-computed values or data-based limits
        if 'perimeter' in self.plot_limits:
            # Expand limits to include theoretical value
            pmin, pmax = self.plot_limits['perimeter']
            pmin = min(pmin, self.theoretical_perimeter * 0.99)
            pmax = max(pmax, self.theoretical_perimeter * 1.01)
            self.axes[1].set_ylim(pmin, pmax)
            
        if 'volume' in self.plot_limits:
            # Expand limits to include theoretical value
            vmin, vmax = self.plot_limits['volume']
            vmin = min(vmin, self.theoretical_volume * 0.99)
            vmax = max(vmax, self.theoretical_volume * 1.01)
            self.axes[2].set_ylim(vmin, vmax)
            
        # Error plot limits
        if len(steps) > 0:
            all_p_errors = np.abs(np.array(self.history['perimeters']) - self.theoretical_perimeter) / self.theoretical_perimeter * 100
            all_v_errors = np.abs(np.array(self.history['volumes']) - self.theoretical_volume) / self.theoretical_volume * 100
            max_error = max(all_p_errors.max(), all_v_errors.max()) if len(all_p_errors) > 0 else 1.0
            self.axes[3].set_ylim(0, max_error * 1.1)
            
    def _set_initial_axis_limits(self):
        """Set initial axis limits for CircleAnimator consistent scaling from frame 0."""
        if hasattr(self, 'plot_limits'):
            # Set x-axis limits for all diagnostic plots
            total_steps = len(self.history['perimeters'])
            for ax in [self.axes[1], self.axes[2], self.axes[3]]:
                ax.set_xlim(0, total_steps)
            
            # Set y-axis limits using pre-computed values
            if 'perimeter' in self.plot_limits:
                # Expand limits to include theoretical value
                pmin, pmax = self.plot_limits['perimeter']
                pmin = min(pmin, self.theoretical_perimeter * 0.99)
                pmax = max(pmax, self.theoretical_perimeter * 1.01)
                self.axes[1].set_ylim(pmin, pmax)
                
            if 'volume' in self.plot_limits:
                # Expand limits to include theoretical value
                vmin, vmax = self.plot_limits['volume']
                vmin = min(vmin, self.theoretical_volume * 0.99)
                vmax = max(vmax, self.theoretical_volume * 1.01)
                self.axes[2].set_ylim(vmin, vmax)
                
            # Error plot limits
            if len(self.history['perimeters']) > 0:
                all_p_errors = np.abs(np.array(self.history['perimeters']) - self.theoretical_perimeter) / self.theoretical_perimeter * 100
                all_v_errors = np.abs(np.array(self.history['volumes']) - self.theoretical_volume) / self.theoretical_volume * 100
                max_error = max(all_p_errors.max(), all_v_errors.max()) if len(all_p_errors) > 0 else 1.0
                self.axes[3].set_ylim(0, max_error * 1.1)


def create_circle_animation(radius: float = 1.0,
                          n_points: int = 32,
                          n_steps: int = 30,
                          save_path: Optional[Path] = None) -> animation.FuncAnimation:
    """Create circle steady state animation.
    
    Args:
        radius: Circle radius
        n_points: Number of boundary points
        n_steps: Number of evolution steps
        save_path: Optional path to save video
        
    Returns:
        Animation object
    """
    animator = CircleAnimator(radius=radius, n_points=n_points, figsize=(14, 10))
    return animator.create_animation(n_steps=n_steps, save_path=save_path)


def setup_animation_backend():
    """Setup matplotlib backend for animation."""
    plt.switch_backend('Agg')  # Use non-interactive backend

    try:
        import av
        print(f"PyAV {av.__version__} available for video generation")
    except ImportError:
        print("Warning: PyAV not available. Video generation will fail.")

    return True