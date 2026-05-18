#!/usr/bin/env python3
"""Animation script for flower shape evolving to circle.

This script demonstrates the Mullins-Sekerka flow by showing how a flower-shaped
boundary evolves toward a circular steady state. This validates the algorithm's 
ability to minimize perimeter while conserving volume.

Usage:
    python scripts/visual_tests/animate_flower.py [--output-dir output_dir] [--n-steps 5000]
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.torch.visualization.animation import VarifoldAnimator, setup_animation_backend
from src.torch.shapes import generate_oriented_flower
from src.torch.solver import MMConfig, compute_volume_shoelace


class FlowerAnimator(VarifoldAnimator):
    """Specialized animator for flower → circle evolution."""

    def __init__(self,
                 n_petals: int = 4,
                 inner_radius: float = 0.5,
                 outer_radius: float = 1.0,
                 n_points: int = 64,
                 dtype: torch.dtype = torch.float32,
                 initial_sampling: str = "parameter",
                 precomputed_positions: np.ndarray = None,
                 precomputed_angles: np.ndarray = None,
                 precomputed_perimeters: np.ndarray = None,
                 precomputed_volumes: np.ndarray = None,
                 **kwargs):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = dtype

        # Generate flower-shaped boundary with standard parameters
        # These parameters are consistent with other test scripts
        varifold = generate_oriented_flower(
            n_points=n_points,
            n_petals=n_petals,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            device=device,
            dtype=dtype,
            initial_sampling=initial_sampling,
        )

        super().__init__(varifold, **kwargs)

        # Store precomputed data if provided
        self._precomputed_positions = precomputed_positions
        self._precomputed_angles = precomputed_angles
        self._precomputed_perimeters = precomputed_perimeters
        self._precomputed_volumes = precomputed_volumes
        
        # Store flower parameters
        self.n_petals = n_petals
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        
        # Compute target circle from volume conservation: π r² = V₀
        self.initial_volume = compute_volume_shoelace(varifold)
        self.target_radius = np.sqrt(self.initial_volume / np.pi)
        self.theoretical_perimeter = 2 * np.pi * self.target_radius
        
    def setup_figure(self):
        """Setup figure with flower-specific layout."""
        from src.torch.visualization.static import setup_matplotlib_style
        
        setup_matplotlib_style()
        self.fig, self.axes = plt.subplots(2, 2, figsize=self.figsize)
        self.axes = self.axes.flatten()
        
        # Boundary evolution plot
        self.axes[0].set_aspect('equal')
        self.axes[0].set_title('Flower → Circle Evolution')
        self.axes[0].grid(True, alpha=0.3)
        self.axes[0].set_xlim(-2.5, 2.5)
        self.axes[0].set_ylim(-2.5, 2.5)
        
        # Perimeter evolution
        self.axes[1].set_title('Perimeter Evolution')
        self.axes[1].set_xlabel('Step')
        self.axes[1].set_ylabel('Perimeter')
        self.axes[1].grid(True, alpha=0.3)
        
        # Volume conservation (divergence theorem)
        self.axes[2].set_title('Volume Conservation (div. thm.)')
        self.axes[2].set_xlabel('Step')
        self.axes[2].set_ylabel(r'$(V - V_0) / V_0$ [\%]')
        self.axes[2].grid(True, alpha=0.3)
        
        # Shape metric (circularity)
        self.axes[3].set_title('Circularity Metric')
        self.axes[3].set_xlabel('Step')
        self.axes[3].set_ylabel(r'$4\pi \cdot \text{Volume}/\text{Perimeter}^2$')
        self.axes[3].grid(True, alpha=0.3)

        
    def _update_diagnostic_plot(self, frame: int):
        """Update diagnostic plots for flower evolution."""
        if frame == 0 or len(self.history['perimeters']) == 0:
            return
            
        steps = np.arange(min(frame + 1, len(self.history['perimeters'])))
        perimeters = np.array(self.history['perimeters'][:len(steps)])
        volumes = np.array(self.history['volumes'][:len(steps)])
        
        # Perimeter plot with target
        if 'perimeter_line' not in self.artists:
            self.artists['perimeter_line'], = self.axes[1].plot(
                steps, perimeters, 'b-', linewidth=2, label='Current'
            )
            self.axes[1].axhline(y=self.theoretical_perimeter, color='b', 
                                linestyle='--', alpha=0.7, label='Circle Target')
            self.axes[1].legend()
        else:
            self.artists['perimeter_line'].set_data(steps, perimeters)
            
        # Volume conservation: relative error (V - V₀) / V₀ in percent
        volume_error_pct = (volumes - self.initial_volume) / self.initial_volume * 100
        if 'volume_line' not in self.artists:
            self.artists['volume_line'], = self.axes[2].plot(
                steps, volume_error_pct, 'r-', linewidth=2, label='Relative error'
            )
            self.axes[2].axhline(y=0.0, color='r',
                                linestyle='--', alpha=0.7, label='Exact conservation')
            self.axes[2].legend()
        else:
            self.artists['volume_line'].set_data(steps, volume_error_pct)
            
        # Circularity metric (1.0 = perfect circle)
        circularity = 4 * np.pi * volumes / (perimeters ** 2)
        
        if 'circularity_line' not in self.artists:
            self.artists['circularity_line'], = self.axes[3].plot(
                steps, circularity, 'g-', linewidth=2, label='Circularity'
            )
            self.axes[3].axhline(y=1.0, color='g', 
                                linestyle='--', alpha=0.7, label='Perfect Circle')
            self.axes[3].legend()
        else:
            self.artists['circularity_line'].set_data(steps, circularity)
            
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
            vmin, vmax = self.plot_limits['volume']
            err_min = (vmin - self.initial_volume) / self.initial_volume * 100
            err_max = (vmax - self.initial_volume) / self.initial_volume * 100
            margin = max(abs(err_min), abs(err_max)) * 0.1
            self.axes[2].set_ylim(min(err_min, -margin), max(err_max, margin))

        # Circularity plot limits
        if len(steps) > 0:
            all_circularity = 4 * np.pi * np.array(self.history['volumes']) / (np.array(self.history['perimeters']) ** 2)
            cmin = min(all_circularity.min(), 0.95)
            cmax = max(all_circularity.max(), 1.05)
            self.axes[3].set_ylim(cmin, cmax)
            
    def _set_initial_axis_limits(self):
        """Set initial axis limits for FlowerAnimator consistent scaling from frame 0."""
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
                vmin, vmax = self.plot_limits['volume']
                err_min = (vmin - self.initial_volume) / self.initial_volume
                err_max = (vmax - self.initial_volume) / self.initial_volume
                margin = max(abs(err_min), abs(err_max)) * 0.1
                self.axes[2].set_ylim(min(err_min, -margin), max(err_max, margin))

            # Circularity plot limits
            if len(self.history['perimeters']) > 0:
                all_circularity = 4 * np.pi * np.array(self.history['volumes']) / (np.array(self.history['perimeters']) ** 2)
                cmin = min(all_circularity.min(), 0.95)
                cmax = max(all_circularity.max(), 1.05)
                self.axes[3].set_ylim(cmin, cmax)

    def load_precomputed_history(self):
        """Load pre-computed history without running simulation."""
        if self._precomputed_positions is None:
            return False

        from src.torch.oriented_varifold import OrientedPointCloudVarifold

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        positions = self._precomputed_positions
        angles = self._precomputed_angles

        # Clear and rebuild varifolds list
        self.varifolds = []
        for i in range(len(positions)):
            varifold = OrientedPointCloudVarifold(
                positions=torch.tensor(positions[i], device=device, dtype=self.dtype),
                angles=torch.tensor(angles[i], device=device, dtype=self.dtype),
            )
            self.varifolds.append(varifold)

        # Load history metrics
        if self._precomputed_perimeters is not None:
            perimeters = self._precomputed_perimeters.tolist()
        else:
            perimeters = []

        if self._precomputed_volumes is not None:
            volumes = self._precomputed_volumes.tolist()
        else:
            volumes = []

        self.history = {
            'perimeters': perimeters,
            'volumes': volumes,
            'times': [0.0] * len(volumes),
        }

        # Pre-compute axis limits
        self._compute_plot_limits()

        print(f"Loaded {len(self.varifolds)} frames from pre-computed history")
        return True


def main():
    """Main function for flower evolution animation."""
    parser = argparse.ArgumentParser(description='Flower → Circle Evolution Animation')
    parser.add_argument('--output-dir', type=str, default='scripts/outputs/movies',
                       help='Output directory for videos and data')
    parser.add_argument('--n-steps', type=int, default=5000,
                       help='Number of evolution steps')
    parser.add_argument('--petals', type=int, default=4,
                       help='Number of flower petals')
    parser.add_argument('--inner-radius', type=float, default=0.5,
                       help='Inner radius of flower petals')
    parser.add_argument('--outer-radius', type=float, default=1.0,
                       help='Outer radius of flower petals')
    parser.add_argument('--n-points', type=int, default=128,
                       help='Number of boundary points')
    parser.add_argument('--frames-only', action='store_true',
                       help='Save frames only, no MP4')
    parser.add_argument('--no-video', action='store_true',
                       help='Skip video generation')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'], help='Computation device')
    parser.add_argument('--time-step', type=float, default=1e-5,
                       help='Time step h for MM scheme')
    parser.add_argument('--bem-method', type=str, default='point',
                       choices=['panel', 'point'],
                       help='BEM matrix computation method (point=recommended, panel=analytical)')
    parser.add_argument('--bem-epsilon-scale', type=float, default=0.1,
                       help='Epsilon scale for point method')
    parser.add_argument('--bem-n-endpoints', type=int, default=3,
                       help='Number of endpoints for BEM (1=center only, 3=endpoint collocation)')
    parser.add_argument('--use-unit-coherence', action='store_true',
                       help='Use coherence=1 instead of computed value')
    parser.add_argument('--optimizer-method', type=str, default='trust-ncg',
                       help='MM-step optimizer (matches the two-ellipses '
                            'showcase / ablation; default: trust-ncg)')
    parser.add_argument('--gtol', type=float, default=1e-8,
                       help='Optimizer gradient tolerance (MMConfig.optimizer_tol)')
    parser.add_argument('--dtype', type=str, default='float64',
                       choices=['float32', 'float64'],
                       help='Data type for computation')
    parser.add_argument('--output-name', type=str, default=None,
                       help='Output video filename (default: flower_n{n_points}_s{n_steps}.mp4)')
    parser.add_argument('--from-history', type=str, default=None,
                       help='Load pre-computed history from .npz file (skips simulation)')
    parser.add_argument('--save-history', type=str, default=None,
                       help='Save evolution history to this .npz path after simulation')
    parser.add_argument('--initial-sampling', type=str, default='arc_length',
                       choices=['parameter', 'arc_length', 'mass_uniform'],
                       help='Initial sampling strategy (default: arc_length)')
    parser.add_argument('--skip-frames', type=int, default=10,
                       help='Only render every N-th frame for faster video generation (default 10 keeps the video ~25 MB instead of ~225 MB at h=1e-5/n=5000)')
    # Redistribute options
    parser.add_argument('--redistribute', action=argparse.BooleanOptionalAction, default=True,
                       help='Enable point redistribution (default on; pass --no-redistribute to disable)')
    parser.add_argument('--redistribute-interval', type=int, default=1,
                       help='Redistribute every N steps')
    parser.add_argument('--redistribute-n-iters', type=int, default=10,
                       help='Number of redistribute iterations')
    parser.add_argument('--redistribute-step-size', type=float, default=0.01,
                       help='Redistribute step size')
    parser.add_argument('--redistribute-tol', type=float, default=1e-4,
                       help='Redistribute convergence tolerance')

    args = parser.parse_args()
    
    # Setup
    setup_animation_backend()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        
    print(f"Using device: {device}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Flower → Circle Evolution Animation ===")
    if args.from_history:
        print(f"Loading from history: {args.from_history}")
    else:
        print(f"Petals: {args.petals}")
        print(f"Inner radius: {args.inner_radius}")
        print(f"Outer radius: {args.outer_radius}")
        print(f"Points: {args.n_points}")
        print(f"Steps: {args.n_steps}")
        print(f"Time step: {args.time_step}")
        print(f"BEM method: {args.bem_method}")
        if args.bem_method == 'point':
            print(f"  Epsilon scale: {args.bem_epsilon_scale}")
        print(f"Use unit coherence: {args.use_unit_coherence}")
        print(f"Dtype: {args.dtype}")
    print(f"Skip frames: {args.skip_frames}")
    print(f"Output: {output_dir}")

    # Create animator
    start_time = time.time()

    try:
        # Determine dtype
        dtype = torch.float64 if args.dtype == 'float64' else torch.float32

        # Load from history if specified
        precomputed_positions = None
        precomputed_angles = None
        precomputed_perimeters = None
        precomputed_volumes = None

        if args.from_history:
            history_data = np.load(args.from_history)
            precomputed_positions = history_data['positions']
            precomputed_angles = history_data['angles']
            precomputed_perimeters = history_data['perimeters']
            precomputed_volumes = history_data['volumes']
            # Infer n_points from history
            args.n_points = precomputed_positions.shape[1]
            print(f"Loaded history: {len(precomputed_positions)} frames, {args.n_points} points")

        # Create config for flower evolution (BEM-based)
        config = MMConfig(
            time_step=args.time_step,
            bem_method=args.bem_method,
            bem_epsilon_scale=args.bem_epsilon_scale,
            bem_n_endpoints=args.bem_n_endpoints,
            use_unit_coherence=args.use_unit_coherence,
            optimizer_method=args.optimizer_method,
            optimizer_tol=args.gtol,
            redistribute=args.redistribute,
            redistribute_interval=args.redistribute_interval,
            redistribute_n_iters=args.redistribute_n_iters,
            redistribute_step_size=args.redistribute_step_size,
            redistribute_tol=args.redistribute_tol,
        )

        animator = FlowerAnimator(
            n_petals=args.petals,
            inner_radius=args.inner_radius,
            outer_radius=args.outer_radius,
            n_points=args.n_points,
            dtype=dtype,
            initial_sampling=args.initial_sampling,
            precomputed_positions=precomputed_positions,
            precomputed_angles=precomputed_angles,
            precomputed_perimeters=precomputed_perimeters,
            precomputed_volumes=precomputed_volumes,
            figsize=(14, 10),
            config=config
        )

        # Generate evolution data or load from history
        print(f"\n--- Evolving Flower ---")
        evolution_start = time.time()

        if args.from_history:
            animator.load_precomputed_history()
        else:
            animator.evolve_varifold(args.n_steps)
            if args.save_history:
                animator.save_history(args.save_history)

        evolution_time = time.time() - evolution_start

        print(f"Evolution completed in {evolution_time:.2f}s")
        
        # Analyze results
        print(f"\n--- Results Analysis ---")
        if len(animator.history['perimeters']) > 0:
            initial_P = animator.history['perimeters'][0]
            final_P = animator.history['perimeters'][-1]
            target_P = animator.theoretical_perimeter
            
            perimeter_reduction = (initial_P - final_P) / initial_P * 100
            perimeter_error = abs(final_P - target_P) / target_P * 100
            
            print(f"Perimeter: {initial_P:.4f} → {final_P:.4f}")
            print(f"  Reduction: {perimeter_reduction:.1f}%")
            print(f"  Target: {target_P:.4f}")
            print(f"  Final error: {perimeter_error:.3f}%")
            
            initial_V = animator.history['volumes'][0] 
            final_V = animator.history['volumes'][-1]
            volume_change = abs(final_V - initial_V) / initial_V * 100
            
            print(f"Volume: {initial_V:.4f} → {final_V:.4f}")
            print(f"  Conservation error: {volume_change:.3f}%")
            
            # Circularity analysis
            initial_circularity = 4 * np.pi * animator.history['volumes'][0] / (animator.history['perimeters'][0] ** 2)
            final_circularity = 4 * np.pi * animator.history['volumes'][-1] / (animator.history['perimeters'][-1] ** 2)
            
            print(f"Circularity: {initial_circularity:.4f} → {final_circularity:.4f}")
            print(f"  Progress to circle: {(final_circularity - initial_circularity) / (1.0 - initial_circularity) * 100:.1f}%")
        
        # Save data
        data = {
            'parameters': {
                'n_petals': args.petals,
                'inner_radius': args.inner_radius,
                'outer_radius': args.outer_radius,
                'n_points': args.n_points,
                'n_steps': args.n_steps,
                'device': device,
                'time_step': args.time_step,
                'bem_method': args.bem_method,
                'bem_epsilon_scale': args.bem_epsilon_scale,
                'use_unit_coherence': args.use_unit_coherence,
                'dtype': args.dtype,
            },
            'theoretical': {
                'initial_volume': float(animator.initial_volume),
                'target_radius': float(animator.target_radius),
                'target_perimeter': float(animator.theoretical_perimeter),
            },
            'history': {
                'perimeters': [float(p) for p in animator.history['perimeters']],
                'volumes': [float(v) for v in animator.history['volumes']],
                'n_completed': len(animator.history['perimeters']),
            },
            'timing': {
                'evolution_time': evolution_time,
                'total_time': time.time() - start_time,
            }
        }
        
        data_path = output_dir / 'flower_data.json'
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved: {data_path}")
        
        if args.frames_only:
            # Save individual frames
            print(f"\n--- Saving Frames ---")
            frame_dir = output_dir / "flower_frames"
            animator.save_frames(frame_dir, args.n_steps)
            print(f"Frames saved to {frame_dir}")
            
        elif not args.no_video:
            # Create and save animation
            print(f"\n--- Creating Animation ---")
            n_completed = len(animator.varifolds) - 1
            if args.output_name:
                video_path = output_dir / args.output_name
            else:
                video_path = output_dir / f'flower_n{args.n_points}_dt{args.time_step}_e{config.bem_n_endpoints}_s{n_completed}.mp4'

            animation_start = time.time()
            animator.create_animation(
                n_steps=args.n_steps,
                interval=300,  # Slower for better viewing
                save_path=video_path,
                skip_frames=args.skip_frames
            )
            
            animation_time = time.time() - animation_start
            print(f"Animation created in {animation_time:.2f}s")
            print(f"Video saved: {video_path}")
            
            # Also save final frame
            final_frame_path = output_dir / 'flower_final_frame.png'
            animator.animate_frame(len(animator.varifolds) - 1)
            animator.fig.savefig(final_frame_path, dpi=300, bbox_inches='tight')
            print(f"Final frame saved: {final_frame_path}")
        
        total_time = time.time() - start_time
        print(f"\n✓ Total time: {total_time:.2f}s")
        
        plt.close('all')
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())