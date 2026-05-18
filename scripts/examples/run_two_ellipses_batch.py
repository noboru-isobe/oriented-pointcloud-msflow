#!/usr/bin/env python3
"""Batch runner for two_ellipses experiments.

Runs specified variants of (redistribute, use_unit_coherence),
producing videos, data JSON, and final frame PNGs for each variant.

Usage:
    # Quick smoke test
    uv run python scripts/experiments/run_two_ellipses_batch.py --n-steps 10 --no-video

    # Full run (default: rdT_ucF, gap=0.1, compile=False)
    uv run python scripts/experiments/run_two_ellipses_batch.py --n-steps 10000 --skip-frames 50
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.torch.visualization.animation import VarifoldAnimator, setup_animation_backend
from src.torch.visualization.static import setup_matplotlib_style
from src.torch.shapes import generate_oriented_two_ellipses
from src.torch.solver import MMConfig
from src.torch.oriented_varifold import OrientedPointCloudVarifold


# ---------------------------------------------------------------------------
# TwoEllipsesAnimator
# ---------------------------------------------------------------------------

class TwoEllipsesAnimator(VarifoldAnimator):
    """Animator specialised for two-ellipse multi-component evolution."""

    def __init__(self,
                 n_per_ellipse: int = 64,
                 a1: float = 0.4, b1: float = 1.0,
                 center1=(-0.6, 0.0),
                 a2: float = 0.4, b2: float = 1.0,
                 center2=(0.6, 0.0),
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float64,
                 **kwargs):
        self.dtype = dtype
        self.n_per_ellipse = n_per_ellipse

        varifold = generate_oriented_two_ellipses(
            n_per_ellipse=n_per_ellipse,
            a1=a1, b1=b1, center1=center1,
            a2=a2, b2=b2, center2=center2,
            device=device, dtype=dtype,
        )

        super().__init__(varifold, **kwargs)

        # Ellipse geometry
        self.a1, self.b1, self.center1 = a1, b1, center1
        self.a2, self.b2, self.center2 = a2, b2, center2

        # Theoretical volume (sum of two ellipses)
        self.theoretical_volume = np.pi * a1 * b1 + np.pi * a2 * b2

        # Initial perimeter (Ramanujan approx for each ellipse)
        def ramanujan(a, b):
            return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

        self.initial_perimeter_theoretical = ramanujan(a1, b1) + ramanujan(a2, b2)

        # Volume-conserving target: each ellipse -> circle with same area
        r1 = np.sqrt(a1 * b1)
        r2 = np.sqrt(a2 * b2)
        self.theoretical_perimeter_target = 2 * np.pi * r1 + 2 * np.pi * r2

    # -- figure setup -----------------------------------------------------

    def setup_figure(self):
        setup_matplotlib_style()
        self.fig, self.axes = plt.subplots(2, 2, figsize=self.figsize)
        self.axes = self.axes.flatten()

        # Boundary
        self.axes[0].set_aspect('equal')
        self.axes[0].set_title('Two Ellipses Evolution')
        self.axes[0].grid(True, alpha=0.3)
        x_lo = min(self.center1[0] - self.a1, self.center2[0] - self.a2) - 0.3
        x_hi = max(self.center1[0] + self.a1, self.center2[0] + self.a2) + 0.3
        y_lo = min(self.center1[1] - self.b1, self.center2[1] - self.b2) - 0.3
        y_hi = max(self.center1[1] + self.b1, self.center2[1] + self.b2) + 0.3
        self.axes[0].set_xlim(x_lo, x_hi)
        self.axes[0].set_ylim(y_lo, y_hi)

        # Perimeter
        self.axes[1].set_title('Perimeter Evolution')
        self.axes[1].set_xlabel('Step')
        self.axes[1].set_ylabel('Perimeter')
        self.axes[1].grid(True, alpha=0.3)

        # Volume error
        self.axes[2].set_title('Volume Conservation (div. thm.)')
        self.axes[2].set_xlabel('Step')
        self.axes[2].set_ylabel(r'$(V - V_0)/V_0$ [%]')
        self.axes[2].grid(True, alpha=0.3)

        # Circularity
        self.axes[3].set_title('Circularity')
        self.axes[3].set_xlabel('Step')
        self.axes[3].set_ylabel(r'$4\pi V / P^2$')
        self.axes[3].grid(True, alpha=0.3)

    # -- diagnostic update ------------------------------------------------

    def _update_diagnostic_plot(self, frame: int):
        if frame == 0 or len(self.history['perimeters']) == 0:
            return

        steps = np.arange(min(frame + 1, len(self.history['perimeters'])))
        perimeters = np.array(self.history['perimeters'][:len(steps)])
        volumes = np.array(self.history['volumes'][:len(steps)])

        # --- Perimeter ---
        if 'perimeter_line' not in self.artists:
            self.artists['perimeter_line'], = self.axes[1].plot(
                steps, perimeters, 'b-', linewidth=2, label='Current')
            self.axes[1].axhline(y=self.initial_perimeter_theoretical,
                                 color='gray', ls=':', alpha=0.7, label='Initial (theory)')
            self.axes[1].axhline(y=self.theoretical_perimeter_target,
                                 color='b', ls='--', alpha=0.7, label='Circle target')
            self.axes[1].legend(fontsize=8)
        else:
            self.artists['perimeter_line'].set_data(steps, perimeters)

        # --- Volume error (relative to initial computed volume) ---
        V0 = self.history['volumes'][0]
        if V0 != 0:
            vol_err_pct = (volumes - V0) / V0 * 100
        else:
            vol_err_pct = np.zeros_like(volumes)
        if 'volume_line' not in self.artists:
            self.artists['volume_line'], = self.axes[2].plot(
                steps, vol_err_pct, 'r-', linewidth=2, label='Relative error')
            self.axes[2].axhline(y=0.0, color='r', ls='--', alpha=0.7,
                                 label='Exact conservation')
            self.axes[2].legend(fontsize=8)
        else:
            self.artists['volume_line'].set_data(steps, vol_err_pct)

        # --- Circularity ---
        with np.errstate(divide='ignore', invalid='ignore'):
            circularity = np.where(perimeters > 0,
                                   4 * np.pi * volumes / (perimeters ** 2), np.nan)
        if 'circularity_line' not in self.artists:
            self.artists['circularity_line'], = self.axes[3].plot(
                steps, circularity, 'g-', linewidth=2, label='Circularity')
            self.axes[3].axhline(y=1.0, color='g', ls='--', alpha=0.7,
                                 label='Perfect circle')
            self.axes[3].legend(fontsize=8)
        else:
            self.artists['circularity_line'].set_data(steps, circularity)

        # --- Axis limits ---
        total_steps = len(self.history['perimeters'])
        for ax in [self.axes[1], self.axes[2], self.axes[3]]:
            ax.set_xlim(0, total_steps)

        if 'perimeter' in self.plot_limits:
            pmin, pmax = self.plot_limits['perimeter']
            pmin = min(pmin, self.theoretical_perimeter_target * 0.98)
            pmax = max(pmax, self.initial_perimeter_theoretical * 1.02)
            self._safe_set_ylim(self.axes[1], pmin, pmax)

        if 'volume' in self.plot_limits and V0 != 0:
            vmin, vmax = self.plot_limits['volume']
            err_min = (vmin - V0) / V0 * 100
            err_max = (vmax - V0) / V0 * 100
            margin = max(abs(err_min), abs(err_max)) * 0.1
            self._safe_set_ylim(self.axes[2], min(err_min, -margin), max(err_max, margin))

        if len(steps) > 0:
            perimeters_all = np.array(self.history['perimeters'])
            valid = perimeters_all > 0
            if valid.any():
                all_circ = (4 * np.pi * np.array(self.history['volumes'])[valid]
                            / (perimeters_all[valid] ** 2))
                finite = np.isfinite(all_circ)
                if finite.any():
                    cmin = all_circ[finite].min() * 0.98
                    cmax = max(all_circ[finite].max(), 1.0) * 1.02
                    self._safe_set_ylim(self.axes[3], cmin, cmax)

    @staticmethod
    def _safe_set_ylim(ax, ymin, ymax):
        """Set ylim only if both values are finite."""
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
            ax.set_ylim(ymin, ymax)

    def _set_initial_axis_limits(self):
        if not hasattr(self, 'plot_limits'):
            return
        total_steps = len(self.history['perimeters'])
        for ax in [self.axes[1], self.axes[2], self.axes[3]]:
            ax.set_xlim(0, total_steps)

        if 'perimeter' in self.plot_limits:
            pmin, pmax = self.plot_limits['perimeter']
            pmin = min(pmin, self.theoretical_perimeter_target * 0.98)
            pmax = max(pmax, self.initial_perimeter_theoretical * 1.02)
            self._safe_set_ylim(self.axes[1], pmin, pmax)

        if 'volume' in self.plot_limits and len(self.history['volumes']) > 0:
            V0 = self.history['volumes'][0]
            if V0 != 0:
                vmin, vmax = self.plot_limits['volume']
                err_min = (vmin - V0) / V0 * 100
                err_max = (vmax - V0) / V0 * 100
                margin = max(abs(err_min), abs(err_max)) * 0.1
                self._safe_set_ylim(self.axes[2], min(err_min, -margin), max(err_max, margin))

        if len(self.history['perimeters']) > 0:
            perimeters = np.array(self.history['perimeters'])
            valid = perimeters > 0
            if valid.any():
                all_circ = (4 * np.pi * np.array(self.history['volumes'])[valid]
                            / (perimeters[valid] ** 2))
                finite = np.isfinite(all_circ)
                if finite.any():
                    cmin = all_circ[finite].min() * 0.98
                    cmax = max(all_circ[finite].max(), 1.0) * 1.02
                    self._safe_set_ylim(self.axes[3], cmin, cmax)


# ---------------------------------------------------------------------------
# Single-variant runner
# ---------------------------------------------------------------------------

def run_single(redistribute: bool, use_unit_coherence: bool,
               args) -> dict:
    """Run one (rd, uc) variant and return timing / result dict."""
    rd_tag = "T" if redistribute else "F"
    uc_tag = "T" if use_unit_coherence else "F"
    label = f"rd{rd_tag}_uc{uc_tag}"

    print(f"\n{'=' * 60}")
    print(f"Variant: {label}  (compile={args.compile})")
    print(f"{'=' * 60}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # File names
    dp_tag = f"_dp{args.dead_threshold}" if args.remove_dead else ""
    comp_tag = "T" if args.compile else "F"
    opt_tag = f"_opt-{args.optimizer_method}" if args.optimizer_method != "bfgs" else ""
    gtol_tag = f"_gtol{args.gtol:.0e}" if args.gtol != 1e-6 else ""
    rd_iters_tag = f"_ri{args.redistribute_n_iters}" if args.redistribute_n_iters != 10 else ""
    rd_post_tag = f"_rp{args.redistribute_n_iters_after_removal}" if args.redistribute_n_iters_after_removal > 0 else ""
    be_tag = f"_be-{args.backend}" if args.backend != "naive" else ""
    base = f"two_ellipses_s{args.n_steps}_rd{rd_tag}_uc{uc_tag}_k{args.mass_knn_k}{dp_tag}_comp{comp_tag}_gap{args.gap}{opt_tag}{gtol_tag}{rd_iters_tag}{rd_post_tag}{be_tag}"
    video_path = output_dir / f"{base}.mp4"
    data_path = output_dir / f"{base}_data.json"
    frame_path = output_dir / f"{base}_final_frame.png"

    dtype = torch.float64 if args.dtype == 'float64' else torch.float32

    config = MMConfig(
        time_step=args.time_step,
        bem_method=args.bem_method,
        bem_epsilon_scale=args.bem_epsilon_scale,
        compile=args.compile,
        optimizer_method=args.optimizer_method,
        optimizer_tol=args.gtol,
        redistribute=redistribute,
        redistribute_n_iters=args.redistribute_n_iters,
        redistribute_n_iters_after_removal=args.redistribute_n_iters_after_removal,
        use_unit_coherence=use_unit_coherence,
        mass_knn_k=args.mass_knn_k,
        remove_dead_points=args.remove_dead,
        dead_point_threshold=args.dead_threshold,
        backend=args.backend,
    )

    # Compute centers from gap
    a1, a2 = 0.4, 0.4
    center_x = (a1 + a2 + args.gap) / 2
    center1 = (-center_x, 0.0)
    center2 = (center_x, 0.0)

    animator = TwoEllipsesAnimator(
        n_per_ellipse=args.n_per_ellipse,
        center1=center1,
        center2=center2,
        device=args.device,
        dtype=dtype,
        figsize=(14, 10),
        config=config,
    )

    # --- Evolve ---
    t0 = time.time()
    animator.evolve_varifold(args.n_steps)
    evolution_time = time.time() - t0
    print(f"  Evolution: {evolution_time:.2f}s")

    n_completed = len(animator.varifolds) - 1

    # --- Save history (.npz) for --from-history replay ---
    if args.save_history:
        history_path = output_dir / f"{base}.npz"
        try:
            animator.save_history(history_path)
        except RuntimeError as e:
            print(f"  [save_history skipped] {e}")

    # --- Analyse ---
    result = {'label': label, 'evolution_time': evolution_time, 'n_completed': n_completed}
    if len(animator.history['perimeters']) > 1:
        P0 = animator.history['perimeters'][0]
        Pf = animator.history['perimeters'][-1]
        V0 = animator.history['volumes'][0]
        Vf = animator.history['volumes'][-1]
        result['initial_perimeter'] = P0
        result['final_perimeter'] = Pf
        result['perimeter_reduction_pct'] = (P0 - Pf) / P0 * 100 if P0 != 0 else 0.0
        result['initial_volume'] = V0
        result['final_volume'] = Vf
        result['volume_error_pct'] = abs(Vf - V0) / V0 * 100 if V0 != 0 else 0.0
        result['initial_circularity'] = 4 * np.pi * V0 / (P0 ** 2) if P0 != 0 else 0.0
        result['final_circularity'] = 4 * np.pi * Vf / (Pf ** 2) if Pf != 0 else 0.0

        print(f"  Perimeter: {P0:.4f} -> {Pf:.4f}  (reduction {result['perimeter_reduction_pct']:.2f}%)")
        print(f"  Volume err: {result['volume_error_pct']:.4f}%")
        print(f"  Circularity: {result['initial_circularity']:.4f} -> {result['final_circularity']:.4f}")

    # --- Save JSON ---
    data = {
        'parameters': {
            'n_per_ellipse': args.n_per_ellipse,
            'n_steps': args.n_steps,
            'time_step': args.time_step,
            'gap': args.gap,
            'compile': args.compile,
            'redistribute': redistribute,
            'use_unit_coherence': use_unit_coherence,
            'dtype': args.dtype,
            'device': str(animator.initial_varifold.positions.device),
            'bem_method': args.bem_method,
            'bem_epsilon_scale': args.bem_epsilon_scale,
        },
        'theoretical': {
            'initial_perimeter': float(animator.initial_perimeter_theoretical),
            'target_perimeter': float(animator.theoretical_perimeter_target),
            'volume': float(animator.theoretical_volume),
        },
        'history': {
            'perimeters': [float(p) for p in animator.history['perimeters']],
            'volumes': [float(v) for v in animator.history['volumes']],
            'n_completed': n_completed,
        },
        'timing': {
            'evolution_time': evolution_time,
        },
    }
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Data saved: {data_path}")

    # --- Video / final frame ---
    if not args.no_video:
        t1 = time.time()
        animator.create_animation(
            n_steps=args.n_steps,
            interval=300,
            save_path=video_path,
            skip_frames=args.skip_frames,
        )
        render_time = time.time() - t1
        print(f"  Video saved: {video_path}  ({render_time:.2f}s)")
        result['render_time'] = render_time

        # Final frame
        animator.animate_frame(len(animator.varifolds) - 1)
        animator.fig.savefig(frame_path, dpi=300, bbox_inches='tight')
        print(f"  Final frame: {frame_path}")

    plt.close('all')
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Two-ellipses batch experiment runner')
    parser.add_argument('--n-steps', type=int, default=10000)
    parser.add_argument('--n-per-ellipse', type=int, default=64)
    parser.add_argument('--time-step', type=float, default=1e-5)
    parser.add_argument('--gap', type=float, default=0.1,
                        help='Gap between two ellipses (default: 0.1)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--dtype', type=str, default='float64',
                        choices=['float32', 'float64'])
    parser.add_argument('--output-dir', type=str,
                        default='scripts/outputs/experiments')
    parser.add_argument('--skip-frames', type=int, default=50)
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--save-history', action='store_true',
                        help='Save evolution history to .npz next to mp4 (constant N required)')
    parser.add_argument('--bem-method', type=str, default='point',
                        choices=['panel', 'point'])
    parser.add_argument('--bem-epsilon-scale', type=float, default=0.1)
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile (default: off)')
    parser.add_argument('--optimizer-method', type=str, default='bfgs',
                        help='Optimizer method (default: bfgs)')
    parser.add_argument('--gtol', type=float, default=1e-6,
                        help='Optimizer gradient tolerance (default: 1e-6)')
    parser.add_argument('--redistribute', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Enable redistribute (default: on)')
    parser.add_argument('--redistribute-n-iters', type=int, default=10,
                        help='Redistribute iterations per step (default: 10)')
    parser.add_argument('--redistribute-n-iters-after-removal', type=int, default=0,
                        help='Extra redistribute iters after dead point removal (default: 0 = off)')
    parser.add_argument('--use-unit-coherence', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Use unit coherence (default: off)')
    parser.add_argument('--mass-knn-k', type=int, default=10,
                        help='k for kNN distance in δ computation (default: 10)')
    parser.add_argument('--remove-dead', action='store_true',
                        help='Enable dead point removal')
    parser.add_argument('--dead-threshold', type=float, default=1e-4,
                        help='Dead point threshold (default: 1e-4)')
    parser.add_argument('--backend', type=str, default='naive',
                        choices=['naive', 'keops'],
                        help='Pairwise kernel backend for coherence; default naive (faster at small N)')
    args = parser.parse_args()

    setup_animation_backend()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {args.device}")

    # --- Warmup (only if compile enabled) ---
    if args.compile:
        print("\n--- Warmup (compile cache) ---")
        dtype = torch.float64 if args.dtype == 'float64' else torch.float32
        warmup_config = MMConfig(
            time_step=args.time_step,
            bem_method=args.bem_method,
            bem_epsilon_scale=args.bem_epsilon_scale,
            compile=True,
        )
        warmup_animator = TwoEllipsesAnimator(
            n_per_ellipse=args.n_per_ellipse,
            device=args.device,
            dtype=dtype,
            figsize=(14, 10),
            config=warmup_config,
        )
        t_warmup = time.time()
        warmup_animator.evolve_varifold(3)
        warmup_time = time.time() - t_warmup
        print(f"Warmup done: {warmup_time:.1f}s (3 steps)")
        del warmup_animator
        plt.close('all')

    VARIANTS = [(args.redistribute, args.use_unit_coherence)]

    results = []
    total_start = time.time()

    for rd, uc in VARIANTS:
        try:
            result = run_single(rd, uc, args)
        except Exception as e:
            rd_tag = "T" if rd else "F"
            uc_tag = "T" if uc else "F"
            label = f"rd{rd_tag}_uc{uc_tag}"
            print(f"  [FAILED] {label}: {e}")
            result = {'label': label, 'evolution_time': 0, 'n_completed': 0}
        results.append(result)

    total_time = time.time() - total_start

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"Timing Summary (compile={args.compile})")
    print(f"{'=' * 70}")
    print(f"{'Variant':<12} {'Steps':>8} {'Evolve(s)':>10} {'Render(s)':>10} "
          f"{'P reduction':>12} {'Vol err':>10}")
    print('-' * 70)
    for r in results:
        render = f"{r.get('render_time', 0):.1f}"
        p_red = f"{r.get('perimeter_reduction_pct', 0):.2f}%"
        v_err = f"{r.get('volume_error_pct', 0):.4f}%"
        print(f"{r['label']:<12} {r['n_completed']:>8} {r['evolution_time']:>10.1f} "
              f"{render:>10} {p_red:>12} {v_err:>10}")
    print(f"\nTotal wall time: {total_time:.1f}s")


if __name__ == '__main__':
    main()
