"""Boundary-integral (BIE) linearized Wasserstein metric, assembled
block-diagonally per bulk component.

Replaces the grid Poisson metric (grid_wasserstein.py) by a dense
single-layer boundary integral formulation with the SAME endpoint
conventions (BoundaryFluxToGrid built without a grid):

    W(s, dtheta) = (h/2) sum_c V_c^T Q_c V_c,
    V_ik = (s_i - r_ik dtheta_i) / h   (carrier measure, unit flux scale)

where for every metric component c the quadratic form Q_c is the
symmetrized interior-Neumann solution operator of the single-layer
representation on the endpoints of c:

    A_c = 1/2 I + K*_c  (interior trace, columns scaled by zeta),
    A_w = D^{1/2} A_c D^{-1/2},  D = diag(zeta),
    bordered system [[A_w, u0], [v0^T, 0]] with the ONE near-null pair
    (u0, v0) of A_w (one phase component -> one constant potential),
    Q_c = sym(D S_c Lambda_c),  Lambda_c = D^{-1/2} (bordered)^{-1} D^{1/2}.

Component structure (no rank estimation):
  * metric components are the winding bulk components, MERGED across
    two bulks as soon as two of their loops come within the bridging
    gap g_bridge = c_bridge * ell (ell = median particle mass);
  * inside a merged component the particles of the cancelling pair
    (facing partner within g_mask with anti-parallel normal) are
    EXCLUDED from the operator and held fixed (frozen displacement)
    until the removal / reconnection event;
  * the compatibility rows are the exact endpoint flux sums per
    metric component (BoundaryFluxToGrid._particle_label_rows), i.e.
    the bulk area rows before a merge and their sum after it.

Fail-closed diagnostics: the relative second-smallest singular value
sigma_tail = sigma_{n-1} / sigma_1 of every block (rank-1 evidence),
and the smallest eigenvalue of the metric Hessian on the admissible
subspace (check_definiteness).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .bem_wasserstein import (
    IncompatibleVelocityError,
    _trace_signs,
    build_bem_matrices_point,
)
from .boundary_flux_grid import BoundaryFluxPolicies, BoundaryFluxToGrid

_T = torch.Tensor


class BIEBlockDegenerateError(RuntimeError):
    """A metric block has no unmasked particle, or its weighted
    operator has more than one near-null direction (rank evidence
    below sigma_tail_min)."""


class IndefiniteBIEMetricError(RuntimeError):
    """The metric Hessian restricted to the admissible subspace has a
    negative eigenvalue beyond tolerance."""


@dataclass
class BIEMetricConfig:
    # kernel regularization r^2 = |x - y|^2 + eps^2, eps = scale * ell.
    # Measured on the flower (concave petals, N=291): scale 0.1 makes the
    # symmetrized form slightly indefinite on the admissible subspace
    # (lambda_min/lambda_max down to -5e-4 between steps 10 and 35),
    # scale >= 0.15 keeps it positive (margin ~1e-3) at the price of the
    # mode-k attenuation (disk k=3: -0.5% at 0.1, -1.2% at 0.15, -1.7% at
    # 0.2; the 1024^2 grid gives -0.8%).
    epsilon_scale: float = 0.15
    # two bulks merge into one metric component when two of their
    # loops come within c_bridge * ell of each other
    bridge_gap_over_ell: float = 1.0
    # inside a merged component, particles with a facing partner of
    # the other bulk within c_mask * ell and normal product <= anti_tol
    # form the cancelling pair (masked + frozen)
    mask_gap_over_ell: float = 1.5
    anti_parallel_tol: float = -0.9
    mask_cancelling_pair: bool = True
    # fail-closed levels
    sigma_tail_min: float = 1e-6
    definiteness_check_every: int = 1
    definiteness_tol: float = 1e-6
    compat_reject_tol: float = 0.1


@dataclass
class BIEBlock:
    component: int
    particle_index: _T          # (n_p,) particle ids in the block
    endpoint_index: _T          # (n_e,) flat endpoint ids (i*K + k)
    sqrt_w: _T                  # (n_e,)
    u0: _T                      # (n_e,) left near-null vector of A_w
    quad: _T                    # (n_e, n_e) symmetric quadratic form
    sigma_tail: float           # sigma_{n-1} / sigma_1
    null_level: float           # sigma_n / sigma_1
    wall_seconds: float = 0.0


def loop_gap_matrix(positions: _T, loop_labels: _T, n_loop: int) -> _T:
    """(n_loop, n_loop) minimal point distance between loops (inf on
    the diagonal). Dense cdist -- N is a few hundred here."""
    G = torch.full((n_loop, n_loop), float("inf"), dtype=positions.dtype,
                   device=positions.device)
    idx = [torch.nonzero(loop_labels == a).flatten()
           for a in range(n_loop)]
    for a in range(n_loop):
        if idx[a].numel() == 0:
            continue
        for b in range(a + 1, n_loop):
            if idx[b].numel() == 0:
                continue
            d = torch.cdist(positions[idx[a]], positions[idx[b]]).min()
            G[a, b] = d
            G[b, a] = d
    return G


def merge_metric_labels(gaps: _T, bulk_of_loop: _T, n_bulk: int,
                        g_bridge: float):
    """Union of bulk components across loop pairs closer than
    g_bridge. Returns (metric_of_loop (n_loop,), merged_pairs) where
    merged_pairs lists (bulk_a, bulk_b, gap) for every bridging loop
    pair. Metric labels are dense, in first-appearance loop order (the
    winding convention), so an unmerged partition reproduces the bulk
    labels exactly."""
    n_loop = int(bulk_of_loop.numel())
    parent = list(range(n_bulk))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged_pairs = []
    for a in range(n_loop):
        for b in range(a + 1, n_loop):
            ba, bb = int(bulk_of_loop[a]), int(bulk_of_loop[b])
            if ba == bb:
                continue
            g = float(gaps[a, b])
            if g <= g_bridge:
                merged_pairs.append((ba, bb, g))
                ra, rb = find(ba), find(bb)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    root_of_bulk = [find(b) for b in range(n_bulk)]
    remap = {}
    metric_of_loop = torch.zeros(n_loop, dtype=torch.long,
                                 device=bulk_of_loop.device)
    for a in range(n_loop):
        r = root_of_bulk[int(bulk_of_loop[a])]
        if r not in remap:
            remap[r] = len(remap)
        metric_of_loop[a] = remap[r]
    return metric_of_loop, merged_pairs


def cancelling_pair_mask(positions: _T, normals: _T, loop_labels: _T,
                         bulk_of_loop: _T, metric_of_loop: _T,
                         g_mask: float, anti_tol: float) -> _T:
    """Particles of a merged metric component that face a particle of
    the OTHER bulk within g_mask with anti-parallel normals (n_i . n_j
    <= anti_tol): the discrete hidden boundary (cancelling pair)."""
    N = positions.shape[0]
    mask = torch.zeros(N, dtype=torch.bool, device=positions.device)
    n_loop = int(bulk_of_loop.numel())
    idx = [torch.nonzero(loop_labels == a).flatten()
           for a in range(n_loop)]
    for a in range(n_loop):
        for b in range(a + 1, n_loop):
            if int(bulk_of_loop[a]) == int(bulk_of_loop[b]):
                continue
            if int(metric_of_loop[a]) != int(metric_of_loop[b]):
                continue
            if idx[a].numel() == 0 or idx[b].numel() == 0:
                continue
            D = torch.cdist(positions[idx[a]], positions[idx[b]])
            anti = normals[idx[a]] @ normals[idx[b]].T <= anti_tol
            hit = (D <= g_mask) & anti
            mask[idx[a]] |= hit.any(dim=1)
            mask[idx[b]] |= hit.any(dim=0)
    return mask


def assemble_block(component: int, particle_index: _T,
                   endpoint_index: _T, p: _T, n: _T, zeta: _T,
                   epsilon: float) -> BIEBlock:
    """Single-layer interior-Neumann block on one metric component:
    bordered solve with the ONE near-null pair, quadratic form
    precomputed by a single multi-RHS LU solve."""
    t0 = time.perf_counter()
    n_e = int(zeta.numel())
    if n_e < 3:
        raise BIEBlockDegenerateError(
            f"metric component {component} has {n_e} endpoints")
    S, K_star = build_bem_matrices_point(p, n, zeta, epsilon)
    half, _ = _trace_signs("interior")
    I = torch.eye(n_e, dtype=zeta.dtype, device=zeta.device)
    A = half * I + K_star
    sqrt_w = zeta.sqrt()
    A_w = (sqrt_w[:, None] * A) / sqrt_w[None, :]
    U, Sv, Vh = torch.linalg.svd(A_w)
    s0 = float(Sv[0])
    sigma_tail = float(Sv[-2] / s0)
    null_level = float(Sv[-1] / s0)
    u0 = U[:, -1].detach()
    v0 = Vh[-1, :].detach()
    Msys = torch.zeros(n_e + 1, n_e + 1, dtype=zeta.dtype,
                       device=zeta.device)
    Msys[:n_e, :n_e] = A_w
    Msys[:n_e, n_e] = u0
    Msys[n_e, :n_e] = v0
    LU = torch.linalg.lu_factor(Msys)
    RHS = torch.zeros(n_e + 1, n_e, dtype=zeta.dtype, device=zeta.device)
    RHS[:n_e, :] = torch.diag(sqrt_w)
    SOL = torch.linalg.lu_solve(*LU, RHS)
    LAM = SOL[:n_e] / sqrt_w[:, None]
    PHI = S @ LAM
    A_q = zeta[:, None] * PHI
    quad = 0.5 * (A_q + A_q.T)
    return BIEBlock(component=component, particle_index=particle_index,
                    endpoint_index=endpoint_index, sqrt_w=sqrt_w,
                    u0=u0, quad=quad.detach(), sigma_tail=sigma_tail,
                    null_level=null_level,
                    wall_seconds=time.perf_counter() - t0)


class BIEWassersteinMetric:
    """Frozen-coefficient block-diagonal BIE metric; the same call
    signature as GridWassersteinMetric / BEMWasserstein and the same
    grid-side attributes the MM stepper reads (phase / poisson are
    None: there is no grid)."""

    def __init__(self, config: Optional[BIEMetricConfig] = None):
        self.config = config or BIEMetricConfig()
        self.phase = None
        self.poisson = None
        self.flux: Optional[BoundaryFluxToGrid] = None
        self.forward_solve = "fail_closed"
        self.gauge_hold_fired = False
        self.gamma_cross = None
        self.max_projection_rel = 0.0
        self.compat_labels: Optional[_T] = None
        self.compat_labels_base: Optional[_T] = None
        self.n_compat_components: Optional[int] = None
        self.n_compat_base: Optional[int] = None
        self.metric_of_loop: Optional[_T] = None
        self.bulk_of_loop: Optional[_T] = None
        self.merged_pairs: list = []
        self.masked: Optional[_T] = None
        self.blocks: list = []
        self.last_sigma_tail: list = []
        self.last_null_level: list = []
        self.last_lambda_min: Optional[float] = None
        self.last_lambda_max: Optional[float] = None
        self.last_definiteness_step: Optional[int] = None
        self.n_forward_solves = 0
        self.solve_wall_seconds = 0.0
        self.setup_wall_seconds = 0.0
        self.epsilon_used: Optional[float] = None
        self.ell: Optional[float] = None
        self.g_bridge: Optional[float] = None
        self.g_mask: Optional[float] = None
        self._last_r_comp: Optional[float] = None
        self._max_r_comp = 0.0
        self._n_setups = 0
        self._N = None
        self._K = None

    # ---- setup ---------------------------------------------------
    def setup_for_step(self, varifold, carrier_masses: _T, coherence: _T,
                       target_volume: float,
                       policies: Optional[BoundaryFluxPolicies] = None,
                       bulk_partition: Optional[dict] = None,
                       gauge_seed=None) -> None:
        t0 = time.perf_counter()
        cfg = self.config
        pol = policies or BoundaryFluxPolicies()
        pos = varifold.positions.detach()
        nor = varifold.normals.detach()
        m = carrier_masses.detach()
        q = coherence.detach()
        N = pos.shape[0]
        if bulk_partition is None or bulk_partition.get(
                "bulk_of_loop") is None:
            raise ValueError(
                "BIEWassersteinMetric.setup_for_step needs the winding "
                "bulk partition (dict with bulk_of_loop, loops, n_bulk)")
        loops = bulk_partition["loops"]
        bulk_of_loop = bulk_partition["bulk_of_loop"]
        n_bulk = int(bulk_partition["n_bulk"])
        n_loop = int(bulk_of_loop.numel())

        self.flux = BoundaryFluxToGrid(pos, nor, m, q, None, pol)
        K = self.flux.n_endpoints
        self._N, self._K = N, K
        ell = float(m.median())
        self.ell = ell
        self.epsilon_used = cfg.epsilon_scale * ell
        self.g_bridge = cfg.bridge_gap_over_ell * ell
        self.g_mask = cfg.mask_gap_over_ell * ell

        gaps = loop_gap_matrix(pos, loops, n_loop)
        metric_of_loop, merged = merge_metric_labels(
            gaps, bulk_of_loop, n_bulk, self.g_bridge)
        self.metric_of_loop = metric_of_loop
        self.bulk_of_loop = bulk_of_loop
        self.merged_pairs = merged
        self.loop_gaps = gaps
        labels = metric_of_loop[loops]
        self.compat_labels = labels
        self.compat_labels_base = labels
        self.n_compat_components = int(labels.max()) + 1
        self.n_compat_base = self.n_compat_components

        if cfg.mask_cancelling_pair and merged:
            masked = cancelling_pair_mask(
                pos, nor, loops, bulk_of_loop, metric_of_loop,
                self.g_mask, cfg.anti_parallel_tol)
        else:
            masked = torch.zeros(N, dtype=torch.bool, device=pos.device)
        self.masked = masked

        blocks = []
        sig_tail, null_lv = [], []
        ep_pos = self.flux.endpoint_positions            # (N, K, 2)
        zeta = self.flux.zeta                            # (N, K)
        for c in range(self.n_compat_components):
            sel = (labels == c) & ~masked
            pidx = torch.nonzero(sel).flatten()
            if pidx.numel() == 0:
                raise BIEBlockDegenerateError(
                    f"metric component {c} has no unmasked particle")
            eidx = (pidx[:, None] * K
                    + torch.arange(K, device=pidx.device)[None, :]
                    ).reshape(-1)
            p = ep_pos[pidx].reshape(-1, 2)
            n = nor[pidx].repeat_interleave(K, dim=0)
            z = zeta[pidx].reshape(-1)
            blk = assemble_block(c, pidx, eidx, p, n, z, self.epsilon_used)
            if blk.sigma_tail < cfg.sigma_tail_min:
                raise BIEBlockDegenerateError(
                    f"metric component {c}: second singular value "
                    f"{blk.sigma_tail:.3e} < {cfg.sigma_tail_min:.1e} "
                    f"(rank-1 evidence lost; {int(masked.sum())} masked)")
            blocks.append(blk)
            sig_tail.append(blk.sigma_tail)
            null_lv.append(blk.null_level)
        self.blocks = blocks
        self.last_sigma_tail = sig_tail
        self.last_null_level = null_lv
        self.n_forward_solves = 0
        self.solve_wall_seconds = 0.0
        self._last_r_comp = None
        self._max_r_comp = 0.0
        self._n_setups += 1
        self.forward_solve = "fail_closed"
        self.setup_wall_seconds = time.perf_counter() - t0

    # ---- evaluation ----------------------------------------------
    def _endpoint_velocity(self, displacements: _T, delta_angles: _T,
                           time_step: float) -> _T:
        fl = self.flux
        v = fl.flux_scale[:, None] * (
            displacements[:, None]
            - fl.r_physical * delta_angles[:, None])         # (N, K)
        return v.reshape(-1) / time_step

    def __call__(self, displacements: _T, delta_angles: _T,
                 time_step: float,
                 drho_offset: "_T | None" = None) -> _T:
        if drho_offset is not None:
            raise ValueError(
                "BIE metric has no grid field: drho_offset unsupported")
        assert self.blocks, "setup_for_step first"
        if self.forward_solve != "fail_closed":
            raise ValueError(
                f"forward_solve must be 'fail_closed' for the BIE "
                f"metric, got {self.forward_solve!r}")
        t0 = time.perf_counter()
        V = self._endpoint_velocity(displacements, delta_angles, time_step)
        h = time_step
        W = V.new_zeros(())
        worst = 0.0
        for blk in self.blocks:
            Vb = V[blk.endpoint_index]
            with torch.no_grad():
                vt = blk.sqrt_w * Vb.detach()
                nv = float(vt.norm())
                r = float((blk.u0 @ vt).abs() / nv) if nv > 0 else 0.0
            worst = max(worst, r)
            W = W + (h / 2) * (Vb @ (blk.quad @ Vb))
        self._last_r_comp = worst
        self._max_r_comp = max(self._max_r_comp, worst)
        if worst > self.config.compat_reject_tol:
            raise IncompatibleVelocityError(
                f"componentwise Neumann compatibility violated: "
                f"max_c |u0^T v~|/|v~| = {worst:.3f} > "
                f"{self.config.compat_reject_tol}")
        self.n_forward_solves += 1
        self.solve_wall_seconds += time.perf_counter() - t0
        return W

    def quadratic(self, displacements, delta_angles, time_step,
                  drho_offset=None):
        return self(displacements, delta_angles, time_step,
                    drho_offset=drho_offset)

    def hessp(self, dir_s: _T, dir_th: _T, time_step: float
              ) -> tuple[_T, _T]:
        """Metric HVP in (s, dtheta): (h) B^T Q B / h^2 = B^T Q B / h."""
        assert self.blocks
        N, K = self._N, self._K
        V = self._endpoint_velocity(dir_s, dir_th, time_step)
        g = torch.zeros_like(V)
        for blk in self.blocks:
            g[blk.endpoint_index] = blk.quad @ V[blk.endpoint_index]
        g = g.reshape(N, K)
        fl = self.flux
        base = fl.flux_scale[:, None] * g
        g_s = base.sum(dim=1)
        g_th = -(base * fl.r_physical).sum(dim=1)
        return g_s, g_th

    def hessian_dense(self, time_step: float) -> _T:
        """Dense (2N, 2N) metric Hessian in (s, dtheta) (audit)."""
        N, K = self._N, self._K
        fl = self.flux
        # V = (1/h) [diag(fs) (x) 1_K | -diag(fs) r] (s; th)
        B = torch.zeros(N * K, 2 * N, dtype=fl.zeta.dtype,
                        device=fl.zeta.device)
        rows = torch.arange(N * K, device=B.device)
        part = rows // K
        B[rows, part] = fl.flux_scale[part]
        B[rows, N + part] = -(fl.flux_scale[part]
                              * fl.r_physical.reshape(-1))
        Q = torch.zeros(N * K, N * K, dtype=B.dtype, device=B.device)
        for blk in self.blocks:
            ii = blk.endpoint_index
            Q[ii[:, None], ii[None, :]] = blk.quad
        return (B.T @ Q @ B) / time_step

    def component_rows(self) -> _T:
        assert self.flux is not None and self.compat_labels is not None
        return self.flux.component_rows(self.compat_labels, None)

    def check_definiteness(self, basis: _T, AB_solve: _T, coherence: _T,
                           q_suppress: bool, time_step: float,
                           step_index: Optional[int] = None
                           ) -> tuple[float, float]:
        """Smallest / largest eigenvalue of the metric Hessian on the
        admissible subspace (pre-scale variable u = basis y; s = D_q u,
        dtheta = D_q AB D_q u when q_suppress). Raises
        IndefiniteBIEMetricError below -tol * lambda_max."""
        N, K = self._N, self._K
        fl = self.flux
        q = coherence.detach()
        Sm = q[:, None] * basis if q_suppress else basis        # (N, p)
        TH = AB_solve @ Sm
        if q_suppress:
            TH = q[:, None] * TH
        Vm = (fl.flux_scale[:, None, None]
              * (Sm[:, None, :] - fl.r_physical[:, :, None]
                 * TH[:, None, :])).reshape(N * K, -1) / time_step
        p = Vm.shape[1]
        H = torch.zeros(p, p, dtype=Vm.dtype, device=Vm.device)
        for blk in self.blocks:
            Vb = Vm[blk.endpoint_index]
            H += time_step * (Vb.T @ (blk.quad @ Vb))
        H = 0.5 * (H + H.T)
        ev = torch.linalg.eigvalsh(H)
        lam_min, lam_max = float(ev[0]), float(ev[-1])
        self.last_lambda_min, self.last_lambda_max = lam_min, lam_max
        self.last_definiteness_step = step_index
        if lam_min < -self.config.definiteness_tol * max(lam_max, 1e-300):
            raise IndefiniteBIEMetricError(
                f"BIE metric indefinite on the admissible subspace: "
                f"lambda_min = {lam_min:.3e}, lambda_max = {lam_max:.3e}"
                f" ({len(self.blocks)} blocks, "
                f"{int(self.masked.sum()) if self.masked is not None else 0}"
                f" masked)")
        return lam_min, lam_max

    # ---- telemetry -----------------------------------------------
    def setup_snapshot_fields(self) -> "dict | None":
        if not self.blocks:
            return None
        return dict(
            n_components=self.n_compat_components,
            bie_sigma_tail=list(self.last_sigma_tail),
            bie_null_level=list(self.last_null_level),
            bie_lambda_min=self.last_lambda_min,
            bie_lambda_max=self.last_lambda_max,
            bie_n_masked=(int(self.masked.sum())
                          if self.masked is not None else 0),
            bie_merged_pairs=[(int(a), int(b), float(g))
                              for a, b, g in self.merged_pairs],
            bie_epsilon=self.epsilon_used,
            bie_g_bridge=self.g_bridge,
            bie_g_mask=self.g_mask,
            bie_ell=self.ell,
            bie_block_sizes=[int(b.endpoint_index.numel())
                             for b in self.blocks],
            bie_setup_wall_seconds=self.setup_wall_seconds,
        )
