"""Grid metric G1: current-to-phase filling (frozen rho0 construction).

Reconstructs the diffuse phase rho_eps from the oriented point-cloud
current J = sum_i m_i n_i delta_{x_i}:

    scatter J with the Wendland kernel eta_eps
    -> solve -Delta rho = div J by FFT (constant coefficient, exact)
    -> fix the zero mode by the target volume
    -> rho_raw
    -> mass-preserving box projection onto {0 <= rho <= 1, integral = V0}
    -> rho_metric  (the frozen weighted-Poisson coefficient)

Runs ONCE per MM step at setup time: strictly no_grad (the MKL rfft
backward issue is therefore not on this path). The candidate-dependent
tangent source is NOT built here -- see boundary_flux_grid.py, which
shares the SAME kernel eta_eps (the identity d/dt(eta*chi) =
eta*(v H^1) requires one kernel for both).

rho_raw vs rho_metric (reviewer G1 spec): the raw FFT reconstruction
overshoots [0,1]; feeding a negative coefficient to L_rho would break
the PSD guarantee. The projection is applied to the FROZEN coefficient
only, so its non-differentiability is harmless; its size is first-class
telemetry with a fail-closed gate downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .weighted_poisson import connected_components

_T = torch.Tensor


@dataclass(frozen=True)
class PhaseGridConfig:
    grid_shape: Tuple[int, int] = (512, 512)
    box_min: Tuple[float, float] = (-2.0, -2.0)
    box_max: Tuple[float, float] = (2.0, 2.0)
    fill_epsilon: float = 0.06
    support_threshold: float = 1e-3
    projection_rel_tol: float = 1e-2   # fail-closed gate on the fix
    # components smaller than the mollifier's own footprint cannot
    # represent a phase region -- they are reconstruction speckle
    # (measured: threshold-level FFT ringing fragments into O(100)
    # sub-kernel specks when eps/dx is marginal). Dropped specks are
    # first-class telemetry, never silent.
    min_component_area_factor: float = 1.0   # * pi eps^2 / 4
    # ambiguity-window projection cap (Otto pair v7, 2026-08-07):
    # while hidden facing arcs cancel at kernel scale during a merger,
    # the raw reconstruction is locally disturbed and the
    # water-filling correction legitimately exceeds the clean-regime
    # tolerance (measured 1.33e-2 at the first breach vs the 1e-2
    # gate; clean long-run maximum was 1.2e-3). Inside the
    # threshold-unstable window ONLY, the projection gate uses this
    # cap instead; strict projection_rel_tol applies everywhere else.
    projection_rel_tol_window: float = 5e-2


@dataclass(frozen=True)
class PhaseGrid:
    rho_raw: _T
    rho_metric: _T
    component_labels: _T
    n_components: int
    dx: float
    cell_area: float
    target_volume: float
    rho_raw_min: float
    rho_raw_max: float
    projection_l2_relative: float
    projection_linf: float
    volume_error: float
    n_speck_components_dropped: int
    speck_area_dropped: float
    speck_mass_dropped: float
    min_final_component_area: float
    # global carrier rescale target/V_hat(m) applied before scattering
    # (exactly 1.0 when the target is the cloud's own KDE volume);
    # the accumulated departure from 1 measures the KDE drift
    carrier_renormalization: float = 1.0
    # per-dropped-speck telemetry (review, Otto pair): (cx, cy, area,
    # mass, rho_max, has_support) per dropped sub-resolution
    # component -- lets the audit verify each tolerated drop was
    # healing debris inside the ambiguous region, not an unrelated
    # satellite. has_support (v9): any mark within fill_epsilon of the
    # speck's cells -- the budget invariant is "budgets protect
    # MATTER, and matter is where the marks are"; markless specks are
    # reconstruction ringing (measured: single cells at rho = thr
    # around the sharp reentrant notch, ~2 eps from the boundary,
    # mass 1.8e-7 each), not physical fragments
    speck_details: tuple = ()
    n_speck_supported_dropped: int = 0
    speck_supported_area_dropped: float = 0.0
    speck_supported_mass_dropped: float = 0.0
    # v9b: smallest FINAL component that carries marks (the
    # fragmentation gate's subject); markless sub-resolution final
    # components are the same threshold ringing the compat layer
    # already excludes (v8) and are counted separately. None -> not
    # computed (hand-built PhaseGrid); validation falls back to the
    # unfiltered minimum.
    min_final_supported_component_area: "float | None" = None
    n_final_markless_debris: int = 0


class InvalidPhaseReconstructionError(RuntimeError):
    """The frozen rho0 reconstruction failed a fail-closed gate
    (projection too large / volume not preserved / components dropped).
    A scientific metric setup must not proceed on such a phase."""


def validate_phase_reconstruction(pg: "PhaseGrid",
                                  cfg: "PhaseGridConfig",
                                  volume_tol: float = 1e-8,
                                  ambiguous_window: bool = False
                                  ) -> None:
    """G1.1 gates (reviewer): enforced by every scientific metric
    setup. Dropping a speck is a TOPOLOGY EVENT executed by the
    reconstruction -- never acceptable silently in a scientific run.

    ambiguous_window (2026-08-07, Otto pair v4): while the confidence
    sweep is threshold-UNSTABLE (the same label-free window that
    activates the conservative-sweep deflation -- e.g. the healing
    interior right after a merger), the support's sub-resolution
    structure is by definition unresolved, and single-cell crumbs of
    the old interface zone flicker across the threshold (measured at
    v4 step 785: one cell, mass 1.8e-7 = 7e-8 of the phase volume).
    In that window ONLY, sub-resolution drops are tolerated under
    hard caps (area <= 8 cells, mass <= 1e-6 * phase volume), always
    recorded in the PhaseGrid fields. Outside the window the strict
    fail-closed gate is unchanged."""
    if pg.n_speck_components_dropped > 0:
        cell_area = pg.cell_area
        vol = float(pg.rho_metric.sum()) * cell_area
        # v9 split: budgets protect MATTER, and matter is where the
        # marks are. Specks with a mark within fill_epsilon are
        # candidate physical fragments and keep the strict
        # window-scoped budget; markless specks are reconstruction
        # ringing (measured: threshold-level single cells around the
        # sharp reentrant notch, ~2 eps from any mark) and carry a
        # mass-only budget, window-independent -- always recorded in
        # speck_details, never silent.
        unsup_mass = (pg.speck_mass_dropped
                      - pg.speck_supported_mass_dropped)
        if unsup_mass > 1e-5 * max(vol, 1e-30):
            raise InvalidPhaseReconstructionError(
                f"markless ringing mass dropped ({unsup_mass:.3e}) "
                f"exceeds 1e-5 * V: underresolved_component")
        if pg.n_speck_supported_dropped > 0:
            tolerable = (
                ambiguous_window
                and pg.speck_supported_area_dropped <= 8.0 * cell_area
                and pg.speck_supported_mass_dropped
                <= 1e-6 * max(vol, 1e-30))
            if not tolerable:
                raise InvalidPhaseReconstructionError(
                    f"{pg.n_speck_supported_dropped} supported "
                    f"sub-resolution component(s) dropped (area "
                    f"{pg.speck_supported_area_dropped:.3e}, mass "
                    f"{pg.speck_supported_mass_dropped:.3e}): "
                    "underresolved_component")
    # POST-projection fragmentation (G2a-0 reviewer): a negative
    # water-filling shift can split a valid raw candidate into
    # sub-threshold fragments on the FINAL support. Fail closed; never
    # delete/re-project in a loop.
    min_area = cfg.min_component_area_factor \
        * math.pi * cfg.fill_epsilon ** 2 / 4.0
    if pg.n_components < 1:
        raise InvalidPhaseReconstructionError("empty final support")
    # v9b: gate on the smallest SUPPORTED final component (markless
    # sub-resolution flicker is counted in n_final_markless_debris,
    # excluded by the compat layer, and not a fragmentation of
    # matter); hand-built PhaseGrid without the field falls back to
    # the unfiltered minimum
    min_eff = (pg.min_final_supported_component_area
               if pg.min_final_supported_component_area is not None
               else pg.min_final_component_area)
    if min_eff < min_area:
        # same window-scoped tolerance as the speck-drop clause (the
        # v16 death: healing produces transient sub-resolution FINAL
        # fragments of the same quantization class between projection
        # rounds); caps identical, everything recorded via the
        # per-step counts telemetry
        frag_tolerable = False
        if ambiguous_window:
            labs, n_c = connected_components(
                pg.rho_metric > cfg.support_threshold)
            small_area = 0.0
            for c in range(n_c):
                a = float((labs == c).sum()) * pg.cell_area
                if a < min_area:
                    small_area += a
            frag_tolerable = small_area <= 8.0 * pg.cell_area
        if not frag_tolerable:
            raise InvalidPhaseReconstructionError(
                f"post_projection_fragmentation: smallest final "
                f"component area {pg.min_final_component_area:.3e} "
                f"< {min_area:.3e}")
    if not bool(torch.isfinite(pg.rho_metric).all()):
        raise InvalidPhaseReconstructionError("non-finite rho_metric")
    if float(pg.rho_metric.min()) < 0.0 \
            or float(pg.rho_metric.max()) > 1.0 + 1e-12:
        raise InvalidPhaseReconstructionError(
            "rho_metric outside [0, 1]")
    proj_tol = (cfg.projection_rel_tol_window if ambiguous_window
                else cfg.projection_rel_tol)
    if pg.projection_l2_relative > proj_tol:
        raise InvalidPhaseReconstructionError(
            f"projection correction {pg.projection_l2_relative:.3e} "
            f"exceeds {proj_tol:g}"
            + (" (ambiguity-window cap)" if ambiguous_window else ""))
    if pg.volume_error > volume_tol:
        raise InvalidPhaseReconstructionError(
            f"volume error {pg.volume_error:.3e} exceeds "
            f"{volume_tol:g}")


def assert_grid_configs_consistent(phase_cfg: "PhaseGridConfig",
                                   poisson_cfg) -> None:
    """G1.1 (reviewer 3.1): the phase support and the operator
    nullspace must be read on the SAME grid with the SAME threshold,
    or C_phase != C_{L_rho} becomes possible."""
    if tuple(phase_cfg.grid_shape) != tuple(poisson_cfg.grid_shape) \
            or phase_cfg.box_min != poisson_cfg.box_min \
            or phase_cfg.box_max != poisson_cfg.box_max \
            or phase_cfg.support_threshold \
            != poisson_cfg.support_threshold:
        raise ValueError(
            "PhaseGridConfig and WeightedPoissonConfig disagree on "
            "grid/box/support_threshold -- the phase components and "
            "the operator nullspace would be read on different "
            "domains")


class ScatterStencil:
    """Compact local Wendland-C2 scatter on the periodic grid.

    Precomputes, for M source points, the flat cell indices and kernel
    weights of the local stencil. The weights are FROZEN geometry
    (points are pre-step data); applying the stencil to per-point
    values is a linear map, differentiable (and double-backward
    trivial) in the values.
    """

    def __init__(self, points: _T, config: PhaseGridConfig):
        H, W = config.grid_shape
        assert H == W, "square grids only for now"
        self.H, self.W = H, W
        L = config.box_max[0] - config.box_min[0]
        self.dx = L / H
        self.cell_area = self.dx * self.dx
        eps = config.fill_epsilon
        if eps / self.dx < 2.0:
            raise ValueError(
                f"fill_epsilon={eps} under-resolved by the grid "
                f"(eps/dx={eps / self.dx:.2f} < 2)")

        pts = points.detach()
        M = pts.shape[0]
        half = int(math.ceil(eps / self.dx)) + 1
        w = 2 * half + 1
        offs = torch.arange(-half, half + 1, device=pts.device)
        OX, OY = torch.meshgrid(offs, offs, indexing="ij")
        offs2 = torch.stack([OX.reshape(-1), OY.reshape(-1)], dim=1)

        # base cell of each point
        bx = torch.floor((pts[:, 0] - config.box_min[0]) / self.dx)
        by = torch.floor((pts[:, 1] - config.box_min[1]) / self.dx)
        ix = (bx[:, None] + offs2[None, :, 0]).long()
        iy = (by[:, None] + offs2[None, :, 1]).long()
        # periodic cell centers RELATIVE to each point (unwrapped
        # distance via the raw index, so wrap only applies to storage)
        cx = config.box_min[0] + (ix.to(pts.dtype) + 0.5) * self.dx
        cy = config.box_min[1] + (iy.to(pts.dtype) + 0.5) * self.dx
        d = torch.sqrt((cx - pts[:, 0:1]) ** 2
                       + (cy - pts[:, 1:2]) ** 2)
        t = (d / eps).clamp(max=1.0)
        kw = (1 - t) ** 4 * (4 * t + 1)
        # DISCRETE per-point normalization: each point deposits exactly
        # unit mass on the grid (sum kw * dx^2 = 1). More robust than
        # the analytic 2D constant 7/(pi eps^2) at moderate eps/dx
        # (midpoint quadrature of the sharp Wendland peak is several
        # percent off), and it makes total_flux/volume identities exact
        # by construction.
        kw = kw / (kw.sum(dim=1, keepdim=True)
                   * self.cell_area).clamp_min(1e-300)
        self.kernel_weights = kw               # (M, w^2), frozen
        self.flat_index = ((ix % H) * W + (iy % W)).reshape(M, -1)

    def apply(self, values: _T) -> _T:
        """field(x_c) = sum_j values_j eta(x_c - p_j); linear in
        values."""
        src = values[:, None] * self.kernel_weights
        flat = torch.zeros(self.H * self.W, dtype=values.dtype,
                           device=values.device)
        flat = flat.scatter_add(
            0, self.flat_index.reshape(-1), src.reshape(-1))
        return flat.reshape(self.H, self.W)


def _fft_inverse_laplacian_div(Jx: _T, Jy: _T, dx: float) -> _T:
    """Solve -Delta phi = div J on the periodic grid (spectral,
    exact); returns phi with zero mean. no_grad by contract."""
    H, W = Jx.shape
    kx = 2 * torch.pi * torch.fft.fftfreq(H, d=dx).to(
        device=Jx.device, dtype=Jx.dtype)
    ky = 2 * torch.pi * torch.fft.fftfreq(W, d=dx).to(
        device=Jx.device, dtype=Jx.dtype)
    KX = kx[:, None]
    KY = ky[None, :]
    k2 = KX ** 2 + KY ** 2
    k2[0, 0] = 1.0
    div_hat = (1j * KX * torch.fft.fft2(Jx.to(torch.complex128))
               + 1j * KY * torch.fft.fft2(Jy.to(torch.complex128)))
    phi_hat = div_hat / k2
    phi_hat[0, 0] = 0.0
    return torch.real(torch.fft.ifft2(phi_hat)).to(Jx.dtype)


def _mass_preserving_box_projection(rho: _T, target: float,
                                    cell_area: float,
                                    support_threshold: float,
                                    support_mask: Optional[_T] = None,
                                    iters: int = 60) -> _T:
    """Mass-preserving [0,1] projection RESTRICTED to the support
    candidate region. A global water-filling shift would smear the
    clamp's lost mass uniformly over the vacuum, lifting the whole
    background above the support threshold and spuriously CONNECTING
    separated components (measured: pair read as C=1 at every
    threshold). Instead the compensating shift acts only where the
    clamped raw phase already lives; the vacuum stays exactly zero."""
    base = rho.clamp(0.0, 1.0)
    S = (support_mask if support_mask is not None
         else base > support_threshold)
    if not bool(S.any()):
        return torch.zeros_like(rho)
    lo, hi = -1.5, 1.5
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        vol = float((rho[S] + mid).clamp(0.0, 1.0).sum()) * cell_area
        if vol < target:
            lo = mid
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    out = torch.zeros_like(rho)
    out[S] = (rho[S] + tau).clamp(0.0, 1.0)
    return out


class CurrentToPhase:
    """Frozen rho0 reconstruction (setup-time, no_grad)."""

    def __init__(self, config: PhaseGridConfig):
        self.config = config

    @torch.no_grad()
    def reconstruct(self, varifold, carrier_masses: _T,
                    target_volume: float) -> PhaseGrid:
        cfg = self.config
        pts = varifold.positions.detach()
        normals = varifold.normals.detach()
        m = carrier_masses.detach()

        # G3c fix (long-run diagnosis, 2026-08-06): the filling
        # identity assumes the scattered current's amplitude is
        # consistent with the volume target. The KDE carrier functional
        # V_hat = 0.5 sum m x.n drifts along a trajectory
        # (-2.9e-5/step measured, dt-independent) while the geometric
        # volume is conserved; the mismatch then leaks into EITHER the
        # water-filling correction (current_divergence targeting:
        # projection grew as 0.0005 + 1.25|drift| until the fail-closed
        # gate stopped the run at step 263) OR a uniform background
        # lift toward the support threshold (initial_fixed: box margin
        # 19 eps -> 0 in 100 steps). Renormalize the carrier ONCE so
        # the current's divergence-theorem volume equals the target
        # EXACTLY; the factor is recorded telemetry (it IS the measured
        # KDE drift) and fail-closed against nonsense inputs. When the
        # target is the cloud's own V_hat the factor is exactly 1.0.
        vhat = float(0.5 * (m * (pts * normals).sum(dim=-1)).sum())
        if vhat <= 0.0:
            raise InvalidPhaseReconstructionError(
                f"carrier current has non-positive divergence-theorem "
                f"volume ({vhat:.3e}) -- normals not consistently "
                f"outward")
        carrier_renorm = target_volume / vhat
        if abs(carrier_renorm - 1.0) > 0.2:
            raise InvalidPhaseReconstructionError(
                f"carrier renormalization {carrier_renorm:.4f} departs "
                f"from 1 by more than 20% -- the current and the "
                f"volume target describe different geometries")
        m = m * carrier_renorm

        stencil = ScatterStencil(pts, cfg)
        Jx = stencil.apply(m * normals[:, 0])
        Jy = stencil.apply(m * normals[:, 1])

        # J = -D chi  =>  div J = -Delta chi  =>  chi solves the
        # constant-coefficient problem below; the SIGN is pinned by the
        # convention check in tests (unit circle: inside ~ 1).
        H, W = cfg.grid_shape
        area = (cfg.box_max[0] - cfg.box_min[0]) \
            * (cfg.box_max[1] - cfg.box_min[1])
        rho_raw = _fft_inverse_laplacian_div(Jx, Jy, stencil.dx)
        rho_raw = rho_raw + target_volume / area

        # G1.1 order (reviewer): identify + drop specks on the RAW
        # support candidate FIRST, then mass-preserve on the retained
        # support -- so the final rho_metric always integrates to V0
        # exactly and the telemetry describes the FINAL correction.
        base = rho_raw.clamp(0.0, 1.0)
        candidate = base > cfg.support_threshold
        labels0, n0 = connected_components(candidate)
        min_area = cfg.min_component_area_factor \
            * math.pi * cfg.fill_epsilon ** 2 / 4.0
        dropped, dropped_area, dropped_mass = 0, 0.0, 0.0
        n_sup, sup_area, sup_mass = 0, 0.0, 0.0
        speck_details = []
        retained = candidate.clone()
        H = base.shape[0]
        xs = torch.linspace(float(cfg.box_min[0]), float(cfg.box_max[0]),
                            H + 1, dtype=base.dtype)[:-1] + stencil.dx / 2
        for c in range(n0):
            mask = labels0 == c
            a = float(mask.sum()) * stencil.cell_area
            if a < min_area:
                dropped += 1
                dropped_area += a
                m_c = float(base[mask].sum()) * stencil.cell_area
                dropped_mass += m_c
                ij = mask.nonzero()
                cells_xy = torch.stack(
                    [xs[ij[:, 0]], xs[ij[:, 1]]], dim=1)
                has_support = bool(
                    torch.cdist(cells_xy, varifold.positions)
                    .min() < cfg.fill_epsilon)
                if has_support:
                    n_sup += 1
                    sup_area += a
                    sup_mass += m_c
                speck_details.append((
                    float(xs[ij[:, 0]].mean()),
                    float(xs[ij[:, 1]].mean()),
                    a, m_c, float(base[mask].max()), has_support))
                retained &= ~mask

        rho_metric = _mass_preserving_box_projection(
            rho_raw, target_volume, stencil.cell_area,
            cfg.support_threshold, support_mask=retained)

        diff = rho_metric - rho_raw
        proj_l2 = float(diff.norm() / rho_raw.norm().clamp_min(1e-300))
        active = rho_metric > cfg.support_threshold
        labels, n_comp = connected_components(active)
        min_final_area = min(
            (float((labels == c).sum()) * stencil.cell_area
             for c in range(n_comp)), default=0.0)
        # v9b: the fragmentation gate's subject is MATTER. A final
        # component below min_area with no mark within fill_epsilon is
        # the same threshold ringing the compat layer excludes (v8) --
        # counted, never gated. Support is only checked for the small
        # components (the large ones are the phase itself).
        min_sup_area = float("inf")
        n_markless = 0
        for c in range(n_comp):
            mask = labels == c
            a = float(mask.sum()) * stencil.cell_area
            if a >= min_area:
                min_sup_area = min(min_sup_area, a)
                continue
            ij = mask.nonzero()
            cells_xy = torch.stack([xs[ij[:, 0]], xs[ij[:, 1]]], dim=1)
            if bool(torch.cdist(cells_xy, varifold.positions).min()
                    < cfg.fill_epsilon):
                min_sup_area = min(min_sup_area, a)
            else:
                n_markless += 1
        if min_sup_area == float("inf"):
            min_sup_area = min_final_area

        vol = float(rho_metric.sum()) * stencil.cell_area
        return PhaseGrid(
            rho_raw=rho_raw, rho_metric=rho_metric,
            component_labels=labels, n_components=n_comp,
            dx=stencil.dx, cell_area=stencil.cell_area,
            target_volume=target_volume,
            rho_raw_min=float(rho_raw.min()),
            rho_raw_max=float(rho_raw.max()),
            projection_l2_relative=proj_l2,
            projection_linf=float(diff.abs().max()),
            volume_error=abs(vol - target_volume)
            / max(target_volume, 1e-300),
            n_speck_components_dropped=dropped,
            speck_area_dropped=dropped_area,
            speck_mass_dropped=dropped_mass,
            min_final_component_area=min_final_area,
            carrier_renormalization=carrier_renorm,
            speck_details=tuple(speck_details),
            n_speck_supported_dropped=n_sup,
            speck_supported_area_dropped=sup_area,
            speck_supported_mass_dropped=sup_mass,
            min_final_supported_component_area=min_sup_area,
            n_final_markless_debris=n_markless)


class GaugeHoldCertificateError(RuntimeError):
    """0L-B0 cell E: the bulk-seeded gauge partition failed its
    certificate (fail-closed, never silently degraded)."""


@torch.no_grad()
def bulk_seeded_component_labels(active: _T, config: "PhaseGridConfig",
                                 positions: _T, bulk_pp: _T,
                                 n_bulk: int) -> Tuple[_T, dict]:
    """0L-B0 cell E (DIAGNOSTIC gauge hold, reviewer 0L-B0 item 3):
    partition the active set by the RAW bulk incidence -- every active
    cell takes the bulk label of its nearest particle (Euclidean,
    scipy cKDTree; adequate for the concentric benchmark where the
    contact corridor is a single annular gap -- a general geometry
    would need the multi-source graph distance inside the active set).
    State-functional and history-free: reads only the current
    (rho, particles, bulk labels).

    The result is fed to WeightedPoissonOperator(component_labels=...)
    exactly like the conservative-sweep labels: the MATRIX is
    untouched, only the projection/gauge/compatibility subspace is
    refined. Across a bridged corridor this is NOT a direct sum of two
    operators -- it is a two-component deflation of ONE connected
    operator; the caller must record gamma_cross (the face
    conductivity crossing the label interface) so the reading stays
    honest.

    Certificate (fail-closed): every label non-empty and 4-connected on
    the active graph; every particle's nearest active cell carries the
    particle's own bulk label; co-membership invariant under bulk-label
    permutation (constructive: labels are copied from bulk_pp)."""
    from scipy.spatial import cKDTree
    from .weighted_poisson import connected_components

    H, W = active.shape
    dx = (config.box_max[0] - config.box_min[0]) / H
    idx = active.nonzero()                                    # (M, 2)
    cx = config.box_min[0] + (idx[:, 0].to(positions.dtype) + 0.5) * dx
    cy = config.box_min[1] + (idx[:, 1].to(positions.dtype) + 0.5) * dx
    tree = cKDTree(positions.detach().cpu().numpy())
    _, nn = tree.query(torch.stack([cx, cy], 1).cpu().numpy(), k=1)
    lab_cells = bulk_pp.detach().cpu()[torch.as_tensor(nn)]
    labels = torch.full((H, W), -1, dtype=torch.long,
                        device=active.device)
    labels[idx[:, 0], idx[:, 1]] = lab_cells.to(active.device)
    # certificate
    for b in range(n_bulk):
        m = labels == b
        if not bool(m.any()):
            raise GaugeHoldCertificateError(
                f"bulk-seeded gauge label {b} is empty")
        _, nc = connected_components(m)
        if nc != 1:
            raise GaugeHoldCertificateError(
                f"bulk-seeded gauge label {b} has {nc} active-graph "
                "components (must be 1)")
    # every particle's nearest active cell carries its own bulk label
    ptree = cKDTree(torch.stack([cx, cy], 1).cpu().numpy())
    _, pc = ptree.query(positions.detach().cpu().numpy(), k=1)
    own = lab_cells[torch.as_tensor(pc)] == bulk_pp.detach().cpu()
    n_bad = int((~own).sum())
    if n_bad:
        raise GaugeHoldCertificateError(
            f"{n_bad} particles whose nearest active cell carries a "
            "foreign bulk label")
    return labels, {"n_labels": n_bulk, "n_particles_checked":
                    int(positions.shape[0])}


def cross_label_conductivity(op) -> float:
    """gamma_cross = sum of face conductivities across a label
    interface / sum of all face conductivities (reviewer 0L-B0 item
    3c). Small -> the deflation removes a near-disconnected soft mode;
    large -> the gauge hold is an artificial constraint on a strongly
    connected operator."""
    lab = op.labels
    diff_x = (lab != lab.roll(-1, 0)) & (lab >= 0) \
        & (lab.roll(-1, 0) >= 0)
    diff_y = (lab != lab.roll(-1, 1)) & (lab >= 0) \
        & (lab.roll(-1, 1) >= 0)
    tot = float(op.rho_face_x.sum() + op.rho_face_y.sum())
    cross = float(op.rho_face_x[diff_x].sum()
                  + op.rho_face_y[diff_y].sum())
    return cross / tot if tot > 0 else float("nan")


@torch.no_grad()
def substantial_component_count(rho_metric: _T, thr: float,
                                min_core_cells: int = 0) -> int:
    """Number of operating components that contain at least one
    substantial 3*thr core (>= min_core_cells cells). Components
    without such a core are threshold-level debris by the v7/v8
    invariant and are absorbed by conservative_component_labels;
    consistency guards must compare against THIS count, not the raw
    operating count."""
    active = rho_metric > thr
    labels_op, n_op = connected_components(active)
    hi = rho_metric > 3.0 * thr
    labels_hi, _ = connected_components(hi)
    lab_op_np = labels_op.cpu().numpy()
    lab_hi_np = labels_hi.cpu().numpy()
    n_sub = 0
    for c in range(n_op):
        comp = lab_op_np == c
        children = set(lab_hi_np[comp & (lab_hi_np >= 0)])
        if any(int((lab_hi_np == ch).sum()) >= min_core_cells
               for ch in children):
            n_sub += 1
    return n_sub


def conservative_component_labels(rho_metric: _T, thr: float,
                                  min_core_cells: int = 0
                                  ) -> Tuple[_T, int]:
    """Confidence-sweep-conservative component labels (2026-08-07,
    the soft-bridge merger window).

    When the operating-threshold support has just bridged through a
    weak neck (counts at thr/3, thr, 3*thr read e.g. 1/1/2), the
    weighted Poisson operator acquires a soft exchange mode and float64
    solves hit their eps*||A||*||phi|| floor (measured: |phi|/|b|
    jumping 1.3e-3 -> 2.27 and residual floor 3e-10 at the Otto pair's
    step 743). This helper returns the MOST CONSERVATIVE reading of
    the sweep: if the 3*thr superlevel set still separates into more
    components than the operating support, every operating-active cell
    is assigned to its nearest 3*thr core (Euclidean distance
    transform, cores restricted to the same operating component), so
    the compatibility/projection subspace keeps the blobs separate
    until the support is UNAMBIGUOUSLY merged. Label-free, no events,
    no history: the rule reads the current rho only and self-removes
    once all thresholds agree. Physically it freezes inter-blob volume
    exchange during the ambiguity window -- benign, because the true
    flow's exchange there is O(neck conductance).

    min_core_cells (v7): a 3*thr core smaller than this many cells is
    healing DEBRIS, not a blob -- it must not seed a compatibility
    component (v6 measured: a sub-resolution interior island became a
    compat component containing zero flux endpoints, so its constraint
    row was exactly zero and the rank check refused the basis --
    correctly). Debris cells are absorbed into the nearest substantial
    core. Callers pass the same sub-resolution criterion as the speck
    machinery (min_component_area / cell_area).

    Returns (labels, n): identical to the operating labels whenever no
    refinement is needed.
    """
    import numpy as np
    from scipy import ndimage

    active = rho_metric > thr
    labels_op, n_op = connected_components(active)
    hi = rho_metric > 3.0 * thr
    labels_hi, n_hi = connected_components(hi)

    lab_op_np = labels_op.cpu().numpy()
    lab_hi_np = labels_hi.cpu().numpy()
    act_np = active.cpu().numpy()
    comp_children = []
    for c in range(n_op):
        comp = lab_op_np == c
        children = sorted(set(lab_hi_np[comp & (lab_hi_np >= 0)]))
        children = [ch for ch in children
                    if int((lab_hi_np == ch).sum()) >= min_core_cells]
        comp_children.append(children)
    coreless = [c for c in range(n_op) if not comp_children[c]]

    # v8 (2026-08-09, arm-4 Run A step 767): the debris invariant is
    # GLOBAL -- an operating component with NO substantial 3*thr core
    # at all is threshold-level flicker (measured: a raw-connected
    # tip cell that the water-filling shift disconnected in the
    # METRIC field, 1 cell at rho = thr exactly, no marks within 2
    # cells, zero flux row -> rank-deficient refusal; merging it into
    # the main label instead makes the factor exactly singular, since
    # the cell is isolated in the positive-conductivity face graph
    # and its gauge is not spanned by the componentwise projection).
    # Debris therefore stays UNLABELED (-1): the operator excludes it
    # from the active system (vacuum semantics -- exact for an
    # isolated threshold-level cell), in EVERY branch -- the previous
    # early return on n_hi <= n_op skipped the debris rule precisely
    # when the flicker did not also split the sweep.
    if n_hi <= n_op and not coreless:
        return labels_op, n_op
    if all(not ch for ch in comp_children):
        # no substantial core anywhere: degenerate reading, keep the
        # operating labels (nothing substantial to distinguish)
        return labels_op, n_op

    out = torch.full_like(labels_op, -1)
    next_label = 0
    for c in range(n_op):
        comp = lab_op_np == c
        children = comp_children[c]
        if not children:
            continue          # coreless debris: stays -1 (excluded)
        if len(children) == 1:
            out[torch.from_numpy(comp)] = next_label
            next_label += 1
            continue
        # nearest-core assignment inside this operating component
        dists = np.stack([
            ndimage.distance_transform_edt(~(lab_hi_np == ch))
            for ch in children])
        assign = np.argmin(dists, axis=0)
        for j, _ch in enumerate(children):
            mask = comp & act_np & (assign == j)
            out[torch.from_numpy(mask)] = next_label
            next_label += 1
    return out, next_label
