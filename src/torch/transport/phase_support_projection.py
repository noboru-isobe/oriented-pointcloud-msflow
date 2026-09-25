"""Phase-support projection: label-free removal of interior fossil
marks, read from the reconstructed phase itself (review decision,
2026-08-07 -- option B).

Context. Through a contact window the oriented point cloud legitimately
carries hidden-boundary marks (opposite sheets cancelling in the
current). After the merger, those marks freeze (velocity and
redistribution scale with q ~ 0) while the true boundary retreats;
the stranded pair leaves a PERMANENT kernel-scale dipole defect inside
the merged phase (measured on the Otto pair run: rho_raw_max growing
1.02 -> 1.14, projection correction 3e-3 -> 1.3e-2, recurring
single-cell fragments). That is no longer a hidden boundary -- it is a
stale degree of freedom of the old interface representation.

This stage removes such marks as a REPRESENTATION PROJECTION: the same
point cloud semantics (phase and smoothed current preserved up to
discretization tolerance), fewer marks. It is NOT the old q-based
dead-point rule -- q is never consulted for the decision (q ~ 0 on
fossils is an effect, not the criterion); the decision reads the
reconstructed set E_rho alone.

Candidate rule (all conditions on the CURRENT rho, no labels, no
history): a mark i is a fossil candidate iff

    min(rho(x_i + l n_i), rho(x_i - l n_i)) > 1 - delta_in     (interior
    |rho(x_i + l n_i) - rho(x_i - l n_i)| < delta_jump          on BOTH
                                                                sides)
    dist(x_i, transition band of rho) > c_depth * r_local

with l = c_ell * fill_epsilon and r_local = max(fill_epsilon, sigma,
delta). A true boundary mark has phase on one side and vacuum on the
other; an annulus inner boundary has the hole on one side; pre-contact
facing arcs have vacuum in the gap -- all retained. Only marks
strictly in the bulk on both sides of their own normal qualify.

Eligibility (v10-corrected): compatibility count 1 AND the operating
and 3*thr levels reading one component -- current-state conditions.
The thr/3 level carries NO veto: its flicker IS the fossil's own
sub-resolution noise, and requiring full (1,1,1) stability creates a
chicken-and-egg deadlock (measured in v10: the projection never fired
and the run died at the same fragmentation point as v9). The contact
window, where hidden sheets are load-bearing, remains protected by
the compat==2 deflation and by the two-sided candidate test itself.

Protocol: ONE-SHOT per step (candidates from the pre-cleanup rho, no
cascade), then SHADOW VERIFICATION before commit: the candidate-free
cloud must reproduce the phase (D_rho), the smoothed current (D_J),
the coherent perimeter (D_P) and the phase volume within fixed
tolerances, and remain single-component. On failure the projection is
REJECTED and the caller fails closed -- never fall back to tolerating
the fossil via gate exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .phase_grid import CurrentToPhase, PhaseGridConfig
from .weighted_poisson import connected_components

_T = torch.Tensor


@dataclass(frozen=True)
class PhaseSupportProjectionConfig:
    c_ell: float = 2.5            # FAR probe: l = c_ell * eps, beyond the self-trench
    delta_in: float = 0.1         # both sides above 1 - delta_in
    delta_jump: float = 0.1       # side difference below this
    # local-repair / far-field-invariance verification (v13.1: the
    # original global tolerances penalized the repair itself -- the
    # phase and current CHANGE at the fossil site because the artifact
    # disappears; what must not change is everything else)
    repair_radius_eps: float = 3.0   # near zone = this * eps around removals
    # ---- PSP-SPEC-1.0 (frozen after the exploratory series; the
    # halo verification confirmed delta-rho is the grid-converged
    # harmonic field of the removed residual dipole: zero mean 7e-19,
    # harmonicity contrast 234x, ~1/r decay, ||grad drho||_2 = 0.67
    # ||dJ||_2, L2 identical at 512^2 and 768^2) ----
    tau_rho_l2: float = 2e-2         # ||drho_raw||_2 / ||rho_raw||_2
                                     # (measured 9.4e-3 on the live
                                     # fossil removal)
    tau_gradrho_over_dj: float = 1.0  # elliptic bound ||grad drho||_2
                                      # <= this * ||dJ||_2 (theory =1,
                                      # measured 0.67)
    tau_quality: float = 1e-3        # raw overshoot/undershoot must
                                     # not grow beyond this
    tau_rho_far_linf: float = 1e-1   # TELEMETRY-ONLY former far gate.
                                     # v19 measurement: the fossil's dipole
                                     # potential has a power-law halo (rho
                                     # is the harmonic potential of the
                                     # compact current), so its HEALING is
                                     # also nonlocal -- far cells at 0.96
                                     # recover toward 1.0 by ~1.8e-2. True
                                     # locality is enforced on the CURRENT
                                     # (tau_j_far; removal is linear in J,
                                     # measured 7e-9); the rho bound only
                                     # forbids gross changes: deleting real
                                     # boundary flips its enclosed region
                                     # by O(1) far away and is caught here
    tau_j_far: float = 1e-3          # far-field smoothed-current invariance
    tau_net: float = 0.15            # |sum m n| / sum m of removed set
    tau_proj_slack: float = 1e-3     # QUALITY NON-DEGRADATION: the shadow
                                     # water-filling correction must not
                                     # exceed the current one (+slack).
                                     # Removing a fossil HEALS the
                                     # reconstruction; removing genuine
                                     # boundary blows the correction up --
                                     # the direct repair-vs-damage test
                                     # (the volume check is vacuous: the
                                     # filling enforces the target
                                     # exactly)
    tau_p: float = 1e-3              # coherent-perimeter tolerance
    tau_vol: float = 1e-4            # phase-volume tolerance (kept as a
                                     # reconstruction diagnostic; the
                                     # REAL conservation check is:)
    tau_vdiv: float = 1e-3           # current-divergence volume drift
                                     # with shadow-recomputed masses


@dataclass
class ProjectionOutcome:
    n_candidates: int
    accepted: bool
    keep_mask: Optional[_T] = None
    D_rho: float = 0.0
    D_J: float = 0.0
    D_P: float = 0.0
    D_V: float = 0.0
    vol_drift: float = 0.0
    n_before: int = 0
    n_after: int = 0
    removed_indices: tuple = ()
    reason: str = ""
    # diagnostics ONLY (review: q is recorded on removed marks but
    # never consulted by the rule)
    removed_q: tuple = field(default_factory=tuple)
    removed_centroid: tuple = field(default_factory=tuple)


def _bilinear_periodic(rho: _T, pts: _T, box_min, box_max) -> _T:
    """Bilinear sample of a cell-centered periodic field at points."""
    H, W = rho.shape
    Lx = box_max[0] - box_min[0]
    dx = Lx / H
    # cell-center coordinates: box_min + (i + 0.5) dx
    u = (pts[:, 0] - box_min[0]) / dx - 0.5
    v = (pts[:, 1] - box_min[1]) / dx - 0.5
    i0 = torch.floor(u).long()
    j0 = torch.floor(v).long()
    fu = (u - i0.to(u.dtype)).clamp(0.0, 1.0)
    fv = (v - j0.to(v.dtype)).clamp(0.0, 1.0)
    i0m, j0m = i0 % H, j0 % W
    i1m, j1m = (i0 + 1) % H, (j0 + 1) % W
    return ((1 - fu) * (1 - fv) * rho[i0m, j0m]
            + fu * (1 - fv) * rho[i1m, j0m]
            + (1 - fu) * fv * rho[i0m, j1m]
            + fu * fv * rho[i1m, j1m])


def _band_distance_at(rho: _T, pts: _T, box_min, box_max,
                      delta_band: float) -> _T:
    """Distance from each point to the diffuse transition band."""
    import numpy as np
    from scipy import ndimage

    H, W = rho.shape
    dx = (box_max[0] - box_min[0]) / H
    band = ((rho >= delta_band) & (rho <= 1.0 - delta_band)).cpu().numpy()
    if not band.any():
        return torch.full((pts.shape[0],), float("inf"),
                          dtype=pts.dtype)
    dist_cells = ndimage.distance_transform_edt(~band)
    dist = torch.from_numpy(np.ascontiguousarray(dist_cells)).to(
        pts.dtype) * dx
    i = ((pts[:, 0] - box_min[0]) / dx - 0.5).round().long() % H
    j = ((pts[:, 1] - box_min[1]) / dx - 0.5).round().long() % W
    return dist[i, j]


def _smoothed_current(varifold, masses: _T, cfg: PhaseGridConfig) -> _T:
    from .phase_grid import ScatterStencil
    st = ScatterStencil(varifold.positions, cfg)
    n = varifold.normals
    return torch.stack([st.apply(masses * n[:, 0]),
                        st.apply(masses * n[:, 1])])


def phase_support_projection(
    varifold, masses: _T, coherence: _T,
    phase_cfg: PhaseGridConfig, target_volume: float,
    r_local: float,
    psp: PhaseSupportProjectionConfig = PhaseSupportProjectionConfig(),
    perimeter=None,
    mass_fn=None,
) -> ProjectionOutcome:
    """One-shot candidate detection + shadow verification.

    perimeter: optional callable(varifold, masses) -> float for the
    D_P check (production coherent-perimeter conventions supplied by
    the caller). If None, D_P is reported as 0 and NOT verified.

    mass_fn (review blocker 1): callable(varifold) -> masses using the
    EXACT next-step mass resolver. Deleting points changes N and every
    survivor's KDE density, so masses[keep] is NOT the post-deletion
    carrier; both shadow states are re-massed through this resolver.
    Falls back to the provided masses (unit-test convenience) when
    None.
    """
    pg = CurrentToPhase(phase_cfg).reconstruct(
        varifold, masses, target_volume)
    rho = pg.rho_metric
    # Eligibility re-check on the fresh reconstruct (current state).
    # v10 chicken-and-egg finding: requiring the FULL (1,1,1) sweep is
    # self-defeating -- the fossil's own sub-resolution crumbs keep
    # the thr/3 level flickering, so the projection that would remove
    # the fossil never fires. The operative guards are the operating
    # and 3*thr levels reading ONE component (the contact window
    # itself is separately protected by the compat==2 deflation and by
    # the two-sided candidate test); thr/3 fuzz -- which IS the fossil
    # noise -- carries no veto.
    # v11 correction: read eligibility on rho_METRIC (the cleaned,
    # projected set). The RAW field still contains the fossil's own
    # above-3thr bumps as islands, so a raw-field count keeps the
    # projection ineligible forever -- the same deadlock as v10 one
    # level up.
    _, n_op = connected_components(rho > phase_cfg.support_threshold)
    _, n_hi = connected_components(rho > 3.0 * phase_cfg.support_threshold)
    if not (n_op == n_hi == 1):
        return ProjectionOutcome(0, False,
                                 reason=f"phase not single-component at "
                                 f"the operating/3thr levels "
                                 f"(n_op={n_op}, n_hi={n_hi})")

    pos, nrm = varifold.positions, varifold.normals
    # v13 candidate rule (measured correction): the seam fossil keeps
    # its own diffuse TRENCH alive (mark <-> field feedback: the
    # residual dipole holds local rho in the band range), so a
    # near-probe interior test and a band-distance test are both
    # structurally unsatisfiable at the fossil (measured: interior
    # candidates 0 for 19 straight steps, depth == 0 after). The rule
    # is now PAIR-BASED -- the direct discretization of a hidden
    # boundary: (i) an antiparallel partner within r_pair = eps
    # (n_i . n_j < -0.8), (ii) FAR probes at +- c_ell*eps along the
    # mark's own normal (beyond the self-trench) reading bulk on both
    # sides. True boundaries have no antiparallel partner within eps
    # (at the notch the sheets diverge); the pre-contact facing arcs
    # are blocked upstream by compat == 2. Sub-resolution features
    # (cracks thinner than the probe span) can be erased; the shadow
    # verification caps that at the discretization tolerance.
    ell = psp.c_ell * phase_cfg.fill_epsilon
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    antiparallel = (nrm @ nrm.T) < -0.8
    has_partner = ((d < phase_cfg.fill_epsilon) & antiparallel).any(dim=1)
    rp = _bilinear_periodic(rho, pos + ell * nrm,
                            phase_cfg.box_min, phase_cfg.box_max)
    rm = _bilinear_periodic(rho, pos - ell * nrm,
                            phase_cfg.box_min, phase_cfg.box_max)
    interior = has_partner \
        & (torch.minimum(rp, rm) > 1.0 - psp.delta_in) \
        & ((rp - rm).abs() < psp.delta_jump)
    n_cand = int(interior.sum())
    if n_cand == 0:
        return ProjectionOutcome(0, False, reason="no candidates")

    keep = ~interior
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    v_minus = OrientedPointCloudVarifold(positions=pos[keep],
                                         angles=varifold.angles[keep])
    m_plus = mass_fn(varifold) if mass_fn is not None else masses
    m_minus = (mass_fn(v_minus) if mass_fn is not None
               else masses[keep])

    # shadow verification, v13.1 semantics (local repair, far-field
    # invariance): the removal is EXPECTED to change phase and current
    # at the fossil site -- that change IS the repair (trench heals,
    # dipole field vanishes). What is verified: (a) nothing changes
    # beyond the repair zone, (b) the removed set is self-cancelling
    # (balanced pair, near-zero net current), (c) volume, component
    # count and perimeter are preserved. Near-zone changes are
    # RECORDED, not capped (vol/C/P already bound them globally).
    pg_minus = CurrentToPhase(phase_cfg).reconstruct(
        v_minus, m_minus, target_volume)
    H, W = rho.shape
    dx = (phase_cfg.box_max[0] - phase_cfg.box_min[0]) / H
    xs = torch.linspace(phase_cfg.box_min[0], phase_cfg.box_max[0],
                        H + 1, dtype=rho.dtype)[:-1] + dx / 2
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    cells = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    # repair zone = neighborhood of the WHOLE hidden-pair structure
    # (v17 measurement: removing a subset of the seam shifts the
    # REMAINING fossil's trench -- coupled through the reconstruction
    # -- which is near-seam but far from the removed subset; 3.2e-2
    # far-Linf from exactly that). Marks with antiparallel partners
    # delimit the diseased zone; true boundaries have no partners, so
    # the far zone stays the clean territory. State-based as ever.
    zone_marks = pos[has_partner | interior]
    d2 = torch.cdist(cells, zone_marks).min(dim=1).values.reshape(H, W)
    near = d2 <= psp.repair_radius_eps * phase_cfg.fill_epsilon
    far = ~near
    drho_field = (pg_minus.rho_metric - rho).abs()
    # far-field invariance MODULO sub-cell interface displacement
    # (v18 measurement: the water-filling's global micro-adjustment
    # moves distant interfaces by ~dx/6; at slope ~1/eps that reads as
    # |drho| ~ 3e-2 pointwise while the bulk changes by ~1e-8 -- a raw
    # Linf misreads benign interface micro-motion as field change).
    # Allowed = half a cell of local interface displacement + tau.
    gx = (rho.roll(-1, 0) - rho.roll(1, 0)).abs() / 2.0
    gy = (rho.roll(-1, 1) - rho.roll(1, 1)).abs() / 2.0
    grad_cell = torch.maximum(gx, gy)        # |drho| per one-cell shift
    excess = (drho_field - 0.5 * grad_cell).clamp_min(0.0)
    d_rho_far = float(excess[far].max()) if bool(far.any()) else 0.0
    # PSP-SPEC-1.0 elliptic quantities on the RAW fields (pre-clip:
    # the linear reconstruction layer where the halo mathematics
    # lives)
    draw = pg_minus.rho_raw - pg.rho_raw
    d_rho_l2 = float(draw.norm() / pg.rho_raw.norm().clamp_min(1e-300))
    gxr = (draw.roll(-1, 0) - draw.roll(1, 0)) / (2 * dx)
    gyr = (draw.roll(-1, 1) - draw.roll(1, 1)) / (2 * dx)
    grad_l2 = float(((gxr ** 2 + gyr ** 2).sum() * dx * dx).sqrt())
    d_rho = float(drho_field.sum()
                  / rho.abs().sum().clamp_min(1e-300))
    J_plus = _smoothed_current(varifold, m_plus, phase_cfg)
    J_minus = _smoothed_current(v_minus, m_minus, phase_cfg)
    dJ_field = (J_minus - J_plus).pow(2).sum(dim=0).sqrt()
    Jn = J_plus.norm().clamp_min(1e-300)
    d_j_far = float(dJ_field[far].pow(2).sum().sqrt() / Jn)
    d_j = float(dJ_field.pow(2).sum().sqrt() / Jn)
    dj_l2_abs = float((((J_minus - J_plus) ** 2).sum()
                       * dx * dx).sqrt())
    # balance: the removed set must be (near) self-cancelling
    mn = (m_plus[interior, None] * nrm[interior])
    net_frac = float(mn.sum(dim=0).norm()
                     / m_plus[interior].sum().clamp_min(1e-300))
    # review blocker 3: the phase-grid volumes are BOTH projected to
    # the same target (tautological); the meaningful conservation
    # check is the current-divergence volume with independently
    # recomputed masses on each side
    vdiv_plus = float(0.5 * (m_plus
                             * (pos * nrm).sum(-1)).sum())
    n_minus_vec = v_minus.normals
    vdiv_minus = float(0.5 * (m_minus
                              * (v_minus.positions
                                 * n_minus_vec).sum(-1)).sum())
    d_vdiv = abs(vdiv_minus - vdiv_plus) / max(abs(vdiv_plus), 1e-300)
    vol_plus = float(pg.rho_metric.sum()) * pg.cell_area
    vol_minus = float(pg_minus.rho_metric.sum()) * pg_minus.cell_area
    vol_drift = abs(vol_minus - vol_plus) / max(abs(vol_plus), 1e-300)
    # D_P semantics (v13.2 measurement): an imperfectly-cancelling
    # fossil carries its own SPURIOUS coherent-perimeter contribution
    # (measured 1.1% on the offset-pair fixture) -- removing it
    # corrects P downward by exactly that amount. The check therefore
    # bounds the UNEXPLAINED COLLATERAL: |dP| minus the removed marks'
    # own direct contribution sum(m_i q_i). Bad deletions are caught
    # by net/D_V/proj-improvement/topo, not by P preservation.
    d_p = 0.0
    if perimeter is not None:
        p_plus = float(perimeter(varifold, m_plus))
        p_minus = float(perimeter(v_minus, m_minus))
        p_removed_direct = (
            float((m_plus[interior] * coherence[interior]).sum())
            if coherence is not None else 0.0)
        d_p = max(0.0, abs(p_minus - p_plus) - p_removed_direct) \
            / (1.0 + abs(p_plus))
    # review blocker 5: the shadow must not be topologically WORSE
    # than the pre state at any sweep level, must not create new
    # specks, and must stay single at op/3thr
    thr = phase_cfg.support_threshold
    _, n_lo_p = connected_components(rho > thr / 3.0)
    _, n_lo_m = connected_components(pg_minus.rho_metric > thr / 3.0)
    _, n_op_minus = connected_components(pg_minus.rho_metric > thr)
    _, n_hi_minus = connected_components(
        pg_minus.rho_metric > 3.0 * thr)
    shadow_topo_ok = (n_op_minus == 1 and n_hi_minus == 1
                      and n_lo_m <= n_lo_p
                      and pg_minus.n_speck_components_dropped
                      <= pg.n_speck_components_dropped)

    proj_improves = (pg_minus.projection_l2_relative
                     <= pg.projection_l2_relative + psp.tau_proj_slack)
    # PSP-SPEC-1.0 quality improvement: the removal must actually
    # reduce (or not worsen) the fossil defect it targets
    quality_ok = (
        float(pg_minus.rho_raw_max) <= float(pg.rho_raw_max)
        + psp.tau_quality
        and float(pg_minus.rho_raw_min) >= float(pg.rho_raw_min)
        - psp.tau_quality)
    ok = (shadow_topo_ok
          and d_rho_l2 <= psp.tau_rho_l2
          and grad_l2 <= psp.tau_gradrho_over_dj * dj_l2_abs + 1e-10
          and d_j_far <= psp.tau_j_far and net_frac <= psp.tau_net
          and proj_improves and quality_ok
          and d_vdiv <= psp.tau_vdiv
          and d_p <= psp.tau_p and vol_drift <= psp.tau_vol)
    q_removed = tuple(float(x) for x in coherence[interior].tolist()) \
        if coherence is not None else ()
    cx = float(pos[interior, 0].mean())
    cy = float(pos[interior, 1].mean())
    return ProjectionOutcome(
        n_cand, ok, keep_mask=keep, D_rho=d_rho, D_J=d_j, D_P=d_p,
        D_V=d_vdiv, vol_drift=vol_drift,
        n_before=int(pos.shape[0]), n_after=int(keep.sum()),
        removed_indices=tuple(int(i) for i
                              in interior.nonzero().flatten().tolist()),
        reason="accepted" if ok else (
            f"shadow verification failed: C={n_op_minus} "
            f"rhoL2={d_rho_l2:.2e} gradL2={grad_l2:.2e}"
            f"/dJ={dj_l2_abs:.2e} "
            f"qual=({float(pg_minus.rho_raw_max):.3f}"
            f"<={float(pg.rho_raw_max):.3f},"
            f"{float(pg_minus.rho_raw_min):.3f}"
            f">={float(pg.rho_raw_min):.3f}) "
            f"Linf_far={d_rho_far:.2e} D_J_far={d_j_far:.2e} "
            f"net={net_frac:.2e} D_V={d_vdiv:.2e} "
            f"topo=({n_lo_m}<={n_lo_p},{n_op_minus},{n_hi_minus}) "
            f"proj {pg.projection_l2_relative:.2e}"
            f"->{pg_minus.projection_l2_relative:.2e} "
            f"D_P={d_p:.2e} (near repair: D_rho={d_rho:.2e} "
            f"D_J={d_j:.2e}, n_cand={n_cand})"),
        removed_q=q_removed, removed_centroid=(cx, cy))
