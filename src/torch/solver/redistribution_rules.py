"""D2: redistribution time-level rules, one shared implementation.

Mirrors the dead_point_rules pattern: every consumer (the D2a operator
calibration, the D2b in-solver audit, the future active variants) calls
the same pure function, so the audit can never drift from what it
audits. The production `redistribute_points` is NOT touched; the
`legacy_hybrid` branch here reproduces it BITWISE given the same inputs
(pinned in tests), which is the operator-level wiring gate.

Rules (0D plan):

    legacy_hybrid       tangents from the input angles, FIXED for all
                        subiterations; density refreshed on the current
                        positions each subiteration; the caller-supplied
                        coherence (pre-MM in the solver) scales the
                        update, FIXED; curvature for the final angle
                        correction from the INPUT positions with raw
                        masses on the INPUT positions.
    post_mm_frozen      identical loop structure, but m and q are
                        recomputed ONCE on the input geometry and then
                        frozen. At operator level (no MM step) this
                        COINCIDES with legacy_hybrid whenever the caller
                        passes q(input) -- the legacy/frozen axis is
                        purely the STALENESS of q across the MM step and
                        only activates in the in-solver audit (D2b).
    substep_refreshed   tangents, coherence and curvature are refreshed
                        every subiteration from the CURRENT geometry;
                        the angle correction is applied incrementally
                        (closest to manuscript Algorithm 3).

The returned trace records, per subiteration, which geometry each
consumed quantity came from and the tangential/normal split of the
update measured against the CURRENT ordered polygon (central-difference
tangents) -- the update is exactly tangential w.r.t. the tangents it
USED, so any normal component w.r.t. the true evolving curve is
parametrization-induced geometric drift, the quantity D2 is after.
"""

from __future__ import annotations

import torch

from src.torch.math_utils.angles import wrap_angles
from src.torch.math_utils.curvature import compute_regularized_curvature
from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import compute_masses
from src.torch.transport import compute_coherence
from .redistribute import _compute_density_log_gradient

_T = torch.Tensor

REDISTRIBUTION_RULES = ("legacy_hybrid", "post_mm_frozen",
                        "substep_refreshed")


class RedistributionGeometryError(RuntimeError):
    """0L-R fail-closed: a PCA geometry certificate (one-
    dimensionality, effective sample size, orientation margin) or the
    loopwise-mode preconditions did not hold."""


def _geom_tangents_normals(pos: _T, loops: list | None = None):
    """Central-difference tangents/normals of the ordered polygon --
    the measuring frame for normal leakage (independent of the stored
    angles).

    LOOP-AWARE (review, D2b-1 correction): with `loops` given, the
    central differences are taken WITHIN each loop. A global roll on a
    concatenated multi-loop array gives four boundary-adjacent points
    cross-loop neighbours -- that bug contaminated the first
    'geometric tangent refresh only' control and its conclusion was
    retracted. Without `loops` the whole array is one loop."""
    if loops is None:
        t = pos.roll(-1, 0) - pos.roll(1, 0)
        t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-30)
        n = torch.stack([-t[:, 1], t[:, 0]], dim=1)
        return t, n
    t = torch.zeros_like(pos)
    n = torch.zeros_like(pos)
    for c in loops:
        p = pos[c]
        tl = p.roll(-1, 0) - p.roll(1, 0)
        tl = tl / tl.norm(dim=1, keepdim=True).clamp_min(1e-30)
        t[c] = tl
        n[c] = torch.stack([-tl[:, 1], tl[:, 0]], dim=1)
    return t, n


def _local_pca_tangents(pos: _T, ang: _T, delta_tan: float,
                        kernel: str,
                        loop_labels: _T | None = None,
                        return_certificate: bool = False):
    """Permutation-INVARIANT geometric tangent estimate (review,
    D2b-1 axis-B correction): weighted local covariance

        C_i = sum_j w_ij (x_j - xbar_i) (x_j - xbar_i)^T,

    principal eigenvector as the tangent, sign fixed by the STORED
    tangent (-sin a, cos a). No array-order information is consumed.

    Neighbourhood gate (0L-R production revision): with certified
    `loop_labels` the weight is w_ij = eta(.) * 1[l_i = l_j] -- the
    co-orientation gate 1[N_i . N_j > 0] is DROPPED there, because
    using the corrupted stored normals to pick the neighbourhood that
    repairs those normals is circular (reviewer 0L-R section 3). With
    loop_labels=None the historical co-orientation-gated behaviour is
    kept bitwise (comparison arm).

    With return_certificate=True also returns the per-point spectral
    certificate dict (lam1, lam2, rho = lam2/lam1, n_eff) -- the
    caller enforces the pre-registered thresholds fail-closed."""
    from src.torch.oriented_varifold.mass import KERNEL_FUNCTIONS
    eta = KERNEL_FUNCTIONS[kernel]
    d = torch.cdist(pos, pos)
    w = eta((d / delta_tan).clamp(max=1.0 + 1e-12))
    if loop_labels is not None:
        same = (loop_labels.unsqueeze(1) == loop_labels.unsqueeze(0))
        w = w * same.to(w.dtype)
    else:
        n_vec = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
        w = w * ((n_vec @ n_vec.T) > 0.0)
    wsum = w.sum(dim=1, keepdim=True).clamp_min(1e-30)
    xbar = (w @ pos) / wsum
    dx = pos.unsqueeze(0) - xbar.unsqueeze(1)          # (i, j, 2)
    C = torch.einsum("ij,ijk,ijl->ikl", w, dx, dx)
    evals, evecs = torch.linalg.eigh(C)
    t = evecs[:, :, -1]                                # top eigenvector
    t_stored = torch.stack([-torch.sin(ang), torch.cos(ang)], dim=1)
    sgn = torch.sign((t * t_stored).sum(dim=1, keepdim=True))
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    t = t * sgn
    if not return_certificate:
        return t
    lam1 = evals[:, -1]
    lam2 = evals[:, 0].clamp_min(0.0)
    cert = dict(
        lam1=lam1, lam2=lam2,
        rho=lam2 / lam1.clamp_min(1e-300),
        n_eff=(w.sum(dim=1) ** 2) / (w * w).sum(dim=1).clamp_min(
            1e-300),
    )
    return t, cert


# c_xi = int_R xi(|z|) dz for the BLM curvature denominator kernel,
# xi(u) = -u rho'(u)/n with n = 2. For wendland_c2 (rho'(u) =
# -20 u (1-u)^3): xi(u) = 10 u^2 (1-u)^3, c_xi = 2 * 10 * B(3,4)
# = 20 * 12/360 ... = 1/3 exactly (pinned numerically in tests).
KAPPA_DEN_C_XI = {"wendland_c2": 1.0 / 3.0}


def _enforce_kappa_denominator(b_raw: _T, loop_labels: _T,
                               epsilon: float, kernel: str,
                               beta_min: float | None,
                               gamma_min: float | None):
    """0L-kappa hardening (reviewer): DIMENSIONLESS two-stage
    denominator certificate. beta_i = eps * B_i / c_xi -> 1 on a
    well-sampled straight sheet (scale-free across eps/resolution);
    gamma_i = B_i / median_loop(B) catches a local gap inside one
    loop. None disables a check (audit contexts). The absolute 1e-8
    clamp inside compute_regularized_curvature is a numerical floor,
    never the certificate."""
    if beta_min is None and gamma_min is None:
        return
    c_xi = KAPPA_DEN_C_XI.get(kernel)
    if c_xi is None:
        raise RedistributionGeometryError(
            f"no c_xi constant registered for kernel {kernel!r}")
    beta = epsilon * b_raw / c_xi
    if beta_min is not None:
        worst = float(beta.min())
        if worst < beta_min:
            raise RedistributionGeometryError(
                f"kappa denominator certificate failed: min beta = "
                f"{worst:.4f} < {beta_min} (same-loop support too "
                f"sparse for a trustworthy curvature)")
    if gamma_min is not None:
        for lb in loop_labels.unique():
            sel = loop_labels == lb
            med = b_raw[sel].median().clamp_min(1e-300)
            worst = float((b_raw[sel] / med).min())
            if worst < gamma_min:
                raise RedistributionGeometryError(
                    f"kappa denominator certificate failed on loop "
                    f"{int(lb)}: min gamma = {worst:.4f} < "
                    f"{gamma_min} (local support gap)")


def _enforce_pca_certificate(cert: dict, rho_max: float | None,
                             neff_min: float | None):
    """Fail-closed spectral certificate (reviewer 0L-R section 4):
    one-dimensionality rho = lam2/lam1 <= rho_max and effective
    sample size n_eff >= neff_min. None disables a check (audit
    contexts); production passes the pre-registered thresholds."""
    if rho_max is not None:
        worst = float(cert["rho"].max())
        if worst > rho_max:
            raise RedistributionGeometryError(
                f"PCA one-dimensionality certificate failed: "
                f"max lam2/lam1 = {worst:.4f} > {rho_max}")
    if neff_min is not None:
        worst = float(cert["n_eff"].min())
        if worst < neff_min:
            raise RedistributionGeometryError(
                f"PCA effective-sample certificate failed: "
                f"min N_eff = {worst:.2f} < {neff_min}")


def pca_normal_reanchor(pos: _T, ang: _T, delta_tan: float,
                        kernel: str, loop_labels: _T | None,
                        rho_max: float | None,
                        neff_min: float | None,
                        ori_margin: float) -> tuple[_T, dict]:
    """0L-R "PCA normal re-anchoring" (approximate boundary-
    admissibility retraction -- NOT called a geometric-normal
    retraction: the PCA normal field is a pointwise local regression,
    not the exact normal field of one global curve; its quality is
    audited against the polygon oracle).

    n_new = sign(n_old . nu_PCA) nu_PCA with the pre-registered
    orientation margin |n_old . nu_PCA| >= ori_margin (= cos 30
    degrees by design: the retraction must be a SMALL correction;
    fail-closed otherwise). Returns (new_angles, stats)."""
    t, cert = _local_pca_tangents(pos, ang, delta_tan, kernel,
                                  loop_labels=loop_labels,
                                  return_certificate=True)
    _enforce_pca_certificate(cert, rho_max, neff_min)
    nu = torch.stack([t[:, 1], -t[:, 0]], dim=1)  # rot(-90): t -> n
    n_old = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
    dot = (n_old * nu).sum(dim=1)
    worst = float(dot.abs().min())
    if worst < ori_margin:
        raise RedistributionGeometryError(
            f"orientation margin failed: min |n_old . nu_PCA| = "
            f"{worst:.4f} < {ori_margin:.4f} -- the re-anchoring "
            f"would not be a small correction; fail-closed")
    sgn = torch.sign(dot)
    nu = nu * sgn.unsqueeze(-1)
    new_ang = torch.atan2(nu[:, 1], nu[:, 0])
    dang = torch.remainder(new_ang - ang + torch.pi,
                           2 * torch.pi) - torch.pi
    stats = dict(max_angle_rad=float(dang.abs().max()),
                 mean_angle_rad=float(dang.abs().mean()),
                 min_alignment=worst,
                 pca_rho_max=float(cert["rho"].max()),
                 pca_neff_min=float(cert["n_eff"].min()))
    return wrap_angles(new_ang), stats


def repair_boundary_normals(
    positions: _T,
    angles: _T,
    delta: float,
    tau: float,
    sigma: float,
    *,
    kernel: str = "wendland_c2",
    coherence_kernel: str = "wendland_c2",
    ell_factor: float = 4.0,
    pca_rho_max: float | None = 0.25,
    pca_neff_min: float | None = 2.5,
    pca_ori_margin: float = 0.8660254037844387,
) -> tuple[_T, dict]:
    """0L R2: ONE-SHOT normal repair for resumed / corrupted states
    (an explicit operation, deliberately NOT a redistribution flag --
    the t=0 production dynamics never call this).

    Re-anchors the stored normals to the loopwise-PCA normals and
    accepts the result only if (reviewer section 5):
      - the certified loop partition is UNCHANGED (0K shared
        resolver before/after),
      - the bulk WINDING INCIDENCE is unchanged (the winding reads
        the normals, so a repair can move it even at fixed
        positions),
      - the PCA spectral certificate and the cos-30-degree
        orientation margin hold.
    Records Delta V_cur, Delta P# (shared semantics: loopwise mass x
    q^WB refreshed on-state), Delta r_l, and the RELATIVE energy
    jump delta_P_plus = (Delta P#)_+ / max(P#, 1). Fail-closed on
    any certificate violation."""
    from src.torch.oriented_varifold import OrientedPointCloudVarifold
    from src.torch.oriented_varifold.loopwise_mass import (
        resolve_loopwise_oriented_mass_source,
    )
    from src.torch.perimeter.coherence_perimeter import (
        compute_coherence_loopwise,
    )
    from src.torch.transport import compute_coherence
    from src.torch.transport.incidence import (
        PartitionAmbiguousError,
        partition_relation,
        winding_bulk_labels,
    )

    def _state(pos, ang):
        v = OrientedPointCloudVarifold(positions=pos, angles=ang)
        res = resolve_loopwise_oriented_mass_source(
            pos, v.normals, delta, tau, kernel,
            ell_factor=ell_factor)
        m, labels = res.m_loop, res.loop_labels_pre
        q_full = compute_coherence(v, m, sigma, coherence_kernel)
        q_self, _ = compute_coherence_loopwise(
            pos, v.normals, m, sigma, coherence_kernel,
            loop_labels=labels)
        r_by_loop = {}
        p_sharp = 0.0
        for lb in labels.unique():
            sel = labels == lb
            A = float((m[sel] * q_full[sel]).sum())
            B = float((m[sel] * q_self[sel]).sum())
            r = A / B
            r_by_loop[int(lb)] = r
            p_sharp += r * B
        v_cur = float(0.5 * (m * (pos * v.normals).sum(-1)).sum())
        spacing = float(m.median())
        bulk, margin = winding_bulk_labels(pos, v.normals, m,
                                           labels, spacing)
        return v, res, labels, p_sharp, v_cur, r_by_loop, bulk, \
            margin

    (v0, res0, lbl0, p0, vc0, r0, bulk0, mrg0) = _state(positions,
                                                        angles)
    new_ang, ret_stats = pca_normal_reanchor(
        positions, angles, delta, kernel, lbl0, pca_rho_max,
        pca_neff_min, pca_ori_margin)
    (v1, res1, lbl1, p1, vc1, r1, bulk1, mrg1) = _state(positions,
                                                        new_ang)
    if partition_relation(lbl0, lbl1) != "equal":
        raise PartitionAmbiguousError(
            "normal repair changed the certified loop partition -- "
            "fail-closed")
    if bulk0 is None or bulk1 is None or partition_relation(
            bulk0[lbl0], bulk1[lbl1]) != "equal":
        raise PartitionAmbiguousError(
            "normal repair changed (or decertified) the bulk "
            "winding incidence -- fail-closed")
    stats = dict(
        retraction=ret_stats,
        delta_p_sharp=p1 - p0,
        delta_p_plus_rel=max(p1 - p0, 0.0) / max(p0, 1.0),
        delta_v_cur=vc1 - vc0,
        r_before=r0, r_after=r1,
        incidence_margin_before=mrg0,
        incidence_margin_after=mrg1,
    )
    return new_ang, stats


def redistribute_with_rule(
    positions: _T,
    angles: _T,
    rule: str,
    *,
    delta: float,
    kernel: str = "wendland_c2",
    n_iters: int = 10,
    step_size: float = 0.01,
    tol: float = 1e-4,
    max_disp_ratio: float = 0.05,
    mass_tau: float = 0.0,
    delta_redist: float | None = None,
    coherence: _T | None = None,
    sigma: float | None = None,
    coherence_kernel: str = "wendland_c2",
    coherence_backend: str = "naive",
    q_override: str | None = None,
    curvature_visible_qm: bool = False,
    checkpoint_every: int | None = None,
    cap_mode: str = "global",
    loops: list | None = None,
    tangent_source: str = "angles",
    pca_delta_tan: float | None = None,
    loop_labels: _T | None = None,
    density_scope: str = "global",
    curvature_scope: str = "union",
    kappa_den_beta_min: float | None = None,
    kappa_den_gamma_min: float | None = None,
    normal_retraction: bool = False,
    pca_rho_max: float | None = None,
    pca_neff_min: float | None = None,
    pca_ori_margin: float = 0.8660254037844387,   # cos 30 deg
    monotone: bool = False,
    monotone_parts: str = "abc",
) -> tuple[_T, _T, dict]:
    """Run the redistribution operator under one time-level rule.

    `coherence` is the caller's (stale, pre-MM in the solver) q --
    consumed only by legacy_hybrid. post_mm_frozen ignores it and
    recomputes q once on the input; substep_refreshed recomputes every
    subiteration (both need `sigma`).

    D2a.1 control knobs (audit-only; None/False reproduces the rule
    exactly as defined above):
      q_override        "unit" forces q == 1 for ANY rule; "fixed"
                        forces the caller-supplied `coherence`, frozen,
                        for ANY rule -- separates the q-suppression
                        confound from the tangent/curvature axis.
      curvature_visible_qm
                        substep_refreshed only: weight the curvature
                        estimate by the VISIBLE mass q*m instead of the
                        raw carrier m (the manuscript-Algorithm-3
                        pairing). The default raw-m variant is
                        deliberately NOT called an exact Algorithm-3
                        implementation.
      checkpoint_every  store (iteration, positions, angles) snapshots
                        every k subiterations in info["checkpoints"]
                        (equal-CV Pareto comparisons).
      cap_mode          "global" (production: one scalar rescale of the
                        whole update field so its max norm equals
                        max_disp) or "local_knn" (COUNTERFACTUAL
                        control, D2b-0.1/D2b-2: pointwise cap
                        |dx_i| <= 0.5 h_i, h_i = 2nd-nearest-neighbour
                        distance -- permutation-invariant; the earlier
                        roll-based "local_h" consumed array order and
                        was replaced; audit-only).
      loops             optional list of index tensors; adds per-loop
                        preclip/postclip maxima, the loop holding the
                        global max, and h_min/h_median per loop to the
                        trace (attribution diagnostics; trace-only).
      tangent_source    "angles" (default: tangents from the stored
                        angles, per the rule -- a PER-PARTICLE mark, so
                        permutation-equivariant like the rest of the
                        production scheme);
                        "local_pca" (COUNTERFACTUAL, D2b-1 axis B:
                        permutation-INVARIANT geometric tangent refresh
                        -- weighted local covariance within delta_redist
                        with a normal-compatibility gate
                        (N_i . N_j > 0), principal direction, sign fixed
                        by the stored tangent; the production-compatible
                        tangent-refresh control);
                        "ordered_loop_oracle" (DIAGNOSTIC ONLY: loop-wise
                        central differences that consume the caller's
                        `loops` CYCLIC ORDER -- array order is a particle
                        label, not connectivity, so this is an
                        ordered-curve oracle valid only on generator-
                        ordered fixtures, never a point-cloud method).

    0N monotone loopwise closure (reviewer 2026-08-18; monotone=True,
    density_scope="loopwise" only; default False is bitwise):
      (i)   per-loop ACTIVE SET: only loops with CV_l >= tol move in a
            subiteration; a loop that reaches CV_l < tol is frozen for
            the rest of the call (same-loop density -> other loops
            cannot change its CV);
      (ii)  per-loop MONOTONE BACKTRACKING on the POST-cap update:
            alpha_l in {1, 1/2, 1/4, ...} is the first factor with
            Phi_l(X + alpha_l u_l) <= Phi_l(X) + eps_round,
            Phi_l = CV_l^2; no factor found -> the loop's update is
            rejected for this subiteration (stalled_no_descent), never
            forced; alpha is per loop (a global alpha would re-couple
            the loops);
      (iii) STAGE-TOTAL cap: max_{i in l} |x_i^k - x_i^source| <=
            max_disp for the WHOLE call (the per-iteration cap alone
            let 10 capped subiterations accumulate a 10 x max_disp
            jolt) -- each subiteration may spend only the remaining
            budget of its loop.
    monotone_parts (0N-2 factorial DIAGNOSTIC; production "abc"):
    "a" = per-loop active set, "b" = per-loop backtracking, "c" =
    stage-total cap; any subset for attribution arms.
    Termination reason recorded in info["stop_reason"]:
    within_tol / stalled_no_descent / stage_budget_exhausted /
    max_iters, plus per-loop accepted alpha and stage displacement.
    """
    if rule not in REDISTRIBUTION_RULES:
        raise ValueError(f"unknown redistribution rule {rule!r}")
    if (rule in ("post_mm_frozen", "substep_refreshed")
            and sigma is None and q_override not in ("unit", "fixed")):
        raise ValueError(f"{rule} needs sigma to recompute coherence")
    if q_override not in (None, "unit", "fixed"):
        raise ValueError(f"unknown q_override {q_override!r}")
    if q_override == "fixed" and coherence is None:
        raise ValueError("q_override='fixed' needs the coherence tensor")
    if curvature_visible_qm and rule != "substep_refreshed":
        raise ValueError(
            "curvature_visible_qm is a substep_refreshed variant knob")
    if cap_mode not in ("global", "local_knn"):
        raise ValueError(f"unknown cap_mode {cap_mode!r}")
    if tangent_source not in ("angles", "local_pca",
                              "ordered_loop_oracle"):
        raise ValueError(f"unknown tangent_source {tangent_source!r}")
    if tangent_source == "ordered_loop_oracle" and loops is None:
        raise ValueError("ordered_loop_oracle needs the loops' cyclic "
                         "order (oracle diagnostic)")
    # 0L-R knobs. loop_labels is the per-particle CERTIFIED partition
    # (mask semantics, 0K family); `loops` above is a list of cyclic
    # orders (oracle/diagnostics) -- deliberately distinct objects.
    if density_scope not in ("global", "loopwise"):
        raise ValueError(f"unknown density_scope {density_scope!r}")
    if curvature_scope not in ("union", "loopwise"):
        raise ValueError(
            f"unknown curvature_scope {curvature_scope!r}")
    if density_scope == "loopwise" and loop_labels is None:
        raise RedistributionGeometryError(
            "density_scope='loopwise' requires certified loop_labels")
    if curvature_scope == "loopwise" and loop_labels is None:
        raise RedistributionGeometryError(
            "curvature_scope='loopwise' requires certified "
            "loop_labels")
    if normal_retraction and loop_labels is None:
        raise RedistributionGeometryError(
            "normal_retraction requires certified loop_labels (the "
            "PCA neighbourhood must be loopwise, not gated by the "
            "corrupted stored normals)")
    if monotone and density_scope != "loopwise":
        raise RedistributionGeometryError(
            "monotone redistribution is defined loopwise (per-loop "
            "active set / merit / cap need certified loop_labels)")
    if monotone and rule == "substep_refreshed":
        raise RedistributionGeometryError(
            "monotone closure is implemented for the single-final-"
            "correction rules (legacy_hybrid / post_mm_frozen)")

    d_redist = delta_redist if delta_redist is not None else delta
    max_disp = max_disp_ratio * d_redist
    pos = positions.clone()
    ang = angles.clone()

    def _q_on(p, a):
        m = compute_masses(p.detach(), delta, mass_tau, kernel)
        vf = OrientedPointCloudVarifold(positions=p.detach(),
                                        angles=a.detach())
        return compute_coherence(vf, m, sigma, coherence_kernel,
                                 backend=coherence_backend)

    if q_override == "unit":
        q_fixed, q_src, refresh_q = None, "unit (forced q == 1)", False
    elif q_override == "fixed":
        q_fixed, q_src, refresh_q = coherence, "caller, frozen (forced)", \
            False
    elif rule == "legacy_hybrid":
        q_fixed, q_src, refresh_q = coherence, "caller (stale pre-MM)", \
            False
    elif rule == "post_mm_frozen":
        q_fixed, q_src, refresh_q = _q_on(positions, angles), \
            "input geometry, once", False
    else:
        q_fixed, q_src, refresh_q = None, \
            "current geometry, every subiteration", True

    def _q_policy_current(p, a):
        """SINGLE source of the current-geometry q under the active
        policy -- both the update scaling and the visible-qm curvature
        weights go through here, so q_override='unit' really forces
        q == 1 everywhere (review: the earlier version recomputed real
        q for the curvature side under unit override)."""
        if q_override == "unit":
            return None                      # q == 1
        if q_override == "fixed":
            return coherence
        if refresh_q:
            return _q_on(p, a)
        return q_fixed

    tangents = torch.stack([-torch.sin(ang), torch.cos(ang)], dim=1)

    cv_history, trace_rows, checkpoints = [], [], []
    actual_iters = 0
    frozen_loops = set()
    stop_reason = "max_iters"
    for it in range(n_iters):
        if tangent_source == "ordered_loop_oracle":
            tangents, _ = _geom_tangents_normals(pos, loops)
        elif tangent_source == "local_pca":
            dtan = (pca_delta_tan if pca_delta_tan is not None
                    else d_redist)
            if loop_labels is not None:
                tangents, pca_cert = _local_pca_tangents(
                    pos, ang, dtan, kernel, loop_labels=loop_labels,
                    return_certificate=True)
                _enforce_pca_certificate(pca_cert, pca_rho_max,
                                         pca_neff_min)
            else:
                tangents = _local_pca_tangents(pos, ang, dtan, kernel)
        elif rule == "substep_refreshed":
            tangents = torch.stack([-torch.sin(ang), torch.cos(ang)],
                                   dim=1)
        grad_log_theta, theta = _compute_density_log_gradient(
            pos, d_redist, kernel,
            loop_labels if density_scope == "loopwise" else None)
        cv_by_loop = None
        if density_scope == "loopwise":
            # 0L-R: per-loop CV -- the global CV misreads the
            # BETWEEN-loop mean-density difference as nonuniformity
            cv_by_loop = {}
            for lb in loop_labels.unique():
                th_l = theta[loop_labels == lb]
                cv_by_loop[int(lb)] = (th_l.std().item()
                                       / (th_l.mean().item() + 1e-30))
            cv = max(cv_by_loop.values())
        else:
            cv = theta.std().item() / (theta.mean().item() + 1e-30)
        cv_history.append(cv)
        if monotone:
            # 0N (i): per-loop active set -- frozen loops never move
            if "a" in monotone_parts:
                active_loops = [lb for lb, c in cv_by_loop.items()
                                if c >= tol and lb not in frozen_loops]
                for lb, c in cv_by_loop.items():
                    if c < tol:
                        frozen_loops.add(lb)
            else:
                # legacy activation: all loops move while max CV >= tol
                active_loops = ([lb for lb in cv_by_loop
                                 if lb not in frozen_loops]
                                if cv >= tol else [])
            if not active_loops:
                stop_reason = "within_tol"
                break
        elif cv < tol:
            break

        w_tan = -(grad_log_theta * tangents).sum(
            dim=1, keepdim=True) * tangents
        update = step_size * w_tan
        q_it = _q_policy_current(pos, ang) if refresh_q else q_fixed
        if q_it is not None:
            update = update * q_it.unsqueeze(-1)

        # NOTE (review): the production cap is a GLOBAL rescaling, not
        # a pointwise clip -- when any point exceeds max_disp the
        # ENTIRE update field is scaled so its maximum norm equals
        # max_disp
        preclip_norm = update.norm(dim=1)
        disp_max = preclip_norm.max()
        if density_scope == "loopwise" and cap_mode == "global":
            # 0L-R: per-loop cap alpha_l -- the global scalar rescale
            # couples the loops (one loop's large update would shrink
            # every other loop's redistribution), breaking the
            # full-operator cross-loop independence
            scale = torch.ones_like(preclip_norm)
            clipped = False
            for lb in loop_labels.unique():
                sel = loop_labels == lb
                mx = preclip_norm[sel].max()
                if float(mx) > max_disp:
                    scale[sel] = max_disp / mx
                    clipped = True
            update = update * scale.unsqueeze(-1)
        elif cap_mode == "local_knn":
            # permutation-INVARIANT local scale (review: the previous
            # "local_h" control used array-order roll and changed the
            # output -- an ordered oracle in disguise): h_i = distance
            # to the 2nd nearest neighbour (on a sampled curve, the
            # two adjacent points are the 1st/2nd NN)
            d = torch.cdist(pos, pos)
            h_knn = d.kthvalue(3, dim=1).values     # self + 2 NN
            cap = 0.5 * h_knn
            scale = (cap / preclip_norm.clamp_min(1e-30)).clamp(max=1.0)
            clipped = bool((scale < 1.0).any().item())
            update = update * scale.unsqueeze(-1)
        else:
            clipped = disp_max.item() > max_disp
            if clipped:
                update = update * (max_disp / disp_max)

        mono_row = None
        if monotone:
            # 0N (i): zero the update outside the active set
            act_mask = torch.zeros(pos.shape[0], dtype=torch.bool,
                                   device=pos.device)
            for lb in active_loops:
                act_mask |= loop_labels == lb
            update = update * act_mask.unsqueeze(-1).to(update.dtype)
            # 0N (iii): stage-total budget per loop
            moved = (pos - positions).norm(dim=1)
            exhausted = []
            for lb in (list(active_loops) if "c" in monotone_parts
                       else []):
                sel = loop_labels == lb
                remaining = max_disp - float(moved[sel].max())
                mx = float(update[sel].norm(dim=1).max())
                if remaining <= 1e-9 * max_disp:
                    update[sel] = 0.0
                    exhausted.append(lb)
                elif mx > remaining:
                    update[sel] = update[sel] * (remaining / mx)
            for lb in exhausted:
                active_loops.remove(lb)
                frozen_loops.add(lb)
            if not active_loops:
                stop_reason = "stage_budget_exhausted"
                trace_rows.append(dict(iteration=it, cv_before=cv,
                                       cv_by_loop=cv_by_loop,
                                       monotone=dict(
                                           exhausted_loops=[int(x) for x
                                                            in exhausted])))
                break
            # 0N (ii): per-loop monotone backtracking on the POST-cap
            # update, merit Phi_l = CV_l^2 of the same-loop density
            phi0 = {lb: cv_by_loop[lb] ** 2 for lb in active_loops}
            alphas = {}
            stalled = []
            eps_round = 1e-12
            for lb in active_loops:
                sel = loop_labels == lb
                u_l = update[sel]
                if float(u_l.norm(dim=1).max()) == 0.0:
                    alphas[lb] = 0.0
                    continue
                if "b" not in monotone_parts:
                    alphas[lb] = 1.0
                    continue
                accepted = None
                for j in range(8):
                    a = 0.5 ** j
                    trial = pos.clone()
                    trial[sel] = pos[sel] + a * u_l
                    _, th_t = _compute_density_log_gradient(
                        trial, d_redist, kernel, loop_labels)
                    th_l = th_t[sel]
                    phi = (th_l.std().item()
                           / (th_l.mean().item() + 1e-30)) ** 2
                    if phi <= phi0[lb] + eps_round:
                        accepted = a
                        break
                if accepted is None:
                    update[sel] = 0.0
                    alphas[lb] = 0.0
                    stalled.append(int(lb))
                else:
                    update[sel] = accepted * u_l
                    alphas[lb] = accepted
            mono_row = dict(active_loops=[int(x) for x in active_loops],
                            frozen_loops=sorted(int(x) for x in
                                                frozen_loops),
                            alpha_by_loop={int(k): v
                                           for k, v in alphas.items()},
                            stalled_loops=stalled,
                            stage_moved_max_by_loop={
                                int(lb): float(moved[loop_labels == lb]
                                               .max())
                                for lb in loop_labels.unique()})
            if stalled and len(stalled) == len(active_loops):
                stop_reason = "stalled_no_descent"
                trace_rows.append(dict(iteration=it, cv_before=cv,
                                       cv_by_loop=cv_by_loop,
                                       monotone=mono_row))
                break

        # normal leakage + stored-vs-geometric tangent mismatch: the
        # measuring frame CONSUMES a cyclic order, so it is only
        # computed when the caller declares one via `loops` (review
        # D2b-2 hygiene 2: with no declared order the fields are
        # unavailable, not silently array-order-dependent)
        if loops is not None:
            t_geom, n_geom = _geom_tangents_normals(pos, loops)
            cosang = (tangents * t_geom).sum(dim=1).abs().clamp(0.0, 1.0)
            upd_n = (update * n_geom).sum(dim=1)
            leak = dict(
                leakage_frame="ordered_curve(loops)",
                normal_update_norm=upd_n.abs().max().item(),
                stored_vs_geom_tangent_max_deg=(
                    torch.rad2deg(torch.acos(cosang)).max().item()))
        else:
            leak = dict(leakage_frame="unavailable (no declared "
                                      "cyclic order)",
                        normal_update_norm=None,
                        stored_vs_geom_tangent_max_deg=None)
        loop_diag = None
        if loops is not None:
            post_norm = update.norm(dim=1)
            seg_all = (pos.roll(-1, 0) - pos).norm(dim=1)
            loop_diag = []
            for c in loops:
                # intra-loop segments only (drop the wrap chord to the
                # next loop by using consecutive pairs within c)
                Pl = pos[c]
                segs = (Pl.roll(-1, 0) - Pl).norm(dim=1)
                loop_diag.append(dict(
                    preclip_max=preclip_norm[c].max().item(),
                    postclip_max=post_norm[c].max().item(),
                    h_min=segs.min().item(),
                    h_median=segs.median().item()))
            _ = seg_all
            argmax_pt = int(preclip_norm.argmax().item())
            loop_of = next((k for k, c in enumerate(loops)
                            if argmax_pt in set(c.tolist())), None)
        trace_rows.append(dict(
            **(dict(by_loop=loop_diag, argmax_loop=loop_of)
               if loops is not None else {}),
            **(dict(cv_by_loop=cv_by_loop)
               if cv_by_loop is not None else {}),
            iteration=it, cv_before=cv,
            tangent_source=("current angles" if rule == "substep_refreshed"
                            else "input angles, fixed"),
            coherence_source=q_src,
            density_geometry="current positions",
            curvature_geometry=("current positions"
                                if rule == "substep_refreshed"
                                else "input positions (final correction)"),
            curvature_mass_used=(
                ("q*m(current), visible" if curvature_visible_qm
                 else "m(current), raw")
                if rule == "substep_refreshed" else "m(input), raw"),
            preclip_max=preclip_norm.max().item(),
            postclip_max=update.norm(dim=1).max().item(),
            clip_active=clipped,
            global_clip_scale=(min(1.0, max_disp / disp_max.item())
                               if disp_max.item() > 0 else 1.0),
            fraction_above_threshold=float(
                (preclip_norm > max_disp).float().mean().item()),
            max_disp=max_disp,
            tangential_update_norm=update.norm(dim=1).max().item(),
            **leak,
            **(dict(monotone=mono_row) if mono_row is not None else {}),
        ))

        if rule == "substep_refreshed":
            # incremental Frenet angle correction on the geometry the
            # tangential displacement happened on
            normals = torch.stack([torch.cos(ang), torch.sin(ang)],
                                  dim=1)
            if curvature_scope == "loopwise":
                from src.torch.oriented_varifold.mass import (
                    compute_masses_oriented_loopwise,
                )
                m_it = compute_masses_oriented_loopwise(
                    pos.detach(), normals, loop_labels, delta,
                    mass_tau, kernel)
            else:
                m_it = compute_masses(pos.detach(), delta, mass_tau,
                                      kernel)
            w_curv = m_it
            if curvature_visible_qm:
                q_curv = _q_policy_current(pos, ang)
                if q_curv is not None:      # None == unit q: w = m
                    w_curv = m_it * q_curv
            kappa, _ = compute_regularized_curvature(
                pos, normals, w_curv, epsilon=delta, kernel=kernel,
                loop_labels=(loop_labels
                             if curvature_scope == "loopwise"
                             else None))
            u_tan_it = (update * tangents).sum(dim=1)
            ang = wrap_angles(ang + kappa * u_tan_it)

        pos = pos + update
        actual_iters = it + 1
        if checkpoint_every and (it + 1) % checkpoint_every == 0:
            checkpoints.append((it + 1, pos.detach().clone(),
                                ang.detach().clone()))

    _, theta_final = _compute_density_log_gradient(
        pos, d_redist, kernel,
        loop_labels if density_scope == "loopwise" else None)
    if density_scope == "loopwise":
        final_by_loop = {}
        for lb in loop_labels.unique():
            th_l = theta_final[loop_labels == lb]
            final_by_loop[int(lb)] = (th_l.std().item()
                                      / (th_l.mean().item() + 1e-30))
        cv_history.append(max(final_by_loop.values()))
    else:
        final_by_loop = None
        cv_history.append(theta_final.std().item()
                          / (theta_final.mean().item() + 1e-30))

    retraction_stats = None
    if normal_retraction:
        # 0L-R: PCA normal re-anchoring REPLACES the legacy final
        # curvature correction -- the two must never compose (both
        # write the angles)
        ang, retraction_stats = pca_normal_reanchor(
            pos, ang, pca_delta_tan if pca_delta_tan is not None
            else d_redist, kernel, loop_labels, pca_rho_max,
            pca_neff_min, pca_ori_margin)
    elif rule != "substep_refreshed":
        # legacy/frozen: one final angle correction from the INPUT
        # geometry (bitwise the production redistribute_points path).
        # 0L-kappa: curvature_scope="loopwise" changes ONLY the
        # estimator inputs -- same-loop kappa AND the 0K loopwise
        # masses (masking the pairwise support while keeping the
        # union mass would re-import the mass-channel contamination);
        # time level, tangents and single-shot application unchanged.
        normals = torch.stack([torch.cos(angles), torch.sin(angles)],
                              dim=1)
        if curvature_scope == "loopwise":
            from src.torch.oriented_varifold.mass import (
                compute_masses_oriented_loopwise,
            )
            masses = compute_masses_oriented_loopwise(
                positions, normals, loop_labels, delta, mass_tau,
                kernel)
            kappa, _, b_raw = compute_regularized_curvature(
                positions, normals, masses, epsilon=delta,
                kernel=kernel, loop_labels=loop_labels,
                return_denominator=True)
            _enforce_kappa_denominator(
                b_raw, loop_labels, delta, kernel,
                kappa_den_beta_min, kappa_den_gamma_min)
        else:
            masses = compute_masses(positions, delta, mass_tau,
                                    kernel)
            kappa, _ = compute_regularized_curvature(
                positions, normals, masses, epsilon=delta,
                kernel=kernel)
        u_tan = ((pos - positions) * tangents).sum(dim=1)
        ang = wrap_angles(angles + kappa * u_tan)
        if monotone:
            unmoved = (pos - positions).norm(dim=1) == 0.0
            ang = torch.where(unmoved, angles, ang)

    if not monotone:
        stop_reason = ("within_tol" if actual_iters < n_iters
                       else "max_iters")
    info = dict(rule=rule, cv_history=cv_history, trace=trace_rows,
                n_iters=actual_iters,
                converged=(stop_reason == "within_tol"),
                stop_reason=stop_reason,
                monotone=monotone,
                stage_moved_max=float((pos - positions).norm(dim=1)
                                      .max()),
                checkpoints=checkpoints,
                cv_final_by_loop=final_by_loop,
                retraction=retraction_stats)
    return pos.detach(), ang.detach(), info
