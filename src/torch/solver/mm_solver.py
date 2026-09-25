"""MM (Minimizing Movements) solver for Mullins-Sekerka flow.

Implements the full evolution loop using BEM-based linearized Wasserstein:

    μ^n = argmin_{y, θ} [ P̂(μ) + W_lin(s) ]

This iterates mm_step for multiple time steps, tracking the evolution
of the boundary.
"""

import torch
from dataclasses import dataclass, field
from typing import Callable

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import compute_masses, compute_recommended_params

from .mm_step import MMConfig, MMStepResult, MMStepper, mm_step
from .dead_point_rules import evaluate_dead_points
from .redistribute import redistribute_points  # noqa: F401 (parity ref)
from .redistribution_rules import redistribute_with_rule


@dataclass
class EvolutionHistory:
    """History of MM evolution.

    Stores varifolds as a list to support variable N (point removal).
    Scalar quantities (perimeter, volume, etc.) use pre-allocated tensors.

    Attributes:
        _varifolds: List of varifolds at each step (length = n_completed + 1).
        perimeters: (n_steps,) tensor of perimeter values.
        wassersteins: (n_steps,) tensor of Wasserstein values.
        objectives: (n_steps,) tensor of objective values.
        volumes: (n_steps+1,) tensor of computed volumes.
        converged: (n_steps,) tensor of convergence flags.
        n_completed: Number of completed steps.
    """
    _varifolds: list[OrientedPointCloudVarifold] = field(repr=False)
    perimeters: torch.Tensor     # (n_steps,)
    wassersteins: torch.Tensor   # (n_steps,)
    objectives: torch.Tensor     # (n_steps,)
    volumes: torch.Tensor        # (n_steps+1,)
    converged: torch.Tensor      # (n_steps,)
    n_completed: int
    # Structured stop reason (review, D2b-1 correction 4): a short
    # history alone does not reproduce WHY a run stopped -- the caught
    # stepper exception is recorded here instead of only printed.
    stop_step: int | None = None
    stop_stage: str | None = None
    stop_exception_type: str | None = None
    stop_message: str | None = None
    # 0L-B2: structured audit trail of a fail-closed quotient event
    # (both shadow arms + the gate table); None otherwise
    stop_details: dict | None = None
    # P2.1 semantics: when a support gate closes on an INVALID terminal
    # state, that frame is kept as a terminal OBSERVATION (the event
    # geometry must not be lost) but must not be consumed as the next
    # MM state. last_valid_step indexes the last step whose committed
    # support was valid; terminal_candidate_valid marks whether the
    # final stored frame may be evolved further.
    terminal_candidate_valid: bool = True
    last_valid_step: int | None = None

    def get_final_valid_varifold(self) -> OrientedPointCloudVarifold:
        """The last VALID committed state -- the one a production
        caller may evolve further. Falls back to the final frame when
        no gate ever invalidated it."""
        if self.terminal_candidate_valid or self.last_valid_step is None:
            return self.get_final_varifold()
        return self._varifolds[self.last_valid_step + 1]

    def __len__(self) -> int:
        return self.n_completed + 1

    def get_varifold(self, step: int) -> OrientedPointCloudVarifold:
        """Get varifold at given step."""
        return self._varifolds[step]

    def get_final_varifold(self) -> OrientedPointCloudVarifold:
        """Get varifold at final completed step."""
        return self.get_varifold(self.n_completed)

    @property
    def positions(self) -> torch.Tensor:
        """(n_completed+1, N, 2) positions tensor (backward compat).

        Raises RuntimeError if N varies across steps.
        """
        vfs = self._varifolds[:self.n_completed + 1]
        return torch.stack([v.positions for v in vfs])

    @property
    def angles(self) -> torch.Tensor:
        """(n_completed+1, N) angles tensor (backward compat).

        Raises RuntimeError if N varies across steps.
        """
        vfs = self._varifolds[:self.n_completed + 1]
        return torch.stack([v.angles for v in vfs])


def create_history(
    n_steps: int,
    device: str,
    dtype: torch.dtype,
) -> EvolutionHistory:
    """Create history with pre-allocated scalar tensors."""
    return EvolutionHistory(
        _varifolds=[],
        perimeters=torch.zeros(n_steps, device=device, dtype=dtype),
        wassersteins=torch.zeros(n_steps, device=device, dtype=dtype),
        objectives=torch.zeros(n_steps, device=device, dtype=dtype),
        volumes=torch.zeros(n_steps + 1, device=device, dtype=dtype),
        converged=torch.zeros(n_steps, device=device, dtype=torch.bool),
        n_completed=0,
    )


def compute_volume_divergence(
    varifold: OrientedPointCloudVarifold,
    masses: torch.Tensor,
) -> float:
    """Compute volume (area in 2D) using the divergence theorem.

    Area = (1/2) Σ_i (x_i cos θ_i + y_i sin θ_i) · m_i

    This does NOT require point ordering and works for multi-component
    boundaries and after topology changes (e.g., fusion).

    Args:
        varifold: Oriented point cloud with positions and angles.
        masses: (N,) KDE masses (arc length weights).

    Returns:
        Area enclosed by the boundary.
    """
    pos = varifold.positions
    normals = torch.stack(
        [torch.cos(varifold.angles), torch.sin(varifold.angles)], dim=1
    )
    area = 0.5 * (pos * normals).sum(dim=1).dot(masses)
    return area.item()


def compute_volume_shoelace(varifold: OrientedPointCloudVarifold) -> float:
    """Compute volume (area in 2D) using the shoelace formula.

    Warning:
        This function assumes points are ordered consecutively around the
        boundary (i.e., point i is adjacent to points i-1 and i+1).
        If points are unordered or represent multiple disconnected components,
        the result will be incorrect. For multi-component boundaries, each
        component must be processed separately.

    Args:
        varifold: Oriented point cloud with positions ordered along boundary.

    Returns:
        Signed area enclosed by the boundary (positive for counter-clockwise).
    """
    positions = varifold.positions
    x = positions[:, 0]
    y = positions[:, 1]

    x_next = torch.roll(x, -1)
    y_next = torch.roll(y, -1)

    area = 0.5 * torch.abs((x * y_next - x_next * y).sum())
    return area.item()


# Keep old name for backward compatibility
compute_volume = compute_volume_shoelace


def _circ_diff(a, b):
    """Elementwise circular angle distance |a - b| mod 2pi -> [0, pi]."""
    import math as _m
    d = (a - b + _m.pi) % (2 * _m.pi) - _m.pi
    return d.abs()


class QuotientSwitchError(RuntimeError):
    """0L-B2: the drop arm of the shadow transaction failed a
    pre-registered gate; the run stops fail-closed with both arms'
    telemetry in .details."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


class QuotientInvariantError(RuntimeError):
    """0L-B2: after an irreversible quotient the contact certificate
    stopped holding on a committed state (never auto-revives C_-)."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


def _wrap_angle(a):
    import math as _m
    return (a + _m.pi) % (2 * _m.pi) - _m.pi


class MMSolver:
    """MM scheme solver for Mullins-Sekerka flow using BEM.

    Example:
        config = MMConfig(time_step=0.01)
        solver = MMSolver(config)
        history = solver.solve(initial_varifold, n_steps=100)
    """

    def __init__(self, config: MMConfig):
        """Initialize solver."""
        self.config = config

    _STALE_DEFAULT = object()

    def _redistribute(self, stepper, varifold, n_iters,
                      stale_q=_STALE_DEFAULT):
        """Route redistribution through the shared time-level helper.
        The caller's stale (pre-MM) coherence feeds legacy_hybrid only;
        post_mm_frozen/substep_refreshed recompute their own q from
        sigma with the production perimeter kernel/backend."""
        cfg = self.config
        if stale_q is self._STALE_DEFAULT:
            # 0L R2 q-policy: chosen DIRECTLY from the stepper's
            # source-frozen tensors by particle index (never by
            # re-matching label IDs -- a re-certified partition may
            # renumber labels; particle ordering is preserved
            # between the MM step and the redistribution)
            if cfg.redistribution_q_policy == "r_loop":
                # L0J: under contact_complex_renormalized this is the
                # loopwise-GLOBAL r_l (constant per loop, pseudo-time
                # only) -- NEVER the side-wise contact factors
                # (reviewer 2026-08-21: perimeter well-balancing and
                # redistribution weighting must not be conflated)
                stale_q = stepper.perimeter_r_per_particle
                if stale_q is None:
                    raise ValueError(
                        "redistribution_q_policy='r_loop' requires "
                        "perimeter_q_mode='self_renormalized' or "
                        "'contact_complex_renormalized' (the r_l "
                        "tensor is produced there)")
            elif cfg.redistribution_q_policy == "q_wb":
                if cfg.perimeter_q_mode == "contact_complex_renormalized":
                    raise ValueError(
                        "redistribution_q_policy='q_wb' is not "
                        "validated with contact_complex_renormalized "
                        "(a non-uniform per-loop weight would change "
                        "the redistribution path; use 'r_loop')")
                stale_q = stepper.perimeter_coherence
            else:
                stale_q = stepper.fixed_coherence
        # 0L-R: the loopwise/PCA/re-anchoring features need the
        # certified loop labels, RE-CERTIFIED on the post-MM state
        # through the SAME two-stage resolver as 0K (comparing
        # certificates built at different mass scales would be
        # meaningless); pre-MM vs post-MM partition must agree
        # (co-membership) -- fail-closed otherwise.
        loop_labels = None
        uses_geometry = (
            cfg.redistribution_density_scope == "loopwise"
            or cfg.redistribution_tangent_source == "local_pca"
            or cfg.redistribution_normal_retraction
            or cfg.redistribution_curvature_scope == "loopwise")
        if uses_geometry:
            from ..oriented_varifold.loopwise_mass import (
                resolve_loopwise_oriented_mass_source,
            )
            from ..transport.incidence import (
                PartitionAmbiguousError,
                partition_relation,
            )
            res_post = resolve_loopwise_oriented_mass_source(
                varifold.positions.detach(),
                varifold.normals.detach(),
                float(stepper._mass_delta_for_kde),
                float(stepper._mass_tau_for_kde),
                cfg.mass_kernel,
                ell_factor=cfg.grid_incidence_ell_factor)
            loop_labels = res_post.loop_labels_pre
            pre = stepper._loop_labels_last
            if pre is not None and partition_relation(
                    pre, loop_labels) != "equal":
                raise PartitionAmbiguousError(
                    "post-MM loop partition disagrees with the "
                    "pre-MM certified partition (co-membership) -- "
                    "redistribution geometry fail-closed")
        pos, ang, info = redistribute_with_rule(
            varifold.positions, varifold.angles, cfg.redistribution_rule,
            delta=stepper.mass_delta, kernel=cfg.mass_kernel,
            n_iters=n_iters,
            step_size=cfg.redistribute_step_size,
            tol=cfg.redistribute_tol,
            max_disp_ratio=cfg.redistribute_max_disp_ratio,
            mass_tau=stepper.mass_tau,
            delta_redist=stepper.mass_delta * cfg.redistribute_delta_ratio,
            coherence=stale_q, sigma=stepper._sigma,
            coherence_kernel=cfg.perimeter_kernel,
            coherence_backend=cfg.backend,
            loop_labels=loop_labels,
            density_scope=cfg.redistribution_density_scope,
            curvature_scope=cfg.redistribution_curvature_scope,
            kappa_den_beta_min=(
                cfg.redistribution_kappa_den_normalized_min),
            kappa_den_gamma_min=(
                cfg.redistribution_kappa_den_relative_min),
            tangent_source=cfg.redistribution_tangent_source,
            normal_retraction=cfg.redistribution_normal_retraction,
            pca_rho_max=cfg.redistribution_pca_rho_max,
            pca_neff_min=cfg.redistribution_pca_neff_min,
            # PCA bandwidth = the FULL mass delta (not delta_redist =
            # delta/2): the certificate calibration showed N_eff ~ 1.3
            # at delta/2 vs ~ 3-6 at delta -- the tangent estimate
            # needs a resolved neighbourhood even where the update
            # kernel is narrower
            pca_delta_tan=stepper.mass_delta,
            monotone=cfg.redistribution_monotone,
            monotone_parts=cfg.redistribution_monotone_parts)
        if cfg.redistribution_normal_retraction:
            # post-retraction shadow certificate: the re-anchoring
            # rewrites the normals, which can change the co-oriented
            # graph itself -- a state whose partition moved must not
            # be handed to the next MM step
            from ..oriented_varifold.loopwise_mass import (
                resolve_loopwise_oriented_mass_source,
            )
            from ..transport.incidence import (
                PartitionAmbiguousError,
                partition_relation,
            )
            vf_ret = OrientedPointCloudVarifold(positions=pos,
                                                angles=ang)
            res_ret = resolve_loopwise_oriented_mass_source(
                vf_ret.positions, vf_ret.normals,
                float(stepper._mass_delta_for_kde),
                float(stepper._mass_tau_for_kde),
                cfg.mass_kernel,
                ell_factor=cfg.grid_incidence_ell_factor)
            if partition_relation(loop_labels,
                                  res_ret.loop_labels_pre) != "equal":
                raise PartitionAmbiguousError(
                    "loop partition changed across the normal "
                    "re-anchoring -- fail-closed")
        return pos, ang, info

    # ---- 0L-B2 committed-level shadow transaction --------------------
    def _advance_committed(self, stepper, source):
        """One committed step = MM step + the production post-MM stage
        (redistribution only, under the quotient-mode guards: no
        deletion / PSP / vertical / advection). Returns (result,
        committed_varifold). Bitwise the solve-loop committed state
        under those guards (pinned)."""
        result = stepper.step(source)
        cur = result.varifold
        if self.config.redistribute and (
                self.config.redistribution_schedule == "every_step"
                and self.config.redistribute_interval == 1):
            new_pos, new_angles, _ = self._redistribute(
                stepper, cur, self.config.redistribute_n_iters)
            cur = OrientedPointCloudVarifold(positions=new_pos,
                                             angles=new_angles)
        elif self.config.redistribute:
            raise RuntimeError(
                "quotient shadow transaction supports redistribution "
                "schedule every_step / interval 1 only")
        return result, cur

    def _geometric_merged_volume(self, stepper, varifold, labels, masses):
        """V_merged^geom = signed sum of certified polygon areas over
        all loops (outward loops +, inward loops -)."""
        from src.torch.transport.loop_geometry import (
            certified_cyclic_order,
            polygon_area,
        )
        pos = varifold.positions.detach()
        nor = varifold.normals.detach()
        orders = certified_cyclic_order(pos, labels, nor, masses,
                                        float(masses.median()))
        total = 0.0
        for lb, lo in orders.items():
            sel = labels == lb
            sign = 1.0 if float((pos[sel] * nor[sel]).sum(1).mean()) > 0 \
                else -1.0
            total += sign * abs(float(polygon_area(pos, lo.order)))
        return total

    def _mm_with_shadow_switch(self, stepper, source, step):
        """0L-B2: keep/drop shadow transaction. mode 'off' -> bitwise
        stepper.step. Otherwise: run the keep (State II) MM step; if a
        certified, not-yet-merged State-II pair exists on the SOURCE
        state, run the drop arm (raw bulk pair merged) from the same
        source, judge BOTH on their committed states, and either commit
        the quotient irreversibly (returning the drop MM result) or
        raise QuotientSwitchError (fail-closed) with both arms."""
        cfg = self.config
        result_keep = stepper.step(source)
        if cfg.grid_quotient_mode == "off":
            return result_keep
        certs = getattr(stepper, "_contact_certificates_source", None)
        ctx = getattr(stepper, "_contact_pair_context", None)
        if not isinstance(certs, list) or ctx is None:
            return result_keep
        classes = (stepper._last_grid_snapshot.quotient_classes
                   if stepper._last_grid_snapshot is not None else None)
        pending = None
        for c in certs:
            if not c.certified:
                continue
            ba, bb = c.candidate.bulk_pair
            if classes is not None and classes[ba] == classes[bb]:
                continue                       # already merged
            pending = c
            break
        if pending is None:
            return result_keep
        # ---- keep arm committed --------------------------------------
        labels_src = stepper._loop_labels_last
        masses_src = stepper.fixed_masses
        h_wall = float(masses_src.median())
        P_src = float(result_keep.frozen_perimeter_initial)
        V_src = self._geometric_merged_volume(stepper, source, labels_src,
                                              masses_src)
        cur_keep = result_keep.varifold
        if cfg.redistribute:
            np_, na_, _ = self._redistribute(stepper, cur_keep,
                                             cfg.redistribute_n_iters)
            cur_keep = OrientedPointCloudVarifold(positions=np_, angles=na_)
        keep_tel = self._arm_telemetry(stepper, source, result_keep,
                                       cur_keep, P_src, V_src, ctx,
                                       labels_src, h_wall)
        # ---- drop arm from the SAME source ---------------------------
        ba, bb = pending.candidate.bulk_pair
        b_of_loop = ctx["loop_to_raw_bulk"]
        bulk_pp = torch.tensor([b_of_loop[int(l)] for l in
                                labels_src.tolist()],
                               device=labels_src.device)
        mask = (bulk_pp == ba) | (bulk_pp == bb)
        saved_groups = list(stepper._quotient_bulk_groups)
        saved_meta = dict(stepper._quotient_meta)
        stepper.commit_quotient(mask, step, pending.as_dict(), (ba, bb),
                                labels_src=labels_src)
        gates = {}
        try:
            result_drop, cur_drop = self._advance_committed(stepper, source)
            drop_tel = self._arm_telemetry(stepper, source, result_drop,
                                           cur_drop, P_src, V_src, ctx,
                                           labels_src, h_wall)
            drop_tel["exception"] = None
        except Exception as ex:          # noqa: BLE001 -- recorded
            result_drop, cur_drop = None, None
            drop_tel = {"exception": f"{type(ex).__name__}: {ex}"}
        # ---- gates (pre-registered) ---------------------------------
        def pos_(x):
            return max(x, 0.0)
        gates["drop_ran"] = drop_tel.get("exception") is None
        if gates["drop_ran"]:
            gates["split_residual"] = (
                pos_(drop_tel["split_residual"])
                <= max(pos_(keep_tel["split_residual"]),
                       cfg.grid_quotient_eps_split))
            dX = float((cur_drop.positions - cur_keep.positions)
                       .abs().max()) / h_wall
            dTh = float(_wrap_angle(cur_drop.angles - cur_keep.angles)
                        .abs().max())
            gates["switch_x"] = dX <= cfg.grid_quotient_eps_switch_x
            gates["switch_theta"] = dTh <= cfg.grid_quotient_eps_switch_theta
            gates["volume"] = (
                abs(drop_tel["V_merged_rel"])
                <= max(abs(keep_tel["V_merged_rel"]),
                       cfg.grid_quotient_eps_volume))
            cd = drop_tel.get("certificate")
            ck = keep_tel.get("certificate")
            gates["certificate_after"] = bool(cd and cd.get("certified"))
            # amendment (reviewer 2026-08-17): normalized-margin
            # monotonicity vs min(source, keep) with the numerical
            # comparison tolerance MARGIN_COMPARE_TOL (eq. 2.1); the B1
            # thresholds themselves are untouched
            from src.torch.transport.contact_certificate import (
                MARGIN_COMPARE_TOL,
                margins_not_worse,
            )
            if cd and ck:
                ok_m, table = margins_not_worse(pending.as_dict(), ck, cd)
            else:
                ok_m, table = False, None
            gates["margins_not_worse"] = bool(ok_m)
            drop_tel.update(dX_over_h=dX, dtheta_inf=dTh,
                            margin_table=table,
                            margin_tol=MARGIN_COMPARE_TOL)
        else:
            for k in ("split_residual", "switch_x", "switch_theta",
                      "volume", "certificate_after", "margins_not_worse"):
                gates[k] = False
        passed = all(gates.values())
        record = dict(step=int(step), bulk_pair=(int(ba), int(bb)),
                      loop_pair=tuple(int(x) for x in
                                      pending.candidate.loop_pair),
                      certificate_source=pending.as_dict(),
                      keep=keep_tel, drop=drop_tel, gates=gates,
                      thresholds=dict(
                          eps_split=cfg.grid_quotient_eps_split,
                          eps_switch_x=cfg.grid_quotient_eps_switch_x,
                          eps_switch_theta=cfg.grid_quotient_eps_switch_theta,
                          eps_volume=cfg.grid_quotient_eps_volume),
                      passed=passed)
        if not passed:
            stepper._quotient_bulk_groups = saved_groups
            stepper._quotient_meta = saved_meta
            raise QuotientSwitchError(
                f"quotient switch REJECTED at step {step}: failing gates "
                f"{[k for k, v in gates.items() if not v]}", record)
        result_drop.quotient_switch = record
        result_drop.quotient_state = stepper.export_quotient_state()
        print(f"== QUOTIENT COMMITTED at step {step}: raw bulks "
              f"{(ba, bb)} (loops {record['loop_pair']}) ==", flush=True)
        return result_drop

    def _arm_telemetry(self, stepper, source, result, committed, P_src,
                       V_src, ctx, labels_src, h_wall):
        P_new = stepper.sharp_perimeter(committed)
        W = float(result.wasserstein)
        tel = dict(
            objective=float(result.objective),
            objective_initial=float(result.objective_initial),
            perimeter_frozen=float(result.perimeter), wasserstein=W,
            P_sharp_source=P_src, P_sharp_committed=P_new,
            split_residual=P_new + W - P_src,
            n_iter=int(result.n_iter), converged=bool(result.converged),
            max_s_over_h=float(result.displacements.abs().max()) / h_wall,
            max_dtheta=float(result.delta_angles.abs().max()),
            contact_rows=result.contact_rows_telemetry,
            bulk_flux_residuals=result.bulk_flux_residuals,
            bulk_flux_residuals_shadow=result.bulk_flux_residuals_shadow,
        )
        try:
            certs, labels_c, m_c = stepper.contact_certificates_on_state(
                committed, ctx, labels_src)
            V_new = self._geometric_merged_volume(stepper, committed,
                                                  labels_c, m_c)
            tel["V_merged_geom"] = V_new
            tel["V_merged_rel"] = (V_new - V_src) / V_src
            tel["certificate"] = (certs[0].as_dict() if certs else None)
            tel["n_candidates"] = len(certs)
        except Exception as ex:          # noqa: BLE001 -- recorded
            tel["V_merged_geom"] = None
            tel["V_merged_rel"] = float("inf")
            tel["certificate"] = None
            tel["certificate_error"] = f"{type(ex).__name__}: {ex}"
        return tel

    def solve(
        self,
        initial_varifold: OrientedPointCloudVarifold,
        n_steps: int,
        callback: Callable[[int, MMStepResult], bool] | None = None,
    ) -> EvolutionHistory:
        """Run MM scheme for multiple steps.

        Args:
            initial_varifold: Initial boundary configuration.
            n_steps: Number of time steps.
            callback: Optional callback(step, result). Return True to stop.

        Returns:
            EvolutionHistory containing all states and diagnostics.
        """
        device = initial_varifold.positions.device
        dtype = initial_varifold.positions.dtype

        # Create history
        history = create_history(n_steps, device, dtype)

        # Store initial state
        history._varifolds.append(OrientedPointCloudVarifold(
            positions=initial_varifold.positions.detach(),
            angles=initial_varifold.angles.detach(),
        ))

        # Create stepper once, reuse across steps (enables torch.compile caching)
        # δ, τ are recomputed adaptively in stepper._setup_step() when config is None
        stepper = MMStepper(self.config)
        # 0L-B2: an irreversible quotient carried over a resume
        if getattr(self, "_pending_quotient_state", None):
            stepper.import_quotient_state(self._pending_quotient_state,
                                          initial_varifold.n_points)
        # L-A: a segmented/resumed run continues the ORIGINAL frozen
        # volume target (same import pattern as the quotient state);
        # unset -> stepper derives it from the initial cloud as before
        if getattr(self, "_pending_target_volume", None) is not None:
            stepper._grid_target_volume_initial = \
                self._pending_target_volume

        # Compute initial volume (divergence theorem: order-independent)
        cfg = self.config
        delta, tau = compute_recommended_params(
            initial_varifold.positions, kernel=cfg.mass_kernel, k_min=cfg.mass_k_min,
        )
        mass_delta = cfg.mass_delta if cfg.mass_delta is not None else delta
        mass_tau = cfg.mass_tau if cfg.mass_tau is not None else tau
        initial_masses = compute_masses(
            initial_varifold.positions, mass_delta, mass_tau, cfg.mass_kernel,
        )
        initial_volume = compute_volume_divergence(initial_varifold, initial_masses)
        history.volumes[0] = initial_volume

        current_varifold = initial_varifold
        self._stepper = stepper  # expose for diagnostics

        # 0E time-level recording (diagnostics ONLY: attaching a timeline
        # must leave the trajectory bitwise unchanged; see
        # src/torch/diagnostics/timeline.py). Set self.support_timeline
        # to a SupportTimeline before solve() to activate. All five
        # levels are emitted on EVERY step, identity records included.
        timeline = getattr(self, "support_timeline", None)

        def _tl(level, vf, step, pre_mm_scales=False, **kw):
            if timeline is None:
                return
            if pre_mm_scales:
                # scales of THIS level via the stepper's OWN resolver
                # (pure query: honors non-adaptive caching, per-point
                # bandwidth tensors, and mass_knn_k -- 0E-wiring.2)
                snap = stepper.resolve_mass_scales(vf)
                d, tau_ = snap.delta_summary, snap.tau_summary
                d_kde, t_kde = snap.delta_for_kde, snap.tau_for_kde
                eps = None
                if cfg.perimeter_sigma is not None:
                    sig = cfg.perimeter_sigma
                else:
                    from src.torch.perimeter.coherence_perimeter import (
                        compute_recommended_sigma,
                    )
                    sig = compute_recommended_sigma(vf.positions,
                                                    cfg.perimeter_c_sigma)
            else:
                d, tau_ = stepper.mass_delta, stepper.mass_tau
                d_kde = stepper._mass_delta_for_kde
                t_kde = stepper._mass_tau_for_kde
                sig = stepper._sigma if stepper._sigma is not None else 0.1
                eps = stepper.bem_wasserstein._epsilon_used
            return timeline.record(
                step, level, vf, delta=d, sigma=sig, eps_bem=eps,
                mass_tau=tau_, mass_kernel=cfg.mass_kernel,
                coherence_kernel=cfg.perimeter_kernel,
                coherence_backend=cfg.backend,
                delta_for_kde=d_kde, tau_for_kde=t_kde, **kw)

        # P2.1: a production safety flag must fail closed BEFORE
        # stepping, not with an AttributeError mid-run
        if self.config.enforce_support_gates \
                and getattr(self, "support_timeline", None) is None:
            raise ValueError(
                "enforce_support_gates=True requires an attached "
                "SupportTimeline (solver.support_timeline) -- the gate "
                "layer consumes its records and historical loops")

        # D2b-2.1: fail-closed schedule validation (an unknown string
        # must never silently run as the adaptive trigger)
        import math as _math
        _sched = self.config.redistribution_schedule
        if _sched not in ("every_step", "fixed_physical_interval",
                          "adaptive_cv_trigger"):
            raise ValueError(
                f"unknown redistribution_schedule {_sched!r}")
        if self.config.redistribute:
            if _sched == "fixed_physical_interval":
                iv = self.config.redistribute_physical_interval
                if not _math.isfinite(iv) or iv <= 0:
                    raise ValueError(
                        "fixed_physical_interval needs a finite "
                        f"positive redistribute_physical_interval; got "
                        f"{iv}")
            if _sched == "adaptive_cv_trigger":
                th = self.config.redistribute_trigger_cv
                if not _math.isfinite(th) or th < 0:
                    raise ValueError(
                        "adaptive_cv_trigger needs a finite nonnegative"
                        f" redistribute_trigger_cv; got {th}")

        # Evolution loop
        _next_redist_time = self.config.redistribute_physical_interval
        _last_redist_step = None
        for step in range(n_steps):
            try:
                _tl("pre_mm", current_varifold, step, pre_mm_scales=True,
                    committed=True, geometry_changed=False)
                result = self._mm_with_shadow_switch(
                    stepper, current_varifold, step)
            except Exception as e:
                print(f"    [ERROR] Exception at step {step + 1}: {e}")
                history.stop_step = step
                history.stop_stage = "mm_step"
                history.stop_exception_type = type(e).__name__
                history.stop_message = str(e)
                history.stop_details = getattr(e, "details", None)
                break

            dec = getattr(stepper, "_last_rank_decision", None)
            _tl("post_mm_candidate", result.varifold, step,
                displacements=result.displacements,
                committed=False,
                m_used=result.masses,
                q_used=stepper.fixed_coherence,
                rank_fields=dict(
                    detected_rank=result.detected_rank,
                    rank_status=result.rank_status,
                    rank_action=result.rank_action,
                    rank_pending=result.rank_pending,
                    rank_c_abs=result.rank_c_abs,
                    rank_c_gap=result.rank_c_gap,
                    rank_evidence_gap_ratio=result.rank_evidence_gap_ratio,
                    rank_gap_ratio=result.rank_gap_ratio,
                    rank_abs_level=result.rank_abs_level,
                    rank_spectrum_head=result.rank_spectrum_head,
                    subspace_angle_deg=result.subspace_angle_deg,
                    align_const_deg=(dec.align_const_deg
                                     if dec is not None else None),
                    align_held_deg=(getattr(dec, "align_held_deg", None)
                                    if dec is not None else None),
                    contain_deg=(dec.contain_deg
                                 if dec is not None else None),
                    r_comp=result.r_comp,
                    objective=result.objective,
                    objective_initial=result.objective_initial,
                    frozen_perimeter_initial=result.frozen_perimeter_initial,
                    perimeter=result.perimeter,
                    wasserstein=result.wasserstein,
                    relative_gradient_norm=result.relative_gradient_norm,
                    gradient_norm_final=result.gradient_norm_final,
                    step_norm=result.step_norm,
                    objective_decreased=result.objective_decreased,
                    optimizer_message=result.optimizer_message,
                    converged=result.converged,
                    n_iter=result.n_iter))
            current_varifold = result.varifold

            # Arm V2: commit the amplitudes the COUPLED step chose.
            # Applied at the MM commit point (pre-redistribution /
            # pre-deletion) so the existing survivor bookkeeping
            # carries them exactly like any other per-mark state; the
            # GC in the final stage collects the exact zeros.
            if (getattr(self.config, "grid_vertical_dof", False)
                    and result.vertical_da is not None):
                amps_v = stepper.carrier_amplitudes
                if (amps_v is None or amps_v.shape[0]
                        != current_varifold.n_points):
                    amps_v = torch.ones(
                        current_varifold.n_points,
                        dtype=current_varifold.positions.dtype,
                        device=current_varifold.positions.device)
                amps_v = amps_v.clone()
                amps_v[result.vertical_idx] = (
                    amps_v[result.vertical_idx]
                    + result.vertical_da).clamp_min(0.0)
                stepper.carrier_amplitudes = amps_v
                result.amplitude_stats = dict(
                    result.vertical_stats or {})

            # Arm 4: passive advection of low-coherence marks by the
            # committed step's own transport field (grid backend only,
            # default off). Runs BEFORE redistribution: the tangential
            # redistribution update scales with q and therefore never
            # touches the marks this stage moves.
            if (self.config.grid_fossil_advection
                    and getattr(stepper, "grid_wasserstein", None)
                    is not None):
                from src.torch.oriented_varifold.mass import (
                    compute_masses as _cm_adv,
                )
                from src.torch.transport import compute_coherence \
                    as _cq_adv
                from src.torch.transport.fossil_advection import (
                    FossilAdvectionConfig,
                    advect_fossils,
                )
                m_adv = _cm_adv(current_varifold.positions,
                                stepper._mass_delta_for_kde,
                                stepper._mass_tau_for_kde,
                                self.config.mass_kernel)
                q_adv = _cq_adv(current_varifold, m_adv,
                                stepper._sigma,
                                self.config.perimeter_kernel,
                                backend=self.config.backend)
                adv_cfg = (self.config.grid_advection_config
                           or FossilAdvectionConfig())
                current_varifold, adv_stats = advect_fossils(
                    current_varifold, q_adv,
                    result.displacements, result.delta_angles,
                    stepper.grid_wasserstein, adv_cfg)
                result.advection_stats = adv_stats
                if adv_stats["skipped_reason"] is not None \
                        and not getattr(self, "_adv_skip_seen", False):
                    self._adv_skip_seen = True
                    print(f"    [advection-skip] step {step + 1}: "
                          f"{adv_stats['skipped_reason']}", flush=True)

            # Redistribute points if enabled (D2b-2: three schedules;
            # "every_step" reproduces the legacy interval logic bitwise)
            redist_trigger_cv = None
            if not self.config.redistribute:
                redist_ran = False
            elif self.config.redistribution_schedule == "every_step":
                redist_ran = ((step + 1)
                              % self.config.redistribute_interval == 0)
            elif self.config.redistribution_schedule \
                    == "fixed_physical_interval":
                phys_t = (step + 1) * self.config.time_step
                redist_ran = phys_t >= _next_redist_time - 1e-15
                # one step may cross SEVERAL interval marks (dt >
                # interval, or noncommensurate dt): advance past ALL of
                # them -- the operator still runs once per step (review)
                while phys_t >= _next_redist_time - 1e-15:
                    _next_redist_time += \
                        self.config.redistribute_physical_interval
            else:   # adaptive_cv_trigger (permutation-invariant metric)
                from .redistribute import _compute_density_log_gradient
                _, theta = _compute_density_log_gradient(
                    current_varifold.positions.detach(),
                    stepper.mass_delta
                    * self.config.redistribute_delta_ratio,
                    self.config.mass_kernel)
                redist_trigger_cv = (theta.std()
                                     / theta.mean().clamp_min(1e-30)
                                     ).item()
                redist_ran = (redist_trigger_cv
                              > self.config.redistribute_trigger_cv)
            pos_before_redist = current_varifold.positions
            ang_before_redist = current_varifold.angles
            # gap to the PREVIOUS invocation, captured BEFORE updating
            # (review closure patch 5: the old order made the firing
            # record always report 0)
            _steps_since = (step - _last_redist_step
                            if _last_redist_step is not None else None)
            redist_info = None
            if redist_ran:
                _last_redist_step = step
                # D2b-1: ALL rules route through the shared helper (its
                # legacy_hybrid branch reproduces redistribute_points
                # bitwise -- pinned; the D0 goldens verify this stayed
                # exact on the committed trajectories)
                new_pos, new_angles, redist_info = self._redistribute(
                    stepper, current_varifold,
                    self.config.redistribute_n_iters)
                current_varifold = OrientedPointCloudVarifold(
                    positions=new_pos, angles=new_angles,
                )
            # used-quantity contract (review, D2b-1 correction 5): the
            # legacy rule consumes the pre-MM tensors; frozen recomputes
            # once and substep per subiteration -- storing the legacy
            # tensors for those rules would be false, and no single
            # tensor represents substep's q. Non-legacy rules store None
            # plus the policy payload; the subiteration trace is the
            # faithful record.
            _rd_legacy = self.config.redistribution_rule == "legacy_hybrid"
            _tl("post_redistribution", current_varifold, step,
                stage_enabled=self.config.redistribute,
                stage_executed=redist_ran, committed=False,
                m_used=(result.masses if _rd_legacy else None),
                q_used=(stepper.fixed_coherence if _rd_legacy else None),
                rank_fields=(dict(
                    redistribution_rule=self.config.redistribution_rule,
                    schedule=self.config.redistribution_schedule,
                    trigger_value=redist_trigger_cv,
                    trigger_threshold=(
                        self.config.redistribute_trigger_cv
                        if self.config.redistribution_schedule
                        == "adaptive_cv_trigger" else None),
                    trigger_reason=(
                        ("cv_above_threshold" if redist_ran
                         else "cv_below_threshold")
                        if self.config.redistribution_schedule
                        == "adaptive_cv_trigger" else
                        self.config.redistribution_schedule),
                    steps_since_last_invocation=_steps_since,
                    q_policy=("caller stale pre-MM q" if _rd_legacy else
                              ("recomputed once on input"
                               if self.config.redistribution_rule
                               == "post_mm_frozen"
                               else "recomputed per subiteration")),
                    m_policy=("m(input), raw"
                              if self.config.redistribution_rule
                              != "substep_refreshed"
                              else "m(current) per subiteration"),
                    coherence_source=(redist_info["trace"][0]
                                      .get("coherence_source")
                                      if redist_info and
                                      redist_info["trace"] else None),
                    n_subiters=redist_info["n_iters"],
                    clip_active_iters=sum(
                        # monotone early-exit rows (stage budget
                        # exhausted / stalled) carry no clip field
                        t.get("clip_active", 0)
                        for t in redist_info["trace"]))
                    if redist_info else None),
                geometry_changed=bool(redist_ran and (
                    (current_varifold.positions
                     - pos_before_redist).abs().max() > 1e-14
                    or _circ_diff(current_varifold.angles,
                                  ang_before_redist).max() > 1e-14)))

            # Remove dead points if enabled
            del_enabled = self.config.remove_dead_points
            del_ran = (del_enabled and
                       (step + 1) % self.config.dead_point_interval == 0)
            n_removed = 0
            post_removal_redist = False
            if del_ran:
                # D1b: ALL rules route through the shared helper (its
                # legacy_lagged branch reproduces the historical
                # remove_dead_points semantics bitwise -- same >=, same
                # rho*max threshold, same degenerate keep-all; the D0
                # goldens verify this stayed exact)
                dec = evaluate_dead_points(
                    current_varifold, result.masses,
                    stepper.fixed_coherence,
                    self.config.dead_point_rule,
                    self.config.dead_point_threshold,
                    delta_for_kde=stepper._mass_delta_for_kde,
                    tau_for_kde=stepper._mass_tau_for_kde,
                    mass_kernel=self.config.mass_kernel,
                    sigma=stepper._sigma,
                    coherence_kernel=self.config.perimeter_kernel,
                    coherence_backend=self.config.backend,
                )
                keep_mask = dec.keep_mask
                if keep_mask.all():
                    new_varifold = current_varifold
                else:
                    new_varifold = OrientedPointCloudVarifold(
                        positions=current_varifold.positions[keep_mask],
                        angles=current_varifold.angles[keep_mask],
                    )
                n_removed = current_varifold.n_points - new_varifold.n_points
                # used quantities: the FULL pre-deletion tensors the RULE
                # actually consumed (dec.m_used/q_used -- for legacy these
                # are the pre-MM tensors bitwise, for semi/refreshed the
                # recomputed ones; storing the legacy tensors for every
                # rule violated the 0E used-quantity contract); keep_mask
                # maps them onto the surviving points
                _tl("post_deletion_raw",
                    new_varifold if n_removed > 0 else current_varifold,
                    step, keep_mask=keep_mask,
                    stage_enabled=True, stage_executed=True,
                    committed=False,
                    m_used=dec.m_used, q_used=dec.q_used,
                    geometry_changed=n_removed > 0, n_removed=n_removed,
                    rank_fields=dict(
                        dead_point_rule=dec.rule,
                        dead_point_threshold=dec.threshold,
                        dead_point_score=dec.score.detach().clone(),
                        dead_point_margin=(
                            dec.threshold_margin.detach().clone()),
                    ))
                post_del_raw_positions = (
                    new_varifold if n_removed > 0
                    else current_varifold).positions
                post_del_raw_angles = (
                    new_varifold if n_removed > 0
                    else current_varifold).angles
                if n_removed > 0:
                    print(f"    [REMOVE] step {step+1}: removed {n_removed} dead points, "
                          f"N={current_varifold.n_points}→{new_varifold.n_points}")
                    current_varifold = new_varifold

                    # Extra redistribute after removal to fix density holes
                    if (self.config.redistribute and
                        self.config.redistribute_n_iters_after_removal > 0):
                        post_removal_redist = True
                        # Slice coherence to match surviving points
                        post_coherence = (stepper.fixed_coherence[keep_mask]
                                          if stepper.fixed_coherence is not None
                                          else None)
                        new_pos, new_angles, rd_info = self._redistribute(
                            stepper, current_varifold,
                            self.config.redistribute_n_iters_after_removal,
                            stale_q=post_coherence,
                        )
                        current_varifold = OrientedPointCloudVarifold(
                            positions=new_pos, angles=new_angles,
                        )
                        cv_hist = rd_info["cv_history"]
                        print(f"    [REDIST] post-removal: CV {cv_hist[0]:.4f}→{cv_hist[-1]:.4f} "
                              f"({rd_info['n_iters']} iters, conv={rd_info['converged']})")
            if not del_ran:
                post_del_raw_positions = current_varifold.positions
                post_del_raw_angles = current_varifold.angles
                _tl("post_deletion_raw", current_varifold, step,
                    stage_enabled=del_enabled, stage_executed=False,
                    committed=False, geometry_changed=False,
                    m_used=result.masses,
                    q_used=stepper.fixed_coherence)

            # Phase-support projection (review option B): label-free
            # interior-fossil removal read from the reconstructed
            # phase; grid backend only, flag-gated, eligibility from
            # the CURRENT state (stably single-component). Shadow-
            # verified inside the module; a rejected verification is a
            # HARD stop (never tolerate the fossil via gate
            # exceptions). Timeline note: runs between the recorded
            # deletion stage and the final record of this step;
            # promotion to a first-class timeline stage is pending.
            _psp_record = None
            if (self.config.grid_phase_support_projection
                    and getattr(stepper, "grid_wasserstein", None)
                    is not None):
                from src.torch.oriented_varifold.mass import (
                    compute_masses as _cm,
                )
                from src.torch.transport.phase_support_projection import (
                    phase_support_projection,
                )
                gm = stepper.grid_wasserstein
                # v10: the window flag flickers on the fossil's
                # own crumb noise -- it must not veto the projection
                # (the module re-checks operating/3thr levels itself)
                eligible = (
                    getattr(gm, "n_compat_components", None) == 1)
                if eligible:
                    cfgs = self.config

                    def _mass_fn(vf):
                        return _cm(vf.positions,
                                   stepper._mass_delta_for_kde,
                                   stepper._mass_tau_for_kde,
                                   cfgs.mass_kernel)

                    def _perim_fn(vf, mm):
                        from src.torch.perimeter.coherence_perimeter \
                            import compute_perimeter_coherence
                        return compute_perimeter_coherence(
                            vf, mm, stepper._sigma,
                            cfgs.perimeter_kernel,
                            backend=cfgs.backend)

                    from src.torch.transport import compute_coherence
                    m_cur = _mass_fn(current_varifold)
                    q_cur = compute_coherence(
                        current_varifold, m_cur, stepper._sigma,
                        cfgs.perimeter_kernel, backend=cfgs.backend)
                    target = (stepper._grid_target_volume_initial
                              if cfgs.grid_volume_target_mode
                              == "initial_fixed" else None)
                    if target is None:
                        target = float(
                            0.5 * (m_cur
                                   * (current_varifold.positions
                                      * current_varifold.normals)
                                   .sum(-1)).sum())
                    r_local = max(
                        cfgs.grid_metric.phase.fill_epsilon,
                        stepper._sigma or 0.0,
                        float(stepper.mass_delta))
                    psp_kwargs = {}
                    if cfgs.grid_psp_config is not None:
                        psp_kwargs["psp"] = cfgs.grid_psp_config
                    psp_out = phase_support_projection(
                        current_varifold, m_cur,
                        q_cur, cfgs.grid_metric.phase,
                        target, r_local,
                        perimeter=_perim_fn, mass_fn=_mass_fn,
                        **psp_kwargs)
                    _psp_record = psp_out
                    if (psp_out.n_candidates == 0
                            and psp_out.reason != "no candidates"
                            and not getattr(self, "_psp_skip_seen",
                                            False)):
                        self._psp_skip_seen = True
                        print(f"    [PSP-skip] step {step + 1}: "
                              f"{psp_out.reason}", flush=True)
                    if psp_out.n_candidates > 0:
                        if not psp_out.accepted:
                            # v20 semantics: a rejected round means
                            # "not now" -- no removal is committed and
                            # the run continues under the ordinary
                            # gates (measured: round 1 accepted, round
                            # 2 transiently rejected on the quality
                            # slack alone while every invariance
                            # criterion passed; mid-healing the state
                            # needs steps to relax between chunks).
                            # This is NOT tolerating a bad deletion --
                            # nothing is committed -- and not a gate
                            # exception: if the remaining fossil
                            # breaks a gate, the run still fails
                            # closed there.
                            print(f"    [PSP-reject] step {step + 1}:"
                                  f" no removal ({psp_out.reason})",
                                  flush=True)
                        if psp_out.accepted:
                            kept = psp_out.keep_mask
                            n_proj = int((~kept).sum())
                            current_varifold = OrientedPointCloudVarifold(
                                positions=current_varifold
                                .positions[kept],
                                angles=current_varifold.angles[kept])
                            # keep the observer timeline's particle ids in
                            # sync (same remap the deletion stage uses)
                            _tl_obj = getattr(self, "support_timeline",
                                              None)
                            if _tl_obj is not None:
                                _tl_obj.apply_keep_mask(kept)
                            print(
                                f"    [PSP] step {step + 1}: projected "
                                f"out {n_proj} interior fossil mark(s) "
                                f"at ({psp_out.removed_centroid[0]:.3f},"
                                f"{psp_out.removed_centroid[1]:.3f}); "
                                f"D_rho={psp_out.D_rho:.2e} "
                                f"D_J={psp_out.D_J:.2e} "
                                f"vol={psp_out.vol_drift:.2e}",
                                flush=True)
            # stage fields of the FINAL record refer to the post-removal
            # redistribution stage (deletion itself is post_deletion_raw's
            # stage -- mixing the two would bake ambiguous semantics into
            # the D0 golden JSON)
            post_redist_enabled = bool(
                del_ran and n_removed > 0 and self.config.redistribute
                and self.config.redistribute_n_iters_after_removal > 0)
            # Arm 5/6: monotone carrier-amplitude fade + garbage
            # collection (grid backend only, default off). Runs LAST
            # among the state-changing stages (after redistribution
            # and deletion, PSP-style) because the GC changes N: any
            # earlier and the redistribution's per-step caches (sized
            # at the MM step's N) mismatch -- the A7 step-984 crash.
            # Updated on the committed post-redistribution state
            # against the pre-step frozen phase; the new amplitudes
            # attenuate the NEXT step's grid carrier masses.
            if (self.config.grid_carrier_amplitude_fade
                    and getattr(stepper, "grid_wasserstein", None)
                    is not None
                    and stepper.grid_wasserstein.phase is not None):
                from src.torch.transport.carrier_amplitude import (
                    CarrierAmplitudeConfig,
                    update_amplitudes,
                )
                amp_cfg = (self.config.grid_amplitude_config
                           or CarrierAmplitudeConfig())
                amps = stepper.carrier_amplitudes
                if (amps is None
                        or amps.shape[0] != current_varifold.n_points):
                    amps = torch.ones(
                        current_varifold.n_points,
                        dtype=current_varifold.positions.dtype,
                        device=current_varifold.positions.device)
                if getattr(self.config, "grid_vertical_dof", False):
                    # arm V2: the coupled step already chose and
                    # committed da; this stage only garbage-collects
                    # the exact zeros below.
                    amp_stats = dict(result.amplitude_stats or {})
                else:
                    from src.torch.oriented_varifold.mass import (
                        compute_masses as _cm_amp,
                    )
                    from src.torch.transport import compute_coherence \
                        as _cq_amp
                    m_amp = _cm_amp(current_varifold.positions,
                                    stepper._mass_delta_for_kde,
                                    stepper._mass_tau_for_kde,
                                    self.config.mass_kernel)
                    q_amp = _cq_amp(current_varifold, m_amp,
                                    stepper._sigma,
                                    self.config.perimeter_kernel,
                                    backend=self.config.backend)
                    if getattr(amp_cfg, "variational", False):
                        # arm V1 (spec-2.0 full prototype, split
                        # scheme): one box-constrained QP replaces
                        # rate/closed-loop/allocation; kept for
                        # ablation (the coupled V2 path supersedes it).
                        from src.torch.transport.carrier_amplitude \
                            import variational_fade
                        amps, amp_stats = variational_fade(
                            current_varifold, amps, m_amp,
                            stepper.grid_wasserstein.poisson,
                            stepper.grid_wasserstein.phase.rho_metric,
                            self.config.grid_metric.phase, amp_cfg,
                            coherence=q_amp,
                            time_step=self.config.time_step)
                    else:
                        # arm 8: closed-loop rate -- the fade runs
                        # greedily under a fixed fraction of the
                        # existing certified projection budget
                        # (measured every step); a fixed rate was a
                        # two-sided gamble (0.05 lost to the front,
                        # 0.25 to the water-filling transient)
                        p_now = (result.grid_setup_snapshot
                                 .projection_l2_relative
                                 if result.grid_setup_snapshot
                                 is not None else 0.0)
                        p_star = (getattr(amp_cfg, "proj_budget_frac",
                                          0.4)
                                  * self.config.grid_metric.phase
                                  .projection_rel_tol)
                        rate_scale = max(
                            0.0, min(1.0, (p_star - p_now) / p_star))
                        amps, amp_stats = update_amplitudes(
                            current_varifold, amps,
                            stepper.grid_wasserstein.phase.rho_metric,
                            self.config.grid_metric.phase, amp_cfg,
                            coherence=q_amp, rate_scale=rate_scale)
                # arm 6 (spec-2.0 garbage collection): a mark at
                # a = 0 contributes exactly nothing to the grid
                # layer; collecting it here is the certified end of
                # the uniform fade. It additionally heals the
                # coherence channel (the ghost stops recruiting
                # neighbors into the frozen wall) and removes the
                # near-null optimizer DOF of its co-moving pair.
                amp_stats["n_collected"] = 0
                if getattr(amp_cfg, "gc", True):
                    keep = amps > 0.0
                    if not bool(keep.all()):
                        from src.torch.oriented_varifold import (
                            OrientedPointCloudVarifold as _OPV,
                        )
                        current_varifold = _OPV(
                            positions=current_varifold
                            .positions[keep],
                            angles=current_varifold.angles[keep])
                        amps = amps[keep]
                        amp_stats["n_collected"] = int((~keep).sum())
                        tl = getattr(self, "support_timeline", None)
                        if tl is not None:
                            tl.apply_keep_mask(keep)
                stepper.carrier_amplitudes = amps
                result.amplitude_stats = amp_stats

            final_rec = _tl(
                "post_deletion_final", current_varifold, step,
                stage_enabled=post_redist_enabled,
                stage_executed=post_removal_redist,
                committed=True,
                geometry_changed=bool(post_removal_redist and (
                    (current_varifold.positions
                     - post_del_raw_positions).abs().max() > 1e-14
                    or _circ_diff(current_varifold.angles,
                                  post_del_raw_angles).max() > 1e-14)))

            # P2 (production integration): the solver CONSUMES the
            # sticky support gates. Default OFF -- legacy trajectories
            # are untouched. When on: whole-loop extinction events at
            # this step are certified inline from the timeline's own
            # records (the phase rank is the caller's claim from the
            # rank machinery; hole extinction leaves it unchanged) and
            # the run continues on success; any other closed gate
            # (failed certification, sticky invalid support, remaining
            # pending topology event) stops CONTINUATION after the
            # committed step, with a structured stop reason.
            if self.config.enforce_support_gates \
                    and self.support_timeline is not None:
                tl_ = self.support_timeline
                gate_stop = None
                # P2.1 rank semantics: passing the same value as before
                # and after is NOT a measurement -- it is the
                # EVENT-CLASS ASSUMPTION that a whole negative hole
                # loop's extinction leaves the phase rank unchanged
                # (spectral reinspection of the post-deletion support
                # is the future upgrade). Without any rank evidence at
                # all the certificate must NOT pass: fail closed.
                rank_now = getattr(
                    getattr(stepper, "_last_rank_decision", None),
                    "rank", None)
                if rank_now is None:
                    rank_now = self.config.bem_component_rank
                for ev in [e for e in tl_.pending_topology_events
                           if e["step"] == step
                           and e["kind"] == "loop_extinction"]:
                    if rank_now is None:
                        gate_stop = (
                            "loop extinction with NO rank evidence "
                            "(no rank machinery and no oracle rank): "
                            "cannot assert the event-class rank "
                            "assumption -- fail closed")
                        break
                    try:
                        tl_.certify_and_accept_loop_extinction(
                            step, phase_rank_before=rank_now,
                            phase_rank_after=rank_now)
                    except ValueError as e:
                        gate_stop = ("loop extinction failed to "
                                     f"certify: {e}")
                if gate_stop is None:
                    perms = (tl_.gated_permissions(final_rec.report)
                             if final_rec is not None
                             and final_rec.report is not None
                             else None)
                    if perms is None:
                        gate_stop = ("no support report on the "
                                     "committed record (fail closed)")
                    elif not perms["allow_same_rank_evolution"]:
                        gate_stop = (
                            "support gate closed: "
                            f"state={final_rec.report.state}, "
                            f"sticky={tl_.requires_reconstruction}, "
                            f"pending={tl_.pending_topology_events}")
                if gate_stop is not None:
                    print(f"    [GATE] step {step + 1}: {gate_stop}")
                    history.stop_step = step
                    history.stop_stage = "support_gate"
                    history.stop_exception_type = "SupportGateClosed"
                    history.stop_message = gate_stop
                    # the terminal frame is an observation, not a
                    # usable MM state: the last valid state is the
                    # previous committed step
                    history.terminal_candidate_valid = False
                    history.last_valid_step = step - 1
                    _support_gate_stop = True
                else:
                    _support_gate_stop = False
            else:
                _support_gate_stop = False

            # Check for NaN
            if (result.perimeter is None or
                torch.isnan(torch.tensor(result.perimeter)) or
                torch.isnan(current_varifold.positions).any()):
                print(f"    [WARNING] NaN at step {step + 1}, stopping")
                break

            # Store in history
            history._varifolds.append(OrientedPointCloudVarifold(
                positions=current_varifold.positions.detach(),
                angles=current_varifold.angles.detach(),
            ))
            history.perimeters[step] = result.perimeter
            history.wassersteins[step] = result.wasserstein
            history.objectives[step] = result.objective
            history.converged[step] = result.converged
            history.n_completed = step + 1

            # Compute current volume (divergence theorem)
            # After dead-point removal, current_varifold may have fewer points
            # than result.masses, so recompute masses from current positions.
            if current_varifold.n_points != len(result.masses):
                vol_masses = compute_masses(
                    current_varifold.positions,
                    stepper.mass_delta, stepper.mass_tau,
                    self.config.mass_kernel,
                )
            else:
                vol_masses = result.masses
            current_volume = compute_volume_divergence(current_varifold, vol_masses)
            history.volumes[step + 1] = current_volume

            # P2: a closed support gate stops CONTINUATION after the
            # committed step (the step itself stays in the history)
            if _support_gate_stop:
                break

            # Callback -- with the COMMITTED state attached (review
            # blocker 4): result.varifold is the MM minimizer; the
            # state the next step consumes (after redistribution,
            # deletion and the phase-support projection) is
            # current_varifold. Checkpoints must read
            # result.committed_varifold, or a resume would resurrect
            # projected-out fossils.
            result.committed_varifold = current_varifold
            # 0L-B2 / 0N-4 State III hard invariant: after an
            # irreversible quotient the CONTACT REFERENCE (pair,
            # identification scale h_*, sigma_* and thresholds frozen
            # at the switch) must keep holding on EVERY committed state
            # (fail-closed; C_- is never revived). The pair is NOT
            # re-searched: re-applying the moving-scale B1 entry
            # certificate to the shrinking fixed-N ghost representation
            # produced the 0N-3 false positive. Live-scale geometry is
            # shadow telemetry inside the reference metrics.
            if (self.config.grid_quotient_mode != "off"
                    and stepper._quotient_bulk_groups
                    and stepper._quotient_meta.get("quotient_at")
                    is not None):
                from src.torch.transport.contact_certificate import (
                    ContactCertificateError,
                )
                try:
                    ref = stepper._quotient_meta.get("contact_reference")
                    if ref is None:
                        raise QuotientInvariantError(
                            "committed quotient without a contact "
                            "reference: the State III invariant cannot "
                            f"be evaluated at step {step} (fail-closed)",
                            {"step": int(step)})
                    if ref.get("particles_a_mask") is None:
                        # legacy (pre-0N-4) checkpoint: resolve the
                        # pair's particle partition ONCE, uniquely
                        # identified by the SAVED raw bulk pair (never
                        # "the first candidate"; certified is NOT
                        # required -- the moving-h certificate can
                        # already be false here)
                        if (result.contact_pair_context is None
                                or stepper._loop_labels_last is None):
                            raise QuotientInvariantError(
                                "legacy contact_reference cannot be "
                                "resolved: no contact pair context on "
                                f"the committed state at step {step}",
                                {"step": int(step)})
                        certs_c, labels_c, _ = \
                            stepper.contact_certificates_on_state(
                                current_varifold,
                                result.contact_pair_context,
                                stepper._loop_labels_last)
                        want = set(ref["raw_bulk_pair"])
                        match = [c for c in certs_c
                                 if not c.candidate.ambiguous
                                 and set(int(x) for x in
                                         c.candidate.bulk_pair) == want]
                        if len(match) != 1:
                            raise QuotientInvariantError(
                                f"legacy contact_reference resolution "
                                f"needs exactly one candidate with bulk "
                                f"pair {sorted(want)}, found "
                                f"{len(match)} at step {step}",
                                {"step": int(step), "candidates": [
                                    c.as_dict() for c in certs_c]})
                        la, lb = match[0].candidate.loop_pair
                        ref["particles_a_mask"] = (
                            labels_c == int(la)).detach().cpu().clone()
                        ref["particles_b_mask"] = (
                            labels_c == int(lb)).detach().cpu().clone()
                        ref["loop_pair_at_switch"] = (int(la), int(lb))
                        # refresh the snapshot taken inside step() so a
                        # checkpoint of THIS step already carries the
                        # resolved reference
                        result.quotient_state = \
                            stepper.export_quotient_state()
                        print(f"    [0N-4] legacy contact_reference "
                              f"resolved at step {step}: loops "
                              f"({int(la)}, {int(lb)}), h_* "
                              f"{ref['h_ab_at_switch']:.6f}")
                    cert_ref = stepper.contact_reference_on_state(
                        current_varifold)
                    if cert_ref is not None:
                        result.contact_certificate_committed = [
                            cert_ref.as_dict()]
                        if not cert_ref.certified:
                            raise QuotientInvariantError(
                                f"State III contact reference FAILED at "
                                f"step {step} for bulks "
                                f"{cert_ref.candidate.bulk_pair}: "
                                f"{cert_ref.failing}",
                                {"certificate": cert_ref.as_dict(),
                                 "step": int(step)})
                except ContactCertificateError as e:
                    e = QuotientInvariantError(
                        f"State III contact reference guard at step "
                        f"{step}: {e}", {"step": int(step)})
                    print(f"    [ERROR] {e}")
                    history.stop_step = step
                    history.stop_stage = "quotient_invariant"
                    history.stop_exception_type = type(e).__name__
                    history.stop_message = str(e)
                    history.stop_details = e.details
                    if callback is not None:
                        callback(step, result)
                    break
                except QuotientInvariantError as e:
                    print(f"    [ERROR] {e}")
                    history.stop_step = step
                    history.stop_stage = "quotient_invariant"
                    history.stop_exception_type = type(e).__name__
                    history.stop_message = str(e)
                    history.stop_details = e.details
                    if callback is not None:
                        callback(step, result)
                    break
            if _psp_record is not None:
                po = _psp_record
                result.phase_support_projection_outcome = dict(
                    n_candidates=po.n_candidates,
                    accepted=po.accepted,
                    n_before=po.n_before, n_after=po.n_after,
                    D_rho=po.D_rho, D_J=po.D_J, D_P=po.D_P,
                    D_V=po.D_V,
                    removed_indices=list(po.removed_indices),
                    removed_q=list(po.removed_q),
                    removed_centroid=list(po.removed_centroid),
                    reason=po.reason)
            if callback is not None:
                if callback(step, result):
                    break

        return history
