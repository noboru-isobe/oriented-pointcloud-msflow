"""Single step of the MM (Minimizing Movements) scheme.

Implements one step of the MM scheme for Mullins-Sekerka flow using
BEM-based linearized Wasserstein distance:

    μ^n = argmin_{y} [ P̂(μ) + W_lin(s, Δθ(s)) ]

where:
    - P̂ is the coherence-based perimeter
    - W_lin is the linearized Wasserstein via BEM (with endpoint collocation)
    - s = Q @ y (volume-conserving displacements)
    - Δθ is derived from s via weak form angle constraint with coherence suppression:
      A @ Δθ = B @ s  =>  Δθ_i = q_i × [A^† (B @ s)]_i
"""

import torch
from dataclasses import dataclass
from typing import Literal

from torchmin import minimize

from src.torch.oriented_varifold import OrientedPointCloudVarifold
from src.torch.oriented_varifold.mass import (
    compute_masses,
    compute_recommended_params,
    compute_per_point_bandwidths_knn,
    compute_per_point_bandwidths_abramson,
)
from src.torch.perimeter.angle_constraint import compute_angle_constraint_matrices
from src.torch.transport import BEMWasserstein, compute_coherence
from src.torch.math_utils.linalg import TruncatedSVD
from src.torch.math_utils.angles import wrap_angles


@dataclass
class MMConfig:
    """Configuration for MM scheme.

    Attributes:
        time_step: Time step h for MM scheme.
        perimeter_sigma: Bandwidth for coherence perimeter (None for auto).
        perimeter_kernel: Kernel for perimeter computation.
        perimeter_c_sigma: Multiplier for auto sigma computation.
        mass_delta: Bandwidth for KDE mass (None for auto).
        mass_tau: Cutoff threshold for mass (None for auto).
        mass_kernel: Kernel for mass computation.
        optimizer_method: Optimization method (bfgs recommended).
        optimizer_max_iter: Maximum optimizer iterations.
        optimizer_tol: Optimizer tolerance.
        optimizer_lr: Step size for BFGS/L-BFGS.
        optimizer_disp: Verbosity level (0 = silent).
        bem_method: BEM method ("point" or "panel").
        bem_epsilon_scale: Epsilon scale for point method.
        bem_n_endpoints: Number of collocation points per segment (1=center only, 3=endpoints).
        bem_trace_side: "interior" (correct) or "legacy_exterior" (reproduces
            published artifacts; temporary default).
    """
    # Time step. Default 1e-4 matches the production-tested scale: animate_flower.py
    # and run_two_ellipses_batch.py both pass 1e-5 explicitly for long runs, and 1e-4
    # gives near-production volume conservation in short integration tests.
    time_step: float = 1e-4

    # Perimeter parameters
    perimeter_sigma: float | None = None
    perimeter_kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2"
    perimeter_c_sigma: float = 3.0

    # Mass parameters
    mass_delta: float | None = None
    mass_tau: float | None = None
    mass_k_min: float = 1.0
    mass_kernel: Literal["wendland_c2", "biweight", "epanechnikov"] = "wendland_c2"

    # Optimizer parameters
    optimizer_method: str = "bfgs"
    optimizer_max_iter: int = 100
    optimizer_tol: float = 1e-6
    optimizer_lr: float = 0.5
    optimizer_disp: int = 0

    # BEM Wasserstein parameters
    bem_method: str = "point"       # "point" (recommended) or "panel"
    bem_epsilon_scale: float = 0.1  # only for "point"; must be > 0
    bem_n_endpoints: int = 3        # 1 = center only (no Δθ), 3 = endpoint collocation
    # Single-layer jump relation. "interior" is the correct trace for the
    # one-phase interior Neumann problem; "legacy_exterior" is what produced
    # every published artifact and stays the default until those are redone.
    # New experiments must request "interior" explicitly.
    bem_trace_side: str = "legacy_exterior"
    # 0A scale API. "sigma" (legacy default) couples eps_BEM to the coherence
    # bandwidth; "carrier_segment_length" is q-independent -- new benchmarks
    # should request it explicitly. "explicit" uses bem_epsilon verbatim.
    bem_epsilon_mode: str = "sigma"
    bem_epsilon: float | None = None
    # Bandwidth of the weak-form angle constraint. None falls back to the
    # perimeter sigma (legacy behaviour); refinement studies must pin it,
    # since otherwise varying perimeter_sigma silently changes the
    # normal-angle transport regularisation too.
    angle_sigma: float | None = None
    # 0C-5a: componentwise-compatible solver. "spectral_bordered" replaces
    # the single global volume constraint by the kernel of the spectral
    # compatibility matrix C = U0^T W^{1/2} G_end (rank = bem_component_rank,
    # oracle-supplied at this stage) and solves the bordered interior BIE
    # with incompatible-RHS rejection. Production default unchanged.
    bem_solver_mode: str = "legacy_global_projection"
    bem_component_rank: int | None = None
    # P1.2 rank initialization (review decision): an INTEGER rank
    # certificate for the first frame -- NOT component labels or mesh
    # connectivity; the representation stays an unordered point cloud
    # and no component surgery ever happens. Consumed only by
    # bem_rank_mode="adaptive_state_machine"; provenance is recorded
    # verbatim in the run metadata.
    initial_rank_certificate: int | None = None
    initial_rank_certificate_provenance: str = ""
    # C_u row-space continuation gate (the stronger condition the
    # AdaptiveRankMachine's own docstring requires before production
    # use at C >= 2): max principal angle between consecutive steps'
    # particle-space compatibility row spaces.
    cu_continuation_max_deg: float = 30.0
    # Speedup flags (default OFF: the default numeric path is
    # bit-identical; adoption gated on P1.2 + review):
    # Part 1 -- spectral W_lin via the hand-differentiated quadratic
    # Function (same forward sequence, closed-form backward, double-
    # backward-compatible for trust-ncg HVPs).
    wlin_analytic_gradient: bool = False
    # Part 2 -- near-null extraction: "full" torch SVD (baseline) or
    # warm-started Gram-free LOBPCG with parity/fallback protocol.
    bem_svd_method: Literal["full", "lobpcg"] = "full"
    # 0C-5b: "auto" reads C label-free from the weighted operator's spectrum
    # (two complementary criteria on the same spectrum, refuses on any
    # non-clean evidence); "oracle" uses bem_component_rank verbatim.
    # 0C-5d-1.1: "state_machine" runs the hysteretic RankStateMachine
    # inside the stepper -- same-rank weak evidence CONTINUES the evolution
    # at the held working rank; hard stops only on no_null_block,
    # failure to initialise, or reject_artificial.
    bem_rank_mode: str = "oracle"
    bem_rank_max: int = 4
    # Detection thresholds (exposed for sensitivity studies; every result
    # JSON must record the values actually used).
    bem_rank_null_tol: float = 0.05
    bem_rank_gap_min: float = 30.0
    # 0C-5c: which masses build the BEM operator (panel geometry,
    # quadrature, K*, rank detection). "visible" = q*m (legacy default),
    # "carrier" = raw m -- keeps the topology sensor q-independent. The
    # flux measure stays a separate choice (wlin_use_coherence_velocity).
    bem_operator_measure: str = "visible"

    # Angle constraint (weak form: A @ Δθ = B @ s)
    angle_constraint_rcond: float | None = None  # SVD truncation threshold (None = auto)

    # Coherence option for debugging
    use_unit_coherence: bool = False  # If True, use coherence=1 instead of computed value

    # W_lin velocity coherence scaling
    wlin_use_coherence_velocity: bool = True  # If False, V_{ik} = (1/h)(...) instead of (q_i/h)(...)

    # Displacement q suppression: s_i *= q_i, dθ_i *= q_i in MM step
    displacement_q_suppress: bool = False

    # torch.compile
    compile: bool = False

    # Dead point removal
    remove_dead_points: bool = False
    dead_point_threshold: float = 1e-4  # remove if m_i*q_i < threshold * max(m*q)
    dead_point_interval: int = 1        # check every N steps

    # Mass bandwidth type
    mass_bandwidth_type: Literal["global", "per_point_knn", "per_point_abramson"] = "global"
    mass_delta_adaptive: bool = False   # True: recompute δ/τ every step, False: fixed at initial
    mass_knn_k: int = 10               # k for kNN distance in δ computation

    # Redistribution parameters
    redistribute: bool = False
    redistribute_n_iters: int = 10
    redistribute_step_size: float = 0.01
    redistribute_tol: float = 1e-4
    redistribute_interval: int = 1
    redistribute_delta_ratio: float = 0.5  # δ_redist = mass_delta * ratio
    redistribute_max_disp_ratio: float = 0.05  # max per-iter displacement as fraction of δ_redist
    redistribute_n_iters_after_removal: int = 0  # extra redistribute after dead point removal (0 = off)
    # D2b-2 scheduling axis. "every_step" is the legacy behavior
    # ((step+1) % redistribute_interval == 0). "fixed_physical_interval"
    # fires when the ACCUMULATED physical time crosses multiples of
    # redistribute_physical_interval (dt-independent invocation rate).
    # "adaptive_cv_trigger" fires when the permutation-invariant
    # self-excluded KDE-density CV (the operator's own stopping
    # quantity; NEVER ordered edge-length CV) exceeds
    # redistribute_trigger_cv.
    redistribution_schedule: Literal[
        "every_step", "fixed_physical_interval",
        "adaptive_cv_trigger"] = "every_step"
    redistribute_physical_interval: float = 0.0
    redistribute_trigger_cv: float = 0.15
    # P2 (production integration): when True the solver CONSUMES the
    # timeline's sticky support gates -- whole-loop extinctions are
    # certified inline (continue on success), any other closed gate
    # stops continuation with stop_stage="support_gate". Default False:
    # legacy trajectories untouched; requires an attached
    # support_timeline to act.
    enforce_support_gates: bool = False

    # Closure patch 2 (reviewer T): observational BEM-setup telemetry.
    # When True each MMStepResult carries a detached BEMSetupSnapshot of
    # what the BEM setup actually consumed. MUST be trajectory-neutral
    # (pinned by the noninterference test). Default off.
    bem_setup_telemetry: bool = False

    # AUDIT-ONLY knob (theta=15 deg stays the production default): the
    # phase-constant alignment threshold of the adaptive rank machine.
    # Three shapes (star, flower, pair@512) show the SAME decelerating
    # align growth killed mid-climb at 15 deg while spectrum, support
    # and geometry are clean; diagnostic runs at 30 deg observe the
    # asymptote so the reviewer can redesign the criterion. Never flip
    # the default without reviewer judgment.
    rank_theta_const_deg: float = 15.0

    # Pairwise kernel backend for coherence (scalar_density / vector_field).
    # "naive" is faster at small N; "keops" lazy is faster at N≳1000.
    backend: Literal["naive", "keops"] = "naive"

    # G2b: which discretization of the SAME linearized MS tangent metric
    # evaluates W_lin. "bem" is the production default (bitwise
    # untouched). "grid_poisson" replaces the BIE by the weighted-
    # Poisson grid metric (current-to-phase filling + boundary-flux
    # scatter + L_rho0^dagger); volume/component admissibility comes
    # from the grid compatibility rows composed through the angle map
    # (compose_constraint_basis), NOT from the q^2 m weights or the
    # spectral machinery -- so it requires
    # bem_solver_mode == "legacy_global_projection" and compile=False
    # (both enforced at construction).
    metric_backend: Literal["bem", "grid_poisson", "bie"] = "bem"
    # Optional GridMetricConfig (src.torch.transport.grid_wasserstein);
    # None builds the calibrated defaults (512^2, box (-2,2)^2,
    # fill_epsilon 0.06, support_threshold 1e-3).
    grid_metric: "object | None" = None
    # "bie": block-diagonal boundary-integral metric
    # (src.torch.transport.bie_wasserstein.BIEWassersteinMetric); the
    # metric components are the winding bulk components merged at the
    # bridging gap, the cancelling pair is masked + frozen, the
    # compatibility rows are the exact endpoint flux sums. Requires the
    # same legacy/augment settings as the grid backend plus
    # carrier measure and unit velocity (bulk rows) and all grid-only
    # arms off. None builds BIEMetricConfig() defaults.
    bie_metric: "object | None" = None
    # G3a (reviewer item 5): which volume the filling projection
    # targets. "current_divergence" recomputes V_hat from the current
    # cloud every step (the BEM-parity semantics: reconstruction error
    # is measured against THIS step's cloud, physical drift is a
    # separate recorded quantity). "initial_fixed" pins the target to
    # the first step's V_hat (scientific conservation track).
    grid_volume_target_mode: Literal[
        "current_divergence", "initial_fixed"] = "current_divergence"
    # Phase-support projection (review option B, 2026-08-07): after a
    # merger, hidden-boundary mark pairs freeze in the bulk (velocity
    # ~ q ~ 0) and leave a permanent kernel-scale dipole defect
    # ("fossil"). When enabled, marks that the RECONSTRUCTED PHASE
    # declares strictly interior (two-sided rho test + band distance,
    # only in the stably single-component state, one-shot per step,
    # shadow-verified: phase/current/perimeter/volume preserved to
    # tolerance) are removed as a representation projection. NOT the
    # q-based dead-point rule -- q is never consulted. Grid backend
    # only; default off (all existing paths bitwise unchanged).
    grid_phase_support_projection: bool = False
    grid_psp_config: "object | None" = None   # PhaseSupportProjectionConfig
                                              # (None -> module defaults);
                                              # recorded in run JSONs
    # Arm 4 (2026-08-09): passive advection of low-coherence marks by
    # the committed step's own transport field grad(phi), phi =
    # L^dagger(B y*). A mark whose normal has lost its meaning
    # (q ~ 0, antiparallel neighborhood) is interior MATERIAL and
    # rides the flow as a tracer instead of freezing in place; the
    # blend weight (1-q)^p is a smooth label-free function of the
    # current state. Grid backend only; default off (all existing
    # paths bitwise unchanged).
    grid_fossil_advection: bool = False
    grid_advection_config: "object | None" = None  # FossilAdvectionConfig
    # Arm 5 (2026-08-09): persistent carrier-amplitude fade of
    # phase-interior marks (spec-2.0-lite). Amplitudes a_i in [0,1]
    # multiply the carrier masses seen by the GRID layer (filling,
    # flux, volume rows); monotone rate-limited decrease gated by the
    # frozen Layer-1 interior test -- the uniform whole-complex
    # reduction the feasibility audit measured as the optimal move.
    # Grid backend only; default off.
    grid_carrier_amplitude_fade: bool = False
    grid_amplitude_config: "object | None" = None  # CarrierAmplitudeConfig
    # Arm 7 (2026-08-10): horizontal-space optimization. A dead mark's
    # displacement direction is a lift-fiber direction with vanishing
    # objective cost (measured: trust-ncg scatters the hidden wall by
    # 310x). Marks with q < grid_dead_dof_q have their displacement DOF
    # removed from the admissible basis (extra unit constraint rows);
    # they move only by advection/redistribution. Grid backend only;
    # default off.
    # 0F: incidence-augmented componentwise volume rows. "augment"
    # ADDS winding-derived physical bulk rows to the grid compatibility
    # rows -- fired only when the winding bulk partition is STRICTLY
    # finer than the grid partition (post-bridge); when the partitions
    # agree the legacy path runs bitwise-unchanged.
    # "bulk_only_diagnostic" is the 0F-0 attribution cell ONLY (never
    # production): bulk rows alone + projected Poisson forward.
    grid_bulk_rows_mode: Literal[
        "off", "augment", "bulk_only_diagnostic"] = "off"
    # 0L-B0: how the admissible rows are built INSIDE the contact
    # layer, i.e. only on the 'b_finer' branch of grid_bulk_rows_mode
    # = "augment" (every other branch is bitwise-unchanged):
    #   "stack"              cell A, current production: [G; C_bulk],
    #                        rank-revealing (rank 3 post-bridge)
    #   "aligned_prequotient" cell B: [G; P_rel C_bulk] -- grid rows +
    #                        the RELATIVE (intra-grid-component) bulk
    #                        exchange rows only; drops the third,
    #                        non-physical direction delta_comp = the
    #                        scatter/active-set mismatch between G and
    #                        C_+ (aggregation projector, permutation-
    #                        equivariant; see relative_bulk_rows)
    #   "bulk_projected"     cell C: exact bulk rows alone + projected
    #                        Poisson forward (grid incompatibility
    #                        removed by explicit projection, recorded)
    #   "grid_only"          cell D, quotient NEGATIVE control: G only
    #                        (C_- dropped immediately; bulk rows kept
    #                        as shadow telemetry)
    # PRODUCTION default since 0L-B0 (reviewer verdict 2026-08-16):
    # "aligned_prequotient" -- the canonical aggregate/relative
    # decomposition; the rank-3 "stack" caused the E4 tilt (cell A)
    # and is kept as the diagnostic control.
    grid_contact_rows_mode: Literal[
        "stack", "aligned_prequotient", "bulk_projected",
        "grid_only"] = "aligned_prequotient"
    # 0L-B0 cell E (DIAGNOSTIC only, never production as-is): when the
    # conservative sweep reads fewer compatibility components than the
    # raw bulk incidence, hold a bulk-seeded two-component gauge/
    # deflation partition (matrix untouched). gamma_cross telemetry
    # says whether that is a soft-mode deflation or an artificial
    # constraint on a strongly connected operator.
    grid_gauge_hold_bulk_seeded: bool = False
    # 0N (reviewer 2026-08-18): monotone loopwise redistribution --
    # per-loop active set, per-loop monotone backtracking on the
    # post-cap update, stage-total displacement cap. Requires
    # redistribution_density_scope="loopwise". False = bitwise legacy.
    redistribution_monotone: bool = False
    redistribution_monotone_parts: str = "abc"   # 0N-2 diagnostic subsets
    # 0L-B2: certificate-gated quotient of raw BULK classes (never of
    # loops). "off" = bitwise State II forever. "certificate_shadow":
    # the SOLVER runs a committed-level shadow transaction (keep vs
    # drop from the same source, judged on the committed state) and
    # commits the quotient irreversibly only when every pre-registered
    # gate passes; the stepper then reads C_quot = Q_quot C_raw and
    # builds rows [G; P_rel C_quot] (exact merger: {G}). Requires a
    # fixed particle count and no deletion / PSP / vertical / advection.
    grid_quotient_mode: Literal["off", "certificate_shadow"] = "off"
    # 0L-B2 switch gates (pre-registered by scripts/experiments/
    # quotient_switch_calibration.py from the clean State-II window,
    # the B1 positive-fixture shadow switch and the B0 B/D one-step;
    # NEVER from the switch run itself). None = not calibrated -> the
    # shadow mode refuses to construct.
    grid_quotient_eps_split: "float | None" = None    # (R_drop)+ floor
    grid_quotient_eps_switch_x: "float | None" = None  # |dX|_inf / h_wall
    grid_quotient_eps_switch_theta: "float | None" = None  # |dtheta|_inf
    grid_quotient_eps_volume: "float | None" = None   # |dV_merged|/V
    grid_incidence_ell_factor: float = 4.0
    # 0J: perimeter-energy coherence weight. "self_renormalized" is
    # the well-balanced correction q^WB = r_l q^self (unique loopwise
    # scalar preserving the self profile and the source energy);
    # "loop_mean" is its circular-limit control. Scope: pre-contact
    # correction for loops approaching along their whole length.
    # L0J (2026-08-21): "contact_complex_renormalized" = q^CC with the
    # sigma-scale ENERGY contact complex (side-wise r_{alpha,l};
    # reduces exactly to self_renormalized under whole-loop contact).
    # Scope: annihilating LOCAL contact; algebraic baseline until the
    # L0J continuity/refresh audits pass (hard membership jumps).
    # "contact_complex_tb" (L0J-M candidate A, DIAGNOSTIC): q^CC plus
    # the translation-balanced side correction -- the side factors
    # solve the energy + translation-response equality system (outside
    # contribution on the RHS), box 0 <= r <= 1/q^self, fail-closed on
    # rank/feasibility. Note: r > r_CC locally (q > q^CC) is ALLOWED
    # and generally necessary (the structural obstruction).
    perimeter_q_mode: Literal[
        "full", "loop_mean", "self_renormalized",
        "contact_complex_renormalized",
        "contact_complex_tb"] = "full"
    grid_freeze_dead_dof: bool = False
    grid_dead_dof_q: float = 0.15
    # Arm V2 (2026-08-11): coupled minimizing movement -- the carrier
    # amplitude decrement da becomes a decision variable of THE SAME
    # per-step minimization (WFR-type unsplit JKO), solved by block
    # coordinate descent inside step(): (s, dtheta) trust-ncg with the
    # removal deposit C@da as a drho offset in the grid metric, then a
    # box QP in da whose linear term carries the cross coupling
    # <c_i, Ldag flux(s, dtheta)>. The split-scheme ladder measured
    # removal tempo monotonically anti-correlated with survival
    # (A8 1014 / A9-A10 965-974 / A10b-c 847-883) because transport
    # could never compensate removal within the step; the cross term
    # is exactly that compensation. Free set from the frozen pre-step
    # gate (state functional only); gate empty -> identical to the
    # pure Otto scheme (da = 0 optimal, bitwise pinned). Grid backend
    # only; default off. Uses grid_amplitude_config for gate/lam.
    grid_vertical_dof: bool = False
    grid_vertical_sweeps: int = 2
    # Phase IIIc-0 (2026-08-13): antipodal-sheet-aware mass estimator
    # ablation. "oriented_kde" weights the KDE density by
    # omega(n_i . n_j) = (1+s)/2 and uses the co-oriented kNN
    # parameter rule -- fixes the measured count-ratio coherence
    # floor (51/231 = 0.22) on coincident antiparallel sheets while
    # being a no-op on clean boundaries (identical (delta, tau) on a
    # clean circle). Default "kde" is bitwise-identical.
    # 0K (2026-08-15): "loopwise_oriented_kde" additionally masks the
    # oriented density to same-certified-loop pairs (global-N
    # normalization kept -- N cancels exactly under the standard tau
    # rule). Guarantee: cross-loop independence + sheetwise H^1
    # consistency. Requires global non-adaptive scalar (delta, tau)
    # (constructor guard); labels come from the two-stage bootstrap
    # certificate in oriented_varifold.loopwise_mass and are frozen
    # within each step.
    mass_estimator: Literal["kde", "oriented_kde",
                            "loopwise_oriented_kde"] = "kde"
    # 0L-A (2026-08-15): the physical bulk volume functional.
    # "current" is the 0F row C_b = sum zeta (s - r dtheta) on the
    # oriented current volume 1/2 sum m x.n. "geometric_area"
    # replaces it with the EXACT differential of the certified
    # polygon area, C^geom_i = g_i . n_i (positions-only, no dtheta
    # column) -- the tilt-robust volume row (contact layer: stored
    # normals tilt off the geometric normal and 1/2 sum m x.n
    # diverges from the true area). Under "geometric_area" the
    # stacked [C_grid; C_geom] is used in EVERY partition branch
    # (equal included -- the pre-merger rush happens at partition
    # equal, where the legacy branch would never fire the physical
    # row); stacking C_cur with C_geom is forbidden (two different
    # volume functionals = overconstraint); the current rows keep
    # running as shadow telemetry.
    grid_bulk_rows_functional: Literal[
        "current", "geometric_area"] = "current"
    # L0J-M (reviewer 2026-08-21): exact polygon FIRST-MOMENT rows
    # (per raw bulk in State I/II; aggregated per quotient class via
    # Q_quot (x) I_2 in State III -- NEVER passed through P_rel: the
    # grid rows G conserve no spatial moment, so there is nothing to
    # de-duplicate). They put the one-phase MS invariant
    # d/dt int_E x dx = 0 into the admissible tangent space; the
    # artificial q^CC translation force then has no admissible
    # direction to act on. Default off = bitwise legacy.
    grid_bulk_first_moment_rows: bool = False
    # L0J-M rows come in two forms (user 2026-08-24): "polygon" =
    # exact polygon-moment differential on the CERTIFIED cyclic order
    # (L0J-M original; positions-only, dtheta block zero); "current" =
    # ORDER-FREE flux-family rows (transport identity, same zeta/r/
    # endpoints as the volume rows; dtheta block nonzero; consumes NO
    # cyclic order and is functional-family-consistent with the
    # current volume constraint). Default "polygon" = bitwise.
    # "current_centered" pins the BARYCENTER instead of the origin-
    # based moment: row = C_M - xbar_b C_V (a linear combination of
    # the two flux rows -- still fully order-free). Rationale
    # (measured, 2026-08-24): pinning M while the area functional
    # carries a drift dA forces xbar = M/A to move by -xbar dA/A
    # (gauge-dependent leakage; polygon rows 1.7e-5 at t=0.002,
    # current rows exactly the model value 8.4e-6); pinning dxbar = 0
    # removes the leakage by construction.
    grid_bulk_first_moment_rows_form: Literal[
        "polygon", "current", "current_centered"] = "polygon"
    # 0L-A blocker 2: the phase-filling volume target. With
    # "geometric_area" the (initial_fixed) target is the summed
    # certified polygon area instead of 1/2 sum m x.n, so the
    # tangent constraint and the reconstruction target read the SAME
    # volume semantics. rows="current" with target="geometric_area"
    # is rejected; rows="geometric_area" with target="current" is
    # allowed (the 0L-A2a row-only causal experiment).
    grid_volume_target_functional: Literal[
        "current", "geometric_area"] = "current"
    # 0L-R (2026-08-15): redistribution integrability. The rush is a
    # redistribution-stage tilt feedback (0L-A); the production
    # candidate closes it with (a) loopwise density/CV/cap -- each
    # loop equalizes its OWN sampling density, per-loop stopping and
    # per-loop cap (a global cap couples the loops); (b) local-PCA
    # tangents recomputed every subiteration on same-loop
    # neighbourhoods; (c) end-of-stage PCA normal re-anchoring (an
    # approximate boundary-admissibility retraction with a cos-30-deg
    # orientation margin). Defaults keep the legacy path bitwise.
    redistribution_density_scope: Literal[
        "global", "loopwise"] = "global"
    redistribution_tangent_source: Literal[
        "angles", "local_pca"] = "angles"
    redistribution_normal_retraction: bool = False
    # 0L-kappa: the Frenet angle correction's curvature estimator.
    # "loopwise" evaluates kappa AND the 0K loopwise masses on
    # same-certified-loop pairs only -- the union-cloud estimator
    # sign-flips in the contact layer (disk true +2.857 vs union
    # -5.609 at endgame-#2 1325; loopwise +2.855) and turns the
    # correction into an ANTI-correction. Only the estimator inputs
    # change; time level / tangents / single-shot application stay
    # legacy. Default "union" is bitwise.
    redistribution_curvature_scope: Literal[
        "union", "loopwise"] = "union"
    # 0L R2: the coherence scaling of the redistribution update.
    # "stale_full" = the historical pre-MM q^full; "r_loop" = the
    # source-frozen particle-aligned r_{l(i)} (constant per loop, so
    # it only rescales each loop's pseudo-time, never the path --
    # the reviewer-preferred semantics); "q_wb" = r_l q^self.
    # r_loop / q_wb require perimeter_q_mode="self_renormalized".
    redistribution_q_policy: Literal[
        "stale_full", "r_loop", "q_wb"] = "stale_full"
    # 0L-kappa hardening: DIMENSIONLESS denominator certificate for
    # the loopwise curvature (beta = eps B / c_xi -> 1 straight-sheet
    # limit; gamma = B / median_loop(B) local-gap detector).
    # PRE-REGISTERED 2026-08-16 = half the clean-fixture lower
    # envelope (circle/ellipse/flower/annulus x h in {0.8,1,1.25}h0:
    # min beta 0.969, min gamma 0.964). None disables (audit).
    redistribution_kappa_den_normalized_min: "float | None" = 0.48
    redistribution_kappa_den_relative_min: "float | None" = 0.48
    # 0L-theta (2026-08-16): the weak-form angle transport. The
    # union-cloud A/B matrices leak the other sheet's motion into a
    # loop's normal rotation once gap < sigma_angle (4th instance of
    # the cross-loop contamination family; measured: a pure radial
    # hole motion rotates the disk normals by 1.7e-4 at endgame-#3
    # 1525, exactly zero under a same-loop mask). "loopwise" builds
    # and solves the map LOOP BY LOOP (direct sum, loop-local rcond
    # -- a global masked solve would let one loop's singular scale
    # truncate another's modes) and assembles a block-diagonal
    # AB_solve transparently for all consumers.
    angle_constraint_scope: Literal["union", "loopwise"] = "union"
    # The measure weight of the weak form. "visible_full" = m q^full
    # (historical); "raw_loopwise" = the 0K loopwise mass alone --
    # the visible weight adds the spurious kinematic term
    # -s d_tau(log q) on nonuniform-q loops and keeps a cross-loop
    # dependence through q^full even under the loopwise kernel.
    angle_constraint_measure: Literal[
        "visible_full", "raw_loopwise"] = "visible_full"
    # Constant-mode consistency: "mean_zero" solves the CONSTRAINED
    # least squares D = Z (A Z)^+ B P (Z = orthonormal basis of
    # ker m^T, P = I - 1 m^T/M) so D 1 = 0 and m^T D = 0 hold
    # exactly -- NOT the projector sandwich P A^+ B P, which breaks
    # the residual minimality off circle-symmetric fixtures.
    angle_constraint_consistency: Literal[
        "none", "mean_zero"] = "none"
    # PCA spectral-certificate thresholds, PRE-REGISTERED 2026-08-15
    # from the clean-fixture envelope at pca bandwidth = mass_delta,
    # h in {0.8, 1.0, 1.25} h0 (circle/ellipse/flower/annulus): worst
    # clean rho = 0.146 (flower, high curvature), worst clean N_eff =
    # 3.11 -> thresholds 0.25 / 2.5 with safety margins; an isotropic
    # blob sits at rho ~ 1 (clear separation). The 1325 contact state
    # is VALIDATION only (rho 0.004, N_eff 4.8) -- never calibration.
    redistribution_pca_rho_max: "float | None" = 0.25
    redistribution_pca_neff_min: "float | None" = 2.5
    # Cap on the trust-ncg iterations of the SWEEP re-minimizations
    # (the y-block after a deposit offset is injected). They start
    # warm from y* and only need a correction; without the cap the
    # merger window's ill-conditioned landscape ground a single
    # re-minimize for hours (measured: exact-merger run stuck 7 h
    # inside one hessp/Poisson loop at step ~1250). None = uncapped.
    grid_vertical_max_iter: "int | None" = 60
    # G3a (plan: velocity scaling, introduced only after G2 parity so
    # metric differences and scaling differences never mix): optimize
    # in v = y / dt with objective F(dt v)/dt. Same minimizer
    # (y* = dt v*), but gradients and curvatures are O(1) in dt, so
    # optimizer_tol acts on a dt-independent gradient scale. Applies
    # to BOTH metric backends; default unchanged.
    optimizer_variable: Literal["displacement", "velocity"] = \
        "displacement"

    # 0D time-level enums. Only the legacy values are implemented until
    # the D1/D2 variants land; the enums exist NOW so the D0 golden
    # configs can pin them explicitly instead of relying on defaults.
    redistribution_rule: Literal["legacy_hybrid", "post_mm_frozen",
                                 "substep_refreshed"] = "legacy_hybrid"
    dead_point_rule: Literal["legacy_lagged", "semi_implicit",
                             "refreshed"] = "legacy_lagged"


@dataclass(frozen=True)
class BEMSetupSnapshot:
    """Detached telemetry of what one MM step's BEM setup ACTUALLY
    consumed (closure patch 2, reviewer T). Observational only: built
    from detached clones, never mutates or implicitly constructs solver
    state, and generation must leave the trajectory bitwise identical.

    CANONICAL pointwise quantities (audit reference frame) are stored
    separately from the ACTUALLY-USED quantities, because the production
    variants differ in which measure each subsystem consumes (operator:
    m vs q*m; volume constraint: q^2 m; velocity: q vs 1). The policy
    strings state the mapping so the JSON can never silently misreport
    e.g. "q^2 m" when a variant used m.
    """
    # canonical pointwise quantities
    positions: torch.Tensor
    carrier_masses_m: torch.Tensor
    coherence_q: torch.Tensor
    canonical_qm: torch.Tensor
    canonical_q2m: torch.Tensor
    # actually consumed by this setup
    operator_weights_used: torch.Tensor
    physical_offsets: torch.Tensor          # signed r_ik (N, K)
    operator_measure_policy: str
    endpoint_weights_used: torch.Tensor
    endpoint_weights_policy: str
    volume_constraint_weights_used: torch.Tensor
    volume_constraint_policy: str
    velocity_scale_used: torch.Tensor | None
    velocity_policy: str
    # scales
    delta_used: float
    tau_used: float
    perimeter_sigma_used: float
    angle_sigma_used: float
    epsilon_bem_used: float | None
    epsilon_mode: str
    solver_mode: str
    # BEM geometry and spectrum
    endpoint_positions: torch.Tensor | None
    endpoints_per_particle: int | None
    singular_values_ascending: torch.Tensor | None
    singular_values_source: str


@dataclass(frozen=True)
class GridMetricSetupSnapshot:
    """Detached telemetry of what one MM step's GRID metric setup
    actually consumed (G2b + G3a reviewer items 5/6; BEMSetupSnapshot
    policy: observational only, held on the per-step result, no new
    timeline level). All floats are plain Python scalars -- nothing
    here retains graph or grid tensors.

    Volume semantics (reviewer item 5): `reconstruction_volume_error`
    measures filling-vs-target agreement for THIS step's target and is
    ~0 by the water-filling projection -- it is NOT physical volume
    conservation. Physical drift is `target_volume_drift_relative`
    (current divergence-theorem volume vs the first step's).
    """
    n_components: int
    # thresholded active-support cell areas -- NOT integral(rho)
    component_active_support_areas: tuple
    # integral of rho_metric over each component (physical phase mass)
    component_phase_masses: tuple
    target_volume_current: float     # V_hat from THIS step's cloud
    target_volume_used: float        # what the filling targeted
    target_volume_initial: float     # first grid step's V_hat
    target_volume_drift_relative: float
    phase_grid_volume: float         # integral rho_metric (whole grid)
    reconstruction_volume_error: float
    # global carrier rescale target/V_hat applied inside the filling
    # (G3c): exactly 1.0 in current_divergence mode; its departure
    # from 1 in initial_fixed mode IS the accumulated KDE drift
    carrier_renormalization: float
    projection_l2_relative: float    # rho_raw -> rho_metric correction
    speck_mass_dropped: float
    fill_epsilon: "float | None"     # None on the BIE backend
    grid_dx: "float | None"
    grid_shape: "tuple | None"
    constraint_rank: int             # composed grid rows (== C)
    n_params: int                    # dim of the admissible basis
    # confidence telemetry (reviewer item 6) -- the grid analogue of
    # the BIE rank-confidence diagnostics:
    component_counts_at_thresholds: tuple   # at thr/3, thr, 3*thr
    constraint_row_singular_values: tuple
    constraint_row_condition: float
    min_component_phase_mass: float
    min_component_active_area: float
    box_boundary_margin_over_eps: "float | None"
    quadratic_path: str
    solver_backend: str
    operator_measure_policy: str
    velocity_policy: str
    endpoints_per_particle: int
    # 0J: per-step perimeter-q correction stats (min/max r_l,
    # min mean self-coherence, cross-loop kernel max); None when
    # perimeter_q_mode == "full"
    perimeter_q_stats: "dict | None" = None
    # 0F incidence-augmentation telemetry (inert when mode="off")
    bulk_rows_mode: str = "off"
    augmentation_fired: bool = False
    n_bulk_components: "int | None" = None
    r_grid: "int | None" = None
    r_bulk: "int | None" = None
    r_stack: "int | None" = None
    delta_compat: "float | None" = None
    incidence_margin: "float | None" = None
    partition_relation_grid_bulk: "str | None" = None
    # 0L-B0 contact-layer telemetry
    contact_rows_mode: "str | None" = None
    eps_row: "float | None" = None            # scale-free G vs C_agg
    eps_row_composed: "float | None" = None   # same after AB composition
    gauge_hold_fired: "bool | None" = None
    gamma_cross: "float | None" = None
    n_compat_base: "int | None" = None
    n_compat_used: "int | None" = None
    partition_relation_base: "str | None" = None
    # 0L-B2 quotient bookkeeping (raw bulk -> quotient class)
    quotient_classes: "list | None" = None
    n_quotient_groups: "int | None" = None
    n_frozen_dof: int = 0
    # BIE backend telemetry (None on the grid backend)
    metric_backend: str = "grid_poisson"
    bie_sigma_tail: "list | None" = None      # sigma_{n-1}/sigma_1 per block
    bie_null_level: "list | None" = None      # sigma_n/sigma_1 per block
    bie_lambda_min: "float | None" = None     # admissible-subspace Hessian
    bie_lambda_max: "float | None" = None
    bie_n_masked: "int | None" = None
    bie_merged_pairs: "list | None" = None    # (bulk_a, bulk_b, gap)
    bie_epsilon: "float | None" = None
    bie_g_bridge: "float | None" = None
    bie_block_sizes: "list | None" = None
    bie_setup_wall_seconds: "float | None" = None


@dataclass(frozen=True)
class GridMetricStepStats:
    """Per-step Poisson solve counters (G3a reviewer item 1),
    snapshotted from the operator after the optimizer finishes. matvec
    and preconditioner-application counts are derivable: one of each
    per PCG iteration (+1 preconditioner apply per solve);
    sparse_direct solves report 0 iterations."""
    n_forward_solves: int
    n_projected_solves: int
    pcg_iterations_total: int
    pcg_iterations_min: int
    pcg_iterations_median: float
    pcg_iterations_max: int
    max_relative_residual: float
    max_inactive_rhs_fraction: float
    solve_wall_seconds: float
    # 0L-B0 cell C: max over the step of ||drho - P drho|| / ||drho||
    # in the projected forward (None unless the projected path ran)
    max_projection_rel: "float | None" = None


@dataclass
class MMStepResult:
    """Result of a single MM step.

    Attributes:
        varifold: Updated varifold after optimization.
        perimeter: Final perimeter value.
        wasserstein: Final linearized Wasserstein value.
        objective: Final objective value (P + W_lin).
        converged: Whether optimizer converged.
        n_iter: Number of optimizer iterations.
        displacements: Normal direction displacements (N,).
        delta_angles: Angle changes (N,).
        masses: (N,) KDE masses from previous step (fixed during optimization).
        effective_masses: (N,) masses * coherence from previous step.
    """
    varifold: OrientedPointCloudVarifold
    perimeter: float
    wasserstein: float
    objective: float
    converged: bool
    n_iter: int
    displacements: torch.Tensor
    delta_angles: torch.Tensor
    masses: torch.Tensor
    effective_masses: torch.Tensor

    # --- final-point diagnostics (0A) ------------------------------------
    # perimeter / wasserstein / objective above are RE-EVALUATED at the
    # returned optimizer point result.x, not taken from whatever point the
    # optimizer happened to evaluate last (those may differ by a line-search
    # trial). The fields below record both endpoints of the frozen objective
    # F(y) = P_frozen(y) + W_lin(y):
    objective_initial: float | None = None          # F(0)
    frozen_perimeter_initial: float | None = None   # P(0)
    wasserstein_initial: float | None = None        # W(0) (should be ~0)
    gradient_norm_initial: float | None = None      # |grad F(0)|
    gradient_norm_final: float | None = None        # |grad F(y*)|
    relative_gradient_norm: float | None = None     # |gF(y*)|/max(1,|gF(0)|)
    step_norm: float | None = None                  # |y*|
    objective_decreased: bool | None = None         # F(y*) <= F(0) (+tol)
    optimizer_message: str | None = None
    n_fev: int | None = None                        # objective evaluations
    # spectral_bordered only: compatibility residual of the last evaluation
    r_comp: float | None = None
    # spectral_bordered only (0C-5b): rank-confidence diagnostics. NOTE
    # r_comp is NOT rank validation — it is machine zero by construction for
    # any selected rank; these fields are the actual evidence.
    # WORKING-rank quantities (what the solve actually used; in
    # state_machine mode this may be a HELD rank that differs from what
    # the evidence proposed):
    detected_rank: int | None = None         # working rank C
    rank_gap_ratio: float | None = None      # s_(C+1)/s_(C) AT working C
    rank_abs_level: float | None = None      # s_(C)/s_max AT working C
    rank_action: str | None = None           # state_machine decision action
    rank_pending: int | None = None          # persistence counter
    # RAW EVIDENCE quantities (what the spectrum proposed, kept separate
    # so disagreement intervals can be reconstructed after the fact):
    rank_status: str | None = None           # clean/weak_gap/...
    rank_c_abs: int | None = None            # absolute-null criterion count
    rank_c_gap: int | None = None            # max-gap criterion split
    rank_evidence_gap_ratio: float | None = None  # winning ratio at c_gap
    rank_spectrum_head: list | None = None   # leading relative spectrum
    subspace_angle_deg: float | None = None  # vs previous step's U0
    # observational BEM-setup telemetry (config.bem_setup_telemetry);
    # held only on this per-step result -- callers keep scalar summaries
    # and select frames, never the full history (memory policy)
    bem_setup_snapshot: "BEMSetupSnapshot | None" = None
    # grid backend only (G2b): what the grid metric setup consumed
    grid_setup_snapshot: "GridMetricSetupSnapshot | None" = None
    # grid backend only (G3a): per-step Poisson solve counters
    grid_step_stats: "GridMetricStepStats | None" = None
    # review blockers 4/6: varifold is the MM MINIMIZER; the state the
    # next step actually consumes (after redistribution, deletion and
    # the phase-support projection) is committed_varifold, assigned by
    # MMSolver just before the callback. Checkpoints MUST use it.
    committed_varifold: "OrientedPointCloudVarifold | None" = None
    phase_support_projection_outcome: "dict | None" = None
    # arm 4: fossil-advection stage telemetry (None when disabled)
    advection_stats: "dict | None" = None
    # arm 5: carrier-amplitude fade telemetry (None when disabled)
    amplitude_stats: "dict | None" = None
    # arm V2: coupled vertical DOF -- da over the free set (indices
    # into the PRE-step varifold) chosen by the step's own coupled
    # minimization; the solver commits amplitudes from these.
    vertical_da: "torch.Tensor | None" = None
    vertical_idx: "torch.Tensor | None" = None
    vertical_stats: "dict | None" = None
    # 0F: realized-step residuals of the physical bulk volume rows
    # (rows_bulk @ (s, dtheta)); None when incidence rows are off or
    # not fired this step
    bulk_flux_residuals: "tuple | None" = None
    # 0L-A: residuals of the SHADOW current-volume rows when the
    # active physical rows are the geometric-area rows (telemetry
    # only -- the shadow functional is never constrained)
    bulk_flux_residuals_shadow: "tuple | None" = None
    # 0L-B0: row actions on the committed MM velocity v = (s, dtheta)
    # inside the contact layer -- G v (grid rows), C_bulk v (raw bulk
    # rows, whether or not constrained), C_rel v (relative bulk rows),
    # plus optimizer conditioning (objective decrease, iterations)
    contact_rows_telemetry: "dict | None" = None
    # 0L-B1: loop -> (grid compatibility component, raw bulk) maps of
    # this step's source state (State-II bookkeeping for the contact
    # certificate); None when incidence rows are off
    contact_pair_context: "dict | None" = None
    # 0L-B1/B2: contact certificates on the SOURCE state (list of
    # dicts; empty when no State-II candidate) and the persistent
    # quotient bookkeeping after this step
    contact_certificate_source: "list | dict | None" = None
    quotient_state: "dict | None" = None
    # 0L-B2 shadow-switch transaction record (solver-filled on the
    # switch step; None otherwise)
    quotient_switch: "dict | None" = None
    contact_certificate_committed: "list | dict | None" = None


@dataclass(frozen=True)
class MassScaleSnapshot:
    """Resolved KDE bandwidths: `*_for_kde` is what compute_masses
    receives (scalar or per-point tensor), `*_summary` the scalar
    diagnostics value."""
    delta_for_kde: "float | torch.Tensor"
    tau_for_kde: "float | torch.Tensor"
    delta_summary: float
    tau_summary: float


@dataclass(frozen=True)
class AngleMapSolution:
    """The pre-solved weak-form angle map AB_solve = A^dagger B.

    Solved exactly once per step (0C-5a.1): the spectral compatibility
    constraint and the evolution parametrization must agree on the SAME map
    Delta-theta(s) — an independently re-solved pseudo-inverse (different
    driver, rcond, or truncation) would put the optimizer on a subspace built
    from a slightly different constraint than the one it evolves with.
    """
    AB_solve: torch.Tensor
    angle_cond: float
    angle_n_truncated: int
    # 0L-theta: per-loop solve telemetry when the map was assembled
    # as a direct sum (angle_cond above = max over blocks,
    # angle_n_truncated = sum). None for the legacy global solve.
    block_stats: "list | None" = None


def solve_angle_map(
    A_angle: torch.Tensor,
    B_angle: torch.Tensor,
    rcond: float | None = None,
) -> AngleMapSolution:
    """Solve A @ Dtheta = B @ s for the linear map Dtheta = (A^dagger B) s.

    CPU: LAPACK gelsd via torch.linalg.lstsq (SVD-based, best residual).
    CUDA: TruncatedSVD pseudo-inverse (gelsd unavailable).
    """
    A_det = A_angle.detach()
    B_det = B_angle.detach()

    if A_det.device.type == "cpu":
        result = torch.linalg.lstsq(A_det, B_det, rcond=rcond, driver="gelsd")
        AB_solve = result.solution
        S = result.singular_values
        if S.numel() > 0 and (S > 0).any():
            angle_cond = (S.max() / S[S > 0].min()).item()
        else:
            angle_cond = float("nan")
        angle_n_truncated = (
            max(0, A_det.shape[0] - result.rank.item())
            if result.rank.numel() > 0 else 0
        )
    else:
        svd = TruncatedSVD.from_matrix(A_det, rcond=rcond)
        AB_solve = svd.Vh.T @ (svd.S_inv.unsqueeze(-1) * (svd.U.T @ B_det))
        S_inv = svd.S_inv
        active = S_inv > 0
        if active.any():
            S_active = 1.0 / S_inv[active]
            angle_cond = (S_active.max() / S_active.min()).item()
        else:
            angle_cond = float("inf")
        angle_n_truncated = svd.n_truncated
    return AngleMapSolution(AB_solve, angle_cond, angle_n_truncated)


def solve_angle_map_blocks(
    positions: torch.Tensor,
    tangents: torch.Tensor,
    angle_weights: torch.Tensor,
    loop_labels: torch.Tensor,
    sigma: float,
    kernel: str,
    rcond: float | None = None,
    consistency: str = "none",
) -> AngleMapSolution:
    """0L-theta: direct-sum angle map D = diag(D_1, ..., D_L).

    Each certified loop's (A_a, B_a) is BUILT on the loop's subset
    (identical to a standalone computation by construction -- the
    kernel is pairwise) and solved with a LOOP-LOCAL rcond, then
    assembled block-diagonally: off-block entries are exactly zero,
    and one loop's singular scale can never truncate another loop's
    modes. consistency="mean_zero" solves the constrained least
    squares D_a = Z_a (A_a Z_a)^+ B_a P_a with Z_a an orthonormal
    basis of ker(m_a^T) and P_a = I - 1 m_a^T / M_a, giving
    D_a 1 = 0 and m_a^T D_a = 0 exactly while PRESERVING residual
    minimality (the projector sandwich P A^+ B P does not)."""
    from ..perimeter.angle_constraint import (
        compute_angle_constraint_matrices_from_weights,
    )
    N = positions.shape[0]
    AB = torch.zeros(N, N, dtype=positions.dtype,
                     device=positions.device)
    stats, conds = [], []
    n_trunc = 0
    for lb in loop_labels.unique():
        sel = (loop_labels == lb).nonzero().flatten()
        A_a, B_a = compute_angle_constraint_matrices_from_weights(
            positions[sel], tangents[sel], angle_weights[sel],
            sigma, kernel)
        if consistency == "mean_zero":
            m_a = angle_weights[sel]
            n_a = sel.numel()
            P_a = torch.eye(n_a, dtype=A_a.dtype) \
                - torch.outer(torch.ones(n_a, dtype=A_a.dtype),
                              m_a) / m_a.sum()
            Qm, _ = torch.linalg.qr(m_a.unsqueeze(1),
                                    mode="complete")
            Z_a = Qm[:, 1:]                     # ker(m_a^T) basis
            sol = solve_angle_map(A_a @ Z_a, B_a @ P_a, rcond=rcond)
            D_a = Z_a @ sol.AB_solve
        else:
            sol = solve_angle_map(A_a, B_a, rcond=rcond)
            D_a = sol.AB_solve
        AB[sel.unsqueeze(1), sel.unsqueeze(0)] = D_a
        stats.append(dict(loop=int(lb), n=int(sel.numel()),
                          cond=sol.angle_cond,
                          n_truncated=sol.angle_n_truncated))
        conds.append(sol.angle_cond)
        n_trunc += sol.angle_n_truncated
    return AngleMapSolution(AB, max(conds), n_trunc,
                            block_stats=stats)


class NormalGraphParametrization:
    """Normal graph parametrization with volume conservation and angle constraint.

    Position update: x_k = x_k^{n-1} + s_k * n_k^{n-1}
    Angle update: Δθ_i = q_i × [A^†B @ s]_i  (coherence-suppressed)
    Displacement: s = Q @ y where y ∈ R^{N-1}

    Volume conservation: Σ s_i w_i = 0 (via Q).
    Angle constraint: A @ Δθ = B @ s, pre-solved as AB_solve = A^† B,
    then Δθ = (AB_solve @ s) * coherence for hidden boundary suppression.
    """

    def __init__(
        self,
        prev_positions: torch.Tensor,
        prev_normals: torch.Tensor,
        prev_angles: torch.Tensor,
        constraint_weights: torch.Tensor,
        A_angle: torch.Tensor,
        B_angle: torch.Tensor,
        coherence: torch.Tensor,
        rcond: float | None = None,
        q_suppress: bool = False,
        constraint_basis: torch.Tensor | None = None,
        angle_solution: "AngleMapSolution | None" = None,
    ):
        """Initialize parametrization.

        Args:
            prev_positions: Positions from previous step (N, 2).
            prev_normals: Unit normals from previous step (N, 2).
            prev_angles: Angles from previous step (N,).
            constraint_weights: Weights w_i for constraint Σ s_i w_i = 0.
            A_angle: (N, N) angle constraint matrix for Δθ.
            B_angle: (N, N) angle constraint matrix for s.
            coherence: (N,) coherence values from previous step for displacement scaling.
            rcond: SVD truncation threshold. None = eps * N (torch.linalg.lstsq style).
            q_suppress: If True, multiply displacements and delta_angles by coherence q_i.
            constraint_basis: (N, n_params) orthonormal basis of the admissible
                displacement space. When given it REPLACES the single-weight
                orthogonal complement (0C-5a: spectral componentwise
                compatibility, whose rows already contain global
                conservation). The legacy path is unchanged when None.
            angle_solution: pre-solved angle map A^dagger B. When given it is
                used verbatim (0C-5a.1: the spectral compatibility constraint
                and the evolution parametrization must use the SAME map, so
                the caller solves once and passes it to both).
        """
        from src.torch.transport.bem_wasserstein import _orthogonal_complement

        self.prev_positions = prev_positions.detach()
        self.prev_normals = prev_normals.detach()
        self.prev_angles = prev_angles.detach()
        self.coherence = coherence.detach()
        self.q_suppress = q_suppress
        self.N = prev_positions.shape[0]
        if constraint_basis is not None:
            self.Q = constraint_basis.detach()
        else:
            self.Q = _orthogonal_complement(constraint_weights)
        self.n_params = self.Q.shape[1]

        # Angle map AB_solve = A^† B: solve once here, or use the caller's.
        if angle_solution is None:
            angle_solution = solve_angle_map(A_angle, B_angle, rcond=rcond)
        self.AB_solve = angle_solution.AB_solve
        self.angle_cond = angle_solution.angle_cond
        self.angle_n_truncated = angle_solution.angle_n_truncated

    def init_params(self) -> torch.Tensor:
        """Create initial parameter vector y=0.

        Returns:
            params: (n_params,) parameter vector y (N-1 on the legacy path).
        """
        device, dtype = self.prev_positions.device, self.prev_positions.dtype
        return torch.zeros(self.n_params, device=device, dtype=dtype)

    def unpack_params(
        self,
        params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack parameter vector into displacements and delta_angles.

        Displacements are scaled by coherence so that hidden boundary points
        (q ≈ 0) have near-zero displacement: s_i = q_i × (Q @ y)_i.

        delta_angles is derived from displacements via the weak form angle constraint
        and then suppressed by coherence:
            Δθ_i = q_i × [A^†B @ s]_i

        This ensures hidden boundary points (q ≈ 0) have near-zero Δθ,
        preventing the normal rotation instability caused by ill-conditioned A.

        Returns:
            (displacements (N,), delta_angles (N,))
        """
        y = params
        s_raw = self.Q @ y
        displacements = self.coherence * s_raw if self.q_suppress else s_raw

        # Derive Δθ from s via pre-computed AB_solve = A^† B
        delta_angles = self.AB_solve @ displacements
        if self.q_suppress:
            delta_angles = delta_angles * self.coherence

        return displacements, delta_angles

    def reconstruct_varifold(
        self,
        params: torch.Tensor,
    ) -> OrientedPointCloudVarifold:
        """Reconstruct varifold from parameters."""
        displacements, delta_theta = self.unpack_params(params)
        positions = self.prev_positions + displacements.unsqueeze(-1) * self.prev_normals
        angles = self.prev_angles + delta_theta
        # Wrap angles to [-π, π) for numerical stability
        angles = wrap_angles(angles)
        return OrientedPointCloudVarifold(positions=positions, angles=angles)


class MMStepper:
    """Stateful MM stepper that reuses BEM objects across steps.

    Designed for torch.compile compatibility: self.objective is a stable method
    (fixed address) so Dynamo can cache the trace across MM steps.

    Usage (standalone):
        stepper = MMStepper(config)
        result = stepper.step(varifold)

    Usage (in solver loop — reuse across steps):
        stepper = MMStepper(config)
        for step in range(n_steps):
            result = stepper.step(current_varifold)
            current_varifold = result.varifold
        # After step(), stepper.mass_delta and stepper.mass_tau hold current values
    """

    def __init__(self, config: MMConfig):
        self.config = config
        self.bem_wasserstein = BEMWasserstein(
            method=config.bem_method,
            epsilon_scale=config.bem_epsilon_scale,
            analytic_gradient=config.wlin_analytic_gradient,
            svd_method=config.bem_svd_method,
            n_endpoints=config.bem_n_endpoints,
            trace_side=config.bem_trace_side,
            epsilon_mode=config.bem_epsilon_mode,
            epsilon_explicit=config.bem_epsilon,
            solver_mode=config.bem_solver_mode,
            component_rank=config.bem_component_rank,
            rank_mode=("deferred" if config.bem_rank_mode in (
                "state_machine", "adaptive_state_machine")
                       else config.bem_rank_mode),
            rank_max=config.bem_rank_max,
            rank_null_tol=config.bem_rank_null_tol,
            rank_gap_min=config.bem_rank_gap_min,
            operator_measure=config.bem_operator_measure,
        )
        # G2b grid metric backend. Guards are constructor-time so a
        # mis-combined config fails before any step runs:
        # - compile traces through torchmin; the grid metric's custom
        #   Functions + no_grad FFT/PCG internals are untested under
        #   Dynamo -> explicit rejection per plan.
        # - spectral_bordered builds the admissible space from the BIE
        #   near-nullspace; the grid backend builds it from phase
        #   components -- the two constraint machineries must never mix.
        self.grid_wasserstein = None
        self.bie_wasserstein = None
        self.metric_object = None
        # arm 5: persistent multiplicity amplitudes (spec-2.0 state
        # extension), updated by MMSolver after each committed step
        self.carrier_amplitudes = None
        # 0J guards: the perimeter-q correction is defined on the raw
        # oriented measure with a genuine coherence field
        if config.perimeter_q_mode != "full":
            if config.grid_carrier_amplitude_fade:
                raise ValueError(
                    "perimeter_q_mode correction requires "
                    "grid_carrier_amplitude_fade=False (loop graph and "
                    "r_l are defined on the raw oriented masses)")
            if config.use_unit_coherence:
                raise ValueError(
                    "perimeter_q_mode correction is meaningless with "
                    "use_unit_coherence=True")
            if (config.perimeter_q_mode == "self_renormalized"
                    and config.backend != "naive"):
                raise NotImplementedError(
                    "self_renormalized requires backend='naive' "
                    "(masked full kernel matrix); no keops fallback")
        # 0K guards (reviewer section 2): the loopwise mass mask must
        # not coexist with a bandwidth that itself responds to
        # cross-loop geometry -- (delta, tau) are scalar, resolved once
        # at t=0 and frozen for the trajectory.
        if config.mass_estimator == "loopwise_oriented_kde":
            if config.mass_bandwidth_type != "global":
                raise ValueError(
                    "loopwise_oriented_kde requires "
                    "mass_bandwidth_type='global' (per-point/adaptive "
                    "bandwidths are explicitly rejected, no fallback)")
            if config.mass_delta_adaptive:
                raise ValueError(
                    "loopwise_oriented_kde requires "
                    "mass_delta_adaptive=False: (delta, tau) are fixed "
                    "at t=0 for the whole trajectory so the bandwidth "
                    "cannot re-open a cross-loop geometry channel")
        # 0L-A guards: the geometric-area semantics need the certified
        # loop machinery (bulk rows on), and mixed semantics with a
        # geometric TARGET but current ROWS would constrain one volume
        # functional while filling toward another.
        if (config.grid_bulk_rows_functional == "geometric_area"
                or config.grid_volume_target_functional
                == "geometric_area"):
            if config.grid_bulk_rows_mode == "off":
                raise ValueError(
                    "geometric_area volume semantics require "
                    "grid_bulk_rows_mode != 'off' (certified loop "
                    "labels and bulk rows carry the polygon orders)")
            if config.grid_carrier_amplitude_fade:
                raise ValueError(
                    "geometric_area volume semantics are defined on "
                    "the raw carrier geometry; amplitude fade is not "
                    "supported")
        # 0L-theta guards
        if config.angle_constraint_measure == "raw_loopwise":
            if config.angle_constraint_scope != "loopwise":
                raise ValueError(
                    "angle_constraint_measure='raw_loopwise' requires "
                    "scope='loopwise' (a raw measure on the union "
                    "kernel is not one of the registered arms)")
            if config.mass_estimator != "loopwise_oriented_kde":
                raise ValueError(
                    "raw_loopwise angle measure requires the 0K "
                    "loopwise mass estimator (never a silent "
                    "position-KDE mass)")
            if config.displacement_q_suppress:
                raise ValueError(
                    "raw_loopwise angle measure with "
                    "displacement_q_suppress=True would re-inject a "
                    "nonuniform pointwise q at the OUTPUT of the "
                    "angle map -- rejected")
        if (config.angle_constraint_consistency == "mean_zero"
                and config.angle_constraint_scope != "loopwise"):
            raise ValueError(
                "angle_constraint_consistency='mean_zero' is defined "
                "on the loopwise direct-sum solve")
        if (config.grid_volume_target_functional == "geometric_area"
                and config.grid_bulk_rows_functional == "current"):
            raise ValueError(
                "grid_volume_target_functional='geometric_area' with "
                "current rows would fill toward the polygon area "
                "while constraining the oriented-current volume -- "
                "mixed semantics rejected (the reverse combination, "
                "geometric rows with current target, is the 0L-A2a "
                "causal experiment and IS allowed)")
        if config.metric_backend == "grid_poisson":
            from src.torch.transport.grid_wasserstein import (
                GridMetricConfig, GridWassersteinMetric)
            if config.compile:
                raise NotImplementedError(
                    "metric_backend='grid_poisson' with compile=True is "
                    "not supported (plan G3: explicit rejection)")
            if config.bem_solver_mode != "legacy_global_projection":
                raise NotImplementedError(
                    "metric_backend='grid_poisson' requires "
                    "bem_solver_mode='legacy_global_projection': the "
                    "grid backend derives admissibility from phase "
                    "components, not the BIE spectrum")
            grid_cfg = config.grid_metric or GridMetricConfig()
            if not isinstance(grid_cfg, GridMetricConfig):
                raise TypeError(
                    f"config.grid_metric must be a GridMetricConfig, "
                    f"got {type(grid_cfg).__name__}")
            self.grid_wasserstein = GridWassersteinMetric(grid_cfg)
        elif config.metric_backend == "bie":
            from src.torch.transport.bie_wasserstein import (
                BIEMetricConfig, BIEWassersteinMetric)
            if config.compile:
                raise NotImplementedError(
                    "metric_backend='bie' with compile=True is not "
                    "supported")
            if config.bem_solver_mode != "legacy_global_projection":
                raise NotImplementedError(
                    "metric_backend='bie' requires "
                    "bem_solver_mode='legacy_global_projection': the "
                    "BIE backend derives admissibility from the winding "
                    "components, not the BEM spectrum")
            bad = []
            if config.grid_bulk_rows_mode != "augment":
                bad.append("grid_bulk_rows_mode != 'augment'")
            if config.bem_operator_measure != "carrier":
                bad.append("bem_operator_measure != 'carrier'")
            if config.wlin_use_coherence_velocity:
                bad.append("wlin_use_coherence_velocity")
            if config.grid_contact_rows_mode == "bulk_projected":
                bad.append("grid_contact_rows_mode == 'bulk_projected'")
            for k in ("grid_vertical_dof", "grid_fossil_advection",
                      "grid_carrier_amplitude_fade",
                      "grid_phase_support_projection"):
                if getattr(config, k, False):
                    bad.append(k)
            if bad:
                raise ValueError(
                    "metric_backend='bie' incompatible with: "
                    + ", ".join(bad))
            bie_cfg = config.bie_metric or BIEMetricConfig()
            if not isinstance(bie_cfg, BIEMetricConfig):
                raise TypeError(
                    f"config.bie_metric must be a BIEMetricConfig, "
                    f"got {type(bie_cfg).__name__}")
            self.bie_wasserstein = BIEWassersteinMetric(bie_cfg)
        elif config.metric_backend != "bem":
            raise ValueError(
                f"unknown metric_backend {config.metric_backend!r}")
        # grid or BIE: the non-BEM metric object whose constraint rows
        # build the admissible basis (None on the BEM backend)
        self.metric_object = (self.grid_wasserstein
                              or self.bie_wasserstein)
        # The object objective() calls; identical (s, dtheta, dt)
        # signature on all backends.
        self.wasserstein_metric = (self.metric_object
                                   or self.bem_wasserstein)
        self._last_grid_snapshot = None
        self._grid_target_volume_initial = None
        # 0L-B2 persistent quotient bookkeeping: list of disjoint
        # particle masks, each = the UNION of the raw bulks merged at a
        # certified switch (bulk classes are identified, loops are not)
        self._quotient_bulk_groups: list = []
        self._contact_certificates_source = None
        self._quotient_meta: dict = {"quotient_at": None,
                                     "certificate_at_switch": None,
                                     "raw_bulk_pair_at_switch": None,
                                     "particle_count": None,
                                     # 0N-4: the State III contact
                                     # reference (identification scale,
                                     # thresholds and pair FROZEN at
                                     # the switch)
                                     "contact_reference": None}
        if config.grid_quotient_mode != "off":
            bad = []
            if config.remove_dead_points:
                bad.append("remove_dead_points")
            if getattr(config, "grid_phase_support_projection", False):
                bad.append("grid_phase_support_projection")
            if getattr(config, "grid_vertical_dof", False):
                bad.append("grid_vertical_dof")
            if getattr(config, "grid_fossil_advection", False):
                bad.append("grid_fossil_advection")
            if config.grid_bulk_rows_mode != "augment":
                bad.append("grid_bulk_rows_mode != 'augment'")
            if config.grid_contact_rows_mode != "aligned_prequotient":
                bad.append("grid_contact_rows_mode != "
                           "'aligned_prequotient'")
            for k in ("grid_quotient_eps_split", "grid_quotient_eps_switch_x",
                      "grid_quotient_eps_switch_theta",
                      "grid_quotient_eps_volume"):
                if getattr(config, k) is None:
                    bad.append(f"{k} not calibrated")
            if bad:
                raise ValueError(
                    "grid_quotient_mode requires a fixed particle set "
                    "and State-II rows; incompatible: " + ", ".join(bad))
        # 0C-5d-1.1: hysteretic working-rank tracking (state_machine mode);
        # initialised from the first clean evidence.
        self.rank_machine = None
        self.adaptive_rank_machine = None
        self._last_step_max_disp = 0.0
        self._last_cu_rowspace_angle_deg = None
        self._prev_spectral_C_full = None
        self._last_rank_decision = None
        self.param: NormalGraphParametrization | None = None
        self.fixed_coherence: torch.Tensor | None = None
        self.mass_delta: float | None = config.mass_delta
        self.mass_tau: float | None = config.mass_tau
        # Per-point bandwidth (set by _setup_step when per_point mode)
        self._mass_delta_for_kde: float | torch.Tensor | None = config.mass_delta
        self._mass_tau_for_kde: float | torch.Tensor | None = config.mass_tau

        # Tensor diagnostics updated by objective(), converted in step()
        self._last_perimeter: torch.Tensor | None = None
        self._last_wasserstein: torch.Tensor | None = None

        # Lazy-compiled objective
        self._compiled_objective = None

        # Optional optimizer callback (e.g. for trust-region diagnostics)
        self.optimizer_callback = None

    def set_mass_params(self, delta: float | None, tau: float | None):
        """Set mass parameters (called by MMSolver after auto-computation)."""
        self.mass_delta = delta
        self.mass_tau = tau

    def resolve_mass_scales(self, varifold) -> "MassScaleSnapshot":
        """The one place bandwidth policy lives. Mirrors the historical
        _setup_step block exactly, including the caching semantics: with
        mass_delta_adaptive=False the values are computed once on the
        first geometry and then frozen -- so a pure query on a later
        geometry returns the CACHED scales, which is precisely what the
        step will use. Pure: never mutates stepper state."""
        config = self.config
        cached_d, cached_t = self._mass_delta_for_kde, self._mass_tau_for_kde
        need_compute = config.mass_delta_adaptive or cached_d is None

        if config.mass_bandwidth_type == "per_point_knn":
            if need_compute or isinstance(cached_d, float):
                d_kde, t_kde = compute_per_point_bandwidths_knn(
                    varifold.positions.detach(),
                    k=config.mass_knn_k,
                    k_min=config.mass_k_min,
                    kernel=config.mass_kernel,
                )
            else:
                d_kde, t_kde = cached_d, cached_t
        elif config.mass_bandwidth_type == "per_point_abramson":
            if need_compute or isinstance(cached_d, float):
                d_kde, t_kde = compute_per_point_bandwidths_abramson(
                    varifold.positions.detach(),
                    k0=config.mass_knn_k,
                    k_min=config.mass_k_min,
                    kernel=config.mass_kernel,
                )
            else:
                d_kde, t_kde = cached_d, cached_t
        else:  # "global"
            if need_compute:
                if config.mass_delta is None or config.mass_tau is None:
                    delta, tau = compute_recommended_params(
                        varifold.positions.detach(),
                        k0=config.mass_knn_k,
                        kernel=config.mass_kernel,
                        k_min=config.mass_k_min,
                    )
                    d_kde = (config.mass_delta
                             if config.mass_delta is not None else delta)
                    t_kde = (config.mass_tau
                             if config.mass_tau is not None else tau)
                else:
                    d_kde = config.mass_delta
                    t_kde = config.mass_tau
            else:
                d_kde, t_kde = cached_d, cached_t

        def summary(x):
            return x.median().item() if isinstance(x, torch.Tensor) else x

        return MassScaleSnapshot(
            delta_for_kde=d_kde, tau_for_kde=t_kde,
            delta_summary=summary(d_kde), tau_summary=summary(t_kde))


    def _masses_for(self, positions, normals, loop_labels=None):
        """IIIc-0 estimator dispatch (default path bitwise-identical).
        0K: "loopwise_oriented_kde" additionally requires the
        source-frozen certified loop labels -- passing None is an
        error, never a silent fallback to the unmasked estimator."""
        if self.config.mass_estimator == "loopwise_oriented_kde":
            if loop_labels is None:
                raise RuntimeError(
                    "loopwise_oriented_kde needs source-frozen loop "
                    "labels; _setup_step must resolve them before any "
                    "mass evaluation")
            from ..oriented_varifold.mass import (
                compute_masses_oriented_loopwise,
            )
            return compute_masses_oriented_loopwise(
                positions, normals, loop_labels,
                self._mass_delta_for_kde, self._mass_tau_for_kde,
                self.config.mass_kernel)
        if self.config.mass_estimator == "oriented_kde":
            from ..oriented_varifold.mass import compute_masses_oriented
            return compute_masses_oriented(
                positions, normals, self._mass_delta_for_kde,
                self._mass_tau_for_kde, self.config.mass_kernel)
        return compute_masses(
            positions, self._mass_delta_for_kde,
            self._mass_tau_for_kde, self.config.mass_kernel)

    def _certified_loop_labels(self, positions, normals, masses_raw):
        """0J: boundary-loop labels from the co-oriented proximity
        graph at ell = grid_incidence_ell_factor * median(m~), with the
        graph-scale certificate: the partition must be co-membership
        identical at ell/median(m~) in {3.5, 4.0, 4.5} (fail-closed --
        the correction consumes the partition directly, so it is more
        sensitive to misclassification than the volume rows).
        Delegates to the shared 0K implementation so the stepper, the
        driver, and the mass resolver can never drift apart."""
        from ..oriented_varifold.loopwise_mass import (
            certified_loop_labels,
        )
        return certified_loop_labels(
            positions, normals, float(masses_raw.median()),
            graph_scale_factors=(3.5, 4.0, 4.5),
            ell_factor=self.config.grid_incidence_ell_factor)

    def _corrected_perimeter_q(self, varifold, masses_raw, sigma):
        """0J: the perimeter-energy weight under perimeter_q_mode.

        self_renormalized: q^WB = r_l q^self with
        r_l = (sum m q^full)_l / (sum m q^self)_l -- the UNIQUE
        loopwise scalar preserving the self profile and the source
        energy. Fail-closed on (a) degenerate denominator
        (B_l <= 100 eps M_l -- never clipped) and (b) r_l outside
        [0, 1+1e-9], which is a SCOPE certificate for the
        antiparallel-contact regime, not a general theorem
        (co-oriented nearby loops can legitimately give r > 1)."""
        config = self.config
        labels = self._loop_labels_last
        q_full = self.fixed_coherence
        m = masses_raw
        q_out = torch.empty_like(q_full)
        stats = {"min_qbar_self": None, "min_r": None, "max_r": None,
                 "cross_max": None, "mode": config.perimeter_q_mode}
        if config.perimeter_q_mode == "loop_mean":
            for lb in labels.unique():
                sel = labels == lb
                q_out[sel] = (m[sel] * q_full[sel]).sum() / m[sel].sum()
            self._qmode_stats = stats
            return q_out
        from ..perimeter.coherence_perimeter import (
            compute_coherence_loopwise,
        )
        q_self, cross_max = compute_coherence_loopwise(
            varifold.positions.detach(), varifold.normals.detach(),
            m, sigma, config.perimeter_kernel, loop_labels=labels)
        eps = torch.finfo(m.dtype).eps
        if config.perimeter_q_mode in ("contact_complex_renormalized",
                                       "contact_complex_tb"):
            return self._contact_complex_q(varifold, m, sigma, labels,
                                           q_full, q_self, cross_max,
                                           eps, stats)
        rs, qbars = [], []
        r_pp = torch.ones_like(q_full)
        for lb in labels.unique():
            sel = labels == lb
            A = float((m[sel] * q_full[sel]).sum())
            B = float((m[sel] * q_self[sel]).sum())
            M = float(m[sel].sum())
            qbars.append(B / M)
            if B <= 100.0 * eps * M:
                raise RuntimeError(
                    f"self-coherence denominator degenerate on loop "
                    f"{int(lb)} (B={B:.3e}, M={M:.3e}) -- fail-closed, "
                    f"no clipping")
            r = A / B
            if not (0.0 <= r <= 1.0 + 1e-9):
                raise RuntimeError(
                    f"r_l = {r:.6f} outside [0, 1+1e-9] on loop "
                    f"{int(lb)}: outside the antiparallel-contact "
                    f"scope certificate")
            rs.append(r)
            q_out[sel] = r * q_self[sel]
            r_pp[sel] = r
        stats.update(min_qbar_self=min(qbars), min_r=min(rs),
                     max_r=max(rs), cross_max=cross_max)
        self._qmode_stats = stats
        # 0L R2 (reviewer): SOURCE-FROZEN particle-aligned r_{l(i)}
        # tensor -- consumers (the r_loop redistribution q-policy)
        # read it by particle index, NEVER by re-matching label IDs
        # (a re-certified partition may renumber its labels; particle
        # ordering is preserved between the MM step and the
        # redistribution, which runs before any deletion).
        self.perimeter_r_per_particle = r_pp
        return q_out

    def _contact_complex_q(self, varifold, m, sigma, labels, q_full,
                           q_self, cross_max, eps, stats):
        """L0J q^CC: side-wise renormalization on the sigma-scale
        ENERGY contact complex (reviewer 2026-08-21).

        SEPARATION OF CONCERNS (reviewer blocker 1): this fixes the
        PERIMETER energy template only. `perimeter_r_per_particle`
        (consumed by the redistribution 'r_loop' policy) keeps its
        established loopwise-GLOBAL semantics -- constant per loop, a
        pure pseudo-time rescale that never changes the redistribution
        path. The side-wise contact factors live in the SEPARATE
        `perimeter_contact_r_per_particle` (telemetry / future
        'r_contact' arm, never fed to redistribution here)."""
        from ..perimeter.contact_complex import contact_complex_q
        pos = varifold.positions.detach()
        nor = varifold.normals.detach()
        res = contact_complex_q(pos, nor, m, labels, sigma,
                                self.config.perimeter_kernel,
                                q_full, q_self)
        q_out, r_cc, side_tel = res["q_cc"], res["r_cc"], res["sides"]
        cx = res["cx"]
        # GLOBAL loopwise r_l for the redistribution policy (unchanged
        # semantics; same guards as self_renormalized)
        r_pp = torch.ones_like(q_full)
        for lb in labels.unique():
            sel = labels == lb
            A = float((m[sel] * q_full[sel]).sum())
            B = float((m[sel] * q_self[sel]).sum())
            M = float(m[sel].sum())
            if B <= 100.0 * eps * M:
                raise RuntimeError(
                    f"self-coherence denominator degenerate on loop "
                    f"{int(lb)} (B={B:.3e}) -- fail-closed")
            r = A / B
            if not (0.0 <= r <= 1.0 + 1e-9):
                raise RuntimeError(
                    f"global r_l = {r:.6f} outside [0, 1+1e-9] on "
                    f"loop {int(lb)}")
            r_pp[sel] = r
        if self.config.perimeter_q_mode == "contact_complex_tb":
            q_out = self._tb_correction(pos, nor, m, labels, q_self,
                                        q_full, res, stats)
        stats.update(cross_max=cross_max,
                     n_complexes=cx["n_complexes"],
                     contact_sides=side_tel,
                     min_r=(min(t["r"] for t in side_tel)
                            if side_tel else None),
                     max_r=(max(t["r"] for t in side_tel)
                            if side_tel else None))
        self._qmode_stats = stats
        self.perimeter_r_per_particle = r_pp
        self.perimeter_contact_r_per_particle = r_cc
        return q_out

    def _tb_correction(self, pos, nor, m, labels, q_self, q_full,
                       res, stats):
        """L0J-M candidate A: replace each contact side's constant
        factor by the translation-balanced minimum-change factors
        (eq 5.3): energy row e = m q^self on the side, translation
        rows f_k,i = q^self_i Dm_i[xi_{l,k}] with the whole-loop
        normal-graph translation mode s = a_k . n (positions-only FD;
        the loopwise 0K mass of the OTHER loop does not respond, and
        the dtheta share is quantified separately in the M0 audit),
        RHS = [E_I(q^full); -F_out] with F_out the outside-of-side
        loop contribution. Fail-closed on rank/feasibility."""
        from ..oriented_varifold.loopwise_mass import (
            resolve_loopwise_oriented_mass_source,
        )
        from ..perimeter.contact_complex import translation_balanced_r
        cfg = self.config

        def m_of(p2):
            return resolve_loopwise_oriented_mass_source(
                p2, nor, self._mass_delta_for_kde,
                self._mass_tau_for_kde, cfg.mass_kernel,
                ell_factor=cfg.grid_incidence_ell_factor).m_loop
        t = 1e-4 * float(m.median())
        q_out = res["q_cc"].clone()
        tb_tel = []
        for comp in res["cx"]["complexes"]:
            for side in comp.sides:
                sel_loop = labels == side.loop
                inside = side.mask
                rows_f, rhs_f = [], []
                for k in (0, 1):
                    s_t = torch.zeros_like(m)
                    s_t[sel_loop] = nor[sel_loop, k]
                    dpos = s_t.unsqueeze(1) * nor
                    Dm = (m_of(pos + t * dpos)
                          - m_of(pos - t * dpos)) / (2 * t)
                    f_full = q_self * Dm
                    rows_f.append(f_full[inside])
                    rhs_f.append(-float(
                        f_full[sel_loop & ~inside].sum()))
                A = torch.stack([(m * q_self)[inside],
                                 rows_f[0], rows_f[1]])
                b = torch.tensor([float((m * q_full)[inside].sum()),
                                  rhs_f[0], rhs_f[1]], dtype=m.dtype)
                sol = translation_balanced_r(
                    A, b, r0=res["r_cc"][inside],
                    w=(m * q_self * q_self)[inside],
                    hi=1.0 / q_self[inside].clamp_min(1e-12))
                tb_tel.append({k_: v for k_, v in sol.items()
                               if k_ != "r"} | {"loop": side.loop})
                if sol["r"] is None:
                    raise RuntimeError(
                        f"translation-balanced correction failed "
                        f"({sol['status']}) on loop {side.loop} -- "
                        "fail-closed")
                q_out[inside] = sol["r"] * q_self[inside]
        stats["tb_sides"] = tb_tel
        return q_out

    def _setup_step(self, varifold: OrientedPointCloudVarifold):
        """Prepare per-step state: parametrization, coherence, BEM cache.

        Corresponds to the setup portion of the old mm_step().
        """
        from src.torch.perimeter.coherence_perimeter import compute_recommended_sigma

        config = self.config

        # Compute mass parameters via the SHARED resolver (0E-wiring.2:
        # the diagnostics timeline must see exactly the scales this step
        # uses, including cached non-adaptive values and per-point
        # bandwidth tensors -- a separate re-implementation drifted).
        snap = self.resolve_mass_scales(varifold)
        self.mass_delta = snap.delta_summary
        self.mass_tau = snap.tau_summary
        self._mass_delta_for_kde = snap.delta_for_kde
        self._mass_tau_for_kde = snap.tau_for_kde

        # Compute sigma for perimeter and coherence
        if config.perimeter_sigma is None:
            sigma = compute_recommended_sigma(varifold.positions, config.perimeter_c_sigma)
        else:
            sigma = config.perimeter_sigma
        self._sigma = sigma

        # Compute masses and coherence from PREVIOUS step (FIXED during optimization)
        prev_positions = varifold.positions.detach()
        prev_normals = varifold.normals.detach()
        # 0K: the loopwise mass estimator resolves the certified loop
        # labels FIRST (two-stage bootstrap certificate: m_pre ->
        # l_pre -> m_loop -> l_post, co-membership equality required),
        # then every downstream consumer -- coherence, r_l, effective
        # masses, q^2 m constraint weights, angle matrices, grid
        # carrier, volume target, incidence/bulk rows -- reads the
        # loopwise masses through the single prev_masses variable.
        # The non-loopwise path is unchanged (labels, when needed by
        # 0J/0F, are computed after coherence exactly as before).
        self._last_mass_loop_snapshot = None
        loop_labels = None
        if config.mass_estimator == "loopwise_oriented_kde":
            from ..oriented_varifold.loopwise_mass import (
                resolve_loopwise_oriented_mass_source,
            )
            mass_res = resolve_loopwise_oriented_mass_source(
                prev_positions, prev_normals,
                self._mass_delta_for_kde, self._mass_tau_for_kde,
                config.mass_kernel,
                ell_factor=config.grid_incidence_ell_factor)
            prev_masses = mass_res.m_loop
            loop_labels = mass_res.loop_labels_pre
            self._last_mass_loop_snapshot = mass_res
        else:
            prev_masses = self._masses_for(prev_positions, prev_normals)
        self.fixed_masses = prev_masses

        if config.use_unit_coherence:
            self.fixed_coherence = torch.ones_like(prev_masses)
        else:
            self.fixed_coherence = compute_coherence(
                varifold, prev_masses, sigma, config.perimeter_kernel,
                backend=config.backend,
            )

        # 0J: certified loop labels, computed ONCE per step from the
        # RAW oriented masses (never the amplitude-attenuated grid
        # measure) and shared by the perimeter-q correction, the
        # incidence rows, and the 0L-R redistribution geometry. Under
        # 0K the labels were already resolved (and certificated
        # twice) before the masses -- reuse them, never recompute
        # within the step. 0L-R (reviewer section 6): the supply
        # condition is UNIFIED here -- the redistribution features
        # must not be hidden-coupled to the bulk-rows flag.
        needs_loop_labels = (
            config.perimeter_q_mode != "full"
            or config.grid_bulk_rows_mode != "off"
            or config.redistribution_density_scope == "loopwise"
            or config.redistribution_tangent_source == "local_pca"
            or config.redistribution_normal_retraction
            or config.redistribution_curvature_scope == "loopwise"
            or config.angle_constraint_scope == "loopwise")
        if loop_labels is None and needs_loop_labels:
            loop_labels = self._certified_loop_labels(
                prev_positions, prev_normals, prev_masses)
        self._loop_labels_last = loop_labels

        # 0J: the perimeter energy weight. Default = fixed_coherence
        # (bitwise). "self_renormalized": q^WB = r_l q^self -- the
        # UNIQUE loopwise-scalar correction that preserves the
        # self-coherence profile AND the source energy
        # (P_WB(X|X) = P_full(X) by construction); consequently
        # D^j P_WB = r_l D^j P_self for j = 1, 2. "loop_mean" is the
        # circular-limit control.
        self.perimeter_coherence = self.fixed_coherence
        self._qmode_stats = None
        self.perimeter_r_per_particle = None
        self.perimeter_contact_r_per_particle = None
        if config.perimeter_q_mode != "full":
            self.perimeter_coherence = self._corrected_perimeter_q(
                varifold, prev_masses, sigma)

        # effective_masses = masses * coherence
        effective_masses = prev_masses * self.fixed_coherence
        self.fixed_effective_masses = effective_masses

        # Constraint weights for volume conservation: w_i = q_i² * m_i
        constraint_weights = self.fixed_coherence * self.fixed_coherence * prev_masses

        # Compute weak form angle constraint matrices
        prev_tangents = torch.stack(
            [-prev_normals[:, 1], prev_normals[:, 0]], dim=1
        )
        # Angle bandwidth decoupled from the perimeter sigma (0A): with the
        # legacy None it follows sigma, so a sigma-refinement study would
        # silently change the angle-transport regularisation as well.
        sigma_angle = (config.angle_sigma
                       if config.angle_sigma is not None else sigma)
        self._sigma_angle = sigma_angle
        # 0L-theta: the measure weight of the weak form, stated
        # explicitly (visible_full = m q^full, historical;
        # raw_loopwise = the 0K loopwise mass alone)
        if config.angle_constraint_measure == "raw_loopwise":
            angle_weights = prev_masses
        else:
            angle_weights = prev_masses * self.fixed_coherence
        if config.angle_constraint_scope == "loopwise":
            from ..perimeter.angle_constraint import (
                compute_angle_constraint_matrices_from_weights,
            )
            # assembled block matrices (same-loop mask) for the
            # stored A/B; the SOLVE happens loop-by-loop below
            A_angle, B_angle = \
                compute_angle_constraint_matrices_from_weights(
                    prev_positions, prev_tangents, angle_weights,
                    sigma_angle, config.perimeter_kernel)
            same = (loop_labels.unsqueeze(1)
                    == loop_labels.unsqueeze(0)).to(A_angle.dtype)
            A_angle = A_angle * same
            B_angle = B_angle * same
        else:
            A_angle, B_angle = compute_angle_constraint_matrices(
                positions=prev_positions,
                tangents=prev_tangents,
                masses=prev_masses,
                coherence=self.fixed_coherence,
                sigma=sigma_angle,
                kernel=config.perimeter_kernel,
            )

        # Update BEM Wasserstein cache (same object, no reallocation).
        # NOTE: in spectral mode this must run BEFORE the parametrization,
        # because the admissible displacement space is built from the BEM
        # operator's left near-nullspace. In grid mode the BEM object is
        # never set up -- the grid metric setup below plays this role.
        if self.metric_object is None:
            self.bem_wasserstein.setup_for_step(
                positions=prev_positions,
                normals=prev_normals,
                effective_masses=effective_masses,
                sigma=sigma,
                c_sigma=config.perimeter_c_sigma,
                carrier_masses=prev_masses,
            )
            self.bem_wasserstein.setup_coherence(
                self.fixed_coherence,
                use_coherence_velocity=config.wlin_use_coherence_velocity,
            )

        # Solve the angle map ONCE; the spectral constraint and the
        # parametrization must not each run their own pseudo-inverse.
        if config.angle_constraint_scope == "loopwise":
            angle_solution = solve_angle_map_blocks(
                prev_positions, prev_tangents, angle_weights,
                loop_labels, sigma_angle, config.perimeter_kernel,
                rcond=config.angle_constraint_rcond,
                consistency=config.angle_constraint_consistency)
        else:
            angle_solution = solve_angle_map(
                A_angle, B_angle, rcond=config.angle_constraint_rcond)

        constraint_basis = None
        self._last_grid_snapshot = None
        if self.metric_object is not None:
            constraint_basis = self._setup_grid_metric(
                varifold, prev_masses, angle_solution)
        elif config.bem_solver_mode == "spectral_bordered":
            if config.bem_rank_mode == "state_machine":
                self._apply_rank_state_machine()
            elif config.bem_rank_mode == "adaptive_state_machine":
                self._apply_adaptive_rank_machine()
            constraint_basis = self._spectral_constraint_basis(
                angle_solution.AB_solve, config)

        # Create parametrization with volume conservation and angle constraint
        self.param = NormalGraphParametrization(
            prev_positions=varifold.positions,
            prev_normals=varifold.normals,
            prev_angles=varifold.angles,
            constraint_weights=constraint_weights,
            A_angle=A_angle,
            B_angle=B_angle,
            coherence=self.fixed_coherence,
            rcond=config.angle_constraint_rcond,
            q_suppress=config.displacement_q_suppress,
            constraint_basis=constraint_basis,
            angle_solution=angle_solution,
        )

        # Observational BEM-setup telemetry (closure patch 2). Detached
        # clones only; must not touch any tensor that requires grad.
        self._last_bem_snapshot = None
        if config.bem_setup_telemetry:
            bem_fields = self.bem_wasserstein.setup_snapshot_fields()
            if bem_fields is not None:
                def _c(t):
                    return t.detach().clone().requires_grad_(False)
                self._last_bem_snapshot = BEMSetupSnapshot(
                    positions=_c(prev_positions),
                    carrier_masses_m=_c(prev_masses),
                    coherence_q=_c(self.fixed_coherence),
                    canonical_qm=_c(effective_masses),
                    canonical_q2m=_c(constraint_weights),
                    volume_constraint_weights_used=_c(constraint_weights),
                    volume_constraint_policy="q2m (parametrization)",
                    delta_used=float(self.mass_delta),
                    tau_used=float(self.mass_tau),
                    perimeter_sigma_used=float(sigma),
                    angle_sigma_used=float(sigma_angle),
                    **bem_fields,
                )

        # Reset diagnostics
        self._last_perimeter = None
        self._last_wasserstein = None

    def _setup_grid_metric(self, varifold, prev_masses, angle_solution):
        """G2b: freeze the grid metric for this step and build the
        admissible basis from the phase-component compatibility rows.

        Runs where the spectral branch would (BEFORE the
        parametrization). The volume target is label-free via the
        divergence theorem on the oriented current,
        V_hat = 0.5 sum_i m_i x_i . n_i, so the filling projection and
        the admissible space consume the SAME discrete volume. Support
        components are frozen for the whole step (topology changes are
        events -- the merger handler's domain, never optimizer moves).
        """
        from src.torch.transport.boundary_flux_grid import (
            BoundaryFluxPolicies)
        from src.torch.transport.grid_wasserstein import (
            compose_constraint_basis)

        from src.torch.transport.weighted_poisson import (
            connected_components)

        config = self.config
        pos = varifold.positions.detach()
        nor = varifold.normals.detach()
        # arm 5: persistent amplitudes attenuate the carrier masses of
        # the GRID layer only (filling, flux, volume rows). The
        # divergence-volume target uses the SAME attenuated measure so
        # the filling projection and the admissible space stay
        # consistent; under initial_fixed the target is pinned to the
        # step-0 value anyway (amplitudes start at 1).
        m_grid = prev_masses
        amps = getattr(self, "carrier_amplitudes", None)
        if (config.grid_carrier_amplitude_fade and amps is not None
                and amps.shape[0] == prev_masses.shape[0]):
            m_grid = prev_masses * amps
        # 0L-A: certified polygon orders, computed once per step on
        # the source state from the certified loop labels (fail-closed
        # LoopOrderError); consumed by the geometric volume target
        # and/or the geometric bulk rows.
        self._loop_orders_last = None
        if (config.grid_bulk_rows_functional == "geometric_area"
                or config.grid_volume_target_functional
                == "geometric_area"
                or (config.grid_bulk_first_moment_rows
                    and config.grid_bulk_first_moment_rows_form
                    == "polygon")):
            from ..transport.loop_geometry import (
                certified_cyclic_order,
            )
            if self._loop_labels_last is None:
                raise RuntimeError(
                    "geometric_area semantics require the certified "
                    "loop labels (set in _setup_step)")
            self._loop_orders_last = certified_cyclic_order(
                pos, self._loop_labels_last, nor, prev_masses,
                float(prev_masses.median()))
        if config.grid_volume_target_functional == "geometric_area":
            from ..transport.loop_geometry import polygon_area
            target_current = float(sum(
                float(polygon_area(pos, lo.order))
                for lo in self._loop_orders_last.values()))
        else:
            target_current = float(
                0.5 * (m_grid * (pos * nor).sum(dim=-1)).sum())
        if target_current <= 0.0:
            raise RuntimeError(
                f"divergence-theorem volume target is non-positive "
                f"({target_current:.3e}) -- normals are not consistently "
                f"outward; the phase filling cannot proceed")
        if self._grid_target_volume_initial is None:
            self._grid_target_volume_initial = target_current
        target_initial = self._grid_target_volume_initial
        target_used = (target_initial
                       if config.grid_volume_target_mode == "initial_fixed"
                       else target_current)
        policies = BoundaryFluxPolicies(
            operator_measure=config.bem_operator_measure,
            use_coherence_velocity=config.wlin_use_coherence_velocity,
            n_endpoints=config.bem_n_endpoints,
        )
        gm = self.metric_object
        is_bie = self.bie_wasserstein is not None
        # 0F/0L-B0: the raw bulk partition is a pure particle-side
        # object -- computed ONCE here (before the grid setup) so the
        # diagnostic gauge hold can seed on it; bitwise the same values
        # _augment_constraint_rows used to compute itself.
        bulk_part = None
        gauge_seed = None
        if config.grid_bulk_rows_mode != "off":
            bulk_part = self._bulk_partition(pos, nor, m_grid)
            if config.grid_gauge_hold_bulk_seeded \
                    and bulk_part["bulk_of_loop"] is not None:
                gauge_seed = (pos, bulk_part["bulk_pp"],
                              bulk_part["n_bulk"])
        if is_bie:
            if bulk_part is None or bulk_part["bulk_of_loop"] is None:
                from ..transport.incidence import IncidenceUnstableError
                raise IncidenceUnstableError(
                    "BIE metric: winding incidence certificate failed "
                    f"(worst integer margin "
                    f"{bulk_part['margin'] if bulk_part else 'n/a'})")
            gm.setup_for_step(varifold, m_grid, self.fixed_coherence,
                              target_used, policies,
                              bulk_partition=bulk_part)
        else:
            gm.setup_for_step(varifold, m_grid, self.fixed_coherence,
                              target_used, policies,
                              gauge_seed=gauge_seed)
        # compat_labels == phase labels in "operating" mode; in
        # "conservative_sweep" mode they keep freshly-bridged blobs
        # separate through the merger ambiguity window (soft-mode
        # deflation -- volume is then conserved PER BLOB there).
        # BIE: particle labels (metric components), exact flux rows.
        rows = gm.flux.component_rows(
            gm.compat_labels, None if is_bie else gm.phase.cell_area)
        # 0F: incidence augmentation. gm persists across steps, so the
        # forward-solve mode must be re-pinned every setup.
        gm.forward_solve = "fail_closed"
        self._bulk_rows_last = None
        self._bulk_rows_shadow = None
        self._contact_rows_last = None
        self._contact_pair_context = None
        aug = {"mode": config.grid_bulk_rows_mode, "fired": False,
               "n_bulk": None, "r_grid": rows.shape[0], "r_bulk": None,
               "r_stack": None, "d_compat": None, "margin": None,
               "relation": None, "relation_base": None,
               "eps_row": None, "eps_row_composed": None}
        rows_used = rows
        self._contact_certificates_source = None
        if config.grid_bulk_rows_mode != "off":
            rows_used = self._augment_constraint_rows(
                pos, nor, m_grid, gm, rows, aug, bulk_part,
                angle_solution.AB_solve)
            # 0L-B1/B2: contact certificate on the SOURCE state with the
            # step's own bookkeeping (State-II candidates only)
            ctx = getattr(self, "_contact_pair_context", None)
            if ctx is not None and self._loop_labels_last is not None:
                from ..transport.contact_certificate import (
                    ContactCertificateError,
                    evaluate_all_candidates,
                )
                try:
                    self._contact_certificates_source = \
                        evaluate_all_candidates(
                            pos, nor, prev_masses, self._loop_labels_last,
                            ctx["loop_to_grid_component"],
                            ctx["loop_to_raw_bulk"],
                            self._sigma if getattr(self, "_sigma", None)
                            else config.perimeter_sigma)
                except ContactCertificateError as ex:
                    if config.grid_quotient_mode != "off":
                        raise
                    self._contact_certificates_source = {
                        "error": str(ex)[:160]}
        frozen = None
        if config.grid_freeze_dead_dof:
            frozen = self.fixed_coherence < config.grid_dead_dof_q
        if is_bie and gm.masked is not None and bool(gm.masked.any()):
            # cancelling pair: excluded from the operator, held fixed
            frozen = gm.masked if frozen is None else (frozen | gm.masked)
        self._rows_used_last = rows_used        # telemetry (parity)
        composed = compose_constraint_basis(
            rows_used, angle_solution.AB_solve, self.fixed_coherence,
            config.displacement_q_suppress, frozen_mask=frozen,
            drop_dependent_under_freeze=is_bie)
        basis = composed.basis
        sv = composed.singular_values
        self._n_frozen_dof = (int(frozen.sum())
                              if frozen is not None else 0)
        n_rows_eff = (composed.n_rows_effective if is_bie
                      else rows_used.shape[0])
        if is_bie:
            every = max(1, int(gm.config.definiteness_check_every))
            if gm.merged_pairs or (gm._n_setups - 1) % every == 0:
                gm.check_definiteness(
                    basis, angle_solution.AB_solve, self.fixed_coherence,
                    config.displacement_q_suppress, config.time_step,
                    step_index=gm._n_setups)

        # arm V2: precompute the vertical (removal) block on the SAME
        # frozen state as the rest of the step's metric
        self._vertical = None
        self._vertical_offset = None
        if config.grid_vertical_dof:
            self._setup_vertical_dof(varifold, prev_masses, gm,
                                     amps if (amps is not None
                                              and amps.shape[0]
                                              == prev_masses.shape[0])
                                     else None)

        if is_bie:
            bf = gm.setup_snapshot_fields() or {}
            gfields = dict(
                n_components=gm.n_compat_components,
                component_active_support_areas=(),
                component_phase_masses=(),
                phase_grid_volume=float("nan"),
                reconstruction_volume_error=0.0,
                carrier_renormalization=1.0,
                projection_l2_relative=0.0,
                speck_mass_dropped=0.0,
                fill_epsilon=None, grid_dx=None, grid_shape=None,
                component_counts_at_thresholds=(),
                min_component_phase_mass=float("nan"),
                min_component_active_area=float("nan"),
                box_boundary_margin_over_eps=None,
                quadratic_path="bie_block_quadratic",
                solver_backend="dense_lu",
                metric_backend="bie",
                bie_sigma_tail=bf.get("bie_sigma_tail"),
                bie_null_level=bf.get("bie_null_level"),
                bie_lambda_min=bf.get("bie_lambda_min"),
                bie_lambda_max=bf.get("bie_lambda_max"),
                bie_n_masked=bf.get("bie_n_masked"),
                bie_merged_pairs=bf.get("bie_merged_pairs"),
                bie_epsilon=bf.get("bie_epsilon"),
                bie_g_bridge=bf.get("bie_g_bridge"),
                bie_block_sizes=bf.get("bie_block_sizes"),
                bie_setup_wall_seconds=bf.get("bie_setup_wall_seconds"),
            )
        else:
            pg = gm.phase
            thr = gm.config.phase.support_threshold
            comp_areas, comp_masses = [], []
            for c in range(pg.n_components):
                mask = pg.component_labels == c
                comp_areas.append(float(mask.sum()) * pg.cell_area)
                comp_masses.append(
                    float(pg.rho_metric[mask].sum()) * pg.cell_area)
            counts = tuple(
                int(connected_components(pg.rho_metric > t)[1])
                for t in (thr / 3.0, thr, 3.0 * thr))
            active = gm.poisson.active
            H, W = active.shape
            ri = active.any(dim=1).nonzero().flatten()
            ci = active.any(dim=0).nonzero().flatten()
            margin_cells = min(int(ri.min()), H - 1 - int(ri.max()),
                               int(ci.min()), W - 1 - int(ci.max()))
            gfields = dict(
                n_components=pg.n_components,
                component_active_support_areas=tuple(comp_areas),
                component_phase_masses=tuple(comp_masses),
                phase_grid_volume=(float(pg.rho_metric.sum())
                                   * pg.cell_area),
                reconstruction_volume_error=pg.volume_error,
                carrier_renormalization=pg.carrier_renormalization,
                projection_l2_relative=pg.projection_l2_relative,
                speck_mass_dropped=pg.speck_mass_dropped,
                fill_epsilon=gm.config.phase.fill_epsilon,
                grid_dx=pg.dx,
                grid_shape=tuple(gm.config.phase.grid_shape),
                component_counts_at_thresholds=counts,
                min_component_phase_mass=min(comp_masses),
                min_component_active_area=min(comp_areas),
                box_boundary_margin_over_eps=(
                    margin_cells * pg.dx / gm.config.phase.fill_epsilon),
                quadratic_path=gm.config.quadratic_path,
                solver_backend=gm.config.poisson.solver_backend,
                metric_backend="grid_poisson",
            )
        self._last_grid_snapshot = GridMetricSetupSnapshot(
            target_volume_current=target_current,
            target_volume_used=target_used,
            target_volume_initial=target_initial,
            target_volume_drift_relative=(
                (target_current - target_initial) / target_initial),
            constraint_rank=n_rows_eff,
            n_params=basis.shape[1],
            constraint_row_singular_values=tuple(
                float(x) for x in sv),
            constraint_row_condition=float(sv[0] / sv[-1]),
            operator_measure_policy=policies.operator_measure,
            velocity_policy=("coherence_q" if
                             policies.use_coherence_velocity else "unit"),
            endpoints_per_particle=policies.n_endpoints,
            perimeter_q_stats=self._qmode_stats,
            bulk_rows_mode=aug["mode"],
            augmentation_fired=aug["fired"],
            n_bulk_components=aug["n_bulk"],
            r_grid=aug["r_grid"],
            r_bulk=aug["r_bulk"],
            r_stack=aug["r_stack"],
            delta_compat=aug["d_compat"],
            incidence_margin=aug["margin"],
            partition_relation_grid_bulk=aug["relation"],
            contact_rows_mode=config.grid_contact_rows_mode,
            eps_row=aug["eps_row"],
            eps_row_composed=aug["eps_row_composed"],
            gauge_hold_fired=gm.gauge_hold_fired,
            gamma_cross=gm.gamma_cross,
            n_compat_base=gm.n_compat_base,
            n_compat_used=gm.n_compat_components,
            partition_relation_base=aug["relation_base"],
            quotient_classes=aug.get("quotient_classes"),
            n_quotient_groups=len(self._quotient_bulk_groups),
            n_frozen_dof=self._n_frozen_dof,
            **gfields,
        )
        return basis

    # ---- 0L-B2 quotient bookkeeping ------------------------------
    def _quotient_map(self, bulk_pp, n_bulk):
        """Q_quot (B_quot x B_raw, 0/1) from the persistent particle
        groups. Every raw bulk must lie ENTIRELY inside one group or
        entirely outside all groups (fail-closed otherwise: a split or
        mixed bulk means the bookkeeping no longer describes the
        current partition). Returns (None, identity classes) when no
        group is committed."""
        if not self._quotient_bulk_groups:
            return None, list(range(n_bulk))
        N = bulk_pp.shape[0]
        group_pp = torch.full((N,), -1, dtype=torch.long,
                              device=bulk_pp.device)
        for g, mask in enumerate(self._quotient_bulk_groups):
            if mask.shape[0] != N:
                raise RuntimeError(
                    "quotient group mask length != particle count "
                    f"({mask.shape[0]} vs {N})")
            group_pp[mask.to(bulk_pp.device)] = g
        classes = []
        next_free = len(self._quotient_bulk_groups)
        for b in range(n_bulk):
            ids = group_pp[bulk_pp == b].unique().tolist()
            if len(ids) != 1:
                raise RuntimeError(
                    f"raw bulk {b} straddles quotient groups {ids} -- "
                    "bookkeeping/partition mismatch (fail-closed)")
            if ids[0] >= 0:
                classes.append(ids[0])
            else:
                classes.append(next_free)
                next_free += 1
        # renumber classes densely, preserving merges
        uniq = sorted(set(classes))
        remap = {c: i for i, c in enumerate(uniq)}
        classes = [remap[c] for c in classes]
        Q = torch.zeros(len(uniq), n_bulk, dtype=torch.float64,
                        device=bulk_pp.device)
        for b, c in enumerate(classes):
            Q[c, b] = 1.0
        return Q, classes

    def export_quotient_state(self) -> dict:
        meta = {k: v for k, v in self._quotient_meta.items()}
        ref = meta.get("contact_reference")
        if ref is not None:
            ref = dict(ref)
            for k in ("particles_a_mask", "particles_b_mask"):
                if ref.get(k) is not None:
                    ref[k] = ref[k].detach().cpu().clone()
            meta["contact_reference"] = ref
        return {
            "quotient_bulk_groups": [m.detach().cpu().clone()
                                     for m in self._quotient_bulk_groups],
            **meta,
        }

    def import_quotient_state(self, state: "dict | None",
                              n_particles: int) -> None:
        if not state:
            return
        pc = state.get("particle_count")
        if pc is not None and int(pc) != int(n_particles):
            raise ValueError(
                f"quotient state was saved for {pc} particles, resume "
                f"cloud has {n_particles} (fixed-N guard)")
        groups = [torch.as_tensor(m).bool()
                  for m in state.get("quotient_bulk_groups", [])]
        for m in groups:
            if m.shape[0] != n_particles:
                raise ValueError("quotient group mask length mismatch")
        # groups must be pairwise disjoint (union-find closure done at
        # commit time; imported state must already satisfy it)
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if bool((groups[i] & groups[j]).any()):
                    raise ValueError("overlapping quotient groups")
        self._quotient_bulk_groups = groups
        for k in self._quotient_meta:
            if k in state:
                self._quotient_meta[k] = state[k]
        ref = self._quotient_meta.get("contact_reference")
        if ref is not None:
            # 0N-4 mask validation (reviewer): bool dtype, length N,
            # non-empty, disjoint
            ma, mb = ref.get("particles_a_mask"), ref.get("particles_b_mask")
            for name, m in (("particles_a_mask", ma),
                            ("particles_b_mask", mb)):
                if m is None:
                    continue
                m = torch.as_tensor(m)
                if m.dtype != torch.bool:
                    raise ValueError(f"contact_reference {name} must be "
                                     f"a bool mask (got {m.dtype})")
                if m.shape[0] != n_particles:
                    raise ValueError(f"contact_reference {name} length "
                                     f"{m.shape[0]} != N {n_particles}")
                if not bool(m.any()):
                    raise ValueError(f"contact_reference {name} is empty")
                ref[name] = m
            if ma is not None and mb is not None \
                    and bool((ref["particles_a_mask"]
                              & ref["particles_b_mask"]).any()):
                raise ValueError("contact_reference particle masks overlap")
        elif self._quotient_meta.get("quotient_at") is not None:
            # legacy checkpoint (pre-0N-4, e.g. the 0N-3 states): the
            # switch semantics are reconstructed EXACTLY from the stored
            # switch certificate; sigma_* stays None here (filled later
            # only when perimeter_sigma is explicitly fixed) and the
            # particle masks are resolved lazily via the saved bulk
            # pair on the first State III evaluation. A certificate
            # without the required fields (fixture stubs) leaves the
            # reference None -- the State III invariant then refuses to
            # run (fail-closed in the solver), it is never guessed.
            cert = self._quotient_meta.get("certificate_at_switch")
            if (not isinstance(cert, dict)
                    or "candidate" not in cert or "metrics" not in cert
                    or "thresholds" not in cert
                    or "h_ab" not in cert.get("metrics", {})):
                return
            self._quotient_meta["contact_reference"] = dict(
                raw_bulk_pair=tuple(
                    int(x) for x in
                    self._quotient_meta["raw_bulk_pair_at_switch"]),
                loop_pair_at_switch=tuple(
                    int(x) for x in cert["candidate"]["loop_pair"]),
                particles_a_mask=None, particles_b_mask=None,
                particle_count=self._quotient_meta.get("particle_count"),
                h_ab_at_switch=float(cert["metrics"]["h_ab"]),
                coverage_radius_at_switch=float(
                    cert["metrics"]["coverage_radius"]),
                sigma_at_switch=None,
                thresholds_at_switch=dict(cert["thresholds"]),
                legacy_reconstructed=True)

    def commit_quotient(self, particle_mask, step_index, certificate,
                        raw_bulk_pair, labels_src=None) -> None:
        """Irreversibly merge the raw bulks covered by particle_mask
        (union-find closure with existing groups). With labels_src
        (the loop labeling of the switch source state) the 0N-4
        contact_reference is stored: the identification scale h_*, the
        mollifier sigma_*, the thresholds and the particle partition of
        the certified pair become quotient metadata, frozen at the
        switch (State III persistence invariant)."""
        mask = particle_mask.bool().clone()
        keep = []
        for g in self._quotient_bulk_groups:
            if bool((g & mask).any()):
                mask |= g
            else:
                keep.append(g)
        keep.append(mask)
        self._quotient_bulk_groups = keep
        self._quotient_meta.update(
            quotient_at=int(step_index),
            certificate_at_switch=certificate,
            raw_bulk_pair_at_switch=tuple(int(x) for x in raw_bulk_pair),
            particle_count=int(mask.shape[0]))
        if labels_src is not None:
            la, lb = certificate["candidate"]["loop_pair"]
            self._quotient_meta["contact_reference"] = dict(
                raw_bulk_pair=tuple(int(x) for x in raw_bulk_pair),
                loop_pair_at_switch=(int(la), int(lb)),
                particles_a_mask=(labels_src == int(la)).detach()
                .cpu().clone(),
                particles_b_mask=(labels_src == int(lb)).detach()
                .cpu().clone(),
                particle_count=int(mask.shape[0]),
                h_ab_at_switch=float(certificate["metrics"]["h_ab"]),
                coverage_radius_at_switch=float(
                    certificate["metrics"]["coverage_radius"]),
                sigma_at_switch=float(self._sigma),
                thresholds_at_switch=dict(certificate["thresholds"]))

    def _resolve_state_masses_labels(self, pos, nor):
        """0K masses + certified loop labels on an arbitrary state
        (shared by the on-state certificate/reference evaluations)."""
        config = self.config
        if config.mass_estimator == "loopwise_oriented_kde":
            from ..oriented_varifold.loopwise_mass import (
                resolve_loopwise_oriented_mass_source,
            )
            res = resolve_loopwise_oriented_mass_source(
                pos, nor, self._mass_delta_for_kde, self._mass_tau_for_kde,
                config.mass_kernel,
                ell_factor=config.grid_incidence_ell_factor)
            return res.m_loop, res.loop_labels_pre
        m = self._masses_for(pos, nor)
        return m, self._certified_loop_labels(pos, nor, m)

    def contact_certificates_on_state(self, varifold, ctx_src,
                                      labels_src):
        """Evaluate the State-II contact certificates on an arbitrary
        (committed) state: re-resolve 0K masses/labels there, transport
        the source bookkeeping maps through the label overlap bijection
        (fail-closed on topology ambiguity), evaluate. Returns
        (list of ContactCertificate, labels_dst, masses_dst)."""
        from ..transport.contact_certificate import (
            evaluate_all_candidates,
            transport_loop_maps,
        )
        pos = varifold.positions.detach()
        nor = varifold.normals.detach()
        m, labels = self._resolve_state_masses_labels(pos, nor)
        maps = transport_loop_maps(labels_src, labels, {
            "g": ctx_src["loop_to_grid_component"],
            "b": ctx_src["loop_to_raw_bulk"]})
        certs = evaluate_all_candidates(
            pos, nor, m, labels, maps["g"], maps["b"],
            self._sigma if getattr(self, "_sigma", None)
            else self.config.perimeter_sigma)
        return certs, labels, m

    def contact_reference_on_state(self, varifold):
        """0N-4 State III persistence invariant: evaluate the FROZEN
        switch semantics (h_*, sigma_*, thresholds_*, particle
        partition) of the committed contact reference on an arbitrary
        state. Returns None when no reference is stored (masks still
        unresolved on a legacy resume must be filled by the caller
        first). Fail-closed guard (reviewer eq. 3.1): the CURRENT loop
        of each stored mask must equal the stored particle set EXACTLY
        -- no splitting, merging, inflow or dropout -- otherwise the
        reference would silently grade a proper subset/superset."""
        from ..transport.contact_certificate import (
            ContactCertificateError,
            evaluate_contact_reference,
        )
        ref = self._quotient_meta.get("contact_reference")
        if ref is None or ref.get("particles_a_mask") is None:
            return None
        if ref.get("sigma_at_switch") is None:
            # legacy backfill: only an explicitly fixed perimeter_sigma
            # reproduces the switch value uniquely; auto-sigma legacy
            # checkpoints cannot be reconstructed -> fail-closed
            if self.config.perimeter_sigma is None:
                raise ContactCertificateError(
                    "legacy quotient state with auto perimeter_sigma: "
                    "sigma_at_switch cannot be reconstructed uniquely")
            ref["sigma_at_switch"] = float(self.config.perimeter_sigma)
        pos = varifold.positions.detach()
        nor = varifold.normals.detach()
        m, labels = self._resolve_state_masses_labels(pos, nor)
        for name in ("particles_a_mask", "particles_b_mask"):
            mask = ref[name].to(labels.device)
            labs = labels[mask].unique()
            if labs.numel() != 1:
                raise ContactCertificateError(
                    f"contact_reference {name} spans {labs.numel()} "
                    "current loops (stored pair partition broken)")
            cur = labels == labs[0]
            if not bool((cur == mask).all()):
                raise ContactCertificateError(
                    f"contact_reference {name}: current loop "
                    f"{int(labs[0])} != stored particle set "
                    f"({int(cur.sum())} vs {int(mask.sum())} particles) "
                    "-- inflow/dropout, fail-closed")
        la = labels[ref["particles_a_mask"].to(labels.device)][0]
        lb = labels[ref["particles_b_mask"].to(labels.device)][0]
        if int(la) == int(lb):
            raise ContactCertificateError(
                "contact_reference masks map to the SAME current loop")
        return evaluate_contact_reference(
            pos, nor, m,
            ref["particles_a_mask"].to(labels.device),
            ref["particles_b_mask"].to(labels.device),
            ref["h_ab_at_switch"], ref["coverage_radius_at_switch"],
            ref["sigma_at_switch"], ref["thresholds_at_switch"],
            ref["raw_bulk_pair"], ref["loop_pair_at_switch"])

    def sharp_perimeter(self, varifold) -> float:
        """P#(X) = sum_i m_i^loop(X) q_i^WB(X): the actual-state
        perimeter energy under the production mass estimator and
        perimeter_q_mode (0J self-renormalized), evaluated WITHOUT
        touching the step state (used by the B2 split residual)."""
        pos = varifold.positions.detach()
        nor = varifold.normals.detach()
        config = self.config
        saved = (self._loop_labels_last, self.fixed_coherence,
                 getattr(self, "_qmode_stats", None))
        try:
            if config.mass_estimator == "loopwise_oriented_kde":
                from ..oriented_varifold.loopwise_mass import (
                    resolve_loopwise_oriented_mass_source,
                )
                res = resolve_loopwise_oriented_mass_source(
                    pos, nor, self._mass_delta_for_kde,
                    self._mass_tau_for_kde, config.mass_kernel,
                    ell_factor=config.grid_incidence_ell_factor)
                m, labels = res.m_loop, res.loop_labels_pre
            else:
                m = self._masses_for(pos, nor)
                labels = self._certified_loop_labels(pos, nor, m)
            sigma = self._sigma if getattr(self, "_sigma", None) \
                else config.perimeter_sigma
            from ..transport.bem_wasserstein import compute_coherence
            q_full = compute_coherence(varifold, m, sigma,
                                       config.perimeter_kernel)
            self._loop_labels_last = labels
            self.fixed_coherence = q_full
            if config.perimeter_q_mode == "full":
                q = q_full
            else:
                q = self._corrected_perimeter_q(varifold, m, sigma)
            return float((m * q).sum())
        finally:
            (self._loop_labels_last, self.fixed_coherence,
             self._qmode_stats) = saved

    def _bulk_partition(self, pos, nor, m_grid):
        """0F raw bulk incidence partition of the particles (winding
        certificate). Returns dict(bulk_of_loop, margin, loops, bulk_pp,
        n_bulk); bulk_of_loop is None when the certificate fails (the
        caller raises IncidenceUnstableError fail-closed)."""
        from ..transport.incidence import (
            oriented_graph_components,
            winding_bulk_labels,
        )
        config = self.config
        spacing = float(m_grid.median())
        # 0J: consume the certified per-step loop cache (computed once
        # in _setup_step from the raw oriented masses)
        loops = self._loop_labels_last
        if loops is None:
            ell = config.grid_incidence_ell_factor * spacing
            loops = oriented_graph_components(pos, nor, ell)
        bulk_of_loop, margin = winding_bulk_labels(
            pos, nor, m_grid, loops, spacing)
        out = dict(bulk_of_loop=bulk_of_loop, margin=margin,
                   loops=loops, bulk_pp=None, n_bulk=None)
        if bulk_of_loop is not None:
            out["bulk_pp"] = bulk_of_loop[loops]
            out["n_bulk"] = int(bulk_of_loop.max()) + 1
        return out

    def _augment_constraint_rows(self, pos, nor, m_grid, gm, rows, aug,
                                 bulk_part=None, AB_solve=None):
        """0F branch (reviewer v2 protocol): the physical bulk rows
        AUGMENT the grid compatibility rows; they never replace them.

        - partitions equal          -> legacy rows (bitwise-identical
                                       path; augmentation does not fire)
        - bulk strictly finer       -> contact-layer rows per
                                       config.grid_contact_rows_mode
                                       (0L-B0 cells A-D; "stack" =
                                       [C_grid; C_bulk] rank-revealing,
                                       post-bridge rank 3 is normal)
        - anything else / ambiguous -> fail-closed
        - "bulk_only_diagnostic"    -> bulk rows alone + projected
                                       Poisson forward (0F-0 cell 3;
                                       diagnostic only)
        """
        from ..transport.grid_wasserstein import (
            bulk_grid_incidence,
            constraint_row_independence,
            reduce_constraint_rows,
            relative_bulk_rows,
            row_space_discrepancy,
        )
        from ..transport.incidence import (
            IncidenceUnstableError,
            PartitionAmbiguousError,
            partition_relation,
        )

        config = self.config
        if bulk_part is None:
            bulk_part = self._bulk_partition(pos, nor, m_grid)
        loops = bulk_part["loops"]
        margin = bulk_part["margin"]
        aug["margin"] = margin
        if bulk_part["bulk_of_loop"] is None:
            raise IncidenceUnstableError(
                f"winding incidence certificate failed (worst integer "
                f"margin {margin:.4f}, {int(loops.max()) + 1} loops)")
        bulk_pp = bulk_part["bulk_pp"]
        n_bulk = bulk_part["n_bulk"]
        aug["n_bulk"] = n_bulk
        cur_rows = gm.flux.bulk_rows(bulk_pp, n_bulk)
        if config.grid_bulk_rows_functional == "geometric_area":
            # 0L-A: the physical row is the exact polygon-area
            # differential; the current-volume rows keep running as
            # shadow telemetry (never stacked with the geometric rows
            # -- two volume functionals at once = overconstraint).
            from ..transport.loop_geometry import area_variation_rows
            bulk_rows = area_variation_rows(
                pos, nor, bulk_pp, self._loop_orders_last, loops,
                n_bulk)
            self._bulk_rows_shadow = cur_rows
            aug["r_geom"] = reduce_constraint_rows(bulk_rows)[1]
        else:
            bulk_rows = cur_rows
        aug["r_bulk"] = reduce_constraint_rows(bulk_rows)[1]

        # ---- L0J-M first-moment rows (reviewer 2026-08-21) --------
        # raw per-bulk rows here; aggregation per quotient class via
        # (Q_quot (x) I_2) at the stacking point. NEVER through P_rel
        # (G conserves no spatial moment). Moment rows are L2-
        # normalized before the rank reduction (dimension [L^2] vs the
        # area rows' [L]; normalization is kernel-invariant); a
        # near-zero row is fail-closed.
        moment_rows_raw = None
        moment_scalars = None       # per-raw-bulk (A_b, M_bx, M_by)
        if config.grid_bulk_first_moment_rows:
            if config.grid_bulk_first_moment_rows_form in (
                    "current", "current_centered"):
                # order-free flux-family rows (no cyclic order used).
                # NOTE (reviewer 2026-08-24, quotient-aggregation
                # correction): the rows stay UNCENTERED here; the
                # centering happens in _with_moments AFTER the
                # (Q (x) I2) aggregation, with the MERGED xbar_c =
                # M_c / A_c -- centering per raw bulk before the
                # aggregation (the first implementation) wrongly
                # retains parts of the per-component centered
                # constraints in State III (eq (2) vs eq (1)).
                moment_rows_raw = gm.flux.first_moment_rows(
                    bulk_pp, n_bulk)
                if config.grid_bulk_first_moment_rows_form \
                        == "current_centered":
                    m_car = gm.flux._m_op          # carrier masses
                    xn = (pos * nor).sum(dim=1)
                    moment_scalars = []
                    for b in range(n_bulk):
                        sel = bulk_pp == b
                        Vb = float(0.5 * (m_car[sel] * xn[sel]).sum())
                        Ms = [float(0.5 * (m_car[sel]
                                           * pos[sel, k] ** 2
                                           * nor[sel, k]).sum())
                              for k in (0, 1)]
                        moment_scalars.append((Vb, Ms[0], Ms[1]))
            else:
                from ..transport.loop_geometry import (
                    first_moment_variation_rows,
                )
                moment_rows_raw = first_moment_variation_rows(
                    pos, nor, bulk_pp, self._loop_orders_last, loops,
                    n_bulk)

        def _with_moments(stacked, Q_quot=None):
            if moment_rows_raw is None:
                return stacked
            M = moment_rows_raw
            CA = cur_rows            # raw current volume rows (B, 2N)
            if moment_scalars is not None:
                # centered form (reviewer eq (1)): aggregate FIRST,
                # then center with the MERGED barycenter M_c / A_c
                from ..transport.boundary_flux_grid import (
                    quotient_centered_moment_rows,
                )
                M = quotient_centered_moment_rows(
                    M, CA, moment_scalars, Q_quot)
            elif Q_quot is not None:
                Bq = int(Q_quot.shape[0])
                M2 = torch.zeros(2 * Bq, M.shape[1], dtype=M.dtype,
                                 device=M.device)
                for c_ in range(Bq):
                    for b in range(n_bulk):
                        if float(Q_quot[c_, b]) > 0:
                            M2[2 * c_] += M[2 * b]
                            M2[2 * c_ + 1] += M[2 * b + 1]
                M = M2
            norms = M.norm(dim=1)
            if bool((norms < 1e-12).any()):
                raise RuntimeError(
                    "first-moment row with near-zero norm -- "
                    "fail-closed (degenerate bulk geometry)")
            aug["moment_row_norms"] = [float(x) for x in norms]
            aug["n_moment_rows"] = int(M.shape[0])
            return torch.cat([stacked, M / norms[:, None]], dim=0)

        if config.grid_bulk_rows_mode == "bulk_only_diagnostic":
            gm.forward_solve = "projected"
            reduced, r, _ = reduce_constraint_rows(bulk_rows)
            aug["fired"] = True
            aug["r_stack"] = r
            self._bulk_rows_last = bulk_rows
            return reduced

        frac = gm.flux.particle_component_fractions(
            gm.compat_labels,
            gm.phase.cell_area if gm.phase is not None else None)
        # ambiguity means the ACTIVE kernel mass is split between
        # different grid components; mass lost to inactive (vacuum)
        # cells -- normal for contact-layer marks -- does not make the
        # attribution ambiguous, so normalize by the active total
        active_total = frac.sum(dim=1)
        if bool((active_total < 1e-6).any()):
            raise PartitionAmbiguousError(
                f"{int((active_total < 1e-6).sum())} particles have "
                f"essentially no active kernel mass -- cannot attribute "
                f"to any grid component")
        rel = frac / active_total.clamp_min(1e-30)[:, None]
        fmax, grid_pp = rel.max(dim=1)
        n_ambiguous = int((fmax < 0.9).sum())
        if n_ambiguous:
            raise PartitionAmbiguousError(
                f"{n_ambiguous} particles split their ACTIVE kernel "
                f"mass across grid components (dominant share < 0.9) "
                f"-- partition attribution ambiguous; refusing to "
                f"compare")
        rel = partition_relation(grid_pp, bulk_pp)
        aug["relation"] = rel
        aug["relation_base"] = rel
        # 0L-B1: State-II bookkeeping maps for the contact certificate
        # (loop -> grid compatibility component / raw bulk); recorded
        # on every admissible step, consumed by the driver telemetry
        # (no action in B1)
        self._contact_pair_context = {
            "loop_to_grid_component": {
                int(lb): int(grid_pp[loops == lb].mode().values)
                for lb in loops.unique()},
            "loop_to_raw_bulk": {
                int(lb): int(bulk_part["bulk_of_loop"][int(lb)])
                for lb in loops.unique()},
            "relation": rel,
        }
        if gm.gauge_hold_fired and gm.compat_labels_base is not None:
            # 0L-B0 cell E: the relation the BASE (sweep) labels would
            # have read -- kept so the b_finer onset stays visible
            fb = gm.flux.particle_component_fractions(
                gm.compat_labels_base,
                gm.phase.cell_area if gm.phase is not None else None)
            relb = fb / fb.sum(dim=1).clamp_min(1e-30)[:, None]
            aug["relation_base"] = partition_relation(
                relb.max(dim=1)[1], bulk_pp)
        if rel not in ("equal", "b_finer"):
            raise PartitionAmbiguousError(
                f"grid vs bulk particle partitions are {rel!r} (grid "
                f"{aug['r_grid']} classes, bulk {n_bulk}) -- only "
                f"equal or bulk-strictly-finer is admissible; "
                f"fail-closed")
        if config.grid_bulk_rows_functional == "geometric_area":
            # 0L-A blocker fix: the geometric row must be active in
            # EVERY admissible branch -- the pre-merger rush happens
            # at partition equal, where the legacy early-return would
            # never fire the physical row and the causal experiment
            # would be void.
            stacked = torch.cat([rows, bulk_rows], dim=0)
            aug["d_compat"] = constraint_row_independence(
                rows, bulk_rows)
            Q_eq, _ = self._quotient_map(bulk_pp, n_bulk)
            reduced, r_stack, _ = reduce_constraint_rows(
                _with_moments(stacked, Q_eq))
            aug["fired"] = True
            aug["r_stack"] = r_stack
            self._bulk_rows_last = bulk_rows
            return reduced
        if rel == "equal":
            if moment_rows_raw is None:
                return rows            # legacy path, bitwise
            Q_eq, _ = self._quotient_map(bulk_pp, n_bulk)
            reduced, r_stack, _ = reduce_constraint_rows(
                _with_moments(rows, Q_eq))
            aug["r_stack"] = r_stack
            return reduced
        # ---- contact layer (b_finer): 0L-B0 cells A-D -------------
        n_grid = rows.shape[0]
        A_gb = bulk_grid_incidence(grid_pp, bulk_pp, n_grid, n_bulk)
        # 0L-B2 State III: quotient of raw bulk classes. C_quot =
        # Q_quot C_raw, incidence over quotient classes, relative rows
        # over quotient classes (exact merger: 2 raw bulks -> 1 class
        # -> P_rel = 0 -> rows = {G}); raw rows stay as shadow.
        Q_quot, quot_classes = self._quotient_map(bulk_pp, n_bulk)
        aug["quotient_classes"] = quot_classes
        if Q_quot is not None:
            C_quot = Q_quot @ bulk_rows
            quot_pp = torch.as_tensor(quot_classes,
                                      device=bulk_pp.device)[bulk_pp]
            A_gq = bulk_grid_incidence(grid_pp, quot_pp, n_grid,
                                       int(Q_quot.shape[0]))
            rel_rows = relative_bulk_rows(C_quot, A_gq)
            C_agg = A_gq.to(bulk_rows.dtype) @ C_quot
        else:
            rel_rows = relative_bulk_rows(bulk_rows, A_gb)
            C_agg = A_gb.to(bulk_rows.dtype) @ bulk_rows
        aug["d_compat"] = constraint_row_independence(rows, bulk_rows)
        aug["eps_row"] = row_space_discrepancy(rows, C_agg)
        if AB_solve is not None:
            N = pos.shape[0]

            def _compose(mat):
                return mat[:, :N] + mat[:, N:] @ AB_solve
            aug["eps_row_composed"] = row_space_discrepancy(
                _compose(rows), _compose(C_agg))
        self._contact_rows_last = {"G": rows, "C_bulk": bulk_rows,
                                   "C_rel": rel_rows}
        mode = config.grid_contact_rows_mode
        aug["fired"] = True
        if mode == "stack":
            stacked = torch.cat([rows, bulk_rows], dim=0)
            self._bulk_rows_last = bulk_rows
        elif mode == "aligned_prequotient":
            stacked = torch.cat([rows, rel_rows], dim=0)
            self._bulk_rows_last = rel_rows
            self._bulk_rows_shadow = bulk_rows
        elif mode == "bulk_projected":
            gm.forward_solve = "projected"
            stacked = bulk_rows
            self._bulk_rows_last = bulk_rows
        elif mode == "grid_only":
            stacked = rows
            self._bulk_rows_shadow = bulk_rows
        else:
            raise ValueError(
                f"unknown grid_contact_rows_mode {mode!r}")
        reduced, r_stack, _ = reduce_constraint_rows(
            _with_moments(stacked, Q_quot))
        aug["r_stack"] = r_stack
        return reduced

    def _setup_vertical_dof(self, varifold, prev_masses, gm, amps):
        """Arm V2 setup: free set F from the frozen pre-step gate,
        response columns c_i, their Ldag images, and the QP data. All
        frozen coefficients -- same philosophy as W_lin. Fail-closed:
        any solve failure leaves self._vertical = None (pure Otto
        step) and records the reason."""
        from src.torch.transport.carrier_amplitude import (
            CarrierAmplitudeConfig,
            _amplitude_response_columns,
            _gate_and_sdead,
        )
        from src.torch.transport.weighted_poisson import (
            IncompatibleGridVelocityError,
            WeightedPoissonSolveError,
        )

        config = self.config
        acfg = config.grid_amplitude_config or CarrierAmplitudeConfig()
        if acfg.lam <= 0.0:
            self._vertical_stats = {"skipped": "lam<=0"}
            return
        a_full = (amps if amps is not None
                  else torch.ones_like(prev_masses))
        gate, s_dead = _gate_and_sdead(
            varifold, a_full, gm.phase.rho_metric, gm.config.phase,
            acfg, coherence=self.fixed_coherence)
        w = gate * s_dead
        free = (w > 0) & (a_full > 0)
        n_free = int(free.sum())
        stats = {
            "coupled": True,
            "lam": float(acfg.lam),
            "n_gated": int((gate > 0).sum()),
            "n_free": n_free,
            "n_free_dropped": 0,
            "skipped": None,
        }
        self._vertical_stats = stats
        if n_free == 0:
            return
        if n_free > acfg.max_free:
            thr = torch.topk(w[free], acfg.max_free).values[-1]
            stats["n_free_dropped"] = n_free - acfg.max_free
            free = free & (w >= thr)
            free &= torch.cumsum(free.to(torch.int64), 0) \
                <= acfg.max_free
            n_free = int(free.sum())
            stats["n_free"] = n_free
        idx = torch.nonzero(free, as_tuple=False).squeeze(1)
        try:
            cols = _amplitude_response_columns(
                varifold.positions.detach()[idx],
                varifold.normals.detach()[idx],
                prev_masses[idx], gm.config.phase, gm.poisson)
            flat = cols.reshape(n_free, -1)
            phis = torch.empty_like(flat)
            for j in range(n_free):
                phi, info = gm.poisson.solve(cols[j])
                if not info.converged:
                    raise WeightedPoissonSolveError(info)
                phis[j] = phi.reshape(-1)
        except (IncompatibleGridVelocityError,
                WeightedPoissonSolveError) as exc:
            stats["skipped"] = type(exc).__name__
            return
        h = config.time_step
        Hmat = (flat @ phis.T) * gm.poisson.cell_area / h
        Hmat = 0.5 * (Hmat + Hmat.T)
        mu = 1e-12 * float(Hmat.diagonal().sum()) / n_free
        Hmat = Hmat + mu * torch.eye(n_free, dtype=Hmat.dtype,
                                     device=Hmat.device)
        self._vertical = {
            "idx": idx,
            "cols": cols,
            "phis": phis,
            "H": Hmat,
            "g_lam": acfg.lam * (w * prev_masses)[idx],
            "lower": -a_full[idx],
            "qp_iters": acfg.qp_iters,
            "cell_area": gm.poisson.cell_area,
        }
        # bookkeeping constant lam * sum_F w m a so the recorded
        # objective is the FULL coupled objective at da = 0; updated
        # with the lam * g^T da part as sweeps fix da (constants w.r.t.
        # (s, dtheta), but they keep objective_decreased honest)
        self._vertical_const = float(
            (self._vertical["g_lam"] * a_full[idx]).sum())

    def _contact_rows_telemetry(self, s, dth):
        """0L-B0: G v / C_bulk v / C_rel v on the committed MM velocity
        (recorded whenever the contact-layer rows were built this
        step, whatever mode constrained them)."""
        rows = getattr(self, "_contact_rows_last", None)
        if rows is None:
            return None
        v = torch.cat([s, dth])
        out = {}
        for k, mat in rows.items():
            if mat is not None and mat.numel():
                out[k] = tuple(float(x) for x in (mat @ v))
        return out or None

    def _collect_grid_step_stats(self):
        """Snapshot the per-step Poisson solve counters (G3a). The
        operator is rebuilt at every setup, so the counters cover
        exactly this step's optimization."""
        if self.bie_wasserstein is not None and self.bie_wasserstein.blocks:
            bw = self.bie_wasserstein
            return GridMetricStepStats(
                n_forward_solves=bw.n_forward_solves,
                n_projected_solves=0,
                pcg_iterations_total=0, pcg_iterations_min=0,
                pcg_iterations_median=0.0, pcg_iterations_max=0,
                # BIE: worst componentwise compatibility residual
                max_relative_residual=bw._max_r_comp,
                max_inactive_rhs_fraction=0.0,
                solve_wall_seconds=bw.solve_wall_seconds,
                max_projection_rel=None)
        if (self.grid_wasserstein is None
                or self.grid_wasserstein.poisson is None):
            return None
        st = self.grid_wasserstein.poisson.stats
        it = sorted(st.pcg_iterations)
        n = len(it)
        med = 0.0 if n == 0 else (
            float(it[n // 2]) if n % 2
            else 0.5 * (it[n // 2 - 1] + it[n // 2]))
        return GridMetricStepStats(
            n_forward_solves=st.n_forward_solves,
            n_projected_solves=st.n_projected_solves,
            pcg_iterations_total=sum(it),
            pcg_iterations_min=it[0] if n else 0,
            pcg_iterations_median=med,
            pcg_iterations_max=it[-1] if n else 0,
            max_relative_residual=st.max_relative_residual,
            max_inactive_rhs_fraction=st.max_inactive_rhs_fraction,
            solve_wall_seconds=st.wall_seconds,
            max_projection_rel=(
                self.grid_wasserstein.max_projection_rel
                if self.grid_wasserstein.forward_solve == "projected"
                else None),
        )

    def _apply_adaptive_rank_machine(self):
        """P1.2 rank initialization (review decision): certified seed +
        AdaptiveRankMachine handoff.

        First frame with initial_rank_certificate = C0: the seed is an
        INTEGER certificate only (never component labels). It is
        accepted only if the spectrum itself reads the same count
        (c_abs == c_gap == C0) and the phase-constant vector sits in
        the C0 candidate span; then the machine's noise floor and
        continuation basis are initialised from the full SVD. Without a
        certificate the machine's own self-consistent initialisation
        rule applies; refusal is a hard stop, never a silent oracle
        fallback. Subsequent frames: the audited noise-floor /
        persistence / phase-constant machinery, plus the C_u row-space
        continuation gate enforced in the constraint-basis stage.
        """
        from src.torch.transport.bem_wasserstein import (
            AmbiguousComponentRankError,
        )
        from src.torch.transport.rank_state import (
            AdaptiveRankMachine,
            RankDecision,
            angle_vec_in_span_deg,
        )

        bem = self.bem_wasserstein
        ev = bem.rank_evidence_last
        U, S, _Vh = bem._spec_svd
        sqrt_w = bem._endpoint_weights_cache.sqrt()

        if self.adaptive_rank_machine is None:
            from src.torch.transport.rank_state import AdaptiveRankConfig
            self.adaptive_rank_machine = AdaptiveRankMachine(
                config=AdaptiveRankConfig(
                    theta_const_deg=self.config.rank_theta_const_deg))
            cert = self.config.initial_rank_certificate
            if cert is not None and not (
                    1 <= int(cert) <= self.config.bem_rank_max):
                raise ValueError(
                    f"initial_rank_certificate={cert} outside "
                    f"[1, bem_rank_max={self.config.bem_rank_max}]")
            if cert is not None:
                if not (ev.c_abs == ev.c_gap == int(cert)):
                    raise AmbiguousComponentRankError(
                        f"rank certificate C0={cert} REJECTED: the "
                        f"spectrum reads c_abs={ev.c_abs}, "
                        f"c_gap={ev.c_gap} -- the certificate must "
                        "agree with the operator, it cannot overrule "
                        "it", ev)
                U0 = U[:, -int(cert):].detach()
                b = (sqrt_w / sqrt_w.norm())
                align = angle_vec_in_span_deg(b, U0)
                m = self.adaptive_rank_machine
                if align >= m.config.theta_const_deg:
                    raise AmbiguousComponentRankError(
                        f"rank certificate C0={cert} REJECTED: phase-"
                        f"constant alignment {align:.1f} deg exceeds "
                        f"{m.config.theta_const_deg} deg", ev)
                s_asc = S.flip(0)
                rel = (s_asc / S[0].clamp_min(1e-300)).tolist()
                m.rank = int(cert)
                m.floor = max(rel[:int(cert)])
                m._prev_basis = U0
                self._last_rank_decision = RankDecision(
                    int(cert), "seeded_certificate", ev.status,
                    align_const_deg=align)
                bem.finalize_with_rank(int(cert))
                return

        l_res = float(self.fixed_masses.median()) \
            if self.fixed_masses is not None else 0.0
        decision = self.adaptive_rank_machine.update(
            S, U, sqrt_w, self._last_step_max_disp, l_res,
            rank_max=self.config.bem_rank_max)
        # review (P1.2 safety blocker): "hard_stop" is NOT an action the
        # machine returns -- the real budget-exhaustion action is
        # ambiguity_budget_exceeded, and it must stop the run
        if decision.action in ("uninitialised", "reject_artificial",
                               "ambiguity_budget_exceeded"):
            raise AmbiguousComponentRankError(
                f"adaptive rank machine: {decision.action} "
                f"(evidence {decision.evidence_status}"
                + (f", align {decision.align_const_deg:.1f} deg"
                   if decision.align_const_deg is not None else "")
                + ")", ev)
        self._last_rank_decision = decision
        bem.finalize_with_rank(decision.rank)

    def _apply_rank_state_machine(self):
        """0C-5d-1.1: decide the WORKING rank with the hysteretic state
        machine, then finalize the deferred BEM spectral setup with it.

        Same-rank weak evidence HOLDS the rank and the evolution continues
        -- the pre-contact confidence decay (small-component null level
        drifting across the fixed gap threshold while the geometry stays
        cleanly separated) must not stop a run. Hard stops: no_null_block,
        failure to initialise on the first frame, and reject_artificial
        (the candidate modes fail phase-constant/continuation validation
        -- hysteresis must not paper over an invalid support).
        """
        from src.torch.transport.bem_wasserstein import (
            AmbiguousComponentRankError,
        )
        from src.torch.transport.rank_state import (
            RankStateConfig,
            RankStateMachine,
        )

        bem = self.bem_wasserstein
        ev = bem.rank_evidence_last
        if ev.status == "no_null_block":
            raise AmbiguousComponentRankError(
                "hard stop: no_null_block (nothing below the null "
                "tolerance)", ev)
        if self.rank_machine is None:
            if ev.status != "clean":
                raise AmbiguousComponentRankError(
                    f"cannot initialise working rank: first evidence "
                    f"status={ev.status}", ev)
            self.rank_machine = RankStateMachine(
                rank=ev.c_gap, config=RankStateConfig())
        U, _S, _Vh = bem._spec_svd
        U_cand = U[:, -max(ev.c_gap, 1):].detach()
        sqrt_w = bem._endpoint_weights_cache.sqrt()
        decision = self.rank_machine.update(ev, U_cand, sqrt_w)
        if decision.action == "reject_artificial":
            raise AmbiguousComponentRankError(
                f"hard stop: reject_artificial "
                f"(phase-constant angle {decision.align_const_deg:.1f} deg"
                + (f", containment {decision.contain_deg:.1f} deg"
                   if decision.contain_deg is not None else "") + ")", ev)
        bem.finalize_with_rank(decision.rank)
        self._last_rank_decision = decision

    def _spectral_constraint_basis(self, angle_map: torch.Tensor,
                                   config) -> torch.Tensor:
        """0C-5a: orthonormal basis of ker C, where

            C = U0^T W^{1/2} D_vel (G - R_end * Amap) D_s

        is the spectral componentwise compatibility constraint on the
        parametrized displacement. U0 is the interior operator's left
        near-null basis (rank = bem_component_rank, oracle-supplied at this
        stage), G the replication of particle displacements to endpoints,
        Amap the production angle map Delta-theta(s) — the SAME pre-solved
        tensor the parametrization evolves with — and D_vel / D_s the
        coherence factors of the velocity and displacement (identity at
        q = 1). The rows already contain global conservation, so the legacy
        global volume constraint must NOT be stacked on top.
        """
        U0, w_end, r_phys, K = self.bem_wasserstein.spectral_pieces()
        N = r_phys.shape[0]
        ab = angle_map
        if config.displacement_q_suppress:
            ab = self.fixed_coherence[:, None] * ab

        sqrt_w = w_end.sqrt()
        # Hard invariant: the SAME tensor that scales the BEM endpoint
        # Neumann data scales the constraint (flux measure consistency).
        vel_scale = self.bem_wasserstein.flux_scale()
        vec = sqrt_w * vel_scale.repeat_interleave(K)      # (NK,)
        X = U0.T * vec[None, :]                            # (C, NK)
        C_G = X.reshape(-1, N, K).sum(-1)                  # (C, N)
        r_flat = r_phys.reshape(-1)
        coeff = (X * r_flat[None, :]).reshape(-1, N, K).sum(-1)
        C_full = C_G - coeff @ ab
        if config.displacement_q_suppress:
            # s = q * (Q y): the constraint acts on the actual displacement
            C_full = C_full * self.fixed_coherence[None, :]

        _, s_vals, Vh = torch.linalg.svd(C_full, full_matrices=True)
        rank = int((s_vals > s_vals.max() * 1e-12).sum())
        Q = Vh[rank:].T.contiguous()
        # particle-space constraint rows, kept for the C_u row-space
        # continuation diagnostic (0C-5d-1): with fixed particle IDs the
        # row spaces of consecutive steps are directly comparable, unlike
        # the weighted endpoint bases U0 whose angles mix in weight change.
        self._spectral_C_full = C_full.detach()
        # P1.2: C_u row-space continuation gate (adaptive mode only) --
        # the stronger condition AdaptiveRankMachine's own docstring
        # requires at C >= 2: consecutive steps' particle-space
        # compatibility ROW SPACES must not jump.
        if self.config.bem_rank_mode == "adaptive_state_machine":
            prev = self._prev_spectral_C_full
            if prev is not None and prev.shape == C_full.shape:
                Qa = torch.linalg.qr(prev.T)[0]
                Qb = torch.linalg.qr(C_full.detach().T)[0]
                cosmin = torch.linalg.svdvals(Qa.T @ Qb).min()
                ang = float(torch.rad2deg(torch.acos(
                    cosmin.clamp(-1.0, 1.0))))
                self._last_cu_rowspace_angle_deg = ang
                if ang > self.config.cu_continuation_max_deg:
                    from src.torch.transport.bem_wasserstein import (
                        AmbiguousComponentRankError,
                    )
                    raise AmbiguousComponentRankError(
                        f"C_u row-space continuation broken: principal "
                        f"angle {ang:.1f} deg > "
                        f"{self.config.cu_continuation_max_deg} deg",
                        self.bem_wasserstein.rank_evidence_last)
            self._prev_spectral_C_full = C_full.detach()

        # 0C-5a.1 hard invariants: dim Ran Q = N - C (the C compatibility
        # rows must be independent), Q lies in ker C, and is orthonormal.
        C_rows = C_full.shape[0]
        eps_alg = 1e-10
        if Q.shape[1] != N - C_rows:
            raise RuntimeError(
                f"spectral constraint rows are rank-deficient: expected "
                f"dim ker C = {N - C_rows}, got {Q.shape[1]} "
                f"(rank {rank} of {C_rows} rows)")
        kernel_resid = ((C_full @ Q).abs().max()
                        / C_full.abs().max().clamp_min(1e-300)).item()
        orth_resid = (Q.T @ Q - torch.eye(
            Q.shape[1], device=Q.device, dtype=Q.dtype)).abs().max().item()
        if kernel_resid > eps_alg or orth_resid > eps_alg:
            raise RuntimeError(
                f"spectral constraint basis failed algebraic check: "
                f"|C Q|/|C| = {kernel_resid:.2e}, "
                f"|Q^T Q - I| = {orth_resid:.2e} (tol {eps_alg})")
        return Q

    def objective(self, params: torch.Tensor) -> torch.Tensor:
        """Compute objective: P + W_lin.

        This is the stable method that torch.compile can trace and cache.
        """
        displacements, delta_angles = self.param.unpack_params(params)
        current_varifold = self.param.reconstruct_varifold(params)

        # Compute masses for CURRENT positions. 0K: the candidate
        # evaluation consumes the SOURCE-frozen loop labels (l(Y) :=
        # l(X)); recomputing graph components per candidate would make
        # the objective non-smooth.
        masses = self._masses_for(current_varifold.positions,
                                  current_varifold.normals,
                                  loop_labels=self._loop_labels_last)

        # Perimeter with FIXED coherence from previous step:
        # P = Σ m_i * q_i^{n-1}, where m_i depends on current positions
        # 0J: the perimeter weight may be the corrected q^WB;
        # perimeter_coherence == fixed_coherence when mode == "full"
        P = (masses * self.perimeter_coherence).sum()

        # Linearized Wasserstein under the configured metric backend
        # (BEM by default; grid_poisson evaluates the same signature).
        # Arm V2: when a vertical sweep fixed a removal deposit, the
        # grid metric prices flux + deposit jointly (cross term =
        # transport compensating removal); None is the pre-V2 path.
        offset = getattr(self, "_vertical_offset", None)
        if offset is not None:
            W_lin = self.wasserstein_metric(
                displacements=displacements,
                delta_angles=delta_angles,
                time_step=self.config.time_step,
                drho_offset=offset,
            )
        else:
            W_lin = self.wasserstein_metric(
                displacements=displacements,
                delta_angles=delta_angles,
                time_step=self.config.time_step,
            )

        total = P + W_lin
        if getattr(self, "_vertical", None) is not None:
            total = total + self._vertical_const

        # Store tensors (not .item()) to avoid graph break under torch.compile.
        # Converted to float in step() after optimization finishes.
        self._last_perimeter = P
        self._last_wasserstein = W_lin

        return total

    def _build_optimizer_options(self) -> dict:
        """Build optimizer options dict from config."""
        config = self.config
        if config.optimizer_method in ("bfgs", "l-bfgs"):
            return {
                "max_iter": config.optimizer_max_iter,
                "gtol": config.optimizer_tol,
                "lr": config.optimizer_lr,
                "disp": config.optimizer_disp,
            }
        elif config.optimizer_method in ("trust-ncg", "dogleg", "trust-exact", "trust-krylov"):
            return {
                "max_iter": config.optimizer_max_iter,
                "gtol": config.optimizer_tol,
            }
        else:
            return {
                "max_iter": config.optimizer_max_iter,
            }

    def _make_debug_wrapper(self, obj_fn):
        """Wrap objective with debug output (for optimizer_disp > 0)."""
        _call_count = [0]
        _prev_params = [None]

        def objective_with_debug(params: torch.Tensor) -> torch.Tensor:
            result = obj_fn(params)
            if self.config.optimizer_disp > 0:
                _call_count[0] += 1
                if _prev_params[0] is not None:
                    step_size = (params - _prev_params[0]).norm().item()
                else:
                    step_size = 0.0
                _prev_params[0] = params.detach().clone()
                if params.requires_grad:
                    grad = torch.autograd.grad(result, params, create_graph=False, retain_graph=True)[0]
                    print(f"  [call {_call_count[0]}] obj={result.item():.6f}, |grad|={grad.norm().item():.4e}, |step|={step_size:.4e}")
            return result

        return objective_with_debug

    def step(self, varifold: OrientedPointCloudVarifold) -> MMStepResult:
        """Perform a single MM step.

        Args:
            varifold: Current varifold μ^{n-1}.

        Returns:
            MMStepResult containing updated varifold and diagnostics.
        """
        # Per-step setup (parametrization, coherence, BEM cache)
        self._setup_step(varifold)

        # Select objective function (compiled or plain)
        if self.config.compile:
            if self._compiled_objective is None:
                self._compiled_objective = torch.compile(self.objective)
            obj_fn = self._compiled_objective
        else:
            obj_fn = self.objective

        # G3a velocity scaling: optimize v with F(dt v)/dt -- same
        # minimizer, O(1) gradient/curvature scales. The physical
        # parameter is always y_star below; every diagnostic that
        # follows is evaluated in y-space.
        dt_scale = self.config.time_step
        if self.config.optimizer_variable == "velocity":
            base_fn = obj_fn

            def obj_fn(v_params):
                return base_fn(dt_scale * v_params) / dt_scale

        # Wrap with debug output if needed
        if self.config.optimizer_disp > 0:
            obj_fn = self._make_debug_wrapper(obj_fn)

        # Initial parameters: y=0
        params_init = self.param.init_params()
        params_init.requires_grad_(True)

        # Run optimization
        result = minimize(
            obj_fn,
            params_init,
            method=self.config.optimizer_method,
            options=self._build_optimizer_options(),
            disp=self.config.optimizer_disp,
            callback=self.optimizer_callback,
        )
        y_star = (dt_scale * result.x
                  if self.config.optimizer_variable == "velocity"
                  else result.x)

        # Arm V2: block-coordinate sweeps of the COUPLED minimizing
        # movement. Each sweep solves the da box-QP at the current
        # (s, dtheta) -- linear term carries the cross coupling
        # <c_i, Ldag flux> -- then re-minimizes (s, dtheta) with the
        # accepted deposit as a drho offset. Both blocks descend the
        # same objective, so this is alternating minimization of the
        # unsplit problem, not operator splitting.
        self._vertical_da = None
        vert = getattr(self, "_vertical", None)
        if vert is not None:
            from src.torch.transport.carrier_amplitude import (
                _box_qp_active_set)
            da = torch.zeros_like(vert["g_lam"])
            for sweep in range(max(1, self.config.grid_vertical_sweeps)):
                with torch.no_grad():
                    s_cur, th_cur = self.param.unpack_params(y_star)
                    drho_flux = self.grid_wasserstein.flux(
                        s_cur.detach(), th_cur.detach())
                    cross = (vert["phis"]
                             @ drho_flux.reshape(-1)) \
                        * vert["cell_area"] / self.config.time_step
                    g_total = vert["g_lam"] + cross
                    da, _, qp_resid = _box_qp_active_set(
                        vert["H"], g_total, vert["lower"],
                        n_outer=vert["qp_iters"])
                    offset = torch.einsum(
                        "f,fhw->hw", da, vert["cols"])
                self._vertical_offset = offset.detach()
                self._vertical_const = float(
                    (vert["g_lam"] * (-vert["lower"])).sum()
                    + (vert["g_lam"] * da).sum())
                init = (y_star / dt_scale
                        if self.config.optimizer_variable == "velocity"
                        else y_star).detach().clone()
                init.requires_grad_(True)
                sweep_opts = self._build_optimizer_options()
                if self.config.grid_vertical_max_iter is not None:
                    sweep_opts["max_iter"] = min(
                        sweep_opts.get(
                            "max_iter",
                            self.config.grid_vertical_max_iter),
                        self.config.grid_vertical_max_iter)
                result = minimize(
                    obj_fn, init,
                    method=self.config.optimizer_method,
                    options=sweep_opts,
                    disp=self.config.optimizer_disp,
                    callback=self.optimizer_callback,
                )
                y_star = (dt_scale * result.x
                          if self.config.optimizer_variable
                          == "velocity" else result.x)
            self._vertical_da = da
            self._vertical_stats.update({
                "n_corner": int((da == vert["lower"]).sum()),
                "qp_resid": float(qp_resid),
                "metric_cost_da": float(0.5 * da @ (vert["H"] @ da)),
                "cross_term": float((cross * da).sum()),
                "removed_amplitude": float(-da.sum()),
                "sweeps": int(max(1, self.config.grid_vertical_sweeps)),
            })

        # Extract optimized varifold and parameters
        new_varifold = self.param.reconstruct_varifold(y_star)
        new_varifold = OrientedPointCloudVarifold(
            positions=new_varifold.positions.detach(),
            angles=new_varifold.angles.detach(),
        )
        displacements, delta_angles = self.param.unpack_params(y_star)

        # Final-point diagnostics (0A): the optimizer's _last_* caches belong
        # to whatever point it evaluated last, which need not be result.x.
        # Re-evaluate F, P, W and the gradient at y = 0 and at y = y_star
        # explicitly (always in y-space regardless of optimizer_variable).
        # Order matters: y* is evaluated LAST so that the caches end at the
        # returned point.
        F0, P0, W0, g0 = self._evaluate_at(torch.zeros_like(y_star))
        Fs, Ps, Ws, gs = self._evaluate_at(y_star)
        tol = 1e-12 * max(1.0, abs(F0))

        self._last_step_max_disp = float(displacements.detach().abs().max())

        return MMStepResult(
            varifold=new_varifold,
            vertical_da=(self._vertical_da.detach()
                         if getattr(self, "_vertical_da", None)
                         is not None else None),
            vertical_idx=(self._vertical["idx"]
                          if getattr(self, "_vertical", None)
                          is not None
                          and self._vertical_da is not None else None),
            vertical_stats=getattr(self, "_vertical_stats", None),
            perimeter=Ps,
            wasserstein=Ws,
            objective=Fs,
            converged=result.success,
            n_iter=result.nit if hasattr(result, 'nit') else -1,
            displacements=displacements.detach(),
            delta_angles=delta_angles.detach(),
            masses=self.fixed_masses,
            effective_masses=self.fixed_effective_masses,
            objective_initial=F0,
            frozen_perimeter_initial=P0,
            wasserstein_initial=W0,
            gradient_norm_initial=g0,
            gradient_norm_final=gs,
            relative_gradient_norm=gs / max(1.0, g0),
            step_norm=y_star.detach().norm().item(),
            objective_decreased=bool(Fs <= F0 + tol),
            optimizer_message=str(getattr(result, "message", None)),
            n_fev=getattr(result, "nfev", None),
            r_comp=self.bem_wasserstein._last_r_comp,
            detected_rank=self.bem_wasserstein.detected_rank,
            rank_status=self.bem_wasserstein.rank_status,
            rank_c_abs=(self.bem_wasserstein.rank_evidence_last.c_abs
                        if self.bem_wasserstein.rank_evidence_last
                        is not None else None),
            rank_c_gap=(self.bem_wasserstein.rank_evidence_last.c_gap
                        if self.bem_wasserstein.rank_evidence_last
                        is not None else None),
            rank_evidence_gap_ratio=(
                self.bem_wasserstein.rank_evidence_last.gap_ratio
                if self.bem_wasserstein.rank_evidence_last
                is not None else None),
            rank_action=(self._last_rank_decision.action
                         if self._last_rank_decision is not None else None),
            rank_pending=(self._last_rank_decision.pending
                          if self._last_rank_decision is not None else None),
            rank_gap_ratio=self.bem_wasserstein.rank_gap_ratio,
            rank_abs_level=self.bem_wasserstein.rank_abs_level,
            rank_spectrum_head=self.bem_wasserstein.rank_spectrum_head,
            subspace_angle_deg=self.bem_wasserstein.subspace_angle_deg,
            bem_setup_snapshot=getattr(self, "_last_bem_snapshot", None),
            grid_setup_snapshot=getattr(self, "_last_grid_snapshot", None),
            grid_step_stats=self._collect_grid_step_stats(),
            bulk_flux_residuals=(
                tuple(float(x) for x in (
                    self._bulk_rows_last
                    @ torch.cat([displacements.detach(),
                                 delta_angles.detach()])))
                if getattr(self, "_bulk_rows_last", None) is not None
                else None),
            bulk_flux_residuals_shadow=(
                tuple(float(x) for x in (
                    self._bulk_rows_shadow
                    @ torch.cat([displacements.detach(),
                                 delta_angles.detach()])))
                if getattr(self, "_bulk_rows_shadow", None)
                is not None else None),
            contact_rows_telemetry=self._contact_rows_telemetry(
                displacements.detach(), delta_angles.detach()),
            contact_pair_context=getattr(self, "_contact_pair_context",
                                         None),
            contact_certificate_source=(
                [c.as_dict() for c in self._contact_certificates_source]
                if isinstance(getattr(self, "_contact_certificates_source",
                                      None), list)
                else getattr(self, "_contact_certificates_source", None)),
            quotient_state=(self.export_quotient_state()
                            if self._quotient_bulk_groups else None),
        )

    def _evaluate_at(self, params: torch.Tensor):
        """Objective, its components, and the gradient norm at a given point.

        Fresh evaluation through the same frozen objective the optimizer
        minimised; used for the 0A final-point diagnostics.
        """
        p = params.detach().clone().requires_grad_(True)
        F = self.objective(p)
        (grad,) = torch.autograd.grad(F, p)
        return (F.item(), self._last_perimeter.item(),
                self._last_wasserstein.item(), grad.norm().item())


def mm_step(
    varifold: OrientedPointCloudVarifold,
    config: MMConfig,
) -> MMStepResult:
    """Perform a single MM step using BEM-based linearized Wasserstein.

    Thin wrapper around MMStepper for backward compatibility.
    For repeated calls (e.g., in a solver loop), prefer using
    MMStepper directly to reuse BEM objects across steps.

    Args:
        varifold: Current varifold μ^{n-1}.
        config: MM scheme configuration.

    Returns:
        MMStepResult containing updated varifold and diagnostics.
    """
    stepper = MMStepper(config)
    return stepper.step(varifold)
