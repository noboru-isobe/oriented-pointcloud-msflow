"""Arm 5: persistent carrier-amplitude fade of phase-interior marks
(spec-2.0-lite -- the uniform whole-complex reduction the feasibility
audit measured as optimal).

The confirmatory suite and arm 4 localized the surviving killer: the
hidden-pair population's CARRIER MASS keeps sourcing the filling map
after the merger (energetic/kinematic visibility is already q-zero,
but the current J = sum m_i n_i delta_x is un-suppressed), and the
closing crack's half-cancelled rho eventually beads into
sub-resolution fragments. The static audit showed (i) no
current-preserving weight reduction exists (K_{eps,F} full rank),
(ii) the OPTIMAL reduction at fixed removed amplitude is the uniform
proportional fade of the whole complex (marginal cost 0.993 r_fossil
per unit amplitude; pairwise deletion costs 2.1x), and the halo study
showed each increment's phase response is a small harmonic dipole
field. This stage implements exactly that move, spread over time:

    a_i^{n+1} = max(a_i^n - rate * g_i, 0)        (monotone, capped)

with the binary gate g_i taken from the FROZEN PSP-SPEC-1.0 Layer-1
candidate rule (antiparallel partner within eps, both far probes in
bulk, no jump) evaluated on the committed state against the pre-step
reconstructed phase. Every gated mark fades at the SAME rate -- the
whole cancellation complex reduces together (no pairwise orphaning),
and the fade is slow against the dynamics but fast against the
measured defect growth. Amplitudes are persistent state carried by
the stepper (spec-2.0's state extension (x, n) -> (x, n, a)); they
multiply the carrier masses seen by the GRID layer only (filling,
flux, volume rows). Monotonicity removes the a -> rho -> a feedback
loop by construction. No labels, no events, no deletion; a mark that
never satisfies the interior test keeps a = 1 forever (pre-contact
facing arcs stay load-bearing: the gap side reads vacuum).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_T = torch.Tensor


@dataclass(frozen=True)
class CarrierAmplitudeConfig:
    # frozen PSP-SPEC-1.0 Layer-1 constants
    c_ell: float = 2.5
    delta_in: float = 0.1
    delta_jump: float = 0.1
    pair_dot: float = -0.8
    # fade dynamics. `rate` is the MAXIMUM per-update decrement (a
    # Lipschitz bound); the solver closes the loop by scaling it with
    # the measured projection headroom (arm 8: a fixed rate is a
    # two-sided gamble -- 0.05 lost to the freezing front at 1011,
    # 0.25 lost to the water-filling transient at 971. The transient
    # is MEASURED every step, so the fade runs greedily under a fixed
    # fraction of the existing certified projection budget instead).
    rate: float = 0.05
    # fraction of projection_rel_tol the fade may consume: rate_scale
    # = clamp((frac*tol - p_now)/(frac*tol), 0, 1). Anchored to the
    # existing gate -- not a new scale.
    proj_budget_frac: float = 0.4
    # v3/arm-6 (notch diagnosis, step 989): garbage-collect marks the
    # moment their amplitude reaches EXACTLY zero. At a = 0 the mark
    # contributes nothing to the grid layer (filling, flux, volume
    # rows), so removal is exactly harmless there -- it is the
    # certified terminal point of the uniform fade, not a one-shot
    # deletion. What removal DOES change is intended healing: the
    # ghost stops collapsing its neighbors' coherence (the measured
    # freezing-front recruitment) and its co-moving pair direction
    # stops being a near-null DOF of the objective (the measured
    # 310x optimizer scatter).
    gc: bool = True
    # v2 (arm-4+5 v1 postmortem, step 810): the phase-only gate caught
    # crack marks the flow was STILL WELDING (q ~ 0.4-0.6); fading
    # them removed load-bearing interface and the seam buckled into
    # self-contact (paired q -> 0.99, P -19%). A fossil is interior
    # AND dynamically dead: the rate is modulated by
    # s(q) = clamp(1 - q/q_dead, 0, 1)^2, so q >= q_dead never fades
    # (measured fossils sit at q ~ 0.02-0.05; healing crack marks at
    # q ~ 0.4-0.6 are untouched until the dynamics abandons them).
    q_dead: float = 0.2
    # arm V1 (variational fade, spec-2.0 full prototype). A9 measured
    # the two structural defects of the PROCEDURAL fade: (i) uniform
    # decrement maximizes time-to-first-zero, so GC never fires
    # (n_zero = 0 the whole run), and (ii) the removed amplitude
    # leaves a standing projection load the external controller
    # cannot pay back. The variational path replaces rate/rate_scale/
    # allocation with one box-constrained QP per step:
    #   min_{da in [-a,0]^F} (1/2h) <C da, Ldag C da> + lam * sum w_i m_i da_i
    # where column c_i = InvLaplacian div(m_i n_i eta_i) is the mark's
    # unit-amplitude phase response (the audit's harmonic dipole) and
    # Ldag is the SAME frozen grid-metric operator as W_lin. Removal
    # cost and projection stress share one yardstick; antiparallel
    # pairs cancel in C, so pair-coordinated extinction (exact zeros
    # at box corners) emerges with no connectivity information.
    variational: bool = False
    # shadow price per unit of gated carrier amplitude (m_i * a_i)
    # removed. Calibrated from the measured H diagonal band: above the
    # pair-cancelled directions' cost, below the isolated-mark cost.
    lam: float = 0.0
    # bound on the QP size; overflow keeps the largest-w marks and is
    # RECORDED in stats (no silent cap).
    max_free: int = 256
    qp_iters: int = 300
    # experiment B (A10c postmortem): which norm prices the removal.
    # "metric" = the W_lin operator (H^-1-type, low-pass -- measured
    # to under-price the sub-cell pair residual that the projection
    # gate charges). "gate_l2" = the gate's own plain L2 on the
    # predicted phase response, H = Gram(C) * cell_area / h -- the
    # norm-consistency falsification arm of the coupled-MM plan.
    qp_norm: str = "metric"


def _gate_and_sdead(varifold, amplitudes: _T, rho: _T, phase_cfg,
                    cfg: CarrierAmplitudeConfig,
                    coherence: "_T | None" = None):
    """Shared eligibility state functions of the procedural and
    variational paths: the arm-9 smooth gate (frozen anchors as
    smoothstep midpoints, vacuum side exactly zero) and the v2
    dynamical-death modulation s(q)."""
    from .phase_support_projection import _bilinear_periodic

    pos, nrm = varifold.positions, varifold.normals
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    has_partner = ((d < phase_cfg.fill_epsilon)
                   & ((nrm @ nrm.T) < cfg.pair_dot)).any(dim=1)
    ell = cfg.c_ell * phase_cfg.fill_epsilon
    rp = _bilinear_periodic(rho, pos + ell * nrm,
                            phase_cfg.box_min, phase_cfg.box_max)
    rm = _bilinear_periodic(rho, pos - ell * nrm,
                            phase_cfg.box_min, phase_cfg.box_max)
    # arm 9: the gate is a SMOOTH state function. The binary test
    # (min(rho+-) > 1 - delta_in AND |jump| < delta_jump) flickers at
    # ~6% duty on the wall's own trench fluctuation (measured on the
    # complete rate ladder: every rate from 0.05 to 0.25 plus the
    # closed loop starved the same way, deaths 971-1014), so the fade
    # throughput was 50-100x below the nominal rate no matter the
    # knob. Smoothsteps with the FROZEN anchors as midpoints turn the
    # flicker into a continuous partial rate; the vacuum side is
    # EXACTLY zero (specificity pins unchanged: circle, annulus hole,
    # pre-contact gap all read bulk < 1 - 2*delta_in on one side).
    def _sstep(t):
        t = t.clamp(0.0, 1.0)
        return 3.0 * t * t - 2.0 * t * t * t

    lo = 1.0 - 2.0 * cfg.delta_in            # 0.8: gate exactly 0
    g_bulk = _sstep((torch.minimum(rp, rm) - lo) / (2.0 * cfg.delta_in))
    jump = (rp - rm).abs()
    g_jump = 1.0 - _sstep((jump - 0.5 * cfg.delta_jump)
                          / (1.5 * cfg.delta_jump))
    gate = has_partner.to(amplitudes.dtype) * g_bulk * g_jump

    if coherence is not None:
        s_dead = (1.0 - coherence / cfg.q_dead).clamp(0.0, 1.0) ** 2
    else:
        s_dead = torch.ones_like(amplitudes)
    return gate, s_dead


def update_amplitudes(varifold, amplitudes: _T, rho: _T, phase_cfg,
                      cfg: CarrierAmplitudeConfig =
                      CarrierAmplitudeConfig(),
                      coherence: "_T | None" = None,
                      rate_scale: float = 1.0):
    """One monotone fade update. rho is the (frozen, pre-step)
    reconstructed metric phase; varifold the committed state;
    coherence the q recomputed on that state (None -> no q
    modulation, unit-test convenience)."""
    gate, s_dead = _gate_and_sdead(varifold, amplitudes, rho,
                                   phase_cfg, cfg, coherence)
    out = (amplitudes - cfg.rate * rate_scale
           * gate * s_dead).clamp_min(0.0)
    out = torch.minimum(out, amplitudes)          # monotone always
    n_eff = int(((gate > 0) & (s_dead > 0)).sum())
    stats = {
        "rate_scale": float(rate_scale),
        "n_gated": int((gate > 0).sum()),
        "mean_gate": float(gate[gate > 0].mean()) if bool(
            (gate > 0).any()) else 0.0,
        "n_fading_eligible": n_eff,
        "n_fading": int(((out < 1.0) & (out > 0.0)).sum()),
        "n_zero": int((out == 0.0).sum()),
        "min_amplitude": float(out.min()),
        "faded_fraction": float(1.0 - out.mean()),
    }
    return out, stats


def _amplitude_response_columns(positions: _T, normals: _T,
                                values: _T, phase_cfg, poisson_op):
    """Columns c_i = InvLaplacian div(values_i n_i eta_i): the linear
    phase response of a unit amplitude decrement at mark i, through
    the SAME Wendland scatter and periodic inverse Laplacian as
    CurrentToPhase, then restricted to the operator's active set and
    gauge-projected (componentwise zero mean) so the metric solve
    sees a compatible source. Returns (F, H, W)."""
    from .phase_grid import ScatterStencil, _fft_inverse_laplacian_div

    st = ScatterStencil(positions, phase_cfg)
    H, W = st.H, st.W
    n_pts = positions.shape[0]
    cols = torch.empty(n_pts, H, W, dtype=values.dtype,
                       device=values.device)
    zeros = torch.zeros(H * W, dtype=values.dtype, device=values.device)
    for i in range(n_pts):
        jx = zeros.clone().scatter_add(
            0, st.flat_index[i],
            values[i] * normals[i, 0] * st.kernel_weights[i])
        jy = zeros.clone().scatter_add(
            0, st.flat_index[i],
            values[i] * normals[i, 1] * st.kernel_weights[i])
        c = _fft_inverse_laplacian_div(jx.reshape(H, W),
                                       jy.reshape(H, W), st.dx)
        c = torch.where(poisson_op.active, c, torch.zeros_like(c))
        cols[i] = poisson_op.project_componentwise_zero_mean(c)
    return cols


def _box_qp_active_set(Hmat: _T, g: _T, lower: _T,
                       n_outer: int = 100):
    """min 0.5 x^T H x + g^T x  s.t.  lower <= x <= 0, by a MONOTONE
    hybrid: working-set Newton (exact corner identification through a
    dense SPD solve on the free block) raced against a line-searched
    projected-gradient path, keeping whichever candidate descends
    most. Bound clamping is exact, so corners land at exactly lower_i
    (the GC trigger).

    A10 postmortem: plain projected gradient with step 1/L crawls
    along the pair-cancelled directions (condition eig_max/eig_min
    ~ 1e9 measured in the calibration probe) -- 0-3 corners/step when
    the calibration predicted the whole cancelled cohort per update.
    A10b postmortem: plain active-set iteration can CYCLE on the same
    conditioning and time out at a non-optimal iterate whose metric
    cost dwarfs the linear gain (measured: cost 2.16 vs gain ~0.02 at
    step 833; objective ABOVE the do-nothing point), ratcheting the
    projection channel with certified-looking removals. The hybrid
    accepts only descent steps from the feasible origin, so f(x) <= 0
    is a structural guarantee, and the KKT-enumeration oracle test
    pins exactness on random ill-conditioned instances."""
    def f(x):
        return float(0.5 * x @ (Hmat @ x) + g @ x)

    def kkt_resid(x):
        grad = Hmat @ x + g
        tiny = 1e-14
        r = torch.zeros_like(grad)
        interior = (x > lower + tiny) & (x < -tiny)
        r[interior] = grad[interior]
        vl = (x <= lower + tiny) & (grad < 0)
        r[vl] = grad[vl]
        vh = (x >= -tiny) & (grad > 0)
        r[vh] = grad[vh]
        return float(r.abs().max()) if r.numel() else 0.0

    L = float(torch.linalg.eigvalsh(Hmat)[-1])
    x = torch.zeros_like(g)
    fx = 0.0
    tiny = 1e-14
    scale = float(g.abs().max()) + 1e-300
    ts = torch.logspace(-12, 0, 25, dtype=g.dtype)
    for _ in range(n_outer):
        grad = Hmat @ x + g
        # KKT working sets (pin only where the multiplier sign holds)
        at_lo = (x <= lower + tiny) & (grad >= 0)
        at_hi = (x >= -tiny) & (grad <= 0)
        free = ~(at_lo | at_hi)
        cand = [x]
        # Newton candidate on the free block (exact corner
        # identification), taken ONLY if it descends
        if bool(free.any()):
            xn = torch.where(at_lo, lower, torch.zeros_like(x))
            Hff = Hmat[free][:, free]
            rhs = -(g[free] + Hmat[free][:, ~free] @ xn[~free])
            try:
                sol = torch.cholesky_solve(
                    rhs.unsqueeze(1), torch.linalg.cholesky(Hff)
                ).squeeze(1)
            except Exception:
                sol = torch.linalg.lstsq(Hff, rhs.unsqueeze(1)
                                         ).solution.squeeze(1)
            xn[free] = sol
            cand.append(xn.clamp(min=lower).clamp(max=0.0))
            # damped Newton path (backtracking handles a wrong
            # working-set guess without accepting an ascent)
            d = cand[-1] - x
            for t in (0.5, 0.25, 0.1):
                cand.append((x + t * d).clamp(min=lower).clamp(max=0.0))
        # projected-gradient path with log line search (guaranteed
        # KKT progress when Newton's working set is wrong)
        for t in ts:
            cand.append((x - (float(t) / max(L, 1e-300)) * grad)
                        .clamp(min=lower).clamp(max=0.0))
        vals = [f(c) for c in cand]
        j = int(min(range(len(vals)), key=vals.__getitem__))
        if vals[j] >= fx - 1e-16 * (abs(fx) + 1.0):
            break                                  # no descent left
        x, fx = cand[j], vals[j]
        if kkt_resid(x) < 1e-10 * scale:
            break
    # MONOTONE GUARANTEE: x descends from the feasible origin, so
    # f(x) <= 0 always -- the A10b failure mode (an unconverged
    # active-set iterate with metric cost far above the linear gain,
    # objective ABOVE the do-nothing point) is structurally
    # unreturnable.
    assert fx <= 1e-12, f"box QP returned an ascent point: f={fx}"
    return x, L, kkt_resid(x)


def variational_fade(varifold, amplitudes: _T, masses: _T,
                     poisson_op, rho: _T, phase_cfg,
                     cfg: CarrierAmplitudeConfig,
                     coherence: "_T | None" = None,
                     time_step: float = 1.0):
    """Arm V1: one variational fade update (see CarrierAmplitudeConfig
    docstring for the QP). Fail-closed: on an incompatible or
    unconverged metric solve the amplitudes pass through unchanged
    and the skip is recorded."""
    from .weighted_poisson import (
        IncompatibleGridVelocityError,
        WeightedPoissonSolveError,
    )

    gate, s_dead = _gate_and_sdead(varifold, amplitudes, rho,
                                   phase_cfg, cfg, coherence)
    w = gate * s_dead
    stats = {
        "variational": True,
        "qp_norm": cfg.qp_norm,
        "lam": float(cfg.lam),
        "n_gated": int((gate > 0).sum()),
        "mean_gate": float(gate[gate > 0].mean()) if bool(
            (gate > 0).any()) else 0.0,
        "n_free": 0,
        "n_free_dropped": 0,
        "n_corner": 0,
        "skipped": None,
    }

    def _finish(out):
        stats.update({
            "n_fading": int(((out < 1.0) & (out > 0.0)).sum()),
            "n_zero": int((out == 0.0).sum()),
            "min_amplitude": float(out.min()),
            "faded_fraction": float(1.0 - out.mean()),
        })
        return out, stats

    free = (w > 0) & (amplitudes > 0)
    n_free = int(free.sum())
    if n_free == 0 or cfg.lam <= 0.0:
        return _finish(amplitudes.clone())
    if n_free > cfg.max_free:
        # keep the largest-w marks; RECORDED, never silent
        thr = torch.topk(w[free], cfg.max_free).values[-1]
        stats["n_free_dropped"] = n_free - cfg.max_free
        free = free & (w >= thr)
        free &= torch.cumsum(free.to(torch.int64), 0) <= cfg.max_free
        n_free = int(free.sum())
    stats["n_free"] = n_free
    idx = torch.nonzero(free, as_tuple=False).squeeze(1)

    try:
        # per-unit da_i the grid carrier changes by m_i (m_grid = m*a)
        cols = _amplitude_response_columns(
            varifold.positions[idx], varifold.normals[idx],
            masses[idx], phase_cfg, poisson_op)
        flat = cols.reshape(n_free, -1)
        if cfg.qp_norm == "gate_l2":
            phis = flat            # plain L2: <c_i, c_j> * cell_area
        elif cfg.qp_norm == "metric":
            phis = torch.empty_like(flat)
            for j in range(n_free):
                phi, info = poisson_op.solve(cols[j])
                if not info.converged:
                    raise WeightedPoissonSolveError(info)
                phis[j] = phi.reshape(-1)
        else:
            raise ValueError(
                f"qp_norm must be 'metric' or 'gate_l2', got "
                f"{cfg.qp_norm!r}")
    except (IncompatibleGridVelocityError,
            WeightedPoissonSolveError) as exc:
        stats["skipped"] = type(exc).__name__
        return _finish(amplitudes.clone())

    Hmat = (flat @ phis.T) * poisson_op.cell_area / time_step
    Hmat = 0.5 * (Hmat + Hmat.T)
    mu = 1e-12 * float(Hmat.diagonal().sum()) / n_free
    Hmat = Hmat + mu * torch.eye(n_free, dtype=Hmat.dtype,
                                 device=Hmat.device)
    h_diag = Hmat.diagonal()
    # per-mark SOLO corner threshold lam*_i = H_ii a_i / (w_i m_i):
    # above it even an uncancelled mark goes to its corner alone.
    # Pair-cancelled subspaces reach corners far below this band
    # (their Rayleigh cost ~ h_eig_min), so the band's percentiles
    # bracket the calibration: lam << p10 = only near-perfectly
    # cancelled complexes extinguish; lam >> p90 = every gated mark.
    lam_corner = (h_diag * amplitudes[idx]
                  / (w * masses)[idx].clamp_min(1e-300))
    qs = torch.quantile(lam_corner,
                        torch.tensor([0.1, 0.5, 0.9],
                                     dtype=lam_corner.dtype))
    stats.update({
        "h_diag_min": float(h_diag.min()),
        "h_diag_median": float(h_diag.median()),
        "h_diag_max": float(h_diag.max()),
        "h_eig_min": float(torch.linalg.eigvalsh(Hmat)[0]),
        "lam_corner_p10": float(qs[0]),
        "lam_corner_p50": float(qs[1]),
        "lam_corner_p90": float(qs[2]),
    })

    g = cfg.lam * (w * masses)[idx]
    lower = -amplitudes[idx]
    da, h_eig_max, resid = _box_qp_active_set(
        Hmat, g, lower, n_outer=cfg.qp_iters)
    stats.update({
        "h_eig_max": h_eig_max,
        "qp_pg_residual": resid,
        "metric_cost": float(0.5 * da @ (Hmat @ da)),
        "linear_gain": float((g * da).sum()),
        "n_corner": int((da == lower).sum()),
    })

    out = amplitudes.clone()
    out[idx] = (amplitudes[idx] + da).clamp_min(0.0)
    out = torch.minimum(out, amplitudes)          # monotone always
    return _finish(out)
