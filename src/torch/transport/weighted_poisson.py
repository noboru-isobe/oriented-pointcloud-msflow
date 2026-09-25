"""Weighted Poisson operator on a fixed periodic grid (Grid metric G0).

The grid analogue of the MS tangent metric's elliptic core: for a
frozen diffuse phase rho >= 0,

    L_rho phi = -div(rho grad phi)

discretized as a cell-centered finite-volume operator with HARMONIC
mean face conductivities and an ACTIVE-SET restriction:

- harmonic mean: a face between an active and an inactive (vacuum)
  cell carries exactly zero conductivity. An arithmetic mean would
  leak flux into the vacuum and reconnect separated phase components
  -- the grid reincarnation of the spurious exchange mode. FORBIDDEN.
- active set: inactive cells are excluded from the unknowns; leaving
  them in would add one artificial null vector per vacuum cell
  (nullity C + #inactive instead of the phase-component count C).
- NO global density floor, ever (same reason as the harmonic mean).

Component structure: phase components are the connected components of
the POSITIVE-CONDUCTIVITY face graph (equivalently: 4-connected
components of the active mask), which is the operator's actual
nullspace structure -- one constant null vector per component. This
replaces the BIE near-nullspace/rank machinery: an annulus support is
ONE component (two boundary loops, C=1), two disks are TWO.

Solvability (compatibility): a right-hand side must integrate to zero
on every component. Incompatible input raises
IncompatibleGridVelocityError -- the componentwise mean is NEVER
silently subtracted (same semantics as the BIE r_comp rejection).

Solver: projected PCG on the componentwise-zero-mean subspace with a
constant-coefficient FFT inverse-Laplacian preconditioner (masked to
the active set and re-projected; symmetric PSD on the subspace). The
FFT lives strictly inside no_grad -- autograd never differentiates
through it (MKL rfft backward is broken in this torch build; see
tests/test_weighted_poisson.py::test_mkl_rfft_backward_canary).

Discrete conventions (fixed here, used by every caller):
    inner product   <u, v> = sum(u * v) * dx^2      (cell measure)
    matvec          (L phi)_c = sum_faces rho_f (phi_c - phi_nbr)/dx^2
    energy          <phi, L phi> = sum_f rho_f (dphi_f)^2 >= 0
so that <delta_rho, L^dagger delta_rho> approximates the continuum
int rho |grad phi|^2 = the Wasserstein tangent energy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

_T = torch.Tensor


@dataclass
class PoissonSolveStats:
    """Cumulative per-operator solve counters (G3a telemetry). The
    operator is rebuilt at every MM-step setup, so these are naturally
    per-step; the stepper snapshots them onto the result. matvec and
    FFT-preconditioner application counts are derivable: each PCG
    iteration is exactly one matvec + one preconditioner apply (plus
    one initial preconditioner apply per solve)."""
    n_forward_solves: int = 0        # public fail-closed solve()
    n_projected_solves: int = 0      # solve_projected (adjoint path)
    pcg_iterations: List[int] = field(default_factory=list)
    wall_seconds: float = 0.0
    max_relative_residual: float = 0.0
    max_inactive_rhs_fraction: float = 0.0


class IncompatibleGridVelocityError(RuntimeError):
    """RHS has nonzero integral on some phase component: the weighted
    Poisson problem is unsolvable there. Fail closed; never subtract
    the mean silently."""

    def __init__(self, message: str, residuals=None):
        super().__init__(message)
        self.residuals = residuals


class WeightedPoissonSolveError(RuntimeError):
    """PCG failed to converge; carries the PCGInfo."""

    def __init__(self, info: "PCGInfo"):
        super().__init__(
            f"weighted Poisson PCG failed: {info.breakdown_reason or ''} "
            f"iters={info.iterations} rel={info.relative_residual:.3e}")
        self.info = info


@dataclass(frozen=True)
class PCGInfo:
    converged: bool
    iterations: int
    absolute_residual: float
    relative_residual: float
    min_pAp: float
    breakdown_reason: Optional[str]


@dataclass(frozen=True)
class WeightedPoissonConfig:
    grid_shape: Tuple[int, int] = (512, 512)
    box_min: Tuple[float, float] = (-2.0, -2.0)
    box_max: Tuple[float, float] = (2.0, 2.0)
    boundary_condition: str = "periodic"
    # MEASURED design choice (2026-08-05; two-sided calibration):
    # - thr=1e-8 (reviewer's first guess): tail cells inject
    #   near-singular conductivities, PCG diverges under refinement
    #   (185/657/DIVERGED over 48/96/192 grids).
    # - thr=1e-2: excellent conditioning (76/102/116 iters) but the
    #   cut tail carries REAL conductivity: a systematic -3% energy
    #   bias on the disk-Fourier audit, insensitive to eps.
    # - thr=1e-3: bias collapses to -0.2..-0.3% (inside the 1% gate)
    #   at ~2.5x the iterations (219-359 at 512^2) -- the adopted
    #   default. Trade-off curve recorded in
    #   results/grid_metric/static_audit.json.
    support_threshold: float = 1e-3
    # componentwise compatibility: |int_C rhs| <= atol + rtol *
    # int_C |rhs| -- PER COMPONENT (G1.1, reviewer: a global scale
    # lets oscillation on a large component mask a violation on a
    # small one, e.g. unequal disks).
    compat_rtol: float = 1e-8
    compat_atol: float = 1e-14
    pcg_rtol: float = 1e-10
    pcg_atol: float = 1e-12
    pcg_max_iter: int = 1000
    # G3a sync-light option: convergence (and curvature-breakdown)
    # checks force a device->host sync; on GPU checking every iteration
    # serializes the loop. Interval 1 preserves the exact legacy
    # check-every-iteration semantics; GPU configs use e.g. 8. A
    # breakdown is then detected up to (interval-1) iterations late --
    # the returned info still reports it and callers still fail closed.
    pcg_check_interval: int = 1
    preconditioner: str = "fft_laplacian"
    # G1 guard (reviewer): fraction of |rhs| mass falling on INACTIVE
    # cells. Scattered boundary flux can land below the support
    # threshold; silently zeroing it would both drop real source mass
    # and (if the compatibility scale included it) mask active-side
    # violations. Fail closed above this fraction; the compatibility
    # normalization uses the ACTIVE |rhs| mass only.
    inactive_rhs_tol: float = 2e-2
    # G0.5 (reviewer): only the CG RECURRENCE is swappable. Everything
    # problem-specific (active set, compatibility fail-closed, gauge
    # projection, harmonic faces, preconditioner, telemetry) lives in
    # the operator and wraps EVERY backend.
    #   native_pcg   -- correctness reference and production default
    #   sparse_direct-- CPU sparse LU (G3a): the operator is FROZEN for
    #                   a whole MM step while trust-ncg requests dozens
    #                   of solves -> factor ONCE, then each solve is
    #                   two triangular sweeps at machine precision.
    #                   Gauge: one pinned cell per component (exact up
    #                   to the removed per-component constant).
    #   scipy_cg     -- CPU-only AUDIT backend (parity checks; never
    #                   used for evolution)
    #   xitorch_cg   -- optional benchmark backend; NOT a main
    #                   dependency (raises with instructions if absent)
    solver_backend: str = "native_pcg"


def connected_components(active: _T) -> Tuple[_T, int]:
    """4-connected components of the active mask on the periodic grid
    (== components of the positive-conductivity face graph, since a
    harmonic-mean face is positive iff both cells are active).
    Pure-torch BFS label propagation; labels are 0..C-1, inactive=-1."""
    H, W = active.shape
    labels = torch.full((H, W), -1, dtype=torch.long,
                        device=active.device)
    comp = 0
    seed = torch.zeros_like(active)
    remaining = active.clone()
    while bool(remaining.any()):
        idx = remaining.nonzero()[0]
        seed.zero_()
        seed[idx[0], idx[1]] = True
        while True:
            grown = (seed
                     | seed.roll(1, 0) | seed.roll(-1, 0)
                     | seed.roll(1, 1) | seed.roll(-1, 1)) & active
            if bool((grown == seed).all()):
                break
            seed = grown
        labels[seed] = comp
        remaining &= ~seed
        comp += 1
    return labels, comp


class WeightedPoissonOperator:
    """Frozen-coefficient symmetric PSD operator + projected PCG."""

    def __init__(self, rho: _T, config: WeightedPoissonConfig,
                 component_labels: "_T | None" = None):
        assert config.boundary_condition == "periodic"
        self.config = config
        H, W = rho.shape
        assert (H, W) == tuple(config.grid_shape)
        self.dx = (config.box_max[0] - config.box_min[0]) / H
        self.cell_area = self.dx * self.dx
        self.rho = rho.detach()

        thr = config.support_threshold
        self.active = self.rho > thr
        if component_labels is None:
            self.labels, self.n_components = \
                connected_components(self.active)
        else:
            # DEFLATION override (soft-bridge merger window): the
            # caller supplies a finer partition of the active set --
            # typically the conservative confidence-sweep labels that
            # keep two blobs separate while the operating-threshold
            # support has just bridged through a weak neck. The
            # MATRIX (conductivities) is untouched; only the
            # projection/gauge/compatibility subspace refines, which
            # removes the soft exchange mode (lambda_soft ~ neck
            # conductance) from the solve space and restores the
            # float64 residual floor.
            lab = component_labels.detach()
            assert lab.shape == self.active.shape
            labeled = lab >= 0
            if bool((labeled & ~self.active).any()):
                raise ValueError(
                    "component_labels override labels cells outside "
                    "the active set")
            if bool((self.active & ~labeled).any()):
                # v8: caller-declared threshold debris (coreless
                # flicker) -- excluded from the system entirely.
                # Vacuum semantics; exact for cells isolated in the
                # positive-conductivity face graph, whose own gauge
                # would otherwise make the factor singular.
                self.active = labeled
            self.labels = lab
            self.n_components = int(lab.max().item()) + 1 \
                if bool(self.active.any()) else 0

        # harmonic-mean face conductivities; a face touching an
        # inactive cell has conductivity EXACTLY zero
        r = self.rho
        rl_x, rr_x = r, r.roll(-1, 0)
        rl_y, rr_y = r, r.roll(-1, 1)
        act_x = self.active & self.active.roll(-1, 0)
        act_y = self.active & self.active.roll(-1, 1)
        self.rho_face_x = torch.where(
            act_x, 2.0 * rl_x * rr_x / (rl_x + rr_x).clamp_min(1e-300),
            torch.zeros_like(r))
        self.rho_face_y = torch.where(
            act_y, 2.0 * rl_y * rr_y / (rl_y + rr_y).clamp_min(1e-300),
            torch.zeros_like(r))

        self._rho_bar = float(self.rho[self.active].mean()) \
            if bool(self.active.any()) else 1.0
        self._k2 = None      # lazy FFT symbol

        # per-component cell counts for gauge projection
        self._comp_masks = [(self.labels == c) for c
                            in range(self.n_components)]
        self._comp_sizes = [int(m.sum()) for m in self._comp_masks]

        self.stats = PoissonSolveStats()
        self._direct_lu = None       # lazy sparse_direct factorization
        self._direct_pins = None

    def _record(self, info: PCGInfo, t0: float, forward: bool):
        st = self.stats
        if forward:
            st.n_forward_solves += 1
        else:
            st.n_projected_solves += 1
        st.pcg_iterations.append(info.iterations)
        st.wall_seconds += time.perf_counter() - t0
        if info.relative_residual == info.relative_residual:  # not NaN
            st.max_relative_residual = max(
                st.max_relative_residual, info.relative_residual)

    # -- core algebra --------------------------------------------------

    def matvec(self, phi: _T) -> _T:
        """(L phi)_c = sum_faces rho_f (phi_c - phi_nbr) / dx^2.
        Symmetric PSD by construction; inactive rows/cols are zero."""
        fx = self.rho_face_x * (phi - phi.roll(-1, 0))
        fy = self.rho_face_y * (phi - phi.roll(-1, 1))
        out = (fx - fx.roll(1, 0)) + (fy - fy.roll(1, 1))
        return out / (self.dx * self.dx)

    def inner(self, u: _T, v: _T) -> _T:
        return (u * v).sum() * self.cell_area

    def project_componentwise_zero_mean(self, x: _T) -> _T:
        """Remove the constant (gauge) mode on each phase component;
        zero out inactive cells."""
        out = torch.where(self.active, x, torch.zeros_like(x))
        for m, n in zip(self._comp_masks, self._comp_sizes):
            if n > 0:
                out = torch.where(
                    m, out - out[m].sum() / n, out)
        return out

    def compatibility_residuals(self, rhs: _T) -> _T:
        """Integral of rhs over each phase component (cell measure)."""
        vals = [rhs[m].sum() * self.cell_area for m in self._comp_masks]
        return torch.stack(vals) if vals else rhs.new_zeros(0)

    # -- preconditioner ------------------------------------------------

    def _precondition(self, r: _T) -> _T:
        """(-rho_bar * Delta_h)^{-1} via FFT on the full grid, then
        mask + gauge-project (symmetric PSD on the subspace). Strictly
        no_grad; forward FFT only."""
        if self.config.preconditioner == "none":
            return r.clone()
        with torch.no_grad():
            H, W = r.shape
            if self._k2 is None:
                kx = 2.0 * torch.pi * torch.fft.fftfreq(
                    H, d=self.dx).to(device=r.device, dtype=r.dtype)
                ky = 2.0 * torch.pi * torch.fft.fftfreq(
                    W, d=self.dx).to(device=r.device, dtype=r.dtype)
                # symbol of the DISCRETE 5-point Laplacian
                k2 = ((2 - 2 * torch.cos(kx * self.dx))[:, None]
                      + (2 - 2 * torch.cos(ky * self.dx))[None, :]) \
                    / (self.dx * self.dx)
                k2[0, 0] = 1.0
                self._k2 = k2
            F = torch.fft.fft2(r.to(torch.complex128))
            F = F / (self._rho_bar * self._k2)
            F[0, 0] = 0.0
            out = torch.real(torch.fft.ifft2(F)).to(r.dtype)
        return self.project_componentwise_zero_mean(out)

    # -- solve ---------------------------------------------------------

    def solve(self, rhs: _T) -> Tuple[_T, PCGInfo]:
        """Solve on the componentwise-zero-mean subspace via the
        configured backend. Compatibility rejection (fail closed) and
        all projections are backend-independent."""
        cfg = self.config
        # inactive-RHS guard (G1): source mass on vacuum cells is real
        # mass the operator cannot see -- reject rather than silently
        # zero it, and keep the compatibility scale ACTIVE-only so
        # inactive mass cannot mask an active-side violation.
        abs_total = float(rhs.abs().sum()) * self.cell_area
        abs_active = float(rhs[self.active].abs().sum()) * self.cell_area
        self.last_inactive_rhs_fraction = (
            (abs_total - abs_active) / (abs_total + 1e-300))
        self.stats.max_inactive_rhs_fraction = max(
            self.stats.max_inactive_rhs_fraction,
            self.last_inactive_rhs_fraction)
        if self.last_inactive_rhs_fraction > cfg.inactive_rhs_tol:
            raise IncompatibleGridVelocityError(
                f"{self.last_inactive_rhs_fraction:.3e} of the |rhs| "
                f"mass falls on inactive cells (tol "
                f"{cfg.inactive_rhs_tol:g}): the scattered source is "
                f"not resolved by the active support")
        res = self.compatibility_residuals(rhs)
        scales = torch.stack([
            rhs[m].abs().sum() * self.cell_area
            for m in self._comp_masks]) if self._comp_masks else res
        bad = res.abs() > (cfg.compat_atol + cfg.compat_rtol * scales)
        if bool(bad.any()):
            raise IncompatibleGridVelocityError(
                f"componentwise compatibility violated "
                f"(per-component |int rhs| <= atol + rtol int |rhs|): "
                f"residuals "
                f"{[f'{float(x):.2e}' for x in res]} vs scales "
                f"{[f'{float(x):.2e}' for x in scales]}",
                residuals=res)

        t0 = time.perf_counter()
        if cfg.solver_backend == "native_pcg":
            phi, info = self._solve_native(rhs)
        elif cfg.solver_backend == "sparse_direct":
            phi, info = self._solve_direct(rhs)
        elif cfg.solver_backend == "scipy_cg":
            phi, info = self._solve_scipy(rhs)
        elif cfg.solver_backend == "xitorch_cg":
            phi, info = self._solve_xitorch(rhs)
        else:
            raise ValueError(
                f"solver_backend must be 'native_pcg', 'sparse_direct',"
                f" 'scipy_cg' or 'xitorch_cg'; got "
                f"{cfg.solver_backend!r}")
        self._record(info, t0, forward=True)
        return phi, info

    def _solve_native(self, rhs: _T) -> Tuple[_T, PCGInfo]:
        """Reference projected PCG (G0-gated recurrence). G3a: scalar
        recurrence quantities stay device-resident 0-dim tensors; the
        only host syncs are the convergence/breakdown checks, taken
        every pcg_check_interval iterations (interval 1 == the legacy
        check-every-iteration semantics; identical fp64 arithmetic).
        With interval > 1 a nonpositive-curvature breakdown can pollute
        x for up to interval-1 iterations before detection -- the info
        still reports it and every caller fails closed on it."""
        cfg = self.config
        with torch.no_grad():
            b = self.project_componentwise_zero_mean(rhs)
            x = torch.zeros_like(b)
            r = b.clone()
            b_norm = float(torch.sqrt(self.inner(b, b)))
            if b_norm == 0.0:
                return x, PCGInfo(True, 0, 0.0, 0.0,
                                  float("inf"), None)
            tol = max(cfg.pcg_rtol * b_norm, cfg.pcg_atol)
            ck = max(1, int(cfg.pcg_check_interval))
            z = self._precondition(r)
            p = z.clone()
            rz = self.inner(r, z)
            min_pAp = None
            for it in range(1, cfg.pcg_max_iter + 1):
                Ap = self.matvec(p)
                pAp = self.inner(p, Ap)
                min_pAp = pAp if min_pAp is None else \
                    torch.minimum(min_pAp, pAp)
                alpha = rz / pAp
                x = x + alpha * p
                r = r - alpha * Ap
                r = self.project_componentwise_zero_mean(r)
                if it % ck == 0 or it == cfg.pcg_max_iter:
                    if float(min_pAp) <= 0.0:
                        return x, PCGInfo(
                            False, it, float("nan"), float("nan"),
                            float(min_pAp),
                            "nonpositive curvature (operator not PSD "
                            "on subspace -- bug)")
                    r_norm = float(torch.sqrt(self.inner(r, r)))
                    if r_norm <= tol:
                        return x, PCGInfo(True, it, r_norm,
                                          r_norm / b_norm,
                                          float(min_pAp), None)
                z = self._precondition(r)
                rz_new = self.inner(r, z)
                beta = rz_new / rz
                p = z + beta * p
                rz = rz_new
            r_norm = float(torch.sqrt(self.inner(r, r)))
            info = PCGInfo(False, cfg.pcg_max_iter, r_norm,
                           r_norm / b_norm, float(min_pAp), "max_iter")
            return x, info

    def _direct_factorize(self):
        """Lazy one-time sparse LU of the active submatrix with one
        gauge-pinned cell per component (adds e_i e_i^T for the FIRST
        active cell of each component). For a componentwise-zero-mean
        rhs the pinned system reproduces the exact projected solution
        up to per-component constants, removed by the final gauge
        projection. CPU float64 only."""
        if self._direct_lu is not None:
            return
        import numpy as np
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import splu

        if self.rho.device.type != "cpu":
            raise ValueError("sparse_direct is a CPU-only backend")
        if self.rho.dtype != torch.float64:
            raise ValueError("sparse_direct requires float64")
        H, W = self.rho.shape
        inv_dx2 = 1.0 / (self.dx * self.dx)
        idx = torch.full((H, W), -1, dtype=torch.long)
        n_act = int(self.active.sum())
        idx[self.active] = torch.arange(n_act)

        # diagonal == matvec's coefficient of phi_c
        diag = (self.rho_face_x + self.rho_face_x.roll(1, 0)
                + self.rho_face_y + self.rho_face_y.roll(1, 1)) \
            * inv_dx2
        rows = [idx[self.active]]
        cols = [idx[self.active]]
        vals = [diag[self.active]]
        for face, dim in ((self.rho_face_x, 0), (self.rho_face_y, 1)):
            m = face > 0
            a, bb = idx[m], idx.roll(-1, dim)[m]
            w = -face[m] * inv_dx2
            rows += [a, bb]
            cols += [bb, a]
            vals += [w, w]
        pins = torch.stack([idx[mk].min() for mk in self._comp_masks])
        rows.append(pins)
        cols.append(pins)
        vals.append(torch.ones(len(pins), dtype=self.rho.dtype))

        A = coo_matrix(
            (torch.cat(vals).numpy(),
             (torch.cat(rows).numpy(), torch.cat(cols).numpy())),
            shape=(n_act, n_act)).tocsc()
        self._direct_lu = splu(A)
        self._direct_pins = pins

    def _solve_direct(self, rhs: _T) -> Tuple[_T, PCGInfo]:
        """G3a sparse-direct backend: factor once per frozen operator,
        then each solve is two triangular sweeps. The true residual is
        measured against the SAME criterion as the PCG backends;
        converged=False fails closed upstream as usual.

        Iterative refinement (2026-08-07): in the near-contact window
        the phase develops thin high-contrast walls and the LU
        conditioning degrades -- the Otto pair run measured the
        single-pass residual rising from the usual 1e-13 to 1.8e-9 at
        gap ~ 2*eps, tripping the fail-closed tolerance. Standard
        fixed-point refinement (solve on the true residual, add the
        correction; each pass is one extra pair of triangular sweeps)
        restores the digits without touching the tolerance. The pass
        count is recorded in PCGInfo.iterations (0 = clean first
        solve, unchanged bitwise in the well-conditioned regime)."""
        import numpy as np

        cfg = self.config
        self._direct_factorize()

        def _lu_apply(vec: _T) -> _T:
            x_act = self._direct_lu.solve(vec[self.active].numpy())
            out = torch.zeros_like(vec)
            out[self.active] = torch.from_numpy(
                np.ascontiguousarray(x_act)).to(vec.dtype)
            return self.project_componentwise_zero_mean(out)

        with torch.no_grad():
            b = self.project_componentwise_zero_mean(rhs)
            b_norm = float(torch.sqrt(self.inner(b, b)))
            if b_norm == 0.0:
                return torch.zeros_like(rhs), PCGInfo(
                    True, 0, 0.0, 0.0, float("inf"), None)
            phi = _lu_apply(b)
            r = self.project_componentwise_zero_mean(
                b - self.matvec(phi))
            r_norm = float(torch.sqrt(self.inner(r, r)))
            tol = max(cfg.pcg_rtol * b_norm, cfg.pcg_atol)
            n_ref = 0
            while r_norm > tol and n_ref < 3:
                phi = self.project_componentwise_zero_mean(
                    phi + _lu_apply(r))
                r = self.project_componentwise_zero_mean(
                    b - self.matvec(phi))
                r_new = float(torch.sqrt(self.inner(r, r)))
                n_ref += 1
                if r_new >= 0.5 * r_norm:      # stagnation: stop trying
                    r_norm = r_new
                    break
                r_norm = r_new
            if r_norm > tol:
                # LU-preconditioned CG polish (2026-08-07): when the
                # support bridges during a merger, the two phase blobs
                # couple through a weak conductive neck and the
                # operator acquires a SOFT EXCHANGE MODE
                # (lambda_soft ~ neck conductance) -- plain refinement
                # stagnates at ~eps_mach * cond (measured 3.1e-10 at
                # the Otto pair's step 743, gap ~ 2*eps). The LU stays
                # an excellent preconditioner there, so a short CG in
                # the LU-preconditioned metric recovers the digits
                # without touching any tolerance.
                p_vec = None
                z = _lu_apply(r)
                rz = float(self.inner(r, z))
                n_cg = 0
                while r_norm > tol and n_cg < 50 and rz > 0:
                    p_vec = (z if p_vec is None
                             else z + (rz / rz_old) * p_vec)
                    Ap = self.project_componentwise_zero_mean(
                        self.matvec(p_vec))
                    pAp = float(self.inner(p_vec, Ap))
                    if pAp <= 0:
                        break
                    alpha = rz / pAp
                    phi = phi + alpha * p_vec
                    r = r - alpha * Ap
                    r_norm = float(torch.sqrt(self.inner(r, r)))
                    z = _lu_apply(r)
                    rz_old, rz = rz, float(self.inner(r, z))
                    n_cg += 1
                phi = self.project_componentwise_zero_mean(phi)
                r = self.project_componentwise_zero_mean(
                    b - self.matvec(phi))
                r_norm = float(torch.sqrt(self.inner(r, r)))
                n_ref += n_cg
            ok = r_norm <= tol
            return phi, PCGInfo(ok, n_ref, r_norm, r_norm / b_norm,
                                float("inf"),
                                None if ok else "direct residual above "
                                "pcg tolerance after refinement + "
                                "LU-preconditioned CG polish")

    def _solve_scipy(self, rhs: _T) -> Tuple[_T, PCGInfo]:
        """CPU AUDIT backend: scipy.sparse.linalg.cg over the same
        matvec/projection/preconditioner. Never used for evolution;
        exists to cross-check the native recurrence."""
        import numpy as np
        from scipy.sparse.linalg import LinearOperator, cg

        if rhs.device.type != "cpu":
            raise ValueError("scipy_cg is a CPU-audit-only backend")
        if rhs.dtype != torch.float64:
            raise ValueError("scipy_cg audit backend requires float64")
        cfg = self.config
        H, W = rhs.shape
        dtype = rhs.dtype

        def mv(x):
            t = torch.from_numpy(np.asarray(x)).to(dtype).reshape(H, W)
            t = self.project_componentwise_zero_mean(t)
            out = self.matvec(t)
            out = self.project_componentwise_zero_mean(out)
            return out.reshape(-1).numpy()

        def pc(x):
            t = torch.from_numpy(np.asarray(x)).to(dtype).reshape(H, W)
            return self._precondition(t).reshape(-1).numpy()

        b = self.project_componentwise_zero_mean(rhs)
        b_norm = float(torch.sqrt(self.inner(b, b)))
        if b_norm == 0.0:
            return torch.zeros_like(rhs), PCGInfo(
                True, 0, 0.0, 0.0, float("inf"), None)
        A = LinearOperator((H * W, H * W), matvec=mv,
                           dtype=np.float64)
        M = LinearOperator((H * W, H * W), matvec=pc,
                           dtype=np.float64)
        it_count = [0]

        def cb(_):
            it_count[0] += 1

        x, code = cg(A, b.reshape(-1).numpy(), M=M,
                     rtol=cfg.pcg_rtol, atol=cfg.pcg_atol,
                     maxiter=cfg.pcg_max_iter, callback=cb)
        phi = torch.from_numpy(x).to(dtype).reshape(H, W)
        phi = self.project_componentwise_zero_mean(phi)
        r = b - self.matvec(phi)
        r = self.project_componentwise_zero_mean(r)
        r_norm = float(torch.sqrt(self.inner(r, r)))
        return phi, PCGInfo(code == 0, it_count[0], r_norm,
                            r_norm / b_norm, float("nan"),
                            None if code == 0 else f"scipy code {code}")

    def _solve_xitorch(self, rhs: _T) -> Tuple[_T, PCGInfo]:
        """Optional benchmark backend. xitorch is deliberately NOT a
        main dependency; promote only after residual/energy parity and
        a measured wall/synchronization win."""
        try:
            import xitorch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "solver_backend='xitorch_cg' requires the optional "
                "xitorch package (benchmark environments only; do not "
                "add it to the main dependencies)") from e
        raise NotImplementedError(
            "xitorch_cg backend is a benchmark stub; implement in an "
            "optional environment per the G0.5 promotion criteria")

    def solve_projected(self, rhs: _T) -> Tuple[_T, PCGInfo]:
        """INTERNAL/backward API (G1.1): project an arbitrary cotangent
        onto the active componentwise-zero-mean subspace and solve --
        no compatibility or inactive-mass gates. The public solve()
        keeps its fail-closed semantics for forward RHS; backward
        cotangents are generically incompatible and must be projected,
        because the adjoint of the projected inverse is
        L^dagger P_comp (never a reason to reject).

        Adjoint backend policy (G3a): sparse_direct moves WITH the
        forward (same exact factorization, symmetric operator -- the
        adjoint of an exact solve is the same exact solve); every other
        configuration keeps the native reference recurrence (audit
        backends never enter the adjoint)."""
        t0 = time.perf_counter()
        b = self.project_componentwise_zero_mean(rhs)
        if self.config.solver_backend == "sparse_direct":
            phi, info = self._solve_direct(b)
        else:
            phi, info = self._solve_native(b)
        self._record(info, t0, forward=False)
        return phi, info

    def solve_or_raise(self, rhs: _T) -> _T:
        phi, info = self.solve(rhs)
        if not info.converged:
            raise WeightedPoissonSolveError(info)
        return phi

    def energy(self, rhs: _T) -> _T:
        """<rhs, L^dagger rhs> -- the tangent metric quadratic form."""
        phi = self.solve_or_raise(rhs)
        return self.inner(rhs, phi)
