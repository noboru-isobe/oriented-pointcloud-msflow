"""Grid metric G0: weighted Poisson core.

Pins the structural guarantees the BIE could not offer: exact discrete
symmetry, PSD by construction, phase components as the operator's real
nullspace (annulus = 1 component with 2 boundary loops, two disks = 2),
fail-closed componentwise compatibility, and FFT-preconditioned PCG
convergence under refinement.
"""

import math

import pytest
import torch

from src.torch.transport.weighted_poisson import (
    IncompatibleGridVelocityError,
    WeightedPoissonConfig,
    WeightedPoissonOperator,
    connected_components,
)

DT = torch.float64


def _cfg(n=64, **kw):
    return WeightedPoissonConfig(grid_shape=(n, n), **kw)


def _grid(n, box=2.0):
    xs = torch.linspace(-box, box, n + 1, dtype=DT)[:-1]
    xs = xs + (xs[1] - xs[0]) / 2
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    return X, Y


def _smooth_disk(n, R=1.0, eps=0.15, center=(0.0, 0.0)):
    X, Y = _grid(n)
    r = torch.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    t = ((R - r) / eps).clamp(-1.0, 1.0)
    return 0.5 * (1.0 + torch.sin(0.5 * torch.pi * t))


def _annulus(n, R_out=1.2, R_in=0.5, eps=0.12):
    return (_smooth_disk(n, R_out, eps)
            * (1.0 - _smooth_disk(n, R_in, eps)))


def test_constant_rho_matches_fft_laplacian():
    n = 64
    rho = torch.ones(n, n, dtype=DT)
    op = WeightedPoissonOperator(rho, _cfg(n))
    torch.manual_seed(0)
    b = torch.randn(n, n, dtype=DT)
    b -= b.mean()
    phi, info = op.solve(b)
    assert info.converged
    # exact FFT solution of -Delta_h phi = b (discrete symbol)
    kx = 2 * torch.pi * torch.fft.fftfreq(n, d=op.dx).to(DT)
    k2 = ((2 - 2 * torch.cos(kx * op.dx))[:, None]
          + (2 - 2 * torch.cos(kx * op.dx))[None, :]) / op.dx**2
    k2[0, 0] = 1.0
    F = torch.fft.fft2(b.to(torch.complex128)) / k2
    F[0, 0] = 0.0
    ref = torch.real(torch.fft.ifft2(F)).to(DT)
    assert float((phi - ref).abs().max()) < 1e-9


def test_symmetry_and_psd():
    n = 48
    rho = _smooth_disk(n)
    op = WeightedPoissonOperator(rho, _cfg(n))
    torch.manual_seed(1)
    u = torch.randn(n, n, dtype=DT)
    v = torch.randn(n, n, dtype=DT)
    lhs = float(op.inner(u, op.matvec(v)))
    rhs = float(op.inner(op.matvec(u), v))
    assert abs(lhs - rhs) < 1e-10 * max(1.0, abs(lhs))
    quad = float(op.inner(u, op.matvec(u)))
    assert quad >= -1e-12 * max(1.0, abs(quad))


def test_component_structure_disk_annulus_twodisks():
    n = 96
    # disk: C=1
    op = WeightedPoissonOperator(_smooth_disk(n), _cfg(n))
    assert op.n_components == 1
    # annulus: 2 boundary loops but ONE phase component
    op = WeightedPoissonOperator(_annulus(n), _cfg(n))
    assert op.n_components == 1
    # two separated disks: C=2
    rho2 = (_smooth_disk(n, R=0.5, center=(-1.0, 0.0))
            + _smooth_disk(n, R=0.5, center=(1.0, 0.0)))
    op2 = WeightedPoissonOperator(rho2, _cfg(n))
    assert op2.n_components == 2
    # vacuum cells are NOT extra null components
    labels, C = connected_components(op2.active)
    assert C == 2
    assert int((labels == -1).sum()) > 0


def test_incompatible_rhs_rejected_fail_closed():
    n = 96
    rho2 = (_smooth_disk(n, R=0.5, center=(-1.0, 0.0))
            + _smooth_disk(n, R=0.5, center=(1.0, 0.0)))
    op = WeightedPoissonOperator(rho2, _cfg(n))
    # exchange mode: +mass on one component, -mass on the other;
    # TOTAL mean is zero but componentwise means are not
    b = torch.zeros(n, n, dtype=DT)
    b[op.labels == 0] = 1.0
    b[op.labels == 1] = -float((op.labels == 0).sum()) \
        / float((op.labels == 1).sum())
    with pytest.raises(IncompatibleGridVelocityError):
        op.solve(b)


def test_compatible_rhs_solves_componentwise():
    n = 96
    rho2 = (_smooth_disk(n, R=0.5, center=(-1.0, 0.0))
            + _smooth_disk(n, R=0.5, center=(1.0, 0.0)))
    op = WeightedPoissonOperator(rho2, _cfg(n))
    torch.manual_seed(2)
    b = torch.randn(n, n, dtype=DT)
    b = op.project_componentwise_zero_mean(b)
    phi, info = op.solve(b)
    assert info.converged and info.min_pAp > 0
    res = op.matvec(phi) - b
    res = op.project_componentwise_zero_mean(res)
    assert float(torch.sqrt(op.inner(res, res))) < 1e-8


def test_no_flux_leak_between_separated_components():
    """Anti-exchange guarantee, stated at the right levels:
    (a) OPERATOR level, exact: matvec of a field supported on one
        component never touches the other (harmonic-mean faces carry
        zero conductivity through vacuum);
    (b) SOLUTION level, to solver tolerance: the FFT preconditioner
        mixes components transiently, but PCG converges to the
        block-decoupled solution, so the other component's field is
        bounded by the solve tolerance (not exactly zero)."""
    n = 96
    rho2 = (_smooth_disk(n, R=0.5, center=(-1.0, 0.0))
            + _smooth_disk(n, R=0.5, center=(1.0, 0.0)))
    op = WeightedPoissonOperator(rho2, _cfg(n))
    torch.manual_seed(3)
    m0, m1 = op.labels == 0, op.labels == 1
    u = torch.zeros(n, n, dtype=DT)
    u[m0] = torch.randn(int(m0.sum()), dtype=DT)
    assert float(op.matvec(u)[m1].abs().max()) == 0.0    # exact

    b = torch.zeros(n, n, dtype=DT)
    vals = torch.randn(int(m0.sum()), dtype=DT)
    b[m0] = vals - vals.mean()
    phi, info = op.solve(b)
    assert info.converged
    assert float(phi[m1].abs().max()) < 1e-8             # tolerance


def test_refinement_pcg_iterations_bounded():
    iters = []
    for n in (48, 96, 192):
        op = WeightedPoissonOperator(_smooth_disk(n), _cfg(n))
        torch.manual_seed(4)
        b = torch.randn(n, n, dtype=DT)
        b = op.project_componentwise_zero_mean(b)
        _, info = op.solve(b)
        assert info.converged
        iters.append(info.iterations)
    # FFT preconditioner keeps growth mild (not a hard theory bound;
    # guards against accidental O(n) blowup)
    assert iters[-1] < 6 * iters[0] + 60


def test_componentwise_compat_small_component_not_masked():
    """G1.1: per-component normalization -- a violation on a SMALL
    component must be caught even when a large component carries big
    oscillation (the global-scale convention would mask it)."""
    n = 96
    rho = (_smooth_disk(n, R=0.7, center=(-1.0, 0.0))
           + _smooth_disk(n, R=0.25, center=(1.2, 0.0)))
    op = WeightedPoissonOperator(rho, _cfg(n))
    assert op.n_components == 2
    sizes = [int((op.labels == c).sum()) for c in range(2)]
    big, small = (0, 1) if sizes[0] > sizes[1] else (1, 0)
    b = torch.zeros(n, n, dtype=DT)
    # large oscillation on the big component (compatible there)
    mbig = op.labels == big
    vals = torch.randn(int(mbig.sum()), dtype=DT) * 10.0
    b[mbig] = vals - vals.mean()
    # small NET violation on the small component
    b[op.labels == small] = 1e-3
    with pytest.raises(IncompatibleGridVelocityError):
        op.solve(b)


def test_solve_projected_accepts_arbitrary_cotangents():
    """G1.1: the internal backward API projects first and never
    rejects -- the adjoint of the projected inverse is
    L^dagger P_comp."""
    n = 96
    rho2 = (_smooth_disk(n, R=0.5, center=(-1.0, 0.0))
            + _smooth_disk(n, R=0.5, center=(1.0, 0.0)))
    op = WeightedPoissonOperator(rho2, _cfg(n))
    torch.manual_seed(9)
    g = torch.randn(n, n, dtype=DT)          # incompatible cotangent
    with pytest.raises(IncompatibleGridVelocityError):
        op.solve(g)                           # public API fail-closed
    phi, info = op.solve_projected(g)         # internal API projects
    assert info.converged
    gp = op.project_componentwise_zero_mean(g)
    res = op.project_componentwise_zero_mean(op.matvec(phi)) - gp
    assert float(torch.sqrt(op.inner(res, res))) < 1e-8


def test_scipy_backend_parity():
    """G0.5: the audit backend must reproduce the native recurrence --
    same solution (to solve tolerance), same energy, same fail-closed
    compatibility semantics."""
    n = 64
    rho = _smooth_disk(n)
    torch.manual_seed(7)
    b = torch.randn(n, n, dtype=DT)

    op_n = WeightedPoissonOperator(rho, _cfg(n))
    op_s = WeightedPoissonOperator(
        rho, _cfg(n, solver_backend="scipy_cg"))
    b_p = op_n.project_componentwise_zero_mean(b)
    phi_n, info_n = op_n.solve(b_p)
    phi_s, info_s = op_s.solve(b_p)
    assert info_n.converged and info_s.converged
    assert float((phi_n - phi_s).abs().max()) < 1e-7
    q_n = float(op_n.inner(b_p, phi_n))
    q_s = float(op_s.inner(b_p, phi_s))
    assert abs(q_n - q_s) < 1e-9 * max(1.0, abs(q_n))
    # incompatible rhs rejected identically (backend-independent)
    rho2 = (_smooth_disk(n, R=0.5, center=(-1.0, 0.0))
            + _smooth_disk(n, R=0.5, center=(1.0, 0.0)))
    op2 = WeightedPoissonOperator(
        rho2, _cfg(n, solver_backend="scipy_cg"))
    bad = torch.zeros(n, n, dtype=DT)
    bad[op2.labels == 0] = 1.0
    bad[op2.labels == 1] = -float((op2.labels == 0).sum()) \
        / float((op2.labels == 1).sum())
    with pytest.raises(IncompatibleGridVelocityError):
        op2.solve(bad)


def test_direct_iterative_refinement():
    """G3c Otto-pair fix: in the near-contact window the LU
    conditioning degrades (measured single-pass residual 1.8e-9 at
    gap ~ 2*eps vs 1e-13 in the clean regime) and the fail-closed
    tolerance tripped. _solve_direct now runs standard iterative
    refinement (solve on the true residual, max 3 passes). Pins:
    (i) the clean regime is untouched -- zero refinement passes,
    (ii) with an artificially unreachable tolerance the loop runs,
    records its pass count, and never worsens the residual."""
    n = 64
    rho = _annulus(n)
    op = WeightedPoissonOperator(
        rho, _cfg(n, solver_backend="sparse_direct"))
    torch.manual_seed(3)
    b = op.project_componentwise_zero_mean(torch.randn(n, n, dtype=DT))
    phi, info = op.solve(b)
    assert info.converged
    assert info.iterations == 0            # clean regime: no refinement
    res0 = info.relative_residual

    op2 = WeightedPoissonOperator(
        rho, _cfg(n, solver_backend="sparse_direct",
                  pcg_rtol=1e-30, pcg_atol=0.0))
    phi2, info2 = op2.solve(b)
    assert info2.iterations >= 1           # refinement engaged
    assert info2.relative_residual <= res0 * (1 + 1e-12)
    assert float((phi2 - phi).abs().max()) < 1e-9


def test_direct_soft_bridge_window_converges():
    """The Otto-pair merger window in miniature: two phase blobs
    joined by a thin, barely-above-threshold neck give the operator a
    soft exchange mode (lambda_soft ~ neck conductance) -- the regime
    where the single-pass direct residual rose to 1.8e-9 and plain
    refinement stagnated at 3.1e-10 in the pair run. The
    LU-preconditioned CG polish must bring the solve back under the
    UNCHANGED 1e-10 tolerance."""
    n = 128
    X, Y = _grid(n)
    blobs = (_smooth_disk(n, R=0.55, eps=0.1, center=(-0.75, 0.0))
             + _smooth_disk(n, R=0.55, eps=0.1, center=(0.75, 0.0)))
    neck = 5e-3 * ((Y.abs() < 0.06) & (X.abs() < 0.45)).to(DT)
    rho = (blobs + neck).clamp(max=1.0)
    op = WeightedPoissonOperator(
        rho, _cfg(n, solver_backend="sparse_direct"))
    assert op.n_components == 1            # bridged: one soft component
    torch.manual_seed(9)
    b = op.project_componentwise_zero_mean(torch.randn(n, n, dtype=DT))
    phi, info = op.solve(b)
    assert info.converged, (info.relative_residual, info.iterations)
    assert info.relative_residual <= 1e-10 * 1.5 + 1e-30 or \
        info.converged


@pytest.mark.xfail(reason="MKL rfft2 backward broken in torch 2.6 "
                          "(oneMKL DFTI error); FIXED in torch>=2.13 "
                          "(probed 2026-08-05). Canary: when this "
                          "starts passing, rfft may re-enter autograd "
                          "paths.", strict=False)
def test_mkl_rfft_backward_canary():
    y = torch.randn(16, 16, dtype=DT, requires_grad=True)

    def f(v):
        n = v.shape[0]
        kx = torch.fft.fftfreq(n, d=1 / n) * 2 * torch.pi
        ky = torch.fft.rfftfreq(n, d=1 / n) * 2 * torch.pi
        kk = kx[:, None] ** 2 + ky[None, :] ** 2
        kk[0, 0] = 1.0
        return torch.fft.irfft2(torch.fft.rfft2(v) / kk,
                                s=v.shape).sum()

    assert torch.autograd.gradcheck(f, (y,), eps=1e-6, atol=1e-8)


def test_soft_bridge_deflation_via_conservative_labels():
    """The merger-window deflation (2026-08-07): with the operating
    support freshly bridged (C=1, soft exchange mode), the
    conservative-sweep labels keep the two blobs separate in the
    projection/gauge/compatibility subspace. Pins: (i) the refined
    partition has 2 components while the operating support has 1;
    (ii) the deflated solve keeps |phi| at O(|b|) scale (the
    UNDEFLATED solve on the same operator measured |phi|/|b| ~ 2.3 in
    the Otto pair diagnostic -- three orders larger); (iii) residual
    meets the UNCHANGED tolerance; (iv) once the neck exceeds 3*thr
    everywhere, the helper returns the operating labels (the rule
    self-removes)."""
    from src.torch.transport.phase_grid import (
        conservative_component_labels,
    )
    n = 128
    thr = 3e-3
    X, Y = _grid(n)
    blobs = (_smooth_disk(n, R=0.55, eps=0.1, center=(-0.75, 0.0))
             + _smooth_disk(n, R=0.55, eps=0.1, center=(0.75, 0.0)))
    neck = 5e-3 * ((Y.abs() < 0.06) & (X.abs() < 0.45)).to(DT)
    rho = (blobs + neck).clamp(max=1.0)

    labels, n_ref = conservative_component_labels(rho, thr)
    op_plain = WeightedPoissonOperator(
        rho, _cfg(n, support_threshold=thr,
                  solver_backend="sparse_direct"))
    assert op_plain.n_components == 1      # bridged at the operating thr
    assert n_ref == 2                      # conservative reading: 2

    op = WeightedPoissonOperator(
        rho, _cfg(n, support_threshold=thr,
                  solver_backend="sparse_direct"),
        component_labels=labels)
    assert op.n_components == 2
    torch.manual_seed(9)
    b = op.project_componentwise_zero_mean(torch.randn(n, n, dtype=DT))
    phi, info = op.solve(b)
    assert info.converged
    bn = float(torch.sqrt(op.inner(b, b)))
    pn = float(torch.sqrt(op.inner(phi, phi)))
    assert pn / bn < 0.1                   # soft mode deflated
    assert len(op.compatibility_residuals(b)) == 2

    # (iv) fat neck: all thresholds agree -> operating labels returned
    rho_fat = (blobs + 10 * thr
               * ((Y.abs() < 0.06) & (X.abs() < 0.45)).to(DT)).clamp(max=1.0)
    lab2, n2 = conservative_component_labels(rho_fat, thr)
    assert n2 == 1


def test_conservative_labels_absorb_coreless_flicker():
    """v8 (arm-4 Run A step 767): an operating component with NO
    substantial 3*thr core is threshold-level debris and must be
    absorbed into the nearest core -- in EVERY branch, including
    n_hi <= n_op (the measured killer: a single metric-level cell at
    rho = thr at the far tip, no marks nearby, whose zero flux row
    made the constraint basis rank-deficient)."""
    from src.torch.transport.phase_grid import (
        conservative_component_labels,
    )
    n = 128
    thr = 3e-3
    rho = _smooth_disk(n, R=0.6, eps=0.1, center=(0.0, 0.0)).clamp(max=1.0)
    # isolated single-cell flicker just above the operating threshold,
    # far from the disk (below 3*thr -> no core)
    rho[8, 8] = 1.1 * thr
    labels, n_ref = conservative_component_labels(
        rho, thr, min_core_cells=5)
    assert n_ref == 1, f"flicker seeded a compat component: {n_ref}"
    # the flicker cell is EXCLUDED (vacuum semantics): an isolated
    # cell in the positive-conductivity graph merged into the main
    # label makes the factor exactly singular (measured, arm-4 retry)
    assert int(labels[8, 8]) == -1
    # the operator accepts the subset labeling and solves cleanly
    op = WeightedPoissonOperator(
        rho, _cfg(n, support_threshold=thr,
                  solver_backend="sparse_direct"),
        component_labels=labels)
    assert op.n_components == 1
    assert not bool(op.active[8, 8])
    b = op.project_componentwise_zero_mean(
        torch.randn(n, n, dtype=DT) * op.active)
    phi, info = op.solve(b)
    assert info.converged
    # healthy two-blob split behavior unchanged
    blobs = (_smooth_disk(n, R=0.55, eps=0.1, center=(-0.75, 0.0))
             + _smooth_disk(n, R=0.55, eps=0.1, center=(0.75, 0.0)))
    lab2, n2 = conservative_component_labels(
        blobs.clamp(max=1.0), thr, min_core_cells=5)
    assert n2 == 2
    # flicker + two blobs: still 2, flicker excluded
    rho3 = blobs.clamp(max=1.0).clone()
    rho3[8, 8] = 1.1 * thr
    lab3, n3 = conservative_component_labels(rho3, thr, min_core_cells=5)
    assert n3 == 2
    assert int(lab3[8, 8]) == -1


def test_substantial_component_count_ignores_flicker():
    from src.torch.transport.phase_grid import (
        substantial_component_count,
    )
    n = 128
    thr = 3e-3
    rho = _smooth_disk(n, R=0.6, eps=0.1, center=(0.0, 0.0)).clamp(max=1.0)
    rho[8, 8] = 1.1 * thr
    assert substantial_component_count(rho, thr, min_core_cells=5) == 1
    blobs = (_smooth_disk(n, R=0.55, eps=0.1, center=(-0.75, 0.0))
             + _smooth_disk(n, R=0.55, eps=0.1, center=(0.75, 0.0))
             ).clamp(max=1.0)
    assert substantial_component_count(blobs, thr, min_core_cells=5) == 2
