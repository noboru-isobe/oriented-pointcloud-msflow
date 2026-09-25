"""Grid metric G1: boundary-flux tangent map B (candidate-dependent).

The tangent source of the filling map under normal motion is the
mollified boundary flux (the continuum identity
d/dt (eta * chi_{E_t}) = eta * (v H^1|_{dE_t})):

    delta_rho(x) = sum_{i,k} q^pol_i zeta_ik (s_i - r_ik dtheta_i)
                                          eta_eps(x - p_ik)

built from the SAME endpoint conventions as the BEM path (positions
p_ik, quadrature weights zeta_ik = m^op_i / K, physical offsets r_ik,
operator-measure and q-velocity policies) and the SAME kernel eta_eps
as the rho0 filling (one kernel, or the identity above breaks). This
map -- not a JVP of the filling with frozen masses -- is the correct
linearization: differentiating the candidate current at frozen m_i
misses the arc-length variation and represents no set's tangent.

The scatter geometry (p_ik, kernel weights) is FROZEN pre-step data;
__call__ is a fixed linear map of (s, dtheta), differentiable and
double-backward-trivial by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .phase_grid import PhaseGridConfig, ScatterStencil

_T = torch.Tensor


@dataclass(frozen=True)
class BoundaryFluxPolicies:
    """Recorded policy choices -- must match the BEM configuration the
    grid metric is compared against (BEMSetupSnapshot conventions)."""
    operator_measure: str = "carrier"        # carrier m | visible q*m
    use_coherence_velocity: bool = False     # q in the endpoint speed?
    n_endpoints: int = 3


class BoundaryFluxToGrid:
    """(s, delta_theta) -> delta_rho, explicit linear map."""

    def __init__(self, positions: _T, normals: _T,
                 carrier_masses: _T, coherence: _T,
                 grid_config: "PhaseGridConfig | None",
                 policies: Optional[BoundaryFluxPolicies] = None):
        """grid_config None (BIE backend): the endpoint conventions
        are built exactly as for the grid, but no scatter stencil
        exists -- only the grid-free members (total_flux, bulk_rows,
        first_moment_rows, and the particle-label forms of
        component_rows / particle_component_fractions) are usable."""
        pol = policies or BoundaryFluxPolicies()
        self.policies = pol
        x = positions.detach()
        n = normals.detach()
        m = carrier_masses.detach()
        q = coherence.detach()
        N = x.shape[0]
        K = pol.n_endpoints

        if pol.operator_measure == "carrier":
            m_op = m
        elif pol.operator_measure == "visible":
            m_op = m * q
        else:
            raise ValueError(pol.operator_measure)

        tangents = torch.stack([-n[:, 1], n[:, 0]], dim=1)
        if K == 1:
            r_vals = torch.zeros(1, dtype=x.dtype, device=x.device)
        else:
            r_vals = torch.linspace(-0.5, 0.5, K, dtype=x.dtype,
                                    device=x.device)
        # physical endpoint offsets and positions (BEM formulas)
        self.r_physical = r_vals[None, :] * m_op[:, None]      # (N, K)
        p = x[:, None, :] + self.r_physical[:, :, None] \
            * tangents[:, None, :]                             # (N,K,2)
        self.zeta = (m_op / K)[:, None].expand(N, K)           # (N, K)
        self.endpoint_positions = p                            # (N,K,2)
        self._m_op = m_op                                      # (N,)
        self.flux_scale = (q if pol.use_coherence_velocity
                           else torch.ones_like(q))
        self.n_particles, self.n_endpoints = N, K
        self.stencil = (ScatterStencil(p.reshape(N * K, 2), grid_config)
                        if grid_config is not None else None)

    def __call__(self, displacements: _T, delta_angles: _T) -> _T:
        """delta_rho on the grid; linear in (s, dtheta)."""
        v = self.flux_scale[:, None] \
            * (displacements[:, None]
               - self.r_physical * delta_angles[:, None])      # (N, K)
        vals = (self.zeta * v).reshape(-1)                     # (N*K,)
        if self.stencil is None:
            raise RuntimeError(
                "BoundaryFluxToGrid built without a grid (BIE backend): "
                "the scatter map delta_rho is undefined")
        return self.stencil.apply(vals)

    def total_flux(self, displacements: _T, delta_angles: _T) -> _T:
        """integral of delta_rho = total volume change rate (the
        kernel integrates to one), BEFORE any grid truncation."""
        v = self.flux_scale[:, None] \
            * (displacements[:, None]
               - self.r_physical * delta_angles[:, None])
        return (self.zeta * v).sum()

    def _particle_label_rows(self, labels: _T) -> _T:
        """Grid-free form of component_rows: `labels` is a PARTICLE
        label vector (N,), and the row of component c is the exact
        endpoint flux sum_{i in c, k} zeta_ik (s_i - r_ik dtheta_i)
        (the same functional as bulk_rows, evaluated on the metric
        component labels of the BIE backend). Dense (C, 2N)."""
        N = self.n_particles
        if labels.numel() != N:
            raise ValueError(
                f"particle label vector has {labels.numel()} entries, "
                f"expected {N}")
        C = int(labels.max()) + 1 if labels.numel() else 0
        zsum = (self.zeta * self.flux_scale[:, None]).sum(dim=1)
        zr = (self.zeta * self.flux_scale[:, None]
              * self.r_physical).sum(dim=1)
        rows = []
        for c in range(C):
            ind = (labels == c).to(self.zeta.dtype)
            rows.append(torch.cat([zsum * ind, -(zr * ind)]))
        return torch.stack(rows) if rows else \
            torch.zeros(0, 2 * N, dtype=self.zeta.dtype,
                        device=self.zeta.device)

    def component_rows(self, labels: _T,
                       cell_area: "float | None" = None) -> _T:
        """Constraint rows C_alpha[(s,dtheta)] = integral of B(s,dth)
        over component alpha -- assembled by applying the TRANSPOSE of
        the frozen scatter to component indicators. Returns a dense
        (C, 2N) matrix over the concatenated (s, dtheta) variables.

        cell_area None (no grid): `labels` are particle labels and the
        rows are the exact endpoint flux sums (_particle_label_rows)."""
        if cell_area is None or self.stencil is None:
            if cell_area is not None:
                raise RuntimeError(
                    "component_rows with a cell area needs the grid "
                    "stencil; this flux object was built without a grid")
            return self._particle_label_rows(labels)
        N, K = self.n_particles, self.n_endpoints
        C = int(labels.max()) + 1 if labels.numel() else 0
        rows = []
        flat_labels = labels.reshape(-1)
        for c in range(C):
            ind = (flat_labels[self.stencil.flat_index] == c)
            w_c = (self.stencil.kernel_weights * ind).sum(dim=1) \
                * cell_area                                    # (N*K,)
            w_c = w_c.reshape(N, K)
            base = self.zeta * w_c * self.flux_scale[:, None]
            row_s = base.sum(dim=1)
            row_th = -(base * self.r_physical).sum(dim=1)
            rows.append(torch.cat([row_s, row_th]))
        return torch.stack(rows) if rows else \
            torch.zeros(0, 2 * N, dtype=self.zeta.dtype,
                        device=self.zeta.device)

    def particle_component_fractions(self, labels: _T,
                                     cell_area: "float | None" = None
                                     ) -> _T:
        """(N, C): fraction of each particle's endpoint kernel mass
        landing in each grid component (kernel weights are normalized
        so each endpoint deposits unit mass; the K endpoints are
        averaged). Row sums <= 1 -- the deficit is inactive-cell mass.
        Used only to compare the grid partition against the winding
        bulk partition (0F); never enters any constraint row.

        cell_area None (no grid): `labels` are particle labels and the
        result is their one-hot encoding (every particle is attributed
        entirely to its own metric component)."""
        N, K = self.n_particles, self.n_endpoints
        C = int(labels.max()) + 1 if labels.numel() else 0
        if cell_area is None or self.stencil is None:
            if cell_area is not None:
                raise RuntimeError(
                    "particle_component_fractions with a cell area "
                    "needs the grid stencil; this flux object was built "
                    "without a grid")
            if labels.numel() != N:
                raise ValueError(
                    f"particle label vector has {labels.numel()} "
                    f"entries, expected {N}")
            out = torch.zeros(N, C, dtype=self.zeta.dtype,
                              device=self.zeta.device)
            if C:
                out[torch.arange(N, device=out.device), labels] = 1.0
            return out
        flat_labels = labels.reshape(-1)
        cols = []
        for c in range(C):
            ind = (flat_labels[self.stencil.flat_index] == c)
            w_c = (self.stencil.kernel_weights * ind).sum(dim=1) \
                * cell_area
            cols.append(w_c.reshape(N, K).mean(dim=1))
        return torch.stack(cols, dim=1) if cols else \
            torch.zeros(N, 0, dtype=self.zeta.dtype,
                        device=self.zeta.device)

    def bulk_rows(self, bulk_label_per_particle: _T,
                  n_bulk: int) -> _T:
        """0F physical componentwise volume rows (dense (B, 2N) over
        the concatenated (s, dtheta) variables): the exact first-order
        volume change of bulk component b,

            C_b(s, dth) = sum_{i in b, k} zeta_ik (s_i - r_ik dth_i),

        with a PARTICLE-level indicator (no grid active-cell fraction
        w_act -- that is exactly what makes these rows different from,
        and generally independent of, the grid component_rows near the
        boundary). Reuses self.zeta / self.r_physical so the segment
        convention can never diverge from the scatter's.

        Policy asserts (fail-closed, reviewer-mandated): the physical
        rows are only defined for the carrier measure with unit flux
        scale -- visible-measure or q-velocity configurations must not
        be silently converted."""
        pol = self.policies
        if pol.operator_measure != "carrier":
            raise RuntimeError(
                "bulk_rows requires operator_measure='carrier' (raw "
                f"oriented masses); got {pol.operator_measure!r} -- "
                "refusing to build q-weighted physical volume rows")
        if pol.use_coherence_velocity:
            raise RuntimeError(
                "bulk_rows requires use_coherence_velocity=False; the "
                "physical volume flux of a still-existing boundary "
                "must not be attenuated by q")
        if not torch.allclose(self.zeta.sum(dim=1), self._m_op,
                              rtol=1e-12, atol=0.0):
            raise RuntimeError(
                "bulk_rows: endpoint quadrature no longer sums to the "
                "carrier mass -- segment convention drifted")
        zsum = self.zeta.sum(dim=1)                       # = m_i
        zr = (self.zeta * self.r_physical).sum(dim=1)
        rows = []
        for b in range(n_bulk):
            ind = (bulk_label_per_particle == b).to(self.zeta.dtype)
            rows.append(torch.cat([zsum * ind, -(zr * ind)]))
        return torch.stack(rows) if rows else \
            torch.zeros(0, 2 * self.n_particles,
                        dtype=self.zeta.dtype, device=self.zeta.device)

    def first_moment_rows(self, bulk_label_per_particle: _T,
                          n_bulk: int) -> _T:
        """ORDER-FREE first-moment rows in the SAME flux family as
        bulk_rows (user 2026-08-24: area and moments must not consume
        a cyclic order -- the oriented point cloud suffices): the
        exact first-order change of M_b = int_{E_b} x dx under the
        endpoint velocity field, via the transport identity
        d/dt int_E x dx = oint x (V . n):

            dM_{b,k}[s, dth] = sum_{i in b, j} zeta_ij x^{ep}_{ij,k}
                               (s_i - r_ij dth_i),

        with the SAME zeta / r_physical / endpoint positions as the
        volume rows (one functional family -- no polygon, no order).
        Shape (2B, 2N), row order (b0_x, b0_y, b1_x, ...); the dtheta
        block is NONZERO here (the endpoint sweep of a rotating
        segment transports moment), unlike the polygon-differential
        rows whose functional is positions-only. Same policy asserts
        as bulk_rows."""
        pol = self.policies
        if pol.operator_measure != "carrier":
            raise RuntimeError(
                "first_moment_rows requires operator_measure="
                f"'carrier'; got {pol.operator_measure!r}")
        if pol.use_coherence_velocity:
            raise RuntimeError(
                "first_moment_rows requires use_coherence_velocity="
                "False (physical moment flux, not q-attenuated)")
        rows = []
        for b in range(n_bulk):
            ind = (bulk_label_per_particle == b).to(self.zeta.dtype)
            for k in (0, 1):
                w = self.zeta * self.endpoint_positions[:, :, k]
                row_s = w.sum(dim=1) * ind
                row_th = -(w * self.r_physical).sum(dim=1) * ind
                rows.append(torch.cat([row_s, row_th]))
        return torch.stack(rows) if rows else \
            torch.zeros(0, 2 * self.n_particles,
                        dtype=self.zeta.dtype, device=self.zeta.device)


def quotient_centered_moment_rows(M_rows: _T, CA_rows: _T,
                                  scalars: list, Q_quot: "_T | None",
                                  a_floor: float = 1e-12) -> _T:
    """Reviewer 2026-08-24 eq (1): quotient-class centered moment
    rows. Aggregate FIRST (rows (Q (x) I2), volume rows Q, scalars
    Q), then center with the MERGED barycenter xbar_c = M_c / A_c:

        Cbar_{c,k} = sum_b Q_cb C_{M_b,k}
                     - (M_{c,k} / A_c) sum_b Q_cb C_{A_b}.

    Centering per raw bulk BEFORE aggregation (eq (2)) is WRONG in
    State III: it retains parts of the per-component centered
    constraints; the difference is
    sum_b Q_cb (M_c/A_c - M_b/A_b) C_{A_b} (eq (3)), nonzero whenever
    the raw barycenters differ. Q_quot None = identity classes.
    scalars = [(A_b, M_bx, M_by)] per raw bulk. Fail-closed on a
    degenerate class volume."""
    n_bulk = CA_rows.shape[0]
    if Q_quot is None:
        idx = [[b] for b in range(n_bulk)]
    else:
        idx = [[b for b in range(n_bulk)
                if float(Q_quot[c, b]) > 0]
               for c in range(int(Q_quot.shape[0]))]
    out = []
    for members in idx:
        Mx_row = sum(M_rows[2 * b] for b in members)
        My_row = sum(M_rows[2 * b + 1] for b in members)
        CA_row = sum(CA_rows[b] for b in members)
        A_q = sum(scalars[b][0] for b in members)
        Mx_q = sum(scalars[b][1] for b in members)
        My_q = sum(scalars[b][2] for b in members)
        if abs(A_q) < a_floor:
            raise RuntimeError(
                f"current_centered: quotient-class volume {A_q:.3e} "
                "degenerate -- fail-closed")
        out.append(Mx_row - (Mx_q / A_q) * CA_row)
        out.append(My_row - (My_q / A_q) * CA_row)
    return torch.stack(out)
