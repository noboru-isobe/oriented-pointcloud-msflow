"""0F: winding-signature bulk incidence (Phase IIIa machinery -> src).

Physical bulk-component detection from the oriented point cloud
ITSELF, independent of grid connectivity. Motivation (IIIc-0Dx): after
the diffuse grid support bridges (t_bridge < t_contact), the grid
compatibility rows collapse to one component and the componentwise
volume constraint degenerates -- the measured disk->annulus volume
pump. The winding signature still reads the true bulk components from
the boundary marks (verified 11 steps before breakdown), so it is used
to AUGMENT (never replace) the grid rows.

Pipeline:
  oriented_graph_components  boundary-loop labels (proximity AND
                             co-orientation -- antiparallel sheets stay
                             separate at distance zero)
  winding_bulk_labels        per-loop bulk id from the componentwise
                             winding signature at material-side probes;
                             fail-closed (None) on any instability
  partition_relation         co-membership comparison of two particle
                             partitions (invariant to label renaming)

Certificate (reviewer-mandated, two layers): the pre-round median
winding must be within EPS_WIND of an integer at EVERY (loop, probe
depth), and the rounded signature must be identical across all probe
depths. Either failure returns None -- the caller decides whether that
is a stop event (production) or a skip (diagnostics).

All functions are pure, operate on detached CPU-or-CUDA tensors, and
are O(N^2) dense (N ~ 10^2-10^3 here; ~ms).
"""

from __future__ import annotations

import math

import torch

_T = torch.Tensor

# Integer-margin certificate threshold, pre-registered 2026-08-13 from
# measured worst margins: IIIa fixtures annulus 0.0290 / two disks
# 0.0301 / disk-two-holes 0.0284 / two annuli 0.0301, and exact-merger
# checkpoint 1575 (3 loops, deep contact layer) 0.0540. EPS_WIND = 0.15
# is ~3x the worst measurement and 30% of the round-ambiguity 0.5.
# Do not recalibrate from outcomes; tests pin the fixture margins.
EPS_WIND = 0.15


class IncidenceUnstableError(RuntimeError):
    """Winding incidence failed its certificate (probe-depth
    disagreement or integer margin) -- fail-closed stop signal."""


class PartitionAmbiguousError(RuntimeError):
    """Grid/bulk particle partitions are incomparable or a particle's
    grid-component attribution is ambiguous -- fail-closed."""


def oriented_graph_components(positions: _T, normals: _T,
                              ell: float) -> _T:
    """Boundary-loop labels: connected components of the graph with an
    edge iff |x_i - x_j| < ell AND n_i . n_j > 0. The co-orientation
    condition keeps coincident antiparallel sheets in separate loops
    regardless of distance. Returns LongTensor (N,) with labels in
    first-visit (particle-index) order."""
    x = positions.detach()
    n = normals.detach()
    N = x.shape[0]
    adj = (torch.cdist(x, x) < ell) & ((n @ n.T) > 0.0)
    labels = torch.full((N,), -1, dtype=torch.long, device=x.device)
    current = 0
    for seed in range(N):
        if int(labels[seed]) >= 0:
            continue
        comp = torch.zeros(N, dtype=torch.bool, device=x.device)
        comp[seed] = True
        frontier = comp.clone()
        while bool(frontier.any()):
            frontier = adj[frontier].any(dim=0) & ~comp
            comp |= frontier
        labels[comp] = current
        current += 1
    return labels


def winding_bulk_labels(positions: _T, normals: _T, masses: _T,
                        loop_labels: _T, spacing: float,
                        delta_fracs: tuple = (1.5, 2.5, 4.0),
                        n_probe: int = 8,
                        eps_wind: float = EPS_WIND):
    """Per-loop bulk ids from the componentwise winding signature.

    For each probe depth `frac`, place n_probe material-side probes
    p = x - frac*spacing*n on each loop b and read the regularized
    winding of every loop a:

        w_a(p) = (1/2pi) sum_{i in a} m_i n_i.(x_i - p)
                                      / (|x_i - p|^2 + (spacing/2)^2)

    The signature of loop b is (round(median_p w_a))_a; equal
    signatures = same bulk component. Returns (bulk_of_loop, margin):
    LongTensor (n_loop,) with ids in first-appearance loop order, and
    the worst |median - round(median)| observed. Returns (None, margin)
    if any margin exceeds eps_wind or the rounded signature differs
    across probe depths (fail-closed)."""
    x = positions.detach()
    n = normals.detach()
    m = masses.detach()
    n_loop = int(loop_labels.max()) + 1 if loop_labels.numel() else 0
    eta2 = (0.5 * spacing) ** 2
    signatures = []
    worst_margin = 0.0
    for frac in delta_fracs:
        sig = []
        for b in range(n_loop):
            idx = torch.nonzero(loop_labels == b).flatten()
            sel = idx[torch.linspace(0, idx.numel() - 1,
                                     min(n_probe, idx.numel()),
                                     device=x.device).long()]
            probes = x[sel] - frac * spacing * n[sel]      # (P, 2)
            row = []
            for a in range(n_loop):
                ia = loop_labels == a
                diff = x[ia][None, :, :] - probes[:, None, :]
                den = (diff ** 2).sum(2) + eta2
                w = ((m[ia][None, :]
                      * (n[ia][None, :, :] * diff).sum(2) / den)
                     .sum(1) / (2.0 * math.pi))
                med = float(w.median())
                margin = abs(med - round(med))
                worst_margin = max(worst_margin, margin)
                if margin > eps_wind:
                    return None, worst_margin
                row.append(int(round(med)))
            sig.append(tuple(row))
        signatures.append(tuple(sig))
    if any(s != signatures[0] for s in signatures[1:]):
        return None, worst_margin
    uniq: dict = {}
    bulk = [uniq.setdefault(s, len(uniq)) for s in signatures[0]]
    return torch.tensor(bulk, dtype=torch.long,
                        device=x.device), worst_margin


def partition_relation(labels_a: _T, labels_b: _T) -> str:
    """Compare two particle partitions AS EQUIVALENCE RELATIONS
    (co-membership; invariant to label renaming). Returns
    "equal" / "a_finer" (a strictly refines b) / "b_finer" /
    "incomparable"."""
    if labels_a.shape != labels_b.shape:
        raise ValueError("partition label arrays must be same length")

    def refines(fine: _T, coarse: _T) -> bool:
        # every fine-class lies inside a single coarse-class
        for v in fine.unique():
            if coarse[fine == v].unique().numel() > 1:
                return False
        return True

    a_ref = refines(labels_a, labels_b)
    b_ref = refines(labels_b, labels_a)
    if a_ref and b_ref:
        return "equal"
    if a_ref:
        return "a_finer"
    if b_ref:
        return "b_finer"
    return "incomparable"
