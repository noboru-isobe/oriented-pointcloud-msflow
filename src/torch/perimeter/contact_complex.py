"""Phase L0J (reviewer 2026-08-21): the sigma-scale ENERGY contact
complex for the contact-complex well-balanced coherence

    q_i^CC = r_{alpha, l(i)} q_i^self   on a contact side,
    q_i^CC = q_i^self                   outside every complex,

    r_{alpha, l} = sum_{i in I_{alpha,l}} m_i q_i^full
                 / sum_{i in I_{alpha,l}} m_i q_i^self.

SCOPE. This is the ENERGY interaction complex, built from the SAME
compact-support kernel the coherence itself reads (support |z| = sigma)
-- it is NOT the h-scale topology quotient complex of the local
contact certificate, and the two must never be identified. Applicable
regime: annihilating (anti-aligned) local contact between two
certified loops; co-oriented proximity and same-loop self-contact are
fail-closed out of scope.

Exact properties (pinned in tests/test_contact_complex.py):
  - sidewise source-energy conservation:
        sum_{I} m q^CC = sum_{I} m q^full   on every contact side;
  - outside every complex the compact support gives
        q^CC = q^self = q^full  EXACTLY (far-field identity);
  - a complex covering the whole loop reduces r_{alpha,l} to the
    loopwise r_l of the first campaign's q^WB (exact inclusion);
  - no new length scale is introduced.

Membership is a HARD binary set: r jumps by a finite amount when a
particle crosses the support edge even though the kernel value itself
vanishes smoothly there (support-edge birth/death). The hard version
is therefore an ALGEBRAIC BASELINE; production adoption is gated on
the L0J continuity/refresh audits, with the tapered core/collar
partition as the pre-registered fallback."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .coherence_perimeter import mollifier_2d

_T = torch.Tensor


class SigmaContactComplexError(RuntimeError):
    """Fail-closed certificate error of the energy contact complex."""


# L-A probe classification (reviewer 2026-08-22 hardening 1): benign
# immaturity and structural ambiguity are DIFFERENT conditions -- the
# activation probe continues through the former and is run-fatal on
# the latter. All subclass SigmaContactComplexError, so every existing
# fail-closed catch keeps working.

class ContactComplexNotMature(SigmaContactComplexError):
    """Benign: a side has too few particles (satellite pair / thin
    fringe) -- keep waiting under WB."""


class ContactComplexAmbiguous(SigmaContactComplexError):
    """Structural: a component meets three or more loops, or several
    mature complexes coexist -- out of the Phase L scope, run-fatal."""


class ContactComplexGeometryError(SigmaContactComplexError):
    """Structural: cyclic-order certificate failure, same-loop
    self-contact, non-interval side, crossing pairing -- run-fatal."""


class ContactComplexOutOfScope(SigmaContactComplexError):
    """Structural: co-oriented proximity (r > 1) or degenerate
    denominators -- outside the annihilating-contact scope."""


# ---------------------------------------------------------------------
# certified cyclic order (source-frozen)
# ---------------------------------------------------------------------

def certify_loop_orders(positions: _T, loop_labels: _T,
                        spacing_factor: float = 3.0) -> dict:
    """Derive AND certify a cyclic order for every loop from the
    SOURCE GEOMETRY alone (reviewer 2026-08-21 blocker A): the order
    is the centroid-angle sort of the loop's point set -- disposable,
    re-derived on every source state, never persisted -- so the whole
    pipeline is equivariant under ARBITRARY within-loop permutations,
    label renaming, rigid rotations, and cyclic reversal (an angle
    sort is unique up to rotation + reversal of the cycle, and every
    downstream consumer -- interval connectivity, correspondence
    monotonicity, arc coordinates -- is invariant under both).

    Certificates on the DERIVED order (fail-closed): star-shapedness
    about the centroid (strictly increasing angles, no ties), every
    consecutive segment at most `spacing_factor` x the loop median,
    and each particle's nearest same-loop neighbour is one of its two
    cyclic neighbours.

    Returns {loop: dict(index=global particle ids in cyclic order,
    rank_of=global lookup (N,) with the cyclic rank or -1, arclen=
    cumulative arc coordinate in that order, length=float)}."""
    n_tot = positions.shape[0]
    out = {}
    for lb in loop_labels.unique():
        members = (loop_labels == lb).nonzero().flatten()
        p = positions[members]
        n = p.shape[0]
        if n < 3:
            raise ContactComplexGeometryError(
                f"loop {int(lb)} has {n} < 3 particles")
        c = p.mean(dim=0)
        ang = torch.atan2(p[:, 1] - c[1], p[:, 0] - c[0])
        order = ang.argsort()
        if bool((ang[order][1:] == ang[order][:-1]).any()):
            raise ContactComplexGeometryError(
                f"loop {int(lb)}: tied centroid angles -- not "
                "star-shaped at this resolution (fail-closed)")
        idx = members[order]
        p_o = positions[idx]
        seg = (torch.roll(p_o, -1, dims=0) - p_o).norm(dim=1)
        med = float(seg.median())
        if med <= 0 or float(seg.max()) > spacing_factor * med:
            raise ContactComplexGeometryError(
                f"loop {int(lb)}: cyclic segment {float(seg.max()):.3e}"
                f" exceeds {spacing_factor} x median {med:.3e} in the "
                "derived centroid-angle order (fail-closed)")
        D = torch.cdist(p_o, p_o)
        D.fill_diagonal_(float("inf"))
        nn = D.argmin(dim=1)
        d = (nn - torch.arange(n)) % n
        ok = (d == 1) | (d == n - 1)
        if not bool(ok.all()):
            bad = int((~ok).nonzero().flatten()[0])
            raise ContactComplexGeometryError(
                f"loop {int(lb)}: particle {bad} (derived order) has "
                "nearest same-loop neighbour that is not a cyclic "
                "neighbour")
        rank_of = torch.full((n_tot,), -1, dtype=torch.long)
        rank_of[idx] = torch.arange(n)
        arclen = torch.cat([torch.zeros(1, dtype=seg.dtype),
                            seg.cumsum(0)[:-1]])
        out[int(lb)] = dict(index=idx, rank_of=rank_of, arclen=arclen,
                            length=float(seg.sum()))
    return out


# ---------------------------------------------------------------------
# the complex
# ---------------------------------------------------------------------

@dataclass
class ContactSide:
    loop: int
    particles: _T              # global particle indices (cyclic order)
    mask: _T                   # global boolean mask
    n: int


@dataclass
class SigmaContactComplex:
    sides: "list[ContactSide]"
    telemetry: dict = field(default_factory=dict)


def _cyclic_interval(ranks_sorted: _T, n: int):
    """Is the sorted rank set a single cyclic interval of Z_n? Returns
    (ok, start) with start = first rank of the interval after the
    cyclic cut."""
    k = ranks_sorted.numel()
    if k == n:
        return True, 0
    gaps = torch.roll(ranks_sorted, -1) - ranks_sorted
    gaps[-1] = ranks_sorted[0] + n - ranks_sorted[-1]
    big = (gaps > 1).nonzero().flatten()
    if big.numel() != 1:
        return False, None
    cut = int(big[0])
    start = int(ranks_sorted[(cut + 1) % k])
    return True, start


def _monotone(seq: "list[int]", allow_wrap: int = 0) -> bool:
    """Monotone in either direction, ties allowed. For a CUT-ALIGNED
    proper interval no cyclic wrap remains (allow_wrap = 0); when the
    destination side is the FULL cycle its seam is an arbitrary cut,
    so exactly one wrap is legitimate (allow_wrap = 1)."""
    if len(seq) <= 2:
        return True
    d = [b - a for a, b in zip(seq[:-1], seq[1:])]
    desc = sum(1 for x in d if x < 0)
    asc = sum(1 for x in d if x > 0)
    return desc <= allow_wrap or asc <= allow_wrap


def sigma_contact_complex(positions: _T, normals: _T, masses: _T,
                          loop_labels: _T, loop_orders: dict,
                          sigma: float,
                          kernel: str = "wendland_c2",
                          ) -> SigmaContactComplex:
    """Build the sigma-scale energy contact complex on a SOURCE state
    (frozen for the MM step). Fail-closed certificates:
      - same-loop self-contact (Euclidean < sigma but cyclic ARC
        distance > 2 sigma) -> out of scope,
      - each component meets exactly two loops,
      - each side is one cyclic interval with >= 2 particles,
      - the directed NN correspondences (both directions) are
        cyclically monotone (non-crossing) -- NOT a raw-kernel-graph
        degree condition."""
    n_tot = positions.shape[0]
    z = positions.unsqueeze(1) - positions.unsqueeze(0)
    K = mollifier_2d(z, sigma, kernel)
    same = loop_labels.unsqueeze(1) == loop_labels.unsqueeze(0)
    dist = z.norm(dim=-1)
    # ---- same-loop self-contact rejection -------------------------
    for lb, od in loop_orders.items():
        idx = od["index"]
        s = od["arclen"]
        L = od["length"]
        d_l = dist[idx][:, idx]
        ds = (s.unsqueeze(1) - s.unsqueeze(0)).abs()
        ds = torch.minimum(ds, L - ds)
        bad = (d_l < sigma) & (ds > 2.0 * sigma)
        if bool(bad.any()):
            i, j = [int(x) for x in bad.nonzero()[0]]
            raise ContactComplexGeometryError(
                f"same-loop self-contact on loop {lb}: particles at "
                f"arc distance {float(ds[i, j]):.4f} > 2 sigma are "
                f"within sigma Euclidean ({float(d_l[i, j]):.4f}) -- "
                "out of the Phase L scope (fail-closed)")
    # ---- cross-support graph and components -----------------------
    cross = (K > 0.0) & (~same)
    active = cross.any(dim=1)
    parent = list(range(n_tot))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = cross.nonzero(as_tuple=True)
    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    comp_of = {}
    for i in range(n_tot):
        if bool(active[i]):
            comp_of.setdefault(find(i), []).append(i)
    complexes = []
    complex_id = torch.full((n_tot,), -1, dtype=torch.long)
    for cid, (root, members) in enumerate(sorted(comp_of.items())):
        mem = torch.tensor(members, dtype=torch.long)
        labs = loop_labels[mem].unique()
        if labs.numel() != 2:
            raise ContactComplexAmbiguous(
                f"contact complex meets {labs.numel()} loops "
                f"({[int(x) for x in labs]}) -- exactly two required "
                "(third-loop ambiguity is fail-closed)")
        sides = []
        for lb in labs.tolist():
            od = loop_orders[int(lb)]
            sel = mem[loop_labels[mem] == lb]
            n_loop = od["index"].numel()
            sel_ranks = od["rank_of"][sel]
            if sel.numel() < 2:
                raise ContactComplexNotMature(
                    f"contact side on loop {int(lb)} has "
                    f"{sel.numel()} < 2 particles")
            ok, start = _cyclic_interval(sel_ranks.sort().values,
                                         n_loop)
            if not ok:
                raise ContactComplexGeometryError(
                    f"contact side on loop {int(lb)} is not a single "
                    "cyclic interval")
            # side particles in cyclic order starting at the cut
            shifted = (sel_ranks - start) % n_loop
            side_sorted = sel[shifted.argsort()]
            m_side = torch.zeros(n_tot, dtype=torch.bool)
            m_side[sel] = True
            sides.append(ContactSide(loop=int(lb),
                                     particles=side_sorted,
                                     mask=m_side, n=int(sel.numel())))
        # order-preserving NN correspondence (both directions); the
        # side particle arrays are already in cyclic order from the
        # cut, so non-crossing == plain monotonicity of positions
        (sa, sb) = sides
        for src, dst in ((sa, sb), (sb, sa)):
            D = dist[src.particles][:, dst.particles]
            pi = D.argmin(dim=1)
            dst_full = dst.n == loop_orders[dst.loop]["index"].numel()
            if not _monotone([int(x) for x in pi],
                             allow_wrap=1 if dst_full else 0):
                raise ContactComplexGeometryError(
                    f"NN correspondence loop {src.loop} -> "
                    f"{dst.loop} is not cyclically monotone "
                    "(crossing pairing, fail-closed)")
        for s_ in sides:
            complex_id[s_.mask] = cid
        Kc = K[mem][:, mem]
        cross_c = ~(loop_labels[mem].unsqueeze(1)
                    == loop_labels[mem].unsqueeze(0))
        complexes.append(SigmaContactComplex(
            sides=sides,
            telemetry=dict(
                n_members=int(mem.numel()),
                loops=[s_.loop for s_ in sides],
                side_sizes=[s_.n for s_ in sides],
                K_cross_max=float(Kc[cross_c].max())
                if bool(cross_c.any()) else 0.0)))
    return dict(complexes=complexes, complex_id=complex_id,
                n_complexes=len(complexes))


def contact_complex_q(positions: _T, normals: _T, masses: _T,
                      loop_labels: _T, sigma: float,
                      kernel: str, q_full: _T, q_self: _T) -> dict:
    """Assemble q^CC from a source state (pure function; the stepper
    delegates here). Returns dict(q_cc, r_cc, sides, cx). Guards:
    per-side denominator non-degeneracy (B <= 100 eps M fail-closed,
    never clipped) and the annihilating-contact scope certificate
    r in [0, 1+1e-9]; Delta = sum m (q_self - q_full) per side is the
    anti-contact telemetry (significantly negative => co-oriented
    proximity, out of scope)."""
    orders = certify_loop_orders(positions, loop_labels)
    cx = sigma_contact_complex(positions, normals, masses, loop_labels,
                               orders, sigma, kernel)
    q_cc = q_self.clone()
    r_cc = torch.ones_like(q_self)
    eps = torch.finfo(masses.dtype).eps
    sides = []
    for comp in cx["complexes"]:
        for side in comp.sides:
            sel = side.mask
            A = float((masses[sel] * q_full[sel]).sum())
            B = float((masses[sel] * q_self[sel]).sum())
            M = float(masses[sel].sum())
            if B <= 100.0 * eps * M:
                raise ContactComplexGeometryError(
                    f"contact-side self-coherence denominator "
                    f"degenerate on loop {side.loop} (B={B:.3e}) -- "
                    "fail-closed, no clipping")
            r = A / B
            if not (0.0 <= r <= 1.0 + 1e-9):
                raise ContactComplexOutOfScope(
                    f"contact-side r = {r:.6f} outside [0, 1+1e-9] on "
                    f"loop {side.loop}: outside the "
                    "annihilating-contact scope certificate")
            q_cc[sel] = r * q_self[sel]
            r_cc[sel] = r
            sides.append(dict(loop=side.loop, n=side.n, r=r,
                              Delta=B - A))
    # sigma is echoed so downstream certificates (L-A eligibility
    # eq (1)) read the SOURCE-FROZEN scale the complex was built with
    return dict(q_cc=q_cc, r_cc=r_cc, sides=sides, cx=cx,
                sigma=float(sigma))


def probe_contact_complex(positions: _T, normals: _T, masses: _T,
                          loop_labels: _T, sigma: float, kernel: str,
                          q_full: _T, q_self: _T,
                          maturity_min_side: int = 3) -> dict:
    """L-A activation probe (reviewer 2026-08-22 hardening 1/2):
    classify the sigma-complex state of a SOURCE state into
        inactive       -- no cross interaction at all,
        maturing       -- satellite-only / a side below the maturity
                          minimum (|I| < 3),
        mature_unique  -- exactly one complex, both sides |I| >= 3,
                          scope certificates hold (activation
                          candidate),
        ambiguous      -- >= 3 loops in one component (run-fatal),
        geometry_error -- order/self-contact/interval/crossing
                          failures (run-fatal),
        out_of_scope   -- co-oriented / degenerate (run-fatal).
    Pure observation: never raises for the benign classes; the fatal
    classes are returned as statuses for the CALLER to fail-close on
    (the probe itself must not kill a WB run that merely looked)."""
    try:
        res = contact_complex_q(positions, normals, masses,
                                loop_labels, sigma, kernel,
                                q_full, q_self)
    except ContactComplexNotMature as e:
        return dict(status="maturing", reason=str(e)[:160], res=None)
    except ContactComplexAmbiguous as e:
        return dict(status="ambiguous", reason=str(e)[:160], res=None)
    except ContactComplexOutOfScope as e:
        return dict(status="out_of_scope", reason=str(e)[:160],
                    res=None)
    except ContactComplexGeometryError as e:
        return dict(status="geometry_error", reason=str(e)[:160],
                    res=None)
    cx = res["cx"]
    if cx["n_complexes"] == 0:
        return dict(status="inactive", reason="no cross support",
                    res=res)
    if cx["n_complexes"] > 1:
        # several coexisting complexes: if more than one is mature
        # this is out of the single-pair Phase L scope
        mature = sum(1 for c in cx["complexes"]
                     if all(s.n >= maturity_min_side for s in c.sides))
        if mature > 1:
            return dict(status="ambiguous",
                        reason=f"{mature} mature complexes", res=res)
        return dict(status="maturing",
                    reason=f"{cx['n_complexes']} complexes, "
                           f"{mature} mature", res=res)
    sides = cx["complexes"][0].sides
    if any(s.n < maturity_min_side for s in sides):
        return dict(status="maturing",
                    reason=f"side sizes {[s.n for s in sides]} < "
                           f"{maturity_min_side}", res=res)
    return dict(status="mature_unique", reason="", res=res)


def translation_balanced_r(A: _T, b: _T, r0: _T, w: _T, hi: _T,
                           sv_floor: float = 1e-10,
                           resid_floor: float = 1e-9) -> dict:
    """L0J-M candidate A (reviewer 2026-08-21 secs 5-7): minimum-change
    side factors r solving the equality system A r = b (energy row +
    translation-response rows, the OUTSIDE contribution already on the
    RHS), in the mass-weighted metric w, with box 0 <= r <= hi.

    Rank-revealing: rows are SVD-screened; a discarded direction is
    admissible only when its target residual is within resid_floor
    (else `rank_deficient_residual`). Bounds semantics (reviewer sec
    6): an out-of-box UNCONSTRAINED projection does NOT prove
    infeasibility -- the box-constrained system is then solved (SLSQP,
    small scale) and `bounded_system_infeasible` is reported only when
    that also fails the equalities. Statuses: ok_unconstrained /
    ok_bounded / bounded_system_infeasible / rank_deficient_residual."""
    sw = w.sqrt()
    B = A / sw.unsqueeze(0)                 # variables u = sw (r - r0)
    c = b - A @ r0
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    keep = S > sv_floor * float(S.max())
    rank = int(keep.sum())
    coeff = U.T @ c
    dropped = float(coeff[~keep].norm()) if int((~keep).sum()) else 0.0
    out = dict(rank=rank, n_rows=int(A.shape[0]),
               cond=(float(S[keep].max() / S[keep].min())
                     if rank else float("inf")),
               dropped_target_residual=dropped)
    if dropped > resid_floor * max(float(b.norm()), 1.0):
        out["status"] = "rank_deficient_residual"
        out["r"] = None
        return out
    u = Vh[keep].T @ (coeff[keep] / S[keep])
    r = r0 + u / sw
    out["unconstrained_out_of_bounds"] = bool(
        (r < -1e-12).any() or (r > hi + 1e-12).any())
    if not out["unconstrained_out_of_bounds"]:
        out["status"] = "ok_unconstrained"
        out["r"] = r
        return out
    # box-constrained minimum-change QP (small scale) -- reviewer sec
    # 6: only THIS failing proves infeasibility
    import numpy as np
    from scipy.optimize import minimize
    An, bn = A.numpy(), b.numpy()
    r0n, wn, hin = r0.numpy(), w.numpy(), hi.numpy()
    res = minimize(
        lambda x: float(0.5 * ((x - r0n) ** 2 * wn).sum()),
        x0=np.clip(r.numpy(), 0.0, hin),
        jac=lambda x: (x - r0n) * wn,
        bounds=[(0.0, float(h)) for h in hin],
        constraints=[{"type": "eq",
                      "fun": lambda x, i=i: float(An[i] @ x - bn[i]),
                      "jac": lambda x, i=i: An[i]}
                     for i in range(An.shape[0])],
        method="SLSQP", options={"maxiter": 200, "ftol": 1e-14})
    eq_resid = float(abs(An @ res.x - bn).max())
    if res.success and eq_resid < resid_floor * max(float(b.norm()),
                                                    1.0):
        out["status"] = "ok_bounded"
        out["r"] = torch.as_tensor(res.x, dtype=A.dtype)
        out["eq_residual"] = eq_resid
    else:
        out["status"] = "bounded_system_infeasible"
        out["r"] = None
        out["eq_residual"] = eq_resid
    return out
