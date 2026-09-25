"""0L-B1: contact / quotient certificate for the contact layer.

State II (CONTACT_LAYER_PREQUOTIENT) keeps the relative bulk row C_-;
State III (QUOTIENT_CONTACT) may drop it ONLY when this certificate
holds for the contact pair. Two separate questions, two separate
scales (reviewer 0L-B1):

  geometry  -- do the two raw boundary loops represent the same
               support to within the POINT-CLOUD resolution h_ab?
               (directional mass-weighted NN gaps normalized by
               h_ab = max(median m_a, median m_b), coverage at
               radius c_h h_ab, per-direction anti-alignment, raw
               loopwise mass balance)
  current   -- did the two mollified oriented currents actually
               cancel at the PERIMETER mollifier scale sigma?
               eps_cur2 = ||psi*(T_a+T_b)||^2 / (||psi*T_a||^2 +
               ||psi*T_b||^2) = (E_a+E_b+2E_ab)/(E_a+E_b): 0 for
               antiparallel coincident, 1 for uncorrelated far
               apart, 2 for co-oriented coincident. Audited at
               sigma_c in {0.8, 1, 1.25} sigma -- no new free scale.

Candidates are restricted to State-II bookkeeping: same grid
compatibility component, different raw bulk components, candidate
graph component of exactly two loops with degree 1 each (a near tie
is `ambiguous`, never label-tie-broken). B1 only OBSERVES: no row is
dropped here.

Thresholds are pre-registered from positive/negative fixture families
(scripts/experiments/contact_certificate_calibration.py), never from
the endgame trajectories.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import torch

_T = torch.Tensor


class ContactCertificateError(RuntimeError):
    """Fail-closed certificate evaluation error (bad inputs, PSD
    violation beyond rounding, missing bookkeeping)."""


# ---------------------------------------------------------------------
# mollified oriented current: Gaussian self-convolution Gram
# ---------------------------------------------------------------------

def mollified_current_gram(positions: _T, normals: _T, masses: _T,
                           sigma: float, idx_a: _T, idx_b: _T) -> float:
    """<psi_sigma * T_a, psi_sigma * T_b>_{L2} for the Gaussian
    mollifier: sum_{i in a, j in b} m_i m_j (n_i . n_j) K(x_i - x_j),
    K(z) = exp(-|z|^2 / 4 sigma^2) / (4 pi sigma^2) (self-convolution
    of the unit-mass Gaussian of width sigma). float64."""
    xa, xb = positions[idx_a].double(), positions[idx_b].double()
    na, nb = normals[idx_a].double(), normals[idx_b].double()
    ma, mb = masses[idx_a].double(), masses[idx_b].double()
    d2 = ((xa[:, None, :] - xb[None, :, :]) ** 2).sum(-1)
    ker = torch.exp(-d2 / (4.0 * sigma * sigma)) \
        / (4.0 * math.pi * sigma * sigma)
    return float(((ma[:, None] * mb[None, :]) * (na @ nb.T) * ker).sum())


def mollified_current_energy(positions: _T, normals: _T, masses: _T,
                             sigma: float, idx: _T) -> float:
    """||psi_sigma * T||_{L2}^2 (PSD; clamped at zero for rounding)."""
    e = mollified_current_gram(positions, normals, masses, sigma, idx, idx)
    return max(e, 0.0)


def pair_current_cancellation(positions: _T, normals: _T, masses: _T,
                              sigma: float, idx_a: _T, idx_b: _T,
                              energy_min: float = 0.0,
                              psd_rtol: float = 1e-12) -> dict:
    """(E_a, E_b, E_ab, eps_cur2). eps_cur2 = (E_a+E_b+2E_ab)/(E_a+E_b).
    Denominator guard: E_a + E_b < energy_min -> not evaluable
    (fail-closed, eps None). PSD guard: a negative combined energy
    beyond rounding (relative to E_a+E_b) raises."""
    E_a = mollified_current_gram(positions, normals, masses, sigma,
                                 idx_a, idx_a)
    E_b = mollified_current_gram(positions, normals, masses, sigma,
                                 idx_b, idx_b)
    E_ab = mollified_current_gram(positions, normals, masses, sigma,
                                  idx_a, idx_b)
    # full 2x2 Gram PSD guard (reviewer B2-0): E_a, E_b >= -eps (E_a+E_b),
    # E_a E_b - E_ab^2 >= -eps (E_a+E_b)^2 -- beyond rounding -> raise,
    # within rounding -> clamp
    scale = abs(E_a) + abs(E_b)
    if E_a < -psd_rtol * scale or E_b < -psd_rtol * scale:
        raise ContactCertificateError(
            f"negative self-energy beyond rounding (E_a {E_a:.3e}, "
            f"E_b {E_b:.3e})")
    if E_a * E_b - E_ab * E_ab < -psd_rtol * scale * scale:
        raise ContactCertificateError(
            f"2x2 current Gram not PSD beyond rounding: E_a {E_a:.3e} "
            f"E_b {E_b:.3e} E_ab {E_ab:.3e}")
    E_a, E_b = max(E_a, 0.0), max(E_b, 0.0)
    den = E_a + E_b
    out = dict(E_a=E_a, E_b=E_b, E_ab=E_ab, energy=den, eps_cur2=None,
               evaluable=den >= energy_min and den > 0.0)
    if not out["evaluable"]:
        return out
    num = den + 2.0 * E_ab
    if num < 0.0:
        if -num > psd_rtol * den:
            raise ContactCertificateError(
                f"combined current energy {num:.3e} < 0 beyond rounding "
                f"(E_a+E_b = {den:.3e}) -- Gram not PSD")
        num = 0.0
    out["eps_cur2"] = num / den
    return out


# ---------------------------------------------------------------------
# geometry at the point-cloud resolution
# ---------------------------------------------------------------------

def _wquantile(values: _T, weights: _T, q: float) -> float:
    order = values.argsort()
    v, w = values[order], weights[order]
    c = w.cumsum(0) / w.sum()
    k = int((c >= q).nonzero()[0]) if bool((c >= q).any()) else -1
    return float(v[k])


def directional_gap_metrics(positions: _T, masses: _T, idx_a: _T,
                            idx_b: _T, h_ab: float,
                            coverage_radius_over_h: float) -> dict:
    """Mass-weighted directional NN gaps a->b and b->a: g50, g90,
    normalized g90/h_ab, coverage at radius c_h h_ab (mass fraction of
    each loop within that distance of the other), and the NN maps."""
    xa, xb = positions[idx_a], positions[idx_b]
    ma, mb = masses[idx_a], masses[idx_b]
    D = torch.cdist(xa, xb)
    d_ab, pi_ab = D.min(dim=1)     # for each i in a: nearest j in b
    d_ba, pi_ba = D.min(dim=0)     # for each j in b: nearest i in a
    g_star = coverage_radius_over_h * h_ab
    out = dict(
        h_ab=h_ab,
        g50_ab=_wquantile(d_ab, ma, 0.5), g90_ab=_wquantile(d_ab, ma, 0.9),
        g50_ba=_wquantile(d_ba, mb, 0.5), g90_ba=_wquantile(d_ba, mb, 0.9),
        cov_ab=float(ma[d_ab <= g_star].sum() / ma.sum()),
        cov_ba=float(mb[d_ba <= g_star].sum() / mb.sum()),
        coverage_radius=g_star,
    )
    out["g_sym"] = 0.5 * (out["g50_ab"] + out["g50_ba"])
    out["gap90_over_h"] = max(out["g90_ab"], out["g90_ba"]) / h_ab
    out["coverage"] = min(out["cov_ab"], out["cov_ba"])
    out["_pi_ab"] = pi_ab
    out["_pi_ba"] = pi_ba
    return out


def directional_anti_alignment(normals: _T, masses: _T, idx_a: _T,
                               idx_b: _T, pi_ab: _T, pi_ba: _T) -> dict:
    """Per-direction mass-normalized anti-alignment
    eps^{a->b} = (1/M_a) sum_i m_i |n_i + n_{pi(i)}|^2 (and b->a);
    eps_anti = max of the two (symmetric under unequal counts)."""
    na, nb = normals[idx_a], normals[idx_b]
    ma, mb = masses[idx_a], masses[idx_b]
    e_ab = float((ma * ((na + nb[pi_ab]) ** 2).sum(1)).sum() / ma.sum())
    e_ba = float((mb * ((nb + na[pi_ba]) ** 2).sum(1)).sum() / mb.sum())
    return dict(anti_ab=e_ab, anti_ba=e_ba, anti=max(e_ab, e_ba))


# ---------------------------------------------------------------------
# thresholds / candidates / certificate
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ContactThresholds:
    """Pre-registered from the B1 fixture families (see
    results/reports/phase3c0l_b1_calibration.json and the report);
    NEVER fitted on endgame trajectories. Upper-bound metrics use the
    geometric mean of the positive maximum and the negative minimum of
    the intended rejection channel; the lower-bound coverage uses the
    midpoint. Values below are the calibration outcome (2026-08-16,
    scripts/experiments/contact_certificate_calibration.py):
      gap90/h   p_max 1.102 (residual offset h) / n_min 1.432
                (eccentric near)                       -> 1.256
      coverage  p_min 1.0 / n_max 0.255 (local contact) -> 0.628
                at radius 1.5 h_ab
      anti      p_max 4.96e-4 / n_min 4.0 (co-oriented) -> 0.0446
      mass      p_max 0.0337 / n_min 0.125 (unequal radius,
                ratio-3 gives 0.5)                    -> 0.0649
      current   p_max 0.0236 / n_min 0.0950 (0.5 sigma) -> 0.0474
    0.25-sigma separation (~h at this sampling) is a TRANSITION
    validation case, not a threshold negative."""
    gap90_over_h_max: float = 1.256
    coverage_radius_over_h: float = 1.5
    coverage_min: float = 0.628
    anti_max: float = 0.0446
    mass_max: float = 0.0649
    current_residual2_max: float = 0.0474
    current_energy_min: float = 1e-12
    search_radius_over_h: float = 6.0     # candidate search (loops)
    sigma_sweep: tuple = (0.8, 1.0, 1.25)


@dataclass
class ContactCandidate:
    loop_pair: tuple
    bulk_pair: tuple
    grid_component: Optional[int]
    ambiguous: bool
    reason: str
    g_sym: float


@dataclass
class ContactCertificate:
    candidate: ContactCandidate
    metrics: dict
    current: dict            # per sigma_c: pair_current_cancellation
    margins: dict            # signed margin per condition (>0 = pass)
    failing: list
    certified: bool
    thresholds: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = dict(candidate=asdict(self.candidate),
                 metrics={k: v for k, v in self.metrics.items()
                          if not k.startswith("_")},
                 current=self.current, margins=self.margins,
                 failing=list(self.failing), certified=self.certified,
                 thresholds=self.thresholds)
        return d


def _loop_indices(loop_labels: _T):
    return {int(lb): (loop_labels == lb).nonzero().flatten()
            for lb in loop_labels.unique()}


def find_contact_candidates(positions: _T, masses: _T, loop_labels: _T,
                            loop_to_grid_component: Optional[dict],
                            loop_to_raw_bulk: Optional[dict],
                            thresholds: ContactThresholds,
                            allow_unrestricted: bool = False
                            ) -> list:
    """State-II candidate pairs. Production requires the bookkeeping
    maps (same grid component AND different raw bulks); the
    unrestricted search (all loop pairs) is for fixtures only and must
    be requested explicitly. Candidate graph: loops a, b adjacent iff
    the SYMMETRIC median NN gap g_sym(a, b) <= search_radius_over_h *
    h_ab. A pair is a candidate iff it forms a connected component of
    exactly two loops with degree 1 each. STRICT rule (reviewer B2-0):
    any candidate-graph component with three or more loops makes every
    loop in it `ambiguous` -- no near-tie tolerance, no label
    tie-break. Production requires BOTH bookkeeping maps to contain
    every loop explicitly (a missing key is fail-closed)."""
    if (loop_to_grid_component is None or loop_to_raw_bulk is None) \
            and not allow_unrestricted:
        raise ContactCertificateError(
            "production candidate search needs loop_to_grid_component "
            "and loop_to_raw_bulk (State-II bookkeeping); pass "
            "allow_unrestricted=True only for fixtures")
    idx = _loop_indices(loop_labels)
    loops = sorted(idx)
    if loop_to_grid_component is not None:
        missing = [lb for lb in loops
                   if lb not in loop_to_grid_component
                   or lb not in loop_to_raw_bulk]
        if missing:
            raise ContactCertificateError(
                f"bookkeeping maps miss loops {missing} (every certified "
                "loop must be mapped explicitly)")
    h = {lb: float(masses[idx[lb]].median()) for lb in loops}
    # pairwise symmetric median gaps
    gsym = {}
    for i, a in enumerate(loops):
        for b in loops[i + 1:]:
            if loop_to_grid_component is not None:
                if loop_to_grid_component.get(a) \
                        != loop_to_grid_component.get(b):
                    continue
                if loop_to_raw_bulk.get(a) == loop_to_raw_bulk.get(b):
                    continue
            hab = max(h[a], h[b])
            m = directional_gap_metrics(positions, masses, idx[a], idx[b],
                                        hab, thresholds.coverage_radius_over_h)
            if m["g_sym"] <= thresholds.search_radius_over_h * hab:
                gsym[(a, b)] = (m["g_sym"], hab)
    # candidate graph adjacency
    adj = {lb: [] for lb in loops}
    for (a, b), (g, hab) in gsym.items():
        adj[a].append((b, g, hab))
        adj[b].append((a, g, hab))
    out = []
    seen = set()
    for a in loops:
        if not adj[a] or a in seen:
            continue
        # component of a
        comp, stack = set(), [a]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(v for v, _, _ in adj[u])
        seen |= comp
        if len(comp) != 2:
            # every loop in this component is ambiguous (multi-partner)
            for u in comp:
                partners = sorted(adj[u], key=lambda t: t[1])
                if len(partners) >= 2:
                    b1, g1, hab = partners[0]
                    b2, g2, _ = partners[1]
                    reason = (f"loop {u}: {len(partners)} candidates "
                              f"(g_sym {g1:.3e}, {g2:.3e}; "
                              f"component size {len(comp)})")
                    out.append(ContactCandidate(
                        loop_pair=(u, b1), bulk_pair=(
                            loop_to_raw_bulk.get(u) if loop_to_raw_bulk
                            else None,
                            loop_to_raw_bulk.get(b1) if loop_to_raw_bulk
                            else None),
                        grid_component=(loop_to_grid_component.get(u)
                                        if loop_to_grid_component
                                        else None),
                        ambiguous=True, reason=reason, g_sym=g1))
            continue
        a_, b_ = sorted(comp)
        g, hab = gsym[(a_, b_)]
        out.append(ContactCandidate(
            loop_pair=(a_, b_),
            bulk_pair=(loop_to_raw_bulk.get(a_) if loop_to_raw_bulk
                       else None,
                       loop_to_raw_bulk.get(b_) if loop_to_raw_bulk
                       else None),
            grid_component=(loop_to_grid_component.get(a_)
                            if loop_to_grid_component else None),
            ambiguous=False, reason="unique mutual pair", g_sym=g))
    return out


def evaluate_contact_certificate(positions: _T, normals: _T, masses: _T,
                                 loop_labels: _T,
                                 candidate: ContactCandidate,
                                 sigma_perimeter: float,
                                 thresholds: ContactThresholds
                                 = ContactThresholds()) -> ContactCertificate:
    """Full certificate (eq. 7 of the reviewer's B1 spec) for one
    candidate: uniqueness AND gap90/h AND coverage AND anti AND mass
    AND max over the sigma sweep of eps_cur2. Records signed margins
    (> 0 passes) for every condition and the failing list."""
    idx = _loop_indices(loop_labels)
    a, b = candidate.loop_pair
    ia, ib = idx[a], idx[b]
    h_ab = max(float(masses[ia].median()), float(masses[ib].median()))
    geo = directional_gap_metrics(positions, masses, ia, ib, h_ab,
                                  thresholds.coverage_radius_over_h)
    anti = directional_anti_alignment(normals, masses, ia, ib,
                                      geo["_pi_ab"], geo["_pi_ba"])
    M_a, M_b = float(masses[ia].sum()), float(masses[ib].sum())
    mass = abs(M_a - M_b) / (M_a + M_b)
    current = {}
    eps_list = []
    for rho in thresholds.sigma_sweep:
        c = pair_current_cancellation(
            positions, normals, masses, rho * sigma_perimeter, ia, ib,
            energy_min=thresholds.current_energy_min)
        current[f"{rho:g}"] = c
        eps_list.append(c["eps_cur2"])
    evaluable = all(e is not None for e in eps_list)
    eps_max = max(eps_list) if evaluable else None
    metrics = dict(geo)
    metrics.update(anti)
    metrics.update(mass_residual=mass, M_a=M_a, M_b=M_b,
                   eps_cur2_max=eps_max, current_evaluable=evaluable)
    margins = dict(
        uniqueness=(0.0 if candidate.ambiguous else 1.0),
        gap90_over_h=thresholds.gap90_over_h_max - geo["gap90_over_h"],
        coverage=geo["coverage"] - thresholds.coverage_min,
        anti=thresholds.anti_max - anti["anti"],
        mass=thresholds.mass_max - mass,
        current=((thresholds.current_residual2_max - eps_max)
                 if evaluable else -float("inf")),
    )
    failing = [k for k, v in margins.items()
               if not (v > 0.0 if k != "uniqueness" else v >= 1.0)]
    return ContactCertificate(
        candidate=candidate, metrics=metrics, current=current,
        margins=margins, failing=failing, certified=(len(failing) == 0),
        thresholds=asdict(thresholds))


def evaluate_all_candidates(positions: _T, normals: _T, masses: _T,
                            loop_labels: _T,
                            loop_to_grid_component: Optional[dict],
                            loop_to_raw_bulk: Optional[dict],
                            sigma_perimeter: float,
                            thresholds: ContactThresholds
                            = ContactThresholds(),
                            allow_unrestricted: bool = False) -> list:
    cands = find_contact_candidates(
        positions, masses, loop_labels, loop_to_grid_component,
        loop_to_raw_bulk, thresholds, allow_unrestricted=allow_unrestricted)
    return [evaluate_contact_certificate(
        positions, normals, masses, loop_labels, c, sigma_perimeter,
        thresholds) for c in cands]


def evaluate_contact_reference(positions: _T, normals: _T, masses: _T,
                               mask_a: _T, mask_b: _T,
                               h_star: float,
                               coverage_radius_at_switch: float,
                               sigma_at_switch: float,
                               thresholds_at_switch: dict,
                               bulk_pair: tuple,
                               loop_pair: tuple) -> ContactCertificate:
    """0N-4 State III persistence invariant (reviewer 2026-08-18): the
    quotient switch certified the equivalence a ~_* b at the RESOLUTION
    of the switch state, so the identification scale h_* = h_ab(X_*),
    the mollifier sigma_* and the thresholds are quotient metadata --
    NOT re-derived from the (shrinking, fixed-N) ghost representation.
    This is distinct from the B1 ENTRY certificate
    (evaluate_contact_certificate), which correctly uses the live
    point-cloud resolution to recognize a contact in State II; scope:
    one certified annihilating pair in a fixed-N exact-merger class.

    Hard invariant (switch scale, thresholds all unchanged from B1):
        g90 / h_*                       <= gap90_over_h_max
        coverage at c_h h_*             >= coverage_min
        eps_anti / eps_mass / eps_cur2  <= their B1 thresholds
    Live-scale geometry (g90 / h_ab(t), coverage at c_h h_ab(t)) is
    recorded as SHADOW TELEMETRY only. Metric key semantics are fixed:
    h_ab / gap90_over_h / coverage_radius / coverage refer to the
    switch scale (the hard invariant, what gates read); *_live keys
    carry the moving-scale values."""
    thr = thresholds_at_switch
    ia = mask_a.nonzero().flatten()
    ib = mask_b.nonzero().flatten()
    c_h = float(thr["coverage_radius_over_h"])
    if abs(c_h * h_star - coverage_radius_at_switch) \
            > 1e-12 * max(1.0, abs(coverage_radius_at_switch)):
        raise ContactCertificateError(
            f"contact_reference inconsistent: coverage_radius_at_switch "
            f"{coverage_radius_at_switch!r} != coverage_radius_over_h * "
            f"h_star = {c_h * h_star!r}")
    geo_sw = directional_gap_metrics(positions, masses, ia, ib,
                                     h_star, c_h)
    h_live = max(float(masses[ia].median()), float(masses[ib].median()))
    geo_live = directional_gap_metrics(positions, masses, ia, ib,
                                       h_live, c_h)
    anti = directional_anti_alignment(normals, masses, ia, ib,
                                      geo_sw["_pi_ab"], geo_sw["_pi_ba"])
    M_a, M_b = float(masses[ia].sum()), float(masses[ib].sum())
    mass = abs(M_a - M_b) / (M_a + M_b)
    current = {}
    eps_list = []
    for rho in thr["sigma_sweep"]:
        c = pair_current_cancellation(
            positions, normals, masses, float(rho) * sigma_at_switch,
            ia, ib, energy_min=float(thr["current_energy_min"]))
        current[f"{float(rho):g}"] = c
        eps_list.append(c["eps_cur2"])
    evaluable = all(e is not None for e in eps_list)
    eps_max = max(eps_list) if evaluable else None
    metrics = dict(geo_sw)                    # switch-scale = hard keys
    metrics.update(
        h_star=h_star,
        gap90_over_h_switch=geo_sw["gap90_over_h"],
        coverage_switch_scale=geo_sw["coverage"],
        h_ab_live=h_live,
        gap90_over_h_live=geo_live["gap90_over_h"],
        coverage_radius_live=geo_live["coverage_radius"],
        coverage_live_scale=geo_live["coverage"],
    )
    metrics.update(anti)
    metrics.update(mass_residual=mass, M_a=M_a, M_b=M_b,
                   eps_cur2_max=eps_max, current_evaluable=evaluable)
    margins = dict(
        uniqueness=1.0,          # pair fixed at the switch, no re-search
        gap90_over_h=float(thr["gap90_over_h_max"])
        - geo_sw["gap90_over_h"],
        coverage=geo_sw["coverage"] - float(thr["coverage_min"]),
        anti=float(thr["anti_max"]) - anti["anti"],
        mass=float(thr["mass_max"]) - mass,
        current=((float(thr["current_residual2_max"]) - eps_max)
                 if evaluable else -float("inf")),
    )
    failing = [k for k, v in margins.items()
               if not (v > 0.0 if k != "uniqueness" else v >= 1.0)]
    cand = ContactCandidate(
        loop_pair=tuple(loop_pair), bulk_pair=tuple(bulk_pair),
        grid_component=None, ambiguous=False,
        reason="contact_reference (State III, h_star frozen)",
        g_sym=geo_sw["g_sym"])
    return ContactCertificate(
        candidate=cand, metrics=metrics, current=current,
        margins=margins, failing=failing, certified=(len(failing) == 0),
        thresholds=dict(thr))


def transport_loop_maps(labels_src: _T, labels_dst: _T,
                        maps: dict) -> dict:
    """Carry loop-keyed bookkeeping maps from the source labeling to a
    re-certified target labeling (reviewer B2 item 7). The overlap
    matrix O_ab = #{i: l_src(i)=a, l_dst(i)=b} must be a bijection
    (each source loop meets exactly one target loop and vice versa) and
    the two partitions must agree as equivalence relations; anything
    else is a topology ambiguity -> ContactCertificateError."""
    if labels_src.shape != labels_dst.shape:
        raise ContactCertificateError("label arrays differ in length")
    src = labels_src.unique()
    dst = labels_dst.unique()
    if src.numel() != dst.numel():
        raise ContactCertificateError(
            f"loop count changed {src.numel()} -> {dst.numel()}")
    corr = {}
    for a in src:
        targets = labels_dst[labels_src == a].unique()
        if targets.numel() != 1:
            raise ContactCertificateError(
                f"source loop {int(a)} maps to {targets.numel()} target "
                "loops")
        b = int(targets[0])
        back = labels_src[labels_dst == b].unique()
        if back.numel() != 1 or int(back[0]) != int(a):
            raise ContactCertificateError(
                f"target loop {b} is not the unique image of source loop "
                f"{int(a)}")
        corr[int(a)] = b
    out = {}
    for name, mp in maps.items():
        out[name] = {corr[a]: v for a, v in mp.items() if a in corr}
    out["_correspondence"] = corr
    return out


# 0L-B2 amendment (reviewer 2026-08-17): the shadow-switch monotonicity
# comparison of certificate margins between the source, keep and drop
# states uses NORMALIZED margins and a numerical comparison tolerance
# eta_margin -- NOT a change of any B1 threshold. The B1 thresholds
# remain the hard certificate; this tolerance only defines the
# comparison precision of non-smooth NN/quantile telemetry across the
# three states (measured: an anti-margin change of 1.98e-9 = 4.4e-8 of
# its threshold tripped the original absolute 1e-9 tolerance).
MARGIN_COMPARE_TOL = 1e-6


def normalized_margins(cert_dict: dict) -> dict:
    """m_hat_k: upper-bound gates (tau - f)/tau; coverage (lower
    bound) (f - tau)/(1 - tau). Uniqueness is excluded (hard gate)."""
    th = cert_dict.get("thresholds") or asdict(ContactThresholds())
    mg = cert_dict.get("margins") or {}
    out = {}
    scale = {"gap90_over_h": th["gap90_over_h_max"],
             "anti": th["anti_max"], "mass": th["mass_max"],
             "current": th["current_residual2_max"]}
    for k, tau in scale.items():
        if k in mg and tau > 0:
            out[k] = mg[k] / tau
    if "coverage" in mg:
        out["coverage"] = mg["coverage"] / max(1.0 - th["coverage_min"],
                                               1e-30)
    return out


def margins_not_worse(cert_src: dict, cert_keep: dict, cert_drop: dict,
                      tol: float = MARGIN_COMPARE_TOL) -> tuple:
    """Eq. (2.1): m_hat_k(drop) >= min(m_hat_k(src), m_hat_k(keep)) - tol
    for every normalized margin. Returns (ok, table)."""
    ms, mk, md = (normalized_margins(cert_src), normalized_margins(cert_keep),
                  normalized_margins(cert_drop))
    table = {}
    ok = True
    for k in ms:
        ref = min(ms[k], mk.get(k, ms[k]))
        val = md.get(k, -float("inf"))
        passed = val >= ref - tol
        table[k] = dict(src=ms[k], keep=mk.get(k), drop=val, ref=ref,
                        deficit=ref - val, passed=passed)
        ok = ok and passed
    return ok, table
