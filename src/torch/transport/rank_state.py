"""Hysteretic rank tracking for the spectral componentwise solver (0C-5d).

The state machine consumes structured RankEvidence per step and decides the
WORKING rank. Design constraints fixed at review time:

- weak_gap evidence (criteria agree, confidence low) HOLDS the current rank
  -- chatter prevention through the thin-neck interval;
- a switch requires k_persist GENUINELY CONSECUTIVE steps of clean
  evidence for the new rank (any interrupting weak/ambiguous frame resets
  the counter) AND physical-mode validation of the candidate basis:
    (a) the global phase-constant direction W^{1/2} 1 must lie in the
        candidate span (it is a null direction of every valid closed
        phase boundary, at any C), and
    (b) the candidate span must be approximately contained in the previous
        working subspace (a merge can only lose null directions, not
        rotate into unrelated ones) -- NOTE this check is only active when
        the endpoint count is unchanged; sequences whose N varies (e.g.
        the clean-union static family) are validated by (a) + persistence
        only, and the row-space continuation of the particle-level
        constraint C_u is the dynamic replacement (0C-5d-1);
  a numerically plausible rank count carried by the WRONG null vector
  (raw-crossing collocation artifacts) fails these checks and is rejected;
- hysteresis prevents rank chatter. It CANNOT turn an invalid raw support
  into a valid merged boundary -- rejection is a stop signal for the
  caller, not a fallback the machine recovers from silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .bem_wasserstein import RankEvidence

_T = torch.Tensor


@dataclass(frozen=True)
class RankStateConfig:
    # KNOWN LIMITATION: k_persist counts steps, so under Delta-t refinement
    # the persistence window T = k_persist * dt shrinks -- this cannot
    # define a convergent switch location (same issue as the redistribution
    # time normalisation, 0A/F4). Post-contact work must switch on physical
    # time or cumulative boundary displacement instead. Fine for the static
    # audit and the pre-contact hold, which never complete a switch.
    k_persist: int = 3           # consecutive clean foreign steps to switch
    theta_const_deg: float = 15.0    # W^{1/2}1 must sit in candidate span
    theta_contain_deg: float = 30.0  # candidate span vs previous subspace


@dataclass(frozen=True)
class RankDecision:
    rank: int                    # working rank AFTER this update
    action: str                  # hold_clean / hold_weak / hold_ambiguous /
                                 # candidate_pending / switch /
                                 # reject_artificial
    evidence_status: str
    pending: int = 0             # persistence counter after this update
    # TELEMETRY SEMANTICS (split 2026-08-06 after the pair@512 theta=30
    # observation): align_const_deg is evaluated against whatever span
    # the ACTIVE BRANCH inspected -- the raw last-r SVD columns in the
    # same-rank branch, the CANDIDATE span in the foreign-proposal
    # branch. When a soft mode reorders the near-null spectrum, that
    # value can jump to ~90 deg while the machine's HELD basis
    # continues smoothly (measured: contain ~ 9-11 deg through the
    # jump). align_held_deg is ALWAYS the phase constant's angle in
    # the machine's tracked basis -- the quantity a threshold may
    # legitimately act on. Never conflate the two again.
    align_const_deg: Optional[float] = None
    contain_deg: Optional[float] = None
    align_held_deg: Optional[float] = None


def angle_vec_in_span_deg(b: _T, U: _T) -> float:
    """Angle between vector b and its projection onto span(U) (orthonormal
    columns): 0 means b lies in the span."""
    cos = (U.T @ b).norm() / b.norm().clamp_min(1e-300)
    return float(torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0))))


def max_angle_span_in_span_deg(A: _T, B: _T) -> float:
    """Largest principal angle of span(A) measured against span(B) (both
    orthonormal columns): 0 means span(A) is contained in span(B)."""
    cos = torch.linalg.svdvals(B.T @ A).min()
    return float(torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0))))


@dataclass(frozen=True)
class AdaptiveRankConfig:
    """0C-5d-2a: replacement for the fixed absolute-null threshold.

    Precision about what was wrong with the fixed tau (review-corrected):
    tau = 0.05 acts on RELATIVE singular values, so it is already
    dimensionless and invariant under geometric rescaling. Its defect is
    h-REFINEMENT ANTI-CONVERGENCE: the thin-neck quasi-null mode has a
    positive continuum limit, so any fixed tau eventually counts it as
    null and the transition detection worsens as h -> 0. (A tau ~ h
    formula would fix that but LOSES rescaling invariance -- the two
    defects are different.) The floor rule below is both scale-free and
    refinement-consistent: eta tracks the ACCEPTED clean frames' null
    block (EMA, frozen during non-clean intervals), and a max-gap
    candidate rank r is floor-consistent when

        max_{j<=r} s_(j)/s_max <= kappa_null * eta,
        s_(r+1)/s_max          >= kappa_sep  * eta.

    Persistence is measured in CUMULATIVE BOUNDARY DISPLACEMENT relative
    to l_res, not steps -- a step-count rule shrinks its physical window
    under Delta-t refinement (same failure as the redistribution
    normalisation, F4). Callers must pass the ACTUAL geometric
    displacement of the boundary per frame (for prescribed static
    families: the geometric distance between consecutive frames, e.g.
    |Delta g|), NOT a fixed per-frame convention -- otherwise the switch
    location inherits the sampling grid.
    """
    kappa_null: float = 3.0
    kappa_sep: float = 10.0
    ema_alpha: float = 0.3
    theta_const_deg: float = 15.0
    theta_contain_deg: float = 30.0
    persist_displacement: float = 2.0      # units of l_res, for a switch
    ambiguity_budget: float = 20.0         # units of l_res, then hard stop


@dataclass
class AdaptiveRankMachine:
    """Hysteretic rank tracking with the noise-floor rule (0C-5d-2a).

    Scope: VALID prescribed boundaries only -- raw post-contact supports
    still hard-stop through the physical-mode validation, and nothing here
    repairs a support. update() consumes the raw ascending relative
    spectrum plus the max-gap proposal; there is no absolute threshold
    anywhere.
    """
    config: AdaptiveRankConfig = field(default_factory=AdaptiveRankConfig)
    rank: Optional[int] = None
    floor: Optional[float] = None          # eta, EMA over accepted frames
    _pending_rank: Optional[int] = None
    _pending_disp: float = 0.0
    _ambiguity_disp: float = 0.0
    _prev_basis: Optional[_T] = None

    def update(self, s_desc: _T, U: _T, sqrt_w: _T,
               step_displacement: float, l_res: float,
               rank_max: int = 4) -> RankDecision:
        """One frame. U is the FULL left singular basis (descending order
        as returned by torch.linalg.svd); candidates are its last columns.
        """
        cfg = self.config
        s_asc = s_desc.flip(0)
        rel = (s_asc / s_desc[0].clamp_min(1e-300)).tolist()
        kmax = min(rank_max, len(rel) - 1)
        ratios = [rel[k] / max(rel[k - 1], 1e-300)
                  for k in range(1, kmax + 1)]
        r = max(range(1, kmax + 1), key=lambda k: ratios[k - 1])
        gap_ratio = ratios[r - 1]

        b_const = sqrt_w / sqrt_w.norm().clamp_min(1e-300)

        # align of the phase constant in the HELD (tracked) basis --
        # the only alignment a threshold may act on (telemetry split,
        # see RankDecision docstring)
        held = None
        if self._prev_basis is not None and \
                self._prev_basis.shape[0] == U.shape[0]:
            held = angle_vec_in_span_deg(b_const, self._prev_basis)

        def _dec(*args, **kw):
            kw.setdefault("align_held_deg", held)
            return RankDecision(*args, **kw)

        if self.rank is None:
            # initialisation: the first frame must be self-consistently
            # clean (its own null block IS the initial floor estimate) AND
            # physically valid -- the phase constant must lie in the span
            if gap_ratio < cfg.kappa_sep:
                return _dec(0, "uninitialised", "weak_gap",
                                    align_const_deg=None)
            align0 = angle_vec_in_span_deg(b_const, U[:, -r:])
            if align0 >= cfg.theta_const_deg:
                return _dec(0, "uninitialised", "clean",
                                    align_const_deg=align0)
            self.rank = r
            self.floor = max(rel[:r])
            self._prev_basis = U[:, -r:].detach()
            return _dec(self.rank, "hold_clean", "clean",
                                align_const_deg=align0)

        null_ok = max(rel[:r]) <= cfg.kappa_null * self.floor
        sep_ok = rel[r] >= cfg.kappa_sep * self.floor

        if r == self.rank:
            self._pending_rank, self._pending_disp = None, 0.0
            # same-rank evidence, weak OR clean, must still carry the
            # phase constant: an invalid support can keep the COUNT while
            # rotating the nullspace into artificial modes (review 4.4).
            # KNOWN GAP (review 5.2): at C >= 2 this passes even if the
            # non-constant null directions rotated -- a artificial 2D
            # subspace containing the constant slips through. The dynamic
            # row-C_u continuation check (solver integration stage) is the
            # stronger condition; do not integrate into production
            # without it.
            align = angle_vec_in_span_deg(b_const, U[:, -r:])
            if align >= cfg.theta_const_deg:
                return _dec(self.rank, "reject_artificial",
                                    "clean" if (null_ok and sep_ok)
                                    else "weak_gap",
                                    align_const_deg=align)
            if null_ok and sep_ok:
                self.floor = ((1 - cfg.ema_alpha) * self.floor
                              + cfg.ema_alpha * max(rel[:r]))
                self._ambiguity_disp = 0.0
                self._prev_basis = U[:, -r:].detach()
                return _dec(self.rank, "hold_clean", "clean",
                                    align_const_deg=align)
            # same-rank weak evidence: continue (floor frozen) within the
            # ambiguity budget, then hard-stop signal
            self._ambiguity_disp += step_displacement
            if self._ambiguity_disp > cfg.ambiguity_budget * l_res:
                return _dec(self.rank, "ambiguity_budget_exceeded",
                                    "weak_gap", align_const_deg=align)
            return _dec(self.rank, "hold_weak", "weak_gap",
                                align_const_deg=align)

        # foreign max-gap proposal
        U_cand = U[:, -r:].detach()
        align = angle_vec_in_span_deg(b_const, U_cand)
        contain = None
        if self._prev_basis is not None and \
                self._prev_basis.shape[0] == U_cand.shape[0]:
            if r < self.rank:
                contain = max_angle_span_in_span_deg(
                    U_cand, self._prev_basis)
            else:
                contain = max_angle_span_in_span_deg(
                    self._prev_basis, U_cand)
        physical_ok = align < cfg.theta_const_deg and (
            contain is None or contain < cfg.theta_contain_deg)
        if not (null_ok and sep_ok):
            self._pending_rank, self._pending_disp = None, 0.0
            # COMMON weak-evidence budget (review, P1.2 safety): a
            # foreign-rank weak interval consumes the same cumulative-
            # displacement budget as same-rank weak evidence -- without
            # this, a persistent foreign weak signal could hold the
            # rank through unbounded geometric change.
            self._ambiguity_disp += step_displacement
            if self._ambiguity_disp > cfg.ambiguity_budget * l_res:
                return _dec(self.rank,
                                    "ambiguity_budget_exceeded",
                                    "weak_gap", align_const_deg=align,
                                    contain_deg=contain)
            return _dec(self.rank, "hold_weak", "weak_gap",
                                align_const_deg=align, contain_deg=contain)
        if not physical_ok:
            self._pending_rank, self._pending_disp = None, 0.0
            return _dec(self.rank, "reject_artificial", "clean",
                                align_const_deg=align, contain_deg=contain)
        if self._pending_rank == r:
            self._pending_disp += step_displacement
        else:
            self._pending_rank, self._pending_disp = r, step_displacement
        if self._pending_disp >= cfg.persist_displacement * l_res:
            self.rank = r
            self.floor = max(rel[:r])
            self._prev_basis = U_cand
            self._pending_rank, self._pending_disp = None, 0.0
            self._ambiguity_disp = 0.0
            return _dec(self.rank, "switch", "clean",
                                align_const_deg=align, contain_deg=contain)
        return _dec(self.rank, "candidate_pending", "clean",
                            pending=int(self._pending_disp / l_res),
                            align_const_deg=align, contain_deg=contain)


@dataclass
class RankStateMachine:
    rank: int
    config: RankStateConfig = field(default_factory=RankStateConfig)
    _pending_rank: Optional[int] = None
    _pending_count: int = 0
    _prev_basis: Optional[_T] = None   # last accepted basis at working rank

    def update(self, ev: RankEvidence, U_candidate: _T,
               sqrt_w: _T) -> RankDecision:
        """One step of the state machine.

        Args:
            ev: structured rank evidence of the current operator.
            U_candidate: (NK, r) orthonormal left near-null basis at the
                evidence's preferred count r = ev.c_gap (callers pass the
                last r left singular vectors regardless of evidence status;
                it is only VALIDATED when a switch is on the table).
            sqrt_w: W^{1/2} diagonal (NK,) of the current operator.
        """
        cfg = self.config

        if ev.status in ("detector_disagreement", "no_null_block"):
            # ambiguity never advances a switch
            self._pending_rank, self._pending_count = None, 0
            return RankDecision(self.rank, "hold_ambiguous", ev.status)

        r = ev.rank  # agreed count (clean and weak_gap both define it)

        if r == self.rank:
            self._pending_rank, self._pending_count = None, 0
            if ev.status == "clean":
                # refresh the continuation reference only on clean evidence
                self._prev_basis = U_candidate.detach()
                return RankDecision(self.rank, "hold_clean", ev.status)
            return RankDecision(self.rank, "hold_weak", ev.status)

        # agreed evidence for a DIFFERENT rank
        if ev.status == "weak_gap":
            # weak foreign evidence RESETS pending: the k_persist rule means
            # genuinely consecutive clean frames (a clean run interrupted by
            # weak evidence starts over -- the safer reading of persistence)
            self._pending_rank, self._pending_count = None, 0
            return RankDecision(self.rank, "hold_weak", ev.status)

        # clean foreign evidence: validate the candidate modes physically
        b_const = sqrt_w / sqrt_w.norm().clamp_min(1e-300)
        align = angle_vec_in_span_deg(b_const, U_candidate)
        contain = None
        if self._prev_basis is not None and \
                self._prev_basis.shape[0] == U_candidate.shape[0]:
            if r < self.rank:
                contain = max_angle_span_in_span_deg(
                    U_candidate, self._prev_basis)
            else:
                contain = max_angle_span_in_span_deg(
                    self._prev_basis, U_candidate)
        ok = align < cfg.theta_const_deg and (
            contain is None or contain < cfg.theta_contain_deg)
        if not ok:
            self._pending_rank, self._pending_count = None, 0
            return RankDecision(self.rank, "reject_artificial", ev.status,
                                align_const_deg=align, contain_deg=contain)

        if self._pending_rank == r:
            self._pending_count += 1
        else:
            self._pending_rank, self._pending_count = r, 1
        if self._pending_count >= cfg.k_persist:
            self.rank = r
            self._prev_basis = U_candidate.detach()
            self._pending_rank, self._pending_count = None, 0
            return RankDecision(self.rank, "switch", ev.status,
                                align_const_deg=align, contain_deg=contain)
        return RankDecision(self.rank, "candidate_pending", ev.status,
                            pending=self._pending_count,
                            align_const_deg=align, contain_deg=contain)
