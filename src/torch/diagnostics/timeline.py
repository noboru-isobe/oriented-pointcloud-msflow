"""0E time-level records, 0D-audit ready (0E-wiring.1 revision).

FIVE levels per step, emitted on EVERY step including identity/no-event
records (a step on which deletion evaluated but removed nothing is
exactly the datum the 0D threshold audit needs):

    pre_mm                 the geometry the step starts from, with the
                           scales THIS level actually determines
    post_mm_candidate      the MM output; `used` quantities are the
                           frozen pre-MM ones the legacy scheme consumed
    post_redistribution    identity record when redistribution is off /
                           out of interval (stage_executed = False)
    post_deletion_raw      right after point removal (all-true keep mask
                           when nothing was removed)
    post_deletion_final    after the post-removal extra redistribution --
                           the geometry actually committed; hiding that
                           extra pass behind "post_deletion" would
                           falsify the 0D comparisons

Each record carries DETACHED geometry, both USED and CURRENT m/q/mq
(used = what the variant consumed at that level; current = recomputed on
this level's geometry with this level's scales), the actual
delta/tau/sigma/eps, loop tombstones (survivor counts, vanished and
under-resolved loops -- the annulus inner-loop disappearance must never
silently drop out of the record), and the SupportReport.

Diagnostics only: attaching a timeline leaves trajectories bitwise
unchanged (asserted over ALL steps and scalar histories in the tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from src.torch.oriented_varifold.mass import compute_masses
from .support import (
    STICKY_INVALID_STATES,
    SupportReport,
    WholeBoundaryLoopExtinctionCertificate,
    certify_loop_extinction,
    classify_support,
    permissions as _permissions,
)

_T = torch.Tensor

TIME_LEVELS = ("pre_mm", "post_mm_candidate", "post_redistribution",
               "post_deletion_raw", "post_deletion_final")


@dataclass
class TimeLevelRecord:
    step: int
    level: str
    report: Optional[SupportReport]
    # detached geometry of this level
    positions: Optional[_T]
    angles: Optional[_T]
    n_points: int
    # bookkeeping
    particle_ids: list
    keep_mask: Optional[list]
    stage_enabled: bool
    stage_executed: bool
    geometry_changed: bool
    n_removed: int
    committed: bool
    # USED quantities (what the scheme consumed at this level) vs
    # CURRENT (recomputed on this geometry with this level's scales)
    m_used: Optional[_T] = None
    q_used: Optional[_T] = None
    mq_used: Optional[_T] = None
    m_current: Optional[_T] = None
    # DIAGNOSTIC real coherence on the current geometry (with the
    # production perimeter kernel/backend); distinct from q_used, which
    # is what the solver consumed (== 1 under use_unit_coherence)
    q_current_diagnostic: Optional[_T] = None
    mq_current_diagnostic: Optional[_T] = None
    delta_used: Optional[float] = None
    tau_used: Optional[float] = None
    sigma_used: Optional[float] = None
    eps_bem_used: Optional[float] = None
    # loop tombstones
    loop_survivor_counts: Optional[list] = None
    vanished_loop_ids: Optional[list] = None
    underresolved_loop_ids: Optional[list] = None
    rank_fields: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class SupportTimeline:
    """Maintains persistent IDs + ordered loop membership and records the
    five time levels. `initial_loops` must partition the initial particle
    IDs exactly once (asserted)."""
    initial_loops: List[_T]
    records: List[TimeLevelRecord] = field(default_factory=list)
    # 0E-wiring.2 item 6: COMPUTING current fields and STORING pointwise
    # tensors are separate switches -- a memory-saving flag must never
    # change the SupportReport classification.
    compute_current_fields: bool = True
    store_pointwise_fields: bool = True
    # D1b.1 item 5: STICKY invalidity. Once any frame shows an invalid
    # support (or the classifier itself fails), geometric reclosure in a
    # LATER frame is not evidence of recovery -- the per-frame classifier
    # is memoryless, so an open -> crossing -> separated_closed sequence
    # is an algorithmic stress log, never a scientific trajectory.
    requires_reconstruction: bool = False
    # D1 closure review (policy B): EVERY boundary-loop-count change is a
    # pending topology event that closes the gates until the timeline
    # itself revalidates it from its OWN records
    # (certify_and_accept_loop_extinction). A caller-built certificate,
    # a foreign timeline's certificate or an old certificate can never
    # clear a later invalid state.
    pending_topology_events: list = field(default_factory=list)
    accepted_certificates: list = field(default_factory=list)
    _ids: Optional[_T] = None
    _vanished_seen: set = field(default_factory=set)
    _last_invalid_step: Optional[int] = None
    # step of the most recent ACTUAL deletion event (n_removed > 0) --
    # used to attach the causal state name "open_after_deletion" only to
    # frames of that step; an open geometry on any other step is
    # "open_support" (cause unattributed)
    _last_deletion_step: Optional[int] = None

    def __post_init__(self):
        allids = torch.cat([c for c in self.initial_loops])
        n0 = int(allids.numel())
        assert allids.unique().numel() == n0, \
            "initial_loops must not repeat particle IDs"
        assert torch.equal(allids.sort().values, torch.arange(n0)), \
            "initial_loops must cover 0..N-1 exactly once"
        self._ids = torch.arange(n0)

    def apply_keep_mask(self, keep_mask: _T):
        self._ids = self._ids[keep_mask]

    def _loop_status(self):
        alive = set(self._ids.tolist())
        counts, vanished, under = [], [], []
        for li, c in enumerate(self.initial_loops):
            n = sum(1 for p in c.tolist() if p in alive)
            counts.append(n)
            if n == 0:
                vanished.append(li)
            elif n < 3:
                under.append(li)
        return counts, vanished, under

    def current_loops(self) -> List[_T]:
        """Ordered current-storage indices per ACTIVE loop (>= 3
        survivors). Vanished/under-resolved loops are excluded from the
        classifier input but always reported via the tombstone fields."""
        id_to_pos = {int(pid): k for k, pid in enumerate(self._ids)}
        loops = []
        for c in self.initial_loops:
            idx = [id_to_pos[int(p)] for p in c.tolist()
                   if int(p) in id_to_pos]
            if len(idx) >= 3:
                loops.append(torch.tensor(idx, dtype=torch.long))
        return loops

    def loop_local_h(self, varifold) -> float:
        """Median consecutive spacing over the ordered active loops --
        NOT the full-cloud nearest-neighbour distance, which reads the
        inter-loop gap as a point spacing near contact."""
        segs = []
        for c in self.current_loops():
            P = varifold.positions[c]
            segs.append((P.roll(-1, 0) - P).norm(dim=1))
        if not segs:
            return float("nan")
        return torch.cat(segs).median().item()

    def record(self, step: int, level: str, varifold, *,
               delta: float, sigma: float,
               eps_bem: Optional[float] = None,
               mass_tau: Optional[float] = None,
               mass_kernel: str = "wendland_c2",
               coherence_kernel: str = "wendland_c2",
               coherence_backend: str = "naive",
               delta_for_kde=None, tau_for_kde=None,
               m_used: Optional[_T] = None,
               q_used: Optional[_T] = None,
               displacements: Optional[_T] = None,
               keep_mask: Optional[_T] = None,
               stage_enabled: bool = True,
               stage_executed: bool = True,
               geometry_changed: bool = True,
               n_removed: int = 0,
               committed: bool = True,
               rank_fields: Optional[dict] = None) -> TimeLevelRecord:
        assert level in TIME_LEVELS, level
        if level == "post_deletion_raw" and keep_mask is not None \
                and n_removed > 0:
            self.apply_keep_mask(keep_mask)
            self._last_deletion_step = step
        counts, vanished, under = self._loop_status()

        h_loc = self.loop_local_h(varifold)
        report, err = None, None
        m_cur = q_cur = None
        try:
            if self.compute_current_fields and mass_tau is not None:
                d_kde = delta_for_kde if delta_for_kde is not None else delta
                t_kde = tau_for_kde if tau_for_kde is not None else mass_tau
                m_cur = compute_masses(varifold.positions.detach(), d_kde,
                                       t_kde, mass_kernel)
                from src.torch.transport import compute_coherence
                # the SOLVER's coherence kernel/backend, not the mass
                # kernel (0E-wiring.2 item 1)
                q_cur = compute_coherence(varifold, m_cur, sigma,
                                          coherence_kernel,
                                          backend=coherence_backend)
            report = classify_support(
                varifold, self.current_loops(), delta=delta, sigma=sigma,
                h=h_loc, eps_bem=eps_bem,
                masses=(m_cur if m_cur is not None else m_used),
                coherence=(q_cur if q_cur is not None else q_used),
                displacements=displacements, particle_ids=self._ids,
                mass_tau=mass_tau, mass_kernel=mass_kernel,
                provenance=level,
                deletion_this_step=(self._last_deletion_step == step))
        except Exception as e:      # diagnostics must never kill the run
            err = f"{type(e).__name__}: {e}"

        snap = self.store_pointwise_fields
        rec = TimeLevelRecord(
            step=step, level=level, report=report,
            positions=(varifold.positions.detach().clone() if snap
                       else None),
            angles=(varifold.angles.detach().clone() if snap else None),
            n_points=varifold.n_points,
            particle_ids=self._ids.tolist(),
            keep_mask=(keep_mask.tolist() if keep_mask is not None
                       else None),
            stage_enabled=stage_enabled, stage_executed=stage_executed,
            geometry_changed=geometry_changed, n_removed=n_removed,
            committed=committed,
            m_used=(m_used.detach().clone() if snap and m_used is not None
                    else None),
            q_used=(q_used.detach().clone() if snap and q_used is not None
                    else None),
            mq_used=((m_used * q_used).detach().clone()
                     if snap and m_used is not None and q_used is not None
                     else None),
            m_current=(m_cur if snap else None),
            q_current_diagnostic=(q_cur if snap else None),
            mq_current_diagnostic=(m_cur * q_cur if snap
                                   and m_cur is not None
                                   and q_cur is not None else None),
            delta_used=delta, tau_used=mass_tau, sigma_used=sigma,
            eps_bem_used=eps_bem,
            loop_survivor_counts=counts, vanished_loop_ids=vanished,
            underresolved_loop_ids=under,
            rank_fields=rank_fields, error=err)
        if err is not None or (report is not None
                               and report.state in STICKY_INVALID_STATES):
            self.requires_reconstruction = True
            self._last_invalid_step = step
        # policy B: any NEW vanished loop is a pending topology event
        for li in vanished:
            if li not in self._vanished_seen:
                self._vanished_seen.add(li)
                self.pending_topology_events.append(
                    dict(kind="loop_extinction", step=step,
                         loop_index=li))
        self.records.append(rec)
        return rec

    def _record_at(self, level: str, step: int) -> TimeLevelRecord:
        for r in self.records:
            if r.level == level and r.step == step:
                return r
        raise ValueError(f"no {level} record at step {step}")

    def certify_and_accept_loop_extinction(
            self, step: int, *,
            phase_rank_before: int, phase_rank_after: int,
            area_rtol: float = 1e-9,
    ) -> WholeBoundaryLoopExtinctionCertificate:
        """Resolve a pending loop-extinction event by REVALIDATING it
        from this timeline's OWN records (never from a caller-supplied
        certificate -- a directly instantiated or foreign certificate
        must not clear anything). The phase-rank values are the caller's
        claim from the BEM rank machinery; everything geometric is
        re-derived here. Resolving an event never touches the sticky
        requires_reconstruction flag -- an extinction certificate cannot
        launder a separate invalid-support episode."""
        ev = next((e for e in self.pending_topology_events
                   if e["kind"] == "loop_extinction" and e["step"] == step),
                  None)
        if ev is None:
            raise ValueError(
                f"no pending loop-extinction event at step {step}; "
                f"pending: {self.pending_topology_events}")
        # pre = post_redistribution: the geometry the deletion ACTUALLY
        # acted on (the per-step redistribution stage runs between the
        # MM candidate and the deletion; identity record when off --
        # discovered in D3, where candidate-based pre missed the
        # event-step redistribution drift by ~1e-5 relative)
        cert = certify_loop_extinction(
            self._record_at("post_redistribution", step),
            self._record_at("post_deletion_raw", step),
            self._record_at("post_deletion_final", step),
            phase_rank_before=phase_rank_before,
            phase_rank_after=phase_rank_after,
            area_rtol=area_rtol)
        if cert.loop_index != ev["loop_index"]:
            raise ValueError(
                f"certified loop {cert.loop_index} does not match the "
                f"pending event's loop {ev['loop_index']}")
        self.pending_topology_events.remove(ev)
        self.accepted_certificates.append(cert)
        return cert

    def gated_permissions(self, report: SupportReport) -> dict:
        """`permissions(report)` AND the trajectory state: while the
        sticky invalid flag is set OR any topology event is pending
        (unresolved boundary-loop-count change), every gate is closed
        regardless of how valid the CURRENT frame looks."""
        base = _permissions(report)
        if self.requires_reconstruction or self.pending_topology_events:
            return {k: False for k in base}
        return base
