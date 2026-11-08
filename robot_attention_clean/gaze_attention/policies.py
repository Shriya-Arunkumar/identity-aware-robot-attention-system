"""
policies.py — pluggable target-selection policies, swappable at runtime and
via config. This is the module the research question lives in.

    target_id = policy.update(observations, dt)

`observations` is the list of per-person dicts produced by main_webcam.py /
replay.py after filtering (see gaze_attention/perception.py and filters.py):
each dict has the pinned keys `id`, `bbox`, `gaze_signal`, `confidence`,
`seeking_attention`, `timestamp`. By the time a policy sees them:

  * `gaze_signal` is the FILTERED signal (`filtered_signal`), and
    `seeking_attention` has been recomputed against it
    (`seeking_attention = gaze_signal >= seeking_threshold`) — this is what
    makes the filter ablation meaningful: the same policy sees a
    differently-stabilized seeking signal depending on which filter ran
    upstream.
  * `confidence` is `policy_confidence`, NOT the filter's own uncertainty
    about its smoothed estimate (`filter_uncertainty`, see filters.py).
    `confidence_threshold` (all policies) and `min_confidence`
    (ConfidenceDebtPolicy) gate on this value. See DEBUG_ATTENTION.md FM1
    for why this distinction matters: a filter that's been running
    steadily converges to low uncertainty regardless of whether the
    underlying evidence was ever trustworthy, so gating on filter
    uncertainty instead of actual confidence would let a
    consistently-low-confidence person pass the gate purely because the
    filter had time to settle. `policy_confidence`'s current ("Version 1")
    definition lives in main_webcam.py/evaluation/replay.py, upstream of
    this module.

`ConversationalSimplePolicy` is the ACTIVE policy (see its own docstring).
`ConfidenceDebtPolicy` and `GreedyDwellPolicy` (moved to
baseline_policies.py — re-exported here for backward-compatible imports;
see that module's docstring) remain available as baselines/ablation
comparisons, never as the default. `ConfidenceDebtPolicy` and
`GreedyDwellPolicy` share identical arbitration mechanics (minimum hold
time + deterministic tie-break by lowest person id) via
`dwell_arbitration._DwellArbitration` — `ConversationalSimplePolicy`
implements its own resolution logic instead (speaker-priority + bounded
short-term fallback scoring), since a single shared `_scores()` hook can't
express "force-select a confirmed speaker, bypassing the score comparison
entirely."

visual_speaking_proxy (optional, off by default; ConfidenceDebtPolicy/
GreedyDwellPolicy only — ConversationalSimplePolicy always reads
speaking/speech_confidence directly, see its own docstring)
--------------------------------------------------
When `speech_mode="mouth_motion"`, `_DwellArbitration` also reads
`observations`' `speaking` and `speech_confidence` keys (see
gaze_attention/speech.py — a mouth-motion visual cue, NOT audio speech
recognition; there is no microphone). A person who is both `speaking` and
`seeking_attention` (this project's "looking_at_robot" proxy) gets a
`speech_weight * speech_confidence` bonus added to their score — supporting
evidence only. This bonus is folded into `scores` BEFORE the existing
hold-time/tie-break gate below runs, so speech can never bypass that gate —
only nudge which candidate looks strongest once the gate is checked. A
person who is speaking but not looking at the robot gets no bonus and thus
cannot cause any switch, immediate or otherwise. When `speech_mode="off"`
(the default), the bonus is always exactly 0.0 and gaze-only behavior is
unchanged.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from gaze_attention.baseline_policies import GreedyDwellPolicy
from gaze_attention.dwell_arbitration import DecisionPolicy, Observation, _DwellArbitration

_logger = logging.getLogger(__name__)

__all__ = [
    "Observation", "DecisionPolicy",
    "GreedyDwellPolicy", "ConfidenceDebtPolicy", "ConversationalSimplePolicy",
    "POLICIES", "make_policy",
]


class ConfidenceDebtPolicy(_DwellArbitration):
    """Confidence-gated, debt-weighted dwell policy — the proposed policy.

    Update rule (applied per person per tick, for everyone present this
    tick — see `_DwellArbitration` for how an absent current target is
    instead protected/frozen rather than touched by any of this):

        if seeking[i]:
            engagement[i] += confidence[i] * dt          # confidence-weighted growth
        else:
            engagement[i] -= engagement_decay * dt        # decays when not seeking
        engagement[i] = clamp(engagement[i], 0, engagement_max)

        if i == current_target:
            debt[i] = 0                                    # reset on selection
        elif seeking[i] and confidence[i] >= min_confidence:
            debt[i] += dt                                  # only overlooked, confident, active seekers
        debt[i] = clamp(debt[i], 0, debt_max)

        score[i] = engagement[i] + debt_weight * debt[i]

    Confidence shapes engagement growth (proportional, not just gated,
    unlike Greedy) but NOT its decay — decay is a flat rate, deliberately
    symmetric with GreedyDwellPolicy's `engagement_decay`, so the two
    policies differ only in how they *grow* engagement, not in how idle
    time is penalized. "Debt" accrues for people who are actively seeking
    attention with trustworthy evidence but are being ignored (not the
    current target) — a confidently-ignored seeker becomes increasingly
    likely to be selected next, which is exactly the behavior
    GreedyDwellPolicy has no mechanism for.

    Without decay, a person who stops seeking keeps whatever engagement
    they'd already banked forever (see DEBUG_ATTENTION.md FM2) — with no
    one else in frame to build competing debt, that made the policy
    "stick" to a disengaged person indefinitely. Decay fixes that directly.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.45,
        debt_weight: float = 0.50,
        engagement_decay: float = 0.10,
        minimum_hold_seconds: float = 0.75,
        target_grace_seconds: float = 0.5,
        engagement_max: float = 30.0,
        debt_max: float = 30.0,
        speech_mode: str = "off",
        speech_weight: float = 0.0,
        use_confidence: bool = True,
    ):
        super().__init__(
            minimum_hold_seconds=minimum_hold_seconds,
            target_grace_seconds=target_grace_seconds,
            speech_mode=speech_mode,
            speech_weight=speech_weight,
        )
        self.min_confidence = confidence_threshold
        self._use_confidence = use_confidence
        self.debt_weight = debt_weight
        self.engagement_decay = engagement_decay
        self.engagement_max = engagement_max
        self.debt_max = debt_max
        self._engagement: Dict[int, float] = {}
        self._debt: Dict[int, float] = {}

    def _scores(self, observations: List[Observation], dt: float) -> Dict[int, float]:
        current_target = self._current_target
        scores: Dict[int, float] = {}
        for obs in observations:
            pid = obs["id"]
            c_eff = obs["confidence"] if self._use_confidence else 1.0
            seeking = obs["seeking_attention"]

            engagement = self._engagement.get(pid, 0.0)
            if seeking:
                engagement += c_eff * dt
            else:
                engagement -= self.engagement_decay * dt
            engagement = float(max(0.0, min(self.engagement_max, engagement)))

            debt = self._debt.get(pid, 0.0)
            if pid == current_target:
                debt = 0.0
            elif seeking and c_eff >= self.min_confidence:
                debt += dt
            debt = float(max(0.0, min(self.debt_max, debt)))

            self._engagement[pid] = engagement
            self._debt[pid] = debt
            scores[pid] = engagement + self.debt_weight * debt
        return scores

    def _prune(self, present_ids: set, protect_id: Optional[int] = None) -> None:
        for store in (self._engagement, self._debt):
            for pid in list(store):
                if pid not in present_ids and pid != protect_id:
                    del store[pid]

    def reset(self) -> None:
        super().reset()
        self._engagement.clear()
        self._debt.clear()

    def get_metrics(self, person_id: int) -> Dict[str, float]:
        """Read-only per-person values for CSV logging/replay. `score`
        includes the speech bonus (0.0 when speech_mode == "off"),
        matching what selection actually used."""
        engagement = self._engagement.get(person_id, 0.0)
        debt = self._debt.get(person_id, 0.0)
        bonus = self._last_speech_bonus.get(person_id, 0.0)
        return {
            "engagement": engagement, "debt": debt,
            "score": engagement + self.debt_weight * debt + bonus,
            "speech_bonus": bonus,
        }


class ConversationalSimplePolicy(DecisionPolicy):
    """Conversational, speaker-first target selection -- fast path.

    Replaces (as the ACTIVE policy) the old approach of folding a small
    speech bonus into ConfidenceDebtPolicy's unbounded lifetime engagement
    score (see `_DwellArbitration._speech_bonus`) — that design let a
    target's already-large accumulated engagement outweigh a person who
    had clearly started talking, so a stale target could stay selected for
    10-15s+ while someone else was visibly speaking. This policy never
    compares a speaker candidate against anyone's historical engagement:
    a confirmed speaker is chosen directly, and the non-speaker fallback
    path only ever scores off short-term (bounded, decaying) state.

    A second round of live testing found this was STILL too slow to react:
    the previous defaults (`speaker_confirm_seconds=0.75`,
    `speaker_release_seconds=1.5`, and a `speaker_min_attention_seconds=5.0`
    floor that blocked release) meant a speaker who'd only just been
    confirmed could keep the target for up to several seconds after going
    silent, and — worse — a SECOND person who started speaking while the
    first was still nominally "active" had no way to take over until the
    first one's slow release timer finally expired. Two changes fix this,
    without adding a new state machine (still the same "compute this tick's
    active_speaker_id, then resolve_target" shape as before):

    1. Much faster default timing (see `__init__` for the exact defaults):
       `speaker_confirm_seconds`, `speaker_release_seconds`, and a new
       `minimum_speaker_attention_seconds` floor (renamed from
       `speaker_min_attention_seconds`, shrunk from 5.0 to 2.0) are all a
       fraction of their old values.
    2. Speaker-to-speaker PREEMPTION (`speaker_switch_seconds`, new): if a
       DIFFERENT person is already speaking (`speaking_true_duration`
       tracked for every present person continuously, regardless of who
       currently holds `active_speaker_id`) and has been stably speaking
       for at least `speaker_switch_seconds`, they immediately take over
       as `active_speaker_id` — the old active speaker does NOT have to
       silence-release first. Release-to-fallback (when NO ONE is
       speaking) still goes through `speaker_release_seconds` +
       `minimum_speaker_attention_seconds`, which protects a
       just-confirmed speaker from instant flicker back to the gaze
       fallback on a brief pause, but that floor never blocks preemption
       by an actual competing speaker — see `_update_active_speaker`.

    Two independent modes, resolved every tick:

    1. Active-speaker mode: a person becomes a speaker candidate purely
       from `speaking` + `speech_confidence` -- `seeking_attention`
       (looking at the robot) is deliberately NOT required. Once
       `speaking_true_duration[pid]` (consecutive seconds `speaking` has
       been True, regardless of gaze direction) reaches
       `speaker_confirm_seconds` AND `speech_confidence >=
       speaker_confidence_threshold`, `pid` is a "valid speaker"
       (`speaker_candidate_id`, the most-established one if several
       qualify). Priority order (requirement 9): valid speaker > current
       active speaker > sustained-gaze candidate > confidence-debt
       fallback -- i.e. speaker-mode always short-circuits before the
       fallback scorer ever runs. Selected immediately -- bypassing
       `minimum_hold_seconds` -- because waiting out an old target's hold
       timer while someone is actively talking is exactly the bug this
       policy exists to fix. Never abandoned by the stare penalty while
       still speaking -- that penalty only ever acts on the non-speaker
       fallback path below.

    2. Fallback mode (no active speaker): target selection uses only
       bounded, per-tick state: `short_term_score` (grows with current
       gaze_signal/confidence while looking at the robot, decays at
       `non_speaking_score_decay`/s otherwise, capped at `SCORE_MAX` --
       never allowed to become a multi-minute-spanning "unbounded
       historical dwell"), `attention_debt` (small bounded fairness credit
       for a confident seeker being overlooked, reset the instant they
       become target), `quiet_penalty` (bounded quietness penalty, see
       below), and `recent_attention_penalty` (a bounded, gradually-ramping
       soft penalty once someone has held the target past
       `max_continuous_gaze_seconds`). Normal `minimum_hold_seconds`/
       `target_grace_seconds` gating applies between two fallback
       candidates, same mechanics as `_DwellArbitration` -- requirement 10:
       ordinary non-speaker hold/grace timing is UNCHANGED, only the
       speaker path was sped up.

    Bounded quietness penalty (fast path)
    ---------------------------------------
    Per person, every tick: `quiet_penalty += quiet_decay_rate * dt` if
    NEITHER speaking NOR looking at the robot; reset to 0 the instant
    `speaking` is True; held neutral (unchanged) while looking without
    speaking -- a quiet person maintaining reliable eye contact must NOT
    be penalized for it. Clamped to `[0, quiet_penalty_max]`. Because
    speaking always zeroes it, an active speaker's own `quiet_penalty` is
    always exactly 0.0 while they're speaking -- structurally unable to
    contribute to overriding an active speaker (who bypasses `_score_of`
    entirely in speaker mode anyway).

    The final per-person score (`_score_of`, also what `get_metrics()`
    reports) is:

        score = short_term_score               # positive gaze evidence
              + ACTIVE_SPEAKER_BONUS (if pid is the active speaker)
              + DEBT_WEIGHT * attention_debt    # optional attention debt
              - quiet_penalty                   # bounded quietness penalty
              - recent_attention_penalty         # bounded stare penalty

    Every term is individually clamped, so the sum can never grow without
    bound in either direction.

    Diagnostics (requirement 11): `speaking_true_duration`,
    `speaking_false_duration`, `last_speaking_time`, `active_speaker_id`,
    `speaker_candidate_id`, `speaker_confirmed_time`/`speaker_release_time`
    (per person, session-clock timestamps), and `last_switch_reason`/
    `last_switch_latency_seconds` (set whenever `current_target` actually
    changes -- one of "initial", "speaker_confirmed", "speaker_preempt",
    "speaker_confirmed_after_release", "fallback_score",
    "fallback_grace_expired"; latency is how long the winning evidence had
    to accumulate before the switch fired) are all public attributes for
    logging -- see main_webcam.py/logger.py.
    """

    SCORE_MAX = 5.0
    DEBT_MAX = 5.0
    DEBT_WEIGHT = 0.5
    PENALTY_CAP = 1.0
    ACTIVE_SPEAKER_BONUS = 5.0  # positive "active-speaker evidence" term, see class docstring

    def __init__(
        self,
        confidence_threshold: float = 0.45,
        minimum_hold_seconds: float = 1.5,
        target_grace_seconds: float = 0.5,
        engagement_decay: float = 0.10,  # accepted for call-signature parity with other policies; unused (this policy uses non_speaking_score_decay instead)
        speech_mode: str = "off",  # gates speaking/speech_confidence consumption below (see _effective_speaking/_effective_speech_confidence) -- "off" means this policy behaves as if no one is ever speaking
        speech_weight: float = 0.0,  # accepted for call-signature parity; unused, see above
        speaker_confirm_seconds: float = 0.25,
        speaker_release_seconds: float = 0.40,
        speaker_switch_seconds: float = 0.25,
        minimum_speaker_attention_seconds: float = 2.0,
        speaker_confidence_threshold: float = 0.25,
        non_speaking_score_decay: float = 0.5,
        recent_attention_penalty_weight: float = 0.2,
        max_continuous_gaze_seconds: float = 8.0,
        quiet_decay_rate: float = 1.0,
        quiet_penalty_max: float = 3.0,
        # -- stabilization patch (flicker fix, see DEBUG_ATTENTION.md) -------
        speaker_switch_confirm_seconds: float = 0.20,
        speech_off_confirm_seconds: float = 0.7,
        speaker_absence_grace_seconds: float = 0.8,
        use_confidence: bool = True,
        # -- debt-oscillation fix (Root Cause #1) ----------------------------
        fallback_switch_margin: float = 0.10,
        debt_requires_speech: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self._use_confidence = use_confidence
        self.speech_mode = speech_mode
        self.fallback_switch_margin = fallback_switch_margin
        self.debt_requires_speech = debt_requires_speech
        self.minimum_hold_seconds = minimum_hold_seconds
        self.target_grace_seconds = target_grace_seconds
        self.speaker_confirm_seconds = speaker_confirm_seconds
        self.speaker_release_seconds = speaker_release_seconds
        self.speaker_switch_seconds = speaker_switch_seconds
        self.minimum_speaker_attention_seconds = minimum_speaker_attention_seconds
        self.speaker_confidence_threshold = speaker_confidence_threshold
        self.non_speaking_score_decay = non_speaking_score_decay
        self.recent_attention_penalty_weight = recent_attention_penalty_weight
        self.max_continuous_gaze_seconds = max_continuous_gaze_seconds
        self.quiet_decay_rate = quiet_decay_rate
        self.quiet_penalty_max = quiet_penalty_max
        self.speaker_switch_confirm_seconds = speaker_switch_confirm_seconds
        self.speech_off_confirm_seconds = speech_off_confirm_seconds
        self.speaker_absence_grace_seconds = speaker_absence_grace_seconds

        # -- requirement 1: tracked per-person state ---------------------------
        self.speaking_true_duration: Dict[int, float] = {}
        self.speaking_false_duration: Dict[int, float] = {}
        self.last_speaking_time: Dict[int, float] = {}
        self.speaker_candidate_id: Optional[int] = None
        self.active_speaker_id: Optional[int] = None

        self.short_term_score: Dict[int, float] = {}
        self.attention_debt: Dict[int, float] = {}
        self.recent_attention_penalty: Dict[int, float] = {}
        self.attention_duration: Dict[int, float] = {}
        self.quiet_penalty: Dict[int, float] = {}

        # requirement 11: diagnostics
        self.speaker_confirmed_time: Dict[int, float] = {}
        self.speaker_release_time: Dict[int, float] = {}
        self.last_switch_reason: Optional[str] = None
        self.last_switch_latency_seconds: float = 0.0
        self.switched_this_tick: bool = False

        self._clock: float = 0.0
        self._speaker_hold_time: Dict[int, float] = {}  # time since promotion, for the minimum_speaker_attention_seconds floor
        self._current_target: Optional[int] = None
        self._time_since_switch: float = 0.0
        self._target_absent_time: float = 0.0
        self._pending_switch_reason: Optional[str] = None
        self._pending_switch_latency: float = 0.0

        # -- stabilization patch state ---------------------------------------
        self._speaker_absent_time: Dict[int, float] = {}  # requirement 5: absence grace per active speaker
        self.pending_challenger_id: Optional[int] = None  # requirement 3
        self.pending_challenger_start_time: Optional[float] = None  # requirement 3
        self.last_decision_log: Dict[str, object] = {}  # requirement 8

        # -- debt-oscillation diagnostics (Root Cause #1) --------------------
        # no_speaker_hold: True on any tick resolved through the fallback
        # path with NO confirmed active speaker -- the state in which the
        # ping-pong used to occur. last_fallback_decision records every
        # ATTEMPTED fallback switch (taken or rejected) with the exact
        # numbers the decision was made from.
        self.no_speaker_hold: bool = False
        self.last_fallback_decision: Dict[str, object] = {}

        # -- Defect C: raw-speaking confirmation-window protection ----------
        # Who (if anyone) currently has raw speaking evidence but is not yet
        # the confirmed active speaker, and how long they have been at it.
        self.raw_speaker_candidate_id: Optional[int] = None
        self.raw_speaker_candidate_since: Optional[float] = None
        self.raw_speaker_candidate_elapsed: float = 0.0
        self.speaker_confirmation_pending: bool = False

        # -- Root Cause #3: latency diagnostics -----------------------------
        # speaking_onset_time[pid] is the session-clock time raw speaking
        # evidence STARTED for pid (set on the 0 -> >0 transition, so it
        # survives the speech-off freeze and is not re-stamped mid-utterance).
        # last_switch_latency_breakdown decomposes the most recent switch into
        # its measured stages -- never inferred from frame counts.
        self.speaking_onset_time: Dict[int, float] = {}
        self.last_switch_latency_breakdown: Dict[str, object] = {}

    @property
    def current_target(self) -> Optional[int]:
        return self._current_target

    def reset(self) -> None:
        self.speaking_true_duration.clear()
        self.speaking_false_duration.clear()
        self.last_speaking_time.clear()
        self.speaker_candidate_id = None
        self.active_speaker_id = None
        self.short_term_score.clear()
        self.attention_debt.clear()
        self.recent_attention_penalty.clear()
        self.attention_duration.clear()
        self.quiet_penalty.clear()
        self.speaker_confirmed_time.clear()
        self.speaker_release_time.clear()
        self.last_switch_reason = None
        self.last_switch_latency_seconds = 0.0
        self.switched_this_tick = False
        self._clock = 0.0
        self._speaker_hold_time.clear()
        self._current_target = None
        self._time_since_switch = 0.0
        self._target_absent_time = 0.0
        self._pending_switch_reason = None
        self._pending_switch_latency = 0.0
        self._speaker_absent_time.clear()
        self.pending_challenger_id = None
        self.pending_challenger_start_time = None
        self.last_decision_log = {}
        self.no_speaker_hold = False
        self.last_fallback_decision = {}
        self.raw_speaker_candidate_id = None
        self.raw_speaker_candidate_since = None
        self.raw_speaker_candidate_elapsed = 0.0
        self.speaker_confirmation_pending = False
        self.speaking_onset_time.clear()
        self.last_switch_latency_breakdown = {}

    def _effective_speaking(self, obs: dict) -> bool:
        """Bug fix: this policy previously read obs["speaking"] directly
        everywhere, ignoring speech_mode entirely -- perception.py's mouth-
        motion visual_speaking_proxy runs unconditionally (see its own
        docstring), so with speech_mode == "off" this class was still
        silently driving speaker_preempt/active-speaker selection off mouth
        motion. Gated at the point of consumption (same pattern as
        _use_confidence): perception.py keeps computing and logging the
        real value regardless; only this policy's use of it is bypassed."""
        if self.speech_mode == "off":
            return False
        return obs.get("speaking", False)

    def _effective_speech_confidence(self, obs: dict) -> float:
        if self.speech_mode == "off":
            return 0.0
        return obs.get("speech_confidence", 0.0)

    def _score_of(self, pid: int) -> float:
        """positive gaze evidence + active-speaker evidence + optional
        attention debt - quiet penalty - bounded stare penalty. Every term
        is individually bounded, so this can never grow without bound.

        This is the REPORTED score (get_metrics/logging/overlay) and is
        deliberately unchanged by the debt-oscillation fix -- debt is still
        fully accrued and still visible here. What changed is only which
        score the fallback SWITCH COMPARISON uses; see
        `_switch_score_of`."""
        return self._score_components(pid, include_debt=True)

    def _score_components(self, pid: int, include_debt: bool) -> float:
        positive_gaze_evidence = self.short_term_score.get(pid, 0.0)
        active_speaker_evidence = self.ACTIVE_SPEAKER_BONUS if pid == self.active_speaker_id else 0.0
        debt_term = self.DEBT_WEIGHT * self.attention_debt.get(pid, 0.0) if include_debt else 0.0
        quiet_penalty = self.quiet_penalty.get(pid, 0.0)
        stare_penalty = self.recent_attention_penalty.get(pid, 0.0)
        return positive_gaze_evidence + active_speaker_evidence + debt_term - quiet_penalty - stare_penalty

    def _has_raw_speech_evidence(self, pid: Optional[int]) -> bool:
        """RAW (unconfirmed) speaking evidence: the speech proxy's `speaking`
        boolean has been true for >0s. Reported in diagnostics only -- it must
        NOT gate debt (see _has_confirmed_speaker and Defect A)."""
        if pid is None:
            return False
        return self.speaking_true_duration.get(pid, 0.0) > 0.0

    def _has_confirmed_speaker(self, pid: Optional[int]) -> bool:
        """CONFIRMED speaker status -- `pid` actually holds active_speaker_id,
        i.e. it cleared speaker_confirm_seconds and the speech-confidence
        gate in _update_active_speaker. This is the only speech notion allowed
        to affect debt."""
        return pid is not None and pid == self.active_speaker_id

    def _debt_counts_for_switch(self, current_id: Optional[int], candidate_id: Optional[int]) -> bool:
        """Root Cause #1: debt may not by itself manufacture a fallback
        switch while nobody is speaking. Debt still accrues and is still
        reported by _score_of/get_metrics -- it is only excluded from the
        switch COMPARISON.

        Defect A: this gate originally used RAW speaking evidence
        (speaking_true_duration > 0), so the instant a candidate's first
        `speaking` frame arrived, their BANKED debt was re-admitted and
        immediately won through the fallback path -- taking the target before
        speaker confirmation had run at all (measured: target moved at
        t=1.067 with active_speaker_id still None, score_difference 0.42 vs
        margin 0.10). A person merely in the PROCESS of becoming a speaker
        must not be able to cash in banked debt. The gate is therefore
        CONFIRMED speaker status, not a raw mouth cue.

        Because _resolve_fallback_target only runs when there is no usable
        active speaker, this means debt is effectively excluded from the
        fallback comparison whenever debt_requires_speech is True -- which is
        the intended conservative behavior. Set debt_requires_speech=False to
        restore the original unconditional debt comparison."""
        if not self.debt_requires_speech:
            return True
        return self._has_confirmed_speaker(current_id) or self._has_confirmed_speaker(candidate_id)

    def _switch_score_of(self, pid: int, include_debt: bool) -> float:
        """Score used ONLY for the fallback switch comparison / candidate
        ranking -- never for reporting."""
        return self._score_components(pid, include_debt=include_debt)

    def get_metrics(self, person_id: int) -> Dict[str, float]:
        return {
            "engagement": self.short_term_score.get(person_id, 0.0),
            "debt": self.attention_debt.get(person_id, 0.0),
            "score": self._score_of(person_id),
            "speech_bonus": 0.0,  # this policy has no separate speech bonus -- speaking drives selection directly
            "speaking_true_duration": self.speaking_true_duration.get(person_id, 0.0),
            "speaking_false_duration": self.speaking_false_duration.get(person_id, 0.0),
            "quiet_penalty": self.quiet_penalty.get(person_id, 0.0),
            "is_active_speaker": 1.0 if person_id == self.active_speaker_id else 0.0,
        }

    def update(self, observations: List[Observation], dt: float) -> Optional[int]:
        present_ids = {obs["id"] for obs in observations}
        obs_by_id = {obs["id"]: obs for obs in observations}
        # Challenger-eligibility guard (mirrors dwell_arbitration.py's): a
        # lost=True observation (perception.py's frozen stand-in for a
        # currently-occluded track) can never become the active speaker or
        # the target, and can never challenge for either -- it carries no
        # new evidence, just a frozen last-known value. This does NOT touch
        # present_ids/target_grace_seconds/speaker_absence_grace_seconds --
        # those remain the only way a lost person retains something they
        # already hold.
        eligible_ids = {obs["id"] for obs in observations if not obs.get("lost", False)}
        self._clock += dt
        previous_target = self._current_target

        self._update_speaking_timers(obs_by_id, present_ids, dt)
        speaker_candidate_id = self._compute_speaker_candidate(obs_by_id, eligible_ids)
        self.speaker_candidate_id = speaker_candidate_id
        self._update_active_speaker(obs_by_id, present_ids, eligible_ids, speaker_candidate_id, dt)

        target = self._resolve_target(present_ids, eligible_ids, dt)

        self._update_fallback_scores(obs_by_id, present_ids, target, dt)
        self._prune(present_ids)

        self.switched_this_tick = target != previous_target
        if self.switched_this_tick:
            self._record_switch(target, previous_target)

        self._current_target = target
        self._log_decision(obs_by_id, previous_target)
        return target

    # -- speaker timers (requirement 1) ------------------------------------------

    def _update_speaking_timers(self, obs_by_id, present_ids, dt) -> None:
        """Requirement 4 (speech-off confirmation): a single speaking=False
        frame must NOT immediately reset speaking_true_duration -- that made
        the confirmation clock fragile to one noisy frame from the
        mouth-motion proxy, on top of whatever hysteresis speech.py already
        applies to the boolean itself. Instead, speaking_false_duration
        accumulates while not speaking, and speaking_true_duration is only
        reset to 0 once speech_off_confirm_seconds of CONTINUOUS silence has
        been confirmed -- until then it stays frozen (not incremented,
        not reset), so a brief pause in natural speech doesn't cost a
        confirmed/near-confirmed speaker their progress."""
        for pid in present_ids:
            speaking = self._effective_speaking(obs_by_id[pid])
            if speaking:
                # Root Cause #3 latency diagnostics: stamp the ONSET of this
                # speaking run (0 -> >0 only), so a mid-utterance pause that
                # freezes speaking_true_duration doesn't restamp it.
                if self.speaking_true_duration.get(pid, 0.0) <= 0.0:
                    self.speaking_onset_time[pid] = self._clock
                self.speaking_true_duration[pid] = self.speaking_true_duration.get(pid, 0.0) + dt
                self.speaking_false_duration[pid] = 0.0
                self.last_speaking_time[pid] = self._clock
            else:
                false_duration = self.speaking_false_duration.get(pid, 0.0) + dt
                self.speaking_false_duration[pid] = false_duration
                if false_duration >= self.speech_off_confirm_seconds:
                    self.speaking_true_duration[pid] = 0.0
                # else: speaking_true_duration left untouched (frozen).
        for store in (self.speaking_true_duration, self.speaking_false_duration,
                      self.last_speaking_time, self.speaking_onset_time):
            for pid in list(store):
                if pid not in present_ids:
                    del store[pid]

    def _compute_speaker_candidate(self, obs_by_id, present_ids, exclude: Optional[int] = None) -> Optional[int]:
        """requirement 3: a "valid speaker" is anyone whose speaking_true_duration
        has reached speaker_confirm_seconds with sufficient speech_confidence.
        When several qualify, the most-established one wins (longest
        speaking_true_duration), ties broken by lowest id -- deterministic,
        same convention used elsewhere in this file. `exclude` (used to look
        for a PREEMPTING challenger, see `_update_active_speaker`) leaves the
        current active speaker out of consideration -- otherwise, once
        they're well-established, they'd always "win" against themselves
        and no one could ever preempt them."""
        candidates = [
            pid for pid in present_ids
            if pid != exclude
            and self._effective_speaking(obs_by_id[pid])
            and self.speaking_true_duration.get(pid, 0.0) >= self.speaker_confirm_seconds
            and self._effective_speech_confidence(obs_by_id[pid]) >= self.speaker_confidence_threshold
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda pid: (-self.speaking_true_duration[pid], pid))

    def _update_active_speaker(self, obs_by_id, present_ids, eligible_ids, speaker_candidate_id, dt) -> None:
        active = self.active_speaker_id

        if active is None:
            if speaker_candidate_id is not None:
                self._promote(speaker_candidate_id, "speaker_confirmed")
            return

        if active not in present_ids:
            # Requirement 5: hold (don't release) the active speaker through
            # a temporary disappearance -- occlusion, a fast head turn, a
            # tracker miss -- up to speaker_absence_grace_seconds. Only
            # release once that grace has elapsed.
            absent_time = self._speaker_absent_time.get(active, 0.0) + dt
            self._speaker_absent_time[active] = absent_time
            if absent_time >= self.speaker_absence_grace_seconds:
                self._release(active)
                self._speaker_absent_time.pop(active, None)
                if speaker_candidate_id is not None:
                    self._promote(speaker_candidate_id, "speaker_confirmed_after_release")
            return

        self._speaker_absent_time.pop(active, None)
        self._speaker_hold_time[active] = self._speaker_hold_time.get(active, 0.0) + dt

        if self.speaking_true_duration.get(active, 0.0) > 0.0:
            # Requirement 6: a DIFFERENT already-valid speaker can preempt
            # once THEY have been stably speaking for speaker_switch_seconds
            # -- do not wait for the current speaker to go silent and
            # release first. Computed with `active` excluded, since the
            # active speaker is (almost always) the most-established
            # candidate and would otherwise mask any real challenger.
            #
            # Stabilization guard: when TWO people are both continuously,
            # validly speaking at once, the exclude-active comparison above
            # means whichever one ISN'T currently active always re-qualifies
            # as a "challenger" -- without a guard, this flips active_speaker_id
            # back and forth every single tick forever (neither ever
            # accumulates the sustained identity the pending-challenger gate
            # in _resolve_target needs). So: an active speaker is only
            # vulnerable to preemption once they've held that status for a
            # protection window STRICTLY LONGER than speaker_switch_confirm_seconds
            # -- long enough for _resolve_target's own confirmation window to
            # fully complete once a preemption does happen, rather than
            # racing it and being undone by the next reverse "preemption"
            # right as it was about to finish.
            promotion_protection_seconds = self.speaker_switch_confirm_seconds + self.speaker_confirm_seconds
            time_since_promoted = self._clock - self.speaker_confirmed_time.get(active, self._clock)
            if time_since_promoted >= promotion_protection_seconds:
                challenger = self._compute_speaker_candidate(obs_by_id, eligible_ids, exclude=active)
                if challenger is not None and self.speaking_true_duration.get(challenger, 0.0) >= self.speaker_switch_seconds:
                    self._promote(challenger, "speaker_preempt")
            return

        # Requirement 5: current speaker has gone quiet -- release after
        # speaker_release_seconds of silence, gated by the
        # minimum_speaker_attention_seconds floor (protects a
        # just-confirmed speaker from instant flicker on a brief pause;
        # does NOT gate preemption above, only release-to-nobody/fallback).
        # "Immediately evaluate all other tracked people" -> re-check
        # speaker_candidate_id for promotion in the SAME tick as release.
        silence = self.speaking_false_duration.get(active, 0.0)
        hold = self._speaker_hold_time.get(active, 0.0)
        if silence >= self.speaker_release_seconds and hold >= self.minimum_speaker_attention_seconds:
            self._release(active)
            if speaker_candidate_id is not None:
                self._promote(speaker_candidate_id, "speaker_confirmed_after_release")

    def _promote(self, pid: int, reason: str) -> None:
        self.active_speaker_id = pid
        self._speaker_hold_time[pid] = 0.0
        self.speaker_confirmed_time[pid] = self._clock
        # Latency: how long pid's speaking evidence had to accumulate
        # before this promotion fired -- captured now, since
        # speaking_true_duration keeps changing on later ticks.
        self._pending_switch_reason = reason
        self._pending_switch_latency = self.speaking_true_duration.get(pid, 0.0)

    def _release(self, pid: int) -> None:
        self.speaker_release_time[pid] = self._clock
        self._speaker_hold_time.pop(pid, None)
        self.active_speaker_id = None

    # -- target resolution (requirement 4/9/10) ----------------------------------

    def _resolve_target(self, present_ids, eligible_ids, dt) -> Optional[int]:
        """Stabilization patch (see DEBUG_ATTENTION.md flicker diagnosis):
        speaker takeovers used to bypass minimum_hold_seconds/target_grace
        entirely and switch on the SAME tick a speaker was confirmed --
        that's what made target selection flicker rapidly whenever the
        current target (or a second person) started speaking. Now:

          - the current target's own presence/grace validity is computed
            ONCE here (requirement 1's uniform hold applies to both the
            fallback path below AND the speaker path in this method --
            single source of truth for _target_absent_time, no double
            counting with _resolve_fallback_target).
          - a speaker who is not yet current_target must remain the SAME
            confirmed pending_challenger for speaker_switch_confirm_seconds
            (requirement 2/3) AND the incumbent's minimum_hold_seconds must
            already have elapsed (requirement 1) before the switch happens.
          - pending_challenger state is reset whenever the candidate
            changes, disappears, stops being confirmed-speaking, or the
            current target transitions from valid to invalid
            (requirement 3).
        """
        # Explicit path bookkeeping (Root Cause #1): cleared here, set only
        # by _resolve_fallback_target, so a speaker-resolved tick never
        # leaves a stale no_speaker_hold from an earlier fallback tick.
        self.no_speaker_hold = False

        current_present = self._current_target is not None and self._current_target in present_ids
        if self._current_target is not None:
            self._target_absent_time = 0.0 if current_present else self._target_absent_time + dt
        current_valid = current_present or (
            self._current_target is not None and self._target_absent_time < self.target_grace_seconds
        )

        speaker = self.active_speaker_id
        speaker_present = speaker is not None and speaker in present_ids
        speaker_in_grace = (
            speaker is not None and not speaker_present
            and self._speaker_absent_time.get(speaker, 0.0) < self.speaker_absence_grace_seconds
        )

        if speaker is None or not (speaker_present or speaker_in_grace):
            self._clear_pending_challenger()
            return self._resolve_fallback_target(present_ids, eligible_ids, dt)

        if speaker == self._current_target:
            # Already the held target (possibly via speaker_in_grace) --
            # nothing left to confirm. Still advance dwell time so
            # minimum_hold_seconds accounting stays correct for whatever
            # happens after this speaker eventually releases.
            self._clear_pending_challenger()
            self._time_since_switch += dt
            return speaker

        if not speaker_present or self.speaking_true_duration.get(speaker, 0.0) <= 0.0:
            # Requirement 3: candidate absent, or no longer confirmed
            # speaking (their speaking_true_duration already reset via the
            # speech-off confirmation above) -- drop any pending progress.
            self._clear_pending_challenger()
            if current_valid:
                self._time_since_switch += dt
                return self._current_target
            return self._resolve_fallback_target(present_ids, eligible_ids, dt)

        if not current_valid:
            # Requirement 3: current target just became invalid -- nothing
            # left to protect, so the confirmed speaker takes over
            # immediately (same as the no-target bootstrap case), discarding
            # any stale pending-confirmation progress rather than reusing it.
            self._clear_pending_challenger()
            self._pending_switch_reason = (
                "speaker_confirmed_after_release" if self._current_target is not None else "speaker_confirmed"
            )
            self._pending_switch_latency = 0.0
            self._current_target = speaker
            self._time_since_switch = 0.0
            self._target_absent_time = 0.0
            return speaker

        # Requirement 1/2: a valid incumbent is protected. The challenger
        # must remain the SAME confirmed candidate for at least
        # speaker_switch_confirm_seconds, and the incumbent's own
        # minimum_hold_seconds dwell floor must already have elapsed.
        if self.pending_challenger_id != speaker:
            self.pending_challenger_id = speaker
            self.pending_challenger_start_time = self._clock
        pending_duration = self._clock - self.pending_challenger_start_time

        if pending_duration >= self.speaker_switch_confirm_seconds and self._time_since_switch >= self.minimum_hold_seconds:
            self._pending_switch_reason = "speaker_preempt"
            self._pending_switch_latency = pending_duration
            self._current_target = speaker
            self._time_since_switch = 0.0
            self._target_absent_time = 0.0
            self._clear_pending_challenger()
            return speaker

        self._time_since_switch += dt
        return self._current_target

    def _clear_pending_challenger(self) -> None:
        self.pending_challenger_id = None
        self.pending_challenger_start_time = None

    def _raw_speaker_undergoing_confirmation(self, eligible_ids) -> Optional[int]:
        """Defect C: the most-established person who has RAW speaking
        evidence but is not (yet) the confirmed active speaker -- i.e. someone
        currently working through speaker_confirm_seconds.

        Restricted to eligible_ids, so a lost=True observation can never
        create this protection (it carries no new evidence). Ordering matches
        _compute_speaker_candidate: longest speaking_true_duration first,
        ties broken by lowest id."""
        pending = [
            pid for pid in eligible_ids
            if self._has_raw_speech_evidence(pid) and not self._has_confirmed_speaker(pid)
        ]
        if not pending:
            return None
        return min(pending, key=lambda pid: (-self.speaking_true_duration.get(pid, 0.0), pid))

    def _resolve_fallback_target(self, present_ids, eligible_ids, dt) -> Optional[int]:
        # _target_absent_time is already up to date for this tick -- see
        # _resolve_target, the sole caller, which owns that bookkeeping so
        # it isn't double-counted here.
        target_present = self._current_target is not None and self._current_target in present_ids

        # Root Cause #1: this whole method is the no-active-speaker path
        # (_resolve_target only calls it when there is no usable speaker),
        # so record that explicitly rather than leaving "why did it pick
        # this" implicit in a single generic comparison.
        self.no_speaker_hold = self.active_speaker_id is None

        # Whether debt participates in THIS tick's comparison. Computed
        # against the incumbent before the candidate is known; refined below
        # once the candidate is picked (both sides matter).
        include_debt = self._debt_counts_for_switch(self._current_target, None)

        # Defect C: while somebody is mid-confirmation (raw speaking evidence,
        # not yet active_speaker_id), a SILENT fallback challenger must not be
        # able to take the target purely by out-accumulating score.
        # _update_fallback_scores freezes a speaking person's short_term_score
        # ("if speaking: pass"), but during the confirmation window they do
        # not hold ACTIVE_SPEAKER_BONUS yet -- so a silent seeker's still-
        # growing score would overtake them, steal the target, and be
        # preempted back a moment later once confirmation completed.
        raw_speaker_pending = self._raw_speaker_undergoing_confirmation(eligible_ids)
        self.raw_speaker_candidate_id = raw_speaker_pending
        if raw_speaker_pending is None:
            self.raw_speaker_candidate_elapsed = 0.0
            self.raw_speaker_candidate_since = None
        else:
            elapsed = self.speaking_true_duration.get(raw_speaker_pending, 0.0)
            self.raw_speaker_candidate_elapsed = elapsed
            self.raw_speaker_candidate_since = self._clock - elapsed
        self.speaker_confirmation_pending = raw_speaker_pending is not None

        # Candidate selection is restricted to eligible_ids (excludes
        # lost=True) -- target_present above intentionally still checks the
        # full present_ids, since that's the existing target_grace_seconds
        # presence/grace mechanism, untouched. Ranked by the same gated
        # score the switch decision uses, so selection and comparison can
        # never disagree about who the strongest challenger is.
        candidate = (
            min(eligible_ids, key=lambda i: (-self._switch_score_of(i, include_debt), i))
            if eligible_ids else None
        )

        if self._current_target is None:
            if candidate is not None:
                self._pending_switch_reason = "initial"
                self._pending_switch_latency = 0.0
                self._current_target = candidate
                self._time_since_switch = 0.0
                self._target_absent_time = 0.0
            return self._current_target

        if target_present:
            # The strongest challenger is the best eligible person OTHER than
            # the incumbent -- evaluated (and recorded) every tick, so the
            # diagnostic record exists even when the incumbent is also the
            # top-ranked candidate overall and no switch is attempted.
            challengers = eligible_ids - {self._current_target}
            challenger = (
                min(challengers, key=lambda i: (-self._switch_score_of(i, include_debt), i))
                if challengers else None
            )
            if challenger is not None:
                candidate = challenger
                # Re-evaluate debt eligibility now that both sides are known.
                include_debt = self._debt_counts_for_switch(self._current_target, candidate)
                current_score = self._switch_score_of(self._current_target, include_debt)
                candidate_score = self._switch_score_of(candidate, include_debt)
                hold_satisfied = self._time_since_switch >= self.minimum_hold_seconds
                # Root Cause #1: strict margin, not a bare `>`. A bare `>`
                # plus debt (reset for the incumbent, growing for everyone
                # else) made the challenger win by construction every tick.
                margin_satisfied = candidate_score > current_score + self.fallback_switch_margin
                # Defect C: block only a SILENT challenger, and only while
                # someone is actually mid-confirmation. A challenger with its
                # own raw speaking evidence is not "a silent seeker" and is
                # left to the ordinary rules.
                challenger_is_silent = not self._has_raw_speech_evidence(candidate)
                blocked_by_confirmation = self.speaker_confirmation_pending and challenger_is_silent
                take_switch = hold_satisfied and margin_satisfied and not blocked_by_confirmation

                if take_switch:
                    decision, reason = "switch", "fallback_score_switch"
                elif blocked_by_confirmation and hold_satisfied and margin_satisfied:
                    # Would have switched; blocked purely by the Defect C
                    # protection -- distinguishable from debt/margin holds.
                    decision, reason = "hold", "speaker_confirmation_pending"
                elif self.no_speaker_hold:
                    decision, reason = "hold", "no_speaker_hold"
                else:
                    decision, reason = "hold", "target_retention"

                self.last_fallback_decision = {
                    "current_target_id": self._current_target,
                    "candidate_id": candidate,
                    "current_score": current_score,
                    "candidate_score": candidate_score,
                    "score_difference": candidate_score - current_score,
                    "switch_margin": self.fallback_switch_margin,
                    # Raw vs confirmed kept separate (Defect A): only the
                    # CONFIRMED column may gate debt.
                    "current_speaking_state": self._has_raw_speech_evidence(self._current_target),
                    "candidate_speaking_state": self._has_raw_speech_evidence(candidate),
                    "current_speaking_confirmed": self._has_confirmed_speaker(self._current_target),
                    "candidate_speaking_confirmed": self._has_confirmed_speaker(candidate),
                    "debt_current": self.attention_debt.get(self._current_target, 0.0),
                    "debt_candidate": self.attention_debt.get(candidate, 0.0),
                    "debt_counted": include_debt,
                    "hold_satisfied": hold_satisfied,
                    "margin_satisfied": margin_satisfied,
                    # Defect C diagnostics -- make it unambiguous whether a
                    # switch was blocked by the confirmation window rather
                    # than by debt or the switch margin.
                    "raw_speaker_candidate_id": self.raw_speaker_candidate_id,
                    "raw_speaker_candidate_since": self.raw_speaker_candidate_since,
                    "raw_speaker_candidate_elapsed": self.raw_speaker_candidate_elapsed,
                    "active_speaker_id": self.active_speaker_id,
                    "speaker_confirmation_pending": self.speaker_confirmation_pending,
                    "fallback_switch_blocked": blocked_by_confirmation,
                    "fallback_block_reason": "speaker_confirmation_pending" if blocked_by_confirmation else None,
                    "decision": decision,
                    "reason": reason,
                }
                _logger.debug("fallback decision: %s", self.last_fallback_decision)

                if take_switch:
                    self._pending_switch_reason = "fallback_score"
                    self._pending_switch_latency = self._time_since_switch  # how long we waited before this switch fired
                    self._current_target = candidate
                    self._time_since_switch = 0.0
                    self._target_absent_time = 0.0
                    return self._current_target

            self._time_since_switch += dt
            return self._current_target

        grace_expired = self._target_absent_time >= self.target_grace_seconds
        hold_expired = self._time_since_switch >= self.minimum_hold_seconds
        if grace_expired and hold_expired:
            self._pending_switch_reason = "fallback_grace_expired"
            self._pending_switch_latency = self._target_absent_time  # how long the target was absent before release
            self._current_target = candidate
            self._time_since_switch = 0.0
            self._target_absent_time = 0.0
        else:
            self._time_since_switch += dt
        return self._current_target

    # -- diagnostics (requirement 11) --------------------------------------------

    def _record_switch(self, target: Optional[int], previous_target: Optional[int]) -> None:
        """Reads the reason/latency captured at the exact decision point
        (_promote or _resolve_fallback_target) -- both are set right before
        the timers they'd otherwise be derived from get reset to 0, so this
        method only ever consumes already-correct values, never recomputes
        from (by now zeroed) state."""
        self.last_switch_reason = getattr(self, "_pending_switch_reason", None) or "unknown"
        self.last_switch_latency_seconds = getattr(self, "_pending_switch_latency", 0.0)
        self._pending_switch_reason = None
        self._pending_switch_latency = 0.0

        # Root Cause #3: decompose this switch into measured stages. All
        # values are session-clock timestamps accumulated from real dt, never
        # inferred from frame counts. Stages are None when they don't apply
        # (e.g. a fallback switch has no speech onset).
        onset = self.speaking_onset_time.get(target) if target is not None else None
        confirmed = self.speaker_confirmed_time.get(target) if target is not None else None
        now = self._clock
        self.last_switch_latency_breakdown = {
            "reason": self.last_switch_reason,
            "from_id": previous_target,
            "to_id": target,
            "speech_onset_time": onset,
            "confirmed_speaker_time": confirmed,
            "speaker_preempt_time": now if self.last_switch_reason == "speaker_preempt" else None,
            "target_switch_time": now,
            "speech_confirmation_latency": (confirmed - onset) if (onset is not None and confirmed is not None) else None,
            "arbitration_latency": (now - confirmed) if confirmed is not None else None,
            "total_target_switch_latency": (now - onset) if onset is not None else None,
        }
        _logger.debug("switch latency breakdown: %s", self.last_switch_latency_breakdown)

    # -- fallback scoring / quiet + stare penalties (requirement 7/8) ------------

    def _update_fallback_scores(self, obs_by_id, present_ids, target, dt) -> None:
        for pid in present_ids:
            obs = obs_by_id[pid]
            seeking = obs.get("seeking_attention", False)
            speaking = self._effective_speaking(obs)
            c_eff = obs.get("confidence", 0.0) if self._use_confidence else 1.0

            # Decay when neither speaking nor looking at the robot; never
            # decay an active speaker while they're speaking.
            score = self.short_term_score.get(pid, 0.0)
            if speaking:
                pass  # active speaker's own score is irrelevant to selection while speaking; leave untouched
            elif seeking:
                score += c_eff * dt
            else:
                score -= self.non_speaking_score_decay * dt
            self.short_term_score[pid] = float(max(0.0, min(self.SCORE_MAX, score)))

            debt = self.attention_debt.get(pid, 0.0)
            if pid == target:
                debt = 0.0
            elif seeking and c_eff >= self.confidence_threshold:
                debt += dt
            self.attention_debt[pid] = float(max(0.0, min(self.DEBT_MAX, debt)))

            # Requirement 7/8: fast bounded quietness penalty -- speaking
            # always zeroes it (so an active speaker's own quiet_penalty is
            # always 0.0 while they're speaking, structurally unable to
            # affect speaker-mode selection); reliable eye contact without
            # speaking holds it neutral (never strongly penalize a quiet
            # person who is maintaining eye contact); only true
            # disengagement (neither speaking nor looking) grows it,
            # directly at quiet_decay_rate per second, clamped.
            penalty = self.quiet_penalty.get(pid, 0.0)
            if speaking:
                penalty = 0.0
            elif seeking:
                pass  # neutral -- do not penalize sustained eye contact
            else:
                penalty = min(self.quiet_penalty_max, penalty + self.quiet_decay_rate * dt)
            self.quiet_penalty[pid] = float(max(0.0, penalty))

            # Soft stare penalty -- only applies to the non-speaker fallback
            # path (an active speaker is never abandoned solely due to
            # this).
            attention_duration = self.attention_duration.get(pid, 0.0)
            if pid == target and pid != self.active_speaker_id:
                attention_duration += dt
            else:
                attention_duration = 0.0
            self.attention_duration[pid] = attention_duration

            stare_penalty = self.recent_attention_penalty.get(pid, 0.0)
            if pid == target and pid != self.active_speaker_id and attention_duration > self.max_continuous_gaze_seconds:
                stare_penalty = min(self.PENALTY_CAP, stare_penalty + self.recent_attention_penalty_weight * dt)
            else:
                stare_penalty = max(0.0, stare_penalty - self.recent_attention_penalty_weight * dt)
            self.recent_attention_penalty[pid] = stare_penalty

    def _prune(self, present_ids: set) -> None:
        protected = {self.active_speaker_id}
        for store in (
            self.short_term_score, self.attention_debt,
            self.recent_attention_penalty, self.attention_duration,
            self.quiet_penalty,
        ):
            for pid in list(store):
                if pid not in present_ids and pid not in protected:
                    del store[pid]
        for pid in list(self._speaker_hold_time):
            if pid not in present_ids and pid != self.active_speaker_id:
                del self._speaker_hold_time[pid]

    # -- decision logging (requirement 8) -----------------------------------

    def _log_decision(self, obs_by_id, previous_target: Optional[int]) -> None:
        """Records the state the switch decision was actually made from --
        current target, the speaker candidate under consideration, its
        speaking timers, dwell time, and any in-progress pending-challenger
        confirmation. Stored on `self.last_decision_log` (inspectable
        directly in tests) and mirrored to a DEBUG log line."""
        focus_id = self.pending_challenger_id or self.active_speaker_id or self._current_target
        pending_duration = (
            self._clock - self.pending_challenger_start_time
            if self.pending_challenger_start_time is not None else 0.0
        )
        log = {
            "current_target": self._current_target,
            "previous_target": previous_target,
            "candidate": focus_id,
            "speaking": self._effective_speaking(obs_by_id.get(focus_id, {})) if focus_id is not None else False,
            "speaking_true_duration": self.speaking_true_duration.get(focus_id, 0.0) if focus_id is not None else 0.0,
            "speaking_false_duration": self.speaking_false_duration.get(focus_id, 0.0) if focus_id is not None else 0.0,
            "dwell_time": self._time_since_switch,
            "pending_challenger_id": self.pending_challenger_id,
            "pending_challenger_duration": pending_duration,
            "switch_reason": self.last_switch_reason,
            "switched_this_tick": self.switched_this_tick,
        }
        self.last_decision_log = log
        _logger.debug("ConversationalSimplePolicy decision: %s", log)


POLICIES = {
    "greedy": GreedyDwellPolicy,
    "confidence_debt": ConfidenceDebtPolicy,
    "conversational_simple": ConversationalSimplePolicy,
}


def make_policy(name: str, **kwargs) -> DecisionPolicy:
    if name not in POLICIES:
        raise ValueError(f"Unknown policy {name!r}; choose from {list(POLICIES)}")
    return POLICIES[name](**kwargs)
