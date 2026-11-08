"""
attention_manager.py — a higher-level, conversational/socially-bounded
attention layer wrapped AROUND a base `DecisionPolicy` (policies.py).

Does not replace or modify GreedyDwellPolicy / ConfidenceDebtPolicy. Those
remain exactly as they were — this module wraps one of them (`base_policy`)
and either defers to it entirely (`speaker_priority_mode="off"`, or whenever
no speaker/quiet-acknowledgment override applies) or overrides its target
for socially-motivated reasons: a confirmed active speaker, or a brief
acknowledgment of a quiet person maintaining eye contact.

`base_policy.update()` is called every tick regardless, so its own
engagement/debt bookkeeping, minimum_hold_seconds, target_grace_seconds, and
confidence_threshold gating all keep running exactly as before — this class
never reaches into or mutates the base policy's internals. When this class's
own override logic has nothing to say (no confirmed speaker, no
acknowledgment in progress), `AttentionManager.update()` returns exactly
`base_policy.current_target` for that tick.

Requires `speech_mode="mouth_motion"` on the base policy (or at least
`speaking`/`seeking_attention` fields on the observations) to have any
effect — with `speaker_priority_mode="off"` (the default) this class is a
transparent passthrough and gaze-only behavior is byte-for-byte unchanged.

Per-person state (see class docstring below for exact semantics):
    speaker_candidate_time, active_speaker, speaking_duration,
    attention_time, recent_attention_penalty, last_target_time.

No hard timeout ever forces the robot to abandon an active speaker
immediately: `max_continuous_gaze_seconds` only starts a *bounded*,
gradually-ramping `recent_attention_penalty`; only once that penalty
saturates does the manager stop *forcing* the speaker as target for
`gaze_break_seconds` (a "soft" gaze break) — and even then it falls back to
the base policy's own arbitration rather than picking someone arbitrarily,
so if no one else is a credible candidate the speaker typically keeps the
target anyway.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from gaze_attention.policies import DecisionPolicy, Observation


class AttentionManager(DecisionPolicy):
    """Wraps a base `DecisionPolicy` with conversational/social gaze rules.

    Per-person variables
    ---------------------
    speaker_candidate_time[pid]
        Consecutive seconds `pid` has been BOTH `speaking` AND
        `seeking_attention` (looking at the robot) — resets to 0.0 the
        instant either condition breaks. Mouth motion while looking away
        never advances this (requirement 2); a single-frame blip can't
        reach `speaker_confirm_seconds` before it resets (requirement 3).
    active_speaker[pid]
        Bool. True for the (at most one) person currently confirmed as the
        active speaker. Becomes True once `speaker_candidate_time[pid]`
        reaches `speaker_confirm_seconds` (~0.5-1.0s) AND `pid`'s confidence
        clears `confidence_threshold` (preserves the existing confidence
        gate — an untrustworthy mouth-motion reading must not grant
        priority). Becomes False again once `pid` has gone
        `speaker_release_seconds` without speaking-and-looking (a soft
        release, not an instant one — brief pauses in natural speech don't
        immediately drop speaker status).
    speaking_duration[pid]
        Consecutive seconds `pid` has been `speaking` (mouth motion alone,
        regardless of gaze direction) — informational/diagnostic, mirrors
        `speaker_candidate_time`'s gating logic minus the looking
        requirement, so a caller can tell "they were talking but not at the
        robot" apart from "not talking at all".
    attention_time[pid]
        Consecutive seconds the manager's OWN chosen target has equalled
        `pid` (i.e. how long the robot has actually been looking at them,
        not how long they've been eligible). Reset to 0.0 whenever `pid`
        stops being the manager's target.
    recent_attention_penalty[pid]
        Bounded, non-negative. Ramps up at `stare_penalty_weight` per
        second once `attention_time[pid]` exceeds
        `max_continuous_gaze_seconds`; decays back to 0 once `pid` is no
        longer being force-selected. Capped at 1.0. Reaching the cap is
        what triggers a scheduled `gaze_break_seconds` soft break — a
        bounded penalty ramp, never an instant cutoff.
    last_target_time[pid]
        Seconds since `pid` last was the manager's chosen target (0.0 while
        they currently are it).

    Configuration (all defaults chosen to sit inside the ranges requested):
        speaker_priority_mode: "off" | "on". "off" -> pure passthrough to
            base_policy (byte-identical to gaze-only behavior).
        speaker_confirm_seconds: ~0.5-1.0s of continuous speaking+looking
            before someone becomes the active speaker.
        speaker_min_attention_seconds: once forced as target, the active
            speaker is attended to at least this long (~3-6s) before a
            quiet-person acknowledgment is allowed to interrupt them.
        speaker_release_seconds: how long an active speaker may go quiet
            (or leave frame) before losing active-speaker status.
        quiet_gaze_confirm_seconds: how long a non-speaking person must
            hold reliable eye contact (~5-6s) to earn a brief
            acknowledgment.
        quiet_acknowledgment_seconds: length of that acknowledgment
            (~0.8-1.5s) before returning to the active speaker.
        max_continuous_gaze_seconds: attention_time threshold beyond which
            the soft stare penalty starts ramping.
        gaze_break_seconds: length of the soft gaze break once the penalty
            saturates.
        stare_penalty_weight: per-second ramp rate of
            recent_attention_penalty while over `max_continuous_gaze_seconds`.
    """

    PENALTY_CAP = 1.0

    def __init__(
        self,
        base_policy: DecisionPolicy,
        speaker_priority_mode: str = "off",
        speaker_confirm_seconds: float = 0.75,
        speaker_min_attention_seconds: float = 4.0,
        speaker_release_seconds: float = 1.5,
        quiet_gaze_confirm_seconds: float = 5.5,
        quiet_acknowledgment_seconds: float = 1.2,
        max_continuous_gaze_seconds: float = 10.0,
        gaze_break_seconds: float = 2.5,
        stare_penalty_weight: float = 0.4,
        confidence_threshold: float = 0.45,
        use_confidence: bool = True,
    ):
        self.base_policy = base_policy
        self.speaker_priority_mode = speaker_priority_mode
        self._use_confidence = use_confidence
        self.speaker_confirm_seconds = speaker_confirm_seconds
        self.speaker_min_attention_seconds = speaker_min_attention_seconds
        self.speaker_release_seconds = speaker_release_seconds
        self.quiet_gaze_confirm_seconds = quiet_gaze_confirm_seconds
        self.quiet_acknowledgment_seconds = quiet_acknowledgment_seconds
        self.max_continuous_gaze_seconds = max_continuous_gaze_seconds
        self.gaze_break_seconds = gaze_break_seconds
        self.stare_penalty_weight = stare_penalty_weight
        self.confidence_threshold = confidence_threshold

        self.speaker_candidate_time: Dict[int, float] = {}
        self.active_speaker: Dict[int, bool] = {}
        self.speaking_duration: Dict[int, float] = {}
        self.attention_time: Dict[int, float] = {}
        self.recent_attention_penalty: Dict[int, float] = {}
        self.last_target_time: Dict[int, float] = {}

        self._speaker_silence_time: Dict[int, float] = {}
        self._quiet_gaze_time: Dict[int, float] = {}
        self._active_speaker_id: Optional[int] = None
        self._acknowledging_id: Optional[int] = None
        self._ack_time_remaining: float = 0.0
        self._gaze_break_time_remaining: float = 0.0
        self._current_target: Optional[int] = None

    @property
    def current_target(self) -> Optional[int]:
        return self._current_target

    def reset(self) -> None:
        self.base_policy.reset()
        self.speaker_candidate_time.clear()
        self.active_speaker.clear()
        self.speaking_duration.clear()
        self.attention_time.clear()
        self.recent_attention_penalty.clear()
        self.last_target_time.clear()
        self._speaker_silence_time.clear()
        self._quiet_gaze_time.clear()
        self._active_speaker_id = None
        self._acknowledging_id = None
        self._ack_time_remaining = 0.0
        self._gaze_break_time_remaining = 0.0
        self._current_target = None

    def get_metrics(self, person_id: int) -> Dict[str, float]:
        """Delegates to the base policy's metrics (engagement/debt/score/
        speech_bonus) and layers the manager's own per-person values on
        top, for CSV logging/overlay parity with the base policy."""
        metrics = dict(self.base_policy.get_metrics(person_id))
        metrics.update({
            "speaker_candidate_time": self.speaker_candidate_time.get(person_id, 0.0),
            "active_speaker": 1.0 if self.active_speaker.get(person_id, False) else 0.0,
            "speaking_duration": self.speaking_duration.get(person_id, 0.0),
            "attention_time": self.attention_time.get(person_id, 0.0),
            "recent_attention_penalty": self.recent_attention_penalty.get(person_id, 0.0),
            "last_target_time": self.last_target_time.get(person_id, 0.0),
        })
        return metrics

    def update(self, observations: List[Observation], dt: float) -> Optional[int]:
        base_target = self.base_policy.update(observations, dt)

        if self.speaker_priority_mode == "off":
            self._current_target = base_target
            return self._current_target

        present_ids = {obs["id"] for obs in observations}
        obs_by_id = {obs["id"]: obs for obs in observations}

        self._update_speaker_timers(obs_by_id, present_ids, dt)
        self._update_quiet_timers(obs_by_id, present_ids, dt)
        self._release_stale_active_speaker(present_ids, dt)
        self._promote_active_speaker(obs_by_id, present_ids)

        target = self._resolve_target(base_target, obs_by_id, present_ids, dt)

        self._update_attention_bookkeeping(target, present_ids, dt)
        self._current_target = target
        return target

    # -- per-tick timer updates --------------------------------------------

    def _update_speaker_timers(self, obs_by_id, present_ids, dt) -> None:
        for pid in present_ids:
            obs = obs_by_id[pid]
            speaking = obs.get("speaking", False)
            looking = obs.get("seeking_attention", False)
            if speaking:
                self.speaking_duration[pid] = self.speaking_duration.get(pid, 0.0) + dt
            else:
                self.speaking_duration[pid] = 0.0
            if speaking and looking:
                self.speaker_candidate_time[pid] = self.speaker_candidate_time.get(pid, 0.0) + dt
            else:
                self.speaker_candidate_time[pid] = 0.0
        for pid in list(self.speaker_candidate_time):
            if pid not in present_ids:
                del self.speaker_candidate_time[pid]
                self.speaking_duration.pop(pid, None)

    def _update_quiet_timers(self, obs_by_id, present_ids, dt) -> None:
        for pid in present_ids:
            obs = obs_by_id[pid]
            quiet_eye_contact = obs.get("seeking_attention", False) and not obs.get("speaking", False)
            if quiet_eye_contact:
                self._quiet_gaze_time[pid] = self._quiet_gaze_time.get(pid, 0.0) + dt
            else:
                self._quiet_gaze_time[pid] = 0.0
        for pid in list(self._quiet_gaze_time):
            if pid not in present_ids:
                del self._quiet_gaze_time[pid]

    def _release_stale_active_speaker(self, present_ids, dt) -> None:
        pid = self._active_speaker_id
        if pid is None:
            return
        # speaker_candidate_time is already up to date for this tick (set
        # in _update_speaker_timers, and dropped entirely if pid left
        # present_ids), so ">0" is exactly "speaking and looking right now".
        speaking_and_looking = pid in present_ids and self.speaker_candidate_time.get(pid, 0.0) > 0.0
        if speaking_and_looking:
            self._speaker_silence_time[pid] = 0.0
            return
        self._speaker_silence_time[pid] = self._speaker_silence_time.get(pid, 0.0) + dt
        if self._speaker_silence_time[pid] >= self.speaker_release_seconds:
            self.active_speaker[pid] = False
            self._speaker_silence_time.pop(pid, None)
            self._active_speaker_id = None

    def _promote_active_speaker(self, obs_by_id, present_ids) -> None:
        if self._active_speaker_id is not None:
            return
        candidates = [
            pid for pid in present_ids
            if self.speaker_candidate_time.get(pid, 0.0) >= self.speaker_confirm_seconds
            and (obs_by_id[pid].get("confidence", 0.0) if self._use_confidence else 1.0) >= self.confidence_threshold
        ]
        if not candidates:
            return
        chosen = min(candidates)  # deterministic tie-break, same convention as _DwellArbitration
        self._active_speaker_id = chosen
        self.active_speaker[chosen] = True
        self._speaker_silence_time[chosen] = 0.0

    # -- target resolution ---------------------------------------------------

    def _resolve_target(self, base_target, obs_by_id, present_ids, dt) -> Optional[int]:
        # An acknowledgment interlude in progress takes priority over
        # everything else until it completes or its subject disappears.
        if self._acknowledging_id is not None:
            if self._acknowledging_id in present_ids and self._ack_time_remaining > 0.0:
                self._ack_time_remaining -= dt
                return self._acknowledging_id
            self._acknowledging_id = None
            self._ack_time_remaining = 0.0
            # falls through to normal resolution below this tick

        speaker = self._active_speaker_id
        if speaker is not None and speaker in present_ids:
            in_break = self._gaze_break_time_remaining > 0.0
            if in_break:
                self._gaze_break_time_remaining -= dt
                return base_target

            # Requirement 6/7: a quiet person's sustained eye contact earns
            # a brief acknowledgment, but only once the current speaker has
            # already had their minimum guaranteed attention, and only if
            # nobody else is mid-acknowledgment.
            if self.attention_time.get(speaker, 0.0) >= self.speaker_min_attention_seconds:
                quiet_candidates = [
                    pid for pid in present_ids
                    if pid != speaker
                    and self._quiet_gaze_time.get(pid, 0.0) >= self.quiet_gaze_confirm_seconds
                ]
                if quiet_candidates:
                    chosen = min(quiet_candidates)
                    self._acknowledging_id = chosen
                    self._ack_time_remaining = self.quiet_acknowledgment_seconds
                    self._quiet_gaze_time[chosen] = 0.0
                    return chosen

            return speaker

        # No (present) active speaker -> full fallback to the base policy.
        return base_target

    # -- attention/penalty bookkeeping ---------------------------------------

    def _update_attention_bookkeeping(self, target, present_ids, dt) -> None:
        for pid in present_ids:
            if pid == target:
                self.attention_time[pid] = self.attention_time.get(pid, 0.0) + dt
                self.last_target_time[pid] = 0.0
            else:
                self.attention_time[pid] = 0.0
                self.last_target_time[pid] = self.last_target_time.get(pid, 0.0) + dt

            penalty = self.recent_attention_penalty.get(pid, 0.0)
            if pid == target and self.attention_time[pid] > self.max_continuous_gaze_seconds:
                penalty = min(self.PENALTY_CAP, penalty + self.stare_penalty_weight * dt)
            else:
                penalty = max(0.0, penalty - self.stare_penalty_weight * dt)
            self.recent_attention_penalty[pid] = penalty

            # Requirement 5: never an abrupt cutoff -- only once the bounded
            # penalty has fully ramped up (saturated) does a soft gaze break
            # get scheduled, and only for the person currently paying it.
            if pid == target and penalty >= self.PENALTY_CAP and self._gaze_break_time_remaining <= 0.0:
                self._gaze_break_time_remaining = self.gaze_break_seconds
                self.attention_time[pid] = 0.0
                self.recent_attention_penalty[pid] = 0.0

        # Everything except the active speaker's own silence timer (which
        # must keep counting through their absence -- that's the whole
        # point of speaker_release_seconds being a soft, time-based grace
        # period rather than an instant drop the moment they leave frame)
        # and any id mid-acknowledgment is pruned once it leaves present_ids.
        protected = {self._acknowledging_id, self._active_speaker_id}
        for store in (
            self.speaker_candidate_time, self.active_speaker, self.speaking_duration,
            self.attention_time, self.recent_attention_penalty, self.last_target_time,
            self._quiet_gaze_time,
        ):
            for pid in list(store):
                if pid not in present_ids and pid not in protected:
                    del store[pid]
        for pid in list(self._speaker_silence_time):
            if pid not in present_ids and pid != self._active_speaker_id:
                del self._speaker_silence_time[pid]
