"""
dwell_arbitration.py — the `DecisionPolicy` interface, the `Observation`
schema, and `_DwellArbitration` (shared minimum-hold-time + deterministic
tie-break + absence-grace-period mechanics), factored out of policies.py so
both policies.py (ConfidenceDebtPolicy, ConversationalSimplePolicy) and
baseline_policies.py (GreedyDwellPolicy) can depend on it without either
module importing the other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TypedDict


class Observation(TypedDict):
    id: int
    bbox: List[float]
    gaze_signal: float
    confidence: float
    seeking_attention: bool
    timestamp: float
    mouth_activity: float
    speaking: bool
    speech_confidence: float


class DecisionPolicy(ABC):
    """Interface every target-selection policy implements."""

    @abstractmethod
    def update(self, observations: List[Observation], dt: float) -> Optional[int]:
        """Advance by one tick given this frame's (filtered) observations and
        return the selected person's id, or None if no one is present."""

    @abstractmethod
    def reset(self) -> None:
        """Return the policy to its initial (no target, no scores) state."""

    @property
    @abstractmethod
    def current_target(self) -> Optional[int]:
        """The currently selected id without advancing the policy."""


class _DwellArbitration(DecisionPolicy):
    """Shared target-selection mechanics: minimum hold time + deterministic
    tie-break + absence grace period. Subclasses only implement
    `_scores(observations, dt)`, which must return {person_id: score} for
    every id present this tick and update any internal per-person
    accumulators (engagement, debt, ...) as a side effect. Ids no longer
    present in `observations` have their internal state pruned after each
    update — a dropped track (person left / occluded past
    max_missed_frames, per tracker.py) starts over from zero if they
    reappear, they don't keep old credit or debt — EXCEPT the current
    target during its own absence grace window (see `target_grace_seconds`
    below), whose state is deliberately protected from pruning so a
    transient miss doesn't silently zero their score out from under them.

    Absence grace period (`target_grace_seconds`)
    -----------------------------------------------
    A single frame where the current target isn't detected (a blink, a fast
    head turn, motion blur — all common with a real webcam) must not by
    itself clear the target or bypass `minimum_hold_seconds`: perception.py
    only emits an observation for faces MediaPipe actually detected *this*
    frame, so a transient miss makes that person vanish from `observations`
    for that tick even though tracker.py's own `max_missed_frames` grace
    period would still recognize them as the same track a moment later.

    So: while the current target is absent from `observations`, this class
    tracks how long they've been continuously absent
    (`_target_absent_time`) and only releases them once BOTH:
      1. `_time_since_switch >= minimum_hold_seconds` (the normal hold-time
         guarantee), AND
      2. `_target_absent_time >= target_grace_seconds` (they've been gone
         longer than a plausible detection blip — `target_grace_seconds`
         approximates tracker.py's `max_missed_frames` grace window in time
         rather than frame count, since policies operate on dt/time only).
    Until both are true, the current target simply stays selected (frozen,
    not reselected), and a transient empty `present_ids` never by itself
    triggers an immediate reselection.
    """

    def __init__(
        self,
        minimum_hold_seconds: float = 0.75,
        target_grace_seconds: float = 0.5,
        speech_mode: str = "off",
        speech_weight: float = 0.0,
    ):
        self.minimum_hold_seconds = minimum_hold_seconds
        self.target_grace_seconds = target_grace_seconds
        self.speech_mode = speech_mode
        self.speech_weight = speech_weight
        self._current_target: Optional[int] = None
        self._time_since_switch: float = 0.0
        self._target_absent_time: float = 0.0
        self._last_speech_bonus: Dict[int, float] = {}

    @property
    def current_target(self) -> Optional[int]:
        return self._current_target

    @abstractmethod
    def _scores(self, observations: List[Observation], dt: float) -> Dict[int, float]:
        """Update internal per-person state and return this tick's scores,
        keyed by id, for every id present in `observations`."""

    @abstractmethod
    def _prune(self, present_ids: set, protect_id: Optional[int] = None) -> None:
        """Drop internal state for ids no longer present, except
        `protect_id` (if given) even though it's not in `present_ids`."""

    def _speech_bonus(self, obs: Observation) -> float:
        """visual_speaking_proxy support (see gaze_attention/speech.py):
        speaking + looking_at_robot (this project's seeking_attention proxy)
        -> speech_weight * speech_confidence. Speaking alone, without
        looking at the robot, is worth nothing -- it must never be able to
        cause a switch by itself. Always 0.0 when speech_mode == "off"."""
        if self.speech_mode == "off":
            return 0.0
        speaking = obs.get("speaking", False)
        speech_confidence = obs.get("speech_confidence", 0.0)
        looking_at_robot = obs.get("seeking_attention", False)
        if speaking and looking_at_robot:
            return self.speech_weight * speech_confidence
        return 0.0

    def update(self, observations: List[Observation], dt: float) -> Optional[int]:
        present_ids = {obs["id"] for obs in observations}
        target_present = self._current_target is not None and self._current_target in present_ids

        # Protect the current target's internal per-person state (engagement/
        # debt/score) from being pruned purely because of a transient miss —
        # otherwise "the target stays selected" would be cosmetic: their
        # score would silently reset to zero the moment they're next scored.
        protect_id = None
        if self._current_target is not None and not target_present and self._target_absent_time < self.target_grace_seconds:
            protect_id = self._current_target

        scores = self._scores(observations, dt)

        # Speech bonus is supporting evidence only: folded into `scores`
        # BEFORE the hold-time/tie-break gate below runs, so it can nudge
        # which candidate looks strongest but can never bypass that gate.
        for obs in observations:
            pid = obs["id"]
            bonus = self._speech_bonus(obs)
            self._last_speech_bonus[pid] = bonus
            scores[pid] = scores[pid] + bonus
        for pid in list(self._last_speech_bonus):
            if pid not in present_ids and pid != protect_id:
                del self._last_speech_bonus[pid]

        self._prune(present_ids, protect_id=protect_id)

        if self._current_target is not None:
            self._target_absent_time = 0.0 if target_present else self._target_absent_time + dt

        # Challenger-eligibility guard: a lost=True observation (perception.py's
        # frozen stand-in for a currently-occluded track, see tracker.py's
        # LOST state) can never become the target and can never challenge for
        # it -- it carries no new evidence, just a frozen last-known value.
        # This does NOT touch target_present/protect_id/pruning above, and
        # does NOT extend target_grace_seconds -- that remains the only way
        # a lost person retains a target they already hold.
        eligible_ids = {obs["id"] for obs in observations if not obs.get("lost", False)}

        # Deterministic tie-break: highest score wins; ties broken by lowest
        # person id (stable regardless of dict/list ordering).
        candidate = min(eligible_ids, key=lambda i: (-scores[i], i)) if eligible_ids else None

        if self._current_target is None:
            # No target held yet — nothing to protect, pick immediately.
            if candidate is not None:
                self._current_target = candidate
                self._time_since_switch = 0.0
                self._target_absent_time = 0.0
            return self._current_target

        if target_present:
            if (
                candidate is not None
                and candidate != self._current_target
                and self._time_since_switch >= self.minimum_hold_seconds
                and scores[candidate] > scores[self._current_target]
            ):
                self._current_target = candidate
                self._time_since_switch = 0.0
                self._target_absent_time = 0.0
            else:
                self._time_since_switch += dt
            return self._current_target

        # Current target is absent this tick: only release them once BOTH
        # the minimum hold time and the absence grace period have elapsed.
        grace_expired = self._target_absent_time >= self.target_grace_seconds
        hold_expired = self._time_since_switch >= self.minimum_hold_seconds
        if grace_expired and hold_expired:
            self._current_target = candidate  # candidate may be None -> clears
            self._time_since_switch = 0.0
            self._target_absent_time = 0.0
        else:
            self._time_since_switch += dt
        return self._current_target

    def reset(self) -> None:
        self._current_target = None
        self._time_since_switch = 0.0
        self._target_absent_time = 0.0
        self._last_speech_bonus.clear()
