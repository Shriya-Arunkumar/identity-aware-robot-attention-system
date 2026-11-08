"""
speech.py — visual_speaking_proxy: a mouth-motion-based auxiliary cue for
"is this person plausibly talking right now."

This is NOT audio speech recognition and NOT speaker identification —
there is no microphone in this project. Mouth motion is a coarse VISUAL
proxy computed from MediaPipe lip landmarks (see perception.py's
`_mouth_opening_from_landmarks`). Because no microphone was used, mouth
motion is treated as a visual proxy for speaking activity and is not
equivalent to speaker identification or speech recognition (see README.md).

    mouth_activity, speaking, speech_confidence = proxy.update(person_id, mouth_opening, dt)

- `mouth_activity` is derived from TEMPORAL CHANGE in `mouth_opening` over a
  rolling `speaking_window_seconds` window — the standard deviation of the
  windowed mouth_opening values, normalized by `activity_scale` and clamped
  to [0, 1]. A static open or closed mouth produces near-zero activity; a
  talking mouth fluctuates and produces higher activity.
- `speaking` is a hysteresis-debounced boolean over `mouth_activity`:
  `speaking_on_threshold` (higher) to start, `speaking_off_threshold`
  (lower) to stop — this prevents flicker right at a single threshold,
  the same reason filters.py exists for the gaze signal.
- `speech_confidence` reflects how much of the rolling window is actually
  filled yet — 0 right after a person appears/reappears (no history),
  rising to 1 once `speaking_window_seconds` worth of history has
  accumulated. This is about DATA SUFFICIENCY for the speaking call, a
  completely different quantity from raw_confidence/policy_confidence
  (gaze trustworthiness, see filters.py/perception.py) — never confuse
  the two. It is NOT a measure of speech strength and must never be used
  as evidence that someone is speaking: it saturates at 1.0 after
  `speaking_window_seconds` of tracking and stays there for a silent
  person and a talking one alike. Signal strength lives in
  `mouth_activity`; temporal support lives in `support_fraction()`.

Temporal support (Defect B)
---------------------------
Because `mouth_activity` is a STD over `speaking_window_seconds`, a single
brief spike keeps it elevated for the whole window — a 2-frame (0.067s)
blip produced 0.467s of `speaking=True`, enough to clear the 0.35s
speaker-confirmation window and let an isolated mouth twitch preempt a
genuine continuous speaker. Persistence of the rolling statistic is not
evidence of sustained speech.

So turning `speaking` ON additionally requires TEMPORAL SUPPORT: the
fraction of individual recent frames whose frame-to-frame mouth-opening
delta exceeded `activity_delta_threshold`, measured over
`support_window_seconds`, must reach `minimum_positive_fraction`. An
isolated spike contributes only its rise and fall; real speech keeps
generating them. Turning OFF is deliberately left on the existing
off-threshold, so the established speech-off grace still governs pauses —
this gate rejects false ONSETS, it does not shorten anyone's speech.

Per-person state, created lazily on first observation. Call
`forget(person_id)` when a track is dropped (person left / occluded past
tracker.py's max_missed_frames) so stale mouth history doesn't leak into a
reused id — a disappeared track's mouth history must reset, not persist.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np


class VisualSpeakingProxy:
    def __init__(
        self,
        speaking_on_threshold: float = 0.15,
        speaking_off_threshold: float = 0.08,
        speaking_window_seconds: float = 0.5,
        activity_scale: float = 0.02,
        # -- Defect B: temporal-support gate --------------------------------
        support_window_seconds: float = 0.30,
        minimum_positive_fraction: float = 0.30,
        activity_delta_threshold: float = 0.01,
    ):
        if speaking_off_threshold > speaking_on_threshold:
            raise ValueError("speaking_off_threshold must be <= speaking_on_threshold (hysteresis)")
        self.speaking_on_threshold = speaking_on_threshold
        self.speaking_off_threshold = speaking_off_threshold
        self.speaking_window_seconds = speaking_window_seconds
        self.activity_scale = activity_scale
        self.support_window_seconds = support_window_seconds
        self.minimum_positive_fraction = minimum_positive_fraction
        self.activity_delta_threshold = activity_delta_threshold

        # id -> deque of [mouth_opening, age_seconds_since_added]
        self._openings: Dict[int, Deque[List[float]]] = {}
        self._speaking: Dict[int, bool] = {}
        self._window_time: Dict[int, float] = {}  # accumulated tracked time, capped at the window
        # Defect B: id -> deque of [was_frame_active(0/1), age_seconds], and
        # the previous raw mouth_opening used to compute the per-frame delta.
        self._support: Dict[int, Deque[List[float]]] = {}
        self._prev_opening: Dict[int, float] = {}
        self._last_support_fraction: Dict[int, float] = {}

    def update(self, person_id: int, mouth_opening: float, dt: float) -> Tuple[float, bool, float]:
        dt = max(dt, 0.0)

        buf = self._openings.setdefault(person_id, deque())
        buf.append([mouth_opening, 0.0])
        for entry in buf:
            entry[1] += dt
        while buf and buf[0][1] > self.speaking_window_seconds:
            buf.popleft()

        values = [opening for opening, _age in buf]
        mouth_activity = (
            float(np.clip(np.std(values) / self.activity_scale, 0.0, 1.0)) if len(values) > 1 else 0.0
        )

        # -- Defect B: temporal support ------------------------------------
        # mouth_activity is a STD over speaking_window_seconds, so a single
        # 2-frame spike keeps it elevated for the whole window (measured:
        # 0.067s of motion -> 0.467s of speaking=True, which cleared the
        # 0.35s speaker_confirm_seconds and let an isolated blip preempt a
        # genuine continuous speaker). Persistence of the ROLLING statistic
        # is not evidence of sustained speech.
        #
        # So: independently count how many INDIVIDUAL recent frames actually
        # showed mouth movement (frame-to-frame delta above
        # activity_delta_threshold), over support_window_seconds. An isolated
        # spike contributes only its rise and fall (2 active frames); real
        # speech keeps producing them. The fraction is taken against the
        # EXPECTED number of samples for the window, not len(buf), so a spike
        # arriving before any history cannot score 2/2 = 1.0.
        prev_opening = self._prev_opening.get(person_id)
        frame_active = (
            1.0 if prev_opening is not None
            and abs(mouth_opening - prev_opening) >= self.activity_delta_threshold
            else 0.0
        )
        self._prev_opening[person_id] = mouth_opening

        support = self._support.setdefault(person_id, deque())
        support.append([frame_active, 0.0])
        for entry in support:
            entry[1] += dt
        while support and support[0][1] > self.support_window_seconds:
            support.popleft()

        expected_samples = max(1, int(round(self.support_window_seconds / dt))) if dt > 0 else 1
        active_count = sum(entry[0] for entry in support)
        support_fraction = float(np.clip(active_count / expected_samples, 0.0, 1.0))
        self._last_support_fraction[person_id] = support_fraction
        temporally_supported = support_fraction >= self.minimum_positive_fraction

        was_speaking = self._speaking.get(person_id, False)
        # Turning ON additionally requires temporal support. Turning OFF is
        # left on the existing off-threshold alone, so the established
        # speech-off grace (speech_off_confirm_seconds, speaker_release_seconds)
        # still governs pauses -- this gate only rejects onsets.
        if not was_speaking and mouth_activity >= self.speaking_on_threshold and temporally_supported:
            speaking = True
        elif was_speaking and mouth_activity <= self.speaking_off_threshold:
            speaking = False
        else:
            speaking = was_speaking
        self._speaking[person_id] = speaking

        window_time = min(self._window_time.get(person_id, 0.0) + dt, self.speaking_window_seconds)
        self._window_time[person_id] = window_time
        speech_confidence = (
            float(np.clip(window_time / self.speaking_window_seconds, 0.0, 1.0))
            if self.speaking_window_seconds > 0
            else 1.0
        )

        return mouth_activity, speaking, speech_confidence

    def support_fraction(self, person_id: int) -> float:
        """Diagnostic: the fraction of recent frames that individually showed
        mouth movement (Defect B temporal support). Distinct from
        `speech_confidence`, which is only data availability."""
        return self._last_support_fraction.get(person_id, 0.0)

    def forget(self, person_id: int) -> None:
        self._openings.pop(person_id, None)
        self._speaking.pop(person_id, None)
        self._window_time.pop(person_id, None)
        self._support.pop(person_id, None)
        self._prev_opening.pop(person_id, None)
        self._last_support_fraction.pop(person_id, None)

    def reset(self) -> None:
        self._openings.clear()
        self._speaking.clear()
        self._window_time.clear()
        self._support.clear()
        self._prev_opening.clear()
        self._last_support_fraction.clear()
