"""
logger.py — one CSV row per person per processed frame.

Auto-creates a timestamped log file per session under `logs/` (repo-relative,
never the current working directory) so re-running the demo never overwrites
a previous session's log. `CSV_FIELDS` is the single source of truth for the
schema — evaluation/replay.py imports it from here rather than redefining it,
so a live-recorded session and a synthetic replay always share one column
layout.

Confidence fields (see DEBUG_ATTENTION.md FM1 for why these are separate,
not one column):
  * raw_confidence      — straight from perception.py, the wrapped gaze
                           model's own per-frame confidence heuristic.
  * filtered_signal      — the smoothed gaze/engagement value (post-filter).
  * filter_uncertainty   — the filter's own uncertainty about its smoothed
                            estimate (see filters.py's docstring) — a
                            filter-diagnostic value, NOT a confidence.
  * policy_confidence    — what the policy actually gates
                            `confidence_threshold` on; see
                            main_webcam.py/evaluation/replay.py for the
                            current "Version 1" definition (= raw_confidence,
                            undocumented transform not permitted).

visual_speaking_proxy fields (see gaze_attention/speech.py — a mouth-motion
VISUAL cue, NOT audio speech recognition; there is no microphone):
  * mouth_activity   — normalized temporal-change-in-mouth-opening measure.
  * speaking          — hysteresis-debounced boolean over mouth_activity.
  * speech_confidence — how much of the rolling window is filled (data
                         sufficiency for the speaking call, NOT gaze
                         confidence — do not confuse with raw_confidence).
  * speech_bonus       — the actual score contribution this tick
                          (speech_weight * speech_confidence when speaking
                          AND looking at the robot, else 0.0; always 0.0
                          when speech_mode == "off").

Gazebo output fields (see main_webcam.py's RosPublisher and
gaze_attention/target_angle.py). IMPORTANT: this process publishes the RAW
(unsmoothed) angle -- smoothing is intentionally a single downstream stage
in eye_gaze_node.py (a slew-rate limiter), not duplicated here. See
CLAUDE.md/README.md's angle-path architecture section for why (a prior
version had a second, redundant exponential-smoothing stage in this
process, which only added latency -- removed):
  * target_angle    — the current tick's RAW angle, derived from the
                       selected target's CURRENT face position
                       (face_center_x -> normalized_x -> raw angle,
                       clamped but NOT smoothed), or the fixed id->angle
                       fallback mapping when face coordinates aren't usable
                       this tick. Recomputed every frame regardless of
                       whether it was actually published this tick (see
                       `published_angle` below). Independent of
                       `selected_target`'s identity-driven policy state
                       (engagement/debt/etc.) -- this column can change
                       every frame even while `selected_target` stays the
                       same, if that person is physically moving. Identical
                       to `raw_target_angle` (see below) -- kept as two
                       columns for schema stability (`target_angle` existed
                       before the smoothing removal) and clarity at the
                       call site.
  * face_center_x, frame_width, normalized_x, raw_target_angle --
    diagnostics for target_angle above. face_center_x/normalized_x are
    blank when the fallback id->angle mapping was used instead of a real
    face position (no bbox available this tick).
  * published_angle — the RAW angle value actually sent out on
                       /attention_target_angle most recently.
                       RosPublisher only publishes when the target id
                       changes or at most every `min_publish_interval_seconds`
                       (a low fixed rate) -- so this can legitimately lag
                       one or more frames behind `target_angle` in between
                       publishes; that's the throttling working as intended,
                       not a bug. All smoothing happens downstream, in
                       eye_gaze_node.py's slew-rate limiter (the only
                       smoothing stage in the whole angle path) -- it's
                       what keeps the physical eye motion continuous
                       despite the lower publish rate and the raw,
                       unsmoothed input.

Fast speaker-first path diagnostics (ConversationalSimplePolicy only --
other policies leave these blank; see policies.py's docstring for the full
"why" behind the fast-path timing fix):
  * speaking_true_duration/speaking_false_duration/quiet_penalty/
    is_active_speaker (per person, from policy.get_metrics()).
  * active_speaker_id/speaker_confirmed_time/speaker_release_time/
    target_switch_reason/switch_latency_seconds (frame-level -- same value
    repeated on every person-row this frame, same convention as
    `selected_target`). `speaker_confirmed_time`/`speaker_release_time` are
    the CURRENT active speaker's own timestamps (blank if there is none).
    `target_switch_reason`/`switch_latency_seconds` are only meaningful on
    the frame where `selected_target` actually changed -- see
    ConversationalSimplePolicy.last_switch_reason's docstring for the full
    set of reason strings.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional

CSV_FIELDS = [
    "timestamp", "frame_index", "person_id", "x1", "y1", "x2", "y2",
    "raw_confidence", "filtered_signal", "filter_uncertainty", "policy_confidence",
    "mouth_activity", "speaking", "speech_confidence",
    "seeking_attention",
    "engagement", "debt", "score", "speech_bonus", "selected_target",
    "target_angle", "published_angle",
    "face_center_x", "frame_width", "normalized_x", "raw_target_angle",
    "speaking_true_duration", "speaking_false_duration", "quiet_penalty", "is_active_speaker",
    "active_speaker_id", "speaker_confirmed_time", "speaker_release_time",
    "target_switch_reason", "switch_latency_seconds",
    "policy_name", "filter_name",
]

_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def new_log_path(logs_dir: Optional[Path] = None) -> Path:
    logs_dir = logs_dir or _LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"session_{stamp}.csv"
    # Extremely unlikely (same-second re-run), but never overwrite.
    suffix = 1
    while path.exists():
        path = logs_dir / f"session_{stamp}_{suffix}.csv"
        suffix += 1
    return path


class SessionLogger:
    """Buffered per-frame CSV writer. `enabled` can be toggled at runtime
    (the 'l' keyboard control) without losing the open file — toggling off
    just stops writes, toggling back on resumes into the same file."""

    def __init__(self, logs_dir: Optional[Path] = None, enabled: bool = True):
        self.path = new_log_path(logs_dir)
        self.enabled = enabled
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def log_frame(
        self,
        frame_index: int,
        observations: List[dict],
        metrics_by_id: dict,
        selected_target: Optional[int],
        policy_name: str,
        filter_name: str,
        target_angle: Optional[float] = None,
        published_angle: Optional[float] = None,
        face_center_x: Optional[float] = None,
        frame_width: Optional[float] = None,
        normalized_x: Optional[float] = None,
        raw_target_angle: Optional[float] = None,
        active_speaker_id: Optional[int] = None,
        speaker_confirmed_time: Optional[float] = None,
        speaker_release_time: Optional[float] = None,
        target_switch_reason: Optional[str] = None,
        switch_latency_seconds: Optional[float] = None,
    ) -> None:
        """One row per observation this frame. `metrics_by_id` maps person id
        -> {"engagement":..., "debt":..., "score":...} (policy.get_metrics).
        Each `obs` is expected to carry all four confidence-related fields
        (`raw_confidence`, `gaze_signal` [=filtered_signal], `filter_uncertainty`,
        `confidence` [=policy_confidence, the pinned key policies.py reads]) —
        see main_webcam.py/evaluation/replay.py for where they're assembled.
        Missing optional fields (e.g. a caller not yet passing
        `filter_uncertainty`) log as an empty cell rather than raising."""
        if not self.enabled:
            return
        for obs in observations:
            metrics = metrics_by_id.get(obs["id"], {"engagement": 0.0, "debt": 0.0, "score": 0.0, "speech_bonus": 0.0})
            bbox = obs["bbox"]
            self._writer.writerow({
                "timestamp": obs["timestamp"],
                "frame_index": frame_index,
                "person_id": obs["id"],
                "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
                "raw_confidence": obs.get("raw_confidence", ""),
                "filtered_signal": obs["gaze_signal"],
                "filter_uncertainty": obs.get("filter_uncertainty", ""),
                "policy_confidence": obs.get("policy_confidence", obs["confidence"]),
                "mouth_activity": obs.get("mouth_activity", ""),
                "speaking": obs.get("speaking", ""),
                "speech_confidence": obs.get("speech_confidence", ""),
                "seeking_attention": obs["seeking_attention"],
                "engagement": metrics["engagement"],
                "debt": metrics["debt"],
                "score": metrics["score"],
                "speech_bonus": metrics.get("speech_bonus", 0.0),
                "selected_target": selected_target,
                "target_angle": target_angle if target_angle is not None else "",
                "published_angle": published_angle if published_angle is not None else "",
                "face_center_x": face_center_x if face_center_x is not None else "",
                "frame_width": frame_width if frame_width is not None else "",
                "normalized_x": normalized_x if normalized_x is not None else "",
                "raw_target_angle": raw_target_angle if raw_target_angle is not None else "",
                "speaking_true_duration": metrics.get("speaking_true_duration", ""),
                "speaking_false_duration": metrics.get("speaking_false_duration", ""),
                "quiet_penalty": metrics.get("quiet_penalty", ""),
                "is_active_speaker": metrics.get("is_active_speaker", ""),
                "active_speaker_id": active_speaker_id if active_speaker_id is not None else "",
                "speaker_confirmed_time": speaker_confirmed_time if speaker_confirmed_time is not None else "",
                "speaker_release_time": speaker_release_time if speaker_release_time is not None else "",
                "target_switch_reason": target_switch_reason or "",
                "switch_latency_seconds": switch_latency_seconds if switch_latency_seconds is not None else "",
                "policy_name": policy_name,
                "filter_name": filter_name,
            })
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "SessionLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
