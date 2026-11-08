"""
target_angle.py — computes the RAW (unsmoothed) Gazebo-facing gaze angle
from the SELECTED person's current face position, decoupled from the
attention policy.

This module intentionally does NOT smooth its output. Smoothing lives in
exactly one place in the whole angle path: eye_gaze_node.py's slew-rate
limiter, downstream of /attention_target_angle. An earlier version of this
module had its own `ExponentialAngleSmoother` running here too, in series
with eye_gaze_node.py's limiter -- two independent filters stacked, which
only added latency (roughly the exponential filter's ~3tau settling time
on top of the limiter's own ramp) without improving behavior, and meant a
manually-published `/attention_target_angle` command (bypassing this
process entirely) was smoothed differently than a webcam-driven one. See
CLAUDE.md/README.md's angle-path architecture section. Do not reintroduce
a smoothing/rate-limiting class here.

Two related but distinct pieces of the ROS/Gazebo interface:
  * WHO is selected (`/attention_target_id`, Int32) — the policy's choice,
    driving policy state (engagement/debt/speaking/target persistence/
    attention history). Untouched by this module.
  * WHERE that selected person's face currently is
    (`/attention_target_angle`, Float32) — a per-frame geometric
    computation from that person's current bbox, independent of how long
    they've held the target or any policy score. If the selected person
    physically moves (without the target id changing), this angle tracks
    that movement; if the target id changes to a different person, this
    angle tracks THEIR position instead. Neither direction ever touches
    engagement/debt/speaking/attention-history state.

    face_center_x = (x1 + x2) / 2
    normalized_x  = face_center_x / frame_width
    raw_angle     = max_angle * (2 * normalized_x - 1)   # -max_angle..+max_angle
    (clamped to [-max_angle, +max_angle] regardless, in case of a
    degenerate bbox/frame_width)

Fallback mapping (requirement 11): when face coordinates aren't usable
(missing bbox, non-positive frame_width, or a bbox that collapses to zero
width) this falls back to a fixed id -> angle mapping (`FALLBACK_ANGLES`)
so the interface still produces *something* sensible rather than raising or
silently freezing -- consistent with this project's "every component needs
a simple fallback" rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

MAX_ANGLE = 0.65  # radians

# ID 0 = left, ID 1 = center, ID 2 = right -- requirement 11.
FALLBACK_ANGLES: Dict[int, float] = {0: -MAX_ANGLE, 1: 0.0, 2: MAX_ANGLE}


def face_center_angle(
    bbox: Optional[Sequence[float]],
    frame_width: Optional[float],
    max_angle: float = MAX_ANGLE,
) -> Optional[float]:
    """Requirement 2-4: face_center_x -> normalized_x -> target_angle,
    clamped to [-max_angle, +max_angle]. Returns None if face coordinates
    aren't usable (missing bbox, non-positive frame_width, or a
    degenerate/zero-width bbox) -- the caller falls back to
    `FALLBACK_ANGLES` in that case (requirement 11)."""
    return _compute(bbox, frame_width, max_angle).raw_angle


def target_angle_for(
    target_id: Optional[int],
    bbox: Optional[Sequence[float]],
    frame_width: Optional[float],
    max_angle: float = MAX_ANGLE,
    fallback_angles: Dict[int, float] = FALLBACK_ANGLES,
    warned: Optional[set] = None,
) -> Optional[float]:
    """The current tick's raw (pre-smoothing) target angle: face-position-based
    when available, else the fixed id -> angle fallback mapping. Returns
    None only when target_id itself is None (no one selected)."""
    return compute_target_angle(target_id, bbox, frame_width, max_angle, fallback_angles, warned).raw_angle


@dataclass
class TargetAngleResult:
    """Requirement 10 diagnostics, all in one place so logging never has to
    recompute (and risk diverging from) the actual decision."""
    raw_angle: Optional[float]
    face_center_x: Optional[float]
    normalized_x: Optional[float]
    used_fallback: bool


def compute_target_angle(
    target_id: Optional[int],
    bbox: Optional[Sequence[float]],
    frame_width: Optional[float],
    max_angle: float = MAX_ANGLE,
    fallback_angles: Dict[int, float] = FALLBACK_ANGLES,
    warned: Optional[set] = None,
) -> TargetAngleResult:
    if target_id is None:
        return TargetAngleResult(None, None, None, used_fallback=False)

    result = _compute(bbox, frame_width, max_angle)
    if result.raw_angle is not None:
        return result

    # Requirement 11: fallback mapping when face coordinates are unavailable.
    if target_id in fallback_angles:
        return TargetAngleResult(fallback_angles[target_id], None, None, used_fallback=True)
    if warned is not None and target_id not in warned:
        print(f"[target_angle] id={target_id} has no fallback angle and no usable face position — defaulting to 0.0 (center).")
        warned.add(target_id)
    return TargetAngleResult(0.0, None, None, used_fallback=True)


def compute_pitch_angle(
    target_id: Optional[int],
    bbox: Optional[Sequence[float]],
    frame_height: Optional[float],
    max_angle: float = 0.5,
    warned: Optional[set] = None,
) -> TargetAngleResult:
    """Compute vertical gaze angle from the selected person's face position.

    Top of image    -> positive pitch
    Center          -> 0
    Bottom of image -> negative pitch

    The result is clamped to [-max_angle, +max_angle].
    """

    if target_id is None:
        return TargetAngleResult(None, None, None, used_fallback=False)

    if (
        bbox is None
        or frame_height is None
        or frame_height <= 0
        or len(bbox) < 4
    ):
        return TargetAngleResult(0.0, None, None, used_fallback=True)

    y1, y2 = bbox[1], bbox[3]

    if y2 <= y1:
        return TargetAngleResult(0.0, None, None, used_fallback=True)

    face_center_y = (y1 + y2) / 2.0
    normalized_y = face_center_y / frame_height

    # Image Y increases downward, so invert it:
    #
    # top    -> positive pitch
    # center -> zero
    # bottom -> negative pitch
    raw_angle = max_angle * (1.0 - 2.0 * normalized_y)

    clamped = max(
        -max_angle,
        min(max_angle, raw_angle),
    )

    return TargetAngleResult(
        clamped,
        face_center_y,
        normalized_y,
        used_fallback=False,
    )


def _compute(bbox: Optional[Sequence[float]], frame_width: Optional[float], max_angle: float) -> TargetAngleResult:
    if bbox is None or frame_width is None or frame_width <= 0 or len(bbox) < 4:
        return TargetAngleResult(None, None, None, used_fallback=False)
    x1, x2 = bbox[0], bbox[2]
    if x2 <= x1:
        return TargetAngleResult(None, None, None, used_fallback=False)  # degenerate/zero-width bbox

    face_center_x = (x1 + x2) / 2.0
    normalized_x = face_center_x / frame_width
    raw_angle = max_angle * (2.0 * normalized_x - 1.0)
    clamped = max(-max_angle, min(max_angle, raw_angle))
    return TargetAngleResult(clamped, face_center_x, normalized_x, used_fallback=False)

