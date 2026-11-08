"""
visualization.py — live OpenCV overlay. Pure drawing functions over a frame
already in hand; no capture/inference logic here (that's main_webcam.py).

Color priority per face (spec lists these as separate categories; ties are
broken in this order, most important first):
  1. green  — this is the currently selected target
  2. red    — confidence below `confidence_threshold` (untrustworthy)
  3. blue   — actively seeking attention but not currently selected
  4. yellow — detected, none of the above (idle/holding steady)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2

GREEN = (0, 200, 0)
YELLOW = (0, 220, 220)
RED = (0, 0, 220)
BLUE = (220, 130, 0)
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
BLACK = (0, 0, 0)


def _face_color(obs: dict, is_selected: bool, confidence_threshold: float) -> tuple:
    if is_selected:
        return GREEN
    if obs["confidence"] < confidence_threshold:
        return RED
    if obs["seeking_attention"]:
        return BLUE
    return YELLOW


def draw_faces(
    frame,
    observations: List[dict],
    metrics_by_id: Dict[int, dict],
    selected_target: Optional[int],
    confidence_threshold: float,
):
    for obs in observations:
        pid = obs["id"]
        is_selected = pid == selected_target
        color = _face_color(obs, is_selected, confidence_threshold)
        x1, y1, x2, y2 = (int(v) for v in obs["bbox"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        metrics = metrics_by_id.get(pid, {"engagement": 0.0, "debt": 0.0, "score": 0.0, "speech_bonus": 0.0})
        seeking_label = "SEEKING" if obs["seeking_attention"] else "idle"
        status = "SELECTED" if is_selected else seeking_label
        lines = [
            f"id {pid}  {status}",
            f"conf={obs['confidence']:.2f} sig={obs['gaze_signal']:.2f}",
            f"eng={metrics['engagement']:.2f} debt={metrics['debt']:.2f} score={metrics['score']:.2f}",
        ]
        # visual_speaking_proxy label (mouth-motion cue, NOT audio speech
        # recognition -- no microphone). Shown whenever mouth data is
        # present, regardless of speech_mode, so it's visible for demo
        # purposes even when policies aren't acting on it.
        if "mouth_activity" in obs:
            speak_label = "SPEAKING" if obs.get("speaking") else "quiet"
            lines.append(f"mouth(visual proxy): {speak_label} act={obs['mouth_activity']:.2f}")
        speech_bonus = metrics.get("speech_bonus", 0.0)
        if speech_bonus > 0:
            lines.append(f"speech_bonus=+{speech_bonus:.2f}")
        if obs.get("fallback"):
            lines.append("FALLBACK PERCEPTION")

        ty = max(15, y1 - 8 - 15 * (len(lines) - 1))
        for line in lines:
            cv2.putText(frame, line, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            ty += 15

    return frame


def draw_hud(
    frame,
    policy_name: str,
    filter_name: str,
    fps: float,
    selected_target: Optional[int],
    ros_connected: bool,
    fallback_active: bool,
    logging_enabled: bool,
    speech_mode: str = "off",
):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 26), BLACK, -1)

    target_text = str(selected_target) if selected_target is not None else "none"
    ros_text = "ROS: connected" if ros_connected else "ROS: OFFLINE (viz-only)"
    ros_color = GREEN if ros_connected else RED
    log_text = "LOG: on" if logging_enabled else "LOG: off"
    speech_text = f"speech={speech_mode}" if speech_mode != "off" else ""

    line = f"policy={policy_name}  filter={filter_name}  target={target_text}  fps={fps:.1f}  {log_text}  {speech_text}"
    cv2.putText(frame, line, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    ros_size = cv2.getTextSize(ros_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(frame, ros_text, (w - ros_size[0] - 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ros_color, 1, cv2.LINE_AA)

    if fallback_active:
        cv2.rectangle(frame, (0, 26), (w, 48), (0, 0, 160), -1)
        cv2.putText(
            frame, "FALLBACK PERCEPTION ACTIVE - not calibrated gaze ground truth",
            (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA,
        )

    return frame


def draw_overlay(
    frame,
    observations: List[dict],
    metrics_by_id: Dict[int, dict],
    selected_target: Optional[int],
    policy_name: str,
    filter_name: str,
    fps: float,
    ros_connected: bool,
    fallback_active: bool,
    logging_enabled: bool,
    confidence_threshold: float,
    speech_mode: str = "off",
):
    """Draw the full overlay (per-face boxes/labels + HUD) in place and
    return the same frame, for chaining into cv2.imshow."""
    draw_faces(frame, observations, metrics_by_id, selected_target, confidence_threshold)
    draw_hud(frame, policy_name, filter_name, fps, selected_target, ros_connected, fallback_active, logging_enabled, speech_mode)
    return frame
