"""Smoke tests for visualization.py — no display needed, just verifies the
drawing calls don't crash and actually touch the frame buffer."""

import numpy as np

from gaze_attention.visualization import _face_color, draw_overlay, GREEN, RED, BLUE, YELLOW


def _obs(id_=0, gaze_signal=0.9, confidence=0.9, seeking=True, bbox=(10, 10, 50, 50), fallback=False):
    return {
        "id": id_, "bbox": list(bbox), "gaze_signal": gaze_signal,
        "confidence": confidence, "seeking_attention": seeking,
        "timestamp": 0.0, "fallback": fallback,
    }


def test_draw_overlay_no_crash_multiple_faces():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    observations = [_obs(0), _obs(1, confidence=0.1, seeking=False), _obs(2, seeking=False, confidence=0.9)]
    metrics = {0: {"engagement": 1.0, "debt": 0.0, "score": 1.0}, 1: {"engagement": 0.0, "debt": 0.0, "score": 0.0}, 2: {"engagement": 0.2, "debt": 0.1, "score": 0.25}}
    out = draw_overlay(
        frame, observations, metrics, selected_target=0,
        policy_name="confidence_debt", filter_name="moving_average", fps=30.0,
        ros_connected=True, fallback_active=False, logging_enabled=True,
        confidence_threshold=0.45,
    )
    assert out.shape == frame.shape
    assert out.sum() > 0  # something was drawn


def test_draw_overlay_no_faces_no_crash():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = draw_overlay(
        frame, [], {}, selected_target=None,
        policy_name="greedy", filter_name="moving_average", fps=0.0,
        ros_connected=False, fallback_active=True, logging_enabled=False,
        confidence_threshold=0.45,
    )
    assert out.sum() > 0  # HUD still drawn


def test_face_color_priority_selected_wins():
    obs = _obs(confidence=0.1, seeking=True)  # would be RED or BLUE otherwise
    assert _face_color(obs, is_selected=True, confidence_threshold=0.45) == GREEN


def test_face_color_low_confidence_before_seeking():
    obs = _obs(confidence=0.1, seeking=True)
    assert _face_color(obs, is_selected=False, confidence_threshold=0.45) == RED


def test_face_color_seeking_not_selected():
    obs = _obs(confidence=0.9, seeking=True)
    assert _face_color(obs, is_selected=False, confidence_threshold=0.45) == BLUE


def test_face_color_default_yellow():
    obs = _obs(confidence=0.9, seeking=False)
    assert _face_color(obs, is_selected=False, confidence_threshold=0.45) == YELLOW
