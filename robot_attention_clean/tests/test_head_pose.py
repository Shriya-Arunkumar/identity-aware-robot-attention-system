"""Change A (mutual-gaze residual) tests -- deterministic synthetic geometry
only, no live camera. Covers the bearing formula, the residual, the Gaussian
q, sigma sensitivity, solvePnP failure fallback, and -- most importantly --
the empirical SIGN verification: a face off-centre that rotates to look at
the camera must yield e ~= 0 and q ~= 1.

The head-pose sign check works by PROJECTING the canonical 3D model at a
known yaw + world offset through the same approximate camera matrix the
production code uses, then feeding the projected 2D points back through the
real _head_pose_from_landmarks -> _bearing_phi -> residual path. This
validates the residual math and the yaw-extraction sign end to end. It does
NOT validate the anatomical FaceMesh index assignment (left vs right eye
corner) -- that is documented in perception.py as requiring a live-face
runtime diagnostic, since a synthetic round-trip through the model points is
self-consistent regardless of that assignment.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from gaze_attention.perception import (
    _HEAD_POSE_LANDMARK_INDICES,
    _HEAD_POSE_MODEL_POINTS,
    _approx_camera_matrix,
    _bearing_phi,
    _head_pose_from_landmarks,
    _mutual_gaze_q,
)

W, H = 640, 480
HFOV = 60.0


# --- 1. bearing at image centre ------------------------------------------------

def test_bearing_at_center_is_zero():
    assert _bearing_phi(W / 2.0, W, HFOV) == pytest.approx(0.0, abs=1e-12)


# --- 2. bearing sign (right of centre > 0, left < 0) --------------------------

def test_bearing_sign_convention():
    # u_x > 0.5 (right half of the image) -> phi > 0; u_x < 0.5 -> phi < 0.
    assert _bearing_phi(W * 0.9, W, HFOV) > 0.0
    assert _bearing_phi(W * 0.1, W, HFOV) < 0.0


def test_bearing_magnitude_matches_hfov_edges():
    # At the extreme edges (u_x = 0 or 1), |phi| == HFOV/2.
    assert _bearing_phi(0.0, W, HFOV) == pytest.approx(-math.radians(HFOV / 2))
    assert _bearing_phi(float(W), W, HFOV) == pytest.approx(math.radians(HFOV / 2))


# --- 3./6. q: residual -> [0,1], sigma sensitivity ----------------------------

def test_q_is_one_at_zero_residual():
    assert _mutual_gaze_q(0.0, 12.0) == pytest.approx(1.0)


def test_q_decreases_with_larger_residual():
    q_small = _mutual_gaze_q(math.radians(5), 12.0)
    q_large = _mutual_gaze_q(math.radians(25), 12.0)
    assert q_small > q_large
    assert 0.0 < q_large < q_small <= 1.0


def test_larger_sigma_is_more_tolerant_of_the_same_residual():
    e = math.radians(15)
    assert _mutual_gaze_q(e, 20.0) > _mutual_gaze_q(e, 8.0)


# --- synthetic projection helper for the sign / end-to-end tests --------------

def _project_face(yaw_deg: float, x_mm: float, z_mm: float = 1500.0):
    """Project the canonical model at a given yaw (about vertical) and world
    x-offset, and return a synthetic 478-length landmark list whose head-pose
    indices sit at the projected image points (normalized). Other landmarks
    are placed at the face centre -- unused by head pose."""
    import cv2

    cam = _approx_camera_matrix(W, H, HFOV)
    yaw = math.radians(yaw_deg)
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                   [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(Ry)
    tvec = np.array([x_mm, 0.0, z_mm], dtype=np.float64)
    img_pts, _ = cv2.projectPoints(_HEAD_POSE_MODEL_POINTS, rvec, tvec, cam, np.zeros(4))
    img_pts = img_pts.reshape(-1, 2)

    n = max(_HEAD_POSE_LANDMARK_INDICES) + 1
    mean_xy = img_pts.mean(axis=0)
    landmarks = [SimpleNamespace(x=mean_xy[0] / W, y=mean_xy[1] / H, z=0.0) for _ in range(n)]
    for slot, idx in enumerate(_HEAD_POSE_LANDMARK_INDICES):
        landmarks[idx] = SimpleNamespace(x=img_pts[slot, 0] / W, y=img_pts[slot, 1] / H, z=0.0)
    # face_center_x from the projected points' horizontal span midpoint.
    face_center_x = float((img_pts[:, 0].min() + img_pts[:, 0].max()) / 2.0)
    return landmarks, face_center_x



def _look_at_camera_yaw_deg(x_mm: float, z_mm: float = 1500.0) -> float:
    """Yaw (degrees) that makes a head at world (x_mm, 0, z_mm) actually face
    the camera at the origin.

    DERIVED from the model geometry, not fitted. The face-forward axis is
    `nose - centroid(other model points)`, which for _HEAD_POSE_MODEL_POINTS
    points along +z_model; in camera coordinates the face therefore looks
    along -forward. Solving for the yaw that aligns that with the
    head->camera direction gives +atan2(x_mm, z_mm):

        head at x=+700 -> +25.0 deg      head at x=-700 -> -25.0 deg

    HISTORY (important): this used to be `-atan2(...)` with the comment
    "sign established empirically". That was WRONG, and it silently cancelled
    an equally-inverted yaw extraction in _head_pose_from_landmarks -- two
    errors hiding each other, which is exactly why this synthetic round trip
    could not catch the bug. Real-camera evidence
    (logs/session_20260820_140240.jsonl) exposed it: corr(yaw, phi) = +0.745
    where it must be ~ -1, mean |e| = 41.4 deg, mean q = 0.083. Both were
    corrected together; see perception._head_pose_from_landmarks and
    docs/LIVE_CAMERA_VALIDATION.md.
    """
    return math.degrees(math.atan2(x_mm, z_mm))


def _residual_for(yaw_deg, x_mm, z_mm=1500.0):
    cam = _approx_camera_matrix(W, H, HFOV)
    landmarks, face_center_x = _project_face(yaw_deg, x_mm, z_mm)
    pose = _head_pose_from_landmarks(landmarks, W, H, cam)
    assert pose is not None
    yaw, pitch = pose
    phi = _bearing_phi(face_center_x, W, HFOV)
    return yaw + phi, yaw, phi


# --- 4. direct-looking edge sanity: e ~= 0, q ~= 1 ----------------------------

def test_direct_looking_from_right_edge_gives_near_zero_residual():
    # Physically to the right of the optical axis, rotated to face the camera.
    x_mm = 700.0
    z_mm = 1500.0
    look_at_yaw_deg = _look_at_camera_yaw_deg(x_mm, z_mm)
    e, yaw, phi = _residual_for(look_at_yaw_deg, x_mm, z_mm)
    assert phi > 0.0, "right of centre must give phi > 0"
    assert abs(math.degrees(e)) < 5.0, f"looking-at-camera residual should be ~0, got {math.degrees(e):.1f}deg"
    assert _mutual_gaze_q(e, 12.0) > 0.9


def test_direct_looking_from_left_edge_gives_near_zero_residual():
    x_mm = -700.0
    z_mm = 1500.0
    look_at_yaw_deg = _look_at_camera_yaw_deg(x_mm, z_mm)
    e, yaw, phi = _residual_for(look_at_yaw_deg, x_mm, z_mm)
    assert phi < 0.0, "left of centre must give phi < 0"
    assert abs(math.degrees(e)) < 5.0, f"looking-at-camera residual should be ~0, got {math.degrees(e):.1f}deg"
    assert _mutual_gaze_q(e, 12.0) > 0.9


def test_centered_forward_gives_zero_residual():
    e, yaw, phi = _residual_for(0.0, 0.0)
    assert abs(math.degrees(e)) < 2.0
    assert abs(math.degrees(phi)) < 2.0


# --- 5. nonzero residual lowers q ---------------------------------------------

def test_off_axis_looking_away_gives_large_residual_low_q():
    # Off to the right but looking straight forward (NOT at the camera).
    e, yaw, phi = _residual_for(0.0, 700.0)
    assert abs(math.degrees(e)) > 15.0, "looking away off-axis must give a large residual"
    q = _mutual_gaze_q(e, 12.0)
    assert q < _mutual_gaze_q(0.0, 12.0)


# --- 7. invalid solvePnP / landmark input falls back safely -------------------

def test_too_few_landmarks_returns_none():
    cam = _approx_camera_matrix(W, H, HFOV)
    short = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(3)]
    assert _head_pose_from_landmarks(short, W, H, cam) is None


def test_none_landmarks_returns_none():
    cam = _approx_camera_matrix(W, H, HFOV)
    assert _head_pose_from_landmarks(None, W, H, cam) is None


def test_nonfinite_landmarks_returns_none():
    cam = _approx_camera_matrix(W, H, HFOV)
    n = max(_HEAD_POSE_LANDMARK_INDICES) + 1
    landmarks = [SimpleNamespace(x=float("nan"), y=0.5, z=0.0) for _ in range(n)]
    assert _head_pose_from_landmarks(landmarks, W, H, cam) is None


# --- 9. empirical sign verification table (printed diagnostic) ----------------

def test_sign_verification_table_prints_and_holds(capsys):
    """Emits the documented image-position / yaw / phi / e table for the four
    required scenarios and asserts the sign relationships hold."""
    rows = []
    scenarios = [
        ("centered, forward",        0.0,   0.0),
        ("right edge, looking at cam", _look_at_camera_yaw_deg(700.0, 1500.0), 700.0),
        ("left edge, looking at cam",  _look_at_camera_yaw_deg(-700.0, 1500.0), -700.0),
        ("right, turned away",        0.0,   700.0),
    ]
    for label, yaw_deg, x_mm in scenarios:
        e, yaw, phi = _residual_for(yaw_deg, x_mm)
        rows.append((label, math.degrees(yaw), math.degrees(phi), math.degrees(e)))
        print(f"{label:28s} yaw={math.degrees(yaw):+6.1f} phi={math.degrees(phi):+6.1f} e={math.degrees(e):+6.1f}")

    # centered-forward: everything ~0
    assert abs(rows[0][3]) < 2.0
    # both looking-at-camera edge cases: residual small despite large phi
    assert abs(rows[1][3]) < 5.0 and rows[1][2] > 0
    assert abs(rows[2][3]) < 5.0 and rows[2][2] < 0
    # turned away off-axis: large residual
    assert abs(rows[3][3]) > 15.0
