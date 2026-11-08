"""
perception.py — multi-face perception adapter.

    observations = perception.process_frame(frame, timestamp)

Wraps the repo's existing single-face gaze/eye-contact estimator,
`gaze_core.gaze_estimator.GazeEstimator` (see repo root), per detected face —
per the project constraint against rewriting an existing gaze model. This
file does not reimplement gaze estimation; it only does the multi-face
plumbing GazeEstimator doesn't do on its own: detect up to `max_faces` faces
via MediaPipe Face Mesh, track them (gaze_attention.tracker.IOUTracker) for
stable ids, call GazeEstimator once per face, and convert its output to this
project's pinned observation schema.

Signal conversion (documented once, here)
------------------------------------------
GazeEstimator.estimate() returns a GazeResult with `confidence` already in
[0, 1] (no conversion needed) and `gaze_h`, `gaze_v` — horizontal/vertical
offset of gaze from dead-center, each roughly in [-0.5, 0.5], 0 = looking
straight at the camera. This project's `gaze_signal` is a single [0, 1]
scalar where 1 = fully attending / seeking, so:

    offset       = sqrt(gaze_h**2 + (gaze_v / 1.5)**2)   # vertical de-weighted,
                                                          # matching GazeEstimator's
                                                          # own looser vertical
                                                          # eye-contact margin
    gaze_signal  = clip(1 - offset / 0.45, 0, 1)

0.45 is chosen so that `gaze_signal == seeking_threshold (0.6)` exactly at
`offset == gaze_center_margin (0.18)` — the same horizontal offset at which
GazeEstimator's own `eye_contact` boolean flips to False. So this project's
`seeking_attention` derivation (gaze_signal >= seeking_threshold) agrees with
GazeEstimator's own eye-contact decision for the common case of a purely
horizontal look-away; it diverges only when GazeEstimator's confidence
heuristic doesn't kick in (which is used separately, unchanged, as this
project's `confidence`).

A GazeResult that is not valid (blink — is_valid False, confidence already
low ~0.10) has its gaze_h/gaze_v stubbed to 0.0 by GazeEstimator, which would
wrongly compute as *perfect* eye contact if converted naively. Invalid frames
instead hold the person's last valid gaze_signal (frozen), matching the
single-person pipeline's own fail-soft design (ThresholdPolicy: "frames
below min_confidence... timers freeze").

Face landmark detection: Tasks API (`mediapipe.tasks.python.vision.FaceLandmarker`)
--------------------------------------------------------------------------------------
Uses the modern MediaPipe Tasks API, not the deprecated `mp.solutions.face_mesh`
("solutions") API — the latter still runs but is legacy per MediaPipe's own
docs. Unlike solutions, Tasks requires an explicit model bundle file (not
shipped inside the `mediapipe` package): `models/face_landmarker.task`,
resolved relative to this repo (`multi_person/models/`); see that
directory's own note for where it came from and how to refetch it if
missing. `FaceLandmarker` runs in `VIDEO` mode (matches solutions
FaceMesh's default `static_image_mode=False` temporal-tracking behavior)
via `detect_for_video(image, timestamp_ms)`, which requires
monotonically-increasing integer millisecond timestamps — this class
tracks the last timestamp it sent and bumps by at least 1ms if a caller
ever passes a non-increasing one, rather than raising. `result.face_landmarks`
is a list of per-face landmark lists directly (no `.landmark` wrapper
attribute like `solutions`' `multi_face_landmarks[i].landmark` had), but
each landmark is still a `NormalizedLandmark` with `.x`/`.y`/`.z` in
[0, 1] — every place in this file that reads `.x`/`.y` off a landmark is
unaffected by the migration.

Fallback perception
--------------------
If GazeEstimator raises for a face (corrupt landmarks, etc.) or
`--fallback-perception` is passed, this module falls back to a heuristic
that does NOT use GazeEstimator at all:
  * gaze_signal <- 1 - (distance of face-box center from frame center,
    normalized by half the frame diagonal), clamped to [0, 1] — "engagement
    from face center position" per the project spec's fallback heuristic.
  * confidence  <- fraction of GazeEstimator's own tracked landmark indices
    (eye corners/lids/iris centers) that fall within the frame bounds —
    "confidence from landmark visibility count" per spec, using in-frame-ness
    as the visibility proxy since MediaPipe Face Mesh doesn't expose a
    per-landmark visibility score.
Each observation carries extra (non-pinned) keys read via `.get()` by
callers that care: `"fallback": bool` labels fallback-derived observations
on screen (never present as gaze ground truth), and `"mouth_activity"`,
`"speaking"`, `"speech_confidence"` carry the visual_speaking_proxy fields
(see gaze_attention/speech.py and this module's `_mouth_opening_from_landmarks`).
Because no microphone was used, mouth motion is treated as a visual proxy
for speaking activity and is not equivalent to speaker identification or
speech recognition (see README.md).
"""

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, TypedDict

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _ROOT.parent  # multi_person/../  == robot-attention-V2/
for p in (_ROOT, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gaze_attention import raw_dt_from_timestamps, sanitize_dt  # noqa: E402
from gaze_attention.speech import VisualSpeakingProxy  # noqa: E402
from gaze_attention.tracker import IOUTracker  # noqa: E402

try:
    from gaze_core.gaze_estimator import GazeEstimator, GazeResult  # noqa: E402
    _GAZE_ESTIMATOR_AVAILABLE = True
except ImportError:
    _GAZE_ESTIMATOR_AVAILABLE = False

# Landmark indices GazeEstimator reads — reused here only to compute the
# fallback "visibility count" confidence proxy, never to redo its math.
_TRACKED_LANDMARK_INDICES = (33, 133, 362, 263, 159, 145, 386, 374, 468, 473)

_OFFSET_TO_SIGNAL_SCALE = 0.45
_VERTICAL_DEWEIGHT = 1.5

# Inner-lip and outer-corner landmark indices for the mouth-opening ratio
# (a "mouth aspect ratio" analogous to GazeEstimator's own EAR) — NOT part
# of the wrapped gaze model, this project's own new geometry only.
_MOUTH_UPPER_INNER = 13
_MOUTH_LOWER_INNER = 14
_MOUTH_LEFT_CORNER = 61
_MOUTH_RIGHT_CORNER = 291

# Tasks API model bundle (see this module's docstring for why Tasks, not
# the deprecated solutions.face_mesh API, and why this file has to exist
# on disk -- Tasks doesn't ship a bundled model the way solutions did).
_FACE_LANDMARKER_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# --- Head-pose estimation (Change A: mutual-gaze residual) -------------------
#
# Canonical 6-point 3D face model (millimetres, nose-tip at origin) paired
# 1:1, IN THE SAME ORDER, with the MediaPipe FaceMesh landmark indices of the
# same physical points. This is the classic OpenCV/solvePnP head-pose point
# set. cv2.solvePnP recovers the rotation that maps these 3D model points to
# their observed 2D image positions.
#
# _HEAD_POSE_MODEL_POINTS[i] corresponds to _HEAD_POSE_LANDMARK_INDICES[i]:
#   nose tip           -> 1
#   chin               -> 152
#   left eye, l corner -> 33     (anatomical assignment, see caveat below)
#   right eye, r corner-> 263
#   left mouth corner  -> 61
#   right mouth corner -> 291
#
# CAVEAT (documented, not hidden): the anatomical left/right assignment of
# the eye/mouth corners to FaceMesh indices is the one part of this that a
# synthetic round-trip test (which projects the model points directly) cannot
# validate -- getting it mirrored would flip the recovered yaw sign. It uses
# the standard, widely-published MediaPipe head-pose correspondence; a live
# runtime diagnostic (see _head_pose_from_landmarks' logging hook) is the way
# to confirm it against a real face. The residual math and yaw-extraction
# sign ARE verified synthetically (see tests/test_head_pose.py).
_HEAD_POSE_LANDMARK_INDICES = (1, 152, 33, 263, 61, 291)
_HEAD_POSE_MODEL_POINTS = np.array([
    [0.0,     0.0,    0.0],     # nose tip
    [0.0,  -330.0,  -65.0],     # chin
    [-225.0, 170.0, -135.0],    # left eye, left corner
    [225.0,  170.0, -135.0],    # right eye, right corner
    [-150.0, -150.0, -125.0],   # left mouth corner
    [150.0,  -150.0, -125.0],   # right mouth corner
], dtype=np.float64)


def _approx_camera_matrix(w: int, h: int, hfov_deg: float) -> np.ndarray:
    """APPROXIMATE, UNCALIBRATED pinhole camera matrix derived purely from
    the configured horizontal FOV (no real camera calibration is available
    in this project). fx is set so the image half-width subtends hfov/2;
    fy = fx (square-pixel assumption); principal point at the image centre.
    Good enough for a coarse head-pose/bearing estimate, NOT metrology."""
    hfov_rad = math.radians(hfov_deg)
    fx = (w / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx
    return np.array([[fx, 0.0, w / 2.0],
                     [0.0, fy, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _head_pose_from_landmarks(landmarks, w: int, h: int, camera_matrix: np.ndarray):
    """Estimate (yaw, pitch) in RADIANS from FaceMesh landmarks via
    cv2.solvePnP against the canonical model above. Returns None (never a
    fabricated estimate) if landmarks are insufficient, solvePnP fails, or
    any recovered value is non-finite.

    Distortion coefficients are zero (approximate/uncalibrated model).
    rvec (Rodrigues rotation vector) -> R via cv2.Rodrigues; yaw/pitch are
    extracted with the standard Tait-Bryan decomposition:
        sy    = sqrt(R00^2 + R10^2)
        yaw   = atan2(-R20, sy)   # rotation about the vertical (image-y) axis
        pitch = atan2(R21, R22)   # rotation about the horizontal (image-x) axis
    YAW SIGN -- ONE documented flip is applied at this boundary (`-yaw` in
    the return below), and nowhere else. The residual formula in
    process_frame stays exactly `e = yaw + phi`.

    Why: the synthetic round trip in tests/test_head_pose.py projects
    _HEAD_POSE_MODEL_POINTS and reads them back through the SAME
    index->model correspondence, so it is structurally blind to a mirrored
    anatomical assignment of the FaceMesh eye/mouth indices (33/263,
    61/291) -- and to main_webcam.py's `cv2.flip(frame, 1)` preview mirror.
    That was the documented outstanding caveat.

    Live-camera evidence (logs/session_20260820_140240.jsonl, 26 304
    pose-valid frames, subject looking at the lens from both the left and
    right edges of frame) settled it:

        corr(yaw, phi) = +0.745      # must be ~ -1 if the head compensates
        mean |e| = 41.4 deg, mean q = 0.083, only 4.9% of frames |e| < 15 deg

    Both terms carried the SAME sign, so `yaw + phi` added instead of
    cancelling. Recomputing the same log with yaw negated:

        mean |e| = 15.3 deg, median 11.5 deg, 63.0% of frames < 15 deg,
        mean q = 0.569

    NOTE (calibration, NOT corrected here): a free scale fit `e = a*yaw +
    phi` over that log minimises at a ~= -0.5 rather than -1.0, i.e. the
    recovered yaw MAGNITUDE is roughly 2x too large. That is expected from
    the approximate, uncalibrated HFOV-derived camera matrix
    (_approx_camera_matrix) and is a separate calibration problem. Only the
    SIGN -- a genuine correctness bug -- is fixed here; a fitted 0.5 would
    be overfitting one session. See docs/LIVE_CAMERA_VALIDATION.md."""
    import cv2

    max_idx = max(_HEAD_POSE_LANDMARK_INDICES)
    if landmarks is None or len(landmarks) <= max_idx:
        return None

    image_points = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in _HEAD_POSE_LANDMARK_INDICES],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(image_points)):
        return None

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rvec, _tvec = cv2.solvePnP(
            _HEAD_POSE_MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    yaw = math.atan2(-R[2, 0], sy)
    pitch = math.atan2(R[2, 1], R[2, 2])
    if not (math.isfinite(yaw) and math.isfinite(pitch)):
        return None
    # THE one documented sign flip (see this function's docstring for the
    # live-camera evidence). Applied here at the extraction boundary so the
    # residual stays the plain `e = yaw + phi`. Pitch is unaffected: a
    # horizontal mirror negates yaw and roll but preserves pitch.
    return -yaw, pitch


def _bearing_phi(face_center_x: float, w: int, hfov_deg: float) -> float:
    """Bearing (RADIANS) of a face from the optical axis given its
    horizontal image position. u_x = face_center_x / w in [0,1];
    phi = atan((2*u_x - 1) * tan(HFOV/2)). Right of centre (u_x > 0.5) ->
    phi > 0; left of centre -> phi < 0; centre -> 0."""
    u_x = face_center_x / w if w else 0.5
    return math.atan((2.0 * u_x - 1.0) * math.tan(math.radians(hfov_deg) / 2.0))


def _mutual_gaze_q(e: float, sigma_deg: float) -> float:
    """Gaussian mapping of the mutual-gaze residual e (RADIANS) to a [0,1]
    interaction cue: q = exp(-e^2 / (2*sigma^2)), sigma in radians."""
    sigma = math.radians(sigma_deg)
    return math.exp(-(e ** 2) / (2.0 * sigma ** 2))


# --- Confidence components (Change B: real confidence variance) --------------


def _face_size_confidence(bbox, frame_h: int, ref_ratio: float) -> float:
    """c_size = clip(face_diagonal / (ref_ratio * frame_h), 0, 1). A face
    filling ref_ratio of the frame height (by bbox diagonal) reaches 1.0;
    smaller faces score proportionally lower. Guards degenerate frame/bbox
    (returns 0.0 -- a face with no measurable size gets no size confidence,
    never NaN)."""
    if frame_h <= 0:
        return 0.0
    x1, y1, x2, y2 = bbox
    d = math.hypot(x2 - x1, y2 - y1)
    denom = ref_ratio * frame_h
    if denom <= 0 or not math.isfinite(d):
        return 0.0
    return float(np.clip(d / denom, 0.0, 1.0))


def _pose_confidence(yaw: float) -> float:
    """c_pose = clip(cos(yaw), 0, 1) ** 2, yaw in RADIANS. A frontal face
    (yaw ~ 0) scores ~1.0; a strongly-turned head scores low; a head turned
    past 90deg clips to 0.0 (cos < 0). Only for AVAILABLE poses -- an
    unavailable pose uses the configured conservative fallback instead (see
    Perception._compute_confidence), never this function with a fake yaw."""
    if not math.isfinite(yaw):
        return 0.0
    return float(np.clip(math.cos(yaw), 0.0, 1.0) ** 2)


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


def _face_bbox(landmarks, w: int, h: int) -> List[float]:
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, x2 = max(0.0, min(xs)), min(float(w), max(xs))
    y1, y2 = max(0.0, min(ys)), min(float(h), max(ys))
    return [x1, y1, x2, y2]


def _mouth_opening_from_landmarks(landmarks, w: int, h: int) -> float:
    """Normalized mouth-opening ratio: inner-lip vertical gap / mouth width
    — a "mouth aspect ratio" analogous to GazeEstimator's own EAR, but this
    project's own geometry (gaze_core never computed this). Guarded against
    divide-by-zero the same way compute_ear/compute_gaze_ratio are."""
    def p(idx: int) -> np.ndarray:
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h], dtype=np.float64)

    upper, lower = p(_MOUTH_UPPER_INNER), p(_MOUTH_LOWER_INNER)
    left, right = p(_MOUTH_LEFT_CORNER), p(_MOUTH_RIGHT_CORNER)
    vertical = np.linalg.norm(upper - lower)
    horizontal = np.linalg.norm(left - right)
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / horizontal)


def _gaze_result_to_signal(result, last_signal: float) -> tuple[float, float]:
    """(gaze_signal, confidence) from a valid-or-not GazeResult, per the
    conversion documented in this module's docstring."""
    if not result.is_valid:
        return last_signal, float(result.confidence)
    offset = math.sqrt(result.gaze_h ** 2 + (result.gaze_v / _VERTICAL_DEWEIGHT) ** 2)
    signal = float(np.clip(1.0 - offset / _OFFSET_TO_SIGNAL_SCALE, 0.0, 1.0))
    return signal, float(result.confidence)


def _fallback_signal(bbox: List[float], landmarks, w: int, h: int) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    frame_cx, frame_cy = w / 2.0, h / 2.0
    dist = math.hypot(cx - frame_cx, cy - frame_cy)
    max_dist = math.hypot(w / 2.0, h / 2.0) or 1.0
    gaze_signal = float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))

    in_bounds = 0
    for idx in _TRACKED_LANDMARK_INDICES:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
            in_bounds += 1
    confidence = in_bounds / len(_TRACKED_LANDMARK_INDICES)
    return gaze_signal, confidence


class Perception:
    """Owns the MediaPipe FaceLandmarker (Tasks API) pipeline, the IOU
    tracker, and the per-person GazeEstimator-wrapping state. One instance
    per session; call `process_frame` once per video frame."""

    def __init__(self, config: dict, force_fallback: bool = False):
        self.config = config
        self.force_fallback = force_fallback or not _GAZE_ESTIMATOR_AVAILABLE
        if self.force_fallback and not _GAZE_ESTIMATOR_AVAILABLE:
            print(
                "[perception] gaze_core.GazeEstimator not importable — "
                "running FALLBACK PERCEPTION for the whole session."
            )

        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        if not _FACE_LANDMARKER_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Face landmark model not found at {_FACE_LANDMARKER_MODEL_PATH}. "
                "The Tasks API (mediapipe.tasks.python.vision.FaceLandmarker) needs "
                "this model bundle on disk -- see multi_person/models/README.md for "
                "how to fetch it."
            )

        self._mp_image_format = mp.ImageFormat.SRGB
        self._face_landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(_FACE_LANDMARKER_MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=config.get("max_faces", 3),
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._last_landmarker_timestamp_ms: Optional[int] = None
        self._tracker = IOUTracker(
            iou_threshold=config.get("iou_threshold", 0.3),  # deprecated, unused -- kept for back-compat
            max_missed_frames=config.get("max_missed_frames", 15),  # deprecated, unused -- kept for back-compat
            n_confirm=config.get("n_confirm", 3),
            t_lost_seconds=config.get("t_lost_seconds", 1.5),
            iou_gate=config.get("iou_gate", 0.2),
            velocity_alpha=config.get("velocity_alpha", 0.5),
        )
        self._estimator = GazeEstimator() if _GAZE_ESTIMATOR_AVAILABLE else None
        self._last_signal: Dict[int, float] = {}
        self.last_frame_used_fallback: bool = False

        # Mutual-gaze residual (Change A) config, read once.
        self._hfov_deg = float(config.get("hfov_deg", 60.0))
        self._mutual_gaze_sigma_deg = float(config.get("mutual_gaze_sigma_deg", 12.0))
        self._mouth_cue_weight = float(config.get("mouth_cue_weight", 0.0))
        # Approximate/uncalibrated camera matrix, rebuilt lazily per frame
        # size (cached; frame dimensions don't change mid-session in
        # practice but this stays correct if they do).
        self._camera_matrix: Optional[np.ndarray] = None
        self._camera_matrix_wh: Optional[tuple] = None

        # Confidence variance (Change B) config, read once.
        self._confidence_min = float(config.get("confidence_min", 0.05))
        self._confidence_max = float(config.get("confidence_max", 1.0))
        self._face_size_reference_ratio = float(config.get("face_size_reference_ratio", 0.18))
        self._stability_window_frames = int(config.get("stability_window_frames", 5))
        self._stability_k = float(config.get("stability_k", 50.0))
        self._pose_unavailable_confidence = float(config.get("pose_unavailable_confidence", 0.5))
        self._stability_init_confidence = float(config.get("stability_init_confidence", 0.5))
        # Per-person rolling history of frame-normalized landmark arrays, for
        # c_stable. Keyed by tracker id; pruned when a track disappears (same
        # lifecycle as _last_signal) so a recycled id never inherits stale
        # landmarks from a previous person.
        self._landmark_history: Dict[int, Deque[np.ndarray]] = {}

        # visual_speaking_proxy (see speech.py): mouth-motion auxiliary cue,
        # NOT audio speech recognition -- there is no microphone.
        self._speech_proxy = VisualSpeakingProxy(
            speaking_on_threshold=config.get("speaking_on_threshold", 0.15),
            speaking_off_threshold=config.get("speaking_off_threshold", 0.08),
            speaking_window_seconds=config.get("speaking_window_seconds", 0.5),
            activity_scale=config.get("speech_activity_scale", 0.02),
            # Defect B temporal-support gate (see speech.py's docstring).
            support_window_seconds=config.get("speech_support_window_seconds", 0.30),
            minimum_positive_fraction=config.get("speech_minimum_positive_fraction", 0.30),
            activity_delta_threshold=config.get("speech_activity_delta_threshold", 0.01),
        )
        self._last_timestamp: Optional[float] = None

    def _stability_confidence(self, pid: int, landmarks, w: int, h: int) -> float:
        """c_stable = 1 / (1 + stability_k * jitter), where jitter is the
        mean temporal std of frame-normalized landmark coordinates over the
        rolling window for this person id.

        Normalization: MediaPipe landmark .x/.y are ALREADY frame-normalized
        to [0,1] (equivalent to pixel/frame_width, pixel/frame_height), so
        they're used directly -- documented here rather than dividing a second
        time. Until a full `stability_window_frames` of history has
        accumulated, returns the conservative `stability_init_confidence`
        (NOT 1.0 -- a brand-new track is never treated as perfectly stable).
        Invalid/non-finite landmarks return the init value WITHOUT recording,
        so garbage never enters the jitter estimate."""
        try:
            coords = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float64)
        except (TypeError, AttributeError):
            return self._stability_init_confidence
        if coords.size == 0 or not np.all(np.isfinite(coords)):
            return self._stability_init_confidence

        hist = self._landmark_history.get(pid)
        if hist is None or hist.maxlen != self._stability_window_frames:
            hist = deque(maxlen=self._stability_window_frames)
            self._landmark_history[pid] = hist
        # If the landmark count changed (shouldn't mid-track, but guard),
        # drop stale history rather than crash on a ragged stack.
        if hist and hist[-1].shape != coords.shape:
            hist.clear()
        hist.append(coords)

        if len(hist) < self._stability_window_frames:
            return self._stability_init_confidence

        stack = np.stack(tuple(hist), axis=0)  # (window, n_landmarks, 2)
        jitter = float(np.mean(np.std(stack, axis=0)))
        if not math.isfinite(jitter):
            return self._stability_init_confidence
        return float(1.0 / (1.0 + self._stability_k * jitter))

    def _compute_confidence(self, pid, bbox, landmarks, yaw, w, h):
        """Real confidence C = clip(c_size * c_pose * c_stable,
        confidence_min, confidence_max) for a valid, non-lost observation.
        Returns (C, c_size, c_pose, c_stable). `yaw` is None when head pose
        was unavailable, in which case c_pose uses the conservative
        `pose_unavailable_confidence` (never 1.0). Always finite and within
        [confidence_min, confidence_max]."""
        c_size = _face_size_confidence(bbox, h, self._face_size_reference_ratio)
        c_pose = _pose_confidence(yaw) if yaw is not None else self._pose_unavailable_confidence
        c_stable = self._stability_confidence(pid, landmarks, w, h)
        product = c_size * c_pose * c_stable
        if not math.isfinite(product):
            product = self._confidence_min
        C = float(np.clip(product, self._confidence_min, self._confidence_max))
        return C, float(c_size), float(c_pose), float(c_stable)

    def process_frame(self, frame, timestamp: float) -> List[Observation]:
        h, w = frame.shape[:2]
        import cv2
        import mediapipe as mp

        # dt for the speech proxy's own temporal windowing -- tracked
        # internally so process_frame(frame, timestamp) keeps its pinned
        # two-argument interface (the gaze filter's dt is computed
        # separately by the caller, e.g. main_webcam.py's own loop).
        raw_dt = raw_dt_from_timestamps(timestamp, self._last_timestamp)
        dt, _ = sanitize_dt(
            raw_dt,
            fallback=self.config.get("dt_fallback_seconds", 0.033),
            max_dt=self.config.get("dt_max_seconds", 0.2),
        )
        self._last_timestamp = timestamp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=self._mp_image_format, data=rgb)

        # VIDEO mode requires strictly-increasing integer ms timestamps --
        # real wall-clock timestamps (main_webcam.py's time.time()) satisfy
        # this naturally; bump defensively rather than raise if a caller
        # ever passes a non-increasing one (e.g. a test reusing timestamp=0.0
        # across repeated calls on the same instance).
        timestamp_ms = int(timestamp * 1000)
        if self._last_landmarker_timestamp_ms is not None and timestamp_ms <= self._last_landmarker_timestamp_ms:
            timestamp_ms = self._last_landmarker_timestamp_ms + 1
        self._last_landmarker_timestamp_ms = timestamp_ms

        result = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)

        faces = result.face_landmarks or []  # list of per-face landmark lists (Tasks API; no ".landmark" wrapper)
        bboxes = [_face_bbox(landmarks, w, h) for landmarks in faces]
        ids = self._tracker.update(bboxes, dt=dt)

        seeking_threshold = self.config.get("seeking_threshold", 0.6)
        observations: List[Observation] = []
        frame_fallback = self.force_fallback

        # Approximate/uncalibrated camera matrix for head-pose solvePnP,
        # cached per frame size (Change A).
        if self._camera_matrix_wh != (w, h):
            self._camera_matrix = _approx_camera_matrix(w, h, self._hfov_deg)
            self._camera_matrix_wh = (w, h)

        for pid, bbox, landmarks in zip(ids, bboxes, faces):
            if pid is None:
                # TENTATIVE track -- not yet confirmed, no public id exists.
                # Skipped here, before any observation is constructed, so an
                # unconfirmed detection can never enter policy state (see
                # tracker.py's three-state lifecycle).
                continue

            used_fallback = self.force_fallback

            if not used_fallback:
                try:
                    result = self._estimator.estimate(landmarks, w, h)
                    signal, confidence = _gaze_result_to_signal(result, self._last_signal.get(pid, 0.0))
                except Exception as exc:  # noqa: BLE001 - per-face inference error, must not kill the session
                    print(f"[perception] GazeEstimator failed for id={pid}: {exc}; using fallback for this face.")
                    used_fallback = True

            if used_fallback:
                signal, confidence = _fallback_signal(bbox, landmarks, w, h)
                frame_fallback = True

            # visual_speaking_proxy: independent of whether gaze used the
            # fallback path -- mouth geometry doesn't depend on GazeEstimator.
            mouth_opening = _mouth_opening_from_landmarks(landmarks, w, h)
            mouth_activity, speaking, speech_confidence = self._speech_proxy.update(pid, mouth_opening, dt)

            # Mutual-gaze residual cue (Change A). When head pose is
            # recoverable, the interaction cue becomes q (a mutual-gaze
            # measure), optionally combined with the existing mouth-motion
            # activity via mouth_cue_weight (default 0.0 -> cue is q alone).
            # When head pose is NOT recoverable, `signal` keeps its existing
            # value (the position-based fallback / GazeEstimator signal
            # computed above) and yaw/pitch/phi/e/q stay None -- no fabricated
            # gaze estimate, existing fallback behavior preserved.
            yaw = pitch = phi = e = q = None
            pose = _head_pose_from_landmarks(landmarks, w, h, self._camera_matrix)
            if pose is not None:
                yaw, pitch = pose
                face_center_x = (bbox[0] + bbox[2]) / 2.0
                phi = _bearing_phi(face_center_x, w, self._hfov_deg)
                e = yaw + phi  # residual; sign verified synthetically, no flip applied
                q = _mutual_gaze_q(e, self._mutual_gaze_sigma_deg)
                # final interaction cue = q + mouth_cue_weight * mouth_activity,
                # clamped to [0,1]. Default weight 0.0 -> cue == q.
                signal = float(np.clip(q + self._mouth_cue_weight * mouth_activity, 0.0, 1.0))

            # Real confidence (Change B). Replaces the old fallback
            # confidence (landmark-visibility fraction, ~constant 1.0) as the
            # confidence for valid, non-lost observations. `confidence` from
            # _gaze_result_to_signal/_fallback_signal above is now discarded
            # -- only its `signal` is used. c_pose uses the conservative
            # pose_unavailable_confidence when yaw is None.
            C, c_size, c_pose, c_stable = self._compute_confidence(pid, bbox, landmarks, yaw, w, h)
            confidence = C

            self._last_signal[pid] = signal
            observations.append(Observation(
                id=pid,
                bbox=bbox,
                gaze_signal=signal,
                confidence=confidence,
                seeking_attention=signal >= seeking_threshold,
                timestamp=timestamp,
                mouth_activity=mouth_activity,
                speaking=speaking,
                speech_confidence=speech_confidence,
            ))
            # Non-pinned extra keys, read via .get() by callers that care.
            observations[-1]["fallback"] = used_fallback  # type: ignore[typeddict-item]
            # Mutual-gaze residual diagnostics (Change A); None when head
            # pose was unavailable this frame (never a fabricated value).
            # Angles are in RADIANS, consistent with the JSONL schema.
            observations[-1]["yaw"] = yaw      # type: ignore[typeddict-item]
            observations[-1]["pitch"] = pitch  # type: ignore[typeddict-item]
            observations[-1]["phi"] = phi      # type: ignore[typeddict-item]
            observations[-1]["e"] = e          # type: ignore[typeddict-item]
            observations[-1]["q"] = q          # type: ignore[typeddict-item]
            # Confidence components (Change B).
            observations[-1]["c_size"] = c_size      # type: ignore[typeddict-item]
            observations[-1]["c_pose"] = c_pose      # type: ignore[typeddict-item]
            observations[-1]["c_stable"] = c_stable  # type: ignore[typeddict-item]
            observations[-1]["C"] = C                # type: ignore[typeddict-item]

        # LOST tracks (temporarily occluded, still within t_lost_seconds):
        # keep emitting a frozen observation so the id stays in present_ids
        # and per-person policy state (engagement/debt/etc.) isn't pruned
        # out from under it -- see tracker.py's lifecycle and
        # dwell_arbitration.py's pruning. gaze_signal is frozen at the last
        # real value (existing _last_signal cache); confidence is HARD-CODED
        # 0.0, never carried forward -- a lost/invisible person must never
        # clear any confidence gate (debt accrual, etc.), which a cached
        # last-known confidence would incorrectly allow. speaking is forced
        # False -- there is no new evidence they're still talking.
        for lost_pid, predicted_bbox in self._tracker.lost_tracks().items():
            frozen_signal = self._last_signal.get(lost_pid, 0.0)
            observations.append(Observation(
                id=lost_pid,
                bbox=list(predicted_bbox),
                gaze_signal=frozen_signal,
                confidence=0.0,
                seeking_attention=frozen_signal >= seeking_threshold,
                timestamp=timestamp,
                mouth_activity=0.0,
                speaking=False,
                speech_confidence=0.0,
            ))
            observations[-1]["fallback"] = True  # type: ignore[typeddict-item]
            observations[-1]["lost"] = True  # type: ignore[typeddict-item]
            # Change A/B fields for a LOST observation: no new evidence, so
            # the mutual-gaze diagnostics are null and confidence is exactly
            # 0.0 (the normal C formula is NOT applied). Components null, not
            # 0.0-from-formula, to make "no measurement" distinct from "a
            # measured zero". A lost person must never gain positive
            # confidence just because old landmarks remain in history.
            for _k in ("yaw", "pitch", "phi", "e", "q", "c_size", "c_pose", "c_stable"):
                observations[-1][_k] = None  # type: ignore[typeddict-item]
            observations[-1]["C"] = 0.0  # type: ignore[typeddict-item]

        # Drop cached last-known signal (and mouth/landmark history) for ids
        # the tracker no longer holds AT ALL (deleted, past t_lost_seconds).
        # LOST-but-retained ids must NOT be pruned, or their frozen signal
        # would reset to 0.0 the moment they're re-detected. Landmark history
        # is pruned here too so a recycled id never inherits a prior person's
        # landmarks (feeding a false-stable c_stable).
        live_ids = {i for i in ids if i is not None} | set(self._tracker.lost_tracks().keys())
        for pid in list(self._last_signal):
            if pid not in live_ids:
                del self._last_signal[pid]
                self._speech_proxy.forget(pid)
        for pid in list(self._landmark_history):
            if pid not in live_ids:
                del self._landmark_history[pid]

        self.last_frame_used_fallback = frame_fallback
        return observations

    def close(self) -> None:
        self._face_landmarker.close()
