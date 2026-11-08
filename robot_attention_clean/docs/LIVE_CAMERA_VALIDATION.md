# Live-camera validation procedure (MANUAL)

> **Status: the SIGN CONVENTION BUG this document was written to catch was
> found and fixed on 2026-08-20** using a recorded live session (see
> "Recorded evidence" below). The structured Checks 1–5 below have **still
> not been executed interactively**; the Results table remains empty. Treat
> the sign convention as **corrected and confirmed against one recorded
> session**, not as fully validated across cameras and subjects.

## Why this exists

The Change-A mutual-gaze residual is

```
yaw    = head-pose yaw from cv2.solvePnP        (radians)
phi    = atan((2*u_x - 1) * tan(HFOV/2))        (bearing of the face)
e      = yaw + phi        (residual formula itself is untouched; ONE
                          documented sign flip lives in the yaw EXTRACTION,
                          see 'Recorded evidence' below)
q      = exp(-e^2 / (2*sigma^2))                (interaction cue, [0,1])
```

`tests/test_head_pose.py` verifies the residual algebra and the yaw-extraction
sign **synthetically**, by projecting the canonical 3D model points and
solving them back. That round trip cannot catch one specific error class: the
**anatomical left/right assignment of FaceMesh landmark indices** to the
canonical model points.

```python
_HEAD_POSE_LANDMARK_INDICES = (1, 152, 33, 263, 61, 291)
#                              ^nose ^chin ^L-eye ^R-eye ^L-mouth ^R-mouth
```

If the anatomical assignment is mirrored, a synthetic test still passes — it
projects and un-projects the *same* correspondence — but the recovered yaw
sign is inverted on a real face. That inverts `e`, which collapses `q`, which
disables the entire mutual-gaze cue. **Only a real face can distinguish these
two cases.**

**This is not hypothetical: exactly that happened.** See "Recorded evidence"
below — the yaw extraction and the synthetic test's own definition of
"looking at the camera" were *both* inverted, cancelling each other, so every
synthetic test passed while the live cue was dead (`mean q = 0.083`).

Note also that the camera matrix is HFOV-derived and **uncalibrated**
(`_approx_camera_matrix`), so absolute yaw magnitudes are coarse. This
procedure checks **signs and trends**, not absolute accuracy.

## Prerequisites

```bash
python3 -m gaze_attention.main_webcam --no-ros
```

Per-frame `yaw`, `pitch`, `phi`, `e`, `q` are written to
`logs/session_<timestamp>.jsonl` (see `jsonl_logger.py`). Read them with:

```bash
python3 -c "
import json,sys
for line in open(sys.argv[1]):
    r=json.loads(line)
    if r['type']=='frame' and r['yaw'] is not None:
        import math
        print(f\"t={r['t']:.2f} id={r['id']} yaw={math.degrees(r['yaw']):+6.1f} \"
              f\"phi={math.degrees(r['phi']):+6.1f} e={math.degrees(r['e']):+6.1f} q={r['q']:.3f}\")
" logs/session_<timestamp>.jsonl
```

Confirm `hfov_deg` in `config/config.yaml` (default 60.0) approximately
matches the real camera, or every `phi` will be systematically wrong.

## Checks

Record **degrees**, not radians. Hold each pose ~3 s and use the median.

### Check 1 — FaceMesh left/right landmark ordering

Sit centered and facing the camera. Read the raw normalized x of the two eye
landmarks:

```bash
python3 -c "
import cv2, mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/face_landmarker.task'),
    running_mode=vision.RunningMode.IMAGE, num_faces=1))
cap=cv2.VideoCapture(0); ok,frame=cap.read(); cap.release()
frame=cv2.flip(frame,1)
res=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
f=res.face_landmarks[0]
print('idx 33  x=%.3f' % f[33].x, ' idx 263 x=%.3f' % f[263].x)
print('idx 61  x=%.3f' % f[61].x, ' idx 291 x=%.3f' % f[291].x)
"
```

**Expected**, given `main_webcam.py` mirrors the frame (`cv2.flip(frame, 1)`)
and `_HEAD_POSE_MODEL_POINTS` places index 33 at model x = **-225** (left) and
index 263 at x = **+225** (right):

```
x(33) < x(263)      and      x(61) < x(291)
```

**If either inequality is reversed, STOP.** The correspondence is mirrored;
every check below will be self-consistently wrong. Fix the index pairing (or
apply one documented sign flip at the boundary in
`_head_pose_from_landmarks`, never inside the residual formula) before
continuing.

### Check 2 — Centered face, looking at the camera

Sit centered, look straight at the lens.

```
phi ≈ 0        (|phi| < ~5 deg)
yaw ≈ 0        (|yaw| < ~10 deg)
e   ≈ 0        (|e|   < ~10 deg)
q   high       (> ~0.6)
```

### Check 3 — Face at the RIGHT edge of frame, still looking at the camera

Move to the right of frame (as seen in the mirrored preview), keep looking at
the lens. The head is now physically turned to compensate.

```
phi > 0                       (right of centre)
yaw < 0                       (turned back toward the lens)
e = yaw + phi ≈ 0             (|e| < ~15 deg)  <-- THE KEY CHECK
q  still high                 (> ~0.5)
```

`yaw` and `phi` must have **opposite signs and similar magnitude**. If they
have the *same* sign, `e` roughly doubles instead of cancelling — that is the
mirrored-landmark failure from Check 1.

### Check 4 — Face at the LEFT edge of frame, still looking at the camera

```
phi < 0
yaw > 0
e ≈ 0                         (|e| < ~15 deg)
q  still high                 (> ~0.5)
```

### Check 5 — Face turned away

From centre, turn your head ~45 deg away while staying in frame.

```
|e| increases markedly vs Check 2
q  decreases markedly vs Check 2  (expect < ~0.3 at 45 deg with sigma=12 deg)
```

Sweep 0 → 45 deg and confirm `q` decreases **monotonically**. A `q` that rises
as you turn away is an inverted residual.

## Recorded evidence — 2026-08-20 — SIGN INVERSION FOUND AND FIXED

Source: `logs/session_20260820_140240.jsonl`, 480 s, **26 304 pose-valid
frames**. Subject reported looking directly at the lens from both the left
and right edges of frame, so `e` should have been ≈ 0 in both positions.

**Symptom.** `e` did not cancel — it tracked `yaw`:

| Correlation | Measured | Expected if correct |
|---|---:|---|
| `corr(yaw, phi)` | **+0.745** | ≈ −1 |
| `corr(e, yaw)` | +0.965 | ≈ 0 |
| `corr(e, phi)` | +0.894 | ≈ 0 |

Binned by horizontal position, both terms carried the **same** sign, so
`e = yaw + phi` added instead of cancelling:

| cx bin | mean phi | mean yaw | mean e | mean q |
|---|---:|---:|---:|---:|
| 0–380 (left) | −16.2° | −22.5° | **−38.7°** | 0.100 |
| 560–700 (right) | +25.3° | +21.0° | **+46.4°** | 0.039 |

**Root cause — two inverted signs cancelling each other.**

1. `_head_pose_from_landmarks` returned yaw with inverted sign.
2. `tests/test_head_pose.py` defined "looking at the camera" as
   `-atan2(x_mm, z_mm)`, annotated *"sign established empirically"* — also
   inverted.

Because the synthetic round trip projects `_HEAD_POSE_MODEL_POINTS` and reads
them back through the **same** correspondence, the two errors cancelled and
every synthetic test passed. Only real-camera data could separate them.

Ruled out: `main_webcam.py`'s `cv2.flip(frame, 1)` preview mirror. A
horizontal mirror negates **both** `yaw` and `phi`, so `e` merely changes
sign and `|e|` is unchanged — verified numerically. The mirror was not the
cause.

The correct look-at-camera yaw was then **derived from the model geometry**
(face-forward axis = `nose − centroid(other points)`), giving
`+atan2(x_mm, z_mm)`: `+25.0°` for a head at x = +700 mm, `−25.0°` at
x = −700 mm — the opposite of what the test had assumed.

**Fix.** One documented flip at the extraction boundary (`return -yaw, pitch`
in `_head_pose_from_landmarks`); the residual formula stays exactly
`e = yaw + phi`. `tests/test_head_pose.py` gained
`_look_at_camera_yaw_deg()`, derived rather than fitted. Pitch is untouched —
a horizontal mirror preserves pitch.

**Result on the same 26 304 frames** (recomputed offline; the log itself is
unmodified):

| Metric | As logged | Corrected |
|---|---:|---:|
| mean \|e\| | 41.4° | **15.3°** |
| median \|e\| | 40.0° | **11.5°** |
| frac \|e\| < 15° | 4.9% | **63.0%** |
| mean q | 0.083 | **0.569** |
| `corr(yaw, phi)` | +0.745 | **−0.745** |
| frames with q ≥ `seeking_threshold` (0.6) | 3.2% | **52.3%** |

Post-fix synthetic sign table (`test_sign_verification_table_prints_and_holds`):

```
centered, forward             yaw=  +0.0 phi=  +0.0 e=  +0.0
right edge, looking at cam    yaw= -25.0 phi= +25.6 e=  +0.6
left edge, looking at cam     yaw= +25.0 phi= -25.6 e=  -0.6
right, turned away            yaw=  -0.0 phi= +27.1 e= +27.1
```

**Outstanding — yaw MAGNITUDE, not sign (NOT fixed).** A free scale fit
`e = a·yaw + phi` over the same log minimises at **a ≈ −0.5**
(mean \|e\| 9.0°), not −1.0, i.e. the recovered yaw is roughly **2× too
large**. This is consistent with the approximate, uncalibrated HFOV-derived
camera matrix (`_approx_camera_matrix`). Only the sign — a genuine
correctness bug — was corrected; baking in a fitted 0.5 would be overfitting
a single session. Proper intrinsic calibration is the real remedy.

## Results — interactive Checks 1–5

**NOT YET RECORDED.** The bug above was found from a recorded session, not
from running these checks interactively. Fill in and date this table after an
actual run; leave it empty otherwise.

| Check | Expected | Measured | Pass/Fail | Date | Notes |
|---|---|---|---|---|---|
| 1 landmark ordering | x(33) < x(263), x(61) < x(291) | | | | |
| 2 centered | phi≈0, yaw≈0, e≈0, q high | | | | |
| 3 right edge | phi>0, yaw<0, e≈0 | | | | |
| 4 left edge | phi<0, yaw>0, e≈0 | | | | |
| 5 turned away | \|e\| up, q down monotonically | | | | |

Camera / resolution / configured `hfov_deg`: ______

## If a check fails

Do **not** silently flip a sign inside the residual formula. Per
`perception.py`'s documented rule, apply exactly **one** documented flip at
the extraction boundary in `_head_pose_from_landmarks`, re-run
`tests/test_head_pose.py`, and record the change and the evidence here.
