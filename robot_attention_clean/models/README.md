# models/

Third-party pretrained model bundles this project depends on, not project
source — kept out of git (`.gitignore`'s `models/*.task`), fetched instead.

## `face_landmarker.task`

Required by `gaze_attention/perception.py`'s `Perception` class (the
MediaPipe **Tasks** API, `mediapipe.tasks.python.vision.FaceLandmarker`).
Unlike the deprecated `mp.solutions.face_mesh` ("solutions") API this
replaced, Tasks does not ship a model bundle inside the `mediapipe` Python
package — it has to be present on disk, pointed to by
`model_asset_path`.

Fetch it with:

```bash
./scripts/fetch_face_landmarker_model.sh
```

Or manually:

```bash
curl -fSL -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

If this file is missing, `Perception()` raises a clear `FileNotFoundError`
at construction time (not a silent fallback) pointing back here — face
landmark detection is required for every code path in `perception.py`,
including the `--fallback-perception` heuristic (which still needs
landmarks for its face-center/landmark-visibility math, it just skips
`gaze_core.GazeEstimator`).

478 landmarks per detected face are always produced (including the 10 iris
landmarks, indices 468-477) — this is a fixed property of this specific
model bundle, not a configurable option the way `solutions.face_mesh`'s
`refine_landmarks` flag was.
