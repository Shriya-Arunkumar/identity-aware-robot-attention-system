#!/usr/bin/env bash
# fetch_face_landmarker_model.sh — downloads the MediaPipe Tasks API model
# bundle gaze_attention/perception.py needs (models/face_landmarker.task).
#
# Not committed to git (see .gitignore's `models/*.task`) -- it's a ~3.7MB
# third-party pretrained asset, not project source, and re-fetching it here
# keeps the repo lean the same way logs/ and venv/ are kept out of git.
# This is the exact command used when the mediapipe solutions->Tasks
# migration was done; run it once after cloning, or any time
# models/face_landmarker.task is missing (Perception() raises a clear
# FileNotFoundError telling you to run this script if it's absent).
#
# Usage:
#   ./scripts/fetch_face_landmarker_model.sh

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
dest="models/face_landmarker.task"

echo "[fetch_face_landmarker_model] downloading ${url} -> ${dest}"
curl -fSL --max-time 60 -o "${dest}" "${url}"
echo "[fetch_face_landmarker_model] done: $(du -h "${dest}" | cut -f1)"
