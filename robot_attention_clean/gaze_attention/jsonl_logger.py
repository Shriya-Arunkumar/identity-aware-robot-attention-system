"""
jsonl_logger.py — JSONL diagnostic logger, ALONGSIDE (never replacing or
modifying) logger.py's SessionLogger CSV. One .jsonl file per session under
logs/, two kinds of lines (distinguished by "type"):

  "frame"  — one line per tracked person per processed frame:
             {type, t, id, bbox, yaw, pitch, phi, e, q, c_size, c_pose,
              c_stable, C, engagement, score, lost}
             yaw/pitch/phi/e/q/c_size/c_pose/c_stable/C are None until the
             mutual-gaze-residual (Change A) and confidence-components
             (Change B) work lands -- this schema is written up front, not
             grown incrementally, so downstream tooling/analysis scripts
             can target the final shape from day one.

  "switch" — one line per actual target change (any policy -- detected
             directly by comparing this tick's target against the previous
             tick's, not by relying on ConversationalSimplePolicy-specific
             bookkeeping, so it works uniformly across policies):
             {type, t, from_id, to_id, reason, scores, confidences}
             `reason` is None for policies that don't track it (only
             ConversationalSimplePolicy currently does -- see
             policies.py's last_switch_reason). `scores`/`confidences` are
             keyed by person id (as strings, for JSON) and cover only the
             two ids involved in the switch (from_id/to_id) -- every
             person's score/confidence for this same tick is already in
             the "frame" lines emitted the same tick, so repeating all of
             them here would just duplicate that.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def new_jsonl_path(logs_dir: Optional[Path] = None) -> Path:
    logs_dir = logs_dir or _LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"session_{stamp}.jsonl"
    # Extremely unlikely (same-second re-run), but never overwrite.
    suffix = 1
    while path.exists():
        path = logs_dir / f"session_{stamp}_{suffix}.jsonl"
        suffix += 1
    return path


class JsonlLogger:
    """Buffered per-frame + per-switch-event JSONL writer. Fully
    independent of SessionLogger (logger.py) -- separate file, separate
    schema, separate enabled flag -- runs alongside it, never modifies it."""

    def __init__(self, logs_dir: Optional[Path] = None, enabled: bool = True):
        self.path = new_jsonl_path(logs_dir)
        self.enabled = enabled
        self._file = open(self.path, "w")

    def log_frame(
        self,
        t: float,
        observations: List[dict],
        metrics_by_id: Dict[int, dict],
    ) -> None:
        """One JSON line per tracked person this frame."""
        if not self.enabled:
            return
        for obs in observations:
            pid = obs["id"]
            metrics = metrics_by_id.get(pid, {})
            row = {
                "type": "frame",
                "t": t,
                "id": pid,
                "bbox": list(obs["bbox"]),
                "yaw": obs.get("yaw"),
                "pitch": obs.get("pitch"),
                "phi": obs.get("phi"),
                "e": obs.get("e"),
                "q": obs.get("q"),
                "c_size": obs.get("c_size"),
                "c_pose": obs.get("c_pose"),
                "c_stable": obs.get("c_stable"),
                "C": obs.get("C"),
                "engagement": metrics.get("engagement"),
                "score": metrics.get("score"),
                "lost": bool(obs.get("lost", False)),
            }
            self._file.write(json.dumps(row) + "\n")
        self._file.flush()

    def log_switch(
        self,
        t: float,
        from_id: Optional[int],
        to_id: Optional[int],
        reason: Optional[str],
        scores: Dict[int, float],
        confidences: Dict[int, float],
    ) -> None:
        """One JSON line per actual target change."""
        if not self.enabled:
            return
        row = {
            "type": "switch",
            "t": t,
            "from_id": from_id,
            "to_id": to_id,
            "reason": reason,
            "scores": {str(k): v for k, v in scores.items()},
            "confidences": {str(k): v for k, v in confidences.items()},
        }
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
