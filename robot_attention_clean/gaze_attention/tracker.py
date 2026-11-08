"""
tracker.py — multi-object tracker with a three-state lifecycle
(TENTATIVE -> CONFIRMED -> LOST -> deleted), constant-velocity bbox
prediction, and global (Hungarian) assignment.

    TENTATIVE   freshly-seen, unconfirmed detection. update() returns None
                for its slot -- no public id exists yet, so it is invisible
                to any caller. A TENTATIVE track that misses even one frame
                is deleted immediately (never given a LOST grace period --
                it was never confirmed enough to earn one).

    CONFIRMED   promoted once a TENTATIVE track accumulates `n_confirm`
                CONSECUTIVE matched frames -- this is the first tick its
                public id is returned. Re-matching a LOST track also
                returns it to CONFIRMED, with the SAME id it had before.

    LOST        a CONFIRMED track that missed this frame. Its bbox is
                frozen (never updated from a detection while LOST);
                `lost_seconds` accumulates wall-clock time (not frame
                count); it remains eligible for re-matching for up to
                `t_lost_seconds`, after which it is deleted for good.

Matching: every CONFIRMED and LOST track's bbox is extrapolated forward by
its own elapsed time-since-last-match, using an EMA'd constant
center-velocity model (`velocity_alpha`), THEN matched against this frame's
detections via a global minimum-cost assignment
(`scipy.optimize.linear_sum_assignment`, cost = 1 - IoU(predicted, det)).
Any assigned pair with IoU below `iou_gate` is rejected and both sides
treated as unmatched. Predicting forward (instead of matching against a
stale last-seen box) plus solving globally (instead of greedily) is what
keeps ids from swapping when two people's boxes cross paths -- a greedy
nearest-IoU match can lock in the wrong pairing right at the crossing
point; the global assignment considers all pairings at once.

TENTATIVE tracks are NOT part of the Hungarian stage (scoped to
CONFIRMED+LOST only). After that stage consumes some detections, any
remaining detections are greedily IoU-matched (threshold `iou_gate`, no
velocity model yet -- a 1-2-hit-old track has no reliable velocity
estimate) against remaining TENTATIVE tracks; anything still unmatched
becomes a brand new TENTATIVE track.

`iou_threshold`/`max_missed_frames` (the old greedy tracker's constructor
params) are accepted and stored for backward compatibility but are UNUSED
by this state machine -- superseded by `iou_gate`/`t_lost_seconds`.
Deprecated, not removed (kept present in config alongside the new keys).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

BBox = Sequence[float]  # [x1, y1, x2, y2]


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 1e-9:
        return 0.0
    return inter_area / union


def _center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _shift_bbox(bbox: BBox, dx: float, dy: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


@dataclass
class _Track:
    bbox: Tuple[float, float, float, float]  # last MATCHED bbox (frozen while LOST)
    velocity: Tuple[float, float] = (0.0, 0.0)  # EMA'd bbox-center velocity, units/sec
    hits: int = 1  # consecutive matched frames; only meaningful pre-confirmation
    state: TrackState = TrackState.TENTATIVE
    lost_seconds: float = 0.0
    time_since_update: float = 0.0  # elapsed time since bbox was last refreshed by a real match

    def predicted_bbox(self) -> Tuple[float, float, float, float]:
        vx, vy = self.velocity
        return _shift_bbox(self.bbox, vx * self.time_since_update, vy * self.time_since_update)


class IOUTracker:
    """Three-state (TENTATIVE/CONFIRMED/LOST) multi-object tracker. See
    module docstring for the full lifecycle and matching description."""

    def __init__(
        self,
        iou_threshold: float = 0.3,  # deprecated, unused -- superseded by iou_gate
        max_missed_frames: int = 15,  # deprecated, unused -- superseded by t_lost_seconds
        n_confirm: int = 3,
        t_lost_seconds: float = 1.5,
        iou_gate: float = 0.2,
        velocity_alpha: float = 0.5,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.n_confirm = n_confirm
        self.t_lost_seconds = t_lost_seconds
        self.iou_gate = iou_gate
        self.velocity_alpha = velocity_alpha

        self._tracks: Dict[int, _Track] = {}
        self._next_id = 0

    def update(self, detections: List[BBox], dt: float = 1.0 / 30.0) -> List[Optional[int]]:
        """Assign a stable id to each detection (same order/length as
        input). Returns None for a detection matched to a still-TENTATIVE
        track -- no public id exists for it yet.

        `dt` is optional (default: a nominal 30fps frame time) so existing
        callers that only ever pass `detections` keep working unchanged;
        real callers should pass the actual elapsed time for the
        constant-velocity prediction to be meaningful.
        """
        dt = max(float(dt), 0.0)

        # -- 1. Age every track by dt before matching. --------------------
        for track in self._tracks.values():
            track.time_since_update += dt
            if track.state == TrackState.LOST:
                track.lost_seconds += dt

        n_dets = len(detections)
        assigned_track_for_det: List[Optional[int]] = [None] * n_dets
        used_dets: set = set()

        # -- 2. Hungarian assignment: CONFIRMED + LOST tracks vs
        #       detections, matched against each track's PREDICTED
        #       (not stale) bbox. ------------------------------------------
        matchable_ids = [
            tid for tid, t in self._tracks.items()
            if t.state in (TrackState.CONFIRMED, TrackState.LOST)
        ]
        if matchable_ids and n_dets:
            predicted = {tid: self._tracks[tid].predicted_bbox() for tid in matchable_ids}
            cost = np.ones((len(matchable_ids), n_dets), dtype=np.float64)
            for i, tid in enumerate(matchable_ids):
                for j, det in enumerate(detections):
                    cost[i, j] = 1.0 - iou(predicted[tid], det)
            row_idx, col_idx = linear_sum_assignment(cost)
            for i, j in zip(row_idx, col_idx):
                if 1.0 - cost[i, j] < self.iou_gate:  # IoU below gate -> reject, both unmatched
                    continue
                tid = matchable_ids[i]
                assigned_track_for_det[j] = tid
                used_dets.add(j)

        # -- 3. Remaining detections vs TENTATIVE tracks: simple greedy
        #       IoU (no velocity model yet -- not enough history). --------
        tentative_ids = [tid for tid, t in self._tracks.items() if t.state == TrackState.TENTATIVE]
        if tentative_ids:
            candidates = []
            for tid in tentative_ids:
                track_bbox = self._tracks[tid].bbox
                for j, det in enumerate(detections):
                    if j in used_dets:
                        continue
                    score = iou(track_bbox, det)
                    if score >= self.iou_gate:
                        candidates.append((score, tid, j))
            candidates.sort(key=lambda c: c[0], reverse=True)
            used_tentative: set = set()
            for score, tid, j in candidates:
                if tid in used_tentative or j in used_dets:
                    continue
                assigned_track_for_det[j] = tid
                used_dets.add(j)
                used_tentative.add(tid)

        # -- 4. Apply matches: refresh matched tracks (bbox/velocity/hits/
        #       state); figure out returned ids. --------------------------
        results: List[Optional[int]] = [None] * n_dets
        matched_track_ids: set = set()
        for j, tid in enumerate(assigned_track_for_det):
            if tid is None:
                continue
            track = self._tracks[tid]
            matched_track_ids.add(tid)
            self._apply_match(track, detections[j])
            results[j] = tid if track.state == TrackState.CONFIRMED else None

        # -- 5. Genuinely-unmatched detections -> brand new TENTATIVE
        #       tracks (confirmed immediately if n_confirm <= 1). ---------
        for j in range(n_dets):
            if assigned_track_for_det[j] is not None:
                continue
            new_id = self._next_id
            self._next_id += 1
            new_track = _Track(bbox=tuple(detections[j]))
            self._tracks[new_id] = new_track
            matched_track_ids.add(new_id)
            if new_track.hits >= self.n_confirm:
                new_track.state = TrackState.CONFIRMED
                results[j] = new_id

        # -- 6. Unmatched existing tracks: CONFIRMED -> LOST; LOST stays
        #       LOST unless past t_lost_seconds; TENTATIVE deleted
        #       immediately. ------------------------------------------------
        for tid in list(self._tracks.keys()):
            if tid in matched_track_ids:
                continue
            track = self._tracks[tid]
            if track.state == TrackState.TENTATIVE:
                del self._tracks[tid]
            elif track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST
                track.lost_seconds = 0.0  # just went missing this tick
            elif track.state == TrackState.LOST:
                if track.lost_seconds >= self.t_lost_seconds:
                    del self._tracks[tid]
                # else: remains LOST, already aged in step 1.

        return results

    def _apply_match(self, track: _Track, detection: BBox) -> None:
        old_cx, old_cy = _center(track.bbox)
        new_cx, new_cy = _center(detection)
        elapsed = max(track.time_since_update, 1e-6)
        raw_vx = (new_cx - old_cx) / elapsed
        raw_vy = (new_cy - old_cy) / elapsed
        old_vx, old_vy = track.velocity
        track.velocity = (
            self.velocity_alpha * raw_vx + (1.0 - self.velocity_alpha) * old_vx,
            self.velocity_alpha * raw_vy + (1.0 - self.velocity_alpha) * old_vy,
        )
        track.bbox = tuple(detection)
        track.time_since_update = 0.0
        track.lost_seconds = 0.0

        if track.state == TrackState.TENTATIVE:
            track.hits += 1
            if track.hits >= self.n_confirm:
                track.state = TrackState.CONFIRMED
        elif track.state == TrackState.LOST:
            track.state = TrackState.CONFIRMED

    def lost_tracks(self) -> Dict[int, Tuple[float, float, float, float]]:
        """Currently-LOST track ids -> their constant-velocity PREDICTED
        bbox (not the frozen last-matched bbox). Purely additive -- does
        not touch update()'s existing signature/behavior/return contract."""
        return {
            tid: t.predicted_bbox()
            for tid, t in self._tracks.items()
            if t.state == TrackState.LOST
        }

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0
