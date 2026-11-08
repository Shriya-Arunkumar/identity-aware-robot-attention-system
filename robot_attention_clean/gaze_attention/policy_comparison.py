"""
policy_comparison.py — live current-vs-baseline target comparison.

Mirrors gaze_attention/filter_graph.py's existing pattern exactly (see that
module's docstring): a pure-data class with no GUI import, a separate
matplotlib visualization layer that degrades to headless CSV-only mode on
any failure, rendered via plt.ion() on the main thread with throttled
non-blocking redraws.

Purely observational. This module NEVER constructs a RosPublisher, never
calls ros_pub.publish, and never receives one -- there is no code path by
which the baseline policy's target can reach ROS/Gazebo. main_webcam.py's
existing `ros_pub.publish(target, ...)` call site is untouched; `target`
there always comes from the ACTIVE policy (ConversationalSimplePolicy by
default), never from anything in this module. See
tests/test_policy_comparison.py's ROS-isolation tests.

Three pieces:
  * `SwitchCounter` -- pure logic, no state beyond the previous target.
    A switch is counted only when the previous target was NOT None and the
    new target differs from it (this also counts a target->None transition
    as a switch, since the target WAS being tracked and then wasn't; the
    only transition that is never counted is the very first assignment,
    None->something, which is an initial pick, not a "switch").
  * `TargetHistory` -- pure data buffers (time/current/baseline target
    id), plus the person-id -> y-axis-position mapping. No matplotlib
    import, so this half is trivially unit-testable without a display.
  * `PolicyComparisonGraph` -- the visualization/logging layer. Always
    writes a CSV row per tick (own file, independent of SessionLogger/
    JsonlLogger -- neither existing logger's schema is touched). Renders a
    step plot (target IDs are discrete categories, not continuous values)
    of both target histories on ONE shared, dynamically-updated y-axis
    (person id -> position), with a None target left as a genuine gap in
    the line (plotted as NaN, which matplotlib does not draw a segment
    through) rather than ever being coerced into a valid person id like 0.
"""

from __future__ import annotations

import csv
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

import cv2

from gaze_attention.visualization import BLACK, WHITE, draw_faces

_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def draw_comparison_hud(
    frame,
    label: str,
    policy_name: str,
    target: Optional[int],
    speaker_id: Optional[int],
    switch_count: int,
    ros_active: bool,
) -> None:
    """Draws the exact required panel text block (see the comparison-mode
    spec: Policy / Target / Speaker / Target switches / ROS status) along
    the BOTTOM of the frame, in place -- deliberately not touching
    visualization.py's own top HUD bar (draw_hud occupies rows 0-48px on
    the current-model panel), so this is purely additive and never changes
    existing overlay semantics. Used for both panels; only `ros_active`
    differs (True only for the current-model panel -- see module
    docstring: this function never implies the baseline controls
    ROS/Gazebo)."""
    h, w = frame.shape[:2]
    lines = [
        label,
        f"Policy: {policy_name}",
        f"Target: {target if target is not None else 'None'}",
        f"Speaker: {speaker_id if speaker_id is not None else 'None'}",
        f"Target switches: {switch_count}",
        f"ROS/Gazebo: {'ACTIVE' if ros_active else 'DISABLED'}",
    ]
    bar_h = 18 * len(lines) + 10
    cv2.rectangle(frame, (0, h - bar_h), (w, h), BLACK, -1)
    for i, text in enumerate(lines):
        color = (0, 200, 0) if (i == 5 and ros_active) else WHITE
        cv2.putText(frame, text, (8, h - bar_h + 18 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_baseline_panel(
    frame,
    observations: List[dict],
    metrics_by_id: Dict[int, dict],
    baseline_target: Optional[int],
    confidence_threshold: float,
    baseline_speaker_id: Optional[int],
    switch_count: int,
) -> None:
    """Baseline panel: reuses draw_faces() UNCHANGED (same box-coloring
    semantics as the current panel -- green = this panel's own selected
    target, per-panel, never implying cross-panel agreement), then adds the
    comparison HUD with ROS/Gazebo always DISABLED -- the baseline never
    controls ROS/Gazebo, and this label makes that visible on-screen."""
    draw_faces(frame, observations, metrics_by_id, baseline_target, confidence_threshold)
    draw_comparison_hud(
        frame, "BASELINE (view only)", "ConfidenceDebtPolicy",
        baseline_target, baseline_speaker_id, switch_count, ros_active=False,
    )


class SwitchCounter:
    """Counts target switches under one fixed, documented rule (see module
    docstring): the initial None->target assignment never counts; every
    other change away from a non-None previous target does, including a
    transition to None (the target being lost)."""

    def __init__(self) -> None:
        self.count: int = 0
        self._previous: Optional[int] = None
        self._seen_any: bool = False

    def update(self, target: Optional[int]) -> bool:
        """Feed the current tick's target. Returns True iff this tick was
        counted as a switch."""
        switched = False
        if self._seen_any and self._previous is not None and target != self._previous:
            self.count += 1
            switched = True
        self._previous = target
        self._seen_any = True
        return switched

    def reset(self) -> None:
        self.count = 0
        self._previous = None
        self._seen_any = False


class TargetHistory:
    """Pure data: elapsed-time-indexed history of both policies' targets,
    plus the person-id -> y-axis-position mapping used to plot discrete
    IDs on a numeric axis without ever inventing a fake id. No GUI import."""

    def __init__(self) -> None:
        self.time_history: List[float] = []
        self.current_target_history: List[Optional[int]] = []
        self.baseline_target_history: List[Optional[int]] = []
        # First-seen order -- stable, and matches the order ids naturally
        # appear in a session, e.g. {12: 0, 18: 1, 25: 2}.
        self._id_to_pos: Dict[int, int] = {}

    def append(self, elapsed_time: float, current_target: Optional[int], baseline_target: Optional[int]) -> None:
        self.time_history.append(elapsed_time)
        self.current_target_history.append(current_target)
        self.baseline_target_history.append(baseline_target)
        for pid in (current_target, baseline_target):
            if pid is not None and pid not in self._id_to_pos:
                self._id_to_pos[pid] = len(self._id_to_pos)

    def position_of(self, pid: Optional[int]) -> float:
        """Numeric y-position for a person id, or NaN for None -- NaN is
        the explicit "gap in the line" marker; it must never be confused
        with a real position (position 0 is a real, valid person id)."""
        if pid is None:
            return math.nan
        return float(self._id_to_pos.get(pid, math.nan))

    def y_ticks_and_labels(self) -> tuple:
        """(positions, labels) for every person id seen so far, e.g.
        ([0, 1, 2], ["Person 12", "Person 18", "Person 25"])."""
        items = sorted(self._id_to_pos.items(), key=lambda kv: kv[1])
        positions = [pos for _pid, pos in items]
        labels = [f"Person {pid}" for pid, _pos in items]
        return positions, labels

    def current_positions(self) -> List[float]:
        return [self.position_of(pid) for pid in self.current_target_history]

    def baseline_positions(self) -> List[float]:
        return [self.position_of(pid) for pid in self.baseline_target_history]


CSV_FIELDS = [
    "timestamp", "elapsed_time", "current_target_id", "baseline_target_id",
    "current_switch_count", "baseline_switch_count",
    "confirmed_speaker_id", "baseline_confirmed_speaker_id",
]


def _new_csv_path(logs_dir: Optional[Path] = None) -> Path:
    logs_dir = logs_dir or _LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"policy_comparison_{stamp}.csv"
    suffix = 1
    while path.exists():
        path = logs_dir / f"policy_comparison_{stamp}_{suffix}.csv"
        suffix += 1
    return path


class PolicyComparisonGraph:
    """The visualization/logging layer. Always writes CSV; optionally also
    renders a live step-plot via matplotlib's interactive mode on the MAIN
    thread (same justification as filter_graph.LiveFilterGraph: a real
    background thread fails with TkAgg). `push()` is called once per tick
    from the main OpenCV loop; a plotting failure degrades to headless
    CSV-only mode and never raises out to the caller."""

    REDRAW_INTERVAL_SECONDS = 0.1

    def __init__(self, headless: bool = False, logs_dir: Optional[Path] = None):
        self.headless = headless
        self.history = TargetHistory()

        self.csv_path = _new_csv_path(logs_dir)
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
        self._csv_writer.writeheader()

        self._last_redraw_time = 0.0
        self._fig = None
        self._window_open = True

        if not headless:
            try:
                self._init_graph()
            except Exception as exc:  # noqa: BLE001 - any GUI failure must degrade gracefully
                print(f"[policy_comparison] live graph unavailable ({exc}) -- falling back to headless CSV-only mode.")
                self.headless = True

    # -- Public API --------------------------------------------------------
    def push(
        self,
        elapsed_time: float,
        current_target: Optional[int],
        baseline_target: Optional[int],
        current_switch_count: int,
        baseline_switch_count: int,
        confirmed_speaker_id: Optional[int],
        baseline_confirmed_speaker_id: Optional[int],
    ) -> None:
        self.history.append(elapsed_time, current_target, baseline_target)

        row = {
            "timestamp": time.time(),
            "elapsed_time": round(elapsed_time, 4),
            "current_target_id": current_target if current_target is not None else "",
            "baseline_target_id": baseline_target if baseline_target is not None else "",
            "current_switch_count": current_switch_count,
            "baseline_switch_count": baseline_switch_count,
            "confirmed_speaker_id": confirmed_speaker_id if confirmed_speaker_id is not None else "",
            "baseline_confirmed_speaker_id": baseline_confirmed_speaker_id if baseline_confirmed_speaker_id is not None else "",
        }
        if not self._csv_file.closed:
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        if not self.headless and self._window_open:
            now = time.monotonic()
            if now - self._last_redraw_time >= self.REDRAW_INTERVAL_SECONDS:
                self._last_redraw_time = now
                self._redraw()

    def close(self) -> None:
        if not self._csv_file.closed:
            self._csv_file.close()
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._fig)
            except Exception:  # noqa: BLE001 - closing must never raise during shutdown
                pass

    # -- Main-thread interactive graph --------------------------------------
    def _init_graph(self) -> None:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.canvas.manager.set_window_title("Target ID over time")
        (line_current,) = ax.step([], [], where="post", label="Current model", color="tab:blue", linewidth=2)
        (line_baseline,) = ax.step([], [], where="post", label="Baseline model", color="tab:orange", linewidth=2, linestyle="--")
        ax.set_title("Target ID over time")
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Person / Speaker ID")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        def on_close(_event):
            self._window_open = False

        fig.canvas.mpl_connect("close_event", on_close)

        fig.show()
        fig.canvas.draw()
        fig.canvas.flush_events()

        self._fig, self._ax = fig, ax
        self._line_current, self._line_baseline = line_current, line_baseline

    def _redraw(self) -> None:
        h = self.history
        if not h.time_history:
            return
        try:
            xs = h.time_history
            self._line_current.set_data(xs, h.current_positions())
            self._line_baseline.set_data(xs, h.baseline_positions())

            positions, labels = h.y_ticks_and_labels()
            if positions:
                self._ax.set_yticks(positions)
                self._ax.set_yticklabels(labels)
                self._ax.set_ylim(min(positions) - 0.5, max(positions) + 0.5)

            self._ax.set_xlim(0, max(xs[-1], 1.0))

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception as exc:  # noqa: BLE001 - a closed/dead window must not crash the demo
            print(f"[policy_comparison] graph redraw failed ({exc}) -- switching to headless CSV-only mode.")
            self.headless = True
            self._window_open = False
