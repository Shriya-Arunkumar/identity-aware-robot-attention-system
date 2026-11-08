"""
filter_graph.py — live filter-comparison visualization for one selected
tracked person. Purely observational: runs its OWN separate
MovingAverageFilter/BayesianFilter instances, fed the same raw
(pre-filter) observations the real pipeline sees, and never touches the
production `filt` object or `policy_observations` main_webcam.py builds
for the actual policy. Enabling/disabling this feature cannot change
target selection — see tests/test_filter_graph.py's
`test_graph_mode_does_not_change_policy_output`.

Two pieces:
  * `FilterComparisonTracker` — pure data, no GUI/matplotlib import. Owns
    one MovingAverageFilter and one BayesianFilter (this project's
    remaining active filters — Kalman was removed from scope, see
    README.md "Filter scope"), each already per-person-stateful via their
    own `filters.py` implementation, so multiple tracked people's state
    stays independent automatically. Kalman is not part of this
    comparison since it no longer exists in this codebase.
  * `LiveFilterGraph` — the visualization/logging layer. ALWAYS writes a
    CSV row per tick for the graphed person (the "headless mode" from the
    spec is simply what you get when the live graph can't or shouldn't
    open — CSV writing never depends on it). Renders via matplotlib's
    interactive mode (`plt.ion()`) on the MAIN thread, redrawn from
    `push()` calls already happening every OpenCV frame -- a genuine
    background thread was tried first and rejected: matplotlib itself
    raises "Starting a Matplotlib GUI outside of the main thread will
    likely fail" (confirmed on this machine's TkAgg backend), so the
    officially-supported non-blocking-animation form of this project's
    "separate thread OR non-blocking animation" requirement is used
    instead. Redraws are throttled (~10 Hz) so the OpenCV loop's frame
    rate is barely affected. If figure creation fails for any reason (no
    display, backend unavailable, `--no-display`), it prints one clear
    message and continues in CSV-only mode — per this project's "every
    component needs a fallback" constraint.

Confidence note (requirement 6): this module never plots or logs
`filter_uncertainty` as if it were a gaze-correctness probability. Only
filtered SIGNAL values (raw/moving_average/bayesian) are graphed. Each
filter's uncertainty output is discarded here (see `FilterComparisonTracker.
update`) precisely to avoid the temptation to mislabel it — see
DEBUG_ATTENTION.md FM1 for why that distinction matters.
"""

from __future__ import annotations

import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Optional

from gaze_attention.filters import BayesianFilter, MovingAverageFilter

_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


class FilterComparisonTracker:
    """Runs moving-average and Bayesian filters in parallel, independent of
    the production pipeline, purely for comparison. Keeps separate internal
    state per tracked person (each filters.py filter already does this)."""

    def __init__(self):
        self._filters = {
            "moving_average": MovingAverageFilter(),
            "bayesian": BayesianFilter(),
        }
        self._known_ids: set = set()

    def update(self, raw_observations: list, dt: float) -> Dict[int, Dict[str, float]]:
        """`raw_observations`: perception.py-style dicts with 'id',
        'gaze_signal' (raw), 'confidence' (raw). Returns
        {person_id: {"raw": float, "moving_average": float, "bayesian": float}}
        for every person present this tick. Both filters receive the exact
        same (raw_signal, raw_confidence, dt) triple, so their outputs are
        directly comparable."""
        current_ids = {obs["id"] for obs in raw_observations}
        for gone_id in self._known_ids - current_ids:
            for f in self._filters.values():
                f.forget(gone_id)
        self._known_ids = current_ids

        results: Dict[int, Dict[str, float]] = {}
        for obs in raw_observations:
            pid = obs["id"]
            raw_signal = obs["gaze_signal"]
            raw_confidence = obs["confidence"]
            row = {"raw": raw_signal}
            for name, f in self._filters.items():
                filtered_signal, _uncertainty = f.update(pid, raw_signal, raw_confidence, dt)
                row[name] = filtered_signal
            results[pid] = row
        return results

    def reset(self) -> None:
        for f in self._filters.values():
            f.reset()
        self._known_ids.clear()


FILTER_NAMES = ("moving_average", "bayesian")
CSV_FIELDS = [
    "timestamp", "person_id", "raw", "moving_average", "bayesian",
    "flip_count_moving_average", "flip_count_bayesian",
    "flip_rate_moving_average", "flip_rate_bayesian",
    "current_target", "fps",
]


def _new_csv_path(logs_dir: Optional[Path] = None) -> Path:
    logs_dir = logs_dir or _LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"filter_graph_{stamp}.csv"
    suffix = 1
    while path.exists():
        path = logs_dir / f"filter_graph_{stamp}_{suffix}.csv"
        suffix += 1
    return path


class LiveFilterGraph:
    """The visualization/logging layer for one graphed person. Always
    writes CSV; optionally also renders a scrolling matplotlib graph via
    interactive mode on the MAIN thread (see module docstring for why not
    a background thread). `push()` is called once per tick from the main
    OpenCV loop; it always writes one CSV row (cheap), and -- when not
    headless -- redraws the figure at a throttled rate (~10 Hz) via
    `fig.canvas.draw_idle()` + `flush_events()`, which return quickly and
    keep the OpenCV loop responsive."""

    REDRAW_INTERVAL_SECONDS = 0.1

    def __init__(
        self,
        person_id: int,
        seeking_threshold: float,
        window_seconds: float = 15.0,
        headless: bool = False,
        logs_dir: Optional[Path] = None,
    ):
        self.person_id = person_id
        self.seeking_threshold = seeking_threshold
        self.window_seconds = window_seconds
        self.headless = headless

        self.csv_path = _new_csv_path(logs_dir)
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
        self._csv_writer.writeheader()

        # Flip tracking: a "flip" is (filtered_signal >= seeking_threshold)
        # toggling from the previous tick, per filter. Rolling event
        # timestamps pruned to window_seconds give both a windowed count
        # and an estimated rate; a separate all-time count is kept too.
        self._last_seeking: Dict[str, Optional[bool]] = {name: None for name in FILTER_NAMES}
        self._flip_events: Dict[str, Deque[float]] = {name: deque() for name in FILTER_NAMES}
        self._flip_count_total: Dict[str, int] = {name: 0 for name in FILTER_NAMES}

        # Rolling plot buffers (main thread only; unused in headless mode).
        self._times: Deque[float] = deque()
        self._raw_vals: Deque[float] = deque()
        self._ma_vals: Deque[float] = deque()
        self._bayes_vals: Deque[float] = deque()
        self._latest_row: dict = {}
        self._last_redraw_time = 0.0
        self._fig = None
        self._window_open = True

        if not headless:
            try:
                self._init_graph()
            except Exception as exc:  # noqa: BLE001 - any GUI failure must degrade gracefully
                print(f"[filter_graph] live graph unavailable ({exc}) -- falling back to headless CSV-only mode.")
                self.headless = True

    # -- Public API ------------------------------------------------------------
    def push(
        self,
        timestamp: float,
        values: Dict[str, float],  # {"raw":..., "moving_average":..., "bayesian":...}
        current_target: Optional[int],
        fps: float,
    ) -> None:
        flip_counts = {}
        flip_rates = {}
        for name in FILTER_NAMES:
            filtered = values.get(name)
            if filtered is None:
                flip_counts[name] = self._flip_count_total[name]
                flip_rates[name] = 0.0
                continue
            seeking = filtered >= self.seeking_threshold
            prev = self._last_seeking[name]
            if prev is not None and seeking != prev:
                self._flip_count_total[name] += 1
                self._flip_events[name].append(timestamp)
            self._last_seeking[name] = seeking

            events = self._flip_events[name]
            while events and timestamp - events[0] > self.window_seconds:
                events.popleft()
            flip_counts[name] = self._flip_count_total[name]
            flip_rates[name] = len(events) / self.window_seconds if self.window_seconds > 0 else 0.0

        row = {
            "timestamp": timestamp,
            "person_id": self.person_id,
            "raw": values.get("raw", ""),
            "moving_average": values.get("moving_average", ""),
            "bayesian": values.get("bayesian", ""),
            "flip_count_moving_average": flip_counts["moving_average"],
            "flip_count_bayesian": flip_counts["bayesian"],
            "flip_rate_moving_average": round(flip_rates["moving_average"], 4),
            "flip_rate_bayesian": round(flip_rates["bayesian"], 4),
            "current_target": current_target,
            "fps": round(fps, 2),
        }
        if not self._csv_file.closed:
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        if not self.headless and self._window_open:
            self._latest_row = row
            self._times.append(timestamp)
            self._raw_vals.append(float(values.get("raw", 0.0) or 0.0))
            self._ma_vals.append(float(values.get("moving_average", 0.0) or 0.0))
            self._bayes_vals.append(float(values.get("bayesian", 0.0) or 0.0))
            while self._times and timestamp - self._times[0] > self.window_seconds:
                self._times.popleft()
                self._raw_vals.popleft()
                self._ma_vals.popleft()
                self._bayes_vals.popleft()

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

    # -- Main-thread interactive graph ------------------------------------------
    def _init_graph(self) -> None:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]})
        fig.canvas.manager.set_window_title(f"Filter comparison -- person {self.person_id}")
        (line_raw,) = ax1.plot([], [], label="raw", color="grey", alpha=0.6, linewidth=1)
        (line_ma,) = ax1.plot([], [], label="moving_average", color="tab:blue")
        (line_bayes,) = ax1.plot([], [], label="bayesian", color="tab:orange")
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlim(-self.window_seconds, 0)
        ax1.set_ylabel("gaze/engagement signal")
        ax1.set_xlabel("seconds ago")
        ax1.legend(loc="upper right")
        ax1.set_title(f"person {self.person_id} -- last {self.window_seconds:.0f}s")

        ax2.axis("off")
        text = ax2.text(0.02, 0.5, "", va="center", family="monospace", fontsize=10)

        def on_close(_event):
            self._window_open = False

        fig.canvas.mpl_connect("close_event", on_close)

        fig.show()
        fig.canvas.draw()
        fig.canvas.flush_events()

        self._fig, self._ax1, self._ax2 = fig, ax1, ax2
        self._line_raw, self._line_ma, self._line_bayes, self._text = line_raw, line_ma, line_bayes, text

    def _redraw(self) -> None:
        if not self._times:
            return
        t0 = self._times[-1]
        xs = [t - t0 for t in self._times]
        self._line_raw.set_data(xs, self._raw_vals)
        self._line_ma.set_data(xs, self._ma_vals)
        self._line_bayes.set_data(xs, self._bayes_vals)

        row = self._latest_row
        self._text.set_text(
            f"flip count   MA={row.get('flip_count_moving_average', 0):<4d} "
            f"Bayes={row.get('flip_count_bayesian', 0)}\n"
            f"flip rate/s  MA={row.get('flip_rate_moving_average', 0):<6.2f} "
            f"Bayes={row.get('flip_rate_bayesian', 0):.2f}\n"
            f"current      raw={float(row.get('raw', 0) or 0):.2f}  "
            f"MA={float(row.get('moving_average', 0) or 0):.2f}  "
            f"Bayes={float(row.get('bayesian', 0) or 0):.2f}\n"
            f"target={row.get('current_target')}   fps={row.get('fps', 0):.1f}"
        )

        try:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception as exc:  # noqa: BLE001 - a closed/dead window must not crash the demo
            print(f"[filter_graph] graph redraw failed ({exc}) -- switching to headless CSV-only mode.")
            self.headless = True
            self._window_open = False
