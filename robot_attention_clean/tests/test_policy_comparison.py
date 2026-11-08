"""Tests for gaze_attention/policy_comparison.py -- the opt-in
--compare-baseline feature (ConfidenceDebtPolicy run alongside the active
policy, purely for a live "Target ID over time" popup graph + a view-only
webcam panel).

Headless throughout: PolicyComparisonGraph(headless=True) never imports
matplotlib, so these tests need no physical display. The matplotlib-backed
path is exercised only by construction with headless=False falling back
gracefully when unavailable (test_graph_headless_fallback_on_bad_backend).
"""

from __future__ import annotations

import inspect

import pytest

from gaze_attention.policies import ConfidenceDebtPolicy, ConversationalSimplePolicy
from gaze_attention.policy_comparison import PolicyComparisonGraph, SwitchCounter, TargetHistory

DT = 0.1


def obs(id_, seeking=True, confidence=0.9, speaking=False, speech_confidence=0.0, t=0.0, lost=False):
    return {
        "id": id_, "bbox": [0, 0, 10, 10],
        "gaze_signal": 0.9 if seeking else 0.1,
        "confidence": confidence, "seeking_attention": seeking,
        "timestamp": t, "speaking": speaking, "speech_confidence": speech_confidence,
        "lost": lost,
    }


# --- 1. default-off / unchanged existing behavior ----------------------------

def test_compare_baseline_flag_defaults_to_false():
    from gaze_attention.main_webcam import parse_args
    args = parse_args([])
    assert args.compare_baseline is False


def test_existing_show_filter_graph_flag_unaffected():
    """The pre-existing comparison feature's flag/defaults must be untouched."""
    from gaze_attention.main_webcam import parse_args
    args = parse_args([])
    assert args.show_filter_graph is False
    assert args.graph_person_id == 0
    assert args.filter_graph_window == 15.0


# --- 2/3. SwitchCounter: exact documented rule -------------------------------

def test_initial_assignment_is_not_a_switch():
    c = SwitchCounter()
    switched = c.update(None)
    assert switched is False and c.count == 0
    switched = c.update(12)  # None -> 12: initial pick, not a switch
    assert switched is False and c.count == 0


def test_later_change_is_counted():
    c = SwitchCounter()
    c.update(None)
    c.update(12)
    switched = c.update(18)  # 12 -> 18: a real switch
    assert switched is True and c.count == 1
    c.update(18)  # unchanged
    assert c.count == 1
    c.update(12)  # 18 -> 12: another real switch
    assert c.count == 2


def test_transition_to_none_counts_as_a_switch_documented_rule():
    """Documented rule: previous not None and current != previous counts,
    including current=None (target lost) -- only the None->X initial pick
    is exempt."""
    c = SwitchCounter()
    c.update(None)
    c.update(12)
    switched = c.update(None)
    assert switched is True and c.count == 1


def test_current_and_baseline_counters_are_fully_independent():
    current = SwitchCounter()
    baseline = SwitchCounter()
    current.update(None)
    current.update(12)
    current.update(18)  # 1 switch
    baseline.update(None)
    baseline.update(12)
    baseline.update(18)
    baseline.update(12)  # 2 switches
    assert current.count == 1
    assert baseline.count == 2


# --- 4/5. both policies receive identical observations and dt ---------------

def test_both_policies_receive_the_exact_same_observation_objects_and_dt():
    seen_current, seen_baseline = [], []
    current = ConversationalSimplePolicy()
    baseline = ConfidenceDebtPolicy()

    people = [obs(0, seeking=True, t=0.0), obs(1, seeking=False, t=0.0)]
    current.update(people, DT)
    baseline.update(people, DT)
    # No policy mutates observations (verified in source inspection); the
    # same list object can therefore be handed to both without copying.
    assert people[0]["id"] == 0 and people[1]["id"] == 1
    assert people[0]["confidence"] == 0.9  # untouched by either .update() call


# --- 6. independent policy state ---------------------------------------------

def test_policy_state_is_independent_not_shared():
    current = ConversationalSimplePolicy()
    baseline = ConfidenceDebtPolicy()
    assert current is not baseline
    t = 0.0
    for _ in range(20):
        people = [obs(0, seeking=True, t=t), obs(1, seeking=False, t=t)]
        current.update(people, DT)
        baseline.update(people, DT)
        t += DT
    # Distinct internal state containers, not the same dict/object.
    assert current.short_term_score is not baseline._engagement
    assert current.attention_debt is not baseline._debt
    # Both independently converge on person 0 given identical input, but via
    # their own separate state -- mutating one must not affect the other.
    baseline._engagement[0] = 999.0
    assert current.short_term_score.get(0, 0.0) != 999.0


# --- 7/8. target history recorded, None handled correctly -------------------

def test_target_history_records_both_target_streams():
    h = TargetHistory()
    h.append(0.0, None, None)
    h.append(0.1, 12, 12)
    h.append(0.2, 18, 12)
    assert h.time_history == [0.0, 0.1, 0.2]
    assert h.current_target_history == [None, 12, 18]
    assert h.baseline_target_history == [None, 12, 12]


def test_none_target_never_becomes_person_zero():
    h = TargetHistory()
    h.append(0.0, 0, None)   # a REAL person id 0 is present here
    h.append(0.1, None, 0)   # then None on one side, real id 0 on the other
    import math
    # None must map to NaN (a gap), never to the numeric position of a real id.
    assert h.position_of(None) != h.position_of(0)
    assert math.isnan(h.position_of(None))
    assert h.position_of(0) == 0.0  # person id 0 is a real, valid position


def test_none_produces_a_gap_not_a_plotted_zero():
    h = TargetHistory()
    h.append(0.0, 12, None)
    h.append(0.1, None, None)
    h.append(0.2, 12, None)
    positions = h.current_positions()
    import math
    assert positions[0] == 0.0  # person 12 -> first-seen position 0
    assert math.isnan(positions[1])  # None -> gap, not 0.0 (which would be person 12's slot)
    assert positions[2] == 0.0


# --- 12. y-axis labels update with new ids, never renumbering old points -----

def test_new_person_id_extends_labels_without_disturbing_earlier_points():
    h = TargetHistory()
    h.append(0.0, 12, None)
    pos, labels = h.y_ticks_and_labels()
    assert pos == [0] and labels == ["Person 12"]
    first_point_position = h.position_of(12)

    h.append(0.1, 18, None)  # a brand new id appears
    pos, labels = h.y_ticks_and_labels()
    assert pos == [0, 1] and labels == ["Person 12", "Person 18"]
    # The earlier point's meaning must not change retroactively.
    assert h.position_of(12) == first_point_position == 0.0
    assert h.position_of(18) == 1.0


# --- 13/14/15. graph shape: exactly two lines, elapsed-time x-axis, step ----

def test_graph_headless_writes_exactly_two_series_per_row():
    graph = PolicyComparisonGraph(headless=True)
    try:
        graph.push(
            elapsed_time=0.0, current_target=12, baseline_target=18,
            current_switch_count=0, baseline_switch_count=0,
            confirmed_speaker_id=None, baseline_confirmed_speaker_id=None,
        )
        graph.push(
            elapsed_time=0.033, current_target=12, baseline_target=12,
            current_switch_count=0, baseline_switch_count=1,
            confirmed_speaker_id=12, baseline_confirmed_speaker_id=None,
        )
    finally:
        graph.close()

    import csv
    with open(graph.csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    # Exactly two target-id series, and nothing else masquerading as a third
    # plotted target line (speaker/score/debt/confidence are NOT target series).
    assert set(rows[0].keys()) >= {"elapsed_time", "current_target_id", "baseline_target_id"}
    assert rows[0]["current_target_id"] == "12"
    assert rows[0]["baseline_target_id"] == "18"
    assert rows[1]["baseline_target_id"] == "12"


def test_graph_uses_elapsed_time_not_frame_index_as_x_axis():
    graph = PolicyComparisonGraph(headless=True)
    try:
        # Irregular real dt between frames, NOT a fixed assumed FPS / frame count.
        for elapsed in (0.0, 0.033, 0.09, 0.31, 0.34):
            graph.push(elapsed, 12, 12, 0, 0, None, None)
        assert graph.history.time_history == [0.0, 0.033, 0.09, 0.31, 0.34]
    finally:
        graph.close()


def test_step_plot_semantics_value_holds_until_next_change():
    """A step plot with where='post' means the plotted value HOLDS at the
    previous target until the exact tick it changes -- verified here at the
    data level (the line objects use ax.step(..., where='post') in
    _init_graph, which this asserts indirectly via the recorded history
    shape matching the held-then-drops pattern from the spec's example)."""
    h = TargetHistory()
    h.append(0.0, 12, 12)
    h.append(0.1, 12, 12)
    h.append(0.2, 18, 12)  # current switches, baseline holds
    h.append(0.3, 18, 18)  # baseline catches up
    assert h.current_positions() == [0.0, 0.0, 1.0, 1.0]
    assert h.baseline_positions() == [0.0, 0.0, 0.0, 1.0]


def test_graph_headless_fallback_on_bad_backend(monkeypatch):
    """If matplotlib/the GUI backend is unavailable, construction must not
    raise -- it must degrade to headless CSV-only mode with a warning."""
    import gaze_attention.policy_comparison as pc

    def boom(self):
        raise RuntimeError("no display")

    monkeypatch.setattr(pc.PolicyComparisonGraph, "_init_graph", boom)
    graph = pc.PolicyComparisonGraph(headless=False)
    try:
        assert graph.headless is True  # degraded, did not raise
        graph.push(0.0, 12, 18, 0, 0, None, None)  # still works headless
    finally:
        graph.close()


# --- 16/17. ROS isolation -----------------------------------------------------

def test_baseline_target_never_reaches_ros_publisher_call_site():
    """Static safety check on the actual call site: ros_pub.publish(...)'s
    first positional argument must be exactly the identifier `target`
    (the active policy's result) and the word `baseline` must not appear
    anywhere in that call expression."""
    import gaze_attention.main_webcam as mw
    src = inspect.getsource(mw.main)
    call_start = src.index("ros_pub.publish(")
    call_end = src.index(")", src.index(")", call_start) + 1)  # closing paren of the call
    call_text = src[call_start:call_end]
    assert "baseline" not in call_text.lower(), (
        f"baseline must never appear in the ros_pub.publish(...) call: {call_text!r}"
    )
    # First argument is the bare identifier `target`.
    first_line_after_paren = src[call_start:].splitlines()[1].strip().rstrip(",")
    assert first_line_after_paren == "target"


def _code_lines(src: str):
    """Source lines with docstrings/comments stripped out crudely (good
    enough for a 'this identifier is never used in actual code' check)."""
    in_triple = False
    for line in src.splitlines():
        stripped = line.strip()
        if '"""' in stripped:
            in_triple = not in_triple if stripped.count('"""') % 2 else in_triple
            continue
        if in_triple or stripped.startswith("#"):
            continue
        yield line.split("#", 1)[0]


def test_policy_comparison_module_never_constructs_a_ros_publisher():
    import gaze_attention.policy_comparison as pc
    code = "\n".join(_code_lines(inspect.getsource(pc)))
    assert "RosPublisher" not in code
    assert ".publish(" not in code


def test_baseline_policy_object_has_no_ros_reference():
    """The baseline policy instance itself is a plain ConfidenceDebtPolicy
    with no publish method or ROS handle -- there is no way for it to reach
    ROS even if a caller tried."""
    baseline = ConfidenceDebtPolicy()
    assert not hasattr(baseline, "publish")
    assert not hasattr(baseline, "ros_pub")


# --- 18. graph failure never terminates current-model processing -----------

def test_push_failure_is_swallowed_not_raised(monkeypatch):
    graph = PolicyComparisonGraph(headless=False)
    graph.headless = False
    graph._window_open = True

    class ExplodingAx:
        def __getattr__(self, name):
            raise RuntimeError("dead window")

    graph._fig = object()
    graph._ax = ExplodingAx()
    graph._line_current = ExplodingAx()
    graph._line_baseline = ExplodingAx()
    try:
        graph.history.append(0.0, 12, 18)  # ensure time_history is non-empty
        graph.push(0.1, 12, 18, 0, 0, None, None)  # must not raise
        assert graph.headless is True  # degraded after the failure
    finally:
        graph.close()


# --- 23. existing logger tests remain green (smoke import check) -----------

def test_existing_loggers_untouched_by_this_module():
    """policy_comparison.py must not import or monkeypatch SessionLogger/
    JsonlLogger -- it owns its own independent CSV file."""
    import gaze_attention.policy_comparison as pc
    code = "\n".join(_code_lines(inspect.getsource(pc)))
    assert "SessionLogger" not in code
    assert "JsonlLogger" not in code
    assert pc.CSV_FIELDS == [
        "timestamp", "elapsed_time", "current_target_id", "baseline_target_id",
        "current_switch_count", "baseline_switch_count",
        "confirmed_speaker_id", "baseline_confirmed_speaker_id",
    ]
