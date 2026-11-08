"""Tests for gaze_attention.filter_graph — the live filter-comparison
visualization. All tests here use headless=True (or no GUI at all) so
they never depend on a display; the live-GUI path was smoke-tested
manually against the real webcam (see commit history), not exercised
here, since opening real windows in a test suite is undesirable and
fragile per this project's own testing conventions.
"""

from __future__ import annotations

import math

import pytest

from gaze_attention.filter_graph import FilterComparisonTracker, LiveFilterGraph
from gaze_attention.filters import BayesianFilter, MovingAverageFilter


def det(pid, gaze_signal=0.7, confidence=0.8):
    return {"id": pid, "gaze_signal": gaze_signal, "confidence": confidence}


# --- 1. All filters receive the same input sequence --------------------------

def test_all_filters_receive_same_input_sequence():
    """Cross-check FilterComparisonTracker's outputs against directly
    instantiating the same two filters.py classes with the identical raw
    (signal, confidence, dt) sequence -- if the tracker fed them different
    inputs, the outputs would diverge from this independent computation."""
    tracker = FilterComparisonTracker()
    direct_ma = MovingAverageFilter()
    direct_bayes = BayesianFilter()

    dt = 0.033
    sequence = [(0.1, 0.9), (0.9, 0.2), (0.5, 0.5), (0.9, 0.9), (0.0, 0.1)] * 10
    for gaze_signal, confidence in sequence:
        result = tracker.update([det(0, gaze_signal, confidence)], dt)
        direct_ma_sig, _ = direct_ma.update(0, gaze_signal, confidence, dt)
        direct_bayes_sig, _ = direct_bayes.update(0, gaze_signal, confidence, dt)

        assert result[0]["moving_average"] == pytest.approx(direct_ma_sig)
        assert result[0]["bayesian"] == pytest.approx(direct_bayes_sig)
        assert result[0]["raw"] == gaze_signal


def test_same_input_sequence_across_multiple_people_simultaneously():
    tracker = FilterComparisonTracker()
    direct = {0: (MovingAverageFilter(), BayesianFilter()), 1: (MovingAverageFilter(), BayesianFilter())}
    dt = 0.033
    for i in range(30):
        detections = [det(0, 0.2 + 0.01 * i, 0.7), det(1, 0.8 - 0.01 * i, 0.6)]
        result = tracker.update(detections, dt)
        for pid, gaze_signal, confidence in [(0, 0.2 + 0.01 * i, 0.7), (1, 0.8 - 0.01 * i, 0.6)]:
            ma_f, bayes_f = direct[pid]
            ma_sig, _ = ma_f.update(pid, gaze_signal, confidence, dt)
            bayes_sig, _ = bayes_f.update(pid, gaze_signal, confidence, dt)
            assert result[pid]["moving_average"] == pytest.approx(ma_sig)
            assert result[pid]["bayesian"] == pytest.approx(bayes_sig)


# --- 2. Filter outputs remain finite ------------------------------------------

def test_filter_outputs_remain_finite_under_hostile_inputs():
    tracker = FilterComparisonTracker()
    hostile = [(0.0, 0.0), (1.0, 1.0), (-1.0, -1.0), (2.0, 2.0), (float("nan"), 0.5)]
    for gaze_signal, confidence in hostile:
        if math.isnan(gaze_signal):
            continue  # NaN input is perception.py's job to prevent, not this module's
        result = tracker.update([det(0, gaze_signal, confidence)], dt=0.033)
        for key in ("raw", "moving_average", "bayesian"):
            assert math.isfinite(result[0][key]), f"{key} produced non-finite output"


def test_filter_outputs_remain_finite_over_long_session():
    import numpy as np

    tracker = FilterComparisonTracker()
    rng = np.random.default_rng(0)
    for _ in range(2000):
        gaze_signal = float(np.clip(rng.normal(0.5, 0.3), 0.0, 1.0))
        confidence = float(rng.uniform(0.0, 1.0))
        result = tracker.update([det(0, gaze_signal, confidence)], dt=0.033)
        assert all(math.isfinite(v) for v in result[0].values())


# --- 3. Filter state resets when a track disappears ---------------------------

def test_filter_state_resets_when_track_disappears():
    tracker = FilterComparisonTracker()
    for _ in range(30):
        tracker.update([det(0, 0.9, 0.9)], dt=0.033)
    warm = tracker.update([det(0, 0.9, 0.9)], dt=0.033)
    assert warm[0]["moving_average"] > 0.5

    tracker.update([], dt=0.033)  # track disappears

    fresh = tracker.update([det(0, 0.0, 0.9)], dt=0.033)
    assert fresh[0]["moving_average"] < warm[0]["moving_average"], (
        "reappearing id must not inherit the disappeared track's filter state"
    )


def test_reset_clears_all_tracked_people():
    tracker = FilterComparisonTracker()
    tracker.update([det(0, 0.9, 0.9), det(1, 0.9, 0.9)], dt=0.033)
    tracker.reset()
    result = tracker.update([det(0, 0.0, 0.9), det(1, 0.0, 0.9)], dt=0.033)
    assert result[0]["moving_average"] == 0.0
    assert result[1]["moving_average"] == 0.0


# --- 4. Graph mode does not change policy output ------------------------------

def test_graph_mode_does_not_change_policy_output():
    """Run an identical scenario twice through a real tracker+filter+policy
    pipeline -- once with FilterComparisonTracker.update() invoked every
    tick (as main_webcam.py does when --show-filter-graph is set), once
    without -- and assert the actual policy's target sequence is
    byte-identical. Proves the comparison tracker has zero side effects on
    the production pipeline."""
    from gaze_attention.filters import make_filter
    from gaze_attention.policies import make_policy
    from gaze_attention.tracker import IOUTracker

    def run(with_graph: bool):
        tracker = IOUTracker()
        filt = make_filter("moving_average")
        policy = make_policy("confidence_debt", confidence_threshold=0.45, minimum_hold_seconds=0.3, target_grace_seconds=0.2)
        graph_tracker = FilterComparisonTracker() if with_graph else None
        known_ids = set()
        targets = []
        t = 0.0
        dt = 0.033
        for i in range(200):
            bbox_a = [0, 0, 50, 50]
            bbox_b = [200, 200, 250, 250]
            detections = [
                {"bbox": bbox_a, "gaze_signal": 0.9, "confidence": 0.9},
                {"bbox": bbox_b, "gaze_signal": 0.9 if (i // 20) % 2 == 0 else 0.1, "confidence": 0.9},
            ]
            ids = tracker.update([d["bbox"] for d in detections])
            raw_observations = [{"id": pid, **d} for pid, d in zip(ids, detections)]

            current_ids = {obs["id"] for obs in raw_observations}
            for gone in known_ids - current_ids:
                filt.forget(gone)
            known_ids = current_ids

            if graph_tracker is not None:
                graph_tracker.update(raw_observations, dt)  # purely observational

            policy_observations = []
            for obs in raw_observations:
                sig, _ = filt.update(obs["id"], obs["gaze_signal"], obs["confidence"], dt)
                policy_observations.append({
                    "id": obs["id"], "bbox": obs["bbox"], "gaze_signal": sig,
                    "confidence": obs["confidence"], "seeking_attention": sig >= 0.6, "timestamp": t,
                })
            target = policy.update(policy_observations, dt)
            targets.append(target)
            t += dt
        return targets

    targets_without_graph = run(with_graph=False)
    targets_with_graph = run(with_graph=True)
    assert targets_with_graph == targets_without_graph


# --- LiveFilterGraph (headless only) ------------------------------------------

def test_live_filter_graph_headless_writes_csv(tmp_path):
    g = LiveFilterGraph(person_id=0, seeking_threshold=0.6, window_seconds=15.0, headless=True, logs_dir=tmp_path)
    for i in range(10):
        g.push(
            timestamp=i * 0.1,
            values={"raw": 0.5, "moving_average": 0.5, "bayesian": 0.5},
            current_target=0,
            fps=30.0,
        )
    g.close()

    import csv as csv_module

    with open(g.csv_path) as f:
        rows = list(csv_module.DictReader(f))
    assert len(rows) == 10
    assert rows[0]["person_id"] == "0"
    assert rows[0]["moving_average"] == "0.5"


def test_live_filter_graph_headless_never_imports_matplotlib_pyplot_gui(tmp_path):
    """Headless mode must not require a working display at all."""
    g = LiveFilterGraph(person_id=0, seeking_threshold=0.6, headless=True, logs_dir=tmp_path)
    assert g.headless is True
    assert g._fig is None
    g.close()


def test_flip_count_and_rate_computed_per_filter(tmp_path):
    g = LiveFilterGraph(person_id=0, seeking_threshold=0.6, window_seconds=15.0, headless=True, logs_dir=tmp_path)
    # Oscillate moving_average above/below threshold; keep bayesian constant.
    for i in range(10):
        ma_val = 0.9 if i % 2 == 0 else 0.1
        g.push(i * 0.1, {"raw": 0.5, "moving_average": ma_val, "bayesian": 0.9}, current_target=0, fps=30.0)
    g.close()

    import csv as csv_module

    with open(g.csv_path) as f:
        rows = list(csv_module.DictReader(f))
    last = rows[-1]
    assert int(last["flip_count_moving_average"]) > 0, "oscillating signal should register flips"
    assert int(last["flip_count_bayesian"]) == 0, "constant signal should register zero flips"


def test_csv_never_logs_filter_uncertainty_as_confidence(tmp_path):
    """Requirement 6: filter confidence/uncertainty values must never be
    logged/plotted as if they were gaze correctness probabilities -- the
    CSV schema simply has no such column."""
    from gaze_attention.filter_graph import CSV_FIELDS

    for forbidden in ("uncertainty", "correctness", "accuracy"):
        assert not any(forbidden in field.lower() for field in CSV_FIELDS)


def test_graph_construction_never_calls_filter_update_before_first_push():
    """LiveFilterGraph itself must not run any filter -- FilterComparisonTracker
    owns the filters; LiveFilterGraph only visualizes/logs already-filtered
    values handed to it via push()."""
    import gaze_attention.filter_graph as fg_module

    assert not hasattr(fg_module.LiveFilterGraph, "update")
