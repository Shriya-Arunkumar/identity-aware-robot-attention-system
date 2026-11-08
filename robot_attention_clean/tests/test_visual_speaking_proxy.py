"""Unit tests for gaze_attention.speech.VisualSpeakingProxy — the
visual_speaking_proxy mouth-motion cue. NOT audio speech recognition;
there is no microphone in this project (see README.md's explicit
limitation statement and gaze_attention/speech.py's docstring).
"""

import math

import pytest

from gaze_attention.speech import VisualSpeakingProxy

DT = 0.033


def test_static_mouth_never_starts_speaking():
    proxy = VisualSpeakingProxy()
    speaking = None
    for _ in range(60):
        activity, speaking, conf = proxy.update(0, 0.10, DT)
    assert speaking is False
    assert activity == pytest.approx(0.0, abs=1e-6)


# --- 1. Mouth motion starts speaking -----------------------------------------

def test_mouth_motion_starts_speaking():
    proxy = VisualSpeakingProxy()
    speaking_history = []
    for i in range(60):
        opening = 0.15 + 0.10 * math.sin(i * 0.9)  # oscillating -> talking-like
        _activity, speaking, _conf = proxy.update(0, opening, DT)
        speaking_history.append(speaking)
    assert speaking_history[0] is False, "should not start speaking on the very first ambiguous sample"
    assert speaking_history[-1] is True, "sustained oscillation should trigger speaking"
    assert True in speaking_history


def test_mouth_activity_reflects_temporal_change_not_raw_opening():
    # A mouth held wide open but STATIC must NOT read as speaking -- it's
    # the CHANGE in opening that matters, not the opening itself.
    proxy = VisualSpeakingProxy()
    activity = speaking = None
    for _ in range(60):
        activity, speaking, _conf = proxy.update(0, 0.40, DT)  # wide open, unchanging
    assert speaking is False
    assert activity == pytest.approx(0.0, abs=1e-6)


# --- 2. Mouth motion stops speaking -------------------------------------------

def test_mouth_motion_stops_speaking():
    proxy = VisualSpeakingProxy()
    for i in range(40):  # get into a speaking state
        opening = 0.15 + 0.10 * math.sin(i * 0.9)
        _activity, speaking, _conf = proxy.update(0, opening, DT)
    assert speaking is True

    for _ in range(60):  # mouth goes static
        activity, speaking, _conf = proxy.update(0, 0.15, DT)
    assert speaking is False, "activity dropping below speaking_off_threshold should stop speaking"
    assert activity == pytest.approx(0.0, abs=1e-6)


def test_hysteresis_prevents_flicker_at_a_single_threshold():
    # A signal that hovers exactly between the on/off thresholds must not
    # flicker speaking True/False every tick -- that's the whole point of
    # having two distinct thresholds instead of one.
    proxy = VisualSpeakingProxy(speaking_on_threshold=0.15, speaking_off_threshold=0.08)
    # Get to a known "speaking" state with strong oscillation.
    for i in range(40):
        proxy.update(0, 0.15 + 0.10 * math.sin(i * 0.9), DT)
    assert proxy._speaking[0] is True

    # Now feed a mild oscillation whose resulting activity sits between
    # the two thresholds -- must stay speaking (no flip back to False)
    # since it hasn't dropped below speaking_off_threshold.
    transitions = []
    for i in range(40, 70):
        opening = 0.15 + 0.03 * math.sin(i * 0.9)
        _activity, speaking, _conf = proxy.update(0, opening, DT)
        transitions.append(speaking)
    # No flip-flopping: once in this band, it should settle, not oscillate
    # tick to tick.
    flips = sum(1 for a, b in zip(transitions, transitions[1:]) if a != b)
    assert flips <= 1, f"expected at most one settling transition, got {flips} flips: {transitions}"


# --- Speech confidence: data sufficiency, not gaze trustworthiness -----------

def test_speech_confidence_starts_low_and_rises_to_one():
    proxy = VisualSpeakingProxy(speaking_window_seconds=0.5)
    _activity, _speaking, conf_first = proxy.update(0, 0.15, DT)
    assert conf_first < 1.0
    conf_last = None
    for _ in range(30):
        _activity, _speaking, conf_last = proxy.update(0, 0.15, DT)
    assert conf_last == pytest.approx(1.0, abs=1e-6)


def test_speech_confidence_never_exceeds_one_or_goes_negative():
    proxy = VisualSpeakingProxy()
    for i in range(200):
        _activity, _speaking, conf = proxy.update(0, 0.1 + 0.05 * math.sin(i), DT)
        assert 0.0 <= conf <= 1.0


# --- 7. Disappeared tracks reset mouth history --------------------------------

def test_disappeared_track_resets_mouth_history():
    proxy = VisualSpeakingProxy()
    # Build up a full window + speaking state for id 0.
    for i in range(40):
        proxy.update(0, 0.15 + 0.10 * math.sin(i * 0.9), DT)
    assert proxy._speaking.get(0) is True
    _activity_before, _speaking_before, conf_before = proxy.update(0, 0.15, DT)
    assert conf_before == pytest.approx(1.0, abs=1e-6), "sanity: window should be full before forgetting"

    proxy.forget(0)

    # A fresh observation for the SAME id right after forget() must behave
    # like a brand new person: low speech_confidence (empty window), and
    # not speaking (no history to judge activity from).
    activity_after, speaking_after, conf_after = proxy.update(0, 0.15, DT)
    assert speaking_after is False, "forgotten track must not inherit stale speaking state"
    assert conf_after < conf_before, "forgotten track's window must have reset, not persisted"
    assert activity_after == 0.0


def test_forget_unknown_id_does_not_raise():
    proxy = VisualSpeakingProxy()
    proxy.forget(999)  # never seen -- must be a no-op, not an error


def test_reset_clears_all_people():
    proxy = VisualSpeakingProxy()
    for i in range(40):
        proxy.update(0, 0.15 + 0.10 * math.sin(i * 0.9), DT)
        proxy.update(1, 0.15 + 0.10 * math.sin(i * 0.9), DT)
    assert proxy._speaking.get(0) is True
    assert proxy._speaking.get(1) is True

    proxy.reset()

    _activity0, speaking0, conf0 = proxy.update(0, 0.15, DT)
    _activity1, speaking1, conf1 = proxy.update(1, 0.15, DT)
    assert speaking0 is False and speaking1 is False
    assert conf0 < 1.0 and conf1 < 1.0


# --- No NaN / crash under noise ----------------------------------------------

def test_no_nan_or_crash_under_noise():
    import numpy as np

    rng = np.random.default_rng(0)
    proxy = VisualSpeakingProxy()
    for i in range(300):
        opening = float(np.clip(rng.normal(0.2, 0.1), 0.0, 1.0))
        activity, speaking, conf = proxy.update(0, opening, DT)
        assert not math.isnan(activity)
        assert not math.isnan(conf)
        assert isinstance(speaking, bool)


def test_zero_dt_never_crashes():
    proxy = VisualSpeakingProxy()
    for _ in range(10):
        activity, speaking, conf = proxy.update(0, 0.2, 0.0)
        assert not math.isnan(activity)


def test_construction_rejects_inverted_hysteresis():
    with pytest.raises(ValueError):
        VisualSpeakingProxy(speaking_on_threshold=0.05, speaking_off_threshold=0.10)
