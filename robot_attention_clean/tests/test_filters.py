"""Unit tests for gaze_attention.filters — synthetic signals, no camera.

Kalman was removed from this prototype's scope (see README.md "Filter
scope" and CLAUDE.md) -- the active filter set is moving_average and
bayesian only."""

import math

import numpy as np
import pytest

from gaze_attention.filters import (
    FILTERS,
    BayesianFilter,
    MovingAverageFilter,
    make_filter,
)

ALL_FILTER_FACTORIES = [MovingAverageFilter, BayesianFilter]


@pytest.mark.parametrize("factory", ALL_FILTER_FACTORIES)
def test_no_nan_or_crash_on_constant_signal(factory):
    f = factory()
    for _ in range(50):
        sig, conf = f.update(person_id=0, gaze_signal=0.8, confidence=0.9, dt=0.033)
        assert not math.isnan(sig)
        assert not math.isnan(conf)
        assert 0.0 <= sig <= 1.0
        assert 0.0 <= conf <= 1.0


@pytest.mark.parametrize("factory", ALL_FILTER_FACTORIES)
def test_no_nan_or_crash_on_noisy_signal(factory):
    rng = np.random.default_rng(0)
    f = factory()
    for _ in range(200):
        sig = float(np.clip(rng.normal(0.5, 0.3), 0.0, 1.0))
        conf = float(rng.uniform(0.0, 1.0))
        out_sig, out_conf = f.update(person_id=1, gaze_signal=sig, confidence=conf, dt=0.05)
        assert not math.isnan(out_sig)
        assert not math.isnan(out_conf)


@pytest.mark.parametrize("factory", ALL_FILTER_FACTORIES)
def test_zero_confidence_never_crashes(factory):
    f = factory()
    for _ in range(30):
        sig, conf = f.update(person_id=0, gaze_signal=1.0, confidence=0.0, dt=0.033)
        assert not math.isnan(sig)
        assert not math.isnan(conf)


@pytest.mark.parametrize("factory", ALL_FILTER_FACTORIES)
def test_tiny_dt_never_crashes(factory):
    f = factory()
    for _ in range(10):
        sig, conf = f.update(person_id=0, gaze_signal=0.5, confidence=0.5, dt=0.0)
        assert not math.isnan(sig)


@pytest.mark.parametrize("factory", ALL_FILTER_FACTORIES)
def test_multiple_people_independent(factory):
    f = factory()
    for _ in range(20):
        f.update(0, gaze_signal=0.9, confidence=0.9, dt=0.03)
        f.update(1, gaze_signal=0.1, confidence=0.9, dt=0.03)
    sig0, _ = f.update(0, gaze_signal=0.9, confidence=0.9, dt=0.03)
    sig1, _ = f.update(1, gaze_signal=0.1, confidence=0.9, dt=0.03)
    assert sig0 > sig1


@pytest.mark.parametrize("factory", ALL_FILTER_FACTORIES)
def test_forget_clears_state(factory):
    f = factory()
    f.update(0, gaze_signal=0.9, confidence=0.9, dt=0.03)
    f.forget(0)
    # After forgetting, a fresh low observation should not be dragged toward
    # the old high value the way a warm/persistent state would.
    sig, _ = f.update(0, gaze_signal=0.1, confidence=0.9, dt=0.03)
    assert sig < 0.9


@pytest.mark.parametrize("factory", ALL_FILTER_FACTORIES)
def test_reset_clears_all_people(factory):
    f = factory()
    f.update(0, gaze_signal=0.9, confidence=0.9, dt=0.03)
    f.update(1, gaze_signal=0.9, confidence=0.9, dt=0.03)
    f.reset()
    sig0, _ = f.update(0, gaze_signal=0.1, confidence=0.9, dt=0.03)
    sig1, _ = f.update(1, gaze_signal=0.1, confidence=0.9, dt=0.03)
    assert sig0 < 0.9 and sig1 < 0.9


def test_moving_average_smooths_step_change():
    f = MovingAverageFilter(window=5)
    for _ in range(5):
        f.update(0, gaze_signal=0.0, confidence=1.0, dt=0.033)
    sig, _ = f.update(0, gaze_signal=1.0, confidence=1.0, dt=0.033)
    # One high sample among five zeros should be smoothed, not passed through.
    assert 0.0 < sig < 1.0


def test_bayesian_low_confidence_relaxes_toward_prior():
    f = BayesianFilter(prior_mu=0.3, relax_rate=2.0)
    # Start near 1.0 with high confidence, then feed low-confidence noise —
    # the estimate should drift toward the prior, not toward the noise.
    for _ in range(10):
        f.update(0, gaze_signal=1.0, confidence=1.0, dt=0.1)
    sig_before, _ = f.update(0, gaze_signal=1.0, confidence=1.0, dt=0.1)
    for _ in range(30):
        sig, _ = f.update(0, gaze_signal=0.9, confidence=0.0, dt=0.1)
    assert sig < sig_before


def test_bayesian_high_confidence_tracks_observation():
    f = BayesianFilter()
    for _ in range(30):
        sig, _ = f.update(0, gaze_signal=0.95, confidence=1.0, dt=0.05)
    assert sig > 0.7


def test_make_filter_factory():
    for name in ("moving_average", "bayesian"):
        f = make_filter(name)
        sig, conf = f.update(0, 0.5, 0.5, 0.033)
        assert not math.isnan(sig)

    with pytest.raises(ValueError):
        make_filter("nonexistent")


# --- Kalman removal (see README.md "Filter scope") ---------------------------

def test_kalman_not_in_active_filter_registry():
    assert "kalman" not in FILTERS
    assert set(FILTERS) == {"moving_average", "bayesian"}


def test_kalman_not_instantiable_via_factory():
    with pytest.raises(ValueError):
        make_filter("kalman")


def test_kalman_filter_class_no_longer_exists():
    import gaze_attention.filters as filters_module

    assert not hasattr(filters_module, "KalmanFilter")


def test_kalman_not_in_default_config():
    from gaze_attention import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["filter"] != "kalman"
    assert DEFAULT_CONFIG["filter"] in FILTERS


def test_kalman_not_in_config_yaml():
    from gaze_attention import load_config

    config = load_config()
    assert config["filter"] != "kalman"
    assert config["filter"] in FILTERS, (
        f"config.yaml's filter={config['filter']!r} is not an active filter; "
        f"choices are {sorted(FILTERS)}"
    )


@pytest.mark.parametrize("factory", ALL_FILTER_FACTORIES)
def test_both_remaining_filters_produce_finite_outputs(factory):
    """Explicit finite-output guard across a hostile input sweep -- both
    remaining filters, every edge case, no NaN/Inf ever."""
    f = factory()
    hostile_inputs = [
        (0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 10.0), (1.0, 0.0, 1e-9),
        (0.5, 0.5, 0.033), (-1.0, -1.0, 0.033), (2.0, 2.0, 0.033),
    ]
    for gaze_signal, confidence, dt in hostile_inputs * 20:
        sig, unc = f.update(0, gaze_signal, confidence, dt)
        assert math.isfinite(sig), f"non-finite signal from {factory.__name__}"
        assert math.isfinite(unc), f"non-finite uncertainty from {factory.__name__}"
        assert 0.0 <= sig <= 1.0
        assert 0.0 <= unc <= 1.0


@pytest.mark.parametrize("factory", ALL_FILTER_FACTORIES)
def test_filter_state_resets_when_track_disappears(factory):
    """A dropped track's filter state must not leak into a reused id."""
    f = factory()
    for _ in range(30):
        f.update(7, gaze_signal=1.0, confidence=1.0, dt=0.033)
    warm_signal, _ = f.update(7, gaze_signal=1.0, confidence=1.0, dt=0.033)
    assert warm_signal > 0.5

    f.forget(7)  # track disappeared (tracker.py's max_missed_frames elapsed)

    # Same id reused for a different person starting from a low signal: the
    # first post-forget reading must not be dragged up by the old state.
    fresh_signal, _ = f.update(7, gaze_signal=0.0, confidence=1.0, dt=0.033)
    assert fresh_signal < warm_signal
    assert math.isfinite(fresh_signal)
