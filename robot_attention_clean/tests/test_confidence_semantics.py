"""Integration tests for the FM1 fix (see DEBUG_ATTENTION.md): raw
confidence, filtered signal, filter uncertainty, and policy confidence are
four distinct fields, and low raw confidence must never silently become
high policy confidence just because a filter has converged.

`_tick()` below mirrors main_webcam.py's/evaluation/replay.py's real
orchestration logic exactly (raw_confidence -> filt.update() ->
policy_confidence = raw_confidence, Version 1) so these tests exercise the
actual integration contract, not just filters.py in isolation — a
filter-only test would not have caught FM1, since each filter's own
`filter_uncertainty` output is *correct on its own terms*; the bug was in
which value the orchestration layer fed to the policy.
"""

from gaze_attention.filters import BayesianFilter
from gaze_attention.policies import ConfidenceDebtPolicy, GreedyDwellPolicy

DT = 0.033


def _tick(filt, pid, gaze_signal, raw_confidence, dt=DT):
    """One tick of the real pipeline's confidence handling."""
    filtered_signal, filter_uncertainty = filt.update(pid, gaze_signal, raw_confidence, dt)
    policy_confidence = raw_confidence  # Version 1 (see main_webcam.py docstring).
    return filtered_signal, filter_uncertainty, policy_confidence


# --- 1. raw confidence 0.10 cannot silently become policy confidence 0.83 ---

def test_raw_confidence_0_10_cannot_become_policy_confidence_0_83_bayesian():
    filt = BayesianFilter()
    raw_confidence = 0.10
    filtered_signal = filter_uncertainty = policy_confidence = None
    for _ in range(90):  # long enough for the filter to fully converge
        filtered_signal, filter_uncertainty, policy_confidence = _tick(filt, 0, 0.85, raw_confidence)

    # The trap (FM1): the filter's OWN uncertainty about its estimate does
    # converge low (looks "confident") even though the input never was.
    assert filter_uncertainty < 0.3, (
        "sanity check: the filter should have converged to low uncertainty "
        "here -- if this fails, the test setup itself is wrong"
    )
    # The fix: policy_confidence must stay exactly the raw value, never
    # drift toward the filter's converged uncertainty-derived reading.
    assert policy_confidence == raw_confidence == 0.10
    assert policy_confidence < 0.83


def test_raw_confidence_0_10_cannot_become_policy_confidence_0_83_moving_average():
    from gaze_attention.filters import MovingAverageFilter

    filt = MovingAverageFilter()
    raw_confidence = 0.10
    policy_confidence = None
    for _ in range(90):
        _, _, policy_confidence = _tick(filt, 0, 0.85, raw_confidence)
    assert policy_confidence == 0.10


# --- 2. low-confidence observations don't pass the gate after convergence ---

def test_low_confidence_never_passes_gate_even_after_filter_converges():
    filt = BayesianFilter()
    policy = ConfidenceDebtPolicy(confidence_threshold=0.45, minimum_hold_seconds=0.05, target_grace_seconds=0.05)
    raw_confidence = 0.10  # well below confidence_threshold=0.45

    t = 0.0
    for _ in range(120):  # far past filter convergence
        filtered_signal, filter_uncertainty, policy_confidence = _tick(filt, 0, 0.9, raw_confidence)
        seeking = filtered_signal >= 0.6
        obs = {
            "id": 0, "bbox": [0, 0, 1, 1],
            "gaze_signal": filtered_signal, "confidence": policy_confidence,
            "seeking_attention": seeking, "timestamp": t,
        }
        policy.update([obs], DT)
        t += DT

    metrics = policy.get_metrics(0)
    assert metrics["debt"] == 0.0, (
        "debt must not accrue for a person whose raw confidence never "
        "cleared the gate, no matter how long the filter has been running"
    )


def test_low_confidence_never_passes_greedy_gate_even_after_filter_converges():
    filt = BayesianFilter()
    policy = GreedyDwellPolicy(confidence_threshold=0.45, minimum_hold_seconds=0.05)
    raw_confidence = 0.10

    t = 0.0
    score = None
    for _ in range(120):
        filtered_signal, filter_uncertainty, policy_confidence = _tick(filt, 0, 0.9, raw_confidence)
        seeking = filtered_signal >= 0.6
        obs = {
            "id": 0, "bbox": [0, 0, 1, 1],
            "gaze_signal": filtered_signal, "confidence": policy_confidence,
            "seeking_attention": seeking, "timestamp": t,
        }
        policy.update([obs], DT)
        score = policy.get_metrics(0)["score"]
        t += DT

    assert score == 0.0, "greedy score must not accumulate below the confidence gate"


# --- 3. filtered signal can be smooth while policy confidence stays low -----

def test_filtered_signal_smooth_while_policy_confidence_stays_low():
    import numpy as np

    filt = BayesianFilter()
    raw_confidence = 0.15
    rng_signal = [0.5 + 0.05 * ((-1) ** i) for i in range(20)]  # small oscillation to filter out

    filtered_signals = []
    policy_confidences = []
    for i in range(200):
        noisy_input = rng_signal[i % len(rng_signal)]
        filtered_signal, filter_uncertainty, policy_confidence = _tick(filt, 0, noisy_input, raw_confidence)
        filtered_signals.append(filtered_signal)
        policy_confidences.append(policy_confidence)

    # Signal has smoothed out (low variance in the tail) ...
    tail_signal = filtered_signals[-30:]
    assert np.var(tail_signal) < 0.001, "filtered_signal should have smoothed out the oscillation"

    # ... while policy_confidence never moved from the constant low raw input.
    assert all(pc == raw_confidence for pc in policy_confidences)
    assert policy_confidences[-1] < 0.45  # still below the default confidence_threshold
