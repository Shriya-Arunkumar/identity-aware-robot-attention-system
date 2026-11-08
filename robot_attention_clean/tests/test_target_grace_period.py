"""Integration tests for the FM3 fix (see DEBUG_ATTENTION.md): a single
missed detection frame must not immediately clear the current target, but
a genuine, extended departure eventually must. Exercises the real
_DwellArbitration machinery in policies.py end to end (target selection +
absence grace + hold time + state protection), not any one method in
isolation.
"""

import pytest

from gaze_attention.policies import ConfidenceDebtPolicy, GreedyDwellPolicy

ALL_POLICY_FACTORIES = [GreedyDwellPolicy, ConfidenceDebtPolicy]
DT = 0.033


def obs(id_, t, gaze_signal=0.9, confidence=0.9, seeking=True, lost=False):
    return {
        "id": id_,
        "bbox": [0, 0, 10, 10],
        "gaze_signal": gaze_signal,
        "confidence": confidence,
        "seeking_attention": seeking,
        "timestamp": t,
        "lost": lost,
    }


@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_target_held_one_second_then_one_missed_frame_stays_selected(factory):
    """A target held well past minimum_hold_seconds must survive a single
    missed-detection frame (the direct FM3 reproduction from
    DEBUG_ATTENTION.md)."""
    policy = factory(minimum_hold_seconds=0.75, target_grace_seconds=0.5)
    t = 0.0
    target = None
    for _ in range(30):  # ~1 second of steady presence
        target = policy.update([obs(0, t)], DT)
        t += DT
    assert target == 0
    assert policy._time_since_switch >= 0.75  # hold time genuinely elapsed

    # One missed frame (e.g. a blink) -- perception.py would emit [] for
    # this person this tick even though the tracker still holds their id.
    target = policy.update([], DT)
    t += DT

    assert target == 0, "a single missed frame must not clear the target"


@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_repeated_missed_frames_beyond_grace_eventually_clears(factory):
    """Missing detection for longer than target_grace_seconds (with hold
    time already satisfied) must eventually release the target."""
    policy = factory(minimum_hold_seconds=0.1, target_grace_seconds=0.2)
    t = 0.0
    for _ in range(10):
        policy.update([obs(0, t)], DT)
        t += DT
    assert policy.current_target == 0

    # Miss far more frames than target_grace_seconds covers (0.2s / 0.033s ~= 6 frames).
    target = None
    for _ in range(20):
        target = policy.update([], DT)
        t += DT

    assert target is None, "a genuine extended absence must eventually clear the target"


@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_new_target_can_be_selected_after_clearing(factory):
    """After a genuine departure clears the target, a new person must be
    selectable -- the fix must not permanently wedge the policy."""
    policy = factory(minimum_hold_seconds=0.1, target_grace_seconds=0.2)
    t = 0.0
    for _ in range(10):
        policy.update([obs(0, t)], DT)
        t += DT
    assert policy.current_target == 0

    for _ in range(20):  # exceed grace -> clears to None
        policy.update([], DT)
        t += DT
    assert policy.current_target is None

    target = None
    for _ in range(10):
        target = policy.update([obs(7, t)], DT)
        t += DT
    assert target == 7


@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_target_reappearing_within_grace_does_not_lose_accumulated_state(factory):
    """The whole point of protecting (not pruning) the absent target's
    internal state: when they reappear within the grace window, their
    engagement/score must have kept accumulating from where it left off,
    not been silently reset to zero by a blip."""
    policy = factory(minimum_hold_seconds=0.75, target_grace_seconds=0.5)
    t = 0.0
    for _ in range(30):
        policy.update([obs(0, t)], DT)
        t += DT
    metrics_before = policy.get_metrics(0)
    assert metrics_before["score"] > 0

    # One missed frame.
    policy.update([], DT)
    t += DT

    # Reappear immediately -- state must not have been wiped by pruning.
    policy.update([obs(0, t)], DT)
    t += DT
    metrics_after = policy.get_metrics(0)

    assert metrics_after["score"] >= metrics_before["score"], (
        "a transient miss must not reset the target's accumulated score to zero"
    )


@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_minimum_hold_still_respected_even_when_target_absent(factory):
    """Point 3/4 of the fix spec: even once the absence grace period has
    elapsed, the target must not be released before minimum_hold_seconds
    has also elapsed."""
    policy = factory(minimum_hold_seconds=5.0, target_grace_seconds=0.1)
    t = 0.0
    # Only briefly held -- well under the 5s minimum hold.
    for _ in range(3):
        policy.update([obs(0, t)], DT)
        t += DT
    assert policy.current_target == 0

    # Absent for far longer than target_grace_seconds (0.1s) but the
    # 5s minimum_hold_seconds has not elapsed yet.
    target = None
    for _ in range(20):  # ~0.66s absent, grace long expired, hold not
        target = policy.update([], DT)
        t += DT

    assert target == 0, "must not release the target before minimum_hold_seconds elapses"


@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_lost_observation_cannot_become_or_challenge_for_target(factory):
    """Step 3 guard (dwell_arbitration.py): an observation with lost=True
    must never become the target and must never displace a lower-scoring
    visible person who already holds it -- even when it would otherwise
    win the tie-break outright on raw score."""
    # GreedyDwellPolicy ignores confidence magnitude above its gate (flat
    # accumulation rate regardless of how high confidence is), so the score
    # gap has to come from seeking-time asymmetry instead: person 0 seeks
    # continuously (lost=True), person 1 only intermittently -- a
    # differentiator both GreedyDwellPolicy and ConfidenceDebtPolicy respect.
    policy = factory(minimum_hold_seconds=0.05, target_grace_seconds=0.05)
    t = 0.0
    target = None
    for i in range(60):
        person_1_seeking = (i % 4) == 0  # seeking only 1 tick in 4
        target = policy.update(
            [
                obs(0, t, gaze_signal=0.95, confidence=0.95, seeking=True, lost=True),  # higher score, but lost
                obs(1, t, gaze_signal=0.6, confidence=0.5, seeking=person_1_seeking, lost=False),  # lower score, visible
            ],
            DT,
        )
        t += DT

    metrics_0 = policy.get_metrics(0)
    metrics_1 = policy.get_metrics(1)
    assert metrics_0["score"] > metrics_1["score"], (
        "test setup sanity check: person 0 must genuinely outscore person 1"
    )
    assert target == 1, "a lost=True observation must never become or displace the target"
    assert policy.current_target == 1


@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_transient_empty_present_ids_does_not_trigger_immediate_reselection(factory):
    """With a second person available, a single blip of the current target
    must not hand the target to the other person early -- selection must
    still respect hold time once the original target reappears within
    grace."""
    policy = factory(minimum_hold_seconds=0.75, target_grace_seconds=0.5)
    t = 0.0
    for _ in range(30):
        policy.update([obs(0, t), obs(1, t, seeking=False)], DT)
        t += DT
    assert policy.current_target == 0

    # Person 0 blips for one frame; person 1 is present but not seeking.
    target = policy.update([obs(1, t, seeking=False)], DT)
    t += DT
    assert target == 0, "transient absence must not immediately hand off to another present person"
