"""Unit tests for gaze_attention.policies — deterministic synthetic scenarios,
no camera. Covers the "must verify" list from the project spec: one-person
case, ID handling, low-confidence non-accumulation, debt only for ignored
active seekers, debt reset on selection, minimum hold time, no NaN/crash.
"""

import math

import pytest

from gaze_attention.policies import (
    ConfidenceDebtPolicy,
    GreedyDwellPolicy,
    make_policy,
)

ALL_POLICY_FACTORIES = [GreedyDwellPolicy, ConfidenceDebtPolicy]


def obs(id_, gaze_signal=0.9, confidence=0.9, seeking=True, t=0.0, bbox=None):
    return {
        "id": id_,
        "bbox": bbox or [0, 0, 10, 10],
        "gaze_signal": gaze_signal,
        "confidence": confidence,
        "seeking_attention": seeking,
        "timestamp": t,
    }


# --- One-person case --------------------------------------------------------

@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_one_person_gets_selected(factory):
    policy = factory()
    target = None
    for i in range(30):
        target = policy.update([obs(0, t=i * 0.033)], dt=0.033)
    assert target == 0


@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_empty_observations_selects_none(factory):
    policy = factory()
    assert policy.update([], dt=0.033) is None


# --- Two people, equal attention: deterministic tie-break -------------------

@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_two_people_equal_attention_first_pick_is_lowest_id(factory):
    policy = factory()
    # First tick: both start at score 0 -> pure tie -> deterministic winner.
    target = policy.update([obs(5, t=0.0), obs(2, t=0.0)], dt=0.033)
    assert target == 2


@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_tiebreak_is_stable_across_repeated_runs(factory):
    results = set()
    for _ in range(5):
        policy = factory()
        target = policy.update([obs(3, t=0.0), obs(1, t=0.0)], dt=0.033)
        results.add(target)
    assert results == {1}


def test_greedy_equal_attention_stays_with_initial_pick():
    # Greedy has no mechanism to break a sustained tie -> the deterministic
    # first pick holds indefinitely, since both scores always move together.
    policy = GreedyDwellPolicy(minimum_hold_seconds=0.05)
    target = None
    for i in range(300):
        target = policy.update([obs(5, t=i * 0.033), obs(2, t=i * 0.033)], dt=0.033)
    assert target == 2


def test_confidence_debt_equal_attention_eventually_switches_for_fairness():
    # The debt term is the fairness mechanism: the person NOT currently
    # selected accrues debt while equally seeking, and eventually overtakes
    # the held target's score even though both are "equally" attentive —
    # this is the intended contrast with GreedyDwellPolicy, not a flaw.
    policy = ConfidenceDebtPolicy(minimum_hold_seconds=0.05, debt_weight=0.5)
    target = None
    for i in range(300):
        target = policy.update([obs(5, t=i * 0.033), obs(2, t=i * 0.033)], dt=0.033)
    assert target == 5


# --- One persistent seeker + one intermittent seeker -------------------------

@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_persistent_seeker_eventually_wins_over_intermittent(factory):
    policy = factory(minimum_hold_seconds=0.1) if factory is GreedyDwellPolicy else factory(minimum_hold_seconds=0.1)
    target = None
    for i in range(200):
        t = i * 0.033
        intermittent_seeking = (i // 10) % 2 == 0  # on/off every ~0.33s
        target = policy.update(
            [
                obs(0, seeking=True, t=t),           # persistent
                obs(1, seeking=intermittent_seeking, t=t),
            ],
            dt=0.033,
        )
    assert target == 0


# --- Low-confidence observations don't wrongly accumulate -------------------

def test_greedy_low_confidence_does_not_accumulate():
    policy = GreedyDwellPolicy(confidence_threshold=0.45)
    for i in range(100):
        policy.update([obs(0, seeking=True, confidence=0.1, t=i * 0.033)], dt=0.033)
    # Never cleared the confidence gate -> score should have decayed to 0,
    # never grown.
    assert policy._score[0] == 0.0


def test_confidence_debt_low_confidence_barely_grows_engagement():
    policy = ConfidenceDebtPolicy()
    high = ConfidenceDebtPolicy()
    for i in range(60):
        policy.update([obs(0, seeking=True, confidence=0.05, t=i * 0.033)], dt=0.033)
        high.update([obs(0, seeking=True, confidence=0.95, t=i * 0.033)], dt=0.033)
    assert policy._engagement[0] < high._engagement[0]


def test_confidence_debt_low_confidence_does_not_accrue_debt():
    policy = ConfidenceDebtPolicy(confidence_threshold=0.45)
    # Person 1 seeks with low confidence while person 0 holds the target —
    # debt should not accrue for person 1 since it never clears the gate.
    for i in range(60):
        policy.update(
            [obs(0, seeking=True, confidence=0.9, t=i * 0.033), obs(1, seeking=True, confidence=0.1, t=i * 0.033)],
            dt=0.033,
        )
    assert policy.current_target == 0
    assert policy._debt[1] == 0.0


# --- Debt increases only for non-selected active seekers; resets on selection

def test_debt_accrues_only_for_ignored_active_seekers():
    policy = ConfidenceDebtPolicy(minimum_hold_seconds=100.0)  # pin the target
    for i in range(60):
        policy.update(
            [obs(0, seeking=True, confidence=0.9, t=i * 0.033), obs(1, seeking=True, confidence=0.9, t=i * 0.033)],
            dt=0.033,
        )
    assert policy.current_target == 0
    assert policy._debt[1] > 0.0
    assert policy._debt[0] == 0.0  # current target never accrues debt


def test_debt_resets_when_target_switches_to_debtor():
    policy = ConfidenceDebtPolicy(minimum_hold_seconds=0.05, debt_weight=5.0)
    # Person 0 starts as target; person 1 seeks and racks up debt until it
    # outscores person 0, forcing a switch — at which point person 1's debt
    # must reset to 0.
    target = None
    for i in range(300):
        t = i * 0.033
        target = policy.update(
            [obs(0, seeking=False, confidence=0.9, t=t), obs(1, seeking=True, confidence=0.9, t=t)],
            dt=0.033,
        )
    assert target == 1
    assert policy._debt[1] == 0.0


# --- Minimum hold time prevents rapid switching ------------------------------

@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_minimum_hold_time_prevents_rapid_switching(factory):
    policy = factory(minimum_hold_seconds=5.0)
    switches = 0
    last = None
    for i in range(150):
        t = i * 0.033
        # Alternate which of two people is "winning" every frame — a policy
        # without hold-time protection would flap every tick.
        winner_is_0 = i % 2 == 0
        target = policy.update(
            [
                obs(0, seeking=winner_is_0, confidence=0.9, t=t),
                obs(1, seeking=not winner_is_0, confidence=0.9, t=t),
            ],
            dt=0.033,
        )
        if last is not None and target != last:
            switches += 1
        last = target
    assert switches == 0


# --- A person entering and leaving -------------------------------------------

@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_person_entering_and_leaving_no_crash_and_reselects(factory):
    # Short grace period so a genuine departure (many consecutive absent
    # frames) resolves within this test's frame budget -- see
    # test_target_absence_grace_period.py-style tests below for the
    # single-frame-blip-vs-genuine-departure distinction itself.
    policy = factory(minimum_hold_seconds=0.05, target_grace_seconds=0.05)
    for i in range(30):
        policy.update([obs(0, t=i * 0.033)], dt=0.033)
    assert policy.current_target == 0

    # Person 0 leaves for good (dropped by tracker -> absent from
    # observations for longer than both the hold and grace periods).
    target = None
    for i in range(10):
        target = policy.update([], dt=0.033)
    assert target is None

    # A new person (id 7) enters.
    for i in range(30):
        target = policy.update([obs(7, t=i * 0.033)], dt=0.033)
    assert target == 7


# --- No NaN / crash across scenarios -----------------------------------------

@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_no_nan_or_crash_under_noise(factory):
    import numpy as np

    rng = np.random.default_rng(1)
    policy = factory()
    for i in range(300):
        n_people = rng.integers(0, 4)
        observations = [
            obs(
                int(pid),
                gaze_signal=float(rng.uniform(0, 1)),
                confidence=float(rng.uniform(0, 1)),
                seeking=bool(rng.uniform() > 0.5),
                t=i * 0.033,
            )
            for pid in rng.choice(10, size=n_people, replace=False)
        ]
        target = policy.update(observations, dt=0.033)
        assert target is None or isinstance(target, int)


def test_make_policy_factory():
    for name in ("greedy", "confidence_debt"):
        p = make_policy(name)
        t = p.update([obs(0)], dt=0.033)
        assert not (isinstance(t, float) and math.isnan(t))

    with pytest.raises(ValueError):
        make_policy("nonexistent")


@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_reset_clears_state(factory):
    policy = factory()
    for i in range(30):
        policy.update([obs(0, t=i * 0.033)], dt=0.033)
    assert policy.current_target == 0
    policy.reset()
    assert policy.current_target is None
