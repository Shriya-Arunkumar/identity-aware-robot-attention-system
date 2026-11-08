"""Tests for FM2: controlled engagement decay in ConfidenceDebtPolicy (see
DEBUG_ATTENTION.md FM2). Without decay, a person who stopped seeking kept
whatever engagement they'd banked forever, with no one else in frame to
build competing debt -- the policy would "stick" indefinitely. These tests
exercise the real ConfidenceDebtPolicy end to end, not the raw arithmetic.
"""

import pytest

from gaze_attention.policies import ConfidenceDebtPolicy

DT = 0.033


def obs(id_, t, seeking=True, confidence=0.9):
    return {
        "id": id_,
        "bbox": [0, 0, 10, 10],
        "gaze_signal": 0.9 if seeking else 0.1,
        "confidence": confidence,
        "seeking_attention": seeking,
        "timestamp": t,
    }


# --- 1. Engagement increases during sustained attention ---------------------

def test_engagement_increases_during_sustained_attention():
    policy = ConfidenceDebtPolicy()
    t = 0.0
    engagements = []
    for _ in range(30):
        policy.update([obs(0, t, seeking=True, confidence=0.8)], DT)
        engagements.append(policy.get_metrics(0)["engagement"])
        t += DT

    assert engagements[-1] > engagements[0]
    # Monotonically non-decreasing while continuously seeking.
    assert all(b >= a - 1e-9 for a, b in zip(engagements, engagements[1:]))


def test_engagement_growth_is_confidence_weighted():
    high = ConfidenceDebtPolicy()
    low = ConfidenceDebtPolicy()
    t = 0.0
    for _ in range(30):
        high.update([obs(0, t, seeking=True, confidence=0.9)], DT)
        low.update([obs(0, t, seeking=True, confidence=0.2)], DT)
        t += DT
    assert high.get_metrics(0)["engagement"] > low.get_metrics(0)["engagement"]


# --- 2. Engagement decays after attention stops ------------------------------

def test_engagement_decays_after_attention_stops():
    policy = ConfidenceDebtPolicy(engagement_decay=0.5)
    t = 0.0
    for _ in range(30):  # build up engagement while seeking
        policy.update([obs(0, t, seeking=True, confidence=0.9)], DT)
        t += DT
    peak = policy.get_metrics(0)["engagement"]
    assert peak > 0.0

    for _ in range(30):  # stop seeking -- still present, just not attending
        policy.update([obs(0, t, seeking=False)], DT)
        t += DT
    after = policy.get_metrics(0)["engagement"]

    assert after < peak, "engagement must decay once seeking_attention goes false"


def test_engagement_decay_rate_is_configurable():
    slow = ConfidenceDebtPolicy(engagement_decay=0.05)
    fast = ConfidenceDebtPolicy(engagement_decay=1.0)
    t = 0.0
    for _ in range(30):
        slow.update([obs(0, t, seeking=True, confidence=0.9)], DT)
        fast.update([obs(0, t, seeking=True, confidence=0.9)], DT)
        t += DT
    for _ in range(30):
        slow.update([obs(0, t, seeking=False)], DT)
        fast.update([obs(0, t, seeking=False)], DT)
        t += DT
    assert fast.get_metrics(0)["engagement"] < slow.get_metrics(0)["engagement"]


def test_engagement_does_not_decay_below_zero():
    policy = ConfidenceDebtPolicy(engagement_decay=5.0)  # aggressive decay
    t = 0.0
    for _ in range(10):
        policy.update([obs(0, t, seeking=True, confidence=0.5)], DT)
        t += DT
    for _ in range(200):  # far more than enough to decay to zero and past
        policy.update([obs(0, t, seeking=False)], DT)
        t += DT
    assert policy.get_metrics(0)["engagement"] == 0.0


# --- 3. Debt increases only for overlooked active seekers --------------------

def test_debt_increases_only_for_overlooked_active_seekers():
    policy = ConfidenceDebtPolicy(confidence_threshold=0.45, minimum_hold_seconds=100.0)
    t = 0.0
    # Person 0 becomes (and stays, hold time pinned) the target; person 1
    # actively seeks with good confidence but is overlooked.
    for _ in range(60):
        policy.update(
            [obs(0, t, seeking=True, confidence=0.9), obs(1, t, seeking=True, confidence=0.9)], DT
        )
        t += DT

    assert policy.current_target == 0
    assert policy.get_metrics(1)["debt"] > 0.0
    assert policy.get_metrics(0)["debt"] == 0.0  # current target never accrues debt


def test_debt_does_not_increase_for_non_seeking_overlooked_person():
    policy = ConfidenceDebtPolicy(minimum_hold_seconds=100.0)
    t = 0.0
    for _ in range(60):
        policy.update(
            [obs(0, t, seeking=True, confidence=0.9), obs(1, t, seeking=False, confidence=0.9)], DT
        )
        t += DT
    assert policy.get_metrics(1)["debt"] == 0.0


def test_debt_does_not_increase_for_low_confidence_overlooked_seeker():
    policy = ConfidenceDebtPolicy(confidence_threshold=0.45, minimum_hold_seconds=100.0)
    t = 0.0
    for _ in range(60):
        policy.update(
            [obs(0, t, seeking=True, confidence=0.9), obs(1, t, seeking=True, confidence=0.2)], DT
        )
        t += DT
    assert policy.get_metrics(1)["debt"] == 0.0


# --- 4. Debt resets on selection ---------------------------------------------

def test_debt_resets_when_person_becomes_selected_target():
    policy = ConfidenceDebtPolicy(minimum_hold_seconds=0.05, debt_weight=5.0, target_grace_seconds=0.05)
    t = 0.0
    # Person 0 holds the target; person 1 seeks and accrues debt until it
    # eventually out-scores person 0 and takes over.
    target = None
    for _ in range(300):
        target = policy.update(
            [obs(0, t, seeking=False, confidence=0.9), obs(1, t, seeking=True, confidence=0.9)], DT
        )
        t += DT

    assert target == 1
    assert policy.get_metrics(1)["debt"] == 0.0, "debt must reset the instant a person becomes the target"


def test_debt_stays_zero_while_holding_target():
    # _scores() checks `current_target` as decided by the PREVIOUS tick, so
    # on tick 1 -- before anyone has ever been selected -- nobody is
    # "current target" yet and debt can briefly accrue for that one tick.
    # From tick 2 onward (once selection has actually happened once),
    # debt must stay pinned at zero for as long as they hold the target.
    policy = ConfidenceDebtPolicy(minimum_hold_seconds=100.0)
    t = 0.0
    policy.update([obs(0, t, seeking=True, confidence=0.9)], DT)
    t += DT
    for _ in range(90):
        policy.update([obs(0, t, seeking=True, confidence=0.9)], DT)
        assert policy.get_metrics(0)["debt"] == 0.0
        t += DT


# --- 5. No unbounded score growth --------------------------------------------

def test_engagement_and_debt_stay_within_configured_max():
    policy = ConfidenceDebtPolicy(engagement_max=5.0, debt_max=5.0, minimum_hold_seconds=100.0)
    t = 0.0
    # Person 0 is pinned as target (never accrues debt); person 1 seeks
    # forever while overlooked (accrues both engagement-decay-immune... no,
    # person 1 IS seeking, so its engagement also grows) and debt.
    for _ in range(5000):
        policy.update(
            [obs(0, t, seeking=True, confidence=0.9), obs(1, t, seeking=True, confidence=0.9)], DT
        )
        t += DT

    m0 = policy.get_metrics(0)
    m1 = policy.get_metrics(1)
    assert m0["engagement"] <= 5.0
    assert m1["engagement"] <= 5.0
    assert m1["debt"] <= 5.0
    assert m0["score"] <= 5.0 + 0.5 * 5.0  # engagement_max + debt_weight * debt_max
    assert m1["score"] <= 5.0 + 0.5 * 5.0


def test_score_does_not_grow_unboundedly_over_a_long_session():
    policy = ConfidenceDebtPolicy(minimum_hold_seconds=100.0)  # default max = 30.0
    t = 0.0
    for _ in range(10000):  # ~330s of simulated time at dt=0.033
        policy.update([obs(0, t, seeking=True, confidence=1.0)], DT)
        t += DT
    metrics = policy.get_metrics(0)
    assert metrics["engagement"] <= 30.0
    assert metrics["score"] <= 30.0 + 0.5 * 30.0
