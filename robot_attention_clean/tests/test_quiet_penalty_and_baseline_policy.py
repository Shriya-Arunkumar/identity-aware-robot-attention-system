"""Tests for:
1. GreedyDwellPolicy being baseline-only, never active by default (moved to
   gaze_attention/baseline_policies.py -- see that module's docstring).
2. The bounded quietness penalty added to ConversationalSimplePolicy (the
   active policy): quiet_duration / quiet_penalty / attention_duration
   bookkeeping, the final score formula (positive gaze evidence +
   active-speaker evidence + optional attention debt - quiet_penalty -
   bounded stare penalty), and its interaction with active-speaker mode and
   the no-speaker fallback.
"""

from __future__ import annotations

import math

from gaze_attention import DEFAULT_CONFIG
from gaze_attention.baseline_policies import GreedyDwellPolicy as BaselineGreedyDwellPolicy
from gaze_attention.policies import (
    ConfidenceDebtPolicy,
    ConversationalSimplePolicy,
    GreedyDwellPolicy,
    POLICIES,
)

DT = 0.1


def obs(id_, seeking=True, confidence=0.9, speaking=False, speech_confidence=0.0, t=0.0):
    return {
        "id": id_,
        "bbox": [0, 0, 10, 10],
        "gaze_signal": 0.9 if seeking else 0.1,
        "confidence": confidence,
        "seeking_attention": seeking,
        "timestamp": t,
        "speaking": speaking,
        "speech_confidence": speech_confidence,
    }


def make_policy(**overrides):
    kwargs = dict(
        confidence_threshold=0.45,
        minimum_hold_seconds=0.75,
        target_grace_seconds=0.5,
        # See test_conversational_simple_policy.py's make_policy() -- these
        # tests rely on speaking/speech_confidence actually driving
        # selection, which now requires speech_mode != "off".
        speech_mode="mouth_motion",
        speaker_confirm_seconds=0.3,
        speaker_release_seconds=0.5,
        minimum_speaker_attention_seconds=0.2,
        speaker_confidence_threshold=0.25,
        non_speaking_score_decay=0.5,
        recent_attention_penalty_weight=0.5,
        max_continuous_gaze_seconds=1.0,
        quiet_decay_rate=0.2,
        quiet_penalty_max=2.0,
    )
    kwargs.update(overrides)
    return ConversationalSimplePolicy(**kwargs)


# --- 1. Greedy dwell is not active by default --------------------------------

def test_greedy_dwell_is_not_active_by_default():
    assert DEFAULT_CONFIG["active_policy"] == "conversational_simple"
    assert DEFAULT_CONFIG["active_policy"] != "greedy"


def test_greedy_dwell_moved_to_baseline_module_and_marked():
    # Same class, re-exported for backward-compatible imports.
    assert GreedyDwellPolicy is BaselineGreedyDwellPolicy
    assert getattr(GreedyDwellPolicy, "BASELINE_ONLY", False) is True


def test_greedy_still_registered_for_explicit_baseline_comparison_only():
    # Retained (not deleted), but only reachable via an explicit name, never
    # as a default construction path.
    assert "greedy" in POLICIES
    assert POLICIES["greedy"] is GreedyDwellPolicy


# --- 2. Quiet, non-gazing person receives increasing penalty -----------------

def test_quiet_non_gazing_person_receives_increasing_penalty():
    policy = make_policy()
    t = 0.0
    penalties = []
    for i in range(20):
        policy.update([obs(0, seeking=False, speaking=False, t=t)], DT)
        penalties.append(policy.quiet_penalty.get(0, 0.0))
        t += DT
    # Strictly non-decreasing, and clearly increasing overall (not stuck at 0).
    assert all(b >= a - 1e-9 for a, b in zip(penalties, penalties[1:]))
    assert penalties[-1] > penalties[0]
    assert penalties[-1] > 0.0


# --- 3. Quiet person maintaining eye contact is not penalized aggressively ---

def test_quiet_person_with_eye_contact_not_penalized_aggressively():
    policy = make_policy()
    t = 0.0
    penalties = []
    # Present, looking at the robot, but never speaking -- reliable eye
    # contact without talking.
    for i in range(30):
        policy.update([obs(0, seeking=True, speaking=False, t=t)], DT)
        penalties.append(policy.quiet_penalty.get(0, 0.0))
        t += DT
    assert all(p == 0.0 for p in penalties), "sustained eye contact must never accrue a quiet penalty"


# --- 4. Speaker score remains positive while speaking -------------------------

def test_speaker_score_remains_positive_while_speaking():
    policy = make_policy()
    t = 0.0
    scores = []
    for i in range(30):
        policy.update([obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t)], DT)
        scores.append(policy._score_of(0))
        t += DT
    # speaker_confirm_seconds=0.3s -> confirmed by tick 3; positive after that.
    assert all(s > 0.0 for s in scores[3:]), "an active speaker's score must stay positive while speaking"


# --- 5. A new speaker can overcome a quiet previous target --------------------

def test_new_speaker_overcomes_quiet_previous_target():
    policy = make_policy()
    t = 0.0
    # Person 0 becomes the settled fallback target via gaze.
    for i in range(20):
        policy.update([obs(0, seeking=True, t=t), obs(1, seeking=False, t=t)], DT)
        t += DT
    assert policy.current_target == 0

    # Person 0 goes quiet (stops looking); person 1 starts speaking. Needs
    # speaker_confirm_seconds (0.3s) + the stabilization patch's
    # speaker_switch_confirm_seconds (0.5s default) before the switch
    # actually fires -- see policies.py's _resolve_target.
    target = None
    for i in range(9):
        target = policy.update(
            [obs(0, seeking=False, speaking=False, t=t), obs(1, seeking=False, speaking=True, speech_confidence=1.0, t=t)],
            DT,
        )
        t += DT
    assert target == 1, "a new confirmed speaker must overcome a now-quiet previous target"


# --- 6. Quiet penalty is bounded ----------------------------------------------

def test_quiet_penalty_is_bounded():
    policy = make_policy(quiet_penalty_max=1.5, quiet_decay_rate=1.0)
    t = 0.0
    for i in range(500):  # far past what would be needed to saturate an unbounded penalty
        policy.update([obs(0, seeking=False, speaking=False, t=t)], DT)
        t += DT
    assert policy.quiet_penalty[0] <= 1.5 + 1e-9


# --- 7. Scores remain finite --------------------------------------------------

def test_scores_remain_finite_over_long_mixed_session():
    policy = make_policy()
    t = 0.0
    for i in range(1000):
        phase = i % 100
        speaking0 = phase < 10
        seeking1 = 30 <= phase < 60
        observations = [
            obs(0, seeking=(phase >= 80), speaking=speaking0, speech_confidence=1.0 if speaking0 else 0.0, t=t),
            obs(1, seeking=seeking1, speaking=False, t=t),
        ]
        policy.update(observations, DT)
        for pid in (0, 1):
            score = policy._score_of(pid)
            assert math.isfinite(score), f"non-finite score for person {pid} at tick {i}: {score}"
        t += DT


# --- 8. Active speaker is not overridden by quietness logic ------------------

def test_active_speaker_not_overridden_by_quietness_logic():
    policy = make_policy()
    t = 0.0
    # Confirm person 0 as the active speaker.
    for i in range(6):
        policy.update([obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=False, speaking=False, t=t)], DT)
        t += DT
    assert policy.current_target == 0

    # Person 1 sits there quiet and not looking for a long time, racking up
    # a large quiet_penalty -- this must have zero effect on who holds the
    # target, since person 0 keeps speaking the whole time.
    target = None
    for i in range(50):
        target = policy.update(
            [obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=False, speaking=False, t=t)],
            DT,
        )
        t += DT
    assert target == 0, "quietness logic on OTHER people must never override the active speaker"
    assert policy.quiet_penalty.get(1, 0.0) > 0.0, "sanity check: person 1 did accrue a quiet penalty"
    # And person 0's own quiet_penalty must be exactly 0 the whole time
    # they're speaking (rule 1: speaking -> quiet_duration = 0).
    assert policy.quiet_penalty.get(0, 0.0) == 0.0


# --- 9. No-speaker fallback remains functional --------------------------------

def test_no_speaker_fallback_remains_functional():
    policy = make_policy()
    t = 0.0
    targets = []
    for i in range(30):
        observations = [
            obs(0, seeking=(i // 15) % 2 == 0, speaking=False, t=t),
            obs(1, seeking=(i // 15) % 2 == 1, speaking=False, t=t),
        ]
        targets.append(policy.update(observations, DT))
        t += DT
    assert targets[-1] in (0, 1)
    assert any(v is not None for v in targets)


# --- extra: ConfidenceDebtPolicy baseline still works, unaffected by the move -

def test_confidence_debt_baseline_unaffected_by_greedy_move():
    policy = ConfidenceDebtPolicy(confidence_threshold=0.45, minimum_hold_seconds=0.1)
    t = 0.0
    for i in range(10):
        policy.update([obs(0, seeking=True, t=t)], DT)
        t += DT
    assert policy.current_target == 0
