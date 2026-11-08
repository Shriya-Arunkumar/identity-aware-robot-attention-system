"""Tests for gaze_attention.attention_manager.AttentionManager -- the
conversational/socially-bounded attention layer wrapped around a base
DecisionPolicy (ConfidenceDebtPolicy here). Covers: persistent-speaker
priority, the requirement that mouth motion alone (without looking at the
robot) or a single frame of it can never create speaker priority, the quiet
sustained-eye-contact acknowledgment, returning to a continuing speaker, the
bounded stare penalty / soft gaze break (never a hard abrupt cutoff), full
fallback to confidence-debt arbitration when there is no speaker, and
byte-for-byte-preserved gaze-only behavior when speaker_priority_mode="off".
"""

from __future__ import annotations

import pytest

from gaze_attention.attention_manager import AttentionManager
from gaze_attention.policies import ConfidenceDebtPolicy

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


def make_manager(**overrides):
    base = ConfidenceDebtPolicy(confidence_threshold=0.45, minimum_hold_seconds=0.2, target_grace_seconds=0.2)
    kwargs = dict(
        speaker_priority_mode="on",
        speaker_confirm_seconds=0.3,
        speaker_min_attention_seconds=0.6,
        speaker_release_seconds=0.5,
        quiet_gaze_confirm_seconds=0.8,
        quiet_acknowledgment_seconds=0.4,
        max_continuous_gaze_seconds=1.0,
        gaze_break_seconds=0.5,
        stare_penalty_weight=0.5,
        confidence_threshold=0.45,
    )
    kwargs.update(overrides)
    return AttentionManager(base, **kwargs)


# --- 1. persistent speaker becomes target ------------------------------------

def test_persistent_speaker_becomes_target():
    mgr = make_manager()
    target = None
    for i in range(20):
        observations = [
            obs(0, seeking=True, speaking=True, t=i * DT),
            obs(1, seeking=False, speaking=False, t=i * DT),
        ]
        target = mgr.update(observations, DT)
    assert target == 0
    assert mgr.active_speaker.get(0) is True


# --- 2. mouth motion while looking away does not create speaker priority ----

def test_mouth_motion_while_looking_away_grants_no_priority():
    mgr = make_manager()
    for i in range(40):
        observations = [
            obs(0, seeking=False, speaking=True, t=i * DT),   # talking but NOT looking at robot
            obs(1, seeking=True, speaking=False, t=i * DT),   # the one actually looking
        ]
        target = mgr.update(observations, DT)
    assert mgr.active_speaker.get(0, False) is False
    assert mgr.speaker_candidate_time.get(0, 0.0) == 0.0
    assert target == 1  # fell through to base-policy arbitration, which favors the looker


# --- 3. one-frame mouth motion does not switch target ------------------------

def test_one_frame_mouth_motion_does_not_switch_target():
    mgr = make_manager()
    t = 0.0
    # Establish person 1 as the settled target via ordinary gaze engagement.
    for i in range(15):
        mgr.update([obs(0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
        t += DT
    established = mgr.current_target
    assert established == 1

    # A single tick of speaking+looking from person 0 must not be enough to
    # confirm speaker status (speaker_confirm_seconds=0.3 > one tick=0.1).
    target = mgr.update([obs(0, seeking=True, speaking=True, t=t), obs(1, seeking=True, t=t)], DT)
    t += DT
    assert mgr.active_speaker.get(0, False) is False
    assert target == established

    # And it resets immediately back to 0 the next tick they're not speaking.
    mgr.update([obs(0, seeking=False, speaking=False, t=t), obs(1, seeking=True, t=t)], DT)
    assert mgr.speaker_candidate_time.get(0, 0.0) == 0.0


# --- 4. quiet sustained eye contact triggers a brief acknowledgment ---------

def test_quiet_sustained_eye_contact_triggers_acknowledgment():
    mgr = make_manager()
    t = 0.0
    targets = []
    # Person 0 speaks+looks continuously (becomes, and remains, the active
    # speaker); person 1 stays quiet but holds reliable eye contact the
    # whole time.
    for i in range(30):
        observations = [
            obs(0, seeking=True, speaking=True, t=t),
            obs(1, seeking=True, speaking=False, t=t),
        ]
        targets.append(mgr.update(observations, DT))
        t += DT

    assert 1 in targets, "quiet person with sustained eye contact should get a brief acknowledgment"
    ack_index = targets.index(1)
    # The acknowledgment is brief -- it must not persist for the rest of the run.
    assert not all(v == 1 for v in targets[ack_index:]), "acknowledgment must be brief, not a permanent switch"


# --- 5. continuing speaker regains priority ----------------------------------

def test_continuing_speaker_regains_priority_after_acknowledgment():
    mgr = make_manager()
    t = 0.0
    targets = []
    for i in range(40):
        observations = [
            obs(0, seeking=True, speaking=True, t=t),
            obs(1, seeking=True, speaking=False, t=t),
        ]
        targets.append(mgr.update(observations, DT))
        t += DT

    assert 1 in targets  # acknowledgment happened at some point
    ack_index = targets.index(1)
    # After the acknowledgment window, the continuing speaker (0) must be
    # back in the driver's seat.
    later = targets[ack_index + 5:]
    assert 0 in later
    assert later[-1] == 0


# --- 6. attention penalty prevents indefinite fixation -----------------------

def test_attention_penalty_prevents_indefinite_fixation():
    mgr = make_manager()
    t = 0.0
    targets = []
    for i in range(60):
        observations = [
            obs(0, seeking=True, speaking=True, t=t),   # speaks the entire time, never stops
            obs(1, seeking=True, speaking=False, t=t),  # a credible alternate the whole time
        ]
        targets.append(mgr.update(observations, DT))
        t += DT

    assert any(v != 0 for v in targets), (
        "an uninterrupted speaker must eventually yield the gaze via the bounded "
        "stare penalty / soft gaze break, not hold it forever"
    )
    assert max(mgr.recent_attention_penalty.values(), default=0.0) <= AttentionManager.PENALTY_CAP


# --- 7. active speaker is not abandoned abruptly ------------------------------

def test_active_speaker_not_abandoned_abruptly():
    mgr = make_manager()
    t = 0.0
    targets = []
    # Person 1 never looks at the robot, so only the stare-penalty/gaze-break
    # path is exercised here (not the quiet-acknowledgment path -- that's
    # covered separately by test 4/5).
    for i in range(15):  # well short of full saturation (~1.0s over + 1.0/0.5=2s ramp)
        observations = [
            obs(0, seeking=True, speaking=True, t=t),
            obs(1, seeking=False, speaking=False, t=t),
        ]
        targets.append(mgr.update(observations, DT))
        t += DT

    # Right around/just past crossing max_continuous_gaze_seconds, the
    # speaker must still hold the target -- the penalty ramp (0.5/s cap 1.0
    # => 2s to saturate) means no switch can happen this early.
    assert targets[-1] == 0
    assert 0.0 < mgr.recent_attention_penalty.get(0, 0.0) < AttentionManager.PENALTY_CAP


# --- 8. no speaker falls back to confidence-debt arbitration -----------------

def test_no_speaker_falls_back_to_confidence_debt_arbitration():
    bare = ConfidenceDebtPolicy(confidence_threshold=0.45, minimum_hold_seconds=0.2, target_grace_seconds=0.2)
    mgr = make_manager()

    t = 0.0
    for i in range(40):
        observations = [
            obs(0, seeking=(i // 10) % 2 == 0, speaking=False, t=t),
            obs(1, seeking=(i // 10) % 2 == 1, speaking=False, t=t),
        ]
        bare_target = bare.update(observations, DT)
        mgr_target = mgr.update(observations, DT)
        assert mgr_target == bare_target
        t += DT


# --- 9. gaze-only mode preserves previous behavior ---------------------------

def test_speaker_priority_mode_off_preserves_gaze_only_behavior():
    bare = ConfidenceDebtPolicy(confidence_threshold=0.45, minimum_hold_seconds=0.2, target_grace_seconds=0.2)
    mgr = make_manager(speaker_priority_mode="off")

    t = 0.0
    for i in range(40):
        # Speaking fields present but must have zero effect in "off" mode.
        observations = [
            obs(0, seeking=True, speaking=True, speech_confidence=1.0, t=t),
            obs(1, seeking=(i // 5) % 2 == 0, speaking=False, t=t),
        ]
        bare_target = bare.update(observations, DT)
        mgr_target = mgr.update(observations, DT)
        assert mgr_target == bare_target
        assert mgr.get_metrics(0)["score"] == bare.get_metrics(0)["score"]
        t += DT
