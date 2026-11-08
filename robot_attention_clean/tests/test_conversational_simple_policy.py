"""Tests for gaze_attention.policies.ConversationalSimplePolicy -- the
active, speaker-first conversational policy. Covers the specific bug this
policy exists to fix (a stuck target ignoring a visibly-talking second
person because of accumulated lifetime engagement), the requirement that
looking at the robot is NOT needed to be recognized as speaking, debounced
one-frame-safe speaker confirmation, speaker persistence/release/resumption,
and the bounded, non-abrupt stare penalty on the fallback (non-speaker)
path.
"""

from __future__ import annotations

import pytest

from gaze_attention.policies import ConfidenceDebtPolicy, ConversationalSimplePolicy

DT = 0.1


def obs(id_, seeking=True, confidence=0.9, speaking=False, speech_confidence=0.0, t=0.0, lost=False):
    return {
        "id": id_,
        "bbox": [0, 0, 10, 10],
        "gaze_signal": 0.9 if seeking else 0.1,
        "confidence": confidence,
        "seeking_attention": seeking,
        "timestamp": t,
        "speaking": speaking,
        "speech_confidence": speech_confidence,
        "lost": lost,
    }


def make_policy(**overrides):
    kwargs = dict(
        confidence_threshold=0.45,
        minimum_hold_seconds=0.75,
        target_grace_seconds=0.5,
        # speech_mode="mouth_motion": every test below that exercises
        # speaking/speech_confidence-driven selection needs this policy to
        # actually consume them -- speech_mode="off" (the class default)
        # now makes it treat speaking as always False (see
        # _effective_speaking/_effective_speech_confidence, the speech_mode
        # gate fix). Tests that specifically want to verify the "off"
        # behavior override this explicitly.
        speech_mode="mouth_motion",
        speaker_confirm_seconds=0.3,
        speaker_release_seconds=0.5,
        minimum_speaker_attention_seconds=0.2,
        speaker_confidence_threshold=0.25,
        non_speaking_score_decay=0.5,
        recent_attention_penalty_weight=0.5,
        max_continuous_gaze_seconds=1.0,
    )
    kwargs.update(overrides)
    return ConversationalSimplePolicy(**kwargs)


# --- 1. Person 2 speaking switches away from Person 1 ------------------------

def test_person_2_speaking_switches_away_from_stuck_person_1():
    """The exact reported bug: Person 1 sits as the settled target (built up
    ordinary gaze engagement well past minimum_hold_seconds), then Person 2
    starts visibly speaking -- the old ConfidenceDebtPolicy+_speech_bonus
    design could leave Person 1 selected for 10-15s+. This policy must
    still switch to Person 2 -- but NOT instantly: the stabilization patch
    (requirements 1/2) requires Person 2 to be confirmed speaking
    (speaker_confirm_seconds=0.3s) AND then remain the SAME pending
    challenger for speaker_switch_confirm_seconds before the switch fires.

    Both windows are DERIVED from the policy's own timers rather than
    hardcoded, so this stays correct whatever those timers are configured to
    -- it asserts the required ORDERING (confirm, then switch-confirm, then
    switch), which is the actual behavior under test."""
    policy = make_policy()
    t = 0.0

    # Person 1 becomes, and stays, the settled target via ordinary gaze.
    for i in range(20):
        policy.update([obs(0, seeking=True, t=t), obs(1, seeking=False, t=t)], DT)
        t += DT
    assert policy.current_target == 0

    # Person 2 starts speaking (not required to look at the robot). Must NOT
    # switch before speaker_confirm + speaker_switch_confirm have elapsed --
    # this is precisely the flicker the stabilization patch prevents. One
    # tick of slack keeps the boundary tick itself out of the assertion.
    too_soon = policy.speaker_confirm_seconds + policy.speaker_switch_confirm_seconds
    ticks_too_soon = max(1, int(too_soon / DT) - 1)
    target = None
    for i in range(ticks_too_soon):
        target = policy.update(
            [obs(0, seeking=True, t=t), obs(1, seeking=False, speaking=True, speech_confidence=1.0, t=t)],
            DT,
        )
        t += DT
    assert target == 0, "must NOT preempt within speaker_confirm+switch_confirm -- that's the flicker fix"

    # After enough continuous, confirmed speaking, the switch does happen.
    for i in range(int(too_soon / DT) + 15):
        target = policy.update(
            [obs(0, seeking=True, t=t), obs(1, seeking=False, speaking=True, speech_confidence=1.0, t=t)],
            DT,
        )
        t += DT
    assert target == 1, "must eventually switch to the visibly-speaking person once fully confirmed"


# --- 2. speaking while looking away still creates speaker priority -----------

def test_speaking_while_looking_away_still_creates_speaker_priority():
    policy = make_policy()
    t = 0.0
    target = None
    for i in range(10):
        target = policy.update(
            [obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=True, t=t)],
            DT,
        )
        t += DT
    assert target == 0, "seeking_attention must NOT be required for speaker priority"


# --- 3. one-frame mouth motion does not switch --------------------------------

def test_one_frame_mouth_motion_does_not_switch():
    policy = make_policy()
    t = 0.0
    for i in range(15):
        policy.update([obs(0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
        t += DT
    established = policy.current_target
    assert established == 1

    target = policy.update(
        [obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=True, t=t)], DT
    )
    t += DT
    assert target == established, "a single frame of speaking is short of speaker_confirm_seconds"

    # Requirement 4: one non-speaking frame must NOT immediately reset
    # speaking_true_duration -- it stays frozen (not reset) until
    # speech_off_confirm_seconds (0.7s default) of continuous silence.
    policy.update([obs(0, seeking=False, speaking=False, t=t), obs(1, seeking=True, t=t)], DT)
    assert policy.speaking_true_duration.get(0, 0.0) == DT, (
        "a single false frame must not reset the confirmation clock -- it should still hold the prior value"
    )


# --- 4. speaker remains selected while speaking -------------------------------

def test_speaker_remains_selected_while_speaking():
    policy = make_policy()
    t = 0.0
    targets = []
    for i in range(40):
        targets.append(policy.update(
            [obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=True, t=t)], DT
        ))
        t += DT
    confirm_index = next(i for i, v in enumerate(targets) if v == 0)
    assert all(v == 0 for v in targets[confirm_index:]), "an uninterrupted speaker must not lose the target"


# --- 5. speaker release returns to gaze selection -----------------------------

def test_speaker_release_returns_to_gaze_selection():
    policy = make_policy()
    t = 0.0
    # Confirm person 0 as speaker.
    for i in range(6):
        policy.update([obs(0, speaking=True, speech_confidence=1.0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
        t += DT
    assert policy.current_target == 0

    # Person 0 stops speaking; person 1 is the only one looking at the robot.
    target = None
    for i in range(15):  # > speaker_release_seconds(0.5s) + minimum_speaker_attention_seconds(0.2s)
        target = policy.update([obs(0, speaking=False, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
        t += DT
    assert target == 1, "after release, short-term gaze selection must take over"


# --- 6. resumed speaker regains priority --------------------------------------

def test_resumed_speaker_regains_priority():
    policy = make_policy()
    t = 0.0
    # Confirm and then release person 0 as speaker.
    for i in range(6):
        policy.update([obs(0, speaking=True, speech_confidence=1.0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
        t += DT
    for i in range(15):
        policy.update([obs(0, speaking=False, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
        t += DT
    assert policy.current_target == 1

    # Person 0 resumes speaking -- needs speaker_confirm_seconds (0.3s) to
    # become active_speaker_id again, THEN speaker_switch_confirm_seconds
    # (0.5s default) as pending challenger against person 1 (the current
    # target), same stabilization gate as the initial-switch case.
    target = None
    for i in range(15):
        target = policy.update(
            [obs(0, speaking=True, speech_confidence=1.0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT
        )
        t += DT
    assert target == 0, "a resumed speaker must regain priority after re-confirmation + switch-confirm"


# --- 7. old accumulated engagement cannot block a valid speaker --------------

def test_old_accumulated_engagement_cannot_block_a_valid_speaker():
    """Person 0 racks up a huge amount of short-term score/debt (as much as
    the bounded accumulators allow) before person 1 starts speaking -- a
    large existing score must not be able to outweigh a confirmed speaker,
    unlike the old _speech_bonus design this policy replaces."""
    policy = make_policy()
    t = 0.0
    for i in range(400):  # long enough to saturate short_term_score/attention_debt at their caps
        policy.update([obs(0, seeking=True, confidence=0.9, t=t), obs(1, seeking=False, t=t)], DT)
        t += DT
    assert policy.current_target == 0
    assert policy.short_term_score.get(0, 0.0) >= ConversationalSimplePolicy.SCORE_MAX - 1e-6

    target = None
    for i in range(15):  # needs speaker_confirm(0.3s) + switch_confirm(0.5s), same stabilization gate
        target = policy.update(
            [obs(0, seeking=True, confidence=0.9, t=t), obs(1, seeking=False, speaking=True, speech_confidence=1.0, t=t)],
            DT,
        )
        t += DT
    assert target == 1, "a maxed-out short-term score must not block a newly confirmed speaker"


# --- 8. non-speaking, look-away person's score decays -------------------------

def test_non_speaking_look_away_scores_decay():
    policy = make_policy()
    t = 0.0
    for i in range(20):
        policy.update([obs(0, seeking=True, confidence=0.9, t=t), obs(1, seeking=False, t=t)], DT)
        t += DT
    grown = policy.short_term_score[0]
    assert grown > 0.0

    for i in range(20):
        policy.update([obs(0, seeking=False, speaking=False, confidence=0.9, t=t), obs(1, seeking=False, t=t)], DT)
        t += DT
    decayed = policy.short_term_score[0]
    assert decayed < grown
    assert decayed >= 0.0  # clamped at zero, never negative


# --- 9. active speaker is not abandoned solely due to stare penalty ----------

def test_active_speaker_not_abandoned_solely_due_to_stare_penalty():
    """Requirement 7/9: the soft stare penalty only ever acts on the
    non-speaker fallback path. An uninterrupted, continuously-speaking
    person must never be forced off the target by attention_time/penalty
    bookkeeping alone."""
    policy = make_policy(max_continuous_gaze_seconds=0.3, recent_attention_penalty_weight=1.0)  # aggressive settings
    t = 0.0
    targets = []
    for i in range(60):  # far past max_continuous_gaze_seconds -- would trigger a break on the fallback path
        targets.append(policy.update(
            [obs(0, speaking=True, speech_confidence=1.0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT
        ))
        t += DT
    confirm_index = next(i for i, v in enumerate(targets) if v == 0)
    assert all(v == 0 for v in targets[confirm_index:]), (
        "the active speaker must be held regardless of attention_time/stare-penalty settings"
    )


# --- lost=True challenger-eligibility guard -----------------------------------

def test_lost_observation_cannot_become_active_speaker_or_target():
    """Mirrors tests/test_target_grace_period.py's dwell_arbitration guard
    test: a lost=True observation must never become the active speaker or
    the target, and must never displace a lower-scoring visible person who
    already holds it -- even while continuously "speaking" with full
    confidence, which would otherwise win outright and immediately (this
    policy's speaker path bypasses minimum_hold_seconds entirely)."""
    policy = make_policy()
    t = 0.0

    # Person 1 becomes, and holds, the target via ordinary gaze (fallback path).
    for _ in range(20):
        policy.update([obs(1, seeking=True, t=t)], DT)
        t += DT
    assert policy.current_target == 1

    # Person 0 "speaks" continuously and confidently, but lost=True -- long
    # enough to blow past speaker_confirm_seconds + speaker_switch_confirm_seconds
    # many times over, which would otherwise make them the active speaker
    # and immediately take the target.
    target = None
    for _ in range(60):
        target = policy.update(
            [
                obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t, lost=True),
                obs(1, seeking=True, t=t),
            ],
            DT,
        )
        t += DT

    assert policy.active_speaker_id != 0, "a lost=True observation must never become the active speaker"
    assert target == 1, "a lost=True observation must never become or displace the target"
    assert policy.current_target == 1


# --- extra: sanity check the bounded fallback isn't literally unbounded ------

def test_fallback_short_term_score_is_bounded_not_unbounded_dwell():
    policy = make_policy()
    t = 0.0
    for i in range(2000):
        policy.update([obs(0, seeking=True, confidence=0.9, t=t)], DT)
        t += DT
    assert policy.short_term_score[0] <= ConversationalSimplePolicy.SCORE_MAX


# --- sanity: greedy/confidence_debt remain available, unaffected -------------

def test_confidence_debt_baseline_still_works_unaffected():
    bare = ConfidenceDebtPolicy(confidence_threshold=0.45, minimum_hold_seconds=0.75, target_grace_seconds=0.5)
    t = 0.0
    for i in range(20):
        bare.update([obs(0, seeking=True, t=t), obs(1, seeking=False, t=t)], DT)
        t += DT
    assert bare.current_target == 0


# ===========================================================================
# Stabilization patch tests (requirement 9) -- see DEBUG_ATTENTION.md flicker
# diagnosis and policies.py's _resolve_target docstring.
# ===========================================================================

def _settle_person_1(policy, t, frames=20):
    """Helper: person 1 becomes, and stays, the settled non-speaking target."""
    for _ in range(frames):
        policy.update([obs(0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
        t += DT
    assert policy.current_target == 1
    return t


# --- a. one-frame challenger blip does not switch -----------------------------

def test_one_frame_challenger_blip_does_not_switch():
    policy = make_policy()
    t = _settle_person_1(policy, 0.0)

    target = policy.update(
        [obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=True, t=t)], DT
    )
    t += DT
    assert target == 1, "a single-frame speaking blip must not switch the target"
    # Follow with silence -- the blip must never accumulate toward a switch.
    for _ in range(10):
        target = policy.update([obs(0, seeking=False, speaking=False, t=t), obs(1, seeking=True, t=t)], DT)
        t += DT
    assert target == 1


# --- b. current target is held for 1.5 seconds --------------------------------

def test_current_target_held_for_1_5_seconds_default():
    """With default minimum_hold_seconds (1.5s), a confirmed challenger must
    not preempt the incumbent before 1.5s of incumbent dwell time has
    elapsed, even once the challenger itself is fully confirmed and has
    satisfied speaker_switch_confirm_seconds well before that."""
    policy = ConversationalSimplePolicy(
        speech_mode="mouth_motion",
        speaker_confirm_seconds=0.3,
        speaker_switch_confirm_seconds=0.3,  # much shorter than the 1.5s hold floor
        speaker_confidence_threshold=0.25,
    )  # minimum_hold_seconds left at the new 1.5s default -- the gate under test
    t = _settle_person_1(policy, 0.0, frames=20)
    dwell_at_loop_start = policy._time_since_switch
    expected_min_wait = max(0.0, policy.minimum_hold_seconds - dwell_at_loop_start)

    target = 1
    ticks = 0
    while target == 1 and ticks < 40:
        target = policy.update(
            [obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=True, t=t)], DT
        )
        t += DT
        ticks += 1
    elapsed_seconds = ticks * DT
    assert target == 0, "must eventually switch once all conditions are satisfied"
    assert elapsed_seconds >= expected_min_wait - 1e-6, (
        f"switched after only {elapsed_seconds:.2f}s but the incumbent's minimum_hold_seconds "
        f"(1.5s) floor required at least {expected_min_wait:.2f}s more dwell time -- the fast "
        f"speaker_confirm/switch_confirm timers must not be able to bypass it"
    )
    assert elapsed_seconds <= expected_min_wait + 0.3, "switched much later than the hold floor required"


# --- c. challenger must remain valid for 0.5 seconds ---------------------------

def test_challenger_must_remain_valid_for_0_5_seconds():
    policy = make_policy(minimum_hold_seconds=0.0)  # isolate the switch-confirm gate from the hold gate
    t = _settle_person_1(policy, 0.0)

    # Drive ticks until person 0 is first tracked as a pending challenger
    # (i.e. has just been confirmed as active speaker, speaker_confirm_seconds=0.3s).
    target = 1
    ticks_to_promotion = 0
    while policy.pending_challenger_id != 0 and ticks_to_promotion < 20:
        target = policy.update(
            [obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=True, t=t)], DT
        )
        t += DT
        ticks_to_promotion += 1
    assert policy.pending_challenger_id == 0, "person 0 should now be tracked as a pending challenger"
    assert target == 1, "becoming a confirmed active speaker must not itself bypass switch-confirm"

    # Continue until the switch actually fires; the pending-challenger
    # window alone (from promotion to switch) must be >= speaker_switch_confirm_seconds.
    ticks_pending = 0
    while target == 1 and ticks_pending < 20:
        target = policy.update(
            [obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=True, t=t)], DT
        )
        t += DT
        ticks_pending += 1
    assert target == 0, "must eventually switch"
    pending_elapsed = ticks_pending * DT
    assert pending_elapsed >= policy.speaker_switch_confirm_seconds - 1e-6, (
        f"switched after only {pending_elapsed:.2f}s as pending challenger, short of "
        f"speaker_switch_confirm_seconds ({policy.speaker_switch_confirm_seconds}s)"
    )


# --- d. one false speaking frame does not reset the timer -----------------------

def test_one_false_speaking_frame_does_not_reset_speaking_true_duration():
    policy = make_policy()
    t = 0.0
    policy.update([obs(0, speaking=True, speech_confidence=1.0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
    t += DT
    policy.update([obs(0, speaking=True, speech_confidence=1.0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
    t += DT
    accumulated = policy.speaking_true_duration[0]
    assert accumulated == pytest.approx(2 * DT)

    policy.update([obs(0, speaking=False, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
    t += DT
    assert policy.speaking_true_duration[0] == pytest.approx(accumulated), (
        "one non-speaking frame must not reset speaking_true_duration -- it should stay frozen"
    )


# --- e. silence longer than 0.7 seconds resets speaking state -------------------

def test_silence_longer_than_0_7_seconds_resets_speaking_state():
    policy = make_policy()
    t = 0.0
    policy.update([obs(0, speaking=True, speech_confidence=1.0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
    t += DT
    assert policy.speaking_true_duration[0] > 0.0

    for i in range(6):  # 0.6s of silence -- under speech_off_confirm_seconds (0.7s)
        policy.update([obs(0, speaking=False, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
        t += DT
    assert policy.speaking_true_duration[0] > 0.0, "must still be frozen (not reset) under 0.7s of silence"

    for i in range(3):  # pushes total silence past 0.7s
        policy.update([obs(0, speaking=False, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
        t += DT
    assert policy.speaking_true_duration[0] == 0.0, "must reset once 0.7s of confirmed silence has elapsed"


# --- f. temporary target disappearance shorter than 0.8s is held ----------------

def test_temporary_active_speaker_disappearance_under_0_8s_is_held():
    policy = make_policy()
    t = 0.0
    target = None
    for i in range(6):  # confirm person 0 as speaker and current target
        target = policy.update(
            [obs(0, speaking=True, speech_confidence=1.0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT
        )
        t += DT
    assert target == 0

    # Person 0 disappears from observations for under 0.8s (speaker_absence_grace_seconds).
    for i in range(7):  # 0.7s -- under the 0.8s grace
        target = policy.update([obs(1, seeking=True, t=t)], DT)
        t += DT
    assert target == 0, "a temporarily absent active speaker must be held through the absence grace"

    # Person 0 reappears, still speaking -- must still be held/regain seamlessly.
    target = policy.update(
        [obs(0, speaking=True, speech_confidence=1.0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT
    )
    assert target == 0


# --- g. target switches after all conditions are satisfied ----------------------

def test_target_switches_after_all_conditions_satisfied():
    policy = ConversationalSimplePolicy(
        speech_mode="mouth_motion",
        minimum_hold_seconds=1.5,
        speaker_confirm_seconds=0.3,
        speaker_switch_confirm_seconds=0.5,
        speaker_confidence_threshold=0.25,
    )
    t = _settle_person_1(policy, 0.0, frames=40)  # 4.0s -- well past the 1.5s hold floor

    # Now person 0 speaks continuously -- confirm (0.3s) + switch-confirm
    # (0.5s) -- and the incumbent's hold floor is already long satisfied,
    # so this is the binding path this time.
    target = 1
    ticks = 0
    while target == 1 and ticks < 30:
        target = policy.update(
            [obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=True, t=t)], DT
        )
        t += DT
        ticks += 1
    elapsed = ticks * DT
    assert target == 0, "switches once confirm + switch-confirm + hold are all satisfied"
    assert elapsed >= policy.speaker_confirm_seconds + policy.speaker_switch_confirm_seconds - 1e-6, (
        "switched before speaker_confirm_seconds + speaker_switch_confirm_seconds had elapsed"
    )
    assert policy.last_switch_reason == "speaker_preempt"
    assert policy.last_decision_log["switched_this_tick"] is True


# ===========================================================================
# speech_mode gate (bug fix): perception.py's mouth-motion visual_speaking_proxy
# computes speaking/speech_confidence unconditionally every frame (see its
# own docstring) -- this policy previously read obs["speaking"]/
# obs["speech_confidence"] directly everywhere, completely ignoring
# speech_mode, so speech_mode="off" (the documented "gaze-only" setting) had
# no effect on the active policy at all. Fixed at the point of consumption
# (_effective_speaking/_effective_speech_confidence): speech_mode="off" now
# makes this policy behave as if no one is ever speaking, regardless of what
# perception.py measured.
# ===========================================================================

def test_speech_mode_off_speaking_true_never_triggers_preempt_or_selection():
    """With speech_mode="off": a person who is speaking=True with maximal
    speech_confidence for many seconds must never become the active speaker,
    never preempt the incumbent, and never be selected as target purely on
    that basis -- as if speaking/speech_confidence weren't in the
    observation at all."""
    policy = make_policy(
        speech_mode="off",
        speaker_confirm_seconds=0.3,
        speaker_switch_confirm_seconds=0.3,
        speaker_confidence_threshold=0.25,
        minimum_hold_seconds=0.5,
    )
    t = 0.0

    # Person 1 becomes the settled target via ordinary gaze (not speaking).
    for i in range(10):
        policy.update([obs(0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
        t += DT
    assert policy.current_target == 1

    # Person 0 now speaks continuously with maximal confidence for well
    # past every speaker-path timing budget (confirm + switch-confirm +
    # hold, several times over). With speech_mode="on" this would have
    # preempted long ago (see test_target_switches_after_all_conditions_satisfied,
    # which uses the same timings and DOES switch).
    for i in range(60):  # 6.0s -- far past any speaker-path budget
        target = policy.update(
            [obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=True, t=t)],
            DT,
        )
        t += DT

    assert target == 1, "speech_mode='off' must prevent a speaking person from ever preempting via the speaker path"
    assert policy.active_speaker_id is None, "no one should ever become the active speaker with speech_mode='off'"
    assert policy.speaker_candidate_id is None, "no speaker candidate should ever be computed with speech_mode='off'"
    assert policy.speaking_true_duration.get(0, 0.0) == 0.0, (
        "speaking_true_duration for the speaking person must never accumulate -- "
        "speech_mode='off' treats speaking as always False at the point of consumption"
    )
    assert policy.last_switch_reason != "speaker_preempt"
    assert policy.last_switch_reason != "speaker_confirmed"


def test_speech_mode_off_vs_mouth_motion_same_observations_diverge():
    """Same exact observation sequence, only speech_mode differs: 'off'
    must never switch to the speaker; 'mouth_motion' must switch, proving
    the gate is the actual deciding factor (not some other config
    difference)."""
    def sequence(policy):
        t = 0.0
        for i in range(10):
            policy.update([obs(0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
            t += DT
        target = None
        for i in range(30):
            target = policy.update(
                [obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=True, t=t)],
                DT,
            )
            t += DT
        return target

    shared = dict(
        speaker_confirm_seconds=0.3,
        speaker_switch_confirm_seconds=0.3,
        speaker_confidence_threshold=0.25,
        minimum_hold_seconds=0.5,
    )
    target_off = sequence(make_policy(speech_mode="off", **shared))
    target_on = sequence(make_policy(speech_mode="mouth_motion", **shared))

    assert target_off == 1, "speech_mode='off' -- must stay on the gaze-selected target"
    assert target_on == 0, "speech_mode='mouth_motion' -- must switch to the confirmed speaker"
