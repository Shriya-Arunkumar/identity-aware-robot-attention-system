"""Tests for the "fast speaker-first path" latency fix in
ConversationalSimplePolicy (gaze_attention/policies.py): live testing found
the previous defaults (speaker_confirm_seconds=0.75, speaker_release_seconds=1.5,
a 5.0s min-attention floor blocking release, and no way for a second
speaker to preempt without the first one silence-releasing first) made
speaker switching noticeably slow. This file exercises the fast
active-speaker-id defaults (speaker_confirm_seconds=0.35,
speaker_release_seconds=0.40, speaker_switch_seconds=0.35,
minimum_speaker_attention_seconds=2.0) and the preemption path, plus the
fast quietness penalty and diagnostic logging fields (requirement 11),
without touching Gazebo or reintroducing GreedyDwellPolicy as the active
policy.

UPDATE (stabilization patch): fast-but-flickery turned out to be the actual
bug report (rapid switching, especially right as the current target starts
speaking) -- see DEBUG_ATTENTION.md. A second layer now sits between
"confirmed active speaker" (still governed by the fast timers above) and
actually moving current_target: a pending-challenger confirmation
(speaker_switch_confirm_seconds, default 0.5s) plus a uniform
minimum_hold_seconds floor (default 1.5s) on the incumbent. The
active-speaker-id bookkeeping this file tests is unchanged; the WALL-CLOCK
latency before current_target itself moves is now longer by design -- the
timing budgets below account for that extra layer.
"""

from __future__ import annotations

import math

from gaze_attention.policies import ConversationalSimplePolicy

DT = 0.05


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
    # Deliberately using the real fast-path DEFAULTS (no overrides) unless
    # a test asks for something different -- these defaults are exactly
    # what's under test here. speech_mode is the one exception: this whole
    # file exercises speaking-driven selection, which now requires
    # speech_mode != "off" (the class default) -- see the speech_mode gate
    # fix in policies.py's _effective_speaking/_effective_speech_confidence.
    kwargs = dict(speech_mode="mouth_motion")
    kwargs.update(overrides)
    return ConversationalSimplePolicy(**kwargs)


def run(policy, ticks, people_fn, t0=0.0):
    """people_fn(i, t) -> list of observations for tick i at time t."""
    t = t0
    targets = []
    for i in range(ticks):
        targets.append(policy.update(people_fn(i, t), DT))
        t += DT
    return targets, t


# --- 1/2. Speaker A stops, speaker B starts, B becomes target within the
#          configured confirmation window ------------------------------------

def test_speaker_a_stops_and_speaker_b_becomes_target_within_confirm_window():
    policy = make_policy()
    t = 0.0

    # A speaks long enough to clear the minimum_speaker_attention_seconds
    # floor (2.0s) so its later release isn't blocked by that floor.
    for i in range(70):  # 3.5s
        policy.update([obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=False, speaking=False, t=t)], DT)
        t += DT
    assert policy.current_target == 0
    assert policy.active_speaker_id == 0

    # A stops; B starts speaking at the same moment.
    switch_tick = None
    worst_case_latency = policy.speaker_release_seconds + policy.speaker_confirm_seconds
    ticks_budget = int(math.ceil(worst_case_latency / DT)) + 4  # small margin
    for i in range(ticks_budget):
        target = policy.update(
            [obs(0, seeking=False, speaking=False, t=t), obs(1, seeking=False, speaking=True, speech_confidence=1.0, t=t)],
            DT,
        )
        t += DT
        if target == 1 and switch_tick is None:
            switch_tick = i
    assert switch_tick is not None, "B must become target within roughly speaker_release_seconds + speaker_confirm_seconds"
    elapsed = (switch_tick + 1) * DT
    assert elapsed <= worst_case_latency + 4 * DT, f"switch took {elapsed:.2f}s, expected within ~{worst_case_latency:.2f}s"
    assert policy.active_speaker_id == 1


# --- 3. Old engagement score cannot block B -----------------------------------

def test_old_accumulated_score_cannot_block_new_speaker():
    policy = make_policy()
    t = 0.0
    # Person 0 saturates its bounded short-term score/debt via ordinary gaze.
    for i in range(400):
        policy.update([obs(0, seeking=True, confidence=0.9, t=t), obs(1, seeking=False, t=t)], DT)
        t += DT
    assert policy.current_target == 0
    assert policy.short_term_score.get(0, 0.0) >= ConversationalSimplePolicy.SCORE_MAX - 1e-6

    target = None
    # speaker_confirm_seconds(0.35s) + speaker_switch_confirm_seconds
    # (0.5s default, stabilization patch) at DT=0.05.
    for i in range(25):
        target = policy.update(
            [obs(0, seeking=True, confidence=0.9, t=t), obs(1, seeking=False, speaking=True, speech_confidence=1.0, t=t)],
            DT,
        )
        t += DT
    assert target == 1, "a maxed-out short-term score must not block a newly confirmed speaker"


# --- 4. A releases quickly after speech stops ---------------------------------

def test_speaker_releases_quickly_after_speech_stops():
    policy = make_policy()
    t = 0.0
    # A speaks well past the minimum_speaker_attention_seconds floor (2.0s).
    for i in range(70):  # 3.5s
        policy.update([obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t)], DT)
        t += DT
    assert policy.active_speaker_id == 0

    # A stops speaking, no one else present/speaking. Release now additionally
    # waits for speech_off_confirm_seconds (0.7s default, stabilization
    # patch) before speaking_true_duration even reaches 0 -- release can't
    # be evaluated until then.
    release_tick = None
    for i in range(40):  # 2.0s budget, generous vs speech_off_confirm_seconds(0.7s) + speaker_release_seconds(0.40s)
        policy.update([obs(0, seeking=False, speaking=False, t=t)], DT)
        t += DT
        if policy.active_speaker_id is None and release_tick is None:
            release_tick = i
    assert release_tick is not None
    elapsed = (release_tick + 1) * DT
    expected_ceiling = policy.speech_off_confirm_seconds + policy.speaker_release_seconds + 4 * DT
    assert elapsed <= expected_ceiling, (
        f"release took {elapsed:.2f}s, expected within ~{expected_ceiling:.2f}s "
        "(speech_off_confirm_seconds must elapse before speaker_release_seconds is even evaluated; "
        "minimum_speaker_attention_seconds floor was already satisfied before A stopped)"
    )


# --- extra: a DIFFERENT already-valid speaker preempts without waiting for
#            the old one's silence-release (requirement 6) -------------------

def test_different_valid_speaker_preempts_without_waiting_for_release():
    policy = make_policy()
    t = 0.0
    # A speaks alone first, becomes active speaker.
    for i in range(10):
        policy.update([obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=False, speaking=False, t=t)], DT)
        t += DT
    assert policy.active_speaker_id == 0

    # B starts speaking too, WHILE A keeps speaking (no silence gap for A at
    # all). speaker_switch_seconds governs active_speaker_id itself; the
    # stabilization patch's speaker_switch_confirm_seconds then gates
    # current_target actually moving to B; and minimum_hold_seconds still
    # applies to the incumbent. The loop length is DERIVED from those timers
    # (plus A's promotion-protection window) rather than hardcoded, so this
    # test asserts "B does preempt without A ever going silent" independent
    # of how the timers are tuned.
    # NOTE: when TWO people speak continuously at once, active_speaker_id (and
    # with it the target) genuinely alternates -- each is repeatedly the
    # "most established challenger" against the other, damped only by
    # promotion_protection_seconds (= speaker_switch_confirm + speaker_confirm).
    # That is pre-existing behavior, not introduced by any timer change:
    # measured over the same window it flips 4x at the old 0.35/0.35/0.5
    # defaults and 8x at 0.25/0.25/0.20, i.e. proportional to the protection
    # window. So this test asserts what it is actually named for -- that B DOES
    # take over while A is still speaking, never having gone silent -- rather
    # than snapshotting one arbitrary tick of an oscillating signal.
    protection = policy.speaker_switch_confirm_seconds + policy.speaker_confirm_seconds
    needed = (protection + policy.speaker_switch_seconds
              + policy.speaker_switch_confirm_seconds + policy.minimum_hold_seconds)
    b_was_active_speaker = False
    b_was_target = False
    for i in range(int(needed / DT) + 10):
        target = policy.update(
            [obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=False, speaking=True, speech_confidence=1.0, t=t)],
            DT,
        )
        b_was_active_speaker = b_was_active_speaker or policy.active_speaker_id == 1
        b_was_target = b_was_target or target == 1
        t += DT
    assert b_was_active_speaker, (
        "B must become the active speaker while A is still speaking -- "
        "no silence-release required"
    )
    assert b_was_target, "B must actually take the target, not merely become active speaker"
    # A never went silent at any point in the loop above.
    assert policy.speaking_true_duration.get(0, 0.0) > 0.0


# --- 5. Quiet person's score decays -------------------------------------------

def test_quiet_persons_score_decays():
    policy = make_policy()
    t = 0.0
    for i in range(20):
        policy.update([obs(0, seeking=True, confidence=0.9, t=t)], DT)
        t += DT
    grown = policy.short_term_score[0]
    assert grown > 0.0

    for i in range(20):
        policy.update([obs(0, seeking=False, speaking=False, confidence=0.9, t=t)], DT)
        t += DT
    decayed = policy.short_term_score[0]
    assert decayed < grown
    assert decayed >= 0.0


# --- 6. Quiet eye-contact person is not over-penalized ------------------------

def test_quiet_eye_contact_person_not_over_penalized():
    policy = make_policy()
    t = 0.0
    for i in range(60):  # 3s of sustained eye contact, never speaking
        policy.update([obs(0, seeking=True, speaking=False, t=t)], DT)
        t += DT
    assert policy.quiet_penalty.get(0, 0.0) == 0.0, "sustained eye contact must never accrue a quiet penalty"


# --- 7. One-frame mouth motion does not switch --------------------------------

def test_one_frame_mouth_motion_does_not_switch():
    policy = make_policy()
    t = 0.0
    # minimum_hold_seconds default is now 1.5s (stabilization patch) -- needs
    # enough ticks for the fallback switch away from the initial tie-break
    # pick to clear that floor.
    for i in range(35):
        policy.update([obs(0, seeking=False, t=t), obs(1, seeking=True, t=t)], DT)
        t += DT
    established = policy.current_target
    assert established == 1

    target = policy.update(
        [obs(0, seeking=False, speaking=True, speech_confidence=1.0, t=t), obs(1, seeking=True, t=t)], DT
    )
    t += DT
    assert target == established, "a single frame of speaking is far short of speaker_confirm_seconds"
    assert policy.active_speaker_id is None

    # Requirement 4 (stabilization patch): one non-speaking frame must not
    # immediately reset speaking_true_duration -- it stays frozen until
    # speech_off_confirm_seconds (0.7s default) of continuous silence.
    policy.update([obs(0, seeking=False, speaking=False, t=t), obs(1, seeking=True, t=t)], DT)
    assert policy.speaking_true_duration.get(0, 0.0) == DT


# --- 8. Target grace still works for ordinary non-speaker misses -------------

def test_target_grace_unchanged_for_ordinary_non_speaker_misses():
    policy = make_policy(minimum_hold_seconds=0.3, target_grace_seconds=0.3)
    t = 0.0
    for i in range(20):
        policy.update([obs(0, seeking=True, t=t), obs(1, seeking=False, t=t)], DT)
        t += DT
    assert policy.current_target == 0

    # Person 0 vanishes for a single tick (a blink/detection blip) -- must
    # NOT immediately clear/reselect the target (this is the ordinary,
    # unrelated-to-speech target_grace_seconds mechanism, unchanged by the
    # fast speaker-first path work).
    target = policy.update([obs(1, seeking=True, t=t)], DT)
    t += DT
    assert target == 0, "a single-frame miss must not immediately release the target (target_grace_seconds)"

    # Person 0 comes back before the grace period elapses -- must still be
    # held as target with no disruption.
    target = policy.update([obs(0, seeking=True, t=t), obs(1, seeking=True, t=t)], DT)
    assert target == 0


# --- 11. Diagnostics: switch reason / latency / per-person timers ------------

def test_switch_diagnostics_are_populated():
    policy = make_policy()
    t = 0.0
    target = policy.update([obs(0, seeking=True, t=t)], DT)
    assert target == 0
    assert policy.last_switch_reason == "initial"

    t += DT
    # confirm person 1 as speaker (0.35s) + pending-challenger switch-confirm
    # (0.5s) -- but person 0 (the incumbent) stays valid throughout, so the
    # uniform minimum_hold_seconds floor (1.5s default, stabilization patch)
    # on person 0's own dwell time is actually the binding constraint here.
    for i in range(35):
        target = policy.update([obs(0, seeking=True, t=t), obs(1, seeking=False, speaking=True, speech_confidence=1.0, t=t)], DT)
        t += DT
    assert target == 1
    assert policy.last_switch_reason in ("speaker_confirmed", "speaker_confirmed_after_release", "speaker_preempt")
    assert policy.last_switch_latency_seconds > 0.0
    assert policy.speaker_confirmed_time.get(1) is not None

    metrics = policy.get_metrics(1)
    assert metrics["is_active_speaker"] == 1.0
    assert metrics["speaking_true_duration"] > 0.0


# --- scores remain finite over a long mixed session ---------------------------

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


# --- switched_this_tick is only True on the actual switching frame -----------

def test_switched_this_tick_is_only_true_on_the_switching_frame():
    """A prior bug had last_switch_reason "stick" forever after the first
    switch, so every later CSV row would spuriously show a switch reason.
    switched_this_tick must be True for exactly one tick per switch."""
    policy = make_policy()
    t = 0.0
    switched_ticks = []
    for i in range(30):
        policy.update([obs(0, seeking=True, t=t), obs(1, seeking=False, t=t)], DT)
        if policy.switched_this_tick:
            switched_ticks.append(i)
        t += DT
    assert switched_ticks == [0], "only the very first (initial) assignment should switch in this scenario"

    # Now force a real speaker-driven switch and confirm it's isolated too.
    # Needs speaker_confirm_seconds(0.35s) + speaker_switch_confirm_seconds
    # (0.5s default, stabilization patch) at DT=0.05.
    switched_ticks = []
    for i in range(25):
        policy.update([obs(0, seeking=True, t=t), obs(1, seeking=False, speaking=True, speech_confidence=1.0, t=t)], DT)
        if policy.switched_this_tick:
            switched_ticks.append(i)
        t += DT
    assert len(switched_ticks) == 1, f"expected exactly one switching tick, got {switched_ticks}"
