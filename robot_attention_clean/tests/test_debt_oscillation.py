"""Root Cause #1 -- the debt-driven target oscillator.

Diagnosis (reproduced deterministically before the fix): in
ConversationalSimplePolicy's fallback (no-active-speaker) path, two equally
engaged people accumulate an IDENTICAL short_term_score, but attention_debt
is force-reset to 0 for whoever currently holds the target and grows at
+dt for everyone else. Since score = short_term + DEBT_WEIGHT*debt and the
switch test was a bare `>`, the challenger's score was STRICTLY greater than
the incumbent's every single tick by construction -- so a switch fired the
instant minimum_hold_seconds expired, then the roles inverted, forever.
Measured period ~0.8s (= minimum_hold_seconds 0.75s quantized to 24 frames
at 30fps), between two SILENT people, with no speech, mouth noise, or
confidence difference involved at all.

Fix: `fallback_switch_margin` (challenger must beat the incumbent by MORE
than the margin, not merely exceed it) plus `debt_requires_speech` (debt is
excluded from the fallback switch COMPARISON while neither side has
confirmed speech evidence -- it is still accrued and still reported).

These tests use persistent ids 12 (Person A) / 18 (Person B) to match the
live diagnostic scenarios.
"""

from __future__ import annotations

import pytest

from gaze_attention.policies import ConversationalSimplePolicy

A, B = 12, 18
DT = 1.0 / 30.0


def obs(pid, seeking=True, confidence=0.9, speaking=False, speech_confidence=0.0, t=0.0, lost=False):
    return {
        "id": pid,
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
        speech_mode="mouth_motion",
        speaker_confirm_seconds=0.35,
        speaker_release_seconds=0.4,
        speaker_switch_seconds=0.35,
        minimum_speaker_attention_seconds=2.0,
        speaker_confidence_threshold=0.25,
        fallback_switch_margin=0.10,
        debt_requires_speech=True,
        # The bounded stare penalty deliberately rotates the target once
        # someone has held it for max_continuous_gaze_seconds (default 8.0),
        # which is a SEPARATE, intended mechanism from the debt oscillator
        # under test here -- at the default it would fire around t=8.5s and
        # confound the 10s no-speaker runs below. Pushed out of range so
        # these tests measure debt behavior only; the stare rotation keeps
        # its own dedicated test (test_stare_penalty_rotation_is_preserved).
        max_continuous_gaze_seconds=60.0,
    )
    kwargs.update(overrides)
    return ConversationalSimplePolicy(**kwargs)


def run(policy, duration, speak_fn, seek_fn=None):
    """Drive `policy` for `duration` synthetic seconds. Returns the list of
    (t, from_id, to_id, reason) switch events."""
    t, switches, prev = 0.0, [], None
    for _ in range(int(duration / DT)):
        sa, sb = speak_fn(t)
        ka, kb = seek_fn(t) if seek_fn else (True, True)
        target = policy.update(
            [obs(A, seeking=ka, speaking=sa, speech_confidence=1.0 if sa else 0.0, t=t),
             obs(B, seeking=kb, speaking=sb, speech_confidence=1.0 if sb else 0.0, t=t)],
            DT,
        )
        if target != prev:
            switches.append((round(t, 3), prev, target, policy.last_switch_reason))
            prev = target
        t += DT
    return switches


# --- Scenario F: nobody speaks -- the regression that motivated the fix ------

def test_scenario_f_no_speaker_no_ping_pong_over_10_seconds():
    """THE bug: two silent, equally-engaged people ping-ponged every ~0.8s.
    Must now settle on one target and hold it for the full 10s."""
    policy = make_policy()
    switches = run(policy, 10.0, lambda t: (False, False))

    assert len(switches) == 1, (
        f"expected only the initial selection, got {len(switches)} switches: {switches}"
    )
    assert switches[0][1] is None and switches[0][3] == "initial"
    assert policy.active_speaker_id is None, "nobody is speaking -- there must be no active speaker"
    assert policy.no_speaker_hold is True, "the no-speaker path must be explicitly recorded"


def test_scenario_f_debt_still_accrues_but_does_not_switch():
    """Debt must NOT be deleted or zeroed by the fix -- only prevented from
    causing a switch. The non-target must still visibly accumulate debt."""
    policy = make_policy()
    switches = run(policy, 10.0, lambda t: (False, False))
    held = switches[0][2]
    other = B if held == A else A

    assert policy.attention_debt.get(other, 0.0) > 1.0, (
        "the overlooked person must still accrue debt -- the fix gates debt's "
        "USE in the switch comparison, it does not remove debt"
    )
    assert policy.attention_debt.get(held, 0.0) == pytest.approx(0.0), "the target's debt is still reset"


def test_debt_only_challenger_cannot_take_the_target():
    """Equal engagement, challenger has strictly larger debt, neither is
    speaking -> the challenger must not win."""
    policy = make_policy()
    run(policy, 5.0, lambda t: (False, False))
    held = policy.current_target
    other = B if held == A else A

    # Precondition: this is genuinely a debt-only advantage.
    assert policy.attention_debt[other] > policy.attention_debt[held]
    assert policy.short_term_score[other] == pytest.approx(policy.short_term_score[held], abs=1e-9)

    for _ in range(300):  # 10 more seconds
        target = policy.update([obs(A, speaking=False), obs(B, speaking=False)], DT)
        assert target == held, "a debt-only challenger must never take the target"


def test_debt_is_excluded_from_the_comparison_when_nobody_speaks():
    """The diagnostic record must show debt was actually excluded, not that
    the switch happened to be blocked by the margin alone."""
    policy = make_policy()
    run(policy, 5.0, lambda t: (False, False))
    d = policy.last_fallback_decision

    assert d["debt_counted"] is False
    assert d["current_speaking_state"] is False and d["candidate_speaking_state"] is False
    assert d["decision"] == "hold"
    assert d["reason"] == "no_speaker_hold"
    assert d["debt_candidate"] > 0.0, "debt existed and was still recorded, just not counted"


# --- a genuine (non-debt) advantage must still switch ------------------------

def test_genuine_engagement_advantage_still_switches():
    """B looks at the robot, A does not -- a real short_term_score advantage,
    not a debt artifact. The margin must not block this."""
    policy = make_policy()
    switches = run(
        policy, 6.0,
        speak_fn=lambda t: (False, False),
        # First 1s: both seeking (A becomes target by id tie-break).
        # After: only B is seeking, so B builds a genuine score advantage.
        seek_fn=lambda t: (True, True) if t < 1.0 else (False, True),
    )
    assert switches[-1][2] == B, f"a genuine engagement advantage must still switch: {switches}"
    assert switches[-1][3] == "fallback_score"
    assert len(switches) == 2, f"exactly one real switch expected, got {switches}"


def test_margin_is_configurable_and_enforced():
    """An impossibly large margin must block even a genuine advantage --
    proving the margin is really the gate and really comes from config."""
    policy = make_policy(fallback_switch_margin=99.0)
    switches = run(
        policy, 6.0,
        speak_fn=lambda t: (False, False),
        seek_fn=lambda t: (True, True) if t < 1.0 else (False, True),
    )
    assert len(switches) == 1, f"a 99.0 margin must block every fallback switch, got {switches}"
    assert policy.last_fallback_decision["margin_satisfied"] is False


def test_debt_requires_speech_false_restores_debt_in_comparison():
    """The flag must be a real switch, not decoration: turning it off puts
    debt back into the comparison (and the old oscillation reappears)."""
    policy = make_policy(debt_requires_speech=False, fallback_switch_margin=0.0)
    switches = run(policy, 10.0, lambda t: (False, False))
    assert len(switches) > 2, (
        "with debt_requires_speech=False and no margin, the original "
        f"debt-driven oscillation should reappear, got {switches}"
    )
    assert policy.last_fallback_decision["debt_counted"] is True


# --- the speaker path must NOT be gated by the fallback margin ---------------

def test_speaker_preempt_still_prompt_and_not_delayed_by_fallback_margin():
    """A confirmed new speaker must still preempt through the speaker path.
    The fallback margin must not touch that latency."""
    baseline = make_policy(fallback_switch_margin=0.0)
    huge = make_policy(fallback_switch_margin=99.0)

    def scenario(policy):
        return run(policy, 6.0, lambda t: (t < 3.0, t >= 3.0))

    sw_baseline = scenario(baseline)
    sw_huge = scenario(huge)

    to_b_baseline = [s for s in sw_baseline if s[2] == B]
    to_b_huge = [s for s in sw_huge if s[2] == B]

    assert to_b_baseline and to_b_huge, "the confirmed speaker must take the target in both runs"
    assert to_b_baseline[0][3] == "speaker_preempt"
    assert to_b_huge[0][3] == "speaker_preempt"
    assert to_b_baseline[0][0] == pytest.approx(to_b_huge[0][0], abs=1e-9), (
        "fallback_switch_margin must not change speaker-preemption latency"
    )


def test_debt_gate_uses_confirmed_speaker_not_raw_cue():
    """Defect A: the debt gate must key on CONFIRMED speaker status
    (active_speaker_id), never on the raw `speaking` cue -- otherwise banked
    debt is cashed in on the first speaking frame, before confirmation."""
    policy = make_policy()
    run(policy, 4.0, lambda t: (t > 1.0, False))

    # A is genuinely the confirmed active speaker here.
    assert policy.active_speaker_id == A
    assert policy._has_raw_speech_evidence(A) is True
    assert policy._has_confirmed_speaker(A) is True
    assert policy._debt_counts_for_switch(A, B) is True

    # B has neither raw nor confirmed evidence -> cannot enable debt alone.
    assert policy._has_confirmed_speaker(B) is False
    assert policy._debt_counts_for_switch(B, None) is False


def test_raw_speech_evidence_alone_does_not_enable_debt():
    """The precise Defect A guard: raw cue true, confirmed false -> debt off."""
    policy = make_policy()
    # One speaking frame only -- raw evidence exists, confirmation does not.
    policy.update(
        [obs(A, speaking=True, speech_confidence=1.0), obs(B, speaking=False)], DT
    )
    assert policy._has_raw_speech_evidence(A) is True, "raw cue is present"
    assert policy._has_confirmed_speaker(A) is False, "but not yet confirmed"
    assert policy._debt_counts_for_switch(A, B) is False, (
        "raw speaking evidence alone must NOT re-admit debt (Defect A)"
    )


# --- explicit decision paths -------------------------------------------------

def test_decision_paths_are_explicit_not_one_generic_comparison():
    """no_speaker_hold / target_retention / fallback_score_switch must be
    distinguishable, per the requirement that paths stay explicit."""
    # no_speaker_hold
    quiet = make_policy()
    run(quiet, 5.0, lambda t: (False, False))
    assert quiet.last_fallback_decision["reason"] == "no_speaker_hold"
    assert quiet.no_speaker_hold is True

    # fallback_score_switch -- captured ON the switching tick (afterwards the
    # record naturally reverts to a hold decision for the next comparison).
    switching = make_policy()
    t, prev, switch_record = 0.0, None, None
    for _ in range(int(6.0 / DT)):
        seeking = (True, True) if t < 1.0 else (False, True)
        target = switching.update(
            [obs(A, seeking=seeking[0], t=t), obs(B, seeking=seeking[1], t=t)], DT
        )
        if target != prev and prev is not None:
            switch_record = dict(switching.last_fallback_decision)
        prev = target
        t += DT
    assert switch_record is not None, "expected a fallback switch to occur"
    assert switch_record["reason"] == "fallback_score_switch"
    assert switch_record["decision"] == "switch"

    # target_retention: a speaker exists (so not no_speaker_hold), but the
    # fallback path still runs for the non-speaker comparison.
    held = make_policy()
    run(held, 4.0, lambda t: (t > 1.0, False))
    assert held.active_speaker_id is not None
    assert held.no_speaker_hold is False


def test_stare_penalty_rotation_is_preserved():
    """The bounded stare penalty is a SEPARATE intended mechanism: after
    max_continuous_gaze_seconds the incumbent's score is penalized enough
    that the target rotates. The debt fix must not disable it."""
    policy = make_policy(max_continuous_gaze_seconds=2.0, recent_attention_penalty_weight=1.0)
    switches = run(policy, 10.0, lambda t: (False, False))
    assert len(switches) >= 2, (
        f"the stare-penalty rotation must still fire, got {switches}"
    )
    assert switches[1][3] == "fallback_score"


def test_fallback_decision_record_has_every_required_field():
    policy = make_policy()
    run(policy, 5.0, lambda t: (False, False))
    for key in (
        "current_target_id", "candidate_id", "current_score", "candidate_score",
        "score_difference", "switch_margin", "current_speaking_state",
        "candidate_speaking_state", "debt_current", "debt_candidate",
        "decision", "reason",
    ):
        assert key in policy.last_fallback_decision, f"missing diagnostic field {key!r}"


# --- preserved behavior ------------------------------------------------------

def test_reported_score_still_includes_debt():
    """get_metrics/_score_of must be unchanged -- the fix affects only the
    switch comparison, not what is reported/logged."""
    policy = make_policy()
    run(policy, 5.0, lambda t: (False, False))
    other = B if policy.current_target == A else A
    reported = policy.get_metrics(other)["score"]
    without_debt = policy._switch_score_of(other, include_debt=False)
    assert reported > without_debt, "the REPORTED score must still include the debt term"
    assert reported == pytest.approx(
        without_debt + ConversationalSimplePolicy.DEBT_WEIGHT * policy.attention_debt[other]
    )


def test_lost_observation_still_cannot_take_target_via_fallback():
    """Pre-existing guard must survive the fix."""
    policy = make_policy()
    t = 0.0
    for _ in range(30):
        policy.update([obs(A, seeking=True, t=t), obs(B, seeking=False, t=t)], DT)
        t += DT
    assert policy.current_target == A
    for _ in range(300):
        target = policy.update(
            [obs(A, seeking=False, t=t), obs(B, seeking=True, t=t, lost=True)], DT
        )
        t += DT
        assert target != B, "a lost observation must never take the target"
