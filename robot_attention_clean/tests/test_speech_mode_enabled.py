"""Root Cause #2 -- connecting the mouth-motion speaker path to target
selection.

Diagnosis: config had `speech_mode: "off"`, which (after the speech_mode
gate fix) made ConversationalSimplePolicy treat everyone as permanently
silent -- active_speaker_id stayed None, `speaker_preempt` never fired, and
an actual speaker change had to wait for fallback scoring. perception.py
computes the mouth cue unconditionally either way; only the policy's use of
it was disabled.

Fix: `speech_mode: "mouth_motion"` in config/config.yaml AND DEFAULT_CONFIG
(kept in sync so a missing/partial config file cannot silently re-disable
the speaker path).

These tests deliberately drive the REAL VisualSpeakingProxy (speech.py) with
synthetic mouth-opening geometry rather than injecting `speaking=True`
booleans -- that is the only way to prove the whole runtime chain
(mouth cue -> proxy -> confirmed speaking state -> policy -> preemption)
is actually connected, which is what Root Cause #2 is about.

No thresholds or timer values are changed here (that is Root Cause #3/#4).
"""

from __future__ import annotations

import pytest

from gaze_attention import DEFAULT_CONFIG, load_config
from gaze_attention.policies import ConversationalSimplePolicy
from gaze_attention.speech import VisualSpeakingProxy

A, B = 12, 18
DT = 1.0 / 30.0


# --- 1. configuration level ---------------------------------------------------

def test_live_config_enables_mouth_motion():
    config = load_config()
    assert config["speech_mode"] == "mouth_motion", (
        "the live config must enable the mouth-motion speaker path"
    )


def test_default_config_matches_live_config():
    """A missing/partial config.yaml must not silently disable the speaker
    path -- that is exactly the failure Root Cause #2 was."""
    assert DEFAULT_CONFIG["speech_mode"] == "mouth_motion"


def test_rest_of_arbitration_config_unchanged():
    """Pins the target/arbitration values that later latency work must NOT
    disturb. speaker_confirm_seconds/speaker_switch_seconds were since
    deliberately lowered together 0.35 -> 0.25 (they gate the same
    transition; see tests/test_arbitration_latency.py's joint-timer
    section), so they are pinned to their new values here rather than
    dropped -- everything else must still be untouched."""
    config = load_config()
    assert config["minimum_hold_seconds"] == 0.75
    assert config["target_grace_seconds"] == 0.5
    assert config["speaker_release_seconds"] == 0.40
    assert config["minimum_speaker_attention_seconds"] == 2.0
    # Joint timers -- intentionally changed, pinned to the agreed values.
    assert config["speaker_confirm_seconds"] == 0.25
    assert config["speaker_switch_seconds"] == 0.25
    assert config["speaker_confidence_threshold"] == 0.25
    assert config["speaking_on_threshold"] == 0.15
    assert config["speaking_off_threshold"] == 0.08
    assert config["speaking_window_seconds"] == 0.5
    # Root Cause #1 fix must remain active and unchanged.
    assert config["fallback_switch_margin"] == 0.10
    assert config["debt_requires_speech"] is True


# --- 2. the real runtime chain ------------------------------------------------

def _proxy():
    """Built from DEFAULT_CONFIG so these tests track the shipped detector
    settings, including the Defect B temporal-support gate."""
    return VisualSpeakingProxy(
        speaking_on_threshold=DEFAULT_CONFIG["speaking_on_threshold"],
        speaking_off_threshold=DEFAULT_CONFIG["speaking_off_threshold"],
        speaking_window_seconds=DEFAULT_CONFIG["speaking_window_seconds"],
        activity_scale=DEFAULT_CONFIG["speech_activity_scale"],
        support_window_seconds=DEFAULT_CONFIG["speech_support_window_seconds"],
        minimum_positive_fraction=DEFAULT_CONFIG["speech_minimum_positive_fraction"],
        activity_delta_threshold=DEFAULT_CONFIG["speech_activity_delta_threshold"],
    )


def _mouth_opening(talking: bool, i: int) -> float:
    """Synthetic mouth-opening ratio. Talking = alternating open/closed
    (temporal variation, which is what speech.py measures via windowed std);
    silent = a constant, unchanging value."""
    if not talking:
        return 0.05
    return 0.25 if i % 2 == 0 else 0.05


def make_policy(speech_mode, **overrides):
    kwargs = dict(
        speech_mode=speech_mode,
        confidence_threshold=DEFAULT_CONFIG["confidence_threshold"],
        minimum_hold_seconds=DEFAULT_CONFIG["minimum_hold_seconds"],
        target_grace_seconds=DEFAULT_CONFIG["target_grace_seconds"],
        speaker_confirm_seconds=DEFAULT_CONFIG["speaker_confirm_seconds"],
        speaker_release_seconds=DEFAULT_CONFIG["speaker_release_seconds"],
        speaker_switch_seconds=DEFAULT_CONFIG["speaker_switch_seconds"],
        minimum_speaker_attention_seconds=DEFAULT_CONFIG["minimum_speaker_attention_seconds"],
        speaker_confidence_threshold=DEFAULT_CONFIG["speaker_confidence_threshold"],
        fallback_switch_margin=DEFAULT_CONFIG["fallback_switch_margin"],
        debt_requires_speech=DEFAULT_CONFIG["debt_requires_speech"],
        max_continuous_gaze_seconds=60.0,  # isolate from the stare rotation
    )
    kwargs.update(overrides)
    return ConversationalSimplePolicy(**kwargs)


def run_through_proxy(policy, duration, talking_fn, seeking=(True, True)):
    """Drive the REAL mouth-motion proxy, then the policy -- the same
    ordering main_webcam.py uses (perception -> speech proxy -> policy).
    Returns (switches, trace)."""
    proxy = _proxy()
    t, i, switches, prev, trace = 0.0, 0, [], None, []
    for _ in range(int(duration / DT)):
        ta, tb = talking_fn(t)
        obs = []
        for pid, talking, seek in ((A, ta, seeking[0]), (B, tb, seeking[1])):
            activity, speaking, sconf = proxy.update(pid, _mouth_opening(talking, i), DT)
            obs.append({
                "id": pid, "bbox": [0, 0, 10, 10],
                "gaze_signal": 0.9 if seek else 0.1,
                "confidence": 0.9, "seeking_attention": seek,
                "timestamp": t, "mouth_activity": activity,
                "speaking": speaking, "speech_confidence": sconf, "lost": False,
            })
        target = policy.update(obs, DT)
        trace.append((round(t, 3), target, policy.active_speaker_id,
                      obs[0]["mouth_activity"], obs[0]["speaking"],
                      obs[1]["mouth_activity"], obs[1]["speaking"]))
        if target != prev:
            switches.append((round(t, 3), prev, target, policy.last_switch_reason))
            prev = target
        i += 1
        t += DT
    return switches, trace


def test_mouth_motion_cue_produces_confirmed_speaking_state():
    """Chain step 1-3: a moving mouth must actually raise mouth_activity and
    flip the confirmed `speaking` boolean through the real proxy."""
    proxy = _proxy()
    speaking_seen = False
    for i in range(30):
        activity, speaking, sconf = proxy.update(A, _mouth_opening(True, i), DT)
        speaking_seen = speaking_seen or speaking
    assert speaking_seen, "a moving mouth must produce speaking=True via the real proxy"

    quiet = _proxy()
    for i in range(30):
        _, speaking, _ = quiet.update(B, _mouth_opening(False, i), DT)
        assert speaking is False, "a static mouth must never produce speaking=True"


def test_speech_is_connected_to_target_selection_end_to_end():
    """Chain step 4-6: with mouth_motion, a real mouth cue must drive
    active_speaker_id and the talking person must end up selected."""
    policy = make_policy("mouth_motion")
    switches, _ = run_through_proxy(
        policy, 6.0,
        # A silent throughout; B starts talking at t=1.0.
        talking_fn=lambda t: (False, t >= 1.0),
    )
    assert policy.active_speaker_id == B, "the talking person must become the active speaker"
    to_b = [s for s in switches if s[2] == B]
    assert to_b, f"the confirmed speaker must take the target, got {switches}"


def test_speech_driven_takeover_uses_the_speaker_path():
    """Defect A regression (was xfail): a speech-driven takeover must come
    through the SPEAKER path after confirmation, never through the fallback
    path on the first raw speaking frame via banked debt."""
    policy = make_policy("mouth_motion")
    switches, _ = run_through_proxy(
        policy, 6.0, talking_fn=lambda t: (False, t >= 1.0),
    )
    to_b = [s for s in switches if s[2] == B]
    assert to_b, f"the confirmed speaker must take the target, got {switches}"
    assert to_b[0][3] in ("speaker_confirmed", "speaker_preempt"), (
        f"the switch must come from the SPEAKER path, not fallback: {to_b[0]}"
    )
    assert all(s[3] != "fallback_score" for s in switches if s[2] == B), (
        f"B must never take the target via fallback_score: {switches}"
    )


def test_speech_mode_off_leaves_the_same_cue_disconnected():
    """Control: byte-identical mouth cues, speech_mode='off' -> no active
    speaker, no speaker-path switch. Proves the config value is the gate."""
    policy = make_policy("off")
    switches, _ = run_through_proxy(
        policy, 6.0, talking_fn=lambda t: (False, t >= 1.0),
    )
    assert policy.active_speaker_id is None
    assert all(s[3] != "speaker_preempt" for s in switches), (
        f"speech_mode='off' must never produce a speaker_preempt: {switches}"
    )


# --- 3. Root Cause #1 must remain intact -------------------------------------

def test_no_speaker_state_still_stable_with_speech_enabled():
    """Scenario F, now with speech_mode enabled and the real proxy: nobody
    talks, so the debt oscillator must still stay suppressed."""
    policy = make_policy("mouth_motion")
    switches, _ = run_through_proxy(policy, 10.0, talking_fn=lambda t: (False, False))
    assert policy.active_speaker_id is None, "nobody is talking -- no active speaker"
    assert len(switches) == 1 and switches[0][3] == "initial", (
        f"the no-speaker state must remain stable, got {switches}"
    )
    assert policy.no_speaker_hold is True


def test_continuous_speaker_produces_no_false_switch_to_silent_person():
    """Scenario D through the real proxy: A talks continuously, B silent."""
    policy = make_policy("mouth_motion")
    switches, _ = run_through_proxy(policy, 10.0, talking_fn=lambda t: (True, False))
    assert all(s[2] != B for s in switches), (
        f"a continuously silent person must never take the target: {switches}"
    )


def test_isolated_mouth_blip_never_turns_speaking_on():
    """Defect B regression: mouth_activity is a std over a 0.5s window, so a
    2-frame (0.067s) blip USED to keep `speaking` True for ~0.467s -- longer
    than speaker_confirm_seconds (0.35s) -- fully confirming a speaker. The
    temporal-support gate now rejects it at onset: the blip contributes only
    its rise and fall, ~2/9 = 0.22 support, below the 0.30 requirement."""
    proxy = _proxy()
    speaking_frames = []
    for i in range(40):
        _, speaking, _ = proxy.update(B, _mouth_opening(5 <= i < 7, i), DT)
        if speaking:
            speaking_frames.append(i * DT)
    assert not speaking_frames, (
        f"an isolated 2-frame blip must never turn speaking on, got {speaking_frames}"
    )
    assert proxy.support_fraction(B) < DEFAULT_CONFIG["speech_minimum_positive_fraction"]


def test_mouth_activity_still_elevates_on_a_blip_but_support_does_not():
    """The two signals must stay SEPARATE: the rolling statistic still
    spikes (unchanged), but temporal support correctly does not."""
    proxy = _proxy()
    peak_activity, peak_support = 0.0, 0.0
    for i in range(40):
        activity, _, _ = proxy.update(B, _mouth_opening(5 <= i < 7, i), DT)
        peak_activity = max(peak_activity, activity)
        peak_support = max(peak_support, proxy.support_fraction(B))
    assert peak_activity >= DEFAULT_CONFIG["speaking_on_threshold"], (
        "raw mouth_activity should still register the spike -- unchanged"
    )
    assert peak_support < DEFAULT_CONFIG["speech_minimum_positive_fraction"], (
        "but temporal support must not, which is what rejects the onset"
    )


def test_speech_confidence_is_data_sufficiency_not_speech_strength():
    """Documented distinction: speech_confidence saturates at 1.0 for a
    SILENT person too, so it can never be evidence of speaking."""
    proxy = _proxy()
    for i in range(40):  # a completely static mouth
        _, speaking, sconf = proxy.update(B, _mouth_opening(False, i), DT)
        assert speaking is False
    assert sconf == pytest.approx(1.0), (
        "speech_confidence reaches 1.0 for a silent person -- it measures data "
        "availability, never speech strength"
    )
    assert proxy.support_fraction(B) == 0.0, "temporal support is the real evidence"


def test_isolated_noise_spike_does_not_steal_the_target():
    """Scenario C through the real proxy: A talks continuously, B produces a
    single isolated mouth blip."""
    policy = make_policy("mouth_motion")
    switches, _ = run_through_proxy(
        policy, 10.0,
        talking_fn=lambda t: (True, 5.0 <= t < 5.0 + DT * 2),
    )
    assert all(s[2] != B for s in switches), (
        f"an isolated mouth-noise spike must not take the target: {switches}"
    )


# --- Defect A: the four speaker-confirmation states --------------------------

def test_state2_raw_cue_before_confirmation_cannot_take_over_via_debt():
    """State 2: candidate has a raw speaking cue but the confirmation timer
    has not completed and active_speaker_id is still None. Banked debt must
    not let them take the target during that window."""
    policy = make_policy("mouth_motion")
    proxy = _proxy()
    t, i = 0.0, 0
    # Phase 1: both silent and seeking for 2s -- A holds the target by id
    # tie-break while B banks a large attention debt.
    while t < 2.0:
        obs = []
        for pid in (A, B):
            act, sp, sc = proxy.update(pid, _mouth_opening(False, i), DT)
            obs.append({"id": pid, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9,
                        "confidence": 0.9, "seeking_attention": True, "timestamp": t,
                        "mouth_activity": act, "speaking": sp, "speech_confidence": sc,
                        "lost": False})
        policy.update(obs, DT)
        i += 1
        t += DT
    assert policy.current_target == A
    banked_debt = policy.attention_debt[B]
    assert banked_debt > 0.5, f"B must have banked real debt, got {banked_debt}"

    # Phase 2: B starts talking. Step tick-by-tick and assert that for every
    # tick where B is NOT yet the confirmed active speaker, the target stays A.
    checked_unconfirmed = 0
    while t < 2.0 + policy.speaker_confirm_seconds:
        obs = []
        for pid, talking in ((A, False), (B, True)):
            act, sp, sc = proxy.update(pid, _mouth_opening(talking, i), DT)
            obs.append({"id": pid, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9,
                        "confidence": 0.9, "seeking_attention": True, "timestamp": t,
                        "mouth_activity": act, "speaking": sp, "speech_confidence": sc,
                        "lost": False})
        target = policy.update(obs, DT)
        if policy.active_speaker_id != B:
            checked_unconfirmed += 1
            assert target == A, (
                f"t={t:.3f}: B took the target while unconfirmed "
                f"(active_speaker_id={policy.active_speaker_id}, debt={policy.attention_debt.get(B)})"
            )
        i += 1
        t += DT
    assert checked_unconfirmed > 0, "expected at least one raw-but-unconfirmed tick"


def test_state3_confirmed_speaker_preempts_promptly():
    """State 3: once the candidate reaches confirmed speaker state, the
    speaker path takes over and logs a speaker reason."""
    policy = make_policy("mouth_motion")
    switches, _ = run_through_proxy(policy, 6.0, talking_fn=lambda t: (False, t >= 1.0))
    assert policy.active_speaker_id == B
    to_b = [s for s in switches if s[2] == B]
    assert to_b and to_b[0][3] in ("speaker_confirmed", "speaker_preempt")


def test_state4_confirmed_target_not_displaced_by_others_banked_debt():
    """State 4: ONCE the current target is the confirmed speaker, another
    person's banked debt must never displace them. Scoped to the
    post-confirmation window -- the pre-confirmation window is governed by
    Defect C (see test_unconfirmed_speaker_should_not_be_overtaken_...)."""
    policy = make_policy("mouth_motion")
    proxy = _proxy()
    t, i, confirmed_since, violations = 0.0, 0, None, []
    while t < 10.0:
        obs = []
        for pid, talking in ((A, t >= 0.5), (B, False)):
            act, sp, sc = proxy.update(pid, _mouth_opening(talking, i), DT)
            obs.append({"id": pid, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9,
                        "confidence": 0.9, "seeking_attention": True, "timestamp": t,
                        "mouth_activity": act, "speaking": sp, "speech_confidence": sc,
                        "lost": False})
        target = policy.update(obs, DT)
        # Only assert once A is BOTH the confirmed speaker and the target.
        if policy.active_speaker_id == A and target == A:
            confirmed_since = confirmed_since or t
        if confirmed_since is not None and target != A:
            violations.append((round(t, 3), target))
        i += 1
        t += DT

    assert confirmed_since is not None, "A must become the confirmed speaker and target"
    assert policy.attention_debt.get(B, 0.0) > 0.0, "B did bank debt"
    assert not violations, (
        f"a confirmed speaker must not be displaced by banked debt: {violations}"
    )


def test_banked_debt_with_unconfirmed_cue_produces_no_fallback_switch():
    """Requirement 3: large debt + raw cue + NOT confirmed -> no fallback
    switch, and the diagnostic record must show debt was not counted.

    Driven with injected `speaking` booleans rather than the real proxy: a
    real mouth blip cannot produce a raw-but-unconfirmed cue because of
    Defect B (a 2-frame blip holds speaking=True for ~0.467s > the 0.35s
    confirm window). Injecting the boolean directly isolates the Defect A
    guard, which is the unit under test here."""
    policy = make_policy("mouth_motion")

    def step(t, a_speaking, b_speaking):
        return policy.update([
            {"id": A, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9, "confidence": 0.9,
             "seeking_attention": True, "timestamp": t, "speaking": a_speaking,
             "speech_confidence": 1.0 if a_speaking else 0.0, "lost": False},
            {"id": B, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9, "confidence": 0.9,
             "seeking_attention": True, "timestamp": t, "speaking": b_speaking,
             "speech_confidence": 1.0 if b_speaking else 0.0, "lost": False},
        ], DT)

    # Phase 1: both silent for 2s -- A holds target, B banks debt.
    t = 0.0
    while t < 2.0:
        step(t, False, False)
        t += DT
    assert policy.current_target == A
    assert policy.attention_debt[B] > 0.5, "B banked real debt"

    # Phase 2: B emits a raw speaking cue for only 3 frames (0.1s) -- well
    # under speaker_confirm_seconds (0.35s), so B never confirms.
    for _ in range(3):
        target = step(t, False, True)
        t += DT
        assert policy.active_speaker_id != B, "B must not confirm from a 0.1s cue"
        assert target == A, (
            f"an unconfirmed candidate must not take the target via banked debt "
            f"(debt={policy.attention_debt.get(B)}, record={policy.last_fallback_decision})"
        )
    assert policy._has_raw_speech_evidence(B) is True, "B did have a raw cue"
    assert policy._has_confirmed_speaker(B) is False
    assert policy.last_fallback_decision["debt_counted"] is False


def test_unconfirmed_speaker_is_not_overtaken_by_a_silent_seeker():
    """Defect C regression (was xfail): while A works through the speaker
    confirmation window, silent B must not steal the target and be preempted
    back a moment later. Real runtime chain."""
    policy = make_policy("mouth_motion")
    switches, _ = run_through_proxy(
        policy, 10.0, talking_fn=lambda t: (t >= 0.5, False),
    )
    assert all(s[2] != B for s in switches), (
        f"a silent person must never take the target from a speaking one: {switches}"
    )
    assert policy.active_speaker_id == A


# --- Defect C: the five required cases ---------------------------------------

def _inject(policy, t, a_speaking, b_speaking, a_seek=True, b_seek=True,
            a_lost=False, b_lost=False, extra=None):
    people = [
        {"id": A, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9 if a_seek else 0.1,
         "confidence": 0.9, "seeking_attention": a_seek, "timestamp": t,
         "speaking": a_speaking, "speech_confidence": 1.0 if a_speaking else 0.0,
         "lost": a_lost},
        {"id": B, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9 if b_seek else 0.1,
         "confidence": 0.9, "seeking_attention": b_seek, "timestamp": t,
         "speaking": b_speaking, "speech_confidence": 1.0 if b_speaking else 0.0,
         "lost": b_lost},
    ]
    if extra:
        people.extend(extra)
    return policy.update(people, DT)


def test_defect_c_1_raw_speaking_candidate_is_protected_during_confirmation():
    """Test 1: B has raw speaking evidence but is not confirmed; A (the
    current target) is silent with a growing fallback score. B must not
    receive the confirmed-speaker bonus and active_speaker_id must stay None
    for the whole confirmation window."""
    policy = make_policy("mouth_motion")
    t = 0.0
    # A settles as target while both are silent.
    while t < 1.0:
        _inject(policy, t, False, False)
        t += DT
    assert policy.current_target == A

    # B starts raw-speaking. Step through the confirmation window only.
    checked = 0
    end = t + policy.speaker_confirm_seconds - DT
    while t < end:
        _inject(policy, t, False, True)
        if policy.active_speaker_id is None:
            checked += 1
            assert policy.speaker_confirmation_pending is True
            assert policy.raw_speaker_candidate_id == B
            assert policy._score_of(B) < ConversationalSimplePolicy.ACTIVE_SPEAKER_BONUS, (
                "an unconfirmed raw speaker must NOT receive the confirmed-speaker bonus"
            )
        t += DT
    assert checked > 0
    assert policy.active_speaker_id is None, "must remain unconfirmed for the whole window"


def test_defect_c_2_confirmation_still_preempts():
    """Test 2: once B confirms, the speaker path takes over as before."""
    policy = make_policy("mouth_motion")
    t, switches, prev = 0.0, [], None
    while t < 6.0:
        target = _inject(policy, t, False, t >= 1.0)
        if target != prev:
            switches.append((round(t, 3), prev, target, policy.last_switch_reason))
            prev = target
        t += DT
    assert policy.active_speaker_id == B
    to_b = [s for s in switches if s[2] == B]
    assert to_b, f"the confirmed speaker must take the target: {switches}"
    assert to_b[0][3] in ("speaker_confirmed", "speaker_preempt")


def test_defect_c_3_no_speaker_fallback_unchanged():
    """Test 3: with no raw speaking evidence anywhere, Root Cause #1
    behavior must be byte-for-byte unchanged."""
    policy = make_policy("mouth_motion")
    t, switches, prev = 0.0, [], None
    while t < 10.0:
        target = _inject(policy, t, False, False)
        if target != prev:
            switches.append((round(t, 3), prev, target, policy.last_switch_reason))
            prev = target
        t += DT
    assert len(switches) == 1 and switches[0][3] == "initial", (
        f"no-speaker fallback must remain stable: {switches}"
    )
    assert policy.speaker_confirmation_pending is False
    assert policy.last_fallback_decision["fallback_switch_blocked"] is False
    assert policy.last_fallback_decision["debt_counted"] is False
    assert policy.last_fallback_decision["reason"] == "no_speaker_hold"


def test_defect_c_4_two_raw_candidates_no_silent_third_can_displace():
    """Test 4: A and B both raw-speaking, neither confirmed; a silent third
    person C must not displace the current target, and nobody may be marked
    confirmed prematurely."""
    C = 25
    policy = make_policy("mouth_motion")

    def step(t, speaking):
        return policy.update([
            {"id": A, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9, "confidence": 0.9,
             "seeking_attention": True, "timestamp": t, "speaking": speaking,
             "speech_confidence": 1.0 if speaking else 0.0, "lost": False},
            {"id": B, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9, "confidence": 0.9,
             "seeking_attention": True, "timestamp": t, "speaking": speaking,
             "speech_confidence": 1.0 if speaking else 0.0, "lost": False},
            {"id": C, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9, "confidence": 0.9,
             "seeking_attention": True, "timestamp": t, "speaking": False,
             "speech_confidence": 0.0, "lost": False},
        ], DT)

    t = 0.0
    while t < 1.0:
        step(t, False)
        t += DT
    held = policy.current_target

    # Both A and B raw-speak; stay inside the confirmation window.
    end = t + policy.speaker_confirm_seconds - DT
    while t < end:
        target = step(t, True)
        assert policy.active_speaker_id is None, "nobody may confirm prematurely"
        assert target != C, "a silent third person must not displace the target"
        assert target == held
        t += DT
    assert policy.speaker_confirmation_pending is True
    assert policy.raw_speaker_candidate_id in (A, B)


def test_defect_c_5_lost_raw_speaker_cannot_create_protection():
    """Test 5: a lost=True observation that appears to be raw-speaking must
    not create the protection, become a candidate, or challenge."""
    policy = make_policy("mouth_motion")
    t = 0.0
    # A settles as the target; B is present and silent.
    while t < 1.0:
        _inject(policy, t, False, False)
        t += DT
    assert policy.current_target == A

    # B now "speaks" but is LOST -- must be invisible to the speaker path.
    while t < 3.0:
        target = _inject(policy, t, False, True, b_lost=True)
        assert policy.raw_speaker_candidate_id != B, (
            "a lost observation must never create confirmation protection"
        )
        assert policy.active_speaker_id != B, "a lost observation must never confirm"
        assert target != B, "a lost observation must never take the target"
        t += DT


def test_defect_c_diagnostics_distinguish_confirmation_block():
    """The record must make it obvious a switch was blocked by the
    confirmation window rather than by debt or the margin."""
    policy = make_policy("mouth_motion", minimum_hold_seconds=0.0, fallback_switch_margin=0.0)
    t = 0.0
    # A becomes the target while both are silent and seeking.
    while t < 1.0:
        _inject(policy, t, False, False)
        t += DT
    assert policy.current_target == A

    # This is the exact Defect C shape: the INCUMBENT (A) raw-speaks, so its
    # short_term_score freezes, while the SILENT challenger (B) keeps
    # growing and would overtake. hold (0.0) and margin (0.0) are both
    # satisfied, so only the Defect C guard can be blocking.
    blocked_seen = False
    end = t + policy.speaker_confirm_seconds - DT
    while t < end:
        _inject(policy, t, True, False)
        d = policy.last_fallback_decision
        # Only the ticks where the switch WOULD otherwise have fired isolate
        # the guard; earlier ticks are held by the margin itself.
        if d.get("fallback_switch_blocked") and d["hold_satisfied"] and d["margin_satisfied"]:
            blocked_seen = True
            assert d["fallback_block_reason"] == "speaker_confirmation_pending"
            assert d["reason"] == "speaker_confirmation_pending"
            assert d["debt_counted"] is False, "not a debt block"
            assert d["speaker_confirmation_pending"] is True
            assert d["raw_speaker_candidate_id"] == A
            assert d["raw_speaker_candidate_elapsed"] > 0.0
            assert d["raw_speaker_candidate_since"] is not None
            assert d["active_speaker_id"] is None
        t += DT
    assert blocked_seen, "expected at least one confirmation-blocked tick"


def test_defect_c_all_required_diagnostic_fields_present():
    policy = make_policy("mouth_motion")
    t = 0.0
    while t < 2.0:
        _inject(policy, t, False, t >= 1.0)
        t += DT
    for key in (
        "raw_speaker_candidate_id", "raw_speaker_candidate_since",
        "raw_speaker_candidate_elapsed", "active_speaker_id",
        "speaker_confirmation_pending", "fallback_switch_blocked",
        "fallback_block_reason", "current_score", "candidate_score",
    ):
        assert key in policy.last_fallback_decision, f"missing diagnostic field {key!r}"


def test_target_retention_and_grace_unchanged():
    """Requirement 5: ordinary target retention / grace behavior is
    untouched by the Defect A guard."""
    policy = make_policy("mouth_motion")
    t = 0.0
    # A becomes the settled target.
    for _ in range(30):
        policy.update(
            [{"id": A, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9, "confidence": 0.9,
              "seeking_attention": True, "timestamp": t, "speaking": False,
              "speech_confidence": 0.0, "lost": False}], DT)
        t += DT
    assert policy.current_target == A

    # A disappears entirely; within target_grace_seconds the target is held.
    held_ticks = int((policy.target_grace_seconds * 0.5) / DT)
    for _ in range(held_ticks):
        target = policy.update([], DT)
        t += DT
        assert target == A, "target must be retained through its grace window"


def test_speaker_and_fallback_paths_remain_separate():
    """The speaker path and the debt-gated fallback path must stay
    distinguishable after enabling speech."""
    policy = make_policy("mouth_motion")
    run_through_proxy(policy, 5.0, talking_fn=lambda t: (t >= 1.0, False))
    assert policy.active_speaker_id == A
    assert policy.no_speaker_hold is False, (
        "with a confirmed speaker, the tick must not be resolved as no_speaker_hold"
    )


# ===========================================================================
# Defect B -- the six required behavior cases, all through the REAL runtime
# chain (mouth geometry -> VisualSpeakingProxy -> policy).
# ===========================================================================

def _run_measured(policy, duration, talking_fn):
    """Drive the real chain and record, per person, the onset of sustained
    talking, when they became the confirmed speaker, and when the target
    actually moved to them."""
    proxy = _proxy()
    t, i, prev = 0.0, 0, None
    switches = []
    confirmed_at = {}
    onset_at = {}
    was_talking = {A: False, B: False}
    while t < duration:
        obs = []
        talking = dict(zip((A, B), talking_fn(t)))
        for pid in (A, B):
            if talking[pid] and not was_talking[pid]:
                onset_at.setdefault(pid, t)
            if not talking[pid]:
                was_talking[pid] = False
            else:
                was_talking[pid] = True
            act, sp, sc = proxy.update(pid, _mouth_opening(talking[pid], i), DT)
            obs.append({"id": pid, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9,
                        "confidence": 0.9, "seeking_attention": True, "timestamp": t,
                        "mouth_activity": act, "speaking": sp, "speech_confidence": sc,
                        "lost": False})
        target = policy.update(obs, DT)
        if policy.active_speaker_id is not None:
            confirmed_at.setdefault(policy.active_speaker_id, t)
        if target != prev:
            switches.append((round(t, 3), prev, target, policy.last_switch_reason))
            prev = target
        i += 1
        t += DT
    return switches, onset_at, confirmed_at


def test_defect_b_1_isolated_spike_never_confirms_or_steals():
    """Test 1: 2-frame spike on B while A speaks continuously."""
    policy = make_policy("mouth_motion")
    switches, _, confirmed = _run_measured(
        policy, 10.0, lambda t: (True, 5.0 <= t < 5.0 + DT * 2))
    assert B not in confirmed, "an isolated spike must never confirm a speaker"
    assert all(s[2] != B for s in switches), f"spike stole the target: {switches}"
    assert all(not (s[2] == B and s[3] == "speaker_preempt") for s in switches)


def test_defect_b_2_real_speaker_onset_confirms_quickly():
    """Test 2: sustained speech must still confirm fast (latency measured)."""
    policy = make_policy("mouth_motion")
    switches, onset, confirmed = _run_measured(
        policy, 6.0, lambda t: (False, t >= 1.0))
    assert B in confirmed, "sustained speech must confirm a speaker"
    confirm_latency = confirmed[B] - onset[B]
    assert confirm_latency < 1.0, f"confirmation took {confirm_latency:.3f}s"
    to_b = [s for s in switches if s[2] == B]
    assert to_b and to_b[0][3] in ("speaker_confirmed", "speaker_preempt")


def test_defect_b_3_continuous_speaker_zero_false_switches():
    """Test 3: A speaks continuously, B silent -> zero false switches."""
    policy = make_policy("mouth_motion")
    switches, _, _ = _run_measured(policy, 10.0, lambda t: (True, False))
    assert all(s[2] != B for s in switches), f"false switch to silent B: {switches}"


def test_defect_b_4_turn_taking_one_clean_switch():
    """Test 4: A speaks, stops; B begins sustained speech -> one clean
    switch to B, no intermediate target."""
    policy = make_policy("mouth_motion")
    switches, _, _ = _run_measured(
        policy, 8.0, lambda t: (t < 3.0, t >= 3.0))
    after = [s for s in switches if s[0] >= 3.0]
    assert len(after) == 1, f"expected exactly one switch after t=3.0: {switches}"
    assert after[0][2] == B
    assert after[0][3] in ("speaker_confirmed", "speaker_preempt")


def test_defect_b_5_no_speaker_interval_stable():
    """Test 5: nobody speaks -> no active speaker, no oscillation."""
    policy = make_policy("mouth_motion")
    switches, _, confirmed = _run_measured(policy, 10.0, lambda t: (False, False))
    assert policy.active_speaker_id is None
    assert not confirmed, "nobody may be confirmed when nobody speaks"
    assert len(switches) == 1 and switches[0][3] == "initial", (
        f"no-speaker interval must stay stable: {switches}"
    )


def test_defect_b_6_brief_pause_does_not_hand_off_to_a_silent_person():
    """Test 6: A speaks, pauses briefly, resumes -> attention stays on A."""
    policy = make_policy("mouth_motion")

    def talking(t):
        # A speaks, pauses 0.3s at t=4.0, resumes. B always silent.
        return (not (4.0 <= t < 4.3), False)

    switches, _, _ = _run_measured(policy, 8.0, talking)
    assert all(s[2] != B for s in switches), (
        f"a brief pause must not hand attention to a silent person: {switches}"
    )


def test_defect_b_latency_breakdown_reported():
    """Explicit separation of speech-confirmation latency from target-switch
    latency, from measured frame timestamps."""
    policy = make_policy("mouth_motion")
    switches, onset, confirmed = _run_measured(
        policy, 8.0, lambda t: (t < 3.0, t >= 3.0))
    to_b = [s for s in switches if s[2] == B][0]
    confirm_latency = confirmed[B] - onset[B]
    switch_latency = to_b[0] - onset[B]
    # Confirmation must be the fast part; the remaining delay is the
    # arbitration confirm window (Root Cause #3, not tuned in this patch).
    assert confirm_latency < switch_latency
    assert confirm_latency < 0.7, f"speech-confirmation latency {confirm_latency:.3f}s"
