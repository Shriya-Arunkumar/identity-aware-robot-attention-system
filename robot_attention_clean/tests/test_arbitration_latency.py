"""Root Cause #3 -- arbitration latency.

Diagnosis: after Defect B fixed the detector, the remaining delay was almost
entirely `speaker_switch_confirm_seconds` (the pending-challenger window in
_resolve_target). It was a CLASS DEFAULT of 0.5s, absent from config.yaml and
DEFAULT_CONFIG, so the single largest contributor to switch latency could not
be tuned without editing source. Measured decomposition of the old path:

    speech_confirmation_latency  0.333s   (speaker_switch_seconds)
    arbitration_latency          0.533s   (speaker_switch_confirm_seconds)
    total                        0.867s

Fix: expose speaker_switch_confirm_seconds / speech_off_confirm_seconds /
speaker_absence_grace_seconds through DEFAULT_CONFIG + config.yaml +
validation + _make_conversational_simple, and lower the switch-confirm window
to 0.20s. The off/grace timers are deliberately NOT shortened.

Everything here drives the REAL runtime chain (mouth geometry ->
VisualSpeakingProxy -> policy), so the Defect B detector fix stays in force.
"""

from __future__ import annotations

import pytest

from gaze_attention import DEFAULT_CONFIG, load_config
from gaze_attention.main_webcam import _make_conversational_simple
from gaze_attention.speech import VisualSpeakingProxy

A, B = 12, 18
DT = 1.0 / 30.0


def _proxy(config):
    return VisualSpeakingProxy(
        speaking_on_threshold=config["speaking_on_threshold"],
        speaking_off_threshold=config["speaking_off_threshold"],
        speaking_window_seconds=config["speaking_window_seconds"],
        activity_scale=config["speech_activity_scale"],
        support_window_seconds=config["speech_support_window_seconds"],
        minimum_positive_fraction=config["speech_minimum_positive_fraction"],
        activity_delta_threshold=config["speech_activity_delta_threshold"],
    )


def _mouth_opening(talking, i):
    if not talking:
        return 0.05
    return 0.25 if i % 2 == 0 else 0.05


def _policy(config=None, isolate_stare=True, **overrides):
    """Built through the SAME factory main_webcam.py uses, so these tests
    exercise the real config -> policy wiring rather than a hand-built
    policy that could silently diverge from production."""
    cfg = dict(config or load_config())
    cfg.update(overrides)
    policy = _make_conversational_simple(cfg)
    if isolate_stare:
        # The stare-penalty rotation is a separate intended mechanism; push it
        # out of range so these tests measure speaker behavior only. Its own
        # behavior is covered by test_debt_oscillation.py's dedicated test.
        policy.max_continuous_gaze_seconds = 60.0
    return policy, cfg


def _run(policy, cfg, duration, talking_fn):
    """Drive the real chain. Returns (switches, breakdown_by_target)."""
    proxy = _proxy(cfg)
    t, i, prev = 0.0, 0, None
    switches, breakdowns = [], {}
    while t < duration:
        obs = []
        talking = talking_fn(t)
        for pid, is_talking in zip((A, B), talking):
            act, sp, sc = proxy.update(pid, _mouth_opening(is_talking, i), DT)
            obs.append({"id": pid, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9,
                        "confidence": 0.9, "seeking_attention": True, "timestamp": t,
                        "mouth_activity": act, "speaking": sp, "speech_confidence": sc,
                        "lost": False})
        target = policy.update(obs, DT)
        if target != prev:
            switches.append((round(t, 3), prev, target, policy.last_switch_reason))
            if target is not None:
                breakdowns[target] = dict(policy.last_switch_latency_breakdown)
            prev = target
        i += 1
        t += DT
    return switches, breakdowns


# --- 1. configuration exposure and data flow ---------------------------------

@pytest.mark.parametrize("key,expected", [
    ("speaker_switch_confirm_seconds", 0.20),
    ("speech_off_confirm_seconds", 0.7),
    ("speaker_absence_grace_seconds", 0.8),
])
def test_timers_exist_in_default_config_and_yaml(key, expected):
    assert DEFAULT_CONFIG[key] == expected
    assert load_config()[key] == expected, f"{key} must be present in config.yaml"


@pytest.mark.parametrize("key", [
    "speaker_switch_confirm_seconds", "speech_off_confirm_seconds",
    "speaker_absence_grace_seconds",
])
def test_config_value_reaches_the_policy_no_class_default_override(key):
    """The whole point of Root Cause #3: a non-default config value must
    actually reach the constructed policy."""
    probe = 0.4321
    policy, _ = _policy(**{key: probe})
    assert getattr(policy, key) == probe, (
        f"{key}={probe} from config did not reach the policy -- a class "
        f"default is overriding it"
    )


@pytest.mark.parametrize("key", [
    "speaker_switch_confirm_seconds", "speech_off_confirm_seconds",
    "speaker_absence_grace_seconds",
])
def test_timers_are_validated(key):
    from gaze_attention import _validate_config
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        cfg = dict(DEFAULT_CONFIG)
        cfg[key] = bad
        with pytest.raises(ValueError):
            _validate_config(cfg)


# --- 2. Test A: fast speaker switch ------------------------------------------

def test_a_fast_speaker_switch_measured():
    policy, cfg = _policy()
    switches, breakdowns = _run(policy, cfg, 8.0, lambda t: (t < 3.0, t >= 3.0))

    to_b = [s for s in switches if s[2] == B]
    assert to_b, f"the new speaker must take the target: {switches}"
    assert to_b[0][3] == "speaker_preempt", "must use the speaker path, not fallback"
    assert all(s[3] != "fallback_score" for s in switches if s[2] == B), (
        "no fallback switch may occur first"
    )

    bd = breakdowns[B]
    assert bd["speech_confirmation_latency"] is not None
    assert bd["arbitration_latency"] is not None
    # The arbitration stage must now be dominated by the (reduced) confirm
    # window, not the old 0.5s one.
    assert bd["arbitration_latency"] < 0.35, (
        f"arbitration latency {bd['arbitration_latency']:.3f}s not reduced"
    )
    assert bd["total_target_switch_latency"] < 0.7, (
        f"total latency {bd['total_target_switch_latency']:.3f}s"
    )


def test_lower_switch_confirm_reduces_arbitration_but_not_speech_confirmation():
    """Requirement 7: same timeline, two switch-confirm values. Lower value
    must reduce ARBITRATION latency and leave SPEECH-confirmation latency and
    false-positive behavior untouched."""
    timeline = lambda t: (t < 3.0, t >= 3.0)

    slow, cfg_slow = _policy(speaker_switch_confirm_seconds=0.5)
    _, bd_slow = _run(slow, cfg_slow, 8.0, timeline)

    fast, cfg_fast = _policy(speaker_switch_confirm_seconds=0.20)
    _, bd_fast = _run(fast, cfg_fast, 8.0, timeline)

    assert bd_fast[B]["arbitration_latency"] < bd_slow[B]["arbitration_latency"], (
        "a lower switch-confirm window must reduce arbitration latency"
    )
    assert bd_fast[B]["speech_confirmation_latency"] == pytest.approx(
        bd_slow[B]["speech_confirmation_latency"], abs=1e-9), (
        "speech-confirmation latency must be unaffected by an arbitration timer"
    )
    assert bd_fast[B]["total_target_switch_latency"] < bd_slow[B]["total_target_switch_latency"]

    # False-positive behavior must be identical at both settings.
    for value in (0.5, 0.20):
        p, c = _policy(speaker_switch_confirm_seconds=value)
        sw, _ = _run(p, c, 10.0, lambda t: (True, 5.0 <= t < 5.0 + DT * 2))
        assert all(s[2] != B for s in sw), (
            f"switch_confirm={value} changed false-positive behavior: {sw}"
        )


# --- 3. Tests B-F -------------------------------------------------------------

def test_b_isolated_noise_spike_still_rejected():
    """Defect B regression must stay green at the new, faster timing."""
    policy, cfg = _policy()
    switches, _ = _run(policy, cfg, 10.0, lambda t: (True, 5.0 <= t < 5.0 + DT * 2))
    assert all(s[2] != B for s in switches), f"spike stole the target: {switches}"
    assert policy.active_speaker_id == A


def test_c_continuous_speaker_zero_false_switches():
    policy, cfg = _policy()
    switches, _ = _run(policy, cfg, 12.0, lambda t: (True, False))
    assert all(s[2] != B for s in switches), f"false switch to silent B: {switches}"


def test_c_stare_penalty_rotation_still_intact():
    """Companion to the isolation above: with the stare penalty ENABLED the
    rotation must still happen, proving it was only isolated, not broken."""
    policy, cfg = _policy(isolate_stare=False)
    policy.max_continuous_gaze_seconds = 1.0
    policy.recent_attention_penalty_weight = 1.0
    switches, _ = _run(policy, cfg, 12.0, lambda t: (False, False))
    assert len(switches) >= 2, f"stare-penalty rotation must still fire: {switches}"


def test_d_brief_pause_does_not_switch():
    policy, cfg = _policy()
    switches, _ = _run(policy, cfg, 10.0,
                       lambda t: (not (4.0 <= t < 4.3), False))
    assert all(s[2] != B for s in switches), (
        f"a brief pause must not hand attention to a silent person: {switches}"
    )


def test_e_genuine_turn_taking_one_switch_per_turn():
    """A speaks, stops; B speaks, stops; A resumes -> one clean switch per
    real turn, no intermediate fallback target, no ping-pong."""
    def talking(t):
        if t < 3.0:
            return (True, False)
        if t < 6.0:
            return (False, True)
        return (True, False)

    policy, cfg = _policy()
    switches, _ = _run(policy, cfg, 10.0, talking)

    real = [s for s in switches if s[1] is not None]
    assert [s[2] for s in real] == [B, A], (
        f"expected exactly one switch per turn (->B, ->A): {switches}"
    )
    assert all(s[3] == "speaker_preempt" for s in real), (
        f"every turn switch must use the speaker path: {switches}"
    )


def test_f_nobody_speaks_stable():
    policy, cfg = _policy()
    switches, _ = _run(policy, cfg, 10.0, lambda t: (False, False))
    assert policy.active_speaker_id is None
    assert len(switches) == 1 and switches[0][3] == "initial", (
        f"no-speaker interval must stay stable: {switches}"
    )
    assert policy.no_speaker_hold is True


# --- 4. off/grace timers stay conservative and separate ----------------------

def test_off_and_grace_timers_not_shortened():
    """Requirement 4: these are distinct concepts from the switch-confirm
    window and must not have been tuned down alongside it."""
    config = load_config()
    assert config["speech_off_confirm_seconds"] == 0.7
    assert config["speaker_absence_grace_seconds"] == 0.8
    assert config["target_grace_seconds"] == 0.5
    assert config["speaker_switch_confirm_seconds"] < config["speech_off_confirm_seconds"], (
        "the switch-confirm window should be the fast one; off/grace stay conservative"
    )


def test_latency_breakdown_has_all_required_stages():
    policy, cfg = _policy()
    _, breakdowns = _run(policy, cfg, 8.0, lambda t: (t < 3.0, t >= 3.0))
    bd = breakdowns[B]
    for key in (
        "reason", "speech_onset_time", "confirmed_speaker_time",
        "speaker_preempt_time", "target_switch_time",
        "speech_confirmation_latency", "arbitration_latency",
        "total_target_switch_latency",
    ):
        assert key in bd, f"missing latency stage {key!r}"
    assert bd["reason"] == "speaker_preempt"
    assert bd["speaker_preempt_time"] is not None
    # Stages must be internally consistent (measured, not inferred).
    assert bd["total_target_switch_latency"] == pytest.approx(
        bd["speech_confirmation_latency"] + bd["arbitration_latency"], abs=1e-9)


# ===========================================================================
# Joint speaker-confirmation/switch timers.
#
# speaker_confirm_seconds and speaker_switch_seconds gate the SAME transition:
# a challenger must clear speaker_confirm_seconds to be a valid candidate
# (_compute_speaker_candidate) AND speaker_switch_seconds to preempt an
# existing active speaker (_update_active_speaker). The binding constraint is
# therefore max(confirm, switch) -- a measured sweep showed lowering only one
# leaves existing-speaker turn-taking latency completely unchanged.
# ===========================================================================

TURN = lambda t: (t < 3.0, t >= 3.0)          # noqa: E731 -- A active, B takes over
COLD = lambda t: (False, t >= 1.0)            # noqa: E731 -- no active speaker yet


@pytest.mark.parametrize("key,expected", [
    ("speaker_confirm_seconds", 0.25),
    ("speaker_switch_seconds", 0.25),
])
def test_joint_timers_configured_consistently(key, expected):
    assert DEFAULT_CONFIG[key] == expected
    assert load_config()[key] == expected, f"{key} must match in config.yaml"


@pytest.mark.parametrize("key", ["speaker_confirm_seconds", "speaker_switch_seconds"])
def test_joint_timers_reach_the_runtime_policy(key):
    """Neither may be shadowed by a class default in the live factory path."""
    probe = 0.4321
    policy, _ = _policy(**{key: probe})
    assert getattr(policy, key) == probe, (
        f"{key}={probe} from config did not reach the policy"
    )


def test_existing_speaker_takeover_is_governed_by_the_joint_constraint():
    """The core finding: turn-taking latency tracks max(confirm, switch).
    Lowering either timer ALONE must leave it unchanged; lowering BOTH must
    reduce it. Measured from timestamps, not frame counts."""
    def total(confirm, switch_s):
        p, c = _policy(speaker_confirm_seconds=confirm, speaker_switch_seconds=switch_s)
        _, bd = _run(p, c, 8.0, TURN)
        return bd[B]["total_target_switch_latency"]

    both_high = total(0.35, 0.35)
    only_switch_low = total(0.35, 0.20)
    only_confirm_low = total(0.20, 0.35)
    both_low = total(0.25, 0.25)

    assert only_switch_low == pytest.approx(both_high, abs=1e-9), (
        "lowering speaker_switch_seconds alone must not change turn-taking latency"
    )
    assert only_confirm_low == pytest.approx(both_high, abs=1e-9), (
        "lowering speaker_confirm_seconds alone must not change turn-taking latency"
    )
    assert both_low < both_high, (
        f"lowering BOTH must reduce turn-taking latency: {both_low} vs {both_high}"
    )


def test_joint_timers_improve_turn_taking_from_the_previous_default():
    """Acceptance criterion 1: measurably better than the 0.567s baseline."""
    policy, cfg = _policy()
    switches, bd = _run(policy, cfg, 8.0, TURN)
    total = bd[B]["total_target_switch_latency"]
    assert total < 0.567, f"turn-taking latency {total:.3f}s did not improve"
    assert bd[B]["reason"] == "speaker_preempt"
    assert all(s[3] != "fallback_score" for s in switches if s[2] == B)


def test_cold_start_behavior_remains_correct():
    """A speaker appearing with NO incumbent active speaker still confirms
    and takes the target through the speaker path."""
    policy, cfg = _policy()
    switches, bd = _run(policy, cfg, 6.0, COLD)
    assert policy.active_speaker_id == B
    assert bd[B]["reason"] == "speaker_preempt"
    assert bd[B]["total_target_switch_latency"] < 0.533, (
        "cold-start latency should also improve"
    )


def test_lowering_confirm_timers_does_not_weaken_false_positive_rejection():
    """Acceptance criteria 3/4: the Defect B temporal-support gate -- not
    these timers -- carries false-positive rejection, so it must be
    unaffected across the whole tested range."""
    spike = lambda t: (True, 5.0 <= t < 5.0 + DT * 2)          # noqa: E731
    noisy = lambda t: (True, any(abs(t - m) < DT * 1.5         # noqa: E731
                                 for m in (2.0, 4.0, 6.0, 8.0)))
    for confirm, switch_s in ((0.35, 0.35), (0.25, 0.25)):
        for name, timeline in (("spike", spike), ("noisy", noisy)):
            p, c = _policy(speaker_confirm_seconds=confirm, speaker_switch_seconds=switch_s)
            sw, _ = _run(p, c, 10.0, timeline)
            assert all(s[2] != B for s in sw), (
                f"{name} stole the target at {confirm}/{switch_s}: {sw}"
            )
            assert p.active_speaker_id != B, (
                f"{name} confirmed a speaker at {confirm}/{switch_s}"
            )


def test_promotion_protection_stays_positive_at_the_new_timers():
    """promotion_protection_seconds = speaker_switch_confirm + speaker_confirm
    is the anti-ping-pong guard. It must remain comfortably positive."""
    policy, _ = _policy()
    protection = policy.speaker_switch_confirm_seconds + policy.speaker_confirm_seconds
    assert protection >= 0.4, f"anti-ping-pong protection shrank to {protection:.3f}s"


# ===========================================================================
# Pre-Root-Cause-4 cleanup: constructor defaults aligned with the accepted
# live configuration, so a directly-constructed policy can no longer silently
# run obsolete timing.
# ===========================================================================

# Every speaker/arbitration timer whose class default is REQUIRED to match the
# accepted configuration value.
ALIGNED_TIMERS = {
    "speaker_confirm_seconds": 0.25,
    "speaker_switch_seconds": 0.25,
    "speaker_switch_confirm_seconds": 0.20,
    "speech_off_confirm_seconds": 0.7,
    "speaker_absence_grace_seconds": 0.8,
}

# Documented, deliberate exception: minimum_hold_seconds' class default (1.5)
# still differs from the configured live value (0.75). It is NOT a speaker
# timer and was never part of the approved latency work, and
# test_conversational_simple_policy.test_current_target_held_for_1_5_seconds_default
# intentionally exercises the 1.5s default. Pinned here so the divergence
# stays visible and cannot be forgotten.
KNOWN_DEFAULT_DIVERGENCES = {"minimum_hold_seconds": (1.5, 0.75)}


@pytest.mark.parametrize("key,expected", sorted(ALIGNED_TIMERS.items()))
def test_live_factory_value_is_the_accepted_value(key, expected):
    """The values actually driving the robot."""
    policy, _ = _policy()
    assert getattr(policy, key) == expected, (
        f"{key} reached the live policy as {getattr(policy, key)}, expected {expected}"
    )


@pytest.mark.parametrize("key,expected", sorted(ALIGNED_TIMERS.items()))
def test_constructor_default_matches_config_so_direct_construction_is_safe(key, expected):
    """A policy built with NO arguments must not silently use obsolete
    timing -- that was the stale-class-default cleanup."""
    import inspect
    from gaze_attention.policies import ConversationalSimplePolicy

    sig = inspect.signature(ConversationalSimplePolicy.__init__)
    default = sig.parameters[key].default
    assert default == expected, (
        f"constructor default for {key} is {default}, but the accepted "
        f"configuration value is {expected} -- direct construction would run "
        f"obsolete timing"
    )
    assert getattr(ConversationalSimplePolicy(), key) == expected


@pytest.mark.parametrize("key,expected", sorted(ALIGNED_TIMERS.items()))
def test_default_config_and_yaml_agree(key, expected):
    assert DEFAULT_CONFIG[key] == expected
    assert load_config()[key] == expected


def test_known_default_divergences_are_still_exactly_as_documented():
    """Guards the ONE remaining deliberate divergence. If it is ever aligned
    (or a new one appears), this fails and forces the docs to be updated."""
    import inspect
    from gaze_attention.policies import ConversationalSimplePolicy

    sig = inspect.signature(ConversationalSimplePolicy.__init__)
    config = load_config()
    for key, (class_default, configured) in KNOWN_DEFAULT_DIVERGENCES.items():
        assert sig.parameters[key].default == class_default
        assert config[key] == configured


def test_no_unrelated_timer_changed_during_the_alignment():
    """The alignment touched constructor defaults only -- these live values
    must be exactly what was previously accepted."""
    config = load_config()
    for key, expected in {
        "minimum_hold_seconds": 0.75,
        "target_grace_seconds": 0.5,
        "speaker_release_seconds": 0.40,
        "minimum_speaker_attention_seconds": 2.0,
        "speaker_confidence_threshold": 0.25,
        "max_continuous_gaze_seconds": 8.0,
        "fallback_switch_margin": 0.10,
        "t_lost_seconds": 1.5,
    }.items():
        assert config[key] == expected, f"{key} changed unexpectedly"


def test_aligning_defaults_did_not_change_live_behavior():
    """A policy built from config and one built with no arguments must now
    agree on every aligned timer -- i.e. the alignment is a no-op for the
    live path, which already passed config explicitly."""
    from gaze_attention.policies import ConversationalSimplePolicy

    live, _ = _policy()
    bare = ConversationalSimplePolicy()
    for key in ALIGNED_TIMERS:
        assert getattr(live, key) == getattr(bare, key), (
            f"{key}: live={getattr(live, key)} bare={getattr(bare, key)}"
        )


def test_live_camera_validation_procedure_documents_the_sign_fix():
    """The mutual-gaze sign convention was inverted and is now corrected; the
    procedure doc must record that evidence AND must still state plainly that
    the interactive Checks 1-5 have not been run."""
    from pathlib import Path

    doc = Path(__file__).resolve().parent.parent / "docs" / "LIVE_CAMERA_VALIDATION.md"
    assert doc.is_file(), "docs/LIVE_CAMERA_VALIDATION.md is missing"
    text = doc.read_text()

    # The recorded evidence for the fix must be present and specific.
    for marker in ("session_20260820_140240", "corr(yaw, phi)",
                   "_HEAD_POSE_LANDMARK_INDICES", "e = yaw + phi"):
        assert marker in text, f"validation doc missing {marker!r}"

    # The interactive checks must NOT be claimed as done.
    assert "NOT YET RECORDED" in text, (
        "the interactive Checks 1-5 have not been run and the doc must say so"
    )

    # The outstanding magnitude/calibration caveat must remain visible.
    assert "uncalibrated" in text.lower()
