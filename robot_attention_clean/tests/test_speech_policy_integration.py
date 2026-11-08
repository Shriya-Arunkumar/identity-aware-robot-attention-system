"""Integration tests: visual_speaking_proxy fields (speaking, speech_confidence)
feeding into the real ConfidenceDebtPolicy/GreedyDwellPolicy (no mocking).
Covers the policy-side requirements: speech_mode="off" reproduces prior
behavior exactly; speaking alone never switches; speech + reliable gaze
raises a challenger's score; and a sustained challenger with speech support
still has to clear the existing hold-time gate, not bypass it (there is no
separate "VERIFY" state in this codebase).
"""

import pytest

from gaze_attention.policies import ConfidenceDebtPolicy, GreedyDwellPolicy

ALL_POLICY_FACTORIES = [GreedyDwellPolicy, ConfidenceDebtPolicy]
DT = 0.033


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


# --- 3. speech-off reproduces previous behavior ------------------------------

@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_speech_off_matches_no_speech_fields_at_all(factory):
    """A policy with speech_mode='off' fed observations that DO carry
    speaking=True/speech_confidence must produce byte-identical scores to
    the same policy fed observations with no speech fields whatsoever --
    proving speech_mode='off' truly reproduces gaze-only behavior."""
    policy_with_fields = factory(speech_mode="off")
    policy_without_fields = factory(speech_mode="off")

    t = 0.0
    for _ in range(60):
        obs_with = [
            obs(0, seeking=True, confidence=0.9, speaking=True, speech_confidence=1.0, t=t),
            obs(1, seeking=True, confidence=0.9, speaking=False, t=t),
        ]
        obs_without = [
            {"id": 0, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9, "confidence": 0.9, "seeking_attention": True, "timestamp": t},
            {"id": 1, "bbox": [0, 0, 10, 10], "gaze_signal": 0.9, "confidence": 0.9, "seeking_attention": True, "timestamp": t},
        ]
        target_with = policy_with_fields.update(obs_with, DT)
        target_without = policy_without_fields.update(obs_without, DT)
        assert target_with == target_without
        t += DT

    m0_with = policy_with_fields.get_metrics(0)
    m0_without = policy_without_fields.get_metrics(0)
    assert m0_with["score"] == m0_without["score"]
    assert m0_with["engagement"] == m0_without["engagement"]
    assert m0_with.get("speech_bonus", 0.0) == 0.0


@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_speech_off_default_is_off(factory):
    """speech_mode defaults to 'off' with no argument at all."""
    policy = factory()
    assert policy.speech_mode == "off"
    t = 0.0
    for _ in range(30):
        policy.update([obs(0, speaking=True, speech_confidence=1.0, t=t)], DT)
        t += DT
    assert policy.get_metrics(0).get("speech_bonus", 0.0) == 0.0


# --- 4. speech alone does not immediately switch ------------------------------

@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_speech_alone_without_looking_never_switches(factory):
    """Speaking without looking at the robot (seeking_attention=False) must
    contribute zero bonus and therefore never cause a switch, no matter how
    long it continues."""
    policy = factory(speech_mode="mouth_motion", speech_weight=10.0, minimum_hold_seconds=0.1, target_grace_seconds=0.1)
    t = 0.0
    for _ in range(30):  # A holds the target
        policy.update([obs(0, seeking=True, t=t), obs(1, seeking=False, t=t)], DT)
        t += DT
    assert policy.current_target == 0

    target = None
    for _ in range(200):  # B speaks loudly (large speech_weight) but never looks
        target = policy.update(
            [obs(0, seeking=True, t=t), obs(1, seeking=False, speaking=True, speech_confidence=1.0, t=t)],
            DT,
        )
        t += DT
    assert target == 0, "speaking without looking at the robot must never cause a switch"


@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_speech_with_looking_does_not_bypass_hold_time(factory):
    """Speaking AND looking still must not switch before minimum_hold_seconds
    has elapsed since the last switch -- speech is supporting evidence, not
    an override of the existing hold-time gate."""
    policy = factory(speech_mode="mouth_motion", speech_weight=10.0, minimum_hold_seconds=5.0, target_grace_seconds=0.1)
    t = 0.0
    for _ in range(5):  # A becomes target, held only briefly
        policy.update([obs(0, seeking=True, t=t), obs(1, seeking=False, t=t)], DT)
        t += DT
    assert policy.current_target == 0

    # B now speaks AND looks (satisfies the bonus condition) for ~2s --
    # well under the 5s hold.
    target = None
    for _ in range(60):
        target = policy.update(
            [obs(0, seeking=False, t=t), obs(1, seeking=True, speaking=True, speech_confidence=1.0, t=t)],
            DT,
        )
        t += DT
    assert target == 0, "a speech-boosted challenger must still respect minimum_hold_seconds"


# --- 5. speech plus reliable gaze increases challenger score -----------------

@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_speech_plus_gaze_increases_score_vs_speech_mode_off(factory):
    speech_on = factory(speech_mode="mouth_motion", speech_weight=0.5, minimum_hold_seconds=100.0)
    speech_off = factory(speech_mode="off", minimum_hold_seconds=100.0)
    t = 0.0

    # Establish person 0 as target FIRST, before person 1 ever appears --
    # otherwise, on the very first tick (no target selected yet), the
    # bonus itself would decide who wins the initial pick, which isn't
    # what this test is about (that's covered by the hold-gate tests).
    for _ in range(10):
        speech_on.update([obs(0, seeking=True, t=t)], DT)
        speech_off.update([obs(0, seeking=True, t=t)], DT)
        t += DT
    assert speech_on.current_target == 0 and speech_off.current_target == 0

    for _ in range(60):
        # Person 0 pinned as target (hold=100s); person 1 is the challenger,
        # speaking and looking at the robot in both runs.
        speech_on.update(
            [obs(0, seeking=True, t=t), obs(1, seeking=True, confidence=0.9, speaking=True, speech_confidence=1.0, t=t)],
            DT,
        )
        speech_off.update(
            [obs(0, seeking=True, t=t), obs(1, seeking=True, confidence=0.9, speaking=True, speech_confidence=1.0, t=t)],
            DT,
        )
        t += DT

    assert speech_on.current_target == 0 and speech_off.current_target == 0, (
        "sanity: person 0 must still hold the target in both runs (100s hold) so the "
        "engagement/debt dynamics for person 1 are directly comparable"
    )
    score_on = speech_on.get_metrics(1)["score"]
    score_off = speech_off.get_metrics(1)["score"]
    assert score_on > score_off, "speech_mode='mouth_motion' should raise a speaking+looking challenger's score"
    assert speech_on.get_metrics(1)["speech_bonus"] > 0.0


@pytest.mark.parametrize("factory", ALL_POLICY_FACTORIES)
def test_speech_bonus_requires_both_speaking_and_looking(factory):
    policy = factory(speech_mode="mouth_motion", speech_weight=0.5)
    t = 0.0
    policy.update(
        [obs(0, seeking=True, t=t), obs(1, seeking=False, speaking=True, speech_confidence=1.0, t=t)], DT
    )
    assert policy.get_metrics(1)["speech_bonus"] == 0.0, "speaking without looking must get no bonus"

    policy2 = factory(speech_mode="mouth_motion", speech_weight=0.5)
    policy2.update(
        [obs(0, seeking=True, t=t), obs(1, seeking=True, speaking=False, speech_confidence=1.0, t=t)], DT
    )
    assert policy2.get_metrics(1)["speech_bonus"] == 0.0, "looking without speaking must get no bonus"

    policy3 = factory(speech_mode="mouth_motion", speech_weight=0.5)
    policy3.update(
        [obs(0, seeking=True, t=t), obs(1, seeking=True, speaking=True, speech_confidence=1.0, t=t)], DT
    )
    assert policy3.get_metrics(1)["speech_bonus"] > 0.0, "speaking AND looking together must get a bonus"


# --- 6. speech during the hold-time gate supports a persistent challenger ---

def test_speech_supports_persistent_challenger_through_hold_gate():
    """A genuinely sustained, confident, speaking-and-looking challenger
    should win the target sooner with speech support than without --
    "supports" a persistent challenger -- while still only switching after
    minimum_hold_seconds has elapsed since the last switch (there is no
    separate VERIFY state; the existing hold-time gate is what's being
    respected here)."""
    common_kwargs = dict(minimum_hold_seconds=0.5, target_grace_seconds=0.1, debt_weight=0.5)

    baseline = ConfidenceDebtPolicy(speech_mode="off", **common_kwargs)
    with_speech = ConfidenceDebtPolicy(speech_mode="mouth_motion", speech_weight=0.5, **common_kwargs)

    def run(policy):
        t = 0.0
        switch_tick = None
        time_since_switch_before_flip = None
        for i in range(5):
            policy.update([obs(0, seeking=True, t=t), obs(1, seeking=False, t=t)], DT)
            t += DT
        assert policy.current_target == 0

        for i in range(1000):
            # Capture the gate's own bookkeeping BEFORE the call that flips
            # it, so we check exactly what update()'s hold-time condition
            # itself compared against minimum_hold_seconds -- robust
            # against any warm-up-phase tick-counting arithmetic.
            pre_time_since_switch = policy._time_since_switch
            target = policy.update(
                [
                    obs(0, seeking=False, t=t),  # A stops seeking, decays
                    obs(1, seeking=True, confidence=0.9, speaking=True, speech_confidence=1.0, t=t),
                ],
                DT,
            )
            t += DT
            if switch_tick is None and target == 1:
                switch_tick = i
                time_since_switch_before_flip = pre_time_since_switch
        return switch_tick, target, time_since_switch_before_flip

    baseline_switch_tick, baseline_final, baseline_pre = run(baseline)
    speech_switch_tick, speech_final, speech_pre = run(with_speech)

    assert baseline_final == 1, "baseline (no speech) should still eventually switch"
    assert speech_final == 1, "speech-supported run should switch"
    assert baseline_switch_tick is not None and speech_switch_tick is not None
    assert speech_switch_tick <= baseline_switch_tick, (
        f"expected speech to accelerate (or at least not delay) the switch: "
        f"baseline switched at tick {baseline_switch_tick}, with-speech at {speech_switch_tick}"
    )
    # Sanity: the hold-time gate's own bookkeeping had actually satisfied
    # minimum_hold_seconds at the moment it allowed the switch -- the gate
    # was never bypassed by speech.
    assert speech_pre >= common_kwargs["minimum_hold_seconds"], (
        f"switch happened with only {speech_pre:.3f}s since the last switch, "
        f"less than minimum_hold_seconds={common_kwargs['minimum_hold_seconds']}"
    )
