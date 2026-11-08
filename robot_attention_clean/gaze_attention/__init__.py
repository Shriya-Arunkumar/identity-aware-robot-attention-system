"""
gaze_attention — confidence-aware multi-person attention arbitration.

Pipeline: perception.process_frame -> filters.<Filter>.update ->
policies.<Policy>.update -> visualization/logger/ROS publish.

`load_config` is the single source of truth for defaults. Every module reads
config values passed in explicitly (never re-reads the yaml itself), so the
whole pipeline works with these DEFAULT_CONFIG values even if
config/config.yaml is missing or partial.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

DEFAULT_CONFIG: dict[str, Any] = {
    "camera_index": 0,
    "max_faces": 3,
    # DEPRECATED (see gaze_attention/tracker.py) -- kept present, unused by
    # matching decisions; superseded by iou_gate/t_lost_seconds below.
    "iou_threshold": 0.3,
    "max_missed_frames": 15,
    # tracker.py's three-state (TENTATIVE/CONFIRMED/LOST) lifecycle params.
    "n_confirm": 3,
    "t_lost_seconds": 1.5,
    "iou_gate": 0.2,
    "velocity_alpha": 0.5,
    "seeking_threshold": 0.6,
    "confidence_threshold": 0.45,
    "engagement_decay": 0.10,
    "debt_weight": 0.50,
    "engagement_max": 30.0,
    "debt_max": 30.0,
    "minimum_hold_seconds": 0.75,
    "target_grace_seconds": 0.5,
    "dt_fallback_seconds": 0.033,
    "dt_max_seconds": 0.2,
    # visual_speaking_proxy (see gaze_attention/speech.py) -- mouth-motion
    # auxiliary cue, NOT audio speech recognition (no microphone). Hysteresis
    # thresholds/window are speech.py-level (always computed); speech_mode/
    # speech_weight are policy-level (whether/how much policies act on it).
    "speaking_on_threshold": 0.15,
    "speaking_off_threshold": 0.08,
    "speaking_window_seconds": 0.5,
    # Defect B (temporal support, see gaze_attention/speech.py). mouth_activity
    # is a STD over speaking_window_seconds, so one 2-frame (0.067s) spike kept
    # it elevated for the full 0.5s window -> 0.467s of speaking=True, enough
    # to clear speaker_confirm_seconds (0.35s) and let an isolated blip preempt
    # a genuine continuous speaker. Turning speaking ON now also requires that
    # a real fraction of INDIVIDUAL recent frames showed movement.
    #   speech_support_window_seconds: window the fraction is measured over.
    #   speech_minimum_positive_fraction: required fraction of active frames.
    #     An isolated spike scores ~2/9 = 0.22; sustained speech ~1.0.
    #   speech_activity_delta_threshold: minimum frame-to-frame change in the
    #     mouth-opening RATIO (scale-normalized, see
    #     perception._mouth_opening_from_landmarks) for a frame to count.
    # activity_scale was previously hardcoded in VisualSpeakingProxy and
    # unreachable from config; exposed here so the detector is fully tunable.
    "speech_support_window_seconds": 0.30,
    "speech_minimum_positive_fraction": 0.30,
    "speech_activity_delta_threshold": 0.01,
    "speech_activity_scale": 0.02,
    # Root Cause #2: "mouth_motion", not "off". With "off",
    # ConversationalSimplePolicy's active-speaker path is entirely inert
    # (active_speaker_id stays None, speaker_preempt never fires). Kept in
    # sync with config/config.yaml deliberately -- a missing/partial config
    # file must not silently disable speaker detection.
    "speech_mode": "mouth_motion",  # off | mouth_motion
    "speech_weight": 0.3,
    # Mutual-gaze residual cue (Change A, see gaze_attention/perception.py).
    #   hfov_deg: camera horizontal field of view, used both to build the
    #     approximate (uncalibrated) camera matrix for cv2.solvePnP and to
    #     map a face's horizontal image position to a bearing angle phi.
    #   mutual_gaze_sigma_deg: width (std, degrees) of the Gaussian mapping
    #     the mutual-gaze residual e = yaw + phi to a [0,1] cue q.
    #   mouth_cue_weight: weight of the existing mouth-motion activity when
    #     folded into the interaction cue. Default 0.0 -> the interaction cue
    #     is the mutual-gaze residual alone; the mouth-motion feature itself
    #     is unchanged and still drives speaking/speech_confidence separately.
    "hfov_deg": 60.0,
    "mutual_gaze_sigma_deg": 12.0,
    "mouth_cue_weight": 0.0,
    # Real confidence variance (Change B, see gaze_attention/perception.py).
    #   C = clip(c_size * c_pose * c_stable, confidence_min, confidence_max)
    #   c_size   = clip(face_diagonal / (face_size_reference_ratio*H), 0, 1)
    #   c_pose   = clip(cos(yaw), 0, 1) ** 2   (yaw from Change A head pose)
    #   c_stable = 1 / (1 + stability_k * landmark_jitter)
    # pose_unavailable_confidence: conservative c_pose used when head pose
    #   could not be recovered (NOT 1.0 -- an unavailable pose must never
    #   look like a perfectly-frontal one; the observation also stays marked
    #   fallback=True, so it is distinguishable).
    # stability_init_confidence: conservative c_stable used until a track has
    #   accumulated a full stability_window_frames of history (a brand-new
    #   track must never be treated as perfectly stable).
    # Change C: ablation flag. When False, all confidence consumers
    # substitute C=1.0 — perception and logging still compute/record the
    # true C and its components, only policy/manager consumption is bypassed.
    "use_confidence": True,
    "confidence_min": 0.05,
    "confidence_max": 1.0,
    "face_size_reference_ratio": 0.18,
    "stability_window_frames": 5,
    "stability_k": 50.0,
    "pose_unavailable_confidence": 0.5,
    "stability_init_confidence": 0.5,
    # AttentionManager (see gaze_attention/attention_manager.py) -- a
    # higher-level conversational/socially-bounded gaze layer wrapped
    # AROUND whichever base policy is selected above. "off" is a
    # transparent passthrough: gaze-only behavior is byte-for-byte
    # unchanged. Requires speech_mode="mouth_motion" to have any real
    # effect (it reads the same speaking/seeking_attention fields).
    # NOTE: superseded as the recommended conversational approach by
    # ConversationalSimplePolicy ("conversational_simple", policies.py)
    # below -- kept available, still off by default.
    # speaker_confirm_seconds/speaker_release_seconds/max_continuous_gaze_seconds
    # are shared by both features (same concepts, same names, both now fast
    # defaults per the live-latency fix below); speaker_min_attention_seconds/
    # stare_penalty_weight below are AttentionManager's OWN keys
    # (ConversationalSimplePolicy has its own separate
    # minimum_speaker_attention_seconds/recent_attention_penalty_weight).
    "speaker_priority_mode": "off",  # off | on
    "speaker_confirm_seconds": 0.25,  # see speaker_switch_seconds -- joint constraint
    "speaker_min_attention_seconds": 5.0,  # AttentionManager only
    "speaker_release_seconds": 0.40,
    "quiet_gaze_confirm_seconds": 5.5,
    "quiet_acknowledgment_seconds": 1.2,
    "max_continuous_gaze_seconds": 8.0,
    "gaze_break_seconds": 2.5,
    "stare_penalty_weight": 0.2,  # AttentionManager only -- see recent_attention_penalty_weight for ConversationalSimplePolicy's own
    # ConversationalSimplePolicy (policies.py) -- the ACTIVE
    # target-selection policy (active_policy). Chooses a confirmed active
    # speaker directly (no comparison against anyone's accumulated lifetime
    # engagement) and falls back to bounded short-term gaze selection when
    # no one is speaking, including a bounded quietness penalty (never
    # penalizes a quiet person maintaining reliable eye contact, never
    # overrides an active speaker, never unbounded). See policies.py's
    # docstring for the full rationale and README.md for why this replaced
    # folding speech in as a small bonus to ConfidenceDebtPolicy's score,
    # and for the "fast speaker-first path" latency fix (speaker_switch_seconds/
    # minimum_speaker_attention_seconds, allowing a different already-valid
    # speaker to preempt without waiting for the old one's release timer).
    # These are INITIAL calibration values, not claimed to be universally
    # correct.
    # Joint speaker-confirmation/switch timers. These two gate the SAME
    # transition and the binding constraint is max(confirm, switch): a
    # challenger must clear speaker_confirm_seconds to be a valid candidate
    # (_compute_speaker_candidate) AND speaker_switch_seconds to preempt an
    # existing active speaker (_update_active_speaker). A measured sweep
    # confirmed that lowering only one changes existing-speaker turn-taking
    # latency by exactly zero -- 0.567s at 0.35/0.35, 0.35/0.20 and 0.20/0.35
    # alike. Lowered together to 0.25: still 7-8 frames at 30fps, and
    # false-positive rejection is carried by speech.py's temporal-support
    # gate (Defect B), not by these timers -- verified unchanged across the
    # sweep. Keeps more margin than 0.20.
    "speaker_switch_seconds": 0.25,
    "minimum_speaker_attention_seconds": 2.0,
    "speaker_confidence_threshold": 0.25,
    # Root Cause #3 (arbitration latency). These three were CLASS DEFAULTS in
    # ConversationalSimplePolicy, absent from config entirely -- so the single
    # largest contributor to switch latency could not be tuned without editing
    # source. Now exposed, validated, and wired through
    # _make_conversational_simple/replay.py.
    #   speaker_switch_confirm_seconds: how long a newly confirmed speaker
    #     must remain the SAME pending challenger before preemption fires.
    #     Was 0.5, which was the entire measured 0.533s arbitration delay.
    #     0.20 keeps a real "is this speaker stable?" check (6 frames at
    #     30fps) while cutting 0.3s; it is NOT zero, so a one-frame
    #     misdetection still cannot switch the target. It also feeds
    #     promotion_protection_seconds (= this + speaker_confirm_seconds),
    #     the anti-ping-pong guard, which stays comfortably positive at 0.55s.
    #   speech_off_confirm_seconds: how long speech evidence must be ABSENT
    #     before speaking_true_duration resets. Deliberately left at 0.7 --
    #     shortening it would make brief pauses cost a speaker their status.
    #   speaker_absence_grace_seconds: how long the active speaker may be
    #     unobserved (occlusion, fast head turn) before release. Left at 0.8.
    "speaker_switch_confirm_seconds": 0.20,
    "speech_off_confirm_seconds": 0.7,
    "speaker_absence_grace_seconds": 0.8,
    # Debt-oscillation fix (Root Cause #1). ConversationalSimplePolicy's
    # fallback (no-active-speaker) path compared candidate vs incumbent with
    # a bare `>`, while attention_debt is force-reset to 0 for whoever
    # currently holds the target and grows for everyone else. With two
    # equally-engaged people that makes the challenger's score STRICTLY
    # greater by construction every tick, so a switch fired the instant
    # minimum_hold_seconds expired -- a guaranteed ~0.8s ping-pong between
    # two silent people, with no speech, mouth noise, or confidence
    # difference involved.
    #   fallback_switch_margin: a challenger must beat the incumbent by MORE
    #     than this to take the target on the fallback path. 0.10 is ~0.11s
    #     of exclusive seeking at confidence 0.9 (short_term_score grows at
    #     confidence*dt, capped at SCORE_MAX=5.0) -- large enough to kill
    #     numerical ties and debt-noise flapping, small enough that a
    #     genuine engagement difference still switches almost immediately.
    #     Applies ONLY to the fallback path; the speaker-preemption path is
    #     deliberately NOT gated by it (a confirmed speaker must not be
    #     delayed by a fallback-only margin).
    #   debt_requires_speech: when True, attention_debt is excluded from the
    #     fallback SWITCH COMPARISON while neither the incumbent nor the
    #     challenger has confirmed speaking evidence. Debt is still accrued,
    #     still logged, and still counted once speech evidence exists -- it
    #     simply may not by itself manufacture a target switch in the
    #     no-speaker state.
    "fallback_switch_margin": 0.10,
    "debt_requires_speech": True,
    "non_speaking_score_decay": 0.5,
    "recent_attention_penalty_weight": 0.2,  # ConversationalSimplePolicy's own stare-penalty ramp rate
    "quiet_decay_rate": 1.0,
    "quiet_penalty_max": 3.0,
    "inference_every_n_frames": 1,
    # GreedyDwellPolicy (gaze_attention/baseline_policies.py) is
    # baseline-only -- never the default/active policy. active_policy must
    # stay "conversational_simple" unless a user explicitly overrides it
    # (--active-policy / config) for an intentional baseline comparison.
    "active_policy": "conversational_simple",
    "policy": "confidence_debt",  # base policy used by AttentionManager when speaker_priority_mode=on, and available as a baseline via active_policy=confidence_debt/greedy
    "filter": "moving_average",  # Kalman removed from this prototype's scope; see README.md "Filter scope"
    "ros_enabled": True,
    # RosPublisher (main_webcam.py) throttling for /attention_target_id +
    # /attention_target_angle: publish only when the selected target id
    # changes, or at most this often otherwise (a low fixed rate) -- not
    # every frame. 0.05s (20 Hz) rather than the old 0.2s (5 Hz): smoothing
    # now lives entirely in eye_gaze_node.py's slew-rate limiter downstream
    # (see target_angle.py's module docstring for why this process no
    # longer smooths), so this stage's job is just to not spam the wire --
    # a faster publish rate here means eye_gaze_node.py sees fresher raw
    # angles more often, which matters now that it's the only thing shaping
    # the motion.
    "ros_min_publish_interval_seconds": 0.05,
    # target_angle.py: /attention_target_angle tracks the SELECTED person's
    # CURRENT face position (face_center_x -> normalized_x -> target_angle),
    # not a fixed per-id lookup -- /attention_target_id (WHO) and
    # /attention_target_angle (WHERE their face currently is) are
    # deliberately independent; see main_webcam.py's RosPublisher
    # docstring. target_angle_max is both the max used to derive the angle
    # from normalized_x AND the clamp bound; also used for the fixed
    # id->angle fallback mapping (target_angle.FALLBACK_ANGLES) when face
    # coordinates aren't usable this tick. NOTE: this process publishes the
    # RAW angle -- no smoothing config here. Smoothing is exactly one stage,
    # in eye_gaze_node.py (ROS params max_angular_speed/control_rate_hz,
    # not part of this config file). Do not add a target_angle_smoothing_*
    # key back without removing the smoothing from wherever else it lives.
    "target_angle_max": 0.65,
    "display_enabled": True,
    "log_enabled": True,
    "debug": False,
}


def raw_dt_from_timestamps(
    current_timestamp: Optional[float], previous_timestamp: Optional[float]
) -> Optional[float]:
    """`current_timestamp - previous_timestamp`, or None if either is
    missing (e.g. the very first frame, before any previous timestamp has
    been recorded) — "compute dt from timestamps where possible"."""
    if current_timestamp is None or previous_timestamp is None:
        return None
    return current_timestamp - previous_timestamp


def sanitize_dt(
    raw_dt: Optional[float],
    fallback: float = DEFAULT_CONFIG["dt_fallback_seconds"],
    max_dt: float = DEFAULT_CONFIG["dt_max_seconds"],
    debug: bool = False,
) -> tuple[float, Optional[float]]:
    """Turn a possibly-missing/negative/zero/non-finite `raw_dt` into a safe
    value every dt-scaled accumulator (engagement, debt, decay, filter
    process noise) can integrate with. Returns `(used_dt, raw_dt)` — the
    caller uses `used_dt`; `raw_dt` is kept only for debug logging.

    - Missing (`None`), non-finite (NaN/Inf), or `<= 0` -> replaced by
      `fallback`. A dropped frame, a stalled clock, or a caller with no
      previous timestamp yet must not be treated as "zero time elapsed"
      (which would silently freeze every accumulator) or as a negative dt
      (which would corrupt them).
    - Always clamped to `[0, max_dt]` afterward, regardless of path. A
      paused window, a dropped frame, or delayed inference can produce one
      abnormally large gap between timestamps; without this clamp, every
      dt-scaled term jumps by a corresponding multiple in a single tick.
      See DEBUG_ATTENTION.md FM4.
    """
    used_dt = raw_dt
    if used_dt is None or not math.isfinite(used_dt) or used_dt <= 0.0:
        used_dt = fallback
    used_dt = max(0.0, min(max_dt, used_dt))
    if debug:
        print(f"[dt] raw_dt={raw_dt} used_dt={used_dt}")
    return used_dt, raw_dt


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load config/config.yaml, falling back to DEFAULT_CONFIG for any
    missing key (or the whole file). Never raises on a missing/partial file —
    that's the "must work with safe defaults" requirement."""
    config = dict(DEFAULT_CONFIG)
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

    if not os.path.isfile(path):
        return config

    import yaml

    with open(path) as f:
        loaded = yaml.safe_load(f) or {}

    config.update(loaded)
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    """Validate Change-A mutual-gaze config values. Raises ValueError on an
    out-of-range/non-finite value rather than silently substituting a
    different default in another code path."""
    hfov = config.get("hfov_deg", DEFAULT_CONFIG["hfov_deg"])
    if not (math.isfinite(hfov) and 0.0 < hfov < 180.0):
        raise ValueError(f"hfov_deg must be finite in (0, 180), got {hfov!r}")

    sigma = config.get("mutual_gaze_sigma_deg", DEFAULT_CONFIG["mutual_gaze_sigma_deg"])
    if not (math.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"mutual_gaze_sigma_deg must be finite and > 0, got {sigma!r}")

    mouth_weight = config.get("mouth_cue_weight", DEFAULT_CONFIG["mouth_cue_weight"])
    if not math.isfinite(mouth_weight):
        raise ValueError(f"mouth_cue_weight must be finite, got {mouth_weight!r}")

    # -- Change B (confidence variance) --------------------------------------
    cmin = config.get("confidence_min", DEFAULT_CONFIG["confidence_min"])
    cmax = config.get("confidence_max", DEFAULT_CONFIG["confidence_max"])
    if not (math.isfinite(cmin) and 0.0 <= cmin <= 1.0):
        raise ValueError(f"confidence_min must be finite in [0, 1], got {cmin!r}")
    if not (math.isfinite(cmax) and 0.0 <= cmax <= 1.0):
        raise ValueError(f"confidence_max must be finite in [0, 1], got {cmax!r}")
    if cmin > cmax:
        raise ValueError(f"confidence_min ({cmin}) must be <= confidence_max ({cmax})")

    ref_ratio = config.get("face_size_reference_ratio", DEFAULT_CONFIG["face_size_reference_ratio"])
    if not (math.isfinite(ref_ratio) and ref_ratio > 0.0):
        raise ValueError(f"face_size_reference_ratio must be finite and > 0, got {ref_ratio!r}")

    window = config.get("stability_window_frames", DEFAULT_CONFIG["stability_window_frames"])
    if not (isinstance(window, int) and window >= 1):
        raise ValueError(f"stability_window_frames must be an int >= 1, got {window!r}")

    k = config.get("stability_k", DEFAULT_CONFIG["stability_k"])
    if not (math.isfinite(k) and k >= 0.0):
        raise ValueError(f"stability_k must be finite and >= 0, got {k!r}")

    pose_fallback = config.get("pose_unavailable_confidence", DEFAULT_CONFIG["pose_unavailable_confidence"])
    if not (math.isfinite(pose_fallback) and 0.0 <= pose_fallback <= 1.0):
        raise ValueError(f"pose_unavailable_confidence must be finite in [0, 1], got {pose_fallback!r}")

    stab_init = config.get("stability_init_confidence", DEFAULT_CONFIG["stability_init_confidence"])
    if not (math.isfinite(stab_init) and 0.0 <= stab_init <= 1.0):
        raise ValueError(f"stability_init_confidence must be finite in [0, 1], got {stab_init!r}")

    # -- Root Cause #1 (debt oscillation) ------------------------------------
    margin = config.get("fallback_switch_margin", DEFAULT_CONFIG["fallback_switch_margin"])
    if not (math.isfinite(margin) and margin >= 0.0):
        raise ValueError(f"fallback_switch_margin must be finite and >= 0, got {margin!r}")

    drs = config.get("debt_requires_speech", DEFAULT_CONFIG["debt_requires_speech"])
    if not isinstance(drs, bool):
        raise ValueError(f"debt_requires_speech must be a bool, got {drs!r}")

    # -- Root Cause #3 (arbitration timers) ----------------------------------
    # Strictly > 0: a zero switch-confirm window would let a single frame of
    # misdetected speech move the target, which is exactly what Defect B's
    # detector fix exists to prevent.
    for key in ("speaker_switch_confirm_seconds", "speech_off_confirm_seconds",
                "speaker_absence_grace_seconds"):
        value = config.get(key, DEFAULT_CONFIG[key])
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{key} must be finite and > 0, got {value!r}")

    # -- Defect B (speech temporal support) ----------------------------------
    support_window = config.get("speech_support_window_seconds", DEFAULT_CONFIG["speech_support_window_seconds"])
    if not (math.isfinite(support_window) and support_window > 0.0):
        raise ValueError(f"speech_support_window_seconds must be finite and > 0, got {support_window!r}")

    fraction = config.get("speech_minimum_positive_fraction", DEFAULT_CONFIG["speech_minimum_positive_fraction"])
    if not (math.isfinite(fraction) and 0.0 <= fraction <= 1.0):
        raise ValueError(f"speech_minimum_positive_fraction must be finite in [0, 1], got {fraction!r}")

    for key in ("speech_activity_delta_threshold", "speech_activity_scale"):
        value = config.get(key, DEFAULT_CONFIG[key])
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{key} must be finite and > 0, got {value!r}")

    # -- Change C (use_confidence ablation flag) -----------------------------
    use_confidence = config.get("use_confidence", DEFAULT_CONFIG["use_confidence"])
    if not isinstance(use_confidence, bool):
        raise ValueError(f"use_confidence must be a bool, got {use_confidence!r}")
