"""
main_webcam.py — the live demo: webcam -> perception -> filter -> policy ->
overlay -> CSV log -> ROS 2 publish, all swappable at runtime.

Run (from multi_person/, with the repo-root venv active):
    python3 -m gaze_attention.main_webcam
    python3 -m gaze_attention.main_webcam --fallback-perception --no-ros
    python3 -m gaze_attention.main_webcam --active-policy greedy --filter moving_average  # explicit baseline comparison only

Default active policy is conversational_simple (see policies.py) --
GreedyDwellPolicy (gaze_attention/baseline_policies.py) and
ConfidenceDebtPolicy are retained for explicit baseline comparison only,
never selected by default.

Keyboard controls (display window focused):
    q  quit
    b  switch to GreedyDwellPolicy (baseline-only, explicit choice)
    p  switch to ConfidenceDebtPolicy (baseline-only, explicit choice)
    c  switch to ConversationalSimplePolicy (the active policy)
    f  cycle filter (moving_average -> bayesian -> ...)
    r  reset policy state
    l  toggle CSV logging on/off
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gaze_attention import load_config, raw_dt_from_timestamps, sanitize_dt
from gaze_attention.attention_manager import AttentionManager
from gaze_attention.filter_graph import FilterComparisonTracker, LiveFilterGraph
from gaze_attention.filters import FILTERS, make_filter
from gaze_attention.jsonl_logger import JsonlLogger
from gaze_attention.logger import SessionLogger
from gaze_attention.perception import Perception
from gaze_attention.policies import POLICIES, make_policy
from gaze_attention.policy_comparison import (
    PolicyComparisonGraph, SwitchCounter, draw_baseline_panel, draw_comparison_hud,
)
from gaze_attention.target_angle import compute_target_angle, compute_pitch_angle 
from gaze_attention.visualization import draw_overlay

_FILTER_CYCLE = list(FILTERS)


class RosPublisher:
    """Publishes the selected attention target and its current gaze angles.

    ROS topics:
      * /attention_target_id       Int32   -- WHO
      * /attention_target_angle    Float32 -- yaw
      * /attention_target_pitch    Float32 -- pitch

    Yaw and pitch are deliberately independent:
      * yaw follows the selected person's horizontal face position.
      * pitch follows the selected person's vertical face position.
    """

    def __init__(self, enabled: bool, min_publish_interval_seconds: float = 0.2):
        self.connected = False
        self.min_publish_interval_seconds = min_publish_interval_seconds

        self.last_published_angle: Optional[float] = None
        self.last_published_pitch: Optional[float] = None

        self._node = None
        self._target_pub = None
        self._angle_pub = None
        self._pitch_pub = None

        self._last_published_target_id: Optional[int] = None
        self._last_publish_time: Optional[float] = None

        if not enabled:
            print("[ros] ROS publishing disabled by config/flag — visualization-only mode.")
            return

        try:
            import rclpy
            from std_msgs.msg import Float32, Int32

            rclpy.init(args=None)
            self._rclpy = rclpy
            self._node = rclpy.create_node("gaze_attention_publisher")

            self._target_pub = self._node.create_publisher(
                Int32, "/attention_target_id", 10
            )

            self._angle_pub = self._node.create_publisher(
                Float32, "/attention_target_angle", 10
            )

            self._pitch_pub = self._node.create_publisher(
                Float32, "/attention_target_pitch", 10
            )

            self._Int32 = Int32
            self._Float32 = Float32

            self.connected = True

            print(
                "[ros] publishers ready: "
                "/attention_target_id, "
                "/attention_target_angle, "
                "/attention_target_pitch"
            )

        except Exception as exc:
            print(
                f"[ros] ROS 2 unavailable ({exc}) — "
                "falling back to visualization-only mode."
            )
            self.connected = False

    def publish(
        self,
        target_id: Optional[int],
        angle: Optional[float],
        pitch: Optional[float],
        now: Optional[float] = None,
    ) -> bool:
        """Publish selected target, yaw angle and pitch angle."""

        if not self.connected or target_id is None:
            return False

        now = time.time() if now is None else now

        changed = target_id != self._last_published_target_id

        due = (
            self._last_publish_time is None
            or (now - self._last_publish_time)
            >= self.min_publish_interval_seconds
        )

        if not (changed or due):
            return False

        angle = float(angle if angle is not None else 0.0)
        pitch = float(pitch if pitch is not None else 0.0)

        # WHO
        self._target_pub.publish(
            self._Int32(data=int(target_id))
        )

        # YAW
        self._angle_pub.publish(
            self._Float32(data=angle)
        )

        # PITCH
        self._pitch_pub.publish(
            self._Float32(data=pitch)
        )

        self._last_published_target_id = target_id
        self._last_publish_time = now

        self.last_published_angle = angle
        self.last_published_pitch = pitch

        return True

    def shutdown(self) -> None:
        if self.connected:
            self._node.destroy_node()
            self._rclpy.shutdown()



def _bbox_for(target_id: Optional[int], policy_observations: list) -> Optional[list]:
    if target_id is None:
        return None
    for obs in policy_observations:
        if obs["id"] == target_id:
            return obs["bbox"]
    return None  # target held through an absence-grace tick -- no bbox this frame, target_angle_for falls back


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Confidence-aware multi-person attention demo (webcam).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--active-policy", choices=list(POLICIES), default=None,
        help="Top-level target-selection policy actually driving the robot. Default: conversational_simple "
             "(speaker-first; see policies.py/README.md). confidence_debt and greedy (baseline_policies.py, "
             "BASELINE-ONLY -- never the default) remain available for explicit comparison only.",
    )
    parser.add_argument(
        "--policy", choices=[n for n in POLICIES if n != "conversational_simple"], default=None,
        help="Base policy used when --active-policy is greedy/confidence_debt (or wrapped by AttentionManager "
             "via --speaker-priority-mode). Has no effect when --active-policy=conversational_simple. "
             "'greedy' is baseline-only (gaze_attention/baseline_policies.py) -- never selected by default.",
    )
    parser.add_argument("--filter", choices=list(FILTERS), default=None)
    parser.add_argument("--fallback-perception", action="store_true")
    parser.add_argument("--no-ros", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Auto-exit after N frames (smoke testing).")
    parser.add_argument("--debug", action="store_true", help="Print raw_dt/used_dt (and other diagnostics) every tick.")
    parser.add_argument(
        "--speech-mode", choices=["off", "mouth_motion"], default=None,
        help="visual_speaking_proxy: mouth-motion auxiliary cue, NOT audio speech recognition (no microphone).",
    )
    parser.add_argument(
        "--speaker-priority-mode", choices=["off", "on"], default=None,
        help="AttentionManager: conversational/socially-bounded gaze layer around the base policy. "
             "Needs --speech-mode mouth_motion to have any effect. Default: off (unchanged gaze-only behavior).",
    )
    parser.add_argument(
        "--show-filter-graph", action="store_true",
        help="Live filter-comparison graph (raw/moving_average/bayesian) for one tracked person, purely "
             "observational -- never affects the active policy/filter. Falls back to headless CSV-only "
             "logging automatically if no display is available.",
    )
    parser.add_argument(
        "--filter-graph-window", type=float, default=15.0,
        help="Scrolling window (seconds) for --show-filter-graph. Default: 15.",
    )
    parser.add_argument(
        "--graph-person-id", type=int, default=0,
        help="Tracked person id to graph with --show-filter-graph. Default: 0.",
    )
    parser.add_argument(
        "--compare-baseline", action="store_true",
        help="Run ConfidenceDebtPolicy alongside the active policy (ConversationalSimplePolicy by "
             "default), purely for comparison: a second, view-only webcam panel plus a live "
             "'Target ID over time' popup graph. The baseline NEVER controls ROS/Gazebo -- the "
             "active policy remains the only source of ROS commands, exactly as without this flag. "
             "Falls back to headless CSV-only logging automatically if no display is available.",
    )
    return parser.parse_args(argv)


def _make_policy(name: str, config: dict):
    kwargs = {
        "confidence_threshold": config["confidence_threshold"],
        "minimum_hold_seconds": config["minimum_hold_seconds"],
        "target_grace_seconds": config["target_grace_seconds"],
        "engagement_decay": config["engagement_decay"],
        "speech_mode": config["speech_mode"],
        "speech_weight": config["speech_weight"],
        "use_confidence": config["use_confidence"],
    }
    if name == "confidence_debt":
        kwargs["debt_weight"] = config["debt_weight"]
        kwargs["engagement_max"] = config["engagement_max"]
        kwargs["debt_max"] = config["debt_max"]
    base_policy = make_policy(name, **kwargs)

    if config["speaker_priority_mode"] == "off":
        return base_policy
    return AttentionManager(
        base_policy,
        speaker_priority_mode=config["speaker_priority_mode"],
        speaker_confirm_seconds=config["speaker_confirm_seconds"],
        speaker_min_attention_seconds=config["speaker_min_attention_seconds"],
        speaker_release_seconds=config["speaker_release_seconds"],
        quiet_gaze_confirm_seconds=config["quiet_gaze_confirm_seconds"],
        quiet_acknowledgment_seconds=config["quiet_acknowledgment_seconds"],
        max_continuous_gaze_seconds=config["max_continuous_gaze_seconds"],
        gaze_break_seconds=config["gaze_break_seconds"],
        stare_penalty_weight=config["stare_penalty_weight"],
        confidence_threshold=config["confidence_threshold"],
        use_confidence=config["use_confidence"],
    )


def _make_conversational_simple(config: dict):
    return make_policy(
        "conversational_simple",
        confidence_threshold=config["confidence_threshold"],
        minimum_hold_seconds=config["minimum_hold_seconds"],
        target_grace_seconds=config["target_grace_seconds"],
        speech_mode=config["speech_mode"],
        speaker_confirm_seconds=config["speaker_confirm_seconds"],
        speaker_release_seconds=config["speaker_release_seconds"],
        speaker_switch_seconds=config["speaker_switch_seconds"],
        minimum_speaker_attention_seconds=config["minimum_speaker_attention_seconds"],
        speaker_confidence_threshold=config["speaker_confidence_threshold"],
        non_speaking_score_decay=config["non_speaking_score_decay"],
        recent_attention_penalty_weight=config["recent_attention_penalty_weight"],
        max_continuous_gaze_seconds=config["max_continuous_gaze_seconds"],
        quiet_decay_rate=config["quiet_decay_rate"],
        quiet_penalty_max=config["quiet_penalty_max"],
        use_confidence=config["use_confidence"],
        fallback_switch_margin=config["fallback_switch_margin"],
        debt_requires_speech=config["debt_requires_speech"],
        # Root Cause #3: previously class defaults, unreachable from config.
        speaker_switch_confirm_seconds=config["speaker_switch_confirm_seconds"],
        speech_off_confirm_seconds=config["speech_off_confirm_seconds"],
        speaker_absence_grace_seconds=config["speaker_absence_grace_seconds"],
    )


def _make_active_policy(config: dict):
    """Dispatches on config["active_policy"]: "conversational_simple" (the
    default, speaker-first policy -- see policies.py's docstring) is
    self-contained and never wrapped; "greedy"/"confidence_debt" go through
    _make_policy (optionally wrapped by AttentionManager, see
    speaker_priority_mode) and remain available as baselines, per the
    explicit requirement not to use them as the active conversational
    policy by default."""
    name = config["active_policy"]
    if name == "conversational_simple":
        return _make_conversational_simple(config)
    return _make_policy(name, config)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    if args.policy:
        config["policy"] = args.policy
        config["active_policy"] = args.policy
    if args.active_policy:
        config["active_policy"] = args.active_policy
    if args.filter:
        config["filter"] = args.filter
    if args.no_ros:
        config["ros_enabled"] = False
    if args.no_display:
        config["display_enabled"] = False
    if args.camera_index is not None:
        config["camera_index"] = args.camera_index
    if args.debug:
        config["debug"] = True
    if args.speech_mode:
        config["speech_mode"] = args.speech_mode
    if args.speaker_priority_mode:
        config["speaker_priority_mode"] = args.speaker_priority_mode

    import cv2

    perception = Perception(config, force_fallback=args.fallback_perception)
    filt = make_filter(config["filter"])
    policy = _make_active_policy(config)
    ros_pub = RosPublisher(enabled=config["ros_enabled"], min_publish_interval_seconds=config["ros_min_publish_interval_seconds"])
    logger = SessionLogger(enabled=config["log_enabled"])
    # JSONL diagnostic logger (Change D) -- independent of SessionLogger,
    # never modifies it, runs alongside it. See jsonl_logger.py's docstring
    # for the schema.
    jsonl_logger = JsonlLogger(enabled=config["log_enabled"])
    angle_warned: set = set()
    known_ids: set = set()
    previous_target: Optional[int] = None  # for switch-event detection, any policy

    # Live filter-comparison graph (requirement: never touches the active
    # policy/filter -- separate tracker instance, fed the same raw
    # observations, purely observational). Headless automatically when
    # there's no display window to show it next to.
    filter_tracker = None
    filter_graph = None
    if args.show_filter_graph:
        filter_tracker = FilterComparisonTracker()
        filter_graph = LiveFilterGraph(
            person_id=args.graph_person_id,
            seeking_threshold=config["seeking_threshold"],
            window_seconds=args.filter_graph_window,
            headless=not config["display_enabled"],
        )
        print(f"[filter_graph] tracking person_id={args.graph_person_id}, "
              f"window={args.filter_graph_window}s, csv -> {filter_graph.csv_path}")

    # Live current-vs-baseline target comparison (requirement: purely
    # observational -- a separate ConfidenceDebtPolicy instance, fed the
    # exact same policy_observations/dt as the active policy, with fully
    # independent internal state. It never reaches ros_pub.publish -- see
    # the "current model must remain the only ROS/Gazebo command source"
    # requirement, and gaze_attention/policy_comparison.py's own docstring.
    baseline_policy = None
    comparison_graph = None
    comparison_start_time: Optional[float] = None
    current_switch_counter = None
    baseline_switch_counter = None
    if args.compare_baseline:
        baseline_policy = _make_policy("confidence_debt", config)
        comparison_graph = PolicyComparisonGraph(headless=not config["display_enabled"])
        comparison_start_time = time.monotonic()
        current_switch_counter = SwitchCounter()
        baseline_switch_counter = SwitchCounter()
        print(f"[policy_comparison] comparing active_policy={config['active_policy']} "
              f"(ROS-connected) vs baseline=confidence_debt (view-only), "
              f"csv -> {comparison_graph.csv_path}")

    print(f"[main_webcam] active_policy={config['active_policy']} filter={config['filter']} "
          f"speech_mode={config['speech_mode']} fallback_perception={args.fallback_perception} "
          f"ros_enabled={ros_pub.connected} log -> {logger.path} jsonl -> {jsonl_logger.path}")

    cap = cv2.VideoCapture(config["camera_index"])
    if not cap.isOpened():
        print(f"[main_webcam] Could not open webcam (index {config['camera_index']}).")
        return 1

    frame_index = 0
    last_t: Optional[float] = None  # None on the first frame -> dt falls back safely (see sanitize_dt)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[main_webcam] Camera read failed — ending session.")
                break
            frame = cv2.flip(frame, 1)

            now = time.time()
            raw_dt = raw_dt_from_timestamps(now, last_t)
            dt, _ = sanitize_dt(
                raw_dt,
                fallback=config["dt_fallback_seconds"],
                max_dt=config["dt_max_seconds"],
                debug=config["debug"],
            )
            last_t = now

            raw_observations = perception.process_frame(frame, now)

            current_ids = {obs["id"] for obs in raw_observations}
            for gone_id in known_ids - current_ids:
                filt.forget(gone_id)
            known_ids = current_ids

            # Filter-comparison graph: a completely separate tracker/filter
            # pair, fed the same raw_observations, never touching `filt` or
            # `policy_observations` below -- see filter_graph.py's docstring.
            filter_graph_results = None
            if filter_tracker is not None:
                filter_graph_results = filter_tracker.update(raw_observations, dt)

            # Confidence fields (see DEBUG_ATTENTION.md FM1): raw_confidence
            # is perception.py's own per-frame output; filter_uncertainty is
            # the filter's diagnostic uncertainty about its smoothed signal
            # estimate (filters.py) and is NEVER used for gating; policy_confidence
            # is what confidence_threshold actually applies to (policies.py's
            # Observation "confidence" key). Version 1: policy_confidence is
            # raw_confidence unmodified -- no filter-derived value is allowed
            # to silently stand in for it. A calibrated transform, if one is
            # ever added, must be documented here explicitly.
            policy_observations = []
            for obs in raw_observations:
                raw_confidence = obs["confidence"]
                filtered_signal, filter_uncertainty = filt.update(obs["id"], obs["gaze_signal"], raw_confidence, dt)
                policy_confidence = raw_confidence
                seeking = filtered_signal >= config["seeking_threshold"]
                policy_observations.append({
                    "id": obs["id"],
                    "bbox": obs["bbox"],
                    "gaze_signal": filtered_signal,          # pinned key policies.py reads
                    "confidence": policy_confidence,          # pinned key policies.py reads
                    "raw_confidence": raw_confidence,
                    "filter_uncertainty": filter_uncertainty,
                    "policy_confidence": policy_confidence,
                    "seeking_attention": seeking,
                    "timestamp": obs["timestamp"],
                    "fallback": obs.get("fallback", False),
                    "lost": obs.get("lost", False),
                    # Mutual-gaze residual diagnostics (Change A) -- carried
                    # through so the JSONL logger (Change D) can record them;
                    # None when head pose was unavailable this frame.
                    "yaw": obs.get("yaw"),
                    "pitch": obs.get("pitch"),
                    "phi": obs.get("phi"),
                    "e": obs.get("e"),
                    "q": obs.get("q"),
                    # Confidence components (Change B) -- carried through for
                    # the JSONL logger; None on lost observations.
                    "c_size": obs.get("c_size"),
                    "c_pose": obs.get("c_pose"),
                    "c_stable": obs.get("c_stable"),
                    "C": obs.get("C"),
                    # visual_speaking_proxy (see speech.py) -- always carried
                    # through regardless of speech_mode, so the overlay/log
                    # can show it even when policies aren't acting on it.
                    "mouth_activity": obs.get("mouth_activity", 0.0),
                    "speaking": obs.get("speaking", False),
                    "speech_confidence": obs.get("speech_confidence", 0.0),
                })

            target = policy.update(policy_observations, dt)
            metrics_by_id = {obs["id"]: policy.get_metrics(obs["id"]) for obs in policy_observations}

            # Baseline comparison (view-only): fed the EXACT SAME
            # policy_observations list and dt as the active policy above --
            # never a copy, since no policy mutates its observations (see
            # policy_comparison.py's docstring) -- so both policies are
            # provably seeing identical input. baseline_target is used only
            # for the comparison panel/graph/log below; it is never read by
            # the ROS publish block a few lines down, which only ever uses
            # `target` (the active policy's result), unchanged from before
            # this flag existed.
            baseline_target = None
            baseline_metrics_by_id = {}
            if baseline_policy is not None:
                baseline_target = baseline_policy.update(policy_observations, dt)
                baseline_metrics_by_id = {
                    obs["id"]: baseline_policy.get_metrics(obs["id"]) for obs in policy_observations
                }

            # The SELECTED person's CURRENT face position drives the angle --
            # independent of policy state (policies.py is untouched). Falls
            # back to the fixed id->angle mapping when face coordinates
            # aren't usable this tick (target held through an absence-grace
            # miss, or fallback-perception with no real bbox).
            #
            # RAW angle, published as-is -- no smoothing here. Smoothing
            # (a single slew-rate limiter) lives entirely downstream in
            # eye_gaze_node.py, the stage closest to the actuator, so it
            # also covers manual `ros2 topic pub` commands that never go
            # through this process at all. Publishing an already-smoothed
            # value here would stack a second, redundant filter in front of
            # that one and needlessly add latency -- see git history for the
            # latency inspection that led to this.
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]

            target_bbox = _bbox_for(target, policy_observations)

            angle_result = compute_target_angle(
                target, target_bbox, frame_width,
                max_angle=config["target_angle_max"], warned=angle_warned,
            )

            pitch_result = compute_pitch_angle(
                target, target_bbox, frame_height,
                max_angle=config.get("target_pitch_max", 0.5),
                warned=angle_warned,
            )

            ros_pub.publish(
                target,
                angle_result.raw_angle,
                pitch_result.raw_angle,
                now,
            )

            
            # Fast speaker-first path diagnostics (ConversationalSimplePolicy
            # only -- getattr defaults keep this a no-op for other policies).
            active_speaker_id = getattr(policy, "active_speaker_id", None)
            speaker_confirmed_time_by_id = getattr(policy, "speaker_confirmed_time", {})
            speaker_release_time_by_id = getattr(policy, "speaker_release_time", {})

            logger.log_frame(
                frame_index, policy_observations, metrics_by_id, target, config["active_policy"], config["filter"],
                target_angle=angle_result.raw_angle, published_angle=ros_pub.last_published_angle,
                active_speaker_id=active_speaker_id,
                speaker_confirmed_time=speaker_confirmed_time_by_id.get(active_speaker_id) if active_speaker_id is not None else None,
                speaker_release_time=max(speaker_release_time_by_id.values()) if speaker_release_time_by_id else None,
                target_switch_reason=getattr(policy, "last_switch_reason", None) if getattr(policy, "switched_this_tick", False) else None,
                switch_latency_seconds=getattr(policy, "last_switch_latency_seconds", None) if getattr(policy, "switched_this_tick", False) else None,
                face_center_x=angle_result.face_center_x,
                frame_width=frame_width,
                normalized_x=angle_result.normalized_x,
                raw_target_angle=angle_result.raw_angle,
            )

            jsonl_logger.log_frame(now, policy_observations, metrics_by_id)
            if target != previous_target:
                switch_scores = {}
                switch_confidences = {}
                for pid in (previous_target, target):
                    if pid is None:
                        continue
                    if pid in metrics_by_id:
                        switch_scores[pid] = metrics_by_id[pid].get("score")
                    obs_match = next((o for o in policy_observations if o["id"] == pid), None)
                    if obs_match is not None:
                        switch_confidences[pid] = obs_match.get("confidence")
                jsonl_logger.log_switch(
                    now, previous_target, target,
                    getattr(policy, "last_switch_reason", None),
                    switch_scores, switch_confidences,
                )
            previous_target = target

            fps = 1.0 / dt
            if filter_graph is not None and filter_graph_results is not None:
                graphed_values = filter_graph_results.get(args.graph_person_id)
                if graphed_values is not None:
                    filter_graph.push(timestamp=now, values=graphed_values, current_target=target, fps=fps)

            # Comparison bookkeeping (switch counts + graph/CSV) -- independent
            # of display_enabled, same as filter_graph's own push() above, so
            # headless runs still produce the comparison CSV.
            if comparison_graph is not None:
                current_switch_counter.update(target)
                baseline_switch_counter.update(baseline_target)
                comparison_elapsed = time.monotonic() - comparison_start_time
                comparison_graph.push(
                    elapsed_time=comparison_elapsed,
                    current_target=target,
                    baseline_target=baseline_target,
                    current_switch_count=current_switch_counter.count,
                    baseline_switch_count=baseline_switch_counter.count,
                    confirmed_speaker_id=getattr(policy, "active_speaker_id", None),
                    baseline_confirmed_speaker_id=getattr(baseline_policy, "active_speaker_id", None),
                )

            if config["display_enabled"]:
                # Baseline panel must be drawn on a COPY of the frame taken
                # BEFORE draw_overlay mutates `frame` in place below --
                # otherwise the baseline panel would show the current
                # model's own overlay underneath its own.
                baseline_frame = frame.copy() if comparison_graph is not None else None

                draw_overlay(
                    frame, policy_observations, metrics_by_id, target,
                    config["active_policy"], config["filter"], fps,
                    ros_pub.connected, perception.last_frame_used_fallback,
                    logger.enabled, config["confidence_threshold"],
                    config["speech_mode"],
                )
                if comparison_graph is not None:
                    # Additive-only HUD strip (see policy_comparison.py) --
                    # never modifies visualization.py's own top HUD bar.
                    draw_comparison_hud(
                        frame, "CURRENT MODEL", config["active_policy"], target,
                        getattr(policy, "active_speaker_id", None),
                        current_switch_counter.count, ros_active=ros_pub.connected,
                    )
                cv2.imshow(
                    "Multi-Person Attention Demo (q=quit b=greedy p=confidence_debt c=conversational_simple f=filter r=reset l=log)",
                    frame,
                )

                if baseline_frame is not None:
                    draw_baseline_panel(
                        baseline_frame, policy_observations, baseline_metrics_by_id,
                        baseline_target, config["confidence_threshold"],
                        getattr(baseline_policy, "active_speaker_id", None),
                        baseline_switch_counter.count,
                    )
                    cv2.imshow("Baseline Comparison (view only) -- ConfidenceDebtPolicy", baseline_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("b"):
                    config["policy"] = "greedy"
                    config["active_policy"] = "greedy"
                    policy = _make_policy("greedy", config)
                    print("[main_webcam] active_policy -> greedy (baseline)")
                elif key == ord("p"):
                    config["policy"] = "confidence_debt"
                    config["active_policy"] = "confidence_debt"
                    policy = _make_policy("confidence_debt", config)
                    print("[main_webcam] active_policy -> confidence_debt (baseline)")
                elif key == ord("c"):
                    config["active_policy"] = "conversational_simple"
                    policy = _make_conversational_simple(config)
                    print("[main_webcam] active_policy -> conversational_simple")
                elif key == ord("f"):
                    idx = _FILTER_CYCLE.index(config["filter"])
                    config["filter"] = _FILTER_CYCLE[(idx + 1) % len(_FILTER_CYCLE)]
                    filt = make_filter(config["filter"])
                    known_ids = set()
                    print(f"[main_webcam] filter -> {config['filter']}")
                elif key == ord("r"):
                    policy.reset()
                    print("[main_webcam] policy state reset")
                elif key == ord("l"):
                    logger.enabled = not logger.enabled
                    jsonl_logger.enabled = logger.enabled
                    print(f"[main_webcam] logging -> {logger.enabled}")

            frame_index += 1
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n[main_webcam] interrupted, shutting down.")
    finally:
        cap.release()
        if config["display_enabled"]:
            cv2.destroyAllWindows()
        logger.close()
        jsonl_logger.close()
        perception.close()
        ros_pub.shutdown()
        if filter_graph is not None:
            filter_graph.close()
        if comparison_graph is not None:
            comparison_graph.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
