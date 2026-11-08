"""
replay.py — no-camera synthetic replay: deterministic scenarios run through
every (filter x policy) combination, exercising the exact pipeline shape
main_webcam.py uses (raw observation -> filter.update -> recompute
seeking_attention on the filtered signal -> policy.update), without any
camera, MediaPipe, ROS, or display dependency.

Two jobs:
  1. `run_all_combinations()` — the "must not NaN/crash, for every
     filter/policy combination" gate from the spec. Run this before trusting
     the webcam path.
  2. `run_scenario()` — produces a per-frame row list with exactly the CSV
     log schema's fields, which `run_policy_replay.sh` / logger.py reuse so a
     recorded webcam session and a synthetic scenario can be analyzed by the
     same evaluation/metrics.py code.

Also loadable as a CSV replay: `frames_from_csv()` turns a previously logged
session back into the same per-frame observation format, so a filter/policy
combination can be re-run against real recorded data (see run_policy_replay.sh).
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Callable, Iterator, List, Optional, TypedDict

# Allow running as `python3 evaluation/replay.py` from multi_person/ without
# installing the package.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gaze_attention import DEFAULT_CONFIG, load_config, raw_dt_from_timestamps, sanitize_dt
from gaze_attention.filters import FILTERS, make_filter
from gaze_attention.logger import CSV_FIELDS
from gaze_attention.policies import POLICIES, make_policy
from gaze_attention.target_angle import compute_target_angle


class RawFrame(TypedDict):
    """One simulated tick of *pre-filter* perception output: which people are
    present and their raw (unfiltered) gaze_signal/confidence this tick."""
    t: float
    dt: float
    people: List[dict]  # [{"id": int, "gaze_signal": float, "confidence": float}, ...]


Scenario = Callable[[], Iterator[RawFrame]]


# ---------------------------------------------------------------------------
# Synthetic scenarios (spec: "Synthetic test scenarios" section).
# ---------------------------------------------------------------------------
def _frames(duration_s: float, dt: float, people_at: Callable[[float], List[dict]]) -> Iterator[RawFrame]:
    t = 0.0
    while t < duration_s:
        yield RawFrame(t=t, dt=dt, people=people_at(t))
        t += dt


def scenario_one_person(dt: float = 0.033, duration_s: float = 5.0) -> Iterator[RawFrame]:
    """A single person, steadily seeking attention with good confidence."""
    return _frames(duration_s, dt, lambda t: [{"id": 0, "gaze_signal": 0.9, "confidence": 0.9}])


def scenario_two_people_equal(dt: float = 0.033, duration_s: float = 8.0) -> Iterator[RawFrame]:
    """Two people, identical signal/confidence throughout — exercises
    deterministic tie-breaking and (for the debt policy) the fairness switch."""
    return _frames(
        duration_s, dt,
        lambda t: [{"id": 0, "gaze_signal": 0.9, "confidence": 0.9}, {"id": 1, "gaze_signal": 0.9, "confidence": 0.9}],
    )


def scenario_persistent_vs_intermittent(dt: float = 0.033, duration_s: float = 10.0) -> Iterator[RawFrame]:
    """Person 0 seeks continuously; person 1 seeks in ~0.33s on/off bursts."""
    def people_at(t: float) -> List[dict]:
        intermittent_on = int(t / 0.33) % 2 == 0
        return [
            {"id": 0, "gaze_signal": 0.9, "confidence": 0.9},
            {"id": 1, "gaze_signal": 0.9 if intermittent_on else 0.1, "confidence": 0.9},
        ]
    return _frames(duration_s, dt, people_at)


def scenario_low_confidence(dt: float = 0.033, duration_s: float = 6.0) -> Iterator[RawFrame]:
    """A single person who appears to seek attention but with uniformly low
    confidence throughout — should never wrongly accumulate engagement/debt."""
    return _frames(duration_s, dt, lambda t: [{"id": 0, "gaze_signal": 0.9, "confidence": 0.05}])


def scenario_enter_and_leave(dt: float = 0.033, duration_s: float = 9.0) -> Iterator[RawFrame]:
    """Person 0 present for the first third, gone for the middle third,
    replaced by person 1 (new id, simulating a dropped-then-new track) for
    the final third."""
    def people_at(t: float) -> List[dict]:
        if t < duration_s / 3:
            return [{"id": 0, "gaze_signal": 0.9, "confidence": 0.9}]
        elif t < 2 * duration_s / 3:
            return []
        else:
            return [{"id": 1, "gaze_signal": 0.9, "confidence": 0.9}]
    return _frames(duration_s, dt, people_at)


SCENARIOS: dict[str, Scenario] = {
    "one_person": scenario_one_person,
    "two_people_equal": scenario_two_people_equal,
    "persistent_vs_intermittent": scenario_persistent_vs_intermittent,
    "low_confidence": scenario_low_confidence,
    "enter_and_leave": scenario_enter_and_leave,
}


# ---------------------------------------------------------------------------
# Core replay loop.
# ---------------------------------------------------------------------------
def run_scenario(
    frames: Iterator[RawFrame],
    filter_name: str,
    policy_name: str,
    config: Optional[dict] = None,
    debug: bool = False,
) -> List[dict]:
    """Run one scenario through one (filter, policy) combination. Returns a
    list of CSV-schema row dicts, one per person per frame.

    Confidence fields, per DEBUG_ATTENTION.md FM1 (mirrors
    main_webcam.py — see its docstring for the full rationale):
      * raw_confidence     — the scenario's own `person["confidence"]`,
                              standing in for perception.py's raw output.
      * filtered_signal    — filt.update()'s smoothed signal.
      * filter_uncertainty — filt.update()'s second return value; a
                              filter-diagnostic value, logged but NEVER fed
                              into policy gating.
      * policy_confidence  — "Version 1": raw_confidence, unmodified. This
                              is what `confidence_threshold` actually gates
                              on (see policies.py's Observation "confidence"
                              key, which carries this value, not
                              filter_uncertainty).
    """
    config = config or DEFAULT_CONFIG
    seeking_threshold = config["seeking_threshold"]

    filt = make_filter(filter_name)
    policy = make_policy(
        policy_name,
        confidence_threshold=config["confidence_threshold"],
        minimum_hold_seconds=config["minimum_hold_seconds"],
        target_grace_seconds=config["target_grace_seconds"],
        engagement_decay=config["engagement_decay"],
        speech_mode=config["speech_mode"],
        speech_weight=config["speech_weight"],
        use_confidence=config["use_confidence"],
        **(
            {"debt_weight": config["debt_weight"], "engagement_max": config["engagement_max"], "debt_max": config["debt_max"]}
            if policy_name == "confidence_debt" else {}
        ),
        **(
            {
                "speaker_confirm_seconds": config["speaker_confirm_seconds"],
                "speaker_release_seconds": config["speaker_release_seconds"],
                "speaker_switch_seconds": config["speaker_switch_seconds"],
                "minimum_speaker_attention_seconds": config["minimum_speaker_attention_seconds"],
                "speaker_confidence_threshold": config["speaker_confidence_threshold"],
                "non_speaking_score_decay": config["non_speaking_score_decay"],
                "recent_attention_penalty_weight": config["recent_attention_penalty_weight"],
                "max_continuous_gaze_seconds": config["max_continuous_gaze_seconds"],
                "quiet_decay_rate": config["quiet_decay_rate"],
                "quiet_penalty_max": config["quiet_penalty_max"],
                "fallback_switch_margin": config["fallback_switch_margin"],
                "debt_requires_speech": config["debt_requires_speech"],
                "speaker_switch_confirm_seconds": config["speaker_switch_confirm_seconds"],
                "speech_off_confirm_seconds": config["speech_off_confirm_seconds"],
                "speaker_absence_grace_seconds": config["speaker_absence_grace_seconds"],
            }
            if policy_name == "conversational_simple" else {}
        ),
    )

    rows: List[dict] = []
    for frame_index, frame in enumerate(frames):
        # dt handling (DEBUG_ATTENTION.md FM4): sanitize the scenario's/CSV's
        # own dt the same way main_webcam.py sanitizes real timestamps --
        # missing/negative/zero/non-finite -> fallback; always clamped to
        # [0, dt_max_seconds] so a bad scenario/CSV row can't produce a huge
        # single-tick score jump.
        dt, raw_dt = sanitize_dt(
            frame.get("dt"),
            fallback=config["dt_fallback_seconds"],
            max_dt=config["dt_max_seconds"],
            debug=debug or config.get("debug", False),
        )

        policy_observations = []
        for person in frame["people"]:
            pid = person["id"]
            raw_confidence = person["confidence"]
            filtered_signal, filter_uncertainty = filt.update(
                pid, person["gaze_signal"], raw_confidence, dt
            )
            policy_confidence = raw_confidence  # Version 1: identity, see docstring above.
            seeking = filtered_signal >= seeking_threshold
            policy_observations.append({
                "id": pid,
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "gaze_signal": filtered_signal,          # pinned key policies.py reads
                "confidence": policy_confidence,          # pinned key policies.py reads
                "raw_confidence": raw_confidence,
                "filter_uncertainty": filter_uncertainty,
                "policy_confidence": policy_confidence,
                "seeking_attention": seeking,
                "timestamp": frame["t"],
                # visual_speaking_proxy: synthetic scenarios don't carry
                # mouth data by default (defaults to not-speaking); a
                # scenario/test can still set these on its "people" dicts.
                "mouth_activity": person.get("mouth_activity", 0.0),
                "speaking": person.get("speaking", False),
                "speech_confidence": person.get("speech_confidence", 0.0),
            })

        selected = policy.update(policy_observations, dt)
        # Same fallback id -> angle mapping main_webcam.py falls back to
        # when face coordinates aren't usable (see target_angle.py) --
        # synthetic scenarios don't carry real face bboxes, so replay
        # always takes the fallback path, never the face-position path.
        # Replay has no RosPublisher/throttling layer, so published_angle
        # == target_angle here always -- there's no separate "wire" for it
        # to lag behind.
        target_angle = compute_target_angle(selected, None, None).raw_angle

        # Fast speaker-first path diagnostics (ConversationalSimplePolicy
        # only -- getattr defaults keep this a no-op for other policies).
        active_speaker_id = getattr(policy, "active_speaker_id", None)
        speaker_confirmed_time_by_id = getattr(policy, "speaker_confirmed_time", {})
        speaker_release_time_by_id = getattr(policy, "speaker_release_time", {})

        for obs in policy_observations:
            metrics = policy.get_metrics(obs["id"])
            rows.append({
                "timestamp": obs["timestamp"],
                "frame_index": frame_index,
                "person_id": obs["id"],
                "x1": obs["bbox"][0], "y1": obs["bbox"][1], "x2": obs["bbox"][2], "y2": obs["bbox"][3],
                "raw_confidence": obs["raw_confidence"],
                "filtered_signal": obs["gaze_signal"],
                "filter_uncertainty": obs["filter_uncertainty"],
                "policy_confidence": obs["policy_confidence"],
                "mouth_activity": obs["mouth_activity"],
                "speaking": obs["speaking"],
                "speech_confidence": obs["speech_confidence"],
                "seeking_attention": obs["seeking_attention"],
                "engagement": metrics["engagement"],
                "debt": metrics["debt"],
                "score": metrics["score"],
                "speech_bonus": metrics.get("speech_bonus", 0.0),
                "selected_target": selected,
                "target_angle": target_angle,
                "published_angle": target_angle,
                "speaking_true_duration": metrics.get("speaking_true_duration", ""),
                "speaking_false_duration": metrics.get("speaking_false_duration", ""),
                "quiet_penalty": metrics.get("quiet_penalty", ""),
                "is_active_speaker": metrics.get("is_active_speaker", ""),
                "active_speaker_id": active_speaker_id if active_speaker_id is not None else "",
                "speaker_confirmed_time": speaker_confirmed_time_by_id.get(active_speaker_id, "") if active_speaker_id is not None else "",
                "speaker_release_time": max(speaker_release_time_by_id.values()) if speaker_release_time_by_id else "",
                "target_switch_reason": (getattr(policy, "last_switch_reason", None) or "") if getattr(policy, "switched_this_tick", False) else "",
                "switch_latency_seconds": getattr(policy, "last_switch_latency_seconds", "") if getattr(policy, "switched_this_tick", False) else "",
                "policy_name": policy_name,
                "filter_name": filter_name,
            })

        # Every filtered/derived value must be finite — the actual gate this
        # harness exists to enforce before the webcam path is trusted.
        for row in rows[-len(policy_observations):]:
            for k in ("raw_confidence", "filtered_signal", "filter_uncertainty", "policy_confidence", "mouth_activity", "speech_confidence", "engagement", "debt", "score", "speech_bonus"):
                v = row[k]
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    raise FloatingPointError(
                        f"non-finite {k}={v} from {filter_name}/{policy_name} at frame {frame_index}"
                    )

    return rows


def run_all_combinations(verbose: bool = True) -> bool:
    """The pre-webcam gate: every scenario x every filter x every policy must
    run start-to-finish with no NaN, no Inf, no exception. Returns True if
    everything passed."""
    config = DEFAULT_CONFIG
    ok = True
    for scenario_name, scenario_fn in SCENARIOS.items():
        for filter_name in FILTERS:
            for policy_name in POLICIES:
                try:
                    rows = run_scenario(scenario_fn(), filter_name, policy_name, config)
                    status = f"OK ({len(rows)} rows)"
                except Exception as exc:  # noqa: BLE001 - this is the crash gate itself
                    ok = False
                    status = f"FAILED: {exc}"
                if verbose:
                    print(f"[{scenario_name:28s}] {filter_name:14s} x {policy_name:15s} -> {status}")
    return ok


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def frames_from_csv(csv_path: Path, config: Optional[dict] = None) -> Iterator[RawFrame]:
    """Replay a previously logged CSV session as RawFrames. Uses the log's
    `raw_confidence` column (the actual pre-filter confidence perception.py
    emitted) as the replay's confidence input — NOT `policy_confidence` or
    `filter_uncertainty`, per this project's confidence-field separation
    (DEBUG_ATTENTION.md FM1). The `filtered_signal` column is fed back in as
    the replay's `gaze_signal` input (there is no logged raw pre-filter
    signal to use instead), so re-running this through a filter re-smooths
    an already-smoothed signal — a pre-existing limitation of CSV replay,
    unchanged by this fix.

    dt is computed per-frame from consecutive real timestamps
    (`raw_dt_from_timestamps` + `sanitize_dt`) — not a single
    whole-session median, which would hide per-frame stalls/drops. The
    first frame has no previous timestamp, so it falls back safely (see
    sanitize_dt) rather than reading as dt=0."""
    import pandas as pd

    config = config or DEFAULT_CONFIG
    df = pd.read_csv(csv_path)
    previous_t: Optional[float] = None
    for frame_index, group in df.groupby("frame_index"):
        t = float(group["timestamp"].iloc[0])
        dt, _ = sanitize_dt(
            raw_dt_from_timestamps(t, previous_t),
            fallback=config["dt_fallback_seconds"],
            max_dt=config["dt_max_seconds"],
        )
        previous_t = t
        people = [
            {"id": int(r.person_id), "gaze_signal": float(r.filtered_signal), "confidence": float(r.raw_confidence)}
            for r in group.itertuples()
        ]
        yield RawFrame(t=t, dt=dt, people=people)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run synthetic replay scenarios.")
    parser.add_argument("--check-all", action="store_true", help="Run every scenario x filter x policy combination.")
    parser.add_argument("--scenario", choices=list(SCENARIOS), help="Run a single scenario.")
    parser.add_argument("--filter", choices=list(FILTERS), default="moving_average")
    parser.add_argument("--policy", choices=list(POLICIES), default="confidence_debt")
    parser.add_argument("--csv-in", type=str, default=None, help="Replay a previously logged CSV instead of a synthetic scenario.")
    parser.add_argument("--out", type=str, default=None, help="Write result rows to this CSV path.")
    parser.add_argument("--debug", action="store_true", help="Print raw_dt/used_dt every tick.")
    args = parser.parse_args()

    config = load_config()
    if args.debug:
        config["debug"] = True

    if args.check_all:
        ok = run_all_combinations()
        sys.exit(0 if ok else 1)

    if args.csv_in:
        frames = frames_from_csv(Path(args.csv_in), config)
    elif args.scenario:
        frames = SCENARIOS[args.scenario]()
    else:
        parser.error("pass --check-all, --scenario, or --csv-in")

    rows = run_scenario(frames, args.filter, args.policy, config, debug=args.debug)
    print(f"{len(rows)} rows produced, no NaN/Inf/crash.")
    if args.out:
        write_csv(rows, Path(args.out))
        print(f"wrote {args.out}")
