"""Unit tests for gaze_attention.jsonl_logger — uses pytest's tmp_path so
nothing lands in the real logs/ directory. Mirrors tests/test_logger.py's
conventions; SessionLogger itself is untouched and has its own tests."""

import json

from gaze_attention.jsonl_logger import JsonlLogger, new_jsonl_path


def _obs(id_=0, confidence=0.9, lost=False, bbox=(1.0, 2.0, 3.0, 4.0)):
    return {"id": id_, "bbox": list(bbox), "confidence": confidence, "lost": lost}


def _read_lines(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_new_jsonl_path_unique(tmp_path):
    p1 = new_jsonl_path(tmp_path)
    p1.touch()
    p2 = new_jsonl_path(tmp_path)
    assert p1 != p2
    assert not p2.exists()


def test_log_frame_writes_one_line_per_person_with_full_schema(tmp_path):
    with JsonlLogger(logs_dir=tmp_path) as log:
        log.log_frame(
            t=1.5,
            observations=[_obs(id_=0), _obs(id_=1, confidence=0.4, lost=True)],
            metrics_by_id={0: {"engagement": 1.0, "score": 1.0}, 1: {"engagement": 0.2, "score": 0.2}},
        )
        path = log.path

    rows = _read_lines(path)
    assert len(rows) == 2
    assert rows[0]["type"] == "frame"
    assert rows[0]["t"] == 1.5
    assert rows[0]["id"] == 0
    assert rows[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert rows[0]["engagement"] == 1.0
    assert rows[0]["score"] == 1.0
    assert rows[0]["lost"] is False
    # Change A/B fields not implemented yet -- must be present as None, not
    # missing, so the schema is stable before those changes land.
    for key in ("yaw", "pitch", "phi", "e", "q", "c_size", "c_pose", "c_stable", "C"):
        assert key in rows[0]
        assert rows[0][key] is None

    assert rows[1]["id"] == 1
    assert rows[1]["lost"] is True


def test_log_switch_writes_one_event_line(tmp_path):
    with JsonlLogger(logs_dir=tmp_path) as log:
        log.log_switch(
            t=3.0, from_id=0, to_id=1, reason="fallback_score",
            scores={0: 1.2, 1: 1.5}, confidences={0: 0.8, 1: 0.9},
        )
        path = log.path

    rows = _read_lines(path)
    assert len(rows) == 1
    row = rows[0]
    assert row["type"] == "switch"
    assert row["t"] == 3.0
    assert row["from_id"] == 0
    assert row["to_id"] == 1
    assert row["reason"] == "fallback_score"
    # JSON object keys are always strings.
    assert row["scores"] == {"0": 1.2, "1": 1.5}
    assert row["confidences"] == {"0": 0.8, "1": 0.9}


def test_log_switch_handles_none_from_id_initial_selection(tmp_path):
    with JsonlLogger(logs_dir=tmp_path) as log:
        log.log_switch(t=0.0, from_id=None, to_id=0, reason="initial", scores={0: 0.1}, confidences={0: 0.9})
        path = log.path
    rows = _read_lines(path)
    assert rows[0]["from_id"] is None
    assert rows[0]["to_id"] == 0


def test_disabled_writes_nothing(tmp_path):
    log = JsonlLogger(logs_dir=tmp_path, enabled=False)
    log.log_frame(t=0.0, observations=[_obs()], metrics_by_id={})
    log.log_switch(t=0.0, from_id=None, to_id=0, reason=None, scores={}, confidences={})
    log.close()
    assert _read_lines(log.path) == []


def test_toggle_enabled_resumes_same_file(tmp_path):
    log = JsonlLogger(logs_dir=tmp_path, enabled=True)
    log.log_frame(t=0.0, observations=[_obs(id_=0)], metrics_by_id={})
    log.enabled = False
    log.log_frame(t=1.0, observations=[_obs(id_=1)], metrics_by_id={})
    log.enabled = True
    log.log_frame(t=2.0, observations=[_obs(id_=2)], metrics_by_id={})
    log.close()
    rows = _read_lines(log.path)
    assert {r["id"] for r in rows} == {0, 2}


def test_missing_metrics_defaults_to_none_not_crash(tmp_path):
    with JsonlLogger(logs_dir=tmp_path) as log:
        log.log_frame(t=0.0, observations=[_obs(id_=5)], metrics_by_id={})
        path = log.path
    rows = _read_lines(path)
    assert rows[0]["engagement"] is None
    assert rows[0]["score"] is None
