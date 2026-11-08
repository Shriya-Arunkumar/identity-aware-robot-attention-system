"""Unit tests for gaze_attention.logger — uses pytest's tmp_path so nothing
lands in the real logs/ directory."""

import csv

from gaze_attention.logger import CSV_FIELDS, SessionLogger, new_log_path


def _obs(id_=0, gaze_signal=0.9, confidence=0.9, seeking=True, t=0.0):
    return {
        "id": id_,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "gaze_signal": gaze_signal,
        "confidence": confidence,
        "seeking_attention": seeking,
        "timestamp": t,
    }


def test_new_log_path_unique(tmp_path):
    p1 = new_log_path(tmp_path)
    p1.touch()
    p2 = new_log_path(tmp_path)
    assert p1 != p2
    assert not p2.exists()


def test_never_overwrites_existing_session(tmp_path, monkeypatch):
    # Force the same timestamp twice in a row to hit the collision branch.
    import gaze_attention.logger as logger_mod

    class _FixedDatetime:
        @staticmethod
        def now():
            import datetime as dt
            return dt.datetime(2026, 1, 1, 12, 0, 0)

    monkeypatch.setattr(logger_mod, "datetime", _FixedDatetime)
    p1 = new_log_path(tmp_path)
    p1.touch()
    p2 = new_log_path(tmp_path)
    assert p1 != p2


def test_logger_writes_header_and_rows(tmp_path):
    with SessionLogger(logs_dir=tmp_path) as log:
        log.log_frame(
            frame_index=0,
            observations=[_obs(id_=0), _obs(id_=1, gaze_signal=0.1, seeking=False)],
            metrics_by_id={0: {"engagement": 1.0, "debt": 0.0, "score": 1.0}, 1: {"engagement": 0.0, "debt": 0.2, "score": 0.1}},
            selected_target=0,
            policy_name="confidence_debt",
            filter_name="moving_average",
        )
        path = log.path

    with open(path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDS
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["person_id"] == "0"
    assert rows[0]["selected_target"] == "0"
    assert rows[1]["debt"] == "0.2"


def test_logger_disabled_writes_nothing(tmp_path):
    log = SessionLogger(logs_dir=tmp_path, enabled=False)
    log.log_frame(0, [_obs()], {0: {"engagement": 0, "debt": 0, "score": 0}}, 0, "greedy", "moving_average")
    log.close()
    with open(log.path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows == []


def test_logger_toggle_enabled_resumes_same_file(tmp_path):
    log = SessionLogger(logs_dir=tmp_path, enabled=True)
    log.log_frame(0, [_obs()], {0: {"engagement": 0, "debt": 0, "score": 0}}, 0, "greedy", "moving_average")
    log.enabled = False
    log.log_frame(1, [_obs()], {0: {"engagement": 0, "debt": 0, "score": 0}}, 0, "greedy", "moving_average")
    log.enabled = True
    log.log_frame(2, [_obs()], {0: {"engagement": 0, "debt": 0, "score": 0}}, 0, "greedy", "moving_average")
    log.close()
    with open(log.path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {r["frame_index"] for r in rows} == {"0", "2"}


def test_missing_metrics_defaults_to_zero(tmp_path):
    with SessionLogger(logs_dir=tmp_path) as log:
        log.log_frame(0, [_obs(id_=5)], metrics_by_id={}, selected_target=None, policy_name="greedy", filter_name="bayesian")
        path = log.path
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["engagement"] == "0.0"
