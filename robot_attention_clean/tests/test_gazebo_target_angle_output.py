"""Tests for connecting the existing selected-target output to Gazebo:
main_webcam.py's RosPublisher (target-id -> angle conversion, throttled
/attention_target_angle publishing) and logger.py's target_angle/
published_angle CSV columns.

RosPublisher is tested directly (not through main_webcam.main(), which
needs a real webcam) with ROS 2 disabled (`enabled=False`) -- that's
enough to exercise the id->angle mapping and the throttling decision logic
without any ROS/rclpy dependency; `connected` stays False either way, and
the throttling math is identical regardless of whether a message would
actually reach a subscriber.
"""

from __future__ import annotations

from gaze_attention.logger import CSV_FIELDS, SessionLogger
from gaze_attention.main_webcam import RosPublisher
from gaze_attention.target_angle import target_angle_for

# No real bbox -- these tests exercise the fallback id->angle mapping path
# (requirement 11), same as main_webcam.py does when a target is held
# through an absence-grace tick with no bbox this frame.


# --- 1. target id -> fallback angle conversion (no face position available) --

def test_target_angle_for_uses_fallback_mapping_without_a_bbox():
    assert target_angle_for(0, None, 640) == -0.65
    assert target_angle_for(1, None, 640) == 0.0
    assert target_angle_for(2, None, 640) == 0.65
    assert target_angle_for(None, None, 640) is None


def test_target_angle_for_defaults_unmapped_id_to_center():
    warned = set()
    assert target_angle_for(99, None, 640, warned=warned) == 0.0
    assert 99 in warned  # warns once, doesn't re-warn every frame


# --- 2. publish() with ROS disabled never raises, never "connects" -----------

def test_publish_disabled_is_a_safe_no_op():
    pub = RosPublisher(enabled=False)
    assert pub.connected is False
    assert pub.publish(0, -0.8, now=0.0) is False
    assert pub.last_published_angle is None


# --- 3. throttling: publish only on change or at the low fixed rate ----------

def test_publish_throttles_when_target_and_time_are_unchanged():
    """With ROS disabled, publish() always returns False for the connection
    reason -- so this drives the throttle decision logic directly through a
    disabled-but-otherwise-real RosPublisher isn't possible without a live
    rclpy node. Instead, verify the throttle bookkeeping (last target id /
    last publish time / last published angle) only advances on an actual
    publish by using a stub that reports itself as connected."""
    pub = RosPublisher(enabled=False, min_publish_interval_seconds=1.0)
    pub.connected = True  # simulate a connected publisher without rclpy
    pub._target_pub = _StubPub()
    pub._angle_pub = _StubPub()
    pub._Int32 = _StubMsg
    pub._Float32 = _StubMsg

    # First publish for a given target always goes out (id changed from None).
    assert pub.publish(0, -0.8, now=0.0) is True
    assert pub.last_published_angle == -0.8

    # Same target, well within the interval -> throttled (no publish).
    assert pub.publish(0, -0.8, now=0.2) is False
    assert pub.last_published_angle == -0.8  # unchanged, since nothing was sent

    # Same target, interval elapsed -> low-fixed-rate publish goes through.
    assert pub.publish(0, -0.8, now=1.1) is True

    # Different target, even immediately -> always publishes (change overrides throttle).
    assert pub.publish(1, 0.0, now=1.15) is True
    assert pub.last_published_angle == 0.0


class _StubPub:
    def publish(self, msg):
        pass


class _StubMsg:
    def __init__(self, data):
        self.data = data


# --- 4. logger.py: target_angle / published_angle columns --------------------

def test_csv_fields_include_target_angle_and_published_angle():
    assert "target_angle" in CSV_FIELDS
    assert "published_angle" in CSV_FIELDS


def test_log_frame_writes_target_angle_and_published_angle(tmp_path):
    logger = SessionLogger(logs_dir=tmp_path)
    obs = [{
        "id": 0, "bbox": [0, 0, 10, 10], "timestamp": 1.0, "gaze_signal": 0.9,
        "confidence": 0.9, "seeking_attention": True,
    }]
    metrics = {0: {"engagement": 1.0, "debt": 0.0, "score": 1.0, "speech_bonus": 0.0}}

    logger.log_frame(0, obs, metrics, selected_target=0, policy_name="conversational_simple",
                      filter_name="moving_average", target_angle=-0.8, published_angle=-0.8)
    logger.close()

    import csv as csv_module
    with open(logger.path) as f:
        rows = list(csv_module.DictReader(f))
    assert rows[0]["target_angle"] == "-0.8"
    assert rows[0]["published_angle"] == "-0.8"


# --- 5. end-to-end test sequence: 0 -> 1 -> 2 -> 0 ----------------------------

def test_target_sequence_0_1_2_0_maps_and_publishes_correctly():
    """For each step, the (fallback, no-bbox) angle mapping must match, and
    RosPublisher must publish on every step (since the target id changes
    every time, the "changed" condition always wins over the throttle
    interval)."""
    pub = RosPublisher(enabled=False, min_publish_interval_seconds=10.0)  # deliberately slow rate
    pub.connected = True
    pub._target_pub = _StubPub()
    pub._angle_pub = _StubPub()
    pub._Int32 = _StubMsg
    pub._Float32 = _StubMsg

    sequence = [0, 1, 2, 0]
    t = 0.0
    published_angles = []
    for target_id in sequence:
        angle = target_angle_for(target_id, None, 640)
        did_publish = pub.publish(target_id, angle, now=t)
        assert did_publish is True, f"a target change (-> {target_id}) must always publish, even mid-throttle-interval"
        published_angles.append(pub.last_published_angle)
        t += 0.05  # well under the 10s throttle interval -- only the id change should matter

    assert published_angles == [-0.65, 0.0, 0.65, -0.65]
