#!/usr/bin/env bash
# manual_eye_test.sh — manually exercises the reactive-gaze scene's
# /attention_target_angle topic, so the two irises can be verified to move
# correctly before the webcam policy is ever connected. No static person
# markers in the world (see two_eye_demo.sdf) -- these are just fixed
# bearings for manual verification, standing in for where a real face
# would be.
#
# Run attention_gazebo_two_eye.launch.py first (in another terminal), then
# run this script. It cycles through left/center/right bearings
# (eye_gaze_node.TARGET_ANGLE_MAP: -0.65, 0.0, +0.65 rad), pausing between
# each so the smoothed approach is visible, then sends an out-of-range
# angle to demonstrate clamping to [-0.8, 0.8] rad.
#
# Usage:
#   source /opt/ros/humble/setup.bash   # if not already sourced
#   ./scripts/manual_eye_test.sh
#
# Equivalent to running these three commands by hand (the exact manual test
# commands, one-shot with --once rather than looping):
#   ros2 topic pub --once /attention_target_angle std_msgs/msg/Float32 "{data: -0.65}"
#   ros2 topic pub --once /attention_target_angle std_msgs/msg/Float32 "{data: 0.0}"
#   ros2 topic pub --once /attention_target_angle std_msgs/msg/Float32 "{data: 0.65}"
# Expected result: -0.65 -> both black irises look left; 0.0 -> both look
# forward; 0.65 -> both look right.

set -euo pipefail

pub() {
    echo "[manual_eye_test] publishing angle=${1} rad"
    ros2 topic pub --once /attention_target_angle std_msgs/msg/Float32 "{data: ${1}}"
}

echo "[manual_eye_test] left bearing (-0.65 rad) -- both irises should look left"
pub -0.65
sleep 2

echo "[manual_eye_test] center bearing (0.0 rad) -- both irises should look forward"
pub 0.0
sleep 2

echo "[manual_eye_test] right bearing (+0.65 rad) -- both irises should look right"
pub 0.65
sleep 2

echo "[manual_eye_test] out-of-range angle (1.5 rad) -- should clamp to +0.8 rad"
pub 1.5
sleep 2

echo "[manual_eye_test] back to center"
pub 0.0

echo "[manual_eye_test] done. Check eye_gaze_node's log output for 'clamped to' and 'iris yaw: ... -> target ...' lines."
