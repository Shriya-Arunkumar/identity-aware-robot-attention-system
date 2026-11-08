#!/usr/bin/env bash
# Manual yaw test for the textured-eye Gazebo Classic world.
# Start gazebo/worlds/eye_test.world and eye_gaze_node first (see README.md).
# Run this with the webcam stopped so it does not override test commands.
# Publishes left/center/right, an out-of-range command, then center.
# Both eyes should follow; the controller clamps 1.5 to +0.7854 radians.

set -euo pipefail

pub() {
    echo "[manual_eye_test] publishing angle=${1} rad"
    ros2 topic pub --once /attention_target_angle std_msgs/msg/Float32 "{data: ${1}}"
}

echo "[manual_eye_test] left bearing (-0.65 rad) -- both eyes should look left"
pub -0.65
sleep 2

echo "[manual_eye_test] center bearing (0.0 rad) -- both eyes should look forward"
pub 0.0
sleep 2

echo "[manual_eye_test] right bearing (+0.65 rad) -- both eyes should look right"
pub 0.65
sleep 2

echo "[manual_eye_test] out-of-range angle (1.5 rad) -- should clamp to +0.7854 rad"
pub 1.5
sleep 2

echo "[manual_eye_test] back to center"
pub 0.0

echo "[manual_eye_test] done. Check eye_gaze_node's log output for 'clamped to' and 'iris yaw: ... -> target ...' lines."
