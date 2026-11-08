# Multi-person robot attention with Gazebo eyes

The project is in `robot_attention_clean/`. It combines a multi-person webcam
preview, attention arbitration, and two textured blue eyes in **Gazebo Classic 11**.
ROS 2 Humble carries the selected person's angle to a custom C++ joint plugin.
Gazebo Sim and `ros_gz_bridge` are not required for this version.

The current world opens facing a procedurally sculpted gray humanoid head.
Both eyes have articulated eyelids and blink together every 3.5 seconds of
simulation time (55 ms closing, 15 ms closed, 105 ms reopening). Blinks run in
the Gazebo plugin even without the webcam process; pausing simulation also pauses
blinking. Timing is configurable in `gazebo/models/eye/model.sdf` under
`blink_interval`, `blink_close_time`, `blink_hold_time`, and `blink_open_time`.
The head is an original approximation, not a scan of a human reference image.
Regenerate its meshes with `python3 scripts/generate_face_meshes.py` from
`robot_attention_clean/`. Rebuild the C++ plugin after changing its animation code.

## Contents

- `gaze_attention/`: perception, tracking, policies, preview, and logging.
- `gazebo/`: the eye world, model, and blue iris texture.
- `ros2_ws/src/`: the ROS eye controller and Gazebo Classic plugin.
- `config/`: attention and camera settings.
- `models/`: MediaPipe Face Landmarker bundle and download information.
- `tests/`, `evaluation/`, `docs/`: validation and offline analysis.

Virtual environments, build outputs, session logs, and caches are excluded from Git.

## Prerequisites

Ubuntu 22.04, ROS 2 Humble, colcon, Gazebo Classic 11, `gazebo_ros`,
`gazebo_dev`, `rclcpp`, and `std_msgs`. The camera process also requires the
Python packages in `robot_attention_clean/requirements.txt` and access to a webcam
and graphical desktop. Create a local Python 3.10 environment named `venv` and
install those requirements when setting up a fresh clone; environments are not shipped.
Source ROS before running Python processes that publish ROS topics.

The perception adapter optionally imports `gaze_core.gaze_estimator` from the
separate single-person project. That library is not included here. If absent,
the code uses its labelled fallback perception; do not interpret fallback values
as the original iris-based gaze estimates. See `gaze_attention/perception.py`.
Speaking activity is inferred from mouth motion, not microphone audio.

## Build once

From the repository root, in a terminal with ROS available:

```bash
source /opt/ros/humble/setup.bash
cd robot_attention_clean/ros2_ws
/usr/bin/colcon --log-base log_current build --build-base build_current --install-base install_current --packages-select eye_yaw_plugin gaze_attention_ros --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

## Run in three terminals

In each terminal, start in the repository root. Keep the same ROS domain settings
in all terminals. Stop other programs using the webcam first.

Terminal 1: simulator and eye plugin.

```bash
source /opt/ros/humble/setup.bash
cd robot_attention_clean
source ros2_ws/install_current/setup.bash
gazebo --verbose -s libgazebo_ros_init.so "$PWD/gazebo/worlds/eye_test.world"
```

Terminal 2: yaw controller.

```bash
source /opt/ros/humble/setup.bash
cd robot_attention_clean
source ros2_ws/install_current/setup.bash
ros2 run gaze_attention_ros eye_gaze_node
```

Terminal 3: camera preview and target publisher.

```bash
source /opt/ros/humble/setup.bash
cd robot_attention_clean
./venv/bin/python3 -m gaze_attention.main_webcam --active-policy conversational_simple --speech-mode mouth_motion
```

The preview displays tracked faces and the selected target. The controller should
report `eye_yaw_plugin connected`; both eyes follow the selected horizontal angle.
Press `q` in the preview and Ctrl+C in the other terminals to stop.

Optional vertical movement requires a fourth terminal, sourced like Terminal 3:

```bash
./venv/bin/python3 -m gaze_attention.pitch_relay
```
