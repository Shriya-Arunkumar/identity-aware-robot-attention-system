#pragma once

#include <algorithm>
#include <cmath>

namespace gaze_attention_gazebo
{
// Both eye instances use world simulation time, so their blinks stay synchronized
// across pauses, resets and different plugin load times. No ROS publisher needed.
inline double BlinkClosure(
  double time, double interval, double close_time, double hold_time, double open_time)
{
  if (!std::isfinite(time) || interval <= 0.0 || close_time <= 0.0 ||
    hold_time < 0.0 || open_time <= 0.0 ||
    close_time + hold_time + open_time >= interval || time < interval)
  {
    return 0.0;
  }
  const double phase = std::fmod(time, interval);
  double progress = 0.0;
  if (phase < close_time) {
    progress = phase / close_time;
  } else if (phase < close_time + hold_time) {
    return 1.0;
  } else if (phase < close_time + hold_time + open_time) {
    progress = 1.0 - (phase - close_time - hold_time) / open_time;
  } else {
    return 0.0;
  }
  progress = std::clamp(progress, 0.0, 1.0);
  return progress * progress * (3.0 - 2.0 * progress);
}
}  // namespace gaze_attention_gazebo
