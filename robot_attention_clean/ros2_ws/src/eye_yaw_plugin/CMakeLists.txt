cmake_minimum_required(VERSION 3.8)
project(eye_yaw_plugin)

find_package(ament_cmake REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_library(eye_yaw_plugin SHARED src/eye_yaw_plugin.cpp)
target_include_directories(eye_yaw_plugin PUBLIC ${GAZEBO_INCLUDE_DIRS})
ament_target_dependencies(eye_yaw_plugin
  gazebo_dev
  gazebo_ros
  rclcpp
  std_msgs
)

install(TARGETS eye_yaw_plugin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# Project configuration (per user requirement: don't rely on manually
# exporting GAZEBO_PLUGIN_PATH/GAZEBO_MODEL_PATH every session) -- this hook
# runs automatically whenever `source install/setup.bash` is sourced. Paths
# are baked in now (configure time) rather than derived at source time from
# AMENT_CURRENT_PREFIX/COLCON_CURRENT_PREFIX, since which of those colcon
# actually sets varies by sourcing mechanism.
get_filename_component(EYE_YAW_PLUGIN_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(PLUGIN_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(GAZEBO_MODELS_DIR "${EYE_YAW_PLUGIN_REPO_ROOT}/gazebo/models")
configure_file(
  hook/50-eye-yaw-plugin-gazebo-paths.sh.in
  ${CMAKE_CURRENT_BINARY_DIR}/hook/50-eye-yaw-plugin-gazebo-paths.sh
  @ONLY
)
ament_environment_hooks(${CMAKE_CURRENT_BINARY_DIR}/hook/50-eye-yaw-plugin-gazebo-paths.sh)

ament_package()
