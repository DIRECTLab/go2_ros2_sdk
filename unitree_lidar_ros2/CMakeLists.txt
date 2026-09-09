cmake_minimum_required(VERSION 3.5)
project(unitree_lidar_ros2)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)

# ---------------------------------------------------------------------------
# Locate unilidar_sdk2
#
# The vendor SDK ships as headers plus a prebuilt static library per
# architecture, and is NOT a ROS package -- it is not on the ament index and
# find_package cannot see it. The rover's copy of this file hardcodes
# $ENV{HOME}/wheeltec_dev/unilidar_sdk2, which only works on that robot. This
# searches the places the SDK is actually cloned instead, and fails loudly with
# the clone command rather than emitting an unfindable-header error 200 lines
# later.
#
# Override explicitly with either:
#   colcon build --cmake-args -DUNILIDAR_SDK_DIR=/path/to/unilidar_sdk2/unitree_lidar_sdk
#   export UNILIDAR_SDK_DIR=/path/to/unilidar_sdk2/unitree_lidar_sdk
# ---------------------------------------------------------------------------
set(UNILIDAR_SDK_DIR "" CACHE PATH
    "Path to unilidar_sdk2/unitree_lidar_sdk (the directory holding include/ and lib/)")

if(NOT UNILIDAR_SDK_DIR AND DEFINED ENV{UNILIDAR_SDK_DIR})
  set(UNILIDAR_SDK_DIR "$ENV{UNILIDAR_SDK_DIR}")
endif()

if(NOT UNILIDAR_SDK_DIR)
  # Searched in order. The first two cover the SDK vendored inside this
  # workspace's src/ (either as a sibling package directory or a git
  # submodule); the rest cover a standalone clone next to the workspace or in
  # the home directory.
  set(_unilidar_candidates
      "${CMAKE_CURRENT_SOURCE_DIR}/../unilidar_sdk2/unitree_lidar_sdk"
      "${CMAKE_CURRENT_SOURCE_DIR}/unilidar_sdk2/unitree_lidar_sdk"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../unilidar_sdk2/unitree_lidar_sdk"
      "$ENV{HOME}/direct/unilidar_sdk2/unitree_lidar_sdk"
      "$ENV{HOME}/unilidar_sdk2/unitree_lidar_sdk"
      "/opt/unilidar_sdk2/unitree_lidar_sdk")
  foreach(_candidate ${_unilidar_candidates})
    if(EXISTS "${_candidate}/include/unitree_lidar_sdk.h")
      get_filename_component(UNILIDAR_SDK_DIR "${_candidate}" ABSOLUTE)
      break()
    endif()
  endforeach()
endif()

if(UNILIDAR_SDK_DIR AND NOT EXISTS "${UNILIDAR_SDK_DIR}/include/unitree_lidar_sdk.h")
  message(FATAL_ERROR
    "UNILIDAR_SDK_DIR was set explicitly but holds no SDK:\n"
    "  ${UNILIDAR_SDK_DIR}\n"
    "Expected ${UNILIDAR_SDK_DIR}/include/unitree_lidar_sdk.h. Point it at the\n"
    "unitree_lidar_sdk directory inside the clone, not the clone root.")
endif()

if(NOT UNILIDAR_SDK_DIR)
  message(FATAL_ERROR
    "unilidar_sdk2 not found. This package drives an external Unitree L2 over\n"
    "serial/UDP and needs the vendor SDK's headers and static library.\n"
    "\n"
    "  git clone https://github.com/unitreerobotics/unilidar_sdk2.git ~/unilidar_sdk2\n"
    "\n"
    "then rebuild, or point at an existing clone:\n"
    "  colcon build --packages-select unitree_lidar_ros2 \\\n"
    "    --cmake-args -DUNILIDAR_SDK_DIR=/path/to/unilidar_sdk2/unitree_lidar_sdk\n"
    "\n"
    "Searched: ${_unilidar_candidates}")
endif()

# The SDK ships one static library per architecture under lib/<arch>, named for
# uname -m: lib/x86_64 and lib/aarch64. CMAKE_SYSTEM_PROCESSOR matches those on
# both the dev host and the robot's Jetson, so no mapping table is needed --
# but a mismatch here surfaces as an inscrutable linker error, so it is checked.
set(UNILIDAR_SDK_LIB_DIR "${UNILIDAR_SDK_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR}")
if(NOT EXISTS "${UNILIDAR_SDK_LIB_DIR}/libunilidar_sdk2.a")
  message(FATAL_ERROR
    "unilidar_sdk2 has no prebuilt library for this architecture.\n"
    "  looked for: ${UNILIDAR_SDK_LIB_DIR}/libunilidar_sdk2.a\n"
    "  CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}\n"
    "The SDK ships x86_64 and aarch64 only.")
endif()

message(STATUS "unilidar_sdk2: ${UNILIDAR_SDK_DIR} (lib: ${CMAKE_SYSTEM_PROCESSOR})")

include_directories(
  ${PCL_INCLUDE_DIRS}
  include
  ${UNILIDAR_SDK_DIR}/include
)

link_directories(
  ${PCL_LIBRARY_DIRS}
  ${UNILIDAR_SDK_LIB_DIR}
)

add_definitions(${PCL_DEFINITIONS})

add_executable(unitree_lidar_ros2_node src/unitree_lidar_ros2_node.cpp)

target_link_libraries( unitree_lidar_ros2_node
  ${Boost_SYSTEM_LIBRARY}
  ${PCL_LIBRARIES}
  libunilidar_sdk2.a
)

ament_target_dependencies(
  unitree_lidar_ros2_node
  rclcpp std_msgs
  sensor_msgs
  geometry_msgs
  tf2_ros
  pcl_conversions
)

install(TARGETS
unitree_lidar_ros2_node
  DESTINATION lib/${PROJECT_NAME}
)

# Launch file and rviz config. Installed into share/<pkg>/launch so
# `ros2 launch unitree_lidar_ros2 launch.py` resolves it the way every other
# package in this workspace does; the rover's copy installs both flat into
# share/<pkg>, which ros2 launch does not search.
install(FILES rviz/view.rviz
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
