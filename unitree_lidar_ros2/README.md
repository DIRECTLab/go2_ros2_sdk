# unitree_lidar_ros2

ROS 2 driver for the Unitree Lidar L2, talking to the sensor directly over
serial or UDP through [`unilidar_sdk2`](https://github.com/unitreerobotics/unilidar_sdk2).

Ported from the rover stack so both robots run the same driver, the same topic
names and the same parameter contract. The node source and headers are the
vendor's, unmodified; only `CMakeLists.txt` and `launch/launch.py` differ, so
the SDK is located portably rather than through a hardcoded home directory.

This drives an **externally mounted** L2. It has nothing to do with the Go2's
own factory-mounted utlidar, which reaches this stack over CycloneDDS through
`go2_robot_sdk`'s `raw_lidar_node`.

## Build

The vendor SDK is headers plus a prebuilt static library per architecture, is
not a ROS package, and is not on the ament index — clone it separately:

```bash
git clone https://github.com/unitreerobotics/unilidar_sdk2.git ~/unilidar_sdk2
colcon build --symlink-install --packages-select unitree_lidar_ros2
```

`CMakeLists.txt` searches the workspace, `~/direct`, `~` and `/opt` for the
clone. Override with `-DUNILIDAR_SDK_DIR=` or the `UNILIDAR_SDK_DIR`
environment variable.

## Run

Normal operation goes through the robot stack, which derives `tf_prefix`-aware
frame ids, publishes the mount transform and wires `fov_mask` onto the output:

```bash
ros2 launch go2_robot_sdk robot.launch.py lidar_source:=external
```

`lidar_source` also takes `internal`, `both` and `none`. Masking is off for this
feed by default — both robots mount the L2 upright, so the two sensors already
observe the same part of the world.

`launch/launch.py` here brings the sensor up on its own for hardware checks. It
publishes no transform, so the cloud has no place in the robot's TF tree.

Full documentation — serial vs UDP wiring, the udev rule, the work-mode/TF
authority constraint, mount calibration and feeding Swarm-SLAM — is in
`go2_robot_sdk`'s [`docs/EXTERNAL_LIDAR.md`](../docs/EXTERNAL_LIDAR.md).
