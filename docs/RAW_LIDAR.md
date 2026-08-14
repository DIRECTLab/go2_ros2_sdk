# Raw LiDAR over CycloneDDS (`raw_lidar_node`)

Publishes the robot's onboard L2 lidar feed — taken straight off the
`rt/utlidar/cloud` DDS topic over the Ethernet link — as a standard
`sensor_msgs/PointCloud2`.

This is **additive**. Nothing about the driver's existing WebRTC point cloud
(`/go2/point_cloud2`) changes, and the node is off by default in the launch file.

## Why this exists rather than reusing the WebRTC path

The WebRTC channel carries `rt/utlidar/voxel_map_compressed`, whose points are
quantised to a 5 cm lattice **on the robot**, before transport
(`lidar_decoder.py` multiplies integer voxel indices by the resolution).
`docs/SWARM_SLAM_FINDINGS.md` §2 measured this: 100% of nearest-neighbour
distances on that feed are exactly 0.0500 m.

Registration front-ends that build local-sampling descriptors — cslam's
FPFH→TEASER, KISS-ICP — need the true per-scan sampling, which only the DDS path
carries. Hence `ChannelFactoryInitialize` + `ChannelSubscriber`, not a WebRTC
derivative.

**Sanity check in RViz2:** a non-repetitive scanner produces a dense
rosette/flower accumulation pattern. If you see a clean concentric lattice
instead, something upstream changed and this is no longer the raw feed.

> **Note on `docs/SWARM_SLAM_FINDINGS.md`.** That investigation (2026-08-05)
> found no `rt/utlidar/cloud` in the robot's DDS presence and concluded raw
> lidar was firmware-gated on that unit. This node is built against a *live-
> introspected* field layout from the current robot, which supersedes that
> finding. The older document is left as-is rather than edited, since its
> measurements stand on their own; treat §2's availability claim as stale.

## Field layout

Confirmed by live introspection. Copied through byte-for-byte — the node does
not re-encode points, so what lands on the ROS topic is what the sensor sent.
Note the padding, and that `intensity` is at offset **16**, not the offset 12
most drivers use.

| name | offset | datatype | dtype |
|---|---|---|---|
| `x` | 0 | 7 (FLOAT32) | float32 |
| `y` | 4 | 7 (FLOAT32) | float32 |
| `z` | 8 | 7 (FLOAT32) | float32 |
| *(pad)* | 12–15 | | |
| `intensity` | 16 | 7 (FLOAT32) | float32 |
| `ring` | 20 | 4 (UINT16) | uint16 |
| *(pad)* | 22–23 | | |
| `time` | 24 | 7 (FLOAT32) | float32 |
| *(pad)* | 28–31 | | |

`point_step = 32`. `height` is always 1; `width` varies per scan (~3600–4000).
That variation is expected for a non-repetitive scanner and is not a bug.

If an incoming message ever declares a layout that disagrees with the table
above, the node logs a prominent warning once; if `point_step` itself
disagrees, it **drops** the message rather than relabelling the payload with
offsets that are now provably wrong.

## Timestamps

`stamp_source` selects the header stamp basis. Default is `raw`.

| value | stamp |
|---|---|
| `raw` (default) | DDS message header stamp **+** the earliest per-point `time` offset |
| `raw_header` | DDS message header stamp, verbatim |
| `receive` | this node's clock |

The per-point `time` field **cannot** be the absolute stamp on its own: it is
FLOAT32, and a 24-bit mantissa quantises epoch seconds (~1.7e9) to 128-second
steps — measured, not assumed. It is an intra-scan offset. So the absolute basis
comes from the robot's own message header, with the earliest per-point offset
applied on top; that is the reference instant deskewing front-ends assume.

`min()` is used rather than the first point's value because firmware differs on
whether offsets run from scan start (≥ 0) or scan end (≤ 0).

Prefer `receive` only for debugging. It stamps with node arrival time, which
folds in network and scheduling jitter and corrupts downstream registration
timing — the erratic-odometry failure mode seen previously with KISS-ICP.

## TF

**This node publishes no TF.** `frame_id` is a parameter and the frame must
already exist in a tree someone else owns.

That is deliberate. The Unitree stack's own dynamic chain
(`unilidar_imu_initial → unilidar_imu → unilidar_lidar`) conflicts with static
transforms and EKF/odometry stacks when both claim overlapping frames — see
`SWARM_SLAM_FINDINGS.md` §3 bug #3 for the same class of problem inside this
repo.

## Parameters

| parameter | default | meaning |
|---|---|---|
| `network_interface` | `''` | Ethernet interface facing the robot, e.g. `eth0`. Empty = all interfaces |
| `dds_domain_id` | `0` | CycloneDDS domain the robot publishes on |
| `dds_topic` | `rt/utlidar/cloud` | DDS topic to subscribe to |
| `output_topic` | `raw_lidar` | ROS2 topic. Relative takes the node namespace; absolute (e.g. `/r0/raw_lidar`) overrides it |
| `frame_id` | `utlidar_lidar` | frame stamped on the cloud. No TF is published for it |
| `stamp_source` | `raw` | `raw` \| `raw_header` \| `receive` |
| `queue_len` | `10` | DDS reader queue depth |
| `verify_layout` | `true` | Compare each message against the confirmed layout and warn on drift |
| `stale_timeout` | `5.0` | Warn if no cloud arrives for this many seconds; `0` disables |

QoS is BEST_EFFORT / KEEP_LAST / depth 1, matching the driver's existing point
cloud publisher and what RViz2, nav2 and SLAM consumers default to for lidar.
It is exposed through `QoSOverridingOptions`, so it can be overridden per-node
at launch without editing code.

> `ros2 topic hz` defaults to a **reliable** subscription and silently reports
> nothing against a best-effort publisher. Use
> `ros2 topic hz /go2/raw_lidar --qos-reliability best_effort`.

## Running it

### Standalone

```bash
ros2 run go2_robot_sdk raw_lidar_node --ros-args -p network_interface:=eth0
```

With everything spelled out:

```bash
ros2 run go2_robot_sdk raw_lidar_node --ros-args -p network_interface:=eth0 -p frame_id:=go2/utlidar_lidar -p output_topic:=/go2/raw_lidar -p stamp_source:=raw
```

### Alongside the rest of the stack

`robot.launch.py` carries it, gated off by default:

```bash
ros2 launch go2_robot_sdk robot.launch.py raw_lidar:=true raw_lidar_iface:=eth0
```

Launch arguments:

| argument | default | meaning |
|---|---|---|
| `raw_lidar` | `false` | enable the node |
| `raw_lidar_iface` | `$GO2_LIDAR_IFACE` or `''` | network interface |
| `raw_lidar_domain` | `0` | DDS domain id |
| `raw_lidar_topic` | `raw_lidar` | resolves to `/go2/raw_lidar` under this launch file's `PushRosNamespace('go2')` |
| `raw_lidar_frame` | `''` | empty derives `<tf_prefix>/utlidar_lidar` |
| `raw_lidar_stamp` | `raw` | stamp source |

### Feeding a Swarm-SLAM (cslam) robot namespace

cslam expects each robot's sensor data under `/r{robot_id}/`. Pass an absolute
topic, which bypasses the `/go2` namespace push:

```bash
ros2 launch go2_robot_sdk robot.launch.py raw_lidar:=true raw_lidar_iface:=eth0 raw_lidar_topic:=/r1/raw_lidar
```

or standalone:

```bash
ros2 run go2_robot_sdk raw_lidar_node --ros-args -p network_interface:=eth0 -p output_topic:=/r1/raw_lidar -p frame_id:=go2/utlidar_lidar
```

Robot id mapping used in `SWARM_SLAM_FINDINGS.md` is `r0` = rover (`ganon`),
`r1` = Go2.

## Requirements

`unitree_sdk2py` is not in rosdep. Install it from source:

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git && cd unitree_sdk2_python && pip install -e .
```

It is imported **lazily**, inside the subscriber's `start()` rather than at
module import — see the troubleshooting note below for why.

## Troubleshooting

**`free(): invalid pointer`, or the process aborts on startup.** This is almost
never a logic bug. A machine running this stack typically carries several
CycloneDDS builds — ROS Humble's apt `libddsc`, `unitree_sdk2`'s thirdparty
copy, and a from-source build for the Python bindings — and linking the wrong
one aborts the process. Check which is actually loaded before debugging
anything else:

```bash
python3 -c "import cyclonedds._clayer as c; print(c.__file__)"
```

```bash
ldd $(python3 -c "import cyclonedds._clayer as c; print(c.__file__)") | grep ddsc
```

The lazy import keeps this failure confined to `raw_lidar_node` instead of
taking down `go2_driver_node` with it.

**No clouds arrive.** The node warns every `stale_timeout` seconds. Confirm the
robot is reachable over the cabled link and that the topic is actually present:

```bash
ros2 run go2_robot_sdk raw_lidar_node --ros-args -p network_interface:=eth0 -p stale_timeout:=2.0
```

If `rt/utlidar/cloud` does not exist in the robot's DDS presence at all, this is
the firmware gating described in `SWARM_SLAM_FINDINGS.md` §2, not a
configuration error.

**`unitree_sdk2py imported, but its sensor_msgs PointCloud2_ IDL type was not
found`.** The IDL package path moved between `unitree_sdk2py` releases. The node
tries both known spellings and reports each failure; if neither works, check the
installed version's `unitree_sdk2py/idl/sensor_msgs/` tree.

**Attribute vs method access.** The Python IDL types expose plain attributes —
`msg.width`, `msg.header.stamp.sec` — unlike the C++ `unitree_sdk2` accessors
(`msg.width()`). Calling them raises `TypeError: 'int' object is not callable`.

## Files

| path | role |
|---|---|
| `go2_robot_sdk/infrastructure/dds/utlidar_cloud_subscriber.py` | ROS-free DDS adapter: channel factory, subscriber, message-shape helpers, confirmed layout constant |
| `go2_robot_sdk/presentation/raw_lidar_node.py` | the `rclpy` node |
| `launch/robot.launch.py` | `create_raw_lidar_nodes()`, gated on `raw_lidar:=true` |
