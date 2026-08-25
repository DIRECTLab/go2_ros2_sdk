# Raw LiDAR over CycloneDDS (`raw_lidar_node`)

Publishes the robot's onboard L2 lidar feed — taken straight off the
`rt/utlidar/cloud` DDS topic over the Ethernet link — as a standard
`sensor_msgs/PointCloud2`.

This is **additive**. Nothing about the driver's existing WebRTC point cloud
(`/go2/point_cloud2`) changes. The node is **on by default** in the launch file;
pass `raw_lidar:=false` to leave it out.

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
Displaying `/go2/raw_lidar` and `/go2/point_cloud2` together makes the contrast
obvious — the WebRTC feed renders as a visibly regular grid, the DDS feed as a
rosette.

**Effect on the `SWARM_SLAM_FINDINGS.md` §1 blockers.** Two of the three
properties that made cross-robot loop closure unreachable were artefacts of the
WebRTC transport, not of the hardware, and this feed removes them:

| §1 blocker | on this feed |
|---|---|
| pre-quantised to a 5 cm lattice | gone |
| short reach (~4.5 m) | gone |
| inverted mount (`rpy="0 2.8782 0"`) | **remains** — factory-fixed |

The mount was §1's binding constraint, so this does not by itself make cslam
close inter-robot loops. It does mean the registration-parameter sweep §7
advised against is worth one revisit, since FPFH now has real local sampling to
histogram rather than a lattice.

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

The default `frame_id` is **`radar`** (`<tf_prefix>/radar` from the launch file),
which is the lidar link `go2.urdf` already defines:

```
radar_joint: base_link → radar, xyz="0.28945 0 -0.046825" rpy="0 2.8782 0"
```

`robot_state_publisher` publishes that edge already, so the cloud resolves with
no extra TF authority anywhere. The `rpy` is the inverted factory mount
(165°) measured in `SWARM_SLAM_FINDINGS.md` §1.

If RViz reports `Message Filter dropping message: frame '<something>' ... queue
is full`, the named frame is not in the tree. Check what is:

```bash
ros2 run tf2_tools view_frames
```

### Is the raw cloud actually in the sensor frame? — verified yes

This needed checking rather than assuming, because the driver's *WebRTC* cloud
is published in `odom`, not a sensor frame (`ros2_publisher.py` uses
`config.frame("odom")`). Had `rt/utlidar/cloud` been likewise pre-transformed,
tagging it `radar` would have applied the 165° mount rotation a second time.

**Verified by overlay (2026-08-17):** displaying `/go2/raw_lidar` and
`/go2/point_cloud2` together with fixed frame `go2/odom` puts both on the same
floor plane, at the same scale, with the same vertical structures. A
double-applied 165° would tip the raw cloud roughly 30° out of the WebRTC
floor and is not present. `radar` is the right frame; the cloud arrives in the
sensor's own coordinates.

To re-confirm numerically without needing TF at all, fit the dominant plane in
the cloud's own coordinates and read its tilt from the Z axis:

- **~15°** — sensor frame, `radar` correct. (164.9° mount ⇒ floor normal
  `[-0.26, 0, -0.97]` in radar coords ⇒ 15.1° from Z.)
- **~1°** — already gravity/body-aligned, which would mean `base_link` instead.

Do not add a static transform to compensate for a frame mismatch; change
`raw_lidar_frame` instead. A second authority on that edge is the conflict this
node exists to avoid.

**The frame's X axis is not the optical axis.** `radar` X maps to
`[-0.97, 0, -0.26]` in `base_link` — mostly backward and slightly down — so the
red X arrow in RViz will not line up with the visible centre of the scan
pattern. That is expected for an inverted mount and is not evidence of a
misconfigured frame.

## Parameters

| parameter | default | meaning |
|---|---|---|
| `network_interface` | `enP8p1s0` | Ethernet interface facing the robot; the Jetson's onboard NIC. **Set it explicitly on any other host.** Empty does *not* mean "all interfaces" — it lets CycloneDDS pick one by its own heuristic, which chooses wrong on any machine that also has Wi-Fi |
| `dds_domain_id` | `0` | CycloneDDS domain the robot publishes on |
| `dds_topic` | `rt/utlidar/cloud` | DDS topic to subscribe to |
| `output_topic` | `raw_lidar` | ROS2 topic. Relative takes the node namespace; absolute (e.g. `/r0/raw_lidar`) overrides it |
| `frame_id` | `radar` | frame stamped on the cloud — the lidar link `go2.urdf` already defines. No TF is published for it |
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
ros2 run go2_robot_sdk raw_lidar_node --ros-args -p network_interface:=eth0 -p frame_id:=go2/radar -p output_topic:=/go2/raw_lidar -p stamp_source:=raw
```

### Alongside the rest of the stack

`robot.launch.py` runs it by default, so the plain bringup already publishes it:

```bash
ros2 launch go2_robot_sdk robot.launch.py
```

On a host whose interface is not the Jetson's, or to leave it out entirely:

```bash
ros2 launch go2_robot_sdk robot.launch.py raw_lidar_iface:=eth0
```

Launch arguments:

| argument | default | meaning |
|---|---|---|
| `raw_lidar` | `true` | run the node; `false` leaves it out |
| `raw_lidar_iface` | `$GO2_LIDAR_IFACE` or `enP8p1s0` | network interface, defaulting to the Jetson's |
| `raw_lidar_domain` | `0` | DDS domain id |
| `raw_lidar_topic` | `raw_lidar` | resolves to `/go2/raw_lidar` under this launch file's `PushRosNamespace('go2')` |
| `raw_lidar_frame` | `''` | empty derives `<tf_prefix>/radar` |
| `raw_lidar_stamp` | `raw` | stamp source |

### Feeding a Swarm-SLAM (cslam) robot namespace

cslam expects each robot's sensor data under `/r{robot_id}/`. Pass an absolute
topic, which bypasses the `/go2` namespace push:

```bash
ros2 launch go2_robot_sdk robot.launch.py raw_lidar_topic:=/r1/raw_lidar
```

or standalone:

```bash
ros2 run go2_robot_sdk raw_lidar_node --ros-args -p network_interface:=eth0 -p output_topic:=/r1/raw_lidar -p frame_id:=go2/radar
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

**No clouds arrive.** The node warns every `stale_timeout` seconds. Work through
these in order — the interface is by far the most common cause.

1. **Find the interface on the robot's cabled subnet.** The WebRTC/Wi-Fi address
   (`ROBOT_IP`, e.g. `192.168.0.x`) is *not* it; DDS lidar traffic is on the
   cabled `192.168.123.0/24` link.

   ```bash
   ip -br addr
   ```

2. **Confirm the link is up**, using whichever `192.168.123.x` host answers
   (`.161` is the robot's usual DDS address, `.18` its onboard computer):

   ```bash
   ping -c3 192.168.123.161
   ```

3. **Pass the interface explicitly**:

   ```bash
   ros2 run go2_robot_sdk raw_lidar_node --ros-args -p network_interface:=eth0 -p stale_timeout:=2.0
   ```

4. **If it is still silent, check the topic exists at all** before blaming
   configuration. Bind ROS's own CycloneDDS to the same interface and enumerate:

   ```bash
   RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eth0"/></Interfaces></General></Domain></CycloneDDS>' ros2 topic list
   ```

   If `/utlidar/cloud` is absent and all you see is the sport/vui command API,
   this is the firmware gating described in `SWARM_SLAM_FINDINGS.md` §2, not a
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
| `launch/robot.launch.py` | `create_raw_lidar_nodes()`, on by default, disable with `raw_lidar:=false` |
