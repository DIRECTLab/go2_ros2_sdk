# External Unitree L2 (`unitree_lidar_ros2`)

Drives a Unitree L2 **bolted onto the Go2** — talking to the sensor directly
over serial or UDP through `unilidar_sdk2` — and publishes it as a standard
`sensor_msgs/PointCloud2`.

Ported from the rover stack (`wheeltec_dev`). Same driver package, same
executable, same topic names, same parameter contract — so the cslam bridge
configuration means the same thing on both robots.

With the L2 mounted upright on top of *both* robots, the two sensors already
observe the same part of the world, so **masking is off by default here** — see
[Masking the external feed](#masking-the-external-feed).

This is **additive**. It is a second, physically separate sensor from the Go2's
factory-mounted utlidar that [`raw_lidar_node`](RAW_LIDAR.md) reads over
CycloneDDS; neither touches the other's topics or frames. Which one runs is
`lidar_source`:

```bash
ros2 launch go2_robot_sdk robot.launch.py lidar_source:=external
```

See [Choosing which lidar runs](#choosing-which-lidar-runs).

## Why this exists

[`docs/SWARM_SLAM_FINDINGS.md`](SWARM_SLAM_FINDINGS.md) §1 found three
properties blocking cross-robot loop closure with the rover. `raw_lidar_node`
removed two of them. The third it explicitly could not:

| §1 blocker | `raw_lidar_node` | external L2 |
|---|---|---|
| pre-quantised to a 5 cm lattice | gone | n/a — never quantised |
| short reach (~4.5 m) | gone | gone (30 m spec) |
| inverted mount (`rpy="0 2.8782 0"`) | **remains** — factory-fixed | **gone** |

The factory lidar sits at roughly 165°, so it sees mostly floor. A flat floor is
translation-degenerate: ScanContext and FPFH cannot register it no matter how
the band is tuned, which is why [`FOV_MASK.md`](FOV_MASK.md) is careful to say
that masking lets you *select* a shared band but cannot conjure walls into one.

An externally mounted, upright L2 is the fix for the mount itself. It is also
the *same sensor model* the rover carries, mounted the same way, which means the
two feeds now share attitude, sampling pattern, reach and noise characteristics.

That is a better fix than masking, and it makes masking largely redundant: the
mask could only ever *select* a shared subset of what two mismatched sensors
saw, whereas identical sensors at identical attitudes have nothing to reconcile.
Hence `unilidar_fov_mask` defaulting off.

## Choosing which lidar runs

The robot has one NIC and one head connector, so in practice the two lidars
compete for the same hardware: bolting on an external L2 generally means the
internal utlidar is unplugged or unreachable. `lidar_source` names that choice
as one argument rather than two booleans you have to remember to keep
consistent.

| `lidar_source` | internal utlidar | external L2 |
|---|---|---|
| `internal` *(default)* | on | off |
| `external` | off | on |
| `both` | on | on |
| `none` | off | off |

**Each feed's `fov_mask` instance follows its feed.** Selecting `external` does
not merely stop the internal driver — it also stops the mask node that was
subscribing to it. That matters because a mask node with no publisher on
`cloud_in` does not fail; it sits there reporting 0 clouds processed every 5
seconds, which reads like a fault in the mask rather than an absent sensor.
(The external feed's own mask is off by default regardless — see
[Masking the external feed](#masking-the-external-feed).)

`both` is worth keeping for one job: it is the configuration in which the two
feeds can be compared side by side, which is what
[`FOV_MASK.md`](FOV_MASK.md)'s tuning loop asks for. It needs both links up at
once — see the interface note under [UDP over ethernet](#udp-over-ethernet).

`raw_lidar` and `unilidar` still exist and now override `lidar_source` per
feed, so the older invocation keeps working:

```bash
ros2 launch go2_robot_sdk robot.launch.py raw_lidar:=false unilidar:=true
```

They default to `''`, meaning "follow `lidar_source`", which is deliberately
distinct from `false`. Pass either explicitly to deviate from the source's
pairing — `lidar_source:=external raw_lidar:=true` is `both` spelled the long
way.

The resolved selection is printed in the launch banner, so a run that came up
with the wrong sensor is visible without reading the node list:

```
   Lidar: external L2 (lidar_source:=external)
```

## Topics and frames

With `tf_prefix:=go2` (the default) and the stack's `/go2` namespace:

| | |
|---|---|
| cloud | `/go2/unilidar/cloud` |
| masked cloud | `/go2/unilidar/cloud_processed` — only with `unilidar_fov_mask:=true` |
| IMU | `/go2/unilidar/imu` (silent in `work_mode: 4`) |
| cloud frame | `go2/unilidar_lidar` |
| mount transform | `go2/base_link` → `go2/unilidar_lidar` |

The rover publishes `/ganon/unilidar/cloud` in `ganon/unilidar_lidar`. Same
shape, different prefix — which is the point. Note the rover's launch file still
defaults its mask **on**, so it also publishes `/ganon/unilidar/cloud_processed`;
whichever topic you feed cslam, feed the same one on both robots.

### The driver publishes no TF, and that is load-bearing

In other work modes the vendor driver broadcasts
`unilidar_imu_initial → unilidar_imu → unilidar_lidar` from its IMU callback,
using the sensor's own orientation estimate. The launch file publishes a static
`base_link → unilidar_lidar` for the mount. **Both claim `unilidar_lidar` as a
child**, which is competing TF authority — the failure this stack has been bitten
by repeatedly, and trap #3 in the rover's own notes.

`work_mode: 4` suppresses the IMU data packets and with them that broadcast, so
the mount transform is the sole authority. It is set in `config/unilidar.yaml`
and is not an incidental value: change it and you get two publishers racing over
one frame, which shows up as a cloud that periodically jumps.

## Wiring: serial or UDP

The SDK supports both, selected by `initialize_type` (`1` = serial, `2` = UDP).
The launch file wraps that in `unilidar_conn:=serial|udp`, because a wrong
`initialize_type` makes the vendor code print one line and call `exit(0)` — which
reads as the node starting and vanishing rather than as a config error.

**The yaml ships `2` (UDP), matching the rover.** Pick whichever matches how the
sensor is actually cabled; there is no way for the software to detect it.

### UDP over ethernet

The sensor ships on `192.168.1.62`, and the host must hold an address on that
subnet — `local_ip`, `192.168.1.2` by default — or the socket bind fails.

`192.168.1.x` is chosen to stay clear of `192.168.123.x`, which is the Go2's own
subnet and what `raw_lidar_node`'s CycloneDDS link uses. Do not renumber the
lidar into `192.168.123.x`; the two links are independent and should stay that
way.

```bash
sudo ip addr add 192.168.1.2/24 dev <iface>
sudo ip link set <iface> up
ping 192.168.1.62
```

The SDK's own tools set the sensor's address and transport, and are built from
the clone:

```bash
cd ~/unilidar_sdk2/unitree_lidar_sdk
./bin/set_to_udp_mode
./bin/set_ip_address
```

> On the Go2 the onboard NIC is usually already committed to the robot's own
> `192.168.123.x` link. Check what `raw_lidar_iface` is using
> (`GO2_LIDAR_IFACE`, `enP8p1s0` on the Jetson) before assigning the lidar an
> interface — running both feeds needs two, or a switch and a second address on
> one.

### Serial over USB

Simpler on a legged robot, and the usual choice when the NIC is taken:

```bash
ros2 launch go2_robot_sdk robot.launch.py lidar_source:=external unilidar_conn:=serial
```

`/dev/ttyACM0` is the default. **That name is not stable** if anything else on
the robot enumerates as `ttyACM` — the rover already has a `ttyACM` udev rule
for its controller. Pin it:

```bash
udevadm info -a -n /dev/ttyACM0 | grep -m2 'idVendor\|idProduct'
```

then, with the values that prints:

```bash
echo 'KERNEL=="ttyACM*", ATTRS{idVendor}=="XXXX", ATTRS{idProduct}=="YYYY", MODE:="0666", GROUP:="dialout", SYMLINK+="unilidar"' \
  | sudo tee /etc/udev/rules.d/99-unilidar.rules
sudo udevadm control --reload && sudo udevadm trigger
```

and launch with `unilidar_serial_port:=/dev/unilidar`.

## The mount transform — measure it

`unilidar_mount_xyz` and `unilidar_mount_ypr` place the sensor on the body. The
defaults (`0.0 0.0 0.15` / `0.0 0.0 0.0`) are a **placeholder for a plate on the
robot's back, not a calibration.**

This matters more than it looks. Every mask primitive except
`sensor_blank_radius` is evaluated in a gravity-aligned frame reached *through*
this transform, so an unmeasured mount does not merely offset the cloud — it
silently invalidates the shared z band, the range band and any cross-robot
comparison built on them. The rover's equivalent numbers
(`-0.008 0.015 0.457` / `-0.83 -0.905 -2.45`) were measured for its mast and
mean nothing here.

`unilidar_mount_ypr` is **yaw pitch roll**, in that order, in radians — the order
`static_transform_publisher` takes positionally. All zeros means upright and
facing forward, which is the whole reason for mounting externally.

```bash
ros2 launch go2_robot_sdk robot.launch.py lidar_source:=external \
  unilidar_mount_xyz:="0.05 0.0 0.18" unilidar_mount_ypr:="0.0 0.0 0.0"
```

Both are parsed and their element count checked before they reach
`static_transform_publisher`, because passing the wrong number of positional
arguments to that node is a mistake this repo's launch files have made before.

### `base_link`, not `base_footprint`

The mount is parented to `<tf_prefix>/base_link`, where the sensor is physically
bolted. On this robot the two frames are coincident anyway — `go2.urdf` leaves
`base_footprint_joint` at zero deliberately, so **`base_footprint` is not
ground-projected here despite the name** (the URDF comment explains why a
correct offset cannot be expressed as a fixed joint, and that the ground
reference is applied at the consumer via `cslam_robot_bridges`' `z_offset`).

The consequence for cross-robot work: a shared `z_min` in
`<prefix>/base_footprint` does **not** mean the same height above the floor on
both robots, because the rover's `base_footprint` really is on the ground and
the Go2's is at body height. Expect to carry a per-robot z offset, exactly as
[`FOV_MASK.md`](FOV_MASK.md) warns for odom frames.

## Masking the external feed

**`unilidar_fov_mask` defaults to `false`, so the external feed is not masked.**
Consumers read `/go2/unilidar/cloud` directly.

That is the right default once the L2 is mounted upright on top of *both*
robots. The mask exists to reconcile two differently-mounted sensors: the
factory utlidar is inverted ~165° and sees mostly floor, so reducing both robots
to one shared band is the only way their clouds describe the same content. Same
sensor at the same attitude on both robots removes that problem at the hardware
level — there is no band left to select, and masking would only discard
geometry. The shared yaml's `sensor_blank_radius: 0.68`, sized for the rover's
mast, would discard a lot of it.

### What turning it off also turns off

The node does two unrelated jobs, and only one of them is masking. It is also
where **accumulation and deskewing** live, and the L2 is a *non-repetitive*
scanner: each sweep covers a fraction of the field of view and successive sweeps
deliberately sample different directions, so one sweep is a sparse slice of the
room rather than a picture of it. [`FOV_MASK.md`](FOV_MASK.md) treats
accumulating over a short window as normal operation for this sensor class, not
a density optimisation.

With the node off, consumers get per-sweep clouds. That is correct for
scan-to-map odometry (KISS-ICP and similar), which is genuinely per-scan and
deskews using the per-point `time` offsets. If cslam turns out to want denser
keyframes, the way to get accumulation *without* masking is to turn the node on
and leave the band unbounded — every mask primitive defaults to unbounded, so a
`fov_mask` with only `decay_time` set is a passthrough that accumulates:

```bash
ros2 launch go2_robot_sdk robot.launch.py lidar_source:=external \
  unilidar_fov_mask:=true fov_mask_params:=/path/to/passthrough.yaml \
  fov_mask_decay:=0.5
```

`fov_mask_params` is shared between the two instances, but with
`lidar_source:=external` only one of them exists, so there is no ambiguity.
Leaving it at the default would point this node at the shared `fov_mask.yaml`
and bring the rover's band back with it.

### Turning masking back on

```bash
ros2 launch go2_robot_sdk robot.launch.py lidar_source:=external unilidar_fov_mask:=true
```

It is a **separate instance** from the one over the internal feed rather than a
switch on it, so `lidar_source:=both unilidar_fov_mask:=true` masks both clouds
at once — which is how you compare what the inverted factory mount sees against
what the upright external one does.

It reads the **same** `config/fov_mask.yaml` and honours the **same**
`fov_mask_*` arguments as the raw instance. That is deliberate: the band is what
the two robots have to hold in common, so it cannot be specialised per feed
without breaking the comparison it exists to make.

The one exception is `sensor_blank_radius`, which describes mount hardware
rather than the shared band — the external mount is not the factory head mount:

```bash
ros2 launch go2_robot_sdk robot.launch.py lidar_source:=external \
  unilidar_fov_mask:=true unilidar_fov_mask_blank_radius:=0.12
```

Size it from the node's own stats line: with it off, the `sens_r` 5th percentile
tells you where the mount returns sit. Raise it until that percentile clears the
mount, then stop.

> **Whichever you choose, choose the same thing on both robots.** If the Go2
> publishes per-sweep unmasked clouds while the rover publishes accumulated
> masked ones, the two feeds are no longer comparable — which was the entire
> point of the shared yaml. The rover's `swarm_slam.launch.py` has `fov_mask`
> defaulting to `true`; turn it off there too, or turn both on.

## Parameter contract

`config/unilidar.yaml`, keyed `/**` — a key of `unitree_lidar_ros2_node:` would
never match, since `PushRosNamespace` makes the node
`/go2/unitree_lidar_ros2_node`. That bare-key/FQN mismatch is bugs #6 and #8 in
`SWARM_SLAM_FINDINGS.md`.

| parameter | type | default | meaning |
|---|---|---|---|
| `initialize_type` | int | `2` | `1` = serial, `2` = UDP. Anything else calls `exit(0)` |
| `work_mode` | int | `4` | suppresses the IMU packets, and with them the driver's competing TF broadcast |
| `use_system_timestamp` | bool | `true` | stamp from this host's clock, not the sensor's |
| `serial_port` | string | `/dev/ttyACM0` | serial device |
| `baudrate` | int | `4000000` | serial baud |
| `lidar_ip` / `lidar_port` | string / int | `192.168.1.62` / `6101` | the **sensor's** address |
| `local_ip` / `local_port` | string / int | `192.168.1.2` / `6201` | **this host's** address on the lidar subnet |
| `cloud_scan_num` | int | `18` | packets grouped into one published sweep |
| `range_min` / `range_max` | double | `0.0` / `30.0` | clip applied in the SDK, before ROS |
| `cloud_frame` / `imu_frame` | string | — | **always overridden by the launch file** with `<tf_prefix>/…` |
| `cloud_topic` / `imu_topic` | string | — | **always overridden by the launch file** |

`range_min` and `range_max` are **doubles**: write `30.0`, never `30`. YAML reads
a bare `30` as an integer, `rclcpp` refuses to coerce it, and the node dies at
startup. Note also that the driver's own `declare_parameter` defaults have
`lidar_ip` and `local_ip` **backwards** relative to how they are used; the yaml
has them the right way round.

Leave the far clip wide here and shape the usable band in `fov_mask` instead —
clipping in the driver would be invisible to the yaml that is shared with the
other robot.

## Launch arguments

```bash
ros2 launch go2_robot_sdk robot.launch.py lidar_source:=external
```

| argument | default | meaning |
|---|---|---|
| `lidar_source` | `internal` | which lidar runs: `internal`, `external`, `both`, `none` |
| `unilidar` | `''` | override `lidar_source` for this feed alone; `''` follows it |
| `unilidar_params` | `config/unilidar.yaml` | path to the driver yaml |
| `unilidar_conn` | `''` | `serial` or `udp`; empty keeps the yaml's `initialize_type` |
| `unilidar_serial_port` | `''` | serial device; empty keeps the yaml's |
| `unilidar_ip` | `''` | the sensor's address; empty keeps the yaml's |
| `unilidar_local_ip` | `''` | this host's address on the lidar subnet |
| `unilidar_topic` | `unilidar/cloud` | cloud topic; the IMU topic follows it |
| `unilidar_frame` | `''` | cloud `frame_id`; empty derives `<tf_prefix>/unilidar_lidar` |
| `unilidar_mount_xyz` | `0.0 0.0 0.15` | sensor origin in `<tf_prefix>/base_link`. **Measure this** |
| `unilidar_mount_ypr` | `0.0 0.0 0.0` | sensor orientation, yaw pitch roll in radians |
| `unilidar_fov_mask` | `false` | mask this feed. Off because both robots now mount the L2 upright, so there is no band to select |
| `unilidar_fov_mask_blank_radius` | `''` | per-mount `sensor_blank_radius` override, when masking is on |

`lidar_source` defaults to `internal`, so the external feed is off unless
asked for. The vendor driver responds to a missing sensor by calling `exit(0)`
or blocking on a socket bind rather than logging and carrying on, so it is not
a feed to leave enabled speculatively on a robot that may not have the hardware
bolted on. Change the `lidar_source` default in `robot.launch.py` once the L2 is
permanent on this robot.

### Standalone

For bringing the sensor up on its own — checking the cable, the IP pair or the
mount — without starting the whole stack:

```bash
ros2 launch unitree_lidar_ros2 launch.py
ros2 launch unitree_lidar_ros2 launch.py initialize_type:=1 serial_port:=/dev/unilidar
```

That package's own launch file predates `lidar_source` and takes the SDK's bare
`initialize_type` — it knows nothing about the robot stack.

This publishes **no** transform, so the cloud has no place in the robot's TF
tree and `fov_mask` will not find `mask_frame` from it. It is a hardware check,
not a bringup path.

## Building

The vendor SDK is headers plus a prebuilt static library per architecture. It is
not a ROS package, is not on the ament index, and `find_package` cannot see it —
so it has to be cloned separately:

```bash
git clone https://github.com/unitreerobotics/unilidar_sdk2.git ~/unilidar_sdk2
```

The package then finds it automatically. `CMakeLists.txt` searches, in order:
`src/unilidar_sdk2/unitree_lidar_sdk` inside this workspace (vendored copy or
git submodule), the same one directory up, `~/direct/unilidar_sdk2`,
`~/unilidar_sdk2`, and `/opt/unilidar_sdk2`. The rover's copy of this file
hardcodes `$ENV{HOME}/wheeltec_dev/unilidar_sdk2`, which only works on that
robot.

For a clone somewhere else:

```bash
colcon build --packages-select unitree_lidar_ros2 \
  --cmake-args -DUNILIDAR_SDK_DIR=/path/to/unilidar_sdk2/unitree_lidar_sdk
```

or export `UNILIDAR_SDK_DIR`. If neither finds it, the build fails with that
clone command rather than an unfindable-header error.

The library is selected by `CMAKE_SYSTEM_PROCESSOR`, matching the SDK's
`lib/x86_64` and `lib/aarch64` — so the same tree builds on an x86 dev host and
on the robot's Jetson with no changes. Any other architecture fails with a clear
message; the SDK ships those two only.

```bash
cd ~/direct/go2_ros2_sdk
colcon build --symlink-install --packages-select unitree_lidar_ros2 lidar_processor go2_robot_sdk
```

## Feeding Swarm-SLAM

`cslam_robot_bridges`' Go2 bridge already takes the cloud topic as an argument,
defaulting to the factory feed. Point it at the external one instead — with
masking off, that is the driver's own topic:

```bash
ros2 launch cslam_robot_bridges go2_cslam.launch.py pointcloud_topic:=unilidar/cloud
```

**The rover's `wheeltec_lidar.launch.py` defaults to `unilidar/cloud_processed`,
not `unilidar/cloud`.** With masking off on this robot, set the rover's
`lidar_cloud_topic:=unilidar/cloud` to match, or the two robots hand cslam
different kinds of cloud. No bridge code change is needed either way.

Two things to keep straight, both from [`FOV_MASK.md`](FOV_MASK.md):

- **Keep `publish_frame: input`** so the cloud stays in its sensor frame and
  cslam's own `sensor_base_frame_id` handling still applies — and make sure that
  value carries the right prefix, which was bug #11.
- **With accumulation on, the topic is published in `decay_frame`** (odom), not
  the sensor frame, which changes what that handling is reconciling. And each
  cslam keyframe carries one odometry pose, so a cloud spanning several poses
  yields a loop-closure transform that is not a transform between two robot
  poses.

The `z_offset` on the bridge is still needed. Nothing here changes the fact that
the Go2's `base_footprint` is not on the ground.

## Verifying

```bash
ros2 topic hz /go2/unilidar/cloud
ros2 run tf2_tools view_frames
```

With `unilidar_fov_mask:=true` there is a second topic to check, and `fov_mask`
subscribes and publishes BEST_EFFORT — so `ros2 topic hz` reports nothing on it
without the flag:

```bash
ros2 topic hz /go2/unilidar/cloud_processed --qos-reliability best_effort
```

**Sanity check in RViz2:** a non-repetitive scanner produces a dense
rosette/flower accumulation pattern, not a concentric lattice. Displaying
`/go2/unilidar/cloud` next to `/go2/raw_lidar` should show two rosettes at
different attitudes — one upright, one tipped ~165° into the floor. That
contrast is the entire reason this sensor was added.

| symptom | cause |
|---|---|
| no external topics at all | `lidar_source` is still `internal`; check the launch banner's `Lidar:` line |
| a mask node reporting 0 clouds processed | on an older revision, before the mask instances followed their feeds. Now they are skipped instead |
| node starts and immediately exits, one line of output | `initialize_type` is neither 1 nor 2 — the vendor code calls `exit(0)` |
| node starts, no clouds, UDP mode | host does not hold `local_ip`, or the sensor is in serial mode |
| `Permission denied` on `/dev/ttyACM0` | user not in `dialout`, or no udev rule |
| clouds publish but `fov_mask` retains 0% | mount transform missing or `mask_frame` unprefixed — the mask node warns on the TF lookup |
| cloud periodically jumps | `work_mode` other than 4, so the driver's IMU TF broadcast is competing with the static mount transform |
| large steady `TF stale … skew` in the stats | clock-domain mismatch; check `use_system_timestamp` is `true` |
