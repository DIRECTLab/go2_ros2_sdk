# FOV mask (`fov_mask` node)

Drops points from a `PointCloud2` that fall outside a configurable region, so
two robots with differently-mounted lidars can be reduced to a comparable subset
of what they observe.

Lives in `lidar_processor`, not `go2_robot_sdk`, because **it has to run on the
rover too**. `ganon` runs `unitree_lidar_ros2`; a mask built into
`raw_lidar_node` could never touch it. One implementation, run once per robot.

It is a pure filter: it publishes a second topic and leaves the input alone.

## What this is for, and what it cannot do

`SWARM_SLAM_FINDINGS.md` §1 identified three properties blocking cross-robot
loop closure. The raw DDS feed removed two of them (quantisation, reach). The
third — the Go2's inverted 165° factory mount, which makes its view
floor-dominated — remains, and this node does **not** fix it.

What it does is let you *select* what both robots keep, so you can test whether
a shared non-degenerate band exists. Be clear about the failure mode: a flat
floor is translation-degenerate, so if the band you select is still mostly
floor, ScanContext and FPFH fail exactly as before, just on fewer points. The
band worth hunting for is one containing **walls** that both sensors see.

## Design notes

**Layout-agnostic.** Fields are read from each message's own descriptors, never
a hardcoded layout, because the two feeds don't share one — the Go2's raw cloud
is a 32-byte `x/y/z/intensity/ring/time` stride, the rover's comes from a
different driver. Surviving points are copied as **whole rows**, so every field
and every padding byte is preserved exactly; `ring` and `time` reach downstream
deskewing intact.

**The mask is evaluated in a gravity-aligned frame, via TF.** This is not
optional for cross-robot work. In the Go2's own sensor frame "horizontal" is
tilted 165°, so an elevation band there selects a completely different physical
region than the same numbers on the rover.

**No TF is published.** Same reasoning as `raw_lidar_node` — see
[`RAW_LIDAR.md`](RAW_LIDAR.md).

## Choosing a mask primitive

Four independent filters, composable, all **unbounded by default** (the node is
a no-op until you configure it).

| primitive | use it for |
|---|---|
| **z band** | the primary knob. ScanContext bins by x,y and stores `max(z + 2.0)` per cell (§4), so z is what actually has to match. Excludes the floor uniformly at *every* range |
| **range band** | capping the rover (30 m spec) to the Go2's reach so both populate the same ScanContext rings — §4 found both robots reaching only 2–3 of 20 |
| **elevation cone** | emulating the cone one sensor physically *cannot* see beyond, on the other robot |
| **azimuth sector** | restricting both to a forward-facing wedge. Wraps if `azim_min > azim_max` |

**Elevation is not the right tool for excluding the floor.** Floor sits at
constant z but at a range-dependent elevation — at 1 m it might be −20°, at 4 m
−6°. A z band removes it cleanly; an elevation band cannot.

`range_mode` picks what "range" means: `horizontal` (√(x²+y²)) matches
ScanContext's x,y binning and is the default; `euclidean` matches datasheet
slant-range specs.

`elev_origin` picks where elevation is measured from: `sensor` (default) uses
the cloud frame's origin expressed in `mask_frame`, making the band a true field
of view; `mask_frame` measures from that frame's origin instead.

## Tuning loop

Nothing computes the angles for you. The node reports what survived, and you
adjust until both robots' retained distributions overlap:

```
masked 47 clouds | kept 1204/3847 (31.3%) | z 5/50/95: 0.21/0.58/1.74 | range 5/50/95: 0.91/2.44/4.31 | elev 5/50/95: -8.2/4.1/22.6
```

Those are the same 5/50/95 percentiles §4 tabulated by hand, live on both
robots. Run it on each, compare, adjust the shared yaml, repeat.

## Configuration

Both robots load the **same yaml**, `go2_robot_sdk/config/fov_mask.yaml`. That
is the point — one file defines the shared band. Copy it into the rover's stack
rather than maintaining two sets of numbers.

`mask_frame` is the one exception: it must name the physically equivalent frame
on each robot, which differs by prefix (`go2/base_footprint` vs
`ganon/base_footprint`). The launch file passes it as a direct override.

> The yaml is keyed `/**:`, not `fov_mask_node:`. Under `PushRosNamespace` the
> node is `/go2/fov_mask_node`, so a bare key never matches and the file is
> silently ignored — the same mismatch recorded as bugs #6 and #8 in
> `SWARM_SLAM_FINDINGS.md`.

| parameter | default | meaning |
|---|---|---|
| `mask_frame` | `base_footprint` | gravity-aligned frame the mask is evaluated in |
| `publish_frame` | `input` | `input` keeps points in their original frame (pure filter); `mask` rewrites x/y/z into `mask_frame` to co-register both feeds |
| `z_min` / `z_max` | unbounded | z band in `mask_frame` |
| `min_range` / `max_range` | `0` / unbounded | radial band |
| `range_mode` | `horizontal` | `horizontal` \| `euclidean` |
| `elev_min_deg` / `elev_max_deg` | `-90` / `90` | elevation cone |
| `elev_origin` | `sensor` | `sensor` \| `mask_frame` |
| `azim_min_deg` / `azim_max_deg` | `-180` / `180` | azimuth sector, wraps if min > max |
| `tf_timeout` | `0.1` | seconds before falling back to the latest transform |
| `stats_period` | `5.0` | seconds between retention reports; `0` disables |

QoS is BEST_EFFORT / KEEP_LAST / depth 1 on both sides — a reliable
subscription would silently receive nothing from either robot's lidar driver.

## Running it

Topics are `cloud_in` and `cloud_masked`, remapped at launch — the same idiom
this stack uses for `pointcloud_to_laserscan` (`cloud_in` → `scan`).

### Alongside the Go2 stack

```bash
ros2 launch go2_robot_sdk robot.launch.py raw_lidar:=true raw_lidar_iface:=eth0 fov_mask:=true
```

Publishes `/go2/raw_lidar_masked` alongside the untouched `/go2/raw_lidar`.

| argument | default | meaning |
|---|---|---|
| `fov_mask` | `false` | enable the node |
| `fov_mask_params` | `config/fov_mask.yaml` | path to the shared mask yaml |
| `fov_mask_frame` | `''` | empty derives `<tf_prefix>/base_footprint` |

### Standalone, on either robot

```bash
ros2 run lidar_processor fov_mask --ros-args --params-file /path/to/fov_mask.yaml -p mask_frame:=ganon/base_footprint -r cloud_in:=/ganon/unilidar/cloud -r cloud_masked:=/ganon/unilidar/cloud_masked
```

### Feeding cslam

Point the bridge's `pointcloud_topic` at the masked topic. Keep
`publish_frame: input` so the cloud stays in its sensor frame and cslam's own
`sensor_base_frame_id` handling still applies — and make sure that value carries
the right prefix, which was bug #11.

## Verifying

```bash
ros2 topic hz /go2/raw_lidar_masked --qos-reliability best_effort
```

If retention reads 0%, the usual causes are a `mask_frame` that doesn't exist
(the node warns on the TF lookup), or a band set in the wrong units — angles are
**degrees** in the params, radians only internally.
