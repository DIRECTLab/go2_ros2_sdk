# FOV mask (`fov_mask` node)

Drops points from a `PointCloud2` that fall outside a configurable region, so
two robots with differently-mounted lidars can be reduced to a comparable subset
of what they observe.

Lives in `lidar_processor`, not `go2_robot_sdk`, because **it has to run on the
rover too**. `ganon` runs `unitree_lidar_ros2`; a mask built into
`raw_lidar_node` could never touch it. One implementation, run once per robot.

It is a pure filter: it publishes its own topics and leaves the input alone. It
can also accumulate points over time onto a third topic — see
[Accumulation](#accumulation-decay_time) below.

| topic | contents |
|---|---|
| `cloud_in` | input, remapped at launch |
| `cloud_processed` | the result: one masked cloud per input scan, or the accumulated history when `decay_time` is set |

One output topic, not two. `decay_time` changes what it carries rather than
adding a second stream — consumers subscribe to one name and don't have to know
how the node is configured.

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

Five independent filters, composable, all **unbounded or off by default** (the
node is a no-op until you configure it).

| primitive | use it for |
|---|---|
| **z band** | the primary knob. ScanContext bins by x,y and stores `max(z + 2.0)` per cell (§4), so z is what actually has to match. Excludes the floor uniformly at *every* range |
| **range band** | capping the rover (30 m spec) to the Go2's reach so both populate the same ScanContext rings — §4 found both robots reaching only 2–3 of 20 |
| **elevation cone** | emulating the cone one sensor physically *cannot* see beyond, on the other robot |
| **azimuth sector** | restricting both to a forward-facing wedge. Wraps if `azim_min > azim_max` |
| **sensor blank radius** | deleting returns off the lidar's own mount. Unlike the four above this is not part of the shared band — it describes hardware, so the value is legitimately per-robot |

**Elevation is not the right tool for excluding the floor.** Floor sits at
constant z but at a range-dependent elevation — at 1 m it might be −20°, at 4 m
−6°. A z band removes it cleanly; an elevation band cannot.

`range_mode` picks what "range" means: `horizontal` (√(x²+y²)) matches
ScanContext's x,y binning and is the default; `euclidean` matches datasheet
slant-range specs.

### `sensor_blank_radius` vs `min_range`

They look similar and are not interchangeable. `min_range` is part of the
*shared band* — it obeys `range_mode` and is measured from `mask_origin`.
`sensor_blank_radius` is *physical self-filtering* — always a sphere, always
measured from the cloud frame's own origin.

|  | `min_range` | `sensor_blank_radius` |
|---|---|---|
| shape | follows `range_mode` (`horizontal` by default → a cylinder) | always a sphere |
| measured from | `mask_origin` | the sensor, always |
| purpose | the shared cross-robot band | deleting the lidar's own mount |

The naive argument for why `min_range` can't do this job is wrong, and it's
worth being precise: horizontal distance is never greater than euclidean, so a
`min_range` of r *does* catch every point an r-sphere catches. It is not too
weak — it is too **strong**. At `range_mode: horizontal` it carves out an
infinite vertical **cylinder** through the sensor, deleting the ceiling directly
overhead and the floor directly underneath at every height.
`sensor_blank_radius` removes a bounded **ball** and nothing past it.

Measured at r = 0.2 m on the rover's mount geometry: both drop the same four
mount returns, but a ceiling return 3 m above the sensor survives the sphere and
is deleted by `min_range`. Same radius, different shape, and the difference is
real data.

Switching `range_mode` to `euclidean` to fix the shape isn't an option either —
it's shared with `max_range`, so you'd stop matching ScanContext's x,y binning.
And with `mask_origin` set to anything other than `sensor`, `min_range`'s
exclusion zone detaches from the sensor entirely, while the mount, bolted to it,
does not.

Because it describes hardware rather than the shared band, the *value* is
legitimately per-robot — the Go2's factory head mount and the rover's L2 mast
are different geometry. Override it with `-p sensor_blank_radius:=0.12` rather
than forking the shared yaml.

## Accumulation (`decay_time`)

Modelled on RViz2's **Decay Time**: points persist for `decay_time` seconds
instead of each scan replacing the last. `0` disables it, matching RViz where 0
means "show only the newest cloud".

Setting it **changes what `cloud_processed` carries**: per-scan masked clouds
when 0, the accumulated history when set. The two never interleave — exactly one
of them is published per input cloud, so the same points can't appear twice.

> **Leave it at 0 when feeding scan-to-map odometry.** KISS-ICP and similar are
> sequential and deskew using the per-point `time` offsets, which are relative
> to their own sweep — an accumulated cloud breaks both assumptions.
>
> **cslam is more tolerant than that**, and it is worth being precise since the
> two often get lumped together. ScanContext bins by x,y and stores
> `max(z + 2.0)` per cell (§4), so sweep boundaries are irrelevant to the
> descriptor, and FPFH/TEASER is plain cloud registration. cslam was in fact
> already consuming the Go2's *accumulated* WebRTC voxel map rather than
> per-sweep data. Two things do still bite:
>
> - **Pose association.** Each keyframe carries one odometry pose. A cloud
>   spanning several poses makes the transform TEASER recovers not a transform
>   between two robot poses.
> - **Frame.** With accumulation on, the topic is published in `decay_frame`
>   (odom), not the sensor frame, which changes what cslam's
>   `sensor_base_frame_id` handling is reconciling — bug #11 territory.

### Why this is not optional on these sensors

Both robots carry Unitree L2s, which are **non-repetitive** scanners: each sweep
covers only a fraction of the field of view, and successive sweeps deliberately
sample *different* directions. That is the rosette pattern you see in RViz
rather than a fixed lattice of rings.

The consequence is that a single sweep is not a picture of the room — it is a
sparse slice of it. Accumulating over `decay_time` is how a complete picture is
formed, so this is part of normal operation for this sensor class, not a density
optimisation layered on top.

Because both robots run the same sensor, **`decay_time` belongs in the shared
yaml and should be identical on both**. Same coverage-vs-time behaviour, same
fraction of the FOV filled, so the two clouds stay directly comparable — which
is the entire point of the shared file. Contrast `sensor_blank_radius`, which
describes mount hardware and is legitimately per-robot.

### Choosing a value

The right value is the sensor's FOV coverage time, and the node already reports
what you need to find it. Sweep `decay_time` upward and watch the stats line:

```
... | decay 41300 pts over 12 clouds
```

Raise it until the accumulated point count stops growing meaningfully and the
scene stops filling in visually — that is the coverage time. Past that you are
only adding age, and with it pose smear.

**While moving, coverage and smear pull against each other.** Accumulating for
`T` seconds at speed `v` smears the cloud by `v·T`; keep that under
`voxel_size` or you blur the very cells you were filling. At 0.5 m/s with
cslam's `voxel_size: 0.3` that is `T < 0.6 s`. Stationary there is no smear and
you can integrate as long as you like.

### `decay_frame` must be fixed with respect to the world

Non-negotiable, and the reason accumulation can't just reuse `mask_frame`.
`mask_frame` defaults to `base_footprint`, which **rides the robot** — stacking
scans there piles every sweep on top of itself at the current pose and produces
mush rather than a swept-out room. `decay_frame` defaults to `odom` and must
name a frame that doesn't move: `odom` or `map`, never `base_link` or
`base_footprint`.

The two settings are independent on purpose: the mask may legitimately be
evaluated in a moving frame, accumulation may not.

For the same reason `cloud_processed` is **published in `decay_frame`** whenever
accumulation is on, whatever `publish_frame` says. An accumulated cloud spans
many sensor poses, so no single sensor frame describes it. With `decay_time: 0`
the topic follows `publish_frame` as usual.

### Bounds

Age-out is measured against each **cloud's own header stamp**, not the wall
clock, so the window follows the data — which matters because the raw feed is
stamped from the robot's clock rather than this node's.

`decay_max_points` (default 2,000,000) caps the buffer, evicting oldest-first.
RViz needs no such cap because it's bounded by a render budget; an unbounded
buffer in a node would simply grow. A publisher changing its field layout
mid-run flushes the buffer, since concatenation needs one stride throughout.

The mask runs **before** accumulation, so blanked and out-of-band points never
enter the buffer.

## Two frame settings, and why they're separate

`mask_frame` and `mask_origin` answer different questions, and conflating them
is a real trap:

- **`mask_frame`** supplies the mask **axes** and the **z datum** — what
  "up" and "height" mean.
- **`mask_origin`** is the point that **range, azimuth and elevation are
  measured from** — where the field of view is anchored.

Without the split, pointing `mask_frame` at a fixed odom frame would make the
range band a shell around *wherever the robot booted* rather than around the
robot. With `mask_origin: sensor` (the default), the axes and z datum come from
`mask_frame` while the FOV still tracks the robot.

z is deliberately **not** re-based on `mask_origin` — it is a datum in
`mask_frame`, not a direction from a point.

| what you want | `mask_frame` | `mask_origin` |
|---|---|---|
| FOV anchored to the robot, height above the floor | `<prefix>/base_footprint` | `sensor` |
| Same, but a datum that doesn't bob with a legged gait | `<prefix>/odom` | `sensor` |
| A fixed region of the world, robot-independent | `<prefix>/odom` | `mask_frame` |
| Anchored at some other frame (e.g. the robot body) | either | `<prefix>/base_link` |

**Picking the z datum for cross-robot work.** `base_footprint` sits on the
ground on both robots by construction, so a z band means the same height above
the floor on each — but on a legged robot it derives from a body that bobs.
`odom` is fixed and gravity-aligned, but each robot's odom origin sits at a
different height (§4 measured the Go2's floor at z = −0.128 m in `go2/odom`), so
a shared z band there needs a per-robot offset to stay comparable.

## Tuning loop

Nothing computes the angles for you. The node reports what survived, and you
adjust until both robots' retained distributions overlap:

```
masked 47 clouds | kept 1204/3847 (31.3%) | z 5/50/95: 0.21/0.58/1.74 | range 5/50/95: 0.91/2.44/4.31 | elev 5/50/95: -8.2/4.1/22.6 | sens_r 5/50/95: 0.34/2.51/4.40
```

`sens_r` is the euclidean distance of survivors from the sensor, and it is how
you size `sensor_blank_radius`: leave the radius at 0, look at where the 5th
percentile sits to find your mount returns, then raise the radius until it
clears them — and stop there, because past that you are discarding real
geometry.

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
| `mask_frame` | `base_footprint` | frame supplying the mask axes and the z datum |
| `mask_origin` | `sensor` | where range/azimuth/elevation are measured from: `sensor` \| `mask_frame` \| an explicit frame id |
| `publish_frame` | `input` | `input` keeps points in their original frame (pure filter); `mask` rewrites x/y/z into `mask_frame` to co-register both feeds |
| `z_min` / `z_max` | unbounded | z band in `mask_frame` |
| `min_range` / `max_range` | `0` / unbounded | radial band |
| `range_mode` | `horizontal` | `horizontal` \| `euclidean` |
| `sensor_blank_radius` | `0.0` | drop points within this euclidean radius of the **sensor** origin, for self-hit removal; `0` disables |
| `elev_min_deg` / `elev_max_deg` | `-90` / `90` | elevation cone, measured from `mask_origin` |
| `azim_min_deg` / `azim_max_deg` | `-180` / `180` | azimuth sector, wraps if min > max |
| `decay_time` | `0.0` | seconds of history to accumulate; switches `cloud_processed` from per-scan to accumulated. `0` disables |
| `decay_frame` | `odom` | frame accumulation happens in; **must be fixed w.r.t. the world** |
| `decay_max_points` | `2000000` | hard cap on the buffer; oldest evicted first |
| `tf_timeout` | `0.1` | seconds before falling back to the latest transform |
| `stats_period` | `5.0` | seconds between retention reports; `0` disables |

QoS is BEST_EFFORT / KEEP_LAST / depth 1 on both sides — a reliable
subscription would silently receive nothing from either robot's lidar driver.

## Running it

Topics are `cloud_in` and `cloud_processed`, remapped at launch — the same idiom
this stack uses for `pointcloud_to_laserscan` (`cloud_in` → `scan`).

### Alongside the Go2 stack

```bash
ros2 launch go2_robot_sdk robot.launch.py raw_lidar:=true raw_lidar_iface:=eth0 fov_mask:=true
```

Publishes `/go2/raw_lidar_processed` alongside the untouched `/go2/raw_lidar`.

| argument | default | meaning |
|---|---|---|
| `fov_mask` | `false` | enable the node |
| `fov_mask_params` | `config/fov_mask.yaml` | path to the shared mask yaml |
| `fov_mask_frame` | `''` | mask axes / z datum; empty derives `<tf_prefix>/base_footprint` |
| `fov_mask_origin` | `''` | measurement origin; empty keeps the yaml's value |
| `fov_mask_decay` | `''` | seconds of history; empty keeps the yaml's value, `0` disables |
| `fov_mask_decay_frame` | `''` | accumulation frame; empty derives `<tf_prefix>/odom` |

Both frame arguments exist because their values carry each robot's `tf_prefix`
and so cannot live in a yaml meant to be shared. For a datum that doesn't bob
with the gait:

```bash
ros2 launch go2_robot_sdk robot.launch.py raw_lidar:=true raw_lidar_iface:=enP8p1s0 fov_mask:=true fov_mask_frame:=go2/odom
```

To make that same topic carry 5 seconds of accumulated history instead:

```bash
ros2 launch go2_robot_sdk robot.launch.py raw_lidar:=true raw_lidar_iface:=enP8p1s0 fov_mask:=true fov_mask_decay:=5.0
```

### Standalone, on either robot

```bash
ros2 run lidar_processor fov_mask --ros-args --params-file /path/to/fov_mask.yaml -p mask_frame:=ganon/base_footprint -r cloud_in:=/ganon/unilidar/cloud -r cloud_processed:=/ganon/unilidar/cloud_processed
```

### Feeding cslam

Point the bridge's `pointcloud_topic` at the masked topic. Keep
`publish_frame: input` so the cloud stays in its sensor frame and cslam's own
`sensor_base_frame_id` handling still applies — and make sure that value carries
the right prefix, which was bug #11.

## Verifying

```bash
ros2 topic hz /go2/raw_lidar_processed --qos-reliability best_effort
```

If retention reads 0%, the usual causes are a `mask_frame` that doesn't exist
(the node warns on the TF lookup), or a band set in the wrong units — angles are
**degrees** in the params, radians only internally.
