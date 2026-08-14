# Two-robot Swarm-SLAM: findings, fixes, and where it stops

Investigation log for running [Swarm-SLAM](https://github.com/MISTLab/Swarm-SLAM) across a
Unitree Go2 (non-EDU, WebRTC) and a Wheeltec Ackermann rover (`ganon`, Unitree L2
lidar). Written 2026-08-05.

**Outcome: intra-robot SLAM works on both robots. Cross-robot loop closure does
not, and is not reachable by configuration on this hardware.** The reasoning and
the measurements behind that are below, so the conclusion can be re-checked
rather than taken on trust.

Robot id mapping used throughout: **`r0` = rover (`ganon`)**, **`r1` = Go2**.

---

## 1. The blocking constraint

Swarm-SLAM's lidar frontend is ScanContext for place recognition plus
FPFH → TEASER for geometric verification. Both require **the two robots to
observe overlapping physical surfaces with comparable sampling**. Neither
tolerates the sensors seeing different parts of the world.

Three properties of the Go2's feed make that unachievable here, and none are
configurable:

| property | value | why it blocks |
|---|---|---|
| pre-quantised | every point on a 5 cm lattice | FPFH is a histogram of local sampling |
| short reach | ~4.5 m maximum | ScanContext bins into 4 m rings |
| inverted mount | `base_link → radar` at `rpy="0 2.8782 0"` (165°) | sees mostly floor; floor is registration-degenerate |

The mount is the binding one. Orientation, not sensor model, is what matters:
two *different* lidars pointed the same way would work; two *identical* lidars
pointed differently would not.

### Why the mount can't be worked around

The Go2's lidar is factory-mounted and inverted, so within any shared height
band its content is floor-dominated (median z **+0.07 m** in the 0–1 m band).
A flat plane is translation-degenerate — it slides freely in x/y and still fits.
The measurable signature: `r1_0` and `r1_1` are the *same cloud* (ScanContext
similarity 1.000, RANSAC translation `[0,0,0]`, fitness 1.000), yet the rover
registered against each with translations `[-3.59, -0.60]` and `[-0.13, -3.14]`.
Two different answers for one pair at fitness 0.86 — the registration finding an
arbitrary alignment because there is no distinctive shared geometry to lock onto.

Matching the rover *to* the Go2 means pointing the rover at the floor too, which
inherits the degeneracy rather than escaping it.

### Swarm-SLAM assumes sensor homogeneity

Every lidar config the authors ship (`kitti_lidar`, `ouster_lidar`,
`graco_lidar`, `m2dgr`) uses one sensor type per experiment with
`voxel_size: 0.5` and `registration_min_inliers` of 60–100. There is also **no
facility to seed the inter-robot transform manually** — the only path to a merged
graph is a verified loop closure, so a known starting offset can't substitute.

---

## 2. Hardware ceiling: no raw lidar on a non-EDU Go2

The WebRTC channel carries only `rt/utlidar/voxel_map_compressed`, transmitted as
**integer voxel indices**. `lidar_decoder.py:43` scales them:

```python
position_array *= res
```

So the quantisation happens on the robot, before transport. Confirmed by
measurement: **100%** of the Go2's nearest-neighbour distances are exactly
0.0500 m, versus 0.002–0.26 m for the rover's raw L2 sweep.

The SDK advertises a CycloneDDS (Ethernet) path as an alternative. It is not
usable on this tier. With the Jetson cabled to the Go2 on `192.168.123.0/24`
and `rmw_cyclonedds_cpp` installed, the robot's entire DDS presence is:

```
/api/obstacles_avoid/{request,response}
/api/sport/{request,response}
/api/vui/{request,response}
/wirelesscontroller_unprocessed
```

That is the high-level command API. **No `/utlidar/cloud`, no
`voxel_map`, no `lowstate`, no `/utlidar/robot_pose`.** Ten ROS domains were
scanned (0–5, 10, 30, 42, 67); only domain 0 has anything, so the sensor topics
are firmware-gated rather than hiding elsewhere.

Consequences:

- The 5 cm lattice is permanent on this hardware.
- The SDK's `_on_cyclonedds_low_state` and `_on_cyclonedds_pose` stubs target
  topics that do not exist on this tier.
- `publish_raw_lidar` / `_on_raw_lidar` in `go2_driver_node.py` (added during
  this investigation) is inert here. It defaults off and would work on an EDU
  unit; it is retained rather than removed for that reason.

---

## 3. Real bugs fixed

These were genuine defects, independent of the cross-robot question. Intra-robot
SLAM did not work before them and does now.

### In this repo (`go2_ros2_sdk`)

| # | Bug | Effect |
|---|---|---|
| 1 | `RobotConfig.from_params()` had `aes_key` in the dataclass field and the `cls(...)` call but **not in the signature** | `TypeError: unexpected keyword argument 'aes_key'` — driver would not start at all |
| 2 | `base_footprint` joint had `xyz="0 0 0"`, coincident with `base_link` | a frame named "footprint" sitting ~0.4 m above the ground |
| 3 | `map` and `odom` declared as fixed links in `go2.urdf` | `robot_state_publisher` asserts `map→odom` and `odom→base_link` as static identities, competing with slam_toolbox and the driver. Verified *not* currently breaking lookups (the driver's dynamic transform wins) but it is two authorities per edge |
| 4 | TF frames unprefixed | multiple robots collided on the global `/tf`; fixed with a `tf_prefix` argument threaded through all five launch files, using `frame_prefix` on `robot_state_publisher` so no URDF edits were needed |

### In the wheeltec stack

| # | Bug | Effect |
|---|---|---|
| 5 | `lidar_static_tf` missing a comma in `swarm_slam.launch.py` — Python concatenated `'0.241' 'base_footprint'` | `static_transform_publisher` got 7 positional args instead of 8 and exited; `base_footprint → unilidar_lidar` was never published |
| 6 | `ekf.yaml` keyed `ekf_filter_node:` but `PushRosNamespace` makes the node `/ganon/ekf_filter_node` | params never matched, so the EKF had no `odom0`/`imu0` inputs and published **nothing** — `/ganon/odom_combined` is an empty topic. `ekf_carto.yaml` is worse: keyed for a node named `carto_ekf_filter_node` |
| 7 | `range_max: 5.0` on the L2 (spec is 30 m) | rover saw 2 of ScanContext's 20 rings |
| 8 | `nav2_params.yaml`, `twist_mux.yaml` and the slam_toolbox params have the same bare-key/FQN mismatch as #6 | nav2's `controller_server` fails to configure (`No critics defined for FollowPath`) and aborts bringup; joystick scaling and deadman never applied. **Not fixed** — see §6 |

### In `cslam_robot_bridges`

| # | Bug | Effect |
|---|---|---|
| 9 | `'max_nbrobots'` typo in `wheeltec_lidar.launch.py` | `max_nb_robots` never reached cslam, which fell back to `default_value='1'`. The rover ran as a single-robot system and could never attempt an inter-robot closure |
| 10 | `wheeltec_lidar.yaml` had `pointcloud_topic: "/ganon/unilidar/cloud"` (absolute) while `odom_topic` was relative | cslam read the raw lidar cloud in `ganon/unilidar_lidar` and odometry in `ganon/base_footprint` — two different frames, bypassing the bridge whose job was to reconcile them |
| 11 | `body_frame` / `sensor_base_frame_id` hardcoded unprefixed in all four bridge launch files | `TF lookup failed (go2/odom -> base_link)` once frames were namespaced |

---

## 4. Measurements worth keeping

Re-deriving these is expensive. Tooling that produced them is in §5.

**Sampling structure**

| | rover (raw) | Go2 |
|---|---|---|
| NN distance 1/50/99 pct | 0.0019 / 0.0130 / 0.2576 m | 0.0500 / 0.0500 / 0.0500 m |
| fraction within 2 mm of median NN | 8.4% | **100.0%** |

**Why stock filters don't normalise this.** PCL's `VoxelGrid`, `pcl_ros`'s
wrapper, and Open3D's `voxel_down_sample` all return the **centroid** of each
occupied cell. On uniform random points at 0.05 m, centroids land **1.9%**
on-lattice; snapping to cell *centres* gives **100%**. This is why Swarm-SLAM's
own `voxel_size: 0.3` downsample never made the two feeds comparable — it
averages, it does not snap.

**Go2 geometry**

- Ground plane in `go2/odom`: **z = −0.128 m**, normal `[0.002, 0.018, 1.000]`,
  tilt **1.1°** from vertical (33284/50081 inliers within 5 cm). The extrinsic
  *orientation* is correct.
- `base_link` in `go2/odom`: **+0.381 m** — so 0.509 m above a level floor.
- Leg reach is 0.426 m (thigh 0.213 + calf 0.213). **0.509 exceeds what the legs
  physically allow**, so odom z is inflated. Prime suspect is the unexplained
  `+ 0.07` added to z in `ros2_publisher.py` lines 71 and 97; removing it gives
  0.439, within reach of a tall stance.
- A fixed `base_footprint` offset cannot be correct anyway: a quadruped's body
  height varies as it walks. The ground correction was therefore applied at the
  consumer (`z_offset` on `go2_bridge`), leaving the SDK's odom convention — which
  its own control path depends on — untouched.

**Vertical coverage, before and after the fixes**

| | rover z 5/50/95 | Go2 z 5/50/95 | shared band |
|---|---|---|---|
| initially | +0.05 / **+3.37** / +4.61 | −0.54 / −0.40 / +0.21 | 22% / 20% of points |
| after re-aim + range + offset | +0.05 / +0.25 / +0.85 | +0.01 / +0.11 / +0.74 | 77% / 22% |

The rover's median return was originally at **+62° elevation** — pointed at the
ceiling. Median-to-median offset between the robots was **3.77 m**.

**ScanContext internals** (`cslam/lidar_pr/scancontext*.py`)

- Descriptor is a 20×60 grid over an 80 m radius, so **4 m per ring**.
- `pt2rs` bins by x,y only — z is explicitly not used for binning.
- Each cell stores `max(z + 2.0)` over the points in it. **The descriptor is a
  max-height map**: which surfaces a sensor illuminates *is* its content.
- Both robots populate only **2–3 of the 20 rings**, because neither exceeds
  ~11 m and the Go2 stops at 4.5 m.
- `distance_sc` uses per-column **cosine** similarity, which is scale-invariant.
  Two near-constant columns (flat floor vs flat ceiling) therefore score ≈1.0
  regardless of magnitude. This produces *spuriously high* similarity — the
  observed 0.887–0.920 on cross-robot pairs, with one Go2 keyframe matching
  twenty-odd rover keyframes.

**Registration funnel** (voxel_size 0.3, `registration_min_inliers: 30`)

| stage | rover↔rover | Go2↔Go2 | cross |
|---|---|---|---|
| raw points | 1187 / 1323 | 62970 / 62867 | 1187 / 62970 |
| after 0.3 m voxel | 266 / 249 | 880 / 883 | 266 / 880 |
| mutual correspondences | 40 | 595 | **62–68** |
| RANSAC fitness | 0.789 | 1.000 | 0.812–0.861 |

Lattice-matching raised cross-robot correspondences from **3–14** inliers to
**62–68** — a real effect, but not the binding one, since the resulting
transforms are mutually inconsistent (see §1).

Note the same-robot "controls" are weak: both robots were stationary during
capture, so those pairs are effectively a cloud registered against itself.

**Density**

Within 4.5 m: one rover sweep gives **1431–1761** distinct 5 cm cells; the Go2's
accumulated map holds **63k–97k**. At the rover's 12 Hz that is ~40 sweeps
(~3.5 s) of accumulation to reach parity — but only while *moving*.
Re-observing the same cells adds nothing, so accumulation saturates well below
the Go2's count from a single viewpoint.

**cslam back-end logs** (rover, 53 optimisation snapshots)

```
total_nb_successful_matches      0      <- inter-robot only
total_nb_failed_matches        182
nb_vertices / nb_edges       92 / 94    <- 91 odometry + 3 intra-robot closures
total_error                 3797.9
inter_robot_loop_closures        0
```

Optimisation moved vertices a mean of 0.98 m and up to 4.64 m — a badly
conditioned graph, consistent with drift from visual odometry losing tracking.
The three loop-closure edges (index deltas 20, 20, 40) assert translations
averaging 0.125 m against odometry edges averaging 0.518 m.

---

## 5. Diagnostic tooling

Written during this investigation. All read-only.

| script | what it answers |
|---|---|
| `zprobe.py` | z and horizontal-range distributions per feed; how much vertical overlap two robots have |
| `groundfit.py` | RANSAC ground-plane fit: how far below a frame's origin the floor sits, and the plane's tilt (validated against synthetic scenes with known truth) |
| `latticetest.py` | is a cloud quantised to a lattice, and at what step |
| `compare_feeds.py` | side-by-side of the two clouds cslam consumes, with pass/fail verdicts on lattice, density, height band, range cap and rings |
| `lcdiag.py` | replays cslam's *own* ScanContext and FPFH code over captured clouds and reports every stage of the funnel, including same-robot controls. Uses the real `solve_teaser` when run on a robot |

`lcdiag.py`'s key output is the **mutual correspondence count** — TEASER's max
clique can never exceed it, so it distinguishes "not enough correspondences"
from "correspondences are wrong".

### Traps these tools exposed

- `ros2 topic hz` defaults to a **reliable** subscription and silently reports
  nothing against a best-effort publisher. Both `/go2/point_cloud2` and the
  bridge outputs are best-effort. Several "NO DATA" readings during this
  investigation were this, not real outages.
- `ros2 topic echo /tf --once` captures **one publisher's** message. Aggregate
  over several seconds before concluding a frame is missing.
- gtsam vertex keys are 17 digits, beyond double precision. `awk` arithmetic on
  them silently produces garbage; use Python integers.

---

## 6. Known-unfixed

| item | why it was left |
|---|---|
| `nav2_params.yaml` bare-key/FQN mismatch | nav2 aborts bringup (`No critics defined for FollowPath`). The fix needs `namespace:=` passed to `nav2_bringup` so its `RewrittenYaml` re-roots the params, which interacts with the outer `PushRosNamespace` and could double to `/go2/go2/...`. Workaround: `nav2:=false` |
| `twist_mux.yaml` same mismatch | joystick axis scaling and `require_enable_button: false` never apply, so `teleop_twist_joy` defaults to requiring button 0 as deadman |
| `map` / `odom` links in `go2.urdf` | competing TF authorities; verified not currently breaking lookups. Removing them makes `base_link` the URDF root, which is conventional, but it is a tree change mid-debug |
| `+ 0.07` in `ros2_publisher.py` z | unexplained, and inflates odom z past what the legs allow. Left alone because the SDK's control path may depend on it; corrected at the consumer instead |
| `spectral_matches.csv` empty while `total_nb_matches_selected: 91` | inconsistent; not understood |
| `/ganon/odom_combined` frame name | the frame `/odom` declares is `ganon/odom_combined`, but it is RTAB-Map's RGB-D odometry, not an EKF fusion. Misleading; renaming ripples through several files |

---

## 7. If picking this up again

**Don't** spend more time on `registration_min_inliers`, `voxel_size`, height
bands, range caps or lattice matching. Those are all done, and the remaining
failure is not a threshold.

The options that could actually change the outcome:

1. **Change the mount geometry** so both sensors observe an overlapping,
   non-degenerate band — walls rather than floor. The Go2's is factory-fixed, so
   this likely means a different lidar on the Go2, or accepting the rover matches
   *to* the floor and living with the degeneracy.
2. **A viewpoint-robust descriptor** in place of ScanContext. Cross-configuration
   lidar place recognition is an open research problem, not a config change.
3. **Accept per-robot SLAM.** Both robots close loops internally and reliably.
   Merging would need an external mechanism — shared fiducials, a common global
   reference, or manual alignment — since cslam offers no way to seed the
   inter-robot transform.

For an **EDU** Go2 the picture changes: raw lidar over Ethernet becomes
available, `publish_raw_lidar` in `go2_driver_node.py` is already wired for it,
and the quantisation constraint disappears. The orientation constraint would
remain.
