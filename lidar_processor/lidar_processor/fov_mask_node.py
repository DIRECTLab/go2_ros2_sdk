# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

"""
FOV Mask Node

Drops points from a PointCloud2 that fall outside a configurable region, so two
robots with differently-mounted lidars can be reduced to a comparable subset of
what they observe.

Written for the cross-robot problem in docs/SWARM_SLAM_FINDINGS.md: the Go2's
lidar is factory-mounted inverted (165 deg) and sees mostly floor, while the
rover's L2 is upright. Run one instance per robot with the same mask and the
retained clouds describe the same band of the world.

Deliberately layout-agnostic. Fields are read from each message's own
descriptors rather than any hardcoded layout, because the two robots' feeds do
not share one -- the Go2's raw feed is a 32-byte x/y/z/intensity/ring/time
stride, the rover's comes from unitree_lidar_ros2. Surviving points are copied
as whole rows, so every field and every padding byte is preserved exactly.

The mask is evaluated in a gravity-aligned frame (`mask_frame`) reached by TF.
That is not optional for cross-robot work: in the Go2's own sensor frame
"horizontal" is tilted 165 deg, so an elevation band there selects a completely
different physical region than the same numbers on the rover.

This node publishes no TF and does not modify the input topic.
"""

import math
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos_overriding_options import QoSOverridingOptions
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# sensor_msgs/PointField enum -> numpy dtype string
_POINTFIELD_NUMPY = {
    1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2',
    5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8',
}

# Sentinels standing in for "unbounded". Params cannot carry inf portably, and
# a finite bound this large is unreachable by any lidar.
UNBOUNDED = 1.0e9

RANGE_MODES = ('horizontal', 'euclidean')
PUBLISH_FRAMES = ('input', 'mask')
# mask_origin also accepts any explicit frame id, resolved through TF.
MASK_ORIGIN_KEYWORDS = ('sensor', 'mask_frame')


class FovMaskNode(Node):
    """Masks a PointCloud2 down to a configurable region of the world."""

    def __init__(self):
        super().__init__('fov_mask_node')

        self._declare_parameters()
        self._read_parameters()

        self.tf_buffer = Buffer()
        # spin_thread=True is not optional here. Without it the listener's /tf
        # subscription is serviced by the same single-threaded executor that
        # runs _on_cloud -- which then blocks inside lookup_transform waiting
        # for a transform that can only arrive if that executor is free to
        # process it. The buffer falls further behind on every cloud, and the
        # lookups fail with "only time T is in the buffer" for a T that keeps
        # receding. An all-static chain still resolves, because static
        # transforms are valid at any time, so the symptom looks selective.
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # BEST_EFFORT on both sides: it is what both robots' lidar drivers
        # publish, and a reliable subscription would silently receive nothing.
        cloud_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Topic names are deliberately generic and remapped at launch, matching
        # how this stack wires pointcloud_to_laserscan (cloud_in -> scan).
        # One output topic carrying whichever form is configured: the per-scan
        # masked cloud, or the accumulated one when decay_time is set. Exactly
        # one of _publish/_accumulate runs per input cloud, so the two never
        # interleave on the wire.
        self.publisher = self.create_publisher(
            PointCloud2, 'cloud_processed', cloud_qos,
            qos_overriding_options=QoSOverridingOptions.with_default_policies())
        self.create_subscription(
            PointCloud2, 'cloud_in', self._on_cloud, cloud_qos)

        # Accumulation buffer: (stamp_seconds, rows) oldest first, every row
        # already rewritten into decay_frame so entries can simply be
        # concatenated. A deque because pruning is always from the front.
        self._decay_buf: deque = deque()
        self._decay_points = 0
        self._decay_layout: Optional[Tuple] = None
        self._time_dtype_cache: Dict[Tuple, Optional[np.dtype]] = {}

        # TF staleness diagnostics. The cloud stamp and the TF stamp can come
        # from different clocks -- the raw lidar feed is stamped from the
        # robot's clock while odom is stamped from this machine's -- and the
        # resulting skew is invisible unless it is measured.
        self._tf_stale = 0
        self._tf_skew_last: Optional[float] = None
        self._deskewed = 0
        self._deskew_failed = 0

        # Rolling stats, reset each reporting window.
        self._msgs = 0
        self._points_in = 0
        self._points_out = 0
        self._dropped_msgs = 0
        self._last_retained: Optional[Dict[str, np.ndarray]] = None
        self._dtype_cache: Dict[Tuple, np.dtype] = {}

        self._log_configuration()

        if self.stats_period > 0.0:
            self.create_timer(self.stats_period, self._report_stats)

    # -- setup -------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameters(
            namespace='',
            parameters=[
                ('mask_frame', 'base_footprint', ParameterDescriptor(
                    description='Gravity-aligned frame supplying the mask AXES and '
                                'the z datum. Set to a robot odom frame for a '
                                'reference that does not bob with a legged gait; set '
                                'to base_footprint for a z datum that sits on the '
                                'ground on both robots by construction.')),
                ('mask_origin', 'sensor', ParameterDescriptor(
                    description='Where range, azimuth and elevation are measured '
                                "FROM -- distinct from mask_frame, which only "
                                "supplies the axes. 'sensor' uses the cloud frame's "
                                "origin, so the field of view follows the robot even "
                                "when mask_frame is a fixed odom frame. "
                                "'mask_frame' measures from that frame's origin. Any "
                                'other value is treated as an explicit frame id and '
                                'resolved through TF.')),
                ('publish_frame', 'input', ParameterDescriptor(
                    description="'input' republishes surviving points untouched in "
                                "the cloud's original frame (a pure filter). 'mask' "
                                "rewrites x/y/z into mask_frame, which co-registers "
                                "the two robots' feeds for direct comparison.")),

                # -- z band: the primitive that matches ScanContext ------------
                ('z_min', -UNBOUNDED, ParameterDescriptor(
                    description='Keep points with mask_frame z >= this. ScanContext '
                                'stores max(z+2.0) per x/y cell, so a z band is the '
                                'most direct way to make two feeds describe the same '
                                'content. Excludes floor uniformly at every range, '
                                'which an elevation band does not.')),
                ('z_max', UNBOUNDED, ParameterDescriptor(
                    description='Keep points with mask_frame z <= this.')),

                # -- range band ------------------------------------------------
                ('min_range', 0.0, ParameterDescriptor(
                    description='Keep points at least this far from the origin.')),
                ('max_range', UNBOUNDED, ParameterDescriptor(
                    description='Keep points at most this far from the origin. Use to '
                                'cap the rover (30 m spec) to the Go2 reach so both '
                                'populate the same ScanContext rings.')),
                ('range_mode', 'horizontal', ParameterDescriptor(
                    description="'horizontal' measures sqrt(x^2+y^2), matching how "
                                "ScanContext bins (by x,y only). 'euclidean' measures "
                                "true slant range, matching sensor datasheet specs.")),

                # -- self-hit blanking -----------------------------------------
                ('sensor_blank_radius', 0.0, ParameterDescriptor(
                    description='Drop points within this euclidean distance of the '
                                'SENSOR origin, in metres, to remove returns off the '
                                "lidar's own mount. Deliberately separate from "
                                'min_range: always a sphere and always measured from '
                                'the cloud frame origin, independent of range_mode '
                                'and mask_origin, because the mount is bolted to the '
                                'sensor and does not move when the measurement origin '
                                'does. 0 disables.')),

                # -- elevation cone --------------------------------------------
                ('elev_min_deg', -90.0, ParameterDescriptor(
                    description='Keep points at least this elevation above horizontal, '
                                'in degrees, measured in the gravity-aligned '
                                'mask_frame. Use to emulate the cone one sensor '
                                'physically cannot see beyond on the other robot.')),
                ('elev_max_deg', 90.0, ParameterDescriptor(
                    description='Keep points at most this elevation, in degrees.')),

                # -- azimuth sector --------------------------------------------
                ('azim_min_deg', -180.0, ParameterDescriptor(
                    description='Keep points at or beyond this yaw, in degrees, in '
                                'mask_frame. If azim_min > azim_max the sector is '
                                'treated as wrapping through +/-180.')),
                ('azim_max_deg', 180.0, ParameterDescriptor(
                    description='Keep points at or below this yaw, in degrees.')),

                # -- decay / accumulation --------------------------------------
                ('decay_time', 0.0, ParameterDescriptor(
                    description='Seconds of history to accumulate, like RViz2\'s '
                                'Decay Time. Points older than this are dropped. '
                                '0 disables accumulation entirely, matching RViz '
                                'where 0 means "show only the newest cloud". '
                                'Changes what cloud_processed carries: per-scan '
                                'masked clouds when 0, the accumulated history when '
                                'set. Leave it at 0 when feeding scan-to-map odometry '
                                'such as KISS-ICP, which is sequential and deskews '
                                'using the per-point time offsets. cslam tolerates '
                                'accumulated clouds -- it was already fed the Go2 '
                                "voxel map -- but ties one odometry pose to each "
                                'keyframe, so a cloud spanning several poses makes '
                                'the loop-closure transform not a pose-to-pose one.')),
                ('decay_frame', 'odom', ParameterDescriptor(
                    description='Frame the accumulation happens in. MUST be fixed '
                                'with respect to the world -- odom or map, never '
                                'base_link or base_footprint. Stacking scans in a '
                                'frame that rides the robot piles every sweep at the '
                                'current pose instead of sweeping out the room. '
                                'Independent of mask_frame, which may legitimately be '
                                'a moving frame.')),
                ('decay_max_points', 2000000, ParameterDescriptor(
                    description='Hard cap on accumulated points; the oldest clouds are '
                                'evicted first once it is reached. RViz has no such '
                                'cap because it is bounded by a render budget, but an '
                                'unbounded buffer here would grow without limit.')),

                ('deskew', True, ParameterDescriptor(
                    description='Motion-compensate each sweep using its per-point '
                                'time offsets: the transform is looked up at sweep '
                                'start and sweep end and interpolated per point, '
                                'instead of one transform for the whole sweep. '
                                'Intra-sweep smear is dominated by rotation and '
                                'scales with range, so it hits exactly the distant '
                                'returns that carry the distinctive geometry. Falls '
                                'back to a single transform when the cloud has no '
                                'usable time field. Only applies to the accumulation '
                                'path, which is the only one with a frame fixed '
                                'enough to deskew into.')),
                ('tf_allow_stale', True, ParameterDescriptor(
                    description='When no transform exists at the cloud\'s stamp, use '
                                'the latest available one instead of dropping the '
                                'cloud. Convenient, but for accumulation it silently '
                                'misplaces whole clouds by however far the robot '
                                'moved in the gap -- set false to drop instead. '
                                'Either way the gap is warned about and counted.')),

                ('tf_timeout', 0.1, ParameterDescriptor(
                    description='Seconds to wait for the TF lookup before falling '
                                'back to the latest available transform.')),
                ('stats_period', 5.0, ParameterDescriptor(
                    description='Seconds between retention reports. 0 disables. The '
                                'reported percentiles are what you tune against: '
                                'adjust the band until both robots retain overlapping '
                                'distributions.')),
            ]
        )

    def _read_parameters(self) -> None:
        get = self.get_parameter
        self.mask_frame = get('mask_frame').get_parameter_value().string_value
        self.mask_origin = get('mask_origin').get_parameter_value().string_value
        self.publish_frame = get('publish_frame').get_parameter_value().string_value
        self.z_min = get('z_min').get_parameter_value().double_value
        self.z_max = get('z_max').get_parameter_value().double_value
        self.min_range = get('min_range').get_parameter_value().double_value
        self.max_range = get('max_range').get_parameter_value().double_value
        self.range_mode = get('range_mode').get_parameter_value().string_value
        self.sensor_blank_radius = get(
            'sensor_blank_radius').get_parameter_value().double_value
        self.elev_min = math.radians(get('elev_min_deg').get_parameter_value().double_value)
        self.elev_max = math.radians(get('elev_max_deg').get_parameter_value().double_value)
        self.azim_min = math.radians(get('azim_min_deg').get_parameter_value().double_value)
        self.azim_max = math.radians(get('azim_max_deg').get_parameter_value().double_value)
        self.decay_time = get('decay_time').get_parameter_value().double_value
        self.decay_frame = get('decay_frame').get_parameter_value().string_value
        self.decay_max_points = get(
            'decay_max_points').get_parameter_value().integer_value
        self.deskew = get('deskew').get_parameter_value().bool_value
        self.tf_allow_stale = get('tf_allow_stale').get_parameter_value().bool_value
        self.tf_timeout = get('tf_timeout').get_parameter_value().double_value
        self.stats_period = get('stats_period').get_parameter_value().double_value

        for name, value, valid in (
            ('range_mode', self.range_mode, RANGE_MODES),
            ('publish_frame', self.publish_frame, PUBLISH_FRAMES),
        ):
            if value not in valid:
                self.get_logger().warn(
                    f"Unknown {name} '{value}', using '{valid[0]}'. "
                    f"Valid values: {', '.join(valid)}")
                setattr(self, name, valid[0])

    # -- cloud handling ----------------------------------------------------

    def _on_cloud(self, msg: PointCloud2) -> None:
        try:
            n_points = int(msg.width) * int(msg.height)
            point_step = int(msg.point_step)
            payload = bytes(msg.data)

            if n_points <= 0 or len(payload) < n_points * point_step:
                self._dropped_msgs += 1
                self.get_logger().warn(
                    f"Dropping malformed cloud: {msg.width}x{msg.height} points but "
                    f"{len(payload)} payload bytes", throttle_duration_sec=10.0)
                return

            if msg.is_bigendian:
                self._dropped_msgs += 1
                self.get_logger().error(
                    "Big-endian clouds are not supported", throttle_duration_sec=30.0)
                return

            xyz_dtype = self._xyz_dtype(msg, point_step)
            if xyz_dtype is None:
                self._dropped_msgs += 1
                self.get_logger().error(
                    "Cloud has no float x/y/z fields; cannot mask it",
                    throttle_duration_sec=30.0)
                return

            transform = self._lookup_transform(self.mask_frame, msg.header)
            if transform is None:
                self._dropped_msgs += 1
                return
            rotation, translation = transform

            origin = self._resolve_origin(msg, translation)
            if origin is None:
                self._dropped_msgs += 1
                return

            view = np.frombuffer(payload, dtype=xyz_dtype, count=n_points)
            xyz = np.stack([view['x'], view['y'], view['z']], axis=1).astype(np.float64)

            # Into the gravity-aligned frame, where the band actually means
            # something consistent across two differently-mounted sensors.
            masked_xyz = xyz @ rotation.T + translation

            keep, metrics = self._evaluate_mask(masked_xyz, origin, xyz)

            self._msgs += 1
            self._points_in += n_points
            self._points_out += int(keep.sum())
            self._last_retained = {k: v[keep] for k, v in metrics.items()}

            # Either/or, never both: the accumulated cloud already contains
            # this scan's surviving points, so publishing both would put the
            # same points on the topic twice.
            if self.decay_time > 0.0:
                self._accumulate(msg, payload, point_step, n_points, keep,
                                 xyz_dtype)
            else:
                self._publish(msg, payload, point_step, n_points, keep,
                              masked_xyz, xyz_dtype)

        except Exception as exc:
            self.get_logger().error(f"Error masking cloud: {exc}",
                                    throttle_duration_sec=5.0)

    def _xyz_dtype(self, msg: PointCloud2, point_step: int) -> Optional[np.dtype]:
        """Structured dtype exposing just x/y/z at this message's own offsets.

        Built from the message's field descriptors rather than an assumed
        layout, since the two robots' feeds differ. Cached, because the layout
        is stable per publisher.
        """
        key = tuple((f.name, f.offset, f.datatype) for f in msg.fields) + (point_step,)
        if key in self._dtype_cache:
            return self._dtype_cache[key]

        names, formats, offsets = [], [], []
        for field in msg.fields:
            if field.name not in ('x', 'y', 'z'):
                continue
            numpy_type = _POINTFIELD_NUMPY.get(int(field.datatype))
            if numpy_type is None or not numpy_type.startswith('f'):
                continue
            names.append(field.name)
            formats.append('<' + numpy_type)
            offsets.append(int(field.offset))

        if len(names) != 3:
            self._dtype_cache[key] = None
            return None

        dtype = np.dtype({'names': names, 'formats': formats,
                          'offsets': offsets, 'itemsize': point_step})
        self._dtype_cache[key] = dtype
        return dtype

    def _lookup_transform(self, target: str, header, source: str = None):
        """(3x3 rotation, 3-vector translation) taking `source` -> `target`."""
        source = source or header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                target, source, Time.from_msg(header.stamp),
                timeout=Duration(seconds=self.tf_timeout))
        except TransformException as exact_exc:
            # Why the exact-stamp lookup failed is the single most diagnostic
            # fact available here, and discarding it turns every cause --
            # extrapolation, a missing edge, a cache that does not reach back
            # far enough -- into the same opaque warning.
            reason = f"{type(exact_exc).__name__}: {exact_exc}"
            if not self.tf_allow_stale:
                self._tf_stale += 1
                self.get_logger().warn(
                    f"No TF {target} <- {source} at the cloud stamp; dropping the "
                    f"cloud (tf_allow_stale is false). {reason}",
                    throttle_duration_sec=10.0)
                return None
            try:
                tf = self.tf_buffer.lookup_transform(target, source, Time())
            except TransformException as exc:
                self.get_logger().warn(
                    f"TF {target} <- {source} unavailable: {exc}",
                    throttle_duration_sec=10.0)
                return None

            # Using a transform from a different instant than the cloud. Fine
            # for a static edge, systematically wrong for a moving one -- the
            # whole cloud lands wherever the robot was at the transform's time.
            self._tf_stale += 1
            skew = (Time.from_msg(header.stamp).nanoseconds
                    - Time.from_msg(tf.header.stamp).nanoseconds) * 1e-9
            self._tf_skew_last = skew
            self.get_logger().warn(
                f"No TF {target} <- {source} at cloud stamp "
                f"{Time.from_msg(header.stamp).nanoseconds * 1e-9:.3f}; fell back to "
                f"one stamped {Time.from_msg(tf.header.stamp).nanoseconds * 1e-9:.3f} "
                f"({skew:+.3f} s away). Reason: {reason}",
                throttle_duration_sec=5.0)

        t = tf.transform.translation
        q = tf.transform.rotation
        return _quat_to_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])

    def _resolve_origin(self, msg: PointCloud2, sensor_translation: np.ndarray):
        """The point in mask_frame that range/azimuth/elevation are measured from.

        Separate from mask_frame on purpose. With mask_frame set to a fixed odom
        frame -- stable, and not bobbing with a legged gait -- the axes and z
        datum come from odom while the field of view still follows the robot,
        rather than becoming a shell around wherever odom happens to sit.
        """
        if self.mask_origin == 'sensor':
            return sensor_translation
        if self.mask_origin == 'mask_frame':
            return np.zeros(3)

        transform = self._lookup_transform(
            self.mask_frame, msg.header, source=self.mask_origin)
        if transform is None:
            return None
        return transform[1]

    def _evaluate_mask(self, pts: np.ndarray, origin: np.ndarray,
                       sensor_xyz: np.ndarray):
        """Boolean keep-mask plus the per-point metrics, for stats.

        Range, azimuth and elevation are all measured from `origin`; z is not,
        because z is a datum in mask_frame rather than a direction from a point.
        That split is what lets mask_frame be a fixed odom frame while the field
        of view still tracks the robot.

        `sensor_xyz` is the untransformed cloud, used only for the self-hit
        blanking sphere, which is anchored to the sensor rather than to
        `origin`.
        """
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        # Offsets from the measurement origin, in the mask frame's axes.
        dx, dy, dz = x - origin[0], y - origin[1], z - origin[2]

        horizontal = np.hypot(dx, dy)
        rng = horizontal if self.range_mode == 'horizontal' else np.sqrt(
            dx * dx + dy * dy + dz * dz)

        elevation = np.arctan2(dz, horizontal)
        azimuth = np.arctan2(dy, dx)

        # Distance from the sensor origin, in the sensor's own frame. Rigid
        # transforms preserve distance, so this is the same sphere whatever
        # mask_frame is -- and it ignores mask_origin and range_mode on
        # purpose: the mount is attached to the sensor, not to the frame you
        # happen to be measuring the field of view from.
        sensor_r = np.sqrt((sensor_xyz * sensor_xyz).sum(axis=1))

        keep = np.isfinite(pts).all(axis=1)
        keep &= (z >= self.z_min) & (z <= self.z_max)
        keep &= (rng >= self.min_range) & (rng <= self.max_range)
        keep &= (elevation >= self.elev_min) & (elevation <= self.elev_max)

        if self.sensor_blank_radius > 0.0:
            keep &= sensor_r >= self.sensor_blank_radius

        if self.azim_min <= self.azim_max:
            keep &= (azimuth >= self.azim_min) & (azimuth <= self.azim_max)
        else:
            # Sector wrapping through +/-180.
            keep &= (azimuth >= self.azim_min) | (azimuth <= self.azim_max)

        return keep, {'z': z, 'range': rng, 'elev': np.degrees(elevation),
                      'sens_r': sensor_r}

    def _publish(self, msg: PointCloud2, payload: bytes, point_step: int,
                 n_points: int, keep: np.ndarray, masked_xyz: np.ndarray,
                 xyz_dtype: np.dtype) -> None:
        # Whole-row copy: every field and every padding byte survives exactly,
        # which matters because the Go2 feed carries ring and time that
        # downstream deskewing needs.
        rows = np.frombuffer(payload, dtype=np.uint8).reshape(n_points, point_step)
        kept_rows = rows[keep].copy()

        out = PointCloud2()
        out.header = msg.header
        out.height = 1
        out.width = int(kept_rows.shape[0])
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = point_step
        out.row_step = point_step * out.width
        out.is_dense = msg.is_dense

        if self.publish_frame == 'mask':
            out.header.frame_id = self.mask_frame
            if out.width:
                writable = kept_rows.view(xyz_dtype).reshape(-1)
                kept_xyz = masked_xyz[keep]
                writable['x'] = kept_xyz[:, 0]
                writable['y'] = kept_xyz[:, 1]
                writable['z'] = kept_xyz[:, 2]

        out.data = kept_rows.tobytes()
        self.publisher.publish(out)

    # -- accumulation ------------------------------------------------------

    def _time_dtype(self, msg: PointCloud2, point_step: int) -> Optional[np.dtype]:
        """Structured dtype exposing just the per-point time column, or None.

        Read from the message's own descriptors like _xyz_dtype, since the two
        robots' feeds do not necessarily agree on where `time` sits.
        """
        key = tuple((f.name, f.offset, f.datatype) for f in msg.fields) + (point_step,)
        if key in self._time_dtype_cache:
            return self._time_dtype_cache[key]

        dtype = None
        for field in msg.fields:
            if field.name != 'time':
                continue
            numpy_type = _POINTFIELD_NUMPY.get(int(field.datatype))
            if numpy_type is None:
                break
            dtype = np.dtype({'names': ['time'], 'formats': ['<' + numpy_type],
                              'offsets': [int(field.offset)], 'itemsize': point_step})
            break

        self._time_dtype_cache[key] = dtype
        return dtype

    def _lookup_pose(self, target: str, source: str, stamp) -> Optional[Tuple]:
        """(quaternion xyzw, translation) at an exact stamp, or None."""
        try:
            tf = self.tf_buffer.lookup_transform(
                target, source, stamp, timeout=Duration(seconds=self.tf_timeout))
        except TransformException:
            return None
        q, t = tf.transform.rotation, tf.transform.translation
        return (np.array([q.x, q.y, q.z, q.w]), np.array([t.x, t.y, t.z]))

    def _deskew(self, msg: PointCloud2, payload: bytes, point_step: int,
                n_points: int, keep: np.ndarray,
                xyz_kept: np.ndarray) -> Optional[np.ndarray]:
        """Per-point motion compensation. Returns points in decay_frame, or None.

        One transform for a whole sweep smears the cloud by however far the
        robot moved during it. Rotation dominates and the error scales with
        range, so it corrupts the distant returns that carry the most
        distinctive geometry. Here the transform is sampled at sweep start and
        sweep end and interpolated per point using the offsets the sensor
        already provides.

        Returns None whenever the sweep cannot be interpolated -- no time
        field, zero-width sweep, or a missing endpoint transform -- and the
        caller falls back to the single-transform path.
        """
        time_dtype = self._time_dtype(msg, point_step)
        if time_dtype is None:
            return None

        times = np.frombuffer(payload, dtype=time_dtype, count=n_points)['time']
        finite = times[np.isfinite(times)]
        if finite.size == 0:
            return None
        t_min, t_max = float(finite.min()), float(finite.max())
        span = t_max - t_min
        if span <= 0.0:
            return None  # single-instant cloud; nothing to compensate

        # header.stamp is the instant of the earliest point when the driver
        # runs stamp_source 'raw'. Under the other stamp sources it is offset
        # by a constant, which shifts the sweep as a whole but leaves the
        # relative interpolation -- the part that removes the smear -- intact.
        start = Time.from_msg(msg.header.stamp)
        end = start + Duration(seconds=span)

        pose0 = self._lookup_pose(self.decay_frame, msg.header.frame_id, start)
        pose1 = self._lookup_pose(self.decay_frame, msg.header.frame_id, end)
        if pose0 is None or pose1 is None:
            return None

        q0, t0 = pose0
        q1, t1 = pose1
        alpha = np.clip((times[keep].astype(np.float64) - t_min) / span, 0.0, 1.0)

        rotations = _quats_to_matrices(_slerp(q0, q1, alpha))
        translations = t0[None, :] + alpha[:, None] * (t1 - t0)[None, :]
        return np.einsum('nij,nj->ni', rotations, xyz_kept) + translations

    def _accumulate(self, msg: PointCloud2, payload: bytes, point_step: int,
                    n_points: int, keep: np.ndarray,
                    xyz_dtype: np.dtype) -> None:
        """Buffer the surviving points and republish the decayed history.

        Modelled on RViz2's Decay Time, with one difference forced by running
        headless: RViz re-transforms its whole buffer into the fixed frame at
        render time, so it can afford to store points in their original frames.
        Here each cloud is rewritten into `decay_frame` once, on arrival, and
        stored ready to concatenate.
        """
        # Concatenation demands one stride and one field layout across the
        # buffer, so a publisher changing layout mid-run has to flush it.
        layout = tuple((f.name, f.offset, f.datatype) for f in msg.fields) + (point_step,)
        if self._decay_layout is not None and layout != self._decay_layout:
            self.get_logger().warn(
                "Cloud layout changed; flushing the decay buffer",
                throttle_duration_sec=10.0)
            self._decay_buf.clear()
            self._decay_points = 0
        self._decay_layout = layout

        rows = np.frombuffer(payload, dtype=np.uint8).reshape(n_points, point_step)
        kept = rows[keep].copy()

        if kept.shape[0]:
            view = np.frombuffer(payload, dtype=xyz_dtype, count=n_points)
            xyz = np.stack([view['x'], view['y'], view['z']], axis=1).astype(np.float64)
            xyz_kept = xyz[keep]

            decayed = None
            if self.deskew:
                decayed = self._deskew(msg, payload, point_step, n_points,
                                       keep, xyz_kept)
                if decayed is None:
                    self._deskew_failed += 1
                else:
                    self._deskewed += 1

            if decayed is None:
                # One transform for the whole sweep. Leaves intra-sweep smear
                # in, which is what deskewing exists to remove. Looked up
                # lazily: when deskewing succeeds this is dead weight, and
                # every TF lookup is a potential block.
                transform = self._lookup_transform(self.decay_frame, msg.header)
                if transform is None:
                    return
                rotation, translation = transform
                decayed = xyz_kept @ rotation.T + translation

            writable = kept.view(xyz_dtype).reshape(-1)
            writable['x'] = decayed[:, 0]
            writable['y'] = decayed[:, 1]
            writable['z'] = decayed[:, 2]

        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self._decay_buf.append((stamp, kept))
        self._decay_points += int(kept.shape[0])

        # Age out. Compared against this cloud's own stamp rather than the wall
        # clock, so the window follows the data -- which matters because the raw
        # feed is stamped from the robot's clock, not this node's.
        cutoff = stamp - self.decay_time
        while self._decay_buf and self._decay_buf[0][0] < cutoff:
            self._decay_points -= int(self._decay_buf.popleft()[1].shape[0])

        # Then the hard cap, oldest first.
        while self._decay_buf and self._decay_points > self.decay_max_points:
            self._decay_points -= int(self._decay_buf.popleft()[1].shape[0])

        if not self._decay_buf:
            return

        merged = (self._decay_buf[0][1] if len(self._decay_buf) == 1
                  else np.concatenate([r for _, r in self._decay_buf], axis=0))

        out = PointCloud2()
        out.header.stamp = msg.header.stamp
        # Always decay_frame, whatever publish_frame says: the accumulated cloud
        # spans many sensor poses, so no single sensor frame describes it.
        out.header.frame_id = self.decay_frame
        out.height = 1
        out.width = int(merged.shape[0])
        out.fields = msg.fields
        out.is_bigendian = msg.is_bigendian
        out.point_step = point_step
        out.row_step = point_step * out.width
        out.is_dense = msg.is_dense
        out.data = merged.tobytes()
        self.publisher.publish(out)

    # -- diagnostics -------------------------------------------------------

    def _report_stats(self) -> None:
        if self._msgs == 0:
            self.get_logger().warn(
                f"No clouds masked in the last {self.stats_period:.0f}s. Check the "
                f"'cloud_in' remapping and that the publisher is BEST_EFFORT."
                + (f" {self._dropped_msgs} dropped." if self._dropped_msgs else ""),
                throttle_duration_sec=15.0)
            return

        retained = 100.0 * self._points_out / max(self._points_in, 1)
        line = (f"masked {self._msgs} clouds | kept {self._points_out}/"
                f"{self._points_in} ({retained:.1f}%)")

        # Percentiles of what survived, in the same 5/50/95 form as the
        # cross-robot tables in SWARM_SLAM_FINDINGS.md section 4. Run this on
        # both robots and tune until the distributions overlap.
        if self._last_retained and self._last_retained['z'].size:
            for label, values in self._last_retained.items():
                p5, p50, p95 = np.percentile(values, [5, 50, 95])
                line += f" | {label} 5/50/95: {p5:.2f}/{p50:.2f}/{p95:.2f}"
        else:
            line += " | nothing retained in the last cloud"

        if self.decay_time > 0.0:
            line += (f" | decay {self._decay_points} pts over "
                     f"{len(self._decay_buf)} clouds")
            if self.deskew:
                line += f" | deskew {self._deskewed} ok/{self._deskew_failed} fell back"

        if self._tf_stale:
            skew = ('' if self._tf_skew_last is None
                    else f", last skew {self._tf_skew_last:+.3f}s")
            line += f" | TF stale x{self._tf_stale}{skew}"

        if self._dropped_msgs:
            line += f" | {self._dropped_msgs} dropped"

        self.get_logger().info(line)
        self._msgs = self._points_in = self._points_out = self._dropped_msgs = 0
        self._tf_stale = self._deskewed = self._deskew_failed = 0

    def _log_configuration(self) -> None:
        def unbounded(v):
            return '<none>' if abs(v) >= UNBOUNDED else f'{v:.2f}'

        self.get_logger().info("FOV Mask Configuration:")
        self.get_logger().info(f"   Mask frame (axes, z datum): {self.mask_frame}")
        self.get_logger().info(
            f"   Mask origin (range/azim/elev measured from): {self.mask_origin}")
        self.get_logger().info(f"   Publish frame: {self.publish_frame}")
        self.get_logger().info(
            f"   z band: [{unbounded(self.z_min)}, {unbounded(self.z_max)}]")
        self.get_logger().info(
            f"   range ({self.range_mode}): "
            f"[{self.min_range:.2f}, {unbounded(self.max_range)}]")
        blank = ('<off>' if self.sensor_blank_radius <= 0.0
                 else f'{self.sensor_blank_radius:.3f} m')
        self.get_logger().info(f"   sensor blank radius: {blank}")
        self.get_logger().info(
            f"   elevation: [{math.degrees(self.elev_min):.1f}, "
            f"{math.degrees(self.elev_max):.1f}] deg")
        if self.decay_time > 0.0:
            self.get_logger().info(
                f"   decay: {self.decay_time:.2f} s in '{self.decay_frame}', "
                f"cap {self.decay_max_points} pts -> cloud_processed")
            self.get_logger().info(
                f"   deskew: {'on' if self.deskew else 'off'}"
                f" | stale TF: {'allowed' if self.tf_allow_stale else 'dropped'}")
        else:
            self.get_logger().info("   decay: <off> (cloud_processed is per-scan)")
        self.get_logger().info(
            f"   azimuth: [{math.degrees(self.azim_min):.1f}, "
            f"{math.degrees(self.azim_max):.1f}] deg")


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Vectorised SLERP between two xyzw quaternions. Returns (N, 4)."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        # Quaternions double-cover rotations; flip so the interpolation takes
        # the short way round instead of spinning nearly all the way about.
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        # Nearly parallel: SLERP is numerically unstable here and lerp is
        # indistinguishable at these angles.
        q = q0[None, :] + alpha[:, None] * (q1 - q0)[None, :]
    else:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        q = (np.sin((1.0 - alpha) * theta)[:, None] * q0[None, :]
             + np.sin(alpha * theta)[:, None] * q1[None, :]) / math.sin(theta)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _quats_to_matrices(q: np.ndarray) -> np.ndarray:
    """(N, 4) xyzw -> (N, 3, 3). Assumes unit quaternions, as _slerp returns."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.empty((q.shape[0], 3, 3))
    m[:, 0, 0] = 1 - 2 * (yy + zz)
    m[:, 0, 1] = 2 * (xy - wz)
    m[:, 0, 2] = 2 * (xz + wy)
    m[:, 1, 0] = 2 * (xy + wz)
    m[:, 1, 1] = 1 - 2 * (xx + zz)
    m[:, 1, 2] = 2 * (yz - wx)
    m[:, 2, 0] = 2 * (xz - wy)
    m[:, 2, 1] = 2 * (yz + wx)
    m[:, 2, 2] = 1 - 2 * (xx + yy)
    return m


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Rotation matrix from a quaternion, without pulling in scipy."""
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        return np.eye(3)
    s = 2.0 / norm
    xs, ys, zs = x * s, y * s, z * s
    return np.array([
        [1.0 - (y * ys + z * zs), x * ys - w * zs,         x * zs + w * ys],
        [x * ys + w * zs,         1.0 - (x * xs + z * zs), y * zs - w * xs],
        [x * zs - w * ys,         y * zs + w * xs,         1.0 - (x * xs + y * ys)],
    ])


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)

    node = None
    try:
        node = FovMaskNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error running fov mask node: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
