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
ELEV_ORIGINS = ('sensor', 'mask_frame')


class FovMaskNode(Node):
    """Masks a PointCloud2 down to a configurable region of the world."""

    def __init__(self):
        super().__init__('fov_mask_node')

        self._declare_parameters()
        self._read_parameters()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # BEST_EFFORT on both sides: it is what both robots' lidar drivers
        # publish, and a reliable subscription would silently receive nothing.
        cloud_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Topic names are deliberately generic and remapped at launch, matching
        # how this stack wires pointcloud_to_laserscan (cloud_in -> scan).
        self.publisher = self.create_publisher(
            PointCloud2, 'cloud_masked', cloud_qos,
            qos_overriding_options=QoSOverridingOptions.with_default_policies())
        self.create_subscription(
            PointCloud2, 'cloud_in', self._on_cloud, cloud_qos)

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
                    description='Gravity-aligned frame the mask is evaluated in. '
                                'Must be the physically equivalent frame on both '
                                'robots for the retained clouds to be comparable.')),
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

                # -- elevation cone --------------------------------------------
                ('elev_min_deg', -90.0, ParameterDescriptor(
                    description='Keep points at least this elevation above horizontal, '
                                'in degrees, measured in the gravity-aligned '
                                'mask_frame. Use to emulate the cone one sensor '
                                'physically cannot see beyond on the other robot.')),
                ('elev_max_deg', 90.0, ParameterDescriptor(
                    description='Keep points at most this elevation, in degrees.')),
                ('elev_origin', 'sensor', ParameterDescriptor(
                    description="Where elevation is measured from. 'sensor' uses the "
                                "cloud frame's origin expressed in mask_frame, making "
                                "the band a true field of view. 'mask_frame' measures "
                                "from that frame's origin instead.")),

                # -- azimuth sector --------------------------------------------
                ('azim_min_deg', -180.0, ParameterDescriptor(
                    description='Keep points at or beyond this yaw, in degrees, in '
                                'mask_frame. If azim_min > azim_max the sector is '
                                'treated as wrapping through +/-180.')),
                ('azim_max_deg', 180.0, ParameterDescriptor(
                    description='Keep points at or below this yaw, in degrees.')),

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
        self.publish_frame = get('publish_frame').get_parameter_value().string_value
        self.z_min = get('z_min').get_parameter_value().double_value
        self.z_max = get('z_max').get_parameter_value().double_value
        self.min_range = get('min_range').get_parameter_value().double_value
        self.max_range = get('max_range').get_parameter_value().double_value
        self.range_mode = get('range_mode').get_parameter_value().string_value
        self.elev_min = math.radians(get('elev_min_deg').get_parameter_value().double_value)
        self.elev_max = math.radians(get('elev_max_deg').get_parameter_value().double_value)
        self.elev_origin = get('elev_origin').get_parameter_value().string_value
        self.azim_min = math.radians(get('azim_min_deg').get_parameter_value().double_value)
        self.azim_max = math.radians(get('azim_max_deg').get_parameter_value().double_value)
        self.tf_timeout = get('tf_timeout').get_parameter_value().double_value
        self.stats_period = get('stats_period').get_parameter_value().double_value

        for name, value, valid in (
            ('range_mode', self.range_mode, RANGE_MODES),
            ('publish_frame', self.publish_frame, PUBLISH_FRAMES),
            ('elev_origin', self.elev_origin, ELEV_ORIGINS),
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

            transform = self._lookup_transform(msg)
            if transform is None:
                self._dropped_msgs += 1
                return
            rotation, translation = transform

            view = np.frombuffer(payload, dtype=xyz_dtype, count=n_points)
            xyz = np.stack([view['x'], view['y'], view['z']], axis=1).astype(np.float64)

            # Into the gravity-aligned frame, where the band actually means
            # something consistent across two differently-mounted sensors.
            masked_xyz = xyz @ rotation.T + translation

            keep, metrics = self._evaluate_mask(masked_xyz, translation)

            self._msgs += 1
            self._points_in += n_points
            self._points_out += int(keep.sum())
            self._last_retained = {k: v[keep] for k, v in metrics.items()}

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

    def _lookup_transform(self, msg: PointCloud2):
        """(3x3 rotation, 3-vector translation) taking cloud frame -> mask_frame."""
        source = msg.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                self.mask_frame, source, Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=self.tf_timeout))
        except TransformException:
            # Falling back to the latest available transform. Correct for the
            # static sensor->base edge; a small lag on a dynamic one, which
            # beats dropping the scan outright.
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.mask_frame, source, Time())
            except TransformException as exc:
                self.get_logger().warn(
                    f"TF {self.mask_frame} <- {source} unavailable: {exc}",
                    throttle_duration_sec=10.0)
                return None

        t = tf.transform.translation
        q = tf.transform.rotation
        return _quat_to_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])

    def _evaluate_mask(self, pts: np.ndarray, sensor_origin: np.ndarray):
        """Boolean keep-mask plus the per-point metrics, for stats."""
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        horizontal = np.hypot(x, y)
        rng = horizontal if self.range_mode == 'horizontal' else np.linalg.norm(pts, axis=1)

        # Elevation from the sensor's optical centre makes the band a true field
        # of view; from the mask frame's origin it is a band of the world.
        if self.elev_origin == 'sensor':
            ex, ey, ez = x - sensor_origin[0], y - sensor_origin[1], z - sensor_origin[2]
        else:
            ex, ey, ez = x, y, z
        elevation = np.arctan2(ez, np.hypot(ex, ey))
        azimuth = np.arctan2(y, x)

        keep = np.isfinite(pts).all(axis=1)
        keep &= (z >= self.z_min) & (z <= self.z_max)
        keep &= (rng >= self.min_range) & (rng <= self.max_range)
        keep &= (elevation >= self.elev_min) & (elevation <= self.elev_max)

        if self.azim_min <= self.azim_max:
            keep &= (azimuth >= self.azim_min) & (azimuth <= self.azim_max)
        else:
            # Sector wrapping through +/-180.
            keep &= (azimuth >= self.azim_min) | (azimuth <= self.azim_max)

        return keep, {'z': z, 'range': rng, 'elev': np.degrees(elevation)}

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

        if self._dropped_msgs:
            line += f" | {self._dropped_msgs} dropped"

        self.get_logger().info(line)
        self._msgs = self._points_in = self._points_out = self._dropped_msgs = 0

    def _log_configuration(self) -> None:
        unbounded = lambda v: '<none>' if abs(v) >= UNBOUNDED else f'{v:.2f}'
        self.get_logger().info("FOV Mask Configuration:")
        self.get_logger().info(f"   Mask frame: {self.mask_frame}")
        self.get_logger().info(f"   Publish frame: {self.publish_frame}")
        self.get_logger().info(
            f"   z band: [{unbounded(self.z_min)}, {unbounded(self.z_max)}]")
        self.get_logger().info(
            f"   range ({self.range_mode}): "
            f"[{self.min_range:.2f}, {unbounded(self.max_range)}]")
        self.get_logger().info(
            f"   elevation from {self.elev_origin}: "
            f"[{math.degrees(self.elev_min):.1f}, {math.degrees(self.elev_max):.1f}] deg")
        self.get_logger().info(
            f"   azimuth: [{math.degrees(self.azim_min):.1f}, "
            f"{math.degrees(self.azim_max):.1f}] deg")


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
