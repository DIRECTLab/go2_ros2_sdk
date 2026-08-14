# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

"""
Raw LiDAR Node

Republishes the Go2's onboard L2 lidar feed -- taken straight off the robot's
``rt/utlidar/cloud`` CycloneDDS topic over Ethernet -- as a standard
sensor_msgs/PointCloud2.

This is a separate path from the driver's WebRTC point cloud
(`ros2_publisher.publish_lidar_data`), on purpose. The WebRTC channel carries
`rt/utlidar/voxel_map_compressed`, whose points are quantised to a 5 cm lattice
on the robot before transport; this node gives the true per-scan sampling that
registration front-ends (KISS-ICP, cslam's FPFH/TEASER) need.

The node publishes no TF. The Unitree stack's own dynamic chain
(unilidar_imu_initial -> unilidar_imu -> unilidar_lidar) will fight static
transforms and EKF/odometry stacks if both claim overlapping frames, so `frame_id`
is a parameter here and TF is left entirely to whatever already owns the tree.
"""

import time as _time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos_overriding_options import QoSOverridingOptions
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import PointCloud2, PointField

from ..infrastructure.dds.utlidar_cloud_subscriber import (
    CONFIRMED_FIELD_LAYOUT,
    CONFIRMED_POINT_STEP,
    DdsUnavailableError,
    UtlidarCloudSubscriber,
    cloud_data_bytes,
    cloud_field_layout,
    cloud_stamp_ns,
)

# PointField enum value -> numpy dtype, for reading the per-point 'time' column
# back out of the payload without unpacking the whole cloud.
_POINTFIELD_NUMPY = {
    1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2',
    5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8',
}

STAMP_SOURCES = ('raw', 'raw_header', 'receive')


class RawLidarNode(Node):
    """Bridges rt/utlidar/cloud (CycloneDDS) to a ROS2 PointCloud2 topic."""

    def __init__(self):
        super().__init__('raw_lidar_node')

        self._declare_parameters()
        self._read_parameters()

        # BEST_EFFORT / KEEP_LAST matches the driver's own point cloud publisher
        # and what RViz2, nav2 and SLAM consumers default to for lidar.
        lidar_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.publisher = self.create_publisher(
            PointCloud2, self.output_topic, lidar_qos,
            qos_overriding_options=QoSOverridingOptions.with_default_policies())

        # Prebuilt from the confirmed layout: identical on every message, so
        # there is no reason to rebuild it per scan.
        self.fields = self._build_fields(CONFIRMED_FIELD_LAYOUT)
        self.point_step = CONFIRMED_POINT_STEP
        self._time_reader = self._build_time_reader(CONFIRMED_FIELD_LAYOUT,
                                                    CONFIRMED_POINT_STEP)

        self._cloud_count = 0
        self._dropped_count = 0
        self._last_cloud_monotonic: Optional[float] = None
        self._layout_logged = False
        self._layout_warned = False

        self.subscriber = UtlidarCloudSubscriber(
            network_interface=self.network_interface,
            domain_id=self.dds_domain_id,
            topic=self.dds_topic,
            queue_len=self.queue_len,
        )

        self._log_configuration()

        try:
            self.subscriber.start(self._on_dds_cloud)
        except DdsUnavailableError as exc:
            self.get_logger().error(f"Raw lidar subscription failed: {exc}")
            raise

        if self.stale_timeout > 0.0:
            self._watchdog = self.create_timer(self.stale_timeout, self._check_stale)

    # -- setup -------------------------------------------------------------

    def _declare_parameters(self) -> None:
        self.declare_parameters(
            namespace='',
            parameters=[
                ('network_interface', '', ParameterDescriptor(
                    description='Ethernet interface facing the robot, e.g. "eth0". '
                                'Empty means every interface.')),
                ('dds_domain_id', 0, ParameterDescriptor(
                    description='CycloneDDS domain id the robot publishes on.')),
                ('dds_topic', 'rt/utlidar/cloud', ParameterDescriptor(
                    description='DDS topic carrying the raw per-scan cloud.')),
                ('output_topic', 'raw_lidar', ParameterDescriptor(
                    description='ROS2 topic to republish on. Relative names take the '
                                'node namespace (/go2/raw_lidar under this repo\'s '
                                'launch files); pass an absolute name such as '
                                '/r0/raw_lidar to place it where Swarm-SLAM expects '
                                'a given robot\'s data.')),
                ('frame_id', 'utlidar_lidar', ParameterDescriptor(
                    description='frame_id stamped on the published cloud. This node '
                                'publishes no TF -- the frame must already exist in '
                                'the tree someone else owns.')),
                ('stamp_source', 'raw', ParameterDescriptor(
                    description="Header stamp basis: 'raw' (robot's DDS header stamp "
                                "shifted by the earliest per-point time offset), "
                                "'raw_header' (that stamp verbatim), or 'receive' "
                                "(this node's clock -- adds network and scheduling "
                                "jitter, so it corrupts registration timing).")),
                ('queue_len', 10, ParameterDescriptor(
                    description='DDS reader queue depth.')),
                ('verify_layout', True, ParameterDescriptor(
                    description='Compare each incoming message against the confirmed '
                                'field layout and warn when it drifts.')),
                ('stale_timeout', 5.0, ParameterDescriptor(
                    description='Warn if no cloud arrives for this many seconds. '
                                '0 disables the check.')),
            ]
        )

    def _read_parameters(self) -> None:
        get = self.get_parameter
        self.network_interface = get('network_interface').get_parameter_value().string_value
        self.dds_domain_id = get('dds_domain_id').get_parameter_value().integer_value
        self.dds_topic = get('dds_topic').get_parameter_value().string_value
        self.output_topic = get('output_topic').get_parameter_value().string_value
        self.frame_id = get('frame_id').get_parameter_value().string_value
        self.stamp_source = get('stamp_source').get_parameter_value().string_value
        self.queue_len = get('queue_len').get_parameter_value().integer_value
        self.verify_layout = get('verify_layout').get_parameter_value().bool_value
        self.stale_timeout = get('stale_timeout').get_parameter_value().double_value

        if self.stamp_source not in STAMP_SOURCES:
            self.get_logger().warn(
                f"Unknown stamp_source '{self.stamp_source}', falling back to 'raw'. "
                f"Valid values: {', '.join(STAMP_SOURCES)}")
            self.stamp_source = 'raw'

    @staticmethod
    def _build_fields(layout: Tuple[Tuple[str, int, int], ...]) -> List[PointField]:
        return [
            PointField(name=name, offset=offset, datatype=datatype, count=1)
            for name, offset, datatype in layout
        ]

    @staticmethod
    def _build_time_reader(layout: Tuple[Tuple[str, int, int], ...],
                           point_step: int):
        """Return a callable(bytes, n_points) -> np.ndarray of per-point times.

        Built as a one-column structured view so reading the time offsets costs a
        strided read rather than unpacking all 32 bytes per point.
        """
        for name, offset, datatype in layout:
            if name != 'time':
                continue
            numpy_type = _POINTFIELD_NUMPY.get(datatype)
            if numpy_type is None:
                return None
            dtype = np.dtype({
                'names': ['time'],
                'formats': ['<' + numpy_type],
                'offsets': [offset],
                'itemsize': point_step,
            })

            def read(payload: bytes, count: int, _dtype=dtype) -> np.ndarray:
                return np.frombuffer(payload, dtype=_dtype, count=count)['time']

            return read
        return None

    # -- DDS callback ------------------------------------------------------

    def _on_dds_cloud(self, msg) -> None:
        """Called on a CycloneDDS listener thread for each incoming scan."""
        try:
            payload = cloud_data_bytes(msg)
            # Plain attributes on the Python IDL types -- msg.width, not msg.width().
            width = int(msg.width)
            height = int(msg.height)
            incoming_point_step = int(msg.point_step)

            if self.verify_layout:
                self._check_layout(msg, incoming_point_step)

            if incoming_point_step != self.point_step:
                # The payload is copied through byte-for-byte and relabelled with
                # the confirmed layout. A different stride makes that relabelling
                # provably wrong, so drop rather than publish nonsense.
                self._dropped_count += 1
                self.get_logger().error(
                    f"Dropping cloud: point_step is {incoming_point_step}, expected "
                    f"{self.point_step}. The raw feed's layout changed; re-verify it "
                    f"before trusting this topic. ({self._dropped_count} dropped)",
                    throttle_duration_sec=10.0)
                return

            n_points = width * height
            if n_points <= 0 or len(payload) < n_points * self.point_step:
                self._dropped_count += 1
                self.get_logger().warn(
                    f"Dropping malformed cloud: {width}x{height} points but "
                    f"{len(payload)} payload bytes",
                    throttle_duration_sec=10.0)
                return

            cloud = PointCloud2()
            cloud.header.frame_id = self.frame_id
            cloud.header.stamp = self._resolve_stamp(msg, payload, n_points)
            # height is always 1 on this sensor: it is a non-repetitive scanner,
            # so width varies per scan (~3600-4000). Not a defect.
            cloud.height = height
            cloud.width = width
            cloud.fields = self.fields
            cloud.is_bigendian = bool(getattr(msg, 'is_bigendian', False))
            cloud.point_step = self.point_step
            cloud.row_step = self.point_step * width
            cloud.data = payload
            cloud.is_dense = bool(getattr(msg, 'is_dense', False))

            self.publisher.publish(cloud)

            self._cloud_count += 1
            self._last_cloud_monotonic = _time.monotonic()
            if not self._layout_logged:
                self._layout_logged = True
                self.get_logger().info(
                    f"First cloud published: {width} points, point_step "
                    f"{incoming_point_step}, frame '{self.frame_id}' -> "
                    f"'{self.output_topic}'")

        except Exception as exc:
            self.get_logger().error(f"Error republishing raw lidar cloud: {exc}",
                                    throttle_duration_sec=5.0)

    def _check_layout(self, msg, incoming_point_step: int) -> None:
        """Warn once if the message stops matching the layout confirmed on this robot."""
        if self._layout_warned:
            return

        incoming = cloud_field_layout(msg)
        if not incoming:
            return  # message declares no fields; nothing to compare against

        expected = list(CONFIRMED_FIELD_LAYOUT)
        if incoming == expected and incoming_point_step == CONFIRMED_POINT_STEP:
            return

        self._layout_warned = True
        self.get_logger().warn(
            "Raw lidar field layout differs from the layout confirmed on this "
            "robot. Published clouds relabel the payload with the confirmed "
            "layout, so they will be wrong until this is re-verified.\n"
            f"  expected: point_step={CONFIRMED_POINT_STEP} {expected}\n"
            f"  received: point_step={incoming_point_step} {incoming}")

    # -- stamping ----------------------------------------------------------

    def _resolve_stamp(self, msg, payload: bytes, n_points: int):
        """Build the header stamp according to stamp_source.

        Note the per-point 'time' field cannot itself be an absolute timestamp:
        it is FLOAT32, whose 24-bit mantissa quantises epoch seconds (~1.7e9) to
        roughly 128 s steps. It is an intra-scan offset, so the absolute basis
        has to come from the DDS message's own header stamp, with the earliest
        per-point offset applied on top -- the reference instant deskewing
        front-ends assume.
        """
        if self.stamp_source == 'receive':
            return self.get_clock().now().to_msg()

        stamp_ns = cloud_stamp_ns(msg)
        if stamp_ns is None:
            self.get_logger().warn(
                "DDS message carries no header stamp; falling back to receive time",
                throttle_duration_sec=30.0)
            return self.get_clock().now().to_msg()

        if self.stamp_source == 'raw':
            offset_ns = self._first_point_offset_ns(payload, n_points)
            if offset_ns is not None and stamp_ns + offset_ns >= 0:
                stamp_ns += offset_ns
            elif offset_ns is not None:
                self.get_logger().warn(
                    f"Per-point time offset ({offset_ns} ns) would push the stamp "
                    "negative; using the unshifted header stamp instead",
                    throttle_duration_sec=30.0)

        return Time(nanoseconds=int(stamp_ns)).to_msg()

    def _first_point_offset_ns(self, payload: bytes, n_points: int) -> Optional[int]:
        """Earliest per-point time offset in the scan, in nanoseconds.

        min() rather than the first point's value, since firmware differs on
        whether offsets run from scan start (>= 0) or scan end (<= 0).
        """
        if self._time_reader is None:
            return None
        try:
            times = self._time_reader(payload, n_points)
            if times.size == 0:
                return None
            earliest = float(times.min())
        except Exception as exc:
            self.get_logger().warn(
                f"Could not read per-point time offsets: {exc}",
                throttle_duration_sec=30.0)
            return None

        if not np.isfinite(earliest):
            return None
        return int(round(earliest * 1e9))

    # -- diagnostics -------------------------------------------------------

    def _check_stale(self) -> None:
        if self._last_cloud_monotonic is None:
            self.get_logger().warn(
                f"No cloud received on '{self.dds_topic}' yet. Check the Ethernet "
                f"link and that network_interface ('{self.network_interface or '<all>'}') "
                f"is the one facing the robot.", throttle_duration_sec=15.0)
            return

        silent_for = _time.monotonic() - self._last_cloud_monotonic
        if silent_for > self.stale_timeout:
            self.get_logger().warn(
                f"No cloud on '{self.dds_topic}' for {silent_for:.1f}s "
                f"({self._cloud_count} received so far)",
                throttle_duration_sec=15.0)

    def _log_configuration(self) -> None:
        self.get_logger().info("Raw LiDAR (CycloneDDS) Configuration:")
        self.get_logger().info(f"   Network interface: {self.network_interface or '<all>'}")
        self.get_logger().info(f"   DDS domain: {self.dds_domain_id}")
        self.get_logger().info(f"   DDS topic: {self.dds_topic}")
        self.get_logger().info(f"   Output topic: {self.output_topic}")
        self.get_logger().info(f"   Frame id: {self.frame_id}")
        self.get_logger().info(f"   Stamp source: {self.stamp_source}")

    def destroy_node(self) -> bool:
        self.subscriber.stop()
        return super().destroy_node()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)

    node = None
    try:
        node = RawLidarNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error running raw lidar node: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
