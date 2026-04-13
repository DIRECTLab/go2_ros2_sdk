#!/usr/bin/env python3
"""
go2_cslam_bridge: pointcloud_bridge node

Bridges the Go2 SDK to Swarm-SLAM by:
  1. Transforming the point cloud from the 'odom' frame it arrives in to the
     robot body frame ('base_link') that ScanContext / Swarm-SLAM expects.
  2. Relaying odometry so the topics land in the /r{N}/ namespace that
     Swarm-SLAM uses.

Parameters
----------
robot_id : int
    Swarm-SLAM robot index (0, 1, 2...).  Used to build /r{N}/ topics.
go2_prefix : str
    Prefix the Go2 SDK uses for this robot's topics.
    Single-robot mode  → '' (empty, topics are just 'odom', 'point_cloud2')
    Multi-robot mode   → 'robot0', 'robot1', ...
body_frame : str
    TF frame name for the robot body. Matches Go2 SDK default:
    single → 'base_link', multi → 'robot{N}/base_link'.
tf_timeout_sec : float
    How long to wait for a TF transform before dropping a cloud.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


# Best-effort QoS matches the Go2 SDK's lidar publisher
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class PointcloudBridge(Node):

    def __init__(self):
        super().__init__('go2_cslam_bridge')

        self.declare_parameter('robot_id', 0)
        self.declare_parameter('go2_prefix', '')
        self.declare_parameter('body_frame', 'base_link')
        self.declare_parameter('tf_timeout_sec', 0.15)

        robot_id = self.get_parameter('robot_id').value
        go2_prefix = self.get_parameter('go2_prefix').value
        self.body_frame = self.get_parameter('body_frame').value
        self.tf_timeout = Duration(
            seconds=self.get_parameter('tf_timeout_sec').value
        )

        # Source topics (Go2 SDK)
        prefix = f'{go2_prefix}/' if go2_prefix else ''
        src_pc_topic = f'{prefix}point_cloud2'
        src_odom_topic = f'{prefix}odom'

        # Destination topics (Swarm-SLAM namespace /r{N}/)
        dst_pc_topic = f'/r{robot_id}/pointcloud'
        dst_odom_topic = f'/r{robot_id}/odom'

        self.get_logger().info(
            f'Bridging robot {robot_id}: '
            f'{src_pc_topic} -> {dst_pc_topic} (transformed to {self.body_frame}), '
            f'{src_odom_topic} -> {dst_odom_topic}'
        )

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Point cloud: subscribe best-effort (matches SDK), publish reliable
        self.pc_pub = self.create_publisher(PointCloud2, dst_pc_topic, 10)
        self.pc_sub = self.create_subscription(
            PointCloud2, src_pc_topic, self._on_pointcloud, _SENSOR_QOS
        )

        # Odometry: relay as-is
        self.odom_pub = self.create_publisher(Odometry, dst_odom_topic, 10)
        self.odom_sub = self.create_subscription(
            Odometry, src_odom_topic, self._on_odom, 10
        )

    # ------------------------------------------------------------------

    def _on_pointcloud(self, msg: PointCloud2) -> None:
        """Transform cloud from odom frame into body frame, then republish."""
        src_frame = msg.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                self.body_frame,
                src_frame,
                msg.header.stamp,
                self.tf_timeout,
            )
        except tf2_ros.TransformException as exc:
            # Retry with latest available transform before giving up
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.body_frame,
                    src_frame,
                    rclpy.time.Time(),
                )
            except tf2_ros.TransformException:
                self.get_logger().warn(
                    f'TF lookup failed ({src_frame} -> {self.body_frame}): {exc}',
                    throttle_duration_sec=5.0,
                )
                return

        transformed = do_transform_cloud(msg, tf)
        self.pc_pub.publish(transformed)

    def _on_odom(self, msg: Odometry) -> None:
        """Relay odometry unchanged."""
        self.odom_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PointcloudBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
