"""Standalone bringup for an external Unitree L2.

Mirrors the rover's copy of this file (namespaced node, inline parameters) so
the same invocation works on both robots. Useful for bringing the sensor up on
its own -- checking the cable, the IP pair or the mount -- without starting the
whole Go2 stack.

For normal operation use go2_robot_sdk's robot.launch.py, which starts this
same node with tf_prefix-derived frame ids, publishes the mount transform and
wires the fov_mask node onto the output. This file publishes NO transform, so
the cloud has no place in the robot's TF tree.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


class LidarLauncher:
    def create_launch_arguments(self):
        return [
            DeclareLaunchArgument(
                'namespace', default_value='unilidar',
                description='Namespace to prepend to the topics'),
            # 1 = serial (USB CDC), 2 = UDP (ethernet). These are the SDK's own
            # initialize_type values, kept rather than renamed so they match
            # the vendor examples and the rover's parameters.
            DeclareLaunchArgument(
                'initialize_type', default_value='2',
                description='SDK transport: 1 = serial over USB, 2 = UDP over ethernet'),
            # work_mode is a bitfield the SDK writes to the sensor. 0 is the
            # vendor default; the rover runs 4, which suppresses the IMU
            # packets and therefore the driver's own TF broadcast -- see
            # docs/EXTERNAL_LIDAR.md in go2_robot_sdk.
            DeclareLaunchArgument(
                'work_mode', default_value='4',
                description='Lidar work mode written to the sensor at startup'),
            DeclareLaunchArgument(
                'serial_port', default_value='/dev/ttyACM0',
                description='Serial device, used when initialize_type is 1'),
            DeclareLaunchArgument(
                'lidar_ip', default_value='192.168.1.62',
                description="The sensor's own address, used when initialize_type is 2"),
            DeclareLaunchArgument(
                'local_ip', default_value='192.168.1.2',
                description='This host\'s address on the lidar subnet'),
            DeclareLaunchArgument(
                'cloud_frame', default_value='unilidar_lidar',
                description='frame_id stamped on the cloud'),
            DeclareLaunchArgument(
                'imu_frame', default_value='unilidar_imu',
                description="frame_id stamped on the sensor's IMU"),
            DeclareLaunchArgument(
                'range_max', default_value='30.0',
                description='Far clip in metres. The L2 spec is 30 m'),
        ]

    def create_lidar_node(self, context):
        # Resolved rather than passed as substitutions: the node declares these
        # as int and double, and launch substitutions always arrive as strings.
        initialize_type = int(LaunchConfiguration('initialize_type').perform(context))
        work_mode = int(LaunchConfiguration('work_mode').perform(context))
        range_max = float(LaunchConfiguration('range_max').perform(context))

        return Node(
            package='unitree_lidar_ros2',
            executable='unitree_lidar_ros2_node',
            namespace=LaunchConfiguration('namespace'),
            name='unitree_lidar_ros2_node',
            output='screen',
            parameters=[{
                'initialize_type': initialize_type,
                'work_mode': work_mode,
                'use_system_timestamp': True,
                'range_min': 0.0,
                'range_max': range_max,
                'cloud_scan_num': 18,

                'serial_port': LaunchConfiguration('serial_port'),
                'baudrate': 4000000,

                'lidar_port': 6101,
                'lidar_ip': LaunchConfiguration('lidar_ip'),
                'local_port': 6201,
                'local_ip': LaunchConfiguration('local_ip'),

                'cloud_frame': LaunchConfiguration('cloud_frame'),
                'cloud_topic': 'unilidar/cloud',
                'imu_frame': LaunchConfiguration('imu_frame'),
                'imu_topic': 'unilidar/imu',
            }],
        )


def generate_launch_description():
    launcher = LidarLauncher()
    return LaunchDescription([
        *launcher.create_launch_arguments(),
        OpaqueFunction(function=lambda context: [launcher.create_lidar_node(context)]),
    ])
