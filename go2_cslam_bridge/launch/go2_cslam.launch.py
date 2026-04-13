"""
go2_cslam.launch.py  —  Run Swarm-SLAM on one Go2 robot.

Run this on each robot (or each robot's machine).  Set ROBOT_ID to the
zero-based index of this robot in the swarm.

Usage (single robot, Go2 SDK in single mode):
    ros2 launch go2_cslam_bridge go2_cslam.launch.py robot_id:=0 go2_prefix:='' body_frame:=base_link

Usage (multi-robot from one machine, Go2 SDK in multi mode):
    ros2 launch go2_cslam_bridge go2_cslam.launch.py robot_id:=0 go2_prefix:=robot0 body_frame:=robot0/base_link
    ros2 launch go2_cslam_bridge go2_cslam.launch.py robot_id:=1 go2_prefix:=robot1 body_frame:=robot1/base_link

Inter-robot communication
    Swarm-SLAM uses DDS multicast for peer discovery.  All robots must share
    the same ROS_DOMAIN_ID (default 0) and be reachable on the same network.
    If you are on separate subnets, run experiment_lidar.launch.py from
    cslam_experiments instead — it sets up the Zenoh DDS bridge automatically.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    robot_id   = LaunchConfiguration('robot_id').perform(context)
    go2_prefix = LaunchConfiguration('go2_prefix').perform(context)
    body_frame = LaunchConfiguration('body_frame').perform(context)
    max_nb_robots = LaunchConfiguration('max_nb_robots').perform(context)
    config_file   = LaunchConfiguration('config_file').perform(context)

    cslam_config_path = os.path.join(
        get_package_share_directory('go2_cslam_bridge'), 'config'
    )
    cslam_experiments_launch = os.path.join(
        get_package_share_directory('cslam_experiments'),
        'launch', 'cslam', 'cslam_lidar.launch.py'
    )

    # ------------------------------------------------------------------ #
    # 1. Bridge node: transforms point cloud odom→body and relays odom    #
    # ------------------------------------------------------------------ #
    bridge_node = Node(
        package='go2_cslam_bridge',
        executable='pointcloud_bridge',
        name='go2_cslam_bridge',
        output='screen',
        parameters=[{
            'robot_id':       int(robot_id),
            'go2_prefix':     go2_prefix,
            'body_frame':     body_frame,
            'tf_timeout_sec': 0.15,
        }],
    )

    # ------------------------------------------------------------------ #
    # 2. Swarm-SLAM lidar stack for this robot                            #
    # ------------------------------------------------------------------ #
    cslam_proc = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(cslam_experiments_launch),
        launch_arguments={
            'config_path':    cslam_config_path + '/',
            'config_file':    config_file,
            'robot_id':       robot_id,
            'namespace':      f'/r{robot_id}',
            'max_nb_robots':  max_nb_robots,
        }.items(),
    )

    return [bridge_node, cslam_proc]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_id', default_value='0',
            description='Zero-based robot index in the swarm'
        ),
        DeclareLaunchArgument(
            'max_nb_robots', default_value='5',
            description='Total number of robots in the swarm'
        ),
        DeclareLaunchArgument(
            'go2_prefix', default_value='',
            description=(
                "Topic prefix the Go2 SDK uses for this robot. "
                "Empty string for single-robot SDK mode; "
                "'robot0', 'robot1', ... for multi-robot mode."
            ),
        ),
        DeclareLaunchArgument(
            'body_frame', default_value='base_link',
            description=(
                "TF body frame for this robot. "
                "Single-robot SDK mode: 'base_link'. "
                "Multi-robot SDK mode: 'robot0/base_link', etc."
            ),
        ),
        DeclareLaunchArgument(
            'config_file', default_value='go2_lidar.yaml',
            description='Swarm-SLAM config file name (looked up in go2_cslam_bridge/config/)'
        ),
        OpaqueFunction(function=launch_setup),
    ])
