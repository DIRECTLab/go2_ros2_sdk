# Mapping launch file - optimized for SLAM and map creation
# Usage: ros2 launch go2_robot_sdk mapping.launch.py

import os
import tempfile
from typing import List
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources import FrontendLaunchDescriptionSource, PythonLaunchDescriptionSource


# Parameter keys whose value names a TF frame, across the nav2 and slam_toolbox
# yaml files this stack loads.
_FRAME_PARAM_KEYS = frozenset({
    'base_frame', 'map_frame', 'odom_frame',
    'base_frame_id', 'global_frame_id', 'odom_frame_id',
    'global_frame', 'robot_base_frame', 'target_frame',
})


def prefix_frame(tf_prefix: str, frame: str) -> str:
    """Namespace a frame id: '' -> 'base_link', 'go2' -> 'go2/base_link'."""
    return f'{tf_prefix}/{frame}' if tf_prefix else frame


def rewrite_frame_params(params_path: str, tf_prefix: str) -> str:
    """Copy a params yaml with every TF frame value prefixed, return the new path.

    Returns params_path untouched when tf_prefix is empty. A flat key->value
    substitution would be wrong here: nav2 reuses 'global_frame' under several
    nodes with different values ('map' for the global costmap, 'odom' for the
    local one), so each value is prefixed in place instead.
    """
    if not tf_prefix:
        return params_path

    with open(params_path, 'r') as handle:
        data = yaml.safe_load(handle)

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _FRAME_PARAM_KEYS and isinstance(value, str) and value:
                    node[key] = prefix_frame(tf_prefix, value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)

    rewritten = tempfile.NamedTemporaryFile(
        mode='w', prefix='go2_frames_', suffix='.yaml', delete=False)
    yaml.safe_dump(data, rewritten, default_flow_style=False)
    rewritten.close()
    return rewritten.name


def _launch_setup(context, *args, **kwargs):
    """Build the mapping-mode nodes, now that tf_prefix can be resolved"""

    tf_prefix = LaunchConfiguration('tf_prefix').perform(context).strip().strip('/')
    # robot_state_publisher prepends frame_prefix verbatim, so it needs the
    # trailing separator: '' -> '', 'go2' -> 'go2/'
    frame_prefix = f'{tf_prefix}/' if tf_prefix else ''

    # Environment variables
    robot_token = os.getenv('ROBOT_TOKEN', '')
    robot_ip = os.getenv('ROBOT_IP', '')
    robot_ip_list = robot_ip.replace(" ", "").split(",") if robot_ip else []
    map_name = os.getenv('MAP_NAME', 'my_map')
    save_map = os.getenv('MAP_SAVE', 'true')
    conn_type = os.getenv('CONN_TYPE', 'webrtc')
    
    # Determine connection mode
    conn_mode = "single" if len(robot_ip_list) == 1 and conn_type != "cyclonedds" else "multi"
    
    # Package paths
    package_dir = get_package_share_directory('go2_robot_sdk')
    urdf_file = 'go2.urdf' if conn_mode == 'single' else 'multi_go2.urdf'
    rviz_config = 'single_robot_conf.rviz' if conn_mode == 'single' else 'multi_robot_conf.rviz'
    
    config_paths = {
        'joystick': os.path.join(package_dir, 'config', 'joystick.yaml'),
        'twist_mux': os.path.join(package_dir, 'config', 'twist_mux.yaml'),
        'slam': os.path.join(package_dir, 'config', 'mapper_params_online_async.yaml'),
        'rviz': os.path.join(package_dir, 'config', rviz_config),
        'urdf': os.path.join(package_dir, 'urdf', urdf_file),
    }
    
    print(f"🗺️  Go2 Mapping Mode:")
    print(f"   Robot IPs: {robot_ip_list}")
    print(f"   Connection: {conn_type} ({conn_mode})")
    print(f"   Map name: {map_name}")
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    with_rviz = LaunchConfiguration('rviz', default='true')
    with_foxglove = LaunchConfiguration('foxglove', default='true')
    with_joystick = LaunchConfiguration('joystick', default='true')
    
    # Load URDF
    with open(config_paths['urdf'], 'r') as file:
        robot_desc = file.read()
    
    # Core nodes
    core_nodes = [
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='go2_robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_desc,
                'frame_prefix': frame_prefix
            }],
        ),
        # Main robot driver
        Node(
            package='go2_robot_sdk',
            executable='go2_driver_node',
            name='go2_driver_node',
            output='screen',
            parameters=[{
                'robot_ip': robot_ip,
                'token': robot_token,
                'conn_type': conn_type,
                'tf_prefix': tf_prefix
            }],
        ),
        # LiDAR processing node
        Node(
            package='lidar_processor_cpp',
            executable='lidar_to_pointcloud_node',
            name='lidar_to_pointcloud',
            remappings=[
                ('robot0/point_cloud2', 'point_cloud2'),
            ] if conn_mode == 'single' else [],
            parameters=[{
                'robot_ip_lst': robot_ip_list,
                'map_name': map_name,
                'map_save': save_map
            }],
        ),
        # Point cloud aggregator - maximized for full coverage
        Node(
            package='lidar_processor_cpp',
            executable='pointcloud_aggregator_node',
            name='pointcloud_aggregator',
            parameters=[{
                'max_range': 20.0,
                'min_range': 0.3,
                'height_filter_min': -1.0,
                'height_filter_max': 3.0,
                'downsample_rate': 1,
                'publish_rate': 20.0,
                'output_frame': prefix_frame(tf_prefix, 'base_link')
            }],
        ),
        # PointCloud to LaserScan converter - maximum coverage
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='go2_pointcloud_to_laserscan',
            remappings=[
                ('cloud_in', '/pointcloud/filtered'),
                ('scan', '/go2/scan'),
            ],
            parameters=[{
                'target_frame': prefix_frame(tf_prefix, 'base_link'),
                'max_height': 3.0,
                'min_height': -1.0,
                'angle_min': -3.14159,
                'angle_max': 3.14159,
                'angle_increment': 0.01745329,
                'scan_time': 0.1,
                'range_min': 0.3,
                'range_max': 20.0,
                'use_inf': True,
                'concurrency_level': 1,
            }],
            output='screen',
        ),
        # TTS Node
        Node(
            package='speech_processor',
            executable='tts_node',
            name='tts_node',
            parameters=[{
                'api_key': os.getenv('ELEVENLABS_API_KEY', ''),
                'provider': 'elevenlabs',
                'voice_name': 'XrExE9yKIg1WjnnlVkGX',
                'local_playback': False,
                'use_cache': True,
                'audio_quality': 'standard'
            }],
        ),
    ]
    
    # Teleop nodes
    teleop_nodes = [
        Node(
            package='joy',
            executable='joy_node',
            condition=IfCondition(with_joystick),
            parameters=[config_paths['joystick']]
        ),
        Node(
            package='teleop_twist_joy',
            executable='teleop_node',
            name='go2_teleop_node',
            condition=IfCondition(with_joystick),
            parameters=[config_paths['twist_mux']],
        ),
        Node(
            package='twist_mux',
            executable='twist_mux',
            output='screen',
            condition=IfCondition(with_joystick),
            parameters=[
                {'use_sim_time': use_sim_time},
                config_paths['twist_mux']
            ],
        ),
    ]
    
    # Visualization nodes
    viz_nodes = [
        Node(
            package='rviz2',
            executable='rviz2',
            condition=IfCondition(with_rviz),
            name='go2_rviz2',
            output='screen',
            arguments=['-d', config_paths['rviz']],
            parameters=[{'use_sim_time': False}]
        ),
    ]
    
    # Include launches
    foxglove_launch = os.path.join(
        get_package_share_directory('foxglove_bridge'),
        'launch', 'foxglove_bridge_launch.xml'
    )
    
    include_launches = [
        # Foxglove Bridge
        IncludeLaunchDescription(
            FrontendLaunchDescriptionSource(foxglove_launch),
            condition=IfCondition(with_foxglove),
        ),
        # SLAM Toolbox for mapping
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[
                rewrite_frame_params(config_paths['slam'], tf_prefix),
                {'use_sim_time': use_sim_time}
            ],
            remappings=[('map', '/go2/map'), ('map_updates', '/go2/map_updates')],
        ),
    ]

    return (
        core_nodes +
        teleop_nodes +
        viz_nodes +
        include_launches
    )


def generate_launch_description():
    """Generate launch description for Go2 mapping mode"""
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false', description='Launch RViz2'),
        DeclareLaunchArgument('foxglove', default_value='false', description='Launch Foxglove Bridge'),
        DeclareLaunchArgument('joystick', default_value='false', description='Launch joystick control'),
        # /tf and /tf_static are global topics, so multiple robots can only
        # share a tree if their frame ids differ. Set to '' for the original
        # unprefixed frames.
        DeclareLaunchArgument(
            'tf_prefix', default_value='go2',
            description='Prefix applied to every TF frame this robot publishes, '
                        'e.g. "go2" yields go2/odom -> go2/base_link'),
        OpaqueFunction(function=_launch_setup),
    ])
