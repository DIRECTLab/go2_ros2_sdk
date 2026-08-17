# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

import os
import tempfile
from typing import List
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace, SetRemap
from launch.actions import (GroupAction, IncludeLaunchDescription,
                            DeclareLaunchArgument, OpaqueFunction)
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


class Go2LaunchConfig:
    """Configuration container for Go2 robot launch parameters"""

    def __init__(self, tf_prefix: str = ''):
        # TF namespacing
        self.tf_prefix = tf_prefix.strip().strip('/')
        # robot_state_publisher prepends frame_prefix verbatim, so it needs the
        # trailing separator: '' -> '', 'go2' -> 'go2/'
        self.frame_prefix = f'{self.tf_prefix}/' if self.tf_prefix else ''

        # Environment variables
        self.robot_token = os.getenv('ROBOT_TOKEN', '')
        self.robot_ip = os.getenv('ROBOT_IP', '')
        self.robot_ip_list = self._parse_ip_list(self.robot_ip)
        self.map_name = os.getenv('MAP_NAME', '3d_map')
        self.save_map = os.getenv('MAP_SAVE', 'true')
        self.conn_type = os.getenv('CONN_TYPE', 'webrtc')

        # Derived configurations
        self.conn_mode = self._determine_connection_mode()
        self.rviz_config = self._get_rviz_config()
        self.urdf_file = self._get_urdf_file()

        # Package paths
        self.package_dir = get_package_share_directory('go2_robot_sdk')
        self.config_paths = self._get_config_paths()

        print(f"� Go2 Launch Configuration:")
        print(f"   Robot IPs: {self.robot_ip_list}")
        print(f"   Connection: {self.conn_type} ({self.conn_mode})")
        print(f"   URDF: {self.urdf_file}")
        print(f"   TF prefix: {self.tf_prefix or '<none>'}")

    def frame(self, name: str) -> str:
        """Namespace a frame id with this robot's tf_prefix"""
        return prefix_frame(self.tf_prefix, name)

    def _parse_ip_list(self, robot_ip: str) -> List[str]:
        """Parse robot IP addresses from environment variable"""
        return robot_ip.replace(" ", "").split(",") if robot_ip else []

    def _determine_connection_mode(self) -> str:
        """Determine connection mode based on IP list and connection type"""
        return "single" if len(self.robot_ip_list) == 1 and self.conn_type != "cyclonedx" else "multi"

    def _get_rviz_config(self) -> str:
        """Get appropriate RViz configuration file"""
        if self.conn_type == 'cyclonedx':
            return "cyclonedx_config.rviz"
        elif self.conn_mode == 'single':
            return "single_robot_conf.rviz"
        else:
            return "multi_robot_conf.rviz"

    def _get_urdf_file(self) -> str:
        """Get appropriate URDF file"""
        return 'go2.urdf' if self.conn_mode == 'single' else 'multi_go2.urdf'

    def _get_config_paths(self) -> dict:
        """Get all configuration file paths"""
        return {
            'joystick': os.path.join(self.package_dir, 'config', 'joystick.yaml'),
            'twist_mux': os.path.join(self.package_dir, 'config', 'twist_mux.yaml'),
            'slam': os.path.join(self.package_dir, 'config', 'mapper_params_online_async.yaml'),
            'nav2': os.path.join(self.package_dir, 'config', 'nav2_params.yaml'),
            'rviz': os.path.join(self.package_dir, 'config', self.rviz_config),
            'urdf': os.path.join(self.package_dir, 'urdf', self.urdf_file),
        }


class Go2NodeFactory:
    """Factory for creating Go2 robot nodes"""

    def __init__(self, config: Go2LaunchConfig):
        self.config = config

    @staticmethod
    def create_launch_arguments() -> List[DeclareLaunchArgument]:
        """Create all launch arguments"""
        return [
            DeclareLaunchArgument('rviz2', default_value='true', description='Launch RViz2'),
            DeclareLaunchArgument('nav2', default_value='true', description='Launch Nav2'),
            DeclareLaunchArgument('slam', default_value='true', description='Launch SLAM'),
            DeclareLaunchArgument('foxglove', default_value='true', description='Launch Foxglove Bridge'),
            DeclareLaunchArgument('joystick', default_value='true', description='Launch joystick'),
            DeclareLaunchArgument('teleop', default_value='true', description='Launch teleoperation'),
            # /tf and /tf_static are global topics, so multiple robots can only
            # share a tree if their frame ids differ. Set to '' for the original
            # unprefixed frames.
            DeclareLaunchArgument(
                'tf_prefix', default_value='go2',
                description='Prefix applied to every TF frame this robot publishes, '
                            'e.g. "go2" yields go2/odom -> go2/base_link'),
            # Raw lidar over CycloneDDS/Ethernet. Off by default: it is an
            # additional feed alongside the driver's WebRTC point_cloud2, not a
            # replacement, and it needs unitree_sdk2py plus a cabled link.
            DeclareLaunchArgument(
                'raw_lidar', default_value='false',
                description='Publish the raw rt/utlidar/cloud feed over CycloneDDS'),
            DeclareLaunchArgument(
                'raw_lidar_iface', default_value=os.getenv('GO2_LIDAR_IFACE', ''),
                description='Ethernet interface facing the robot, e.g. "eth0". '
                            'Empty means every interface'),
            DeclareLaunchArgument(
                'raw_lidar_domain', default_value='0',
                description='CycloneDDS domain id the robot publishes on'),
            DeclareLaunchArgument(
                'raw_lidar_topic', default_value='raw_lidar',
                description='Topic for the raw cloud. Relative resolves under this '
                            'stack\'s /go2 namespace; pass an absolute name such as '
                            '/r0/raw_lidar to feed a Swarm-SLAM robot namespace'),
            DeclareLaunchArgument(
                'raw_lidar_frame', default_value='',
                description='frame_id for the raw cloud. Empty derives '
                            '<tf_prefix>/radar, the lidar link go2.urdf already '
                            'defines. This node publishes no TF'),
            DeclareLaunchArgument(
                'raw_lidar_stamp', default_value='raw',
                description="Header stamp basis: 'raw', 'raw_header' or 'receive'"),
            # FOV mask over the raw cloud, for cross-robot comparison. Off by
            # default and purely additive: it publishes a second topic and
            # leaves the unmasked feed alone.
            DeclareLaunchArgument(
                'fov_mask', default_value='false',
                description='Mask the raw cloud down to a configurable region so it '
                            'is comparable with another robot (requires raw_lidar)'),
            DeclareLaunchArgument(
                'fov_mask_params', default_value='',
                description='Path to a mask yaml. Empty uses config/fov_mask.yaml. '
                            'Point both robots at the same file'),
            DeclareLaunchArgument(
                'fov_mask_frame', default_value='',
                description='Gravity-aligned frame the mask is evaluated in. '
                            'Empty derives <tf_prefix>/base_footprint'),
        ]

    def create_robot_state_nodes(self) -> List[Node]:
        """Create robot state publisher nodes"""
        nodes = []
        use_sim_time = LaunchConfiguration('use_sim_time', default='false')

        if self.config.conn_mode == 'single':
            # Single robot configuration
            robot_desc = self._load_urdf_content(self.config.config_paths['urdf'])

            nodes.extend([
                Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    name='go2_robot_state_publisher',
                    output='screen',
                    parameters=[{
                        'use_sim_time': use_sim_time,
                        'robot_description': robot_desc,
                        'frame_prefix': self.config.frame_prefix
                    }],
                    arguments=[self.config.config_paths['urdf']]
                ),
                self._create_pointcloud_to_laserscan_node()
            ])
        else:
            # Multi-robot configuration
            base_urdf = self._load_urdf_content(self.config.config_paths['urdf'])

            for i, _ in enumerate(self.config.robot_ip_list):
                robot_desc = base_urdf.format(robot_num=f"robot{i}")

                nodes.extend([
                    Node(
                        package='robot_state_publisher',
                        executable='robot_state_publisher',
                        name='go2_robot_state_publisher',
                        output='screen',
                        namespace=f"robot{i}",
                        parameters=[{
                            'use_sim_time': use_sim_time,
                            'robot_description': robot_desc
                        }],
                        arguments=[self.config.config_paths['urdf']]
                    ),
                    self._create_pointcloud_to_laserscan_node(f"robot{i}")
                ])

        return nodes

    def _load_urdf_content(self, urdf_path: str) -> str:
        """Load URDF file content"""
        with open(urdf_path, 'r') as file:
            return file.read()

    def _create_pointcloud_to_laserscan_node(self, namespace: str = None) -> Node:
        """Create pointcloud to laserscan conversion node"""
        if namespace:
            # Multi-robot setup
            return Node(
                package='pointcloud_to_laserscan',
                executable='pointcloud_to_laserscan_node',
                name=f'{namespace}_pointcloud_to_laserscan',
                remappings=[
                    ('cloud_in', f'{namespace}/point_cloud2'),
                    ('scan', f'{namespace}/scan'),
                ],
                parameters=[{
                    'target_frame': self.config.frame(f'{namespace}/base_link'),
                    'max_height': 0.1
                }],
                output='screen',
            )
        else:
            # Single robot setup
            return Node(
                package='pointcloud_to_laserscan',
                executable='pointcloud_to_laserscan_node',
                name='go2_pointcloud_to_laserscan',
                remappings=[
                    ('cloud_in', 'point_cloud2'),
                    ('scan', 'scan'),
                ],
                parameters=[{
                    'target_frame': self.config.frame('base_link'),
                    'max_height': 0.5
                }],
                output='screen',
            )

    def create_core_nodes(self) -> List[Node]:
        """Create core Go2 robot nodes"""
        return [
            # Main robot driver (clean architecture)
            Node(
                package='go2_robot_sdk',
                executable='go2_driver_node',
                name='go2_driver_node',
                output='screen',
                parameters=[{
                    'robot_ip': self.config.robot_ip,
                    'token': self.config.robot_token,
                    'conn_type': self.config.conn_type,
                    'tf_prefix': self.config.tf_prefix
                }],
            ),
            # LiDAR processing node (new separate package)
            Node(
                package='lidar_processor',
                executable='lidar_to_pointcloud',
                name='lidar_to_pointcloud',
                parameters=[{
                    'robot_ip_lst': self.config.robot_ip_list,
                    'map_name': self.config.map_name,
                    'map_save': self.config.save_map
                }],
            ),
            # Advanced point cloud aggregator
            Node(
                package='lidar_processor',
                executable='pointcloud_aggregator',
                name='pointcloud_aggregator',
                parameters=[{
                    'max_range': 20.0,
                    'min_range': 0.1,
                    'height_filter_min': -2.0,
                    'height_filter_max': 3.0,
                    'downsample_rate': 5,
                    'publish_rate': 10.0,
                    'output_frame': self.config.frame('base_link')
                }],
            ),
            # TTS Node (new separate package)
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

    def create_raw_lidar_nodes(self, context) -> List[Node]:
        """Create the raw lidar node (CycloneDDS over Ethernet).

        Purely additive: the driver's existing WebRTC point_cloud2 topic is
        untouched, and nothing here runs unless raw_lidar:=true.
        """
        with_raw_lidar = LaunchConfiguration('raw_lidar', default='false')

        # Empty frame means "derive from tf_prefix", so the cloud lands in the
        # same tree the rest of this launch file builds.
        frame_id = LaunchConfiguration('raw_lidar_frame').perform(context)
        if not frame_id:
            # The lidar's own frame, supplied by the static transform below
            # rather than by the URDF, so the raw node still publishes no TF.
            frame_id = self.config.frame('utlidar_lidar')

        # Resolved here rather than passed as a substitution: the node declares
        # dds_domain_id as an integer, and substitutions arrive as strings.
        domain_id = int(LaunchConfiguration('raw_lidar_domain').perform(context))

        return [
            Node(
                package='go2_robot_sdk',
                executable='raw_lidar_node',
                name='raw_lidar_node',
                output='screen',
                condition=IfCondition(with_raw_lidar),
                parameters=[{
                    'network_interface': LaunchConfiguration('raw_lidar_iface'),
                    'dds_domain_id': domain_id,
                    'output_topic': LaunchConfiguration('raw_lidar_topic'),
                    'frame_id': frame_id,
                    'stamp_source': LaunchConfiguration('raw_lidar_stamp'),
                }],
            ),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='lidar_static_tf',
                output='screen',
                condition=IfCondition(with_raw_lidar),
                # x y z yaw pitch roll parent_frame child_frame
                arguments=[
                           '0.28945', '0', '0.4', # offset from base_link (adjust to match physical mount)
                           '2.1', '-3', '0.2', # rotation (yaw pitch roll in radians)
                           self.config.frame('base_link'),
                           self.config.frame('utlidar_lidar')],
            ),
        ]

    def create_fov_mask_nodes(self, context) -> List[Node]:
        """Create the FOV mask node over the raw cloud.

        Additive: publishes <raw_lidar_topic>_masked and leaves the unmasked
        feed untouched, so both can be compared side by side. Nothing runs
        unless fov_mask:=true.
        """
        with_fov_mask = LaunchConfiguration('fov_mask', default='false')

        params_file = LaunchConfiguration('fov_mask_params').perform(context)
        if not params_file:
            params_file = os.path.join(self.config.package_dir, 'config', 'fov_mask.yaml')

        mask_frame = LaunchConfiguration('fov_mask_frame').perform(context)
        if not mask_frame:
            mask_frame = self.config.frame('base_footprint')

        raw_topic = LaunchConfiguration('raw_lidar_topic').perform(context)

        return [
            Node(
                package='lidar_processor',
                executable='fov_mask',
                name='fov_mask_node',
                output='screen',
                condition=IfCondition(with_fov_mask),
                parameters=[
                    params_file,
                    # Direct override: mask_frame is the one value that cannot be
                    # shared between robots, since it carries each one's prefix.
                    {'mask_frame': mask_frame},
                ],
                remappings=[
                    ('cloud_in', raw_topic),
                    ('cloud_masked', f'{raw_topic}_masked'),
                ],
            ),
        ]

    def create_teleop_nodes(self) -> List[Node]:
        """Create teleoperation and joystick nodes"""
        use_sim_time = LaunchConfiguration('use_sim_time', default='false')
        with_joystick = LaunchConfiguration('joystick', default='true')
        with_teleop = LaunchConfiguration('teleop', default='true')

        return [
            # Joystick node
            Node(
                package='joy',
                executable='joy_node',
                condition=IfCondition(with_joystick),
                parameters=[self.config.config_paths['joystick']]
            ),
            # Teleop twist joy node
            Node(
                package='teleop_twist_joy',
                executable='teleop_node',
                name='go2_teleop_node',
                condition=IfCondition(with_joystick),
                parameters=[self.config.config_paths['twist_mux']],
            ),
            # Twist multiplexer
            Node(
                package='twist_mux',
                executable='twist_mux',
                output='screen',
                condition=IfCondition(with_teleop),
                parameters=[
                    {'use_sim_time': use_sim_time},
                    self.config.config_paths['twist_mux']
                ],
            ),
        ]

    def create_visualization_nodes(self) -> List[Node]:
        """Create visualization nodes (RViz, Foxglove)"""
        with_rviz2 = LaunchConfiguration('rviz2', default='true')

        return [
            # RViz2
            Node(
                package='rviz2',
                executable='rviz2',
                condition=IfCondition(with_rviz2),
                name='go2_rviz2',
                output='screen',
                arguments=['-d', self.config.config_paths['rviz']],
                parameters=[{'use_sim_time': False}]
            ),
        ]

    def create_include_launches(self) -> List[IncludeLaunchDescription | Node]:
        """Create included launch descriptions"""
        use_sim_time = LaunchConfiguration('use_sim_time', default='false')
        with_foxglove = LaunchConfiguration('foxglove', default='true')
        with_slam = LaunchConfiguration('slam', default='true')
        with_nav2 = LaunchConfiguration('nav2', default='true')

        foxglove_launch = os.path.join(
            get_package_share_directory('foxglove_bridge'),
            'launch', 'foxglove_bridge_launch.xml'
        )

        return [
            # Foxglove Bridge
            IncludeLaunchDescription(
                FrontendLaunchDescriptionSource(foxglove_launch),
                condition=IfCondition(with_foxglove),
            ),
            # SLAM Toolbox (direct Node so remappings take effect)
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                output='screen',
                condition=IfCondition(with_slam),
                parameters=[
                    rewrite_frame_params(self.config.config_paths['slam'],
                                         self.config.tf_prefix),
                    {
                        'use_sim_time': use_sim_time,
                        # The yaml above is keyed 'slam_toolbox:', but
                        # PushRosNamespace makes this node /go2/slam_toolbox, so
                        # that key never matches and the file is ignored. These
                        # are passed as direct overrides, which apply whatever
                        # the node's namespace is.
                        'map_frame': self.config.frame('map'),
                        'odom_frame': self.config.frame('odom'),
                        'base_frame': self.config.frame('base_link'),
                    }
                ],
                remappings=[
                    ('/scan', '/go2/scan'),
                    ('/map', '/go2/map'),
                    ('/map_updates', '/go2/map_updates'),
                ],
            ),
            # Nav2
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    os.path.join(get_package_share_directory('nav2_bringup'),
                                'launch', 'navigation_launch.py')
                ]),
                condition=IfCondition(with_nav2),
                launch_arguments={
                    'params_file': rewrite_frame_params(
                        self.config.config_paths['nav2'], self.config.tf_prefix),
                    'use_sim_time': use_sim_time
                }.items(),
            ),
        ]


def _launch_setup(context, *args, **kwargs):
    """Build the Go2 stack, now that tf_prefix can be resolved"""

    # Initialize configuration and factory
    config = Go2LaunchConfig(LaunchConfiguration('tf_prefix').perform(context))
    factory = Go2NodeFactory(config)

    # Create all components
    robot_state_nodes = factory.create_robot_state_nodes()
    core_nodes = factory.create_core_nodes()
    raw_lidar_nodes = factory.create_raw_lidar_nodes(context)
    fov_mask_nodes = factory.create_fov_mask_nodes(context)
    teleop_nodes = factory.create_teleop_nodes()
    visualization_nodes = factory.create_visualization_nodes()
    include_launches = factory.create_include_launches()


    # Combine all elements
    launch_entities = (
        robot_state_nodes +
        core_nodes +
        raw_lidar_nodes +
        fov_mask_nodes +
        teleop_nodes +
        visualization_nodes +
        include_launches
    )
    group = GroupAction([
        PushRosNamespace('go2'),
        SetRemap('tf', '/tf'),
        SetRemap('tf_static', '/tf_static'),
        *launch_entities
    ])
    return [group]


def generate_launch_description():
    """Generate the launch description for Go2 robot system"""
    return LaunchDescription(
        Go2NodeFactory.create_launch_arguments() + [
            OpaqueFunction(function=_launch_setup),
        ]
    )

