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


# Which of the two lidars `lidar_source` selects, as (internal, external).
#
# The robot has one NIC and one head connector, so in practice the internal
# utlidar and an externally mounted L2 compete for the same hardware and the
# choice is exclusive -- hence one argument that names the intent, rather than
# two booleans the caller has to remember to keep consistent. 'both' is kept
# because it is the configuration in which the two feeds can be compared
# side by side, which is what FOV_MASK.md's tuning loop asks for.
_LIDAR_SOURCES = {
    'internal': (True, False),
    'external': (False, True),
    'both': (True, True),
    'none': (False, False),
}

# launch's own truthy set, so `raw_lidar:=1` and `raw_lidar:=True` behave here
# the way they do in an IfCondition.
_TRUE_VALUES = frozenset({'true', '1', 'yes', 'on'})
_FALSE_VALUES = frozenset({'false', '0', 'no', 'off'})


def prefix_frame(tf_prefix: str, frame: str) -> str:
    """Namespace a frame id: '' -> 'base_link', 'go2' -> 'go2/base_link'."""
    return f'{tf_prefix}/{frame}' if tf_prefix else frame


def parse_bool(arg: str, value: str, default: bool = None) -> bool:
    """Parse a launch argument as a bool, with '' meaning "use the default".

    The tri-state matters: raw_lidar and unilidar have to distinguish "the
    caller said nothing, so derive it from lidar_source" from "the caller said
    false", and an IfCondition cannot express that.
    """
    text = value.strip().lower()
    if not text:
        if default is None:
            raise RuntimeError(f'{arg} requires a value')
        return default
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise RuntimeError(
        f"{arg} must be true or false (or '' to derive it), got '{value}'")


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
            'unilidar': os.path.join(self.package_dir, 'config', 'unilidar.yaml'),
            'rviz': os.path.join(self.package_dir, 'config', self.rviz_config),
            'urdf': os.path.join(self.package_dir, 'urdf', self.urdf_file),
        }


class Go2NodeFactory:
    """Factory for creating Go2 robot nodes"""

    def __init__(self, config: Go2LaunchConfig):
        self.config = config
        # Memoised so the selection is resolved -- and reported -- once, even
        # though four node factories ask for it.
        self._lidar_flags = None

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
            # Which lidar feeds this stack. One switch rather than two
            # booleans because the choice is effectively exclusive: the robot
            # has one NIC and one head connector, so an externally mounted L2
            # generally means the internal utlidar is unplugged or unreachable.
            #
            # 'internal' is the default, which is exactly today's behaviour --
            # raw_lidar on, unilidar off.
            DeclareLaunchArgument(
                'lidar_source', default_value='internal',
                choices=list(_LIDAR_SOURCES),
                description="Which lidar to run: 'internal' (the factory "
                            "utlidar over CycloneDDS), 'external' (an L2 bolted "
                            "on, over serial/UDP), 'both' (for comparing the two "
                            "feeds side by side) or 'none'. Each feed's own "
                            'fov_mask instance follows it, so nothing is left '
                            'subscribing to a topic no one publishes'),
            # Raw lidar over CycloneDDS/Ethernet. Empty derives from
            # lidar_source; pass true/false to override that for this feed
            # alone. Additive either way -- the driver's WebRTC point_cloud2 is
            # unaffected. Needs unitree_sdk2py and a cabled link on the robot's
            # subnet; without them the node logs the failure and exits, leaving
            # the rest of the stack running.
            DeclareLaunchArgument(
                'raw_lidar', default_value='',
                description='Publish the raw rt/utlidar/cloud feed over '
                            "CycloneDDS. Empty follows lidar_source (on for "
                            "'internal' and 'both'); true/false overrides it"),
            DeclareLaunchArgument(
                'raw_lidar_iface',
                default_value=os.getenv('GO2_LIDAR_IFACE', 'enP8p1s0'),
                description='Ethernet interface facing the robot. Defaults to the '
                            "Jetson's onboard NIC; override with the GO2_LIDAR_IFACE "
                            'environment variable or this argument on other hosts. '
                            'Must be up and hold an address on the robot subnet -- '
                            'CycloneDDS rejects it otherwise'),
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
            # FOV mask over the raw cloud. On by default so the processed feed
            # is available without extra arguments; still purely additive, since
            # it publishes its own topic and leaves the unmasked feed alone.
            # Its input comes from raw_lidar, so it is skipped outright when
            # that feed is off rather than left warning about a topic nobody
            # publishes.
            DeclareLaunchArgument(
                'fov_mask', default_value='true',
                description='Mask the raw cloud down to a configurable region so it '
                            'is comparable with another robot. Ignored unless the '
                            'internal feed is running'),
            DeclareLaunchArgument(
                'fov_mask_params', default_value='',
                description='Path to a mask yaml. Empty uses config/fov_mask.yaml. '
                            'Point both robots at the same file'),
            DeclareLaunchArgument(
                'fov_mask_frame', default_value='',
                description='Frame supplying the mask axes and z datum. Empty '
                            'derives <tf_prefix>/base_footprint; pass e.g. '
                            'go2/odom for a datum that does not bob with the gait'),
            DeclareLaunchArgument(
                'fov_mask_origin', default_value='',
                description="Where range/azimuth/elevation are measured from. Empty "
                            "keeps the yaml's value ('sensor', which keeps the field "
                            "of view attached to the robot). Accepts 'sensor', "
                            "'mask_frame', or an explicit frame id"),
            DeclareLaunchArgument(
                'fov_mask_decay', default_value='',
                description="Seconds of history to accumulate, like RViz2's Decay "
                            'Time. Switches cloud_processed from per-scan masked '
                            "clouds to the accumulated history. Empty keeps the yaml's "
                            'value; 0 disables accumulation'),
            DeclareLaunchArgument(
                'fov_mask_decay_frame', default_value='',
                description='Frame the accumulation happens in. Empty derives '
                            '<tf_prefix>/odom. Must be fixed with respect to the '
                            'world -- never base_link or base_footprint'),
            # ---------------------------------------------------------------
            # Externally mounted Unitree L2, driven directly over serial or UDP
            # by unitree_lidar_ros2 (unilidar_sdk2). This is a SECOND, entirely
            # separate sensor from the robot's factory-mounted utlidar that
            # raw_lidar above reads over CycloneDDS -- both can run at once and
            # neither touches the other's topics or frames.
            #
            # Off unless lidar_source asks for it. The vendor driver responds
            # to a missing sensor by calling exit(0) or blocking on a socket
            # bind rather than logging and carrying on, so this is not a feed
            # to leave enabled speculatively on a robot that may not have the
            # hardware bolted on.
            DeclareLaunchArgument(
                'unilidar', default_value='',
                description='Start the external Unitree L2 driver '
                            '(unitree_lidar_ros2) and publish its mount '
                            "transform. Empty follows lidar_source (on for "
                            "'external' and 'both'); true/false overrides it"),
            DeclareLaunchArgument(
                'unilidar_params', default_value='',
                description='Path to the driver yaml. Empty uses '
                            'config/unilidar.yaml'),
            # Which transport is wired up is a property of this robot, not of
            # the sensor, so it is an argument rather than a yaml edit.
            DeclareLaunchArgument(
                'unilidar_conn', default_value='',
                description="Transport: 'serial' (USB CDC) or 'udp' (ethernet). "
                            "Empty keeps the yaml's initialize_type"),
            DeclareLaunchArgument(
                'unilidar_serial_port', default_value='',
                description="Serial device for unilidar_conn:=serial. Empty keeps "
                            "the yaml's /dev/ttyACM0. Prefer a udev symlink such as "
                            '/dev/unilidar -- ttyACM numbering is not stable'),
            DeclareLaunchArgument(
                'unilidar_ip', default_value='',
                description="The sensor's own address for unilidar_conn:=udp. "
                            "Empty keeps the yaml's 192.168.1.62"),
            DeclareLaunchArgument(
                'unilidar_local_ip', default_value='',
                description="This host's address on the lidar subnet. Empty keeps "
                            "the yaml's 192.168.1.2. The host must actually hold "
                            'this address or the UDP bind fails'),
            DeclareLaunchArgument(
                'unilidar_topic', default_value='unilidar/cloud',
                description='Topic for the external cloud; the IMU topic follows '
                            'it. Relative resolves under this stack\'s /go2 '
                            "namespace, matching the rover's "
                            '/<robot>/unilidar/cloud. Pass an absolute name such '
                            'as /r0/unilidar/cloud to feed a Swarm-SLAM robot '
                            'namespace directly'),
            DeclareLaunchArgument(
                'unilidar_frame', default_value='',
                description='frame_id for the external cloud. Empty derives '
                            '<tf_prefix>/unilidar_lidar, which the mount transform '
                            'below supplies'),
            # The mount geometry is the one thing here that cannot have a
            # correct default: it describes where this particular sensor is
            # bolted. Measure it. Everything downstream -- the shared z band,
            # the range band, cross-robot registration -- is expressed in a
            # gravity-aligned frame reached through this transform, so an
            # unmeasured mount silently invalidates all of it.
            DeclareLaunchArgument(
                'unilidar_mount_xyz', default_value='0.0 0.0 0.15',
                description='Sensor origin as "x y z" metres in <tf_prefix>/'
                            'base_link. MEASURE THIS -- the default is a placeholder '
                            'for a plate on the robot\'s back, not a calibration'),
            DeclareLaunchArgument(
                'unilidar_mount_ypr', default_value='0.0 0.0 0.0',
                description='Sensor orientation as "yaw pitch roll" radians, the '
                            'order static_transform_publisher takes positionally. '
                            'All zeros means upright and facing forward, which is '
                            "the point of mounting it externally -- the Go2's "
                            'factory lidar is the inverted one'),
            DeclareLaunchArgument(
                'unilidar_fov_mask', default_value='false',
                description='Run a fov_mask instance over the external cloud, '
                            'publishing <unilidar_topic>_processed. Off by '
                            'default: with the L2 mounted upright on top of both '
                            'robots, the two sensors already observe the same '
                            'part of the world and there is no band to select. '
                            'Turn it on to reduce both feeds to a shared region, '
                            'or to accumulate sweeps (fov_mask_decay) -- the same '
                            'node does both. Shares the fov_mask_* arguments '
                            'above, since the band is what both robots hold in '
                            'common. Ignored unless the external feed is running'),
            DeclareLaunchArgument(
                'unilidar_fov_mask_blank_radius', default_value='',
                description="Override sensor_blank_radius for the external feed "
                            'only. Separate from the shared yaml on purpose: this '
                            "value describes mount hardware, and the external "
                            "mount is not the factory head mount. Empty keeps the "
                            "yaml's value"),
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

    def lidar_flags(self, context) -> tuple:
        """Resolve which lidar feeds run, as (internal, external) booleans.

        lidar_source names the intent; raw_lidar and unilidar override it per
        feed when they are given a value. The tri-state is what makes that
        work: '' means "follow lidar_source", which is different from false and
        cannot be expressed as an IfCondition -- so the gating for both feeds
        is resolved here in Python and the node lists come back empty rather
        than carrying a condition.

        Resolving it rather than deferring to IfCondition also means the
        fov_mask instances can be skipped when their input feed is off. A mask
        node with no publisher on cloud_in does not fail; it sits there warning
        that it has processed 0 clouds, which reads like a bug in the mask.
        """
        if self._lidar_flags is not None:
            return self._lidar_flags

        source = LaunchConfiguration('lidar_source').perform(context).strip().lower()
        if source not in _LIDAR_SOURCES:
            raise RuntimeError(
                f"lidar_source must be one of {sorted(_LIDAR_SOURCES)}, "
                f"got '{source}'")
        internal_default, external_default = _LIDAR_SOURCES[source]

        internal = parse_bool(
            'raw_lidar',
            LaunchConfiguration('raw_lidar').perform(context),
            internal_default)
        external = parse_bool(
            'unilidar',
            LaunchConfiguration('unilidar').perform(context),
            external_default)

        selected = [name for name, on in
                    (('internal utlidar', internal), ('external L2', external)) if on]
        print(f"   Lidar: {', '.join(selected) or '<none>'} "
              f"(lidar_source:={source})")

        self._lidar_flags = (internal, external)
        return self._lidar_flags

    def create_raw_lidar_nodes(self, context) -> List[Node]:
        """Create the raw lidar node (CycloneDDS over Ethernet).

        Purely additive: the driver's existing WebRTC point_cloud2 topic is
        untouched, and nothing here runs unless the internal feed is selected
        (lidar_source:=internal|both, or raw_lidar:=true outright).
        """
        internal, _ = self.lidar_flags(context)
        if not internal:
            return []

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
                # x y z yaw pitch roll parent_frame child_frame
                arguments=[
                           '0.28945', '0', '0.45', # offset from base_link (adjust to match physical mount)
                           '2.1', '-2.95', '0.23', # rotation (yaw pitch roll in radians)
                           self.config.frame('base_link'),
                           self.config.frame('utlidar_lidar')],
            ),
        ]

    def _fov_mask_config(self, context) -> tuple:
        """Resolve the shared mask yaml and the per-robot overrides on top of it.

        Both fov_mask instances -- the one over the factory utlidar and the one
        over the external L2 -- read the SAME yaml and the SAME fov_mask_*
        arguments. That is deliberate and is the whole point of the file: the
        band is what the two ROBOTS have to hold in common, so it cannot be
        specialised per feed without breaking the comparison it exists to make.

        Only the values that carry a tf_prefix are overridden here, since a
        yaml shared with another robot cannot name this robot's frames.
        """
        params_file = LaunchConfiguration('fov_mask_params').perform(context)
        if not params_file:
            params_file = os.path.join(self.config.package_dir, 'config', 'fov_mask.yaml')

        mask_frame = LaunchConfiguration('fov_mask_frame').perform(context)
        if not mask_frame:
            mask_frame = self.config.frame('base_footprint')

        # Direct overrides: these carry each robot's tf_prefix, so they cannot
        # live in a yaml meant to be shared between robots. Empty means the
        # yaml's own value stands.
        overrides = {'mask_frame': mask_frame}
        mask_origin = LaunchConfiguration('fov_mask_origin').perform(context)
        if mask_origin:
            overrides['mask_origin'] = mask_origin

        # Accumulation must happen in a frame that does not ride the robot, so
        # this derives odom rather than the base_footprint mask_frame default.
        decay_frame = LaunchConfiguration('fov_mask_decay_frame').perform(context)
        overrides['decay_frame'] = decay_frame or self.config.frame('odom')

        # Resolved and cast here: the node declares decay_time as a double, and
        # launch substitutions arrive as strings.
        decay_time = LaunchConfiguration('fov_mask_decay').perform(context)
        if decay_time:
            overrides['decay_time'] = float(decay_time)

        return params_file, overrides

    def create_fov_mask_nodes(self, context) -> List[Node]:
        """Create the FOV mask node over the raw cloud.

        Additive: publishes <raw_lidar_topic>_processed and leaves the
        unmasked feed untouched, so the two can be compared side by side. That
        one topic carries per-scan masked clouds, or the accumulated history
        when fov_mask_decay is set.

        Skipped entirely when the internal feed is off, not merely when
        fov_mask:=false -- its input is that feed, and a mask node with no
        publisher on cloud_in sits there reporting 0 clouds processed, which
        reads like a fault in the mask rather than an absent sensor.
        """
        internal, _ = self.lidar_flags(context)
        if not internal or not parse_bool(
                'fov_mask', LaunchConfiguration('fov_mask').perform(context)):
            return []

        params_file, overrides = self._fov_mask_config(context)
        raw_topic = LaunchConfiguration('raw_lidar_topic').perform(context)

        return [
            Node(
                package='lidar_processor',
                executable='fov_mask',
                name='fov_mask_node',
                output='screen',
                parameters=[params_file, overrides],
                remappings=[
                    ('cloud_in', raw_topic),
                    ('cloud_processed', f'{raw_topic}_processed'),
                ],
            ),
        ]

    def create_unilidar_nodes(self, context) -> List[Node]:
        """Create the external Unitree L2 driver and its mount transform.

        Mirrors the rover's swarm_slam.launch.py: the same driver package, the
        same unilidar/cloud topic, the same <prefix>/unilidar_lidar frame, and a
        static transform placing that frame on the body. Both robots therefore
        present an upright external L2 the same way, which is what lets one
        shared mask yaml describe both.

        Additive: the factory utlidar feed that raw_lidar_node publishes is
        untouched, and nothing here runs unless the external feed is selected
        (lidar_source:=external|both, or unilidar:=true outright).

        This node publishes NO TF of its own in work_mode 4 -- the mount
        transform below is the sole authority on <prefix>/unilidar_lidar. In
        other work modes the vendor driver broadcasts
        unilidar_imu_initial -> unilidar_imu -> unilidar_lidar from its IMU
        callback, which would make two publishers claim that child frame. That
        exact competing-authority shape has broken TF lookups in this stack
        before, so the work mode is not an incidental setting.
        """
        _, external = self.lidar_flags(context)
        if not external:
            return []

        params_file = LaunchConfiguration('unilidar_params').perform(context)
        if not params_file:
            params_file = self.config.config_paths['unilidar']

        # Empty frame means "derive from tf_prefix", so the cloud lands in the
        # same tree the rest of this launch file builds. Bare, unprefixed frame
        # ids have broken TF lookups here repeatedly.
        cloud_frame = LaunchConfiguration('unilidar_frame').perform(context)
        if not cloud_frame:
            cloud_frame = self.config.frame('unilidar_lidar')

        cloud_topic = LaunchConfiguration('unilidar_topic').perform(context)

        # The IMU topic sits alongside the cloud, so retuning unilidar_topic
        # moves both together rather than leaving the IMU behind on a stale
        # name. 'unilidar/cloud' -> 'unilidar/imu'; a bare 'cloud' -> 'imu'.
        imu_topic = (cloud_topic.rsplit('/', 1)[0] + '/imu') if '/' in cloud_topic else 'imu'

        # The frame ids and topic names are always overridden, never left to the
        # yaml: their values carry this robot's tf_prefix. The IMU frame gets a
        # prefix too even though nothing consumes it in work_mode 4, so that
        # turning the IMU on later does not inject an unprefixed frame.
        overrides = {
            'cloud_frame': cloud_frame,
            'cloud_topic': cloud_topic,
            'imu_frame': self.config.frame('unilidar_imu'),
            'imu_topic': imu_topic,
        }

        # 'serial' / 'udp' rather than the SDK's bare 1 / 2, because a wrong
        # initialize_type makes the vendor code print one line and call
        # exit(0) -- which reads as the node starting and vanishing.
        conn = LaunchConfiguration('unilidar_conn').perform(context).strip().lower()
        if conn:
            if conn not in ('serial', 'udp'):
                raise RuntimeError(
                    f"unilidar_conn must be 'serial' or 'udp', got '{conn}'")
            overrides['initialize_type'] = 1 if conn == 'serial' else 2

        serial_port = LaunchConfiguration('unilidar_serial_port').perform(context)
        if serial_port:
            overrides['serial_port'] = serial_port

        lidar_ip = LaunchConfiguration('unilidar_ip').perform(context)
        if lidar_ip:
            overrides['lidar_ip'] = lidar_ip

        local_ip = LaunchConfiguration('unilidar_local_ip').perform(context)
        if local_ip:
            overrides['local_ip'] = local_ip

        # static_transform_publisher takes these positionally as
        # x y z yaw pitch roll, and passing the wrong count is a classic
        # missing-comma bug in this repo's launch files -- so they are parsed
        # and checked here rather than splatted straight into arguments, where
        # a miscount silently shifts the parent and child frame ids along.
        def mount(arg: str, labels: str) -> List[str]:
            fields = LaunchConfiguration(arg).perform(context).split()
            if len(fields) != 3:
                raise RuntimeError(
                    f'{arg} must be three numbers "{labels}", got '
                    f'{len(fields)}: {fields}')
            try:
                return [str(float(field)) for field in fields]
            except ValueError as exc:
                raise RuntimeError(f'{arg} is not numeric: {fields}') from exc

        mount_xyz = mount('unilidar_mount_xyz', 'x y z')
        mount_ypr = mount('unilidar_mount_ypr', 'yaw pitch roll')

        return [
            Node(
                package='unitree_lidar_ros2',
                executable='unitree_lidar_ros2_node',
                name='unitree_lidar_ros2_node',
                output='screen',
                parameters=[params_file, overrides],
            ),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='unilidar_static_tf',
                output='screen',
                # x y z yaw pitch roll parent_frame child_frame
                #
                # Parented to base_link rather than base_footprint because that
                # is where the sensor is physically bolted. On this robot the
                # two are coincident anyway -- go2.urdf leaves
                # base_footprint_joint at zero deliberately, so base_footprint
                # is NOT ground-projected here despite the name -- but naming
                # base_link keeps the transform meaningful if that is ever
                # fixed.
                arguments=[*mount_xyz, *mount_ypr,
                           self.config.frame('base_link'),
                           cloud_frame],
            ),
        ]

    def create_unilidar_fov_mask_nodes(self, context) -> List[Node]:
        """Create a second FOV mask instance over the external L2 cloud.

        A separate instance rather than a switch on the existing one: the two
        feeds are different sensors on different mounts and both are worth
        having masked at once, which is how you compare what the factory
        inverted mount sees against what the external upright one does.

        Publishes <unilidar_topic>_processed -- so
        /go2/unilidar/cloud_processed by default, matching the rover's
        /<robot>/unilidar/cloud_processed. That is the topic to hand cslam:

            pointcloud_topic:=unilidar/cloud_processed

        Like the raw instance, skipped entirely when its input feed is off
        rather than left subscribing to a topic nobody publishes.

        Off by default, unlike the raw instance. The mask exists to reconcile
        two differently-mounted sensors: the factory utlidar is inverted ~165
        deg and sees mostly floor, so reducing both robots to one band is the
        only way their clouds describe the same content. An L2 mounted upright
        on top of both robots removes that problem at the hardware level --
        same sensor, same attitude, no band to select -- so masking here would
        only discard geometry. The shared yaml's sensor_blank_radius of 0.68 m,
        sized for the rover's mast, would discard a lot of it.

        Note that this node also does accumulation and deskewing, which are
        NOT masking: the L2 is a non-repetitive scanner, so one sweep is a
        sparse slice of the room. With the node off, consumers get per-sweep
        clouds off unilidar/cloud. To accumulate without masking, turn it on
        and leave the band unbounded -- every mask primitive defaults to
        unbounded, so a fov_mask with only decay_time set is a passthrough that
        accumulates.
        """
        _, external = self.lidar_flags(context)
        if not external or not parse_bool(
                'unilidar_fov_mask',
                LaunchConfiguration('unilidar_fov_mask').perform(context)):
            return []

        params_file, overrides = self._fov_mask_config(context)

        # sensor_blank_radius is the one mask value that is legitimately
        # per-feed: it deletes returns off the sensor's own mount, and the
        # external mount is not the factory head mount. Everything else stays
        # shared -- see _fov_mask_config.
        blank_radius = LaunchConfiguration(
            'unilidar_fov_mask_blank_radius').perform(context)
        if blank_radius:
            overrides['sensor_blank_radius'] = float(blank_radius)

        cloud_topic = LaunchConfiguration('unilidar_topic').perform(context)

        return [
            Node(
                package='lidar_processor',
                executable='fov_mask',
                name='unilidar_fov_mask_node',
                output='screen',
                parameters=[params_file, overrides],
                remappings=[
                    ('cloud_in', cloud_topic),
                    ('cloud_processed', f'{cloud_topic}_processed'),
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
    unilidar_nodes = factory.create_unilidar_nodes(context)
    unilidar_fov_mask_nodes = factory.create_unilidar_fov_mask_nodes(context)
    teleop_nodes = factory.create_teleop_nodes()
    visualization_nodes = factory.create_visualization_nodes()
    include_launches = factory.create_include_launches()


    # Combine all elements
    launch_entities = (
        robot_state_nodes +
        core_nodes +
        raw_lidar_nodes +
        fov_mask_nodes +
        unilidar_nodes +
        unilidar_fov_mask_nodes +
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

