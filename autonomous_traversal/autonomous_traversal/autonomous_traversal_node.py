"""
Autonomous traversal node for the Unitree Go2 robot.

Subscribes to /point_cloud2, /scan, /odom and /map. Publishes /cmd_vel.

State machine:
  MOVING_FORWARD        -- obstacle detected       -->  TURNING
  TURNING               -- aligned & clear         -->  MOVING_FORWARD
                                                        (or NAVIGATING_TO_FRONTIER
                                                         if a nav_goal is active)
  TURNING               -- aligned & blocked       -->  TURNING (next best sector)
  TURNING               -- all sectors blocked     -->  NAVIGATING_TO_FRONTIER
                                                        (if remembered waypoints exist)
  TURNING               -- all sectors blocked,    -->  STOPPED
                           no waypoints
  NAVIGATING_TO_FRONTIER -- obstacle ahead         -->  TURNING (then resumes nav)
  NAVIGATING_TO_FRONTIER -- arrived at waypoint    -->  MOVING_FORWARD
                                                        (or next waypoint)
  NAVIGATING_TO_FRONTIER -- all waypoints gone     -->  STOPPED
  STOPPED               -- terminal state          -->  (nothing)

Sector-scoring strategy (three-tier, best available used):

  Tier 1 – Visible-frontier score (point cloud + map)
    The 3-D point cloud is projected to 2-D (bird's-eye) and transformed from
    base_link to map frame using the robot's current odom pose.  Every projected
    point whose map cell is *unknown* (-1) is a "visible frontier": the lidar
    can already see that area but SLAM has not yet integrated it.  Each sector
    is scored by the fraction of all visible-frontier points whose bearing falls
    inside it.  This directly steers the robot toward space it can see but has
    not yet mapped.

  Tier 2 – Ray-exploration score (map only, fallback when no frontiers visible)
    A ray is cast from the robot's position through each sector's centre angle.
    The score is the fraction of traversed map cells that are unknown.

  Tier 3 – Lidar-openness score (always computed, blended with tier 1/2)
    Mean valid range in the sector, normalised by sensor max range.  Keeps the
    robot from steering into tight corners.

  Final score = exploration_weight × (tier-1 or tier-2)
              + (1 − exploration_weight) × lidar_score

  When no map has arrived yet only the lidar score is used.

Frontier waypoint memory:
  Every cloud callback clusters the current visible-frontier points into world-
  frame (x, y) centroids and merges them into a persistent list capped at
  max_frontier_waypoints.  Waypoints whose map cells are no longer unknown are
  pruned before each frontier-navigation attempt.  When immediate turning options
  are exhausted the robot navigates to the nearest surviving waypoint, turning
  to face it and resuming forward motion once aligned.
"""

import math
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
from sensor_msgs_py import point_cloud2 as pc2_utils

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan, PointCloud2


class State(Enum):
    MOVING_FORWARD         = auto()
    TURNING                = auto()
    NAVIGATING_TO_FRONTIER = auto()
    STOPPED                = auto()


class AutonomousTraversalNode(Node):
    def __init__(self):
        super().__init__('autonomous_traversal_node')

        # --- parameters ---
        self.declare_parameter('forward_speed',          0.4)
        self.declare_parameter('turn_speed',             0.5)
        self.declare_parameter('obstacle_threshold',     1.0)   # m – trigger a turn
        self.declare_parameter('free_threshold',         1.5)   # m – minimum clearance to move
        self.declare_parameter('front_half_angle_deg',  30.0)
        self.declare_parameter('sector_width_deg',      30.0)
        self.declare_parameter('align_tolerance_deg',    8.0)
        self.declare_parameter('exploration_weight',     0.7)   # 0=lidar-only  1=frontier-only
        self.declare_parameter('ray_length',             4.0)   # m – map raycast length (tier 2)
        self.declare_parameter('occupied_threshold',    65)     # occupancy value for "occupied"
        self.declare_parameter('cloud_z_min',           -0.3)   # m – height band for projection
        self.declare_parameter('cloud_z_max',            1.5)
        self.declare_parameter('cloud_downsample',       5)     # keep every Nth point
        # frontier waypoint memory
        self.declare_parameter('frontier_goal_radius',     0.8)  # m – arrival threshold
        self.declare_parameter('frontier_heading_tol_deg', 15.0) # deg – align before driving
        self.declare_parameter('max_frontier_waypoints',   20)   # cap on stored waypoints
        self.declare_parameter('min_waypoint_sep',         2.5)  # m – cluster / dedup radius
        self.declare_parameter('min_frontier_nav_dist',    2.5)  # m – ignore waypoints closer than this
        self.declare_parameter('max_frontier_bearings',    300)  # subsample cap for sector scoring
        # topics
        self.declare_parameter('scan_topic',    '/scan')
        self.declare_parameter('odom_topic',    '/odom')
        self.declare_parameter('map_topic',     '/map')
        self.declare_parameter('cloud_topic',   '/point_cloud2')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.forward_speed        = self.get_parameter('forward_speed').value
        self.turn_speed           = self.get_parameter('turn_speed').value
        self.obstacle_threshold   = self.get_parameter('obstacle_threshold').value
        self.free_threshold       = self.get_parameter('free_threshold').value
        self.front_half_angle     = math.radians(self.get_parameter('front_half_angle_deg').value)
        self.sector_width         = math.radians(self.get_parameter('sector_width_deg').value)
        self.align_tolerance      = math.radians(self.get_parameter('align_tolerance_deg').value)
        self.exploration_weight   = self.get_parameter('exploration_weight').value
        self.ray_length           = self.get_parameter('ray_length').value
        self.occupied_threshold   = self.get_parameter('occupied_threshold').value
        self.cloud_z_min          = self.get_parameter('cloud_z_min').value
        self.cloud_z_max          = self.get_parameter('cloud_z_max').value
        self.cloud_downsample     = self.get_parameter('cloud_downsample').value
        self.frontier_goal_radius   = self.get_parameter('frontier_goal_radius').value
        self.frontier_heading_tol   = math.radians(self.get_parameter('frontier_heading_tol_deg').value)
        self.max_frontier_waypoints = self.get_parameter('max_frontier_waypoints').value
        self.min_waypoint_sep       = self.get_parameter('min_waypoint_sep').value
        self.min_frontier_nav_dist  = self.get_parameter('min_frontier_nav_dist').value
        self.max_frontier_bearings  = self.get_parameter('max_frontier_bearings').value

        # --- state ---
        self.state            = State.MOVING_FORWARD
        self.current_yaw      = 0.0
        self.robot_x          = 0.0
        self.robot_y          = 0.0
        self.turn_target_yaw: Optional[float]              = None
        self._turn_candidates: List[Tuple[float, float]]   = []
        # frontier navigation
        self.nav_goal:             Optional[Tuple[float, float]] = None
        self._frontier_waypoints:  List[Tuple[float, float]]    = []

        # sensor data
        self.map_data:           Optional[OccupancyGrid] = None
        self._frontier_bearings: np.ndarray = np.empty(0)

        # diagnostics
        self._scan_count  = 0
        self._odom_count  = 0
        self._map_count   = 0
        self._cloud_count = 0
        self._frontier_count_last = 0

        # --- pub / sub ---
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.cmd_pub  = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.create_subscription(
            LaserScan,     self.get_parameter('scan_topic').value,  self.scan_callback,  sensor_qos)
        self.create_subscription(
            Odometry,      self.get_parameter('odom_topic').value,  self.odom_callback,  sensor_qos)
        self.create_subscription(
            OccupancyGrid, self.get_parameter('map_topic').value,   self.map_callback,   map_qos)
        self.create_subscription(
            PointCloud2,   self.get_parameter('cloud_topic').value, self.cloud_callback, sensor_qos)

        self.create_timer(3.0, self._diag_callback)

        self.get_logger().info(
            f'Autonomous traversal started | '
            f'forward={self.forward_speed} m/s  turn={self.turn_speed} rad/s  '
            f'obstacle<{self.obstacle_threshold} m  free>{self.free_threshold} m  '
            f'exploration_weight={self.exploration_weight}  '
            f'max_waypoints={self.max_frontier_waypoints}'
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _diag_callback(self):
        map_info = (
            f'{self.map_data.info.width}×{self.map_data.info.height} '
            f'@ {self.map_data.info.resolution:.2f} m/cell'
            if self.map_data else 'none'
        )
        tier = ('frontier+lidar' if len(self._frontier_bearings) > 0 and self.map_data
                else 'ray+lidar'  if self.map_data
                else 'lidar-only')
        nav_str = (f'→({self.nav_goal[0]:.1f},{self.nav_goal[1]:.1f})'
                   if self.nav_goal else 'none')
        self.get_logger().info(
            f'[diag] state={self.state.name}  '
            f'scan={self._scan_count}  odom={self._odom_count}  '
            f'cloud={self._cloud_count}  map_updates={self._map_count}  '
            f'map={map_info}  frontiers={self._frontier_count_last}  '
            f'waypoints={len(self._frontier_waypoints)}  nav_goal={nav_str}  '
            f'scoring={tier}'
        )
        self._scan_count = self._odom_count = self._map_count = self._cloud_count = 0

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------
    def map_callback(self, msg: OccupancyGrid):
        self.map_data = msg
        self._map_count += 1

    def odom_callback(self, msg: Odometry):
        self._odom_count += 1
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def cloud_callback(self, msg: PointCloud2):
        """
        Project the 3-D cloud to 2-D, transform to map frame, then find all
        points that land on unknown map cells.

        Two outputs:
          • self._frontier_bearings  – robot-relative angles used for sector scoring
          • self._frontier_waypoints – persistent world-frame (x,y) list used for
                                       recovery navigation when immediate options run out
        """
        self._cloud_count += 1

        if self.map_data is None:
            return

        # --- read & downsample ---
        try:
            raw = np.array(
                [(p[0], p[1], p[2])
                 for p in pc2_utils.read_points(
                     msg, field_names=('x', 'y', 'z'), skip_nans=True)],
                dtype=np.float32,
            )
        except Exception as e:
            self.get_logger().warn(f'cloud_callback read error: {e}')
            return

        if len(raw) == 0:
            self._frontier_bearings = np.empty(0)
            return

        pts = raw[::self.cloud_downsample]

        # --- height filter ---
        pts = pts[(pts[:, 2] >= self.cloud_z_min) & (pts[:, 2] <= self.cloud_z_max)]
        if len(pts) == 0:
            self._frontier_bearings = np.empty(0)
            return

        # --- transform base_link → map frame using current odom pose ---
        cos_y = math.cos(self.current_yaw)
        sin_y = math.sin(self.current_yaw)
        map_x = self.robot_x + pts[:, 0] * cos_y - pts[:, 1] * sin_y
        map_y = self.robot_y + pts[:, 0] * sin_y + pts[:, 1] * cos_y

        # --- look up map cells ---
        res      = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        width    = self.map_data.info.width
        height   = self.map_data.info.height

        mx = ((map_x - origin_x) / res).astype(np.int32)
        my = ((map_y - origin_y) / res).astype(np.int32)

        in_bounds = (mx >= 0) & (mx < width) & (my >= 0) & (my < height)
        mx, my    = mx[in_bounds], my[in_bounds]
        map_x, map_y = map_x[in_bounds], map_y[in_bounds]

        if len(mx) == 0:
            self._frontier_bearings = np.empty(0)
            return

        # --- find points landing on unknown cells ---
        indices     = (my * width + mx).astype(np.int32)
        cell_values = np.array(self.map_data.data, dtype=np.int16)[indices]
        frontier_mask = (cell_values == -1)

        fx = map_x[frontier_mask]
        fy = map_y[frontier_mask]

        if len(fx) == 0:
            self._frontier_bearings   = np.empty(0)
            self._frontier_count_last = 0
            return

        # --- distance filter: discard points too close to the robot ---
        # Nearby unknown cells are about to be mapped as the robot drives through
        # them; including them floods the bearing array and generates useless
        # waypoints right next to the robot.
        dists = np.hypot(fx - self.robot_x, fy - self.robot_y)
        far_mask = dists >= self.min_frontier_nav_dist
        fx_far, fy_far = fx[far_mask], fy[far_mask]

        if len(fx_far) == 0:
            self._frontier_bearings   = np.empty(0)
            self._frontier_count_last = 0
            return

        # --- robot-relative bearings (for live sector scoring) ---
        world_bearings = np.arctan2(fy_far - self.robot_y, fx_far - self.robot_x)
        rel_bearings   = (world_bearings - self.current_yaw + math.pi) % (2 * math.pi) - math.pi

        # Subsample if still very large so sector scoring stays fast
        if len(rel_bearings) > self.max_frontier_bearings:
            step = len(rel_bearings) // self.max_frontier_bearings
            rel_bearings = rel_bearings[::step]

        self._frontier_bearings   = rel_bearings
        self._frontier_count_last = len(fx_far)

        # --- merge into persistent waypoint memory ---
        self._add_frontier_waypoints(fx_far, fy_far)

    def scan_callback(self, msg: LaserScan):
        self._scan_count += 1

        if self.state == State.STOPPED:
            self._publish_stop()
            return

        front_min = self._front_cone_min(msg)

        if self.state == State.MOVING_FORWARD:
            if front_min is None or front_min < self.obstacle_threshold:
                self._start_turn(msg)
            else:
                self._publish_forward()

        elif self.state == State.TURNING:
            yaw_err = self._angle_diff(self.turn_target_yaw, self.current_yaw)

            if abs(yaw_err) <= self.align_tolerance:
                if front_min is not None and front_min >= self.free_threshold:
                    self.turn_target_yaw = None
                    # Resume frontier navigation if one was active, otherwise free-roam
                    if self.nav_goal is not None:
                        self.get_logger().info(
                            'Clear path after turn. Resuming frontier navigation.'
                        )
                        self.state = State.NAVIGATING_TO_FRONTIER
                    else:
                        self.get_logger().info('Clear path. Moving forward.')
                        self.state = State.MOVING_FORWARD
                        self._publish_forward()
                else:
                    self._try_next_candidate()
            else:
                self._publish_turn(1.0 if yaw_err > 0 else -1.0)

        elif self.state == State.NAVIGATING_TO_FRONTIER:
            self._navigate_to_frontier(msg, front_min)

    # ------------------------------------------------------------------
    # Frontier navigation
    # ------------------------------------------------------------------
    def _navigate_to_frontier(self, msg: LaserScan, front_min: Optional[float]):
        """Drive toward self.nav_goal, handling obstacles and arrival."""
        if self.nav_goal is None:
            self._start_frontier_nav()
            return

        goal_x, goal_y = self.nav_goal
        dist = math.hypot(goal_x - self.robot_x, goal_y - self.robot_y)

        # --- arrived ---
        if dist < self.frontier_goal_radius:
            self.get_logger().info(
                f'Arrived at frontier waypoint ({goal_x:.2f}, {goal_y:.2f}). '
                f'{len(self._frontier_waypoints) - 1} waypoint(s) remaining.'
            )
            self._frontier_waypoints = [
                w for w in self._frontier_waypoints
                if math.hypot(w[0] - goal_x, w[1] - goal_y) > self.frontier_goal_radius
            ]
            self.nav_goal = None
            # Try the next waypoint; if none left, resume free-roam
            if self._frontier_waypoints:
                self._start_frontier_nav()
            else:
                self.get_logger().info('All frontier waypoints reached. Resuming free-roam.')
                self.state = State.MOVING_FORWARD
                self._publish_forward()
            return

        # --- obstacle en route: hand off to turn logic ---
        if front_min is not None and front_min < self.obstacle_threshold:
            self.get_logger().info(
                f'Obstacle during frontier nav to ({goal_x:.2f}, {goal_y:.2f}). Turning.'
            )
            # nav_goal stays set so TURNING knows to resume nav afterwards
            self._start_turn(msg)
            return

        # --- steer toward goal ---
        goal_bearing = math.atan2(goal_y - self.robot_y, goal_x - self.robot_x)
        heading_err  = self._angle_diff(goal_bearing, self.current_yaw)

        if abs(heading_err) > self.frontier_heading_tol:
            self._publish_turn(1.0 if heading_err > 0 else -1.0)
        else:
            self._publish_forward()

    def _start_frontier_nav(self):
        """Prune mapped waypoints, pick the nearest surviving one, begin nav."""
        self._prune_frontier_waypoints()

        # Drop any waypoints that are now too close (the robot has moved near them
        # since they were first recorded)
        reachable = [
            w for w in self._frontier_waypoints
            if math.hypot(w[0] - self.robot_x, w[1] - self.robot_y) >= self.min_frontier_nav_dist
        ]
        discarded = len(self._frontier_waypoints) - len(reachable)
        if discarded:
            self.get_logger().info(
                f'Discarded {discarded} waypoint(s) now within {self.min_frontier_nav_dist:.1f} m.'
            )
        self._frontier_waypoints = reachable

        if not self._frontier_waypoints:
            self.get_logger().info('No distant frontier waypoints available. Stopping.')
            self.state = State.STOPPED
            self._publish_stop()
            return

        # Pick the nearest waypoint that still meets the distance requirement
        best = min(
            self._frontier_waypoints,
            key=lambda w: math.hypot(w[0] - self.robot_x, w[1] - self.robot_y),
        )
        dist = math.hypot(best[0] - self.robot_x, best[1] - self.robot_y)

        self.nav_goal = best
        self.state    = State.NAVIGATING_TO_FRONTIER
        self.get_logger().info(
            f'Navigating to frontier waypoint ({best[0]:.2f}, {best[1]:.2f})  '
            f'dist={dist:.2f} m  ({len(self._frontier_waypoints)} waypoint(s) in memory)'
        )

    def _add_frontier_waypoints(self, fx: np.ndarray, fy: np.ndarray):
        """
        Cluster new frontier points onto a grid and merge centroids that are
        far enough from existing waypoints into the persistent list.
        """
        sep = self.min_waypoint_sep

        # Snap to grid → collect points per cell
        grid: dict = {}
        for x, y in zip(fx.tolist(), fy.tolist()):
            key = (round(x / sep), round(y / sep))
            if key not in grid:
                grid[key] = []
            grid[key].append((x, y))

        added = 0
        for pts in grid.values():
            if len(self._frontier_waypoints) >= self.max_frontier_waypoints:
                break
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            too_close = any(
                math.hypot(cx - wx, cy - wy) < sep
                for wx, wy in self._frontier_waypoints
            )
            if not too_close:
                self._frontier_waypoints.append((cx, cy))
                added += 1

        if added > 0:
            self.get_logger().debug(
                f'Added {added} frontier waypoint(s). Total: {len(self._frontier_waypoints)}'
            )

    def _prune_frontier_waypoints(self):
        """Remove waypoints whose map cells are no longer unknown."""
        if self.map_data is None:
            return

        res      = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        width    = self.map_data.info.width
        height   = self.map_data.info.height
        data     = self.map_data.data

        surviving = []
        for wx, wy in self._frontier_waypoints:
            mx = int((wx - origin_x) / res)
            my = int((wy - origin_y) / res)
            if not (0 <= mx < width and 0 <= my < height):
                surviving.append((wx, wy))   # outside map bounds → still unknown
                continue
            if data[my * width + mx] == -1:
                surviving.append((wx, wy))   # still unknown → keep

        removed = len(self._frontier_waypoints) - len(surviving)
        if removed > 0:
            self.get_logger().info(
                f'Pruned {removed} mapped frontier waypoint(s). '
                f'{len(surviving)} remaining.'
            )
        self._frontier_waypoints = surviving

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------
    def _start_turn(self, msg: LaserScan):
        candidates = self._rank_sectors(msg)
        if not candidates:
            self.get_logger().warn('No valid scan sectors. Stopping.')
            self.state = State.STOPPED
            self._publish_stop()
            return

        # Prefer the side with more immediate clearance so the robot turns
        # away from the obstacle rather than into it.
        left_clear, right_clear = self._side_clearance(msg)
        preferred_sign = 1.0 if left_clear >= right_clear else -1.0

        preferred, fallback = [], []
        for score, world_yaw in candidates:
            yaw_err = self._angle_diff(world_yaw, self.current_yaw)
            (preferred if yaw_err * preferred_sign >= 0 else fallback).append((score, world_yaw))

        self._turn_candidates = preferred + fallback

        n_frontiers = len(self._frontier_bearings)
        tier = ('frontier+lidar' if n_frontiers > 0 and self.map_data
                else 'ray+lidar'  if self.map_data
                else 'lidar-only')
        side_str = (f'left={left_clear:.2f} m  right={right_clear:.2f} m  '
                    f'→ prefer {"left(CCW)" if preferred_sign > 0 else "right(CW)"}')
        self.get_logger().info(
            f'Obstacle (<{self.obstacle_threshold} m). '
            f'{len(candidates)} candidate(s) [{tier}, {n_frontiers} frontier pts] | '
            f'{side_str}'
        )
        self._try_next_candidate()

    def _try_next_candidate(self):
        if not self._turn_candidates:
            # All immediate options exhausted — fall back to remembered frontiers
            self._prune_frontier_waypoints()
            if self._frontier_waypoints:
                self.get_logger().info(
                    f'Immediate candidates exhausted. '
                    f'Falling back to {len(self._frontier_waypoints)} remembered '
                    f'frontier waypoint(s).'
                )
                self.nav_goal = None        # _start_frontier_nav will assign it
                self._start_frontier_nav()
            else:
                self.get_logger().info(
                    'All candidates and frontier waypoints exhausted. Stopping.'
                )
                self.state = State.STOPPED
                self._publish_stop()
            return

        score, world_yaw = self._turn_candidates.pop(0)
        self.turn_target_yaw = world_yaw
        self.state = State.TURNING

        yaw_err  = self._angle_diff(world_yaw, self.current_yaw)
        turn_dir = 'CCW' if yaw_err > 0 else 'CW'
        self.get_logger().info(
            f'Turning {turn_dir} → {math.degrees(world_yaw):.1f}° '
            f'(score={score:.3f}, Δyaw={math.degrees(yaw_err):.1f}°, '
            f'{len(self._turn_candidates)} alt(s) left)'
        )
        self._publish_turn(1.0 if yaw_err > 0 else -1.0)

    # ------------------------------------------------------------------
    # Sector ranking
    # ------------------------------------------------------------------
    def _rank_sectors(self, msg: LaserScan) -> List[Tuple[float, float]]:
        """
        Score every non-rear sector and return them sorted best-first as
        (score, world_yaw) pairs.
        """
        has_frontiers = len(self._frontier_bearings) > 0 and self.map_data is not None
        has_map       = self.map_data is not None

        results: List[Tuple[float, float]] = []
        n_sectors = max(1, int(round((msg.angle_max - msg.angle_min) / self.sector_width)))

        for s in range(n_sectors):
            lo  = msg.angle_min + s * self.sector_width
            hi  = lo + self.sector_width
            mid = (lo + hi) / 2.0

            if abs(mid) > math.radians(135.0):   # exclude rear arc
                continue

            ranges_in_sector = [
                r for i, r in enumerate(msg.ranges)
                if lo <= (msg.angle_min + i * msg.angle_increment) < hi
                and msg.range_min < r < msg.range_max
                and math.isfinite(r)
            ]
            if not ranges_in_sector:
                continue

            world_yaw   = self._normalize_angle(self.current_yaw + mid)
            lidar_score = (sum(ranges_in_sector) / len(ranges_in_sector)) / msg.range_max

            if has_frontiers:
                expl_score = self._frontier_sector_score(lo, hi)
            elif has_map:
                expl_score = self._ray_exploration_score(world_yaw)
            else:
                expl_score = 0.0

            score = (self.exploration_weight * expl_score
                     + (1.0 - self.exploration_weight) * lidar_score)
            results.append((score, world_yaw))

        results.sort(key=lambda t: t[0], reverse=True)
        return results

    def _frontier_sector_score(self, sector_lo: float, sector_hi: float) -> float:
        """Fraction of visible-frontier bearings inside [sector_lo, sector_hi)."""
        if len(self._frontier_bearings) == 0:
            return 0.0
        in_sector = np.sum(
            (self._frontier_bearings >= sector_lo) & (self._frontier_bearings < sector_hi)
        )
        return float(in_sector) / len(self._frontier_bearings)

    def _ray_exploration_score(self, world_angle: float) -> float:
        """Tier-2: fraction of ray cells that are unknown, stopping at occupied cells."""
        if self.map_data is None:
            return 0.0

        res      = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        width    = self.map_data.info.width
        height   = self.map_data.info.height
        data     = self.map_data.data

        n_steps = max(1, int(self.ray_length / res))
        cos_a   = math.cos(world_angle)
        sin_a   = math.sin(world_angle)
        unknown = 0
        total   = 0

        for k in range(1, n_steps + 1):
            d  = k * res
            mx = int((self.robot_x + d * cos_a - origin_x) / res)
            my = int((self.robot_y + d * sin_a - origin_y) / res)

            if not (0 <= mx < width and 0 <= my < height):
                break

            cell = data[my * width + mx]
            total += 1

            if cell == -1:
                unknown += 1
            elif cell >= self.occupied_threshold:
                break

        return unknown / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Laser helpers
    # ------------------------------------------------------------------
    def _side_clearance(self, msg: LaserScan) -> Tuple[float, float]:
        """
        Mean range in the 20°–100° flank on each side (front cone excluded so
        the blocked obstacle doesn't bias the comparison).
        Returns (left_mean, right_mean) in metres.
        """
        left_lo,  left_hi  =  math.radians(20),  math.radians(100)
        right_lo, right_hi = -math.radians(100), -math.radians(20)

        left_ranges, right_ranges = [], []
        for i, r in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if not (msg.range_min < r < msg.range_max and math.isfinite(r)):
                continue
            if left_lo <= angle <= left_hi:
                left_ranges.append(r)
            elif right_lo <= angle <= right_hi:
                right_ranges.append(r)

        left_mean  = sum(left_ranges)  / len(left_ranges)  if left_ranges  else 0.0
        right_mean = sum(right_ranges) / len(right_ranges) if right_ranges else 0.0
        return left_mean, right_mean

    def _front_cone_min(self, msg: LaserScan) -> Optional[float]:
        valid = [
            r for i, r in enumerate(msg.ranges)
            if abs(msg.angle_min + i * msg.angle_increment) <= self.front_half_angle
            and msg.range_min < r < msg.range_max
            and math.isfinite(r)
        ]
        return min(valid) if valid else None

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------
    def _publish_forward(self):
        msg = Twist()
        msg.linear.x = self.forward_speed
        self.cmd_pub.publish(msg)

    def _publish_turn(self, direction: float = 1.0):
        msg = Twist()
        msg.angular.z = self.turn_speed * direction
        self.cmd_pub.publish(msg)

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    # ------------------------------------------------------------------
    # Math utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Signed shortest angular difference a − b in (−π, π]."""
        diff = (a - b) % (2 * math.pi)
        if diff > math.pi:
            diff -= 2 * math.pi
        return diff

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Wrap angle to (−π, π]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousTraversalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
