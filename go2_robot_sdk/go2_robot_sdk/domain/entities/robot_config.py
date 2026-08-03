# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List


@dataclass
class RobotConfig:
    """Robot configuration parameters"""
    robot_ip_list: List[str]
    token: str
    aes_key: str
    conn_type: str
    enable_video: bool
    decode_lidar: bool
    publish_raw_voxel: bool
    obstacle_avoidance: bool
    conn_mode: str  # 'single' or 'multi'
    # Prepended to every TF frame this robot publishes, so several robots can
    # share one /tf tree. Empty means unprefixed (original behaviour).
    tf_prefix: str = ''

    @classmethod
    def from_params(cls, robot_ip: str, token: str, conn_type: str,
                   enable_video: bool, decode_lidar: bool,
                   publish_raw_voxel: bool, obstacle_avoidance: bool,
                   tf_prefix: str = ''):
        """Создание конфигурации из параметров"""
        robot_ip_list = robot_ip.replace(" ", "").split(",")
        conn_mode = "single" if (
            len(robot_ip_list) == 1 and conn_type != "cyclonedds") else "multi"

        return cls(
            robot_ip_list=robot_ip_list,
            token=token,
            aes_key=aes_key,
            conn_type=conn_type,
            enable_video=enable_video,
            decode_lidar=decode_lidar,
            publish_raw_voxel=publish_raw_voxel,
            obstacle_avoidance=obstacle_avoidance,
            conn_mode=conn_mode,
            tf_prefix=tf_prefix.strip().strip('/')
        )

    def frame(self, name: str) -> str:
        """Namespace a frame id: '' -> 'base_link', 'go2' -> 'go2/base_link'."""
        return f"{self.tf_prefix}/{name}" if self.tf_prefix else name
