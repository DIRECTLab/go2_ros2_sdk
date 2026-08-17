"""
Presentation layer - user interface (ROS2 node)
"""
from .go2_driver_node import Go2DriverNode

# raw_lidar_node is deliberately NOT re-exported here. Its entry point imports
# the submodule directly, so re-exporting would only force every go2_driver_node
# process to load numpy and the DDS adapter it never uses.
__all__ = ['Go2DriverNode'] 