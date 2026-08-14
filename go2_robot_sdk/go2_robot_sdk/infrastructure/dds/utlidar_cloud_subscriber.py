# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

"""
CycloneDDS adapter for the Go2's onboard L2 lidar.

The robot publishes its per-scan raw cloud on the DDS topic ``rt/utlidar/cloud``,
reachable only over the Ethernet link. The WebRTC channel this repo normally uses
carries ``rt/utlidar/voxel_map_compressed`` instead, which is quantised to a 5 cm
lattice *on the robot* before transport -- see docs/SWARM_SLAM_FINDINGS.md -- so
it is not a substitute when the true per-scan sampling matters.

This adapter is deliberately ROS-free: it hands raw unitree_sdk2py IDL messages
to a callback and knows nothing about sensor_msgs.

unitree_sdk2py is imported lazily inside :meth:`UtlidarCloudSubscriber.start`
rather than at module import. That is not stylistic -- a machine carrying several
CycloneDDS builds (ROS's apt ``libddsc``, unitree_sdk2's thirdparty copy, a
from-source build for the Python bindings) can abort the *process* on import when
the wrong one is linked. Keeping the import lazy confines that blast radius to
the node that actually wants DDS.
"""

import logging
import threading
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Field layout of rt/utlidar/cloud, confirmed by live introspection against the
# robot. Deliberately not derived from generic PointCloud2 conventions: the
# padding at 12-15, 22-23 and 28-31 is real, and intensity does NOT sit at
# offset 12 the way most drivers place it.
#   name       offset  datatype (sensor_msgs/PointField enum)
CONFIRMED_FIELD_LAYOUT: Tuple[Tuple[str, int, int], ...] = (
    ('x',          0, 7),   # FLOAT32
    ('y',          4, 7),   # FLOAT32
    ('z',          8, 7),   # FLOAT32
    ('intensity', 16, 7),   # FLOAT32
    ('ring',      20, 4),   # UINT16
    ('time',      24, 7),   # FLOAT32
)
CONFIRMED_POINT_STEP = 32

# Guards ChannelFactoryInitialize, which is process-global in unitree_sdk2py and
# raises if called twice.
_factory_lock = threading.Lock()
_factory_initialized = False
_factory_signature: Optional[Tuple[int, str]] = None


class DdsUnavailableError(RuntimeError):
    """unitree_sdk2py could not be imported or initialised."""


def _import_sdk():
    """Import unitree_sdk2py, returning (ChannelFactoryInitialize, ChannelSubscriber, PointCloud2_)."""
    try:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize, ChannelSubscriber)
    except Exception as exc:
        raise DdsUnavailableError(
            f"unitree_sdk2py is not importable ({exc}). Install it with "
            "'pip install -e .' from a checkout of "
            "https://github.com/unitreerobotics/unitree_sdk2_python"
        ) from exc

    # The IDL package moved between unitree_sdk2py releases; try both spellings
    # before giving up so this node is not pinned to one revision.
    point_cloud_type = None
    errors = []
    for module_path in ('unitree_sdk2py.idl.sensor_msgs.msg.dds_',
                        'unitree_sdk2py.idl.sensor_msgs.msg.dds_._PointCloud2_'):
        try:
            module = __import__(module_path, fromlist=['PointCloud2_'])
            point_cloud_type = getattr(module, 'PointCloud2_')
            break
        except Exception as exc:
            errors.append(f"{module_path}: {exc}")

    if point_cloud_type is None:
        raise DdsUnavailableError(
            "unitree_sdk2py imported, but its sensor_msgs PointCloud2_ IDL type "
            "was not found. Tried:\n  " + "\n  ".join(errors))

    return ChannelFactoryInitialize, ChannelSubscriber, point_cloud_type


def _as_bytes(data) -> bytes:
    """Normalise a DDS octet sequence to bytes.

    cyclonedds-python hands back bytes for some builds and a numpy uint8 array
    for others, depending on how the sequence is annotated.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    try:
        return data.tobytes()          # numpy array
    except AttributeError:
        return bytes(bytearray(data))  # list of ints


def cloud_data_bytes(msg) -> bytes:
    """Point payload of a DDS cloud message, as bytes."""
    return _as_bytes(msg.data)


def cloud_stamp_ns(msg) -> Optional[int]:
    """Header stamp of a DDS cloud message in nanoseconds, or None if absent.

    These are plain attributes on the Python IDL types, not the accessor methods
    the C++ unitree_sdk2 exposes -- ``msg.header.stamp.sec``, never
    ``msg.header.stamp.sec()``.
    """
    try:
        stamp = msg.header.stamp
    except AttributeError:
        return None

    sec = getattr(stamp, 'sec', None)
    if sec is None:
        return None
    # 'nanosec' is the ROS2 spelling; 'nsec' shows up in some IDL generations.
    nanosec = getattr(stamp, 'nanosec', None)
    if nanosec is None:
        nanosec = getattr(stamp, 'nsec', 0)
    return int(sec) * 1_000_000_000 + int(nanosec)


def cloud_field_layout(msg) -> List[Tuple[str, int, int]]:
    """(name, offset, datatype) triples the DDS message declares for itself."""
    layout = []
    for field in getattr(msg, 'fields', []) or []:
        layout.append((
            str(getattr(field, 'name', '')),
            int(getattr(field, 'offset', -1)),
            int(getattr(field, 'datatype', -1)),
        ))
    return layout


class UtlidarCloudSubscriber:
    """Subscribes to the robot's raw lidar DDS topic over Ethernet.

    Callbacks fire on a CycloneDDS listener thread, not on the caller's thread.
    """

    def __init__(self,
                 network_interface: str = '',
                 domain_id: int = 0,
                 topic: str = 'rt/utlidar/cloud',
                 queue_len: int = 10):
        self.network_interface = network_interface.strip()
        self.domain_id = int(domain_id)
        self.topic = topic
        self.queue_len = int(queue_len)
        self._subscriber = None

    def start(self, on_cloud: Callable[[object], None]) -> None:
        """Initialise the DDS channel factory and begin delivering clouds."""
        global _factory_initialized, _factory_signature

        channel_factory_initialize, channel_subscriber, point_cloud_type = _import_sdk()

        # ChannelFactoryInitialize takes networkInterface=None to mean "every
        # interface"; an empty parameter string has to become None, not ''.
        interface = self.network_interface or None
        signature = (self.domain_id, self.network_interface)

        with _factory_lock:
            if not _factory_initialized:
                try:
                    channel_factory_initialize(self.domain_id, interface)
                except Exception as exc:
                    raise DdsUnavailableError(
                        f"ChannelFactoryInitialize(domain={self.domain_id}, "
                        f"iface={interface!r}) failed: {exc}") from exc
                _factory_initialized = True
                _factory_signature = signature
            elif _factory_signature != signature:
                logger.warning(
                    "DDS channel factory already initialised as %s; ignoring "
                    "request for %s (it is process-global)",
                    _factory_signature, signature)

        subscriber = channel_subscriber(self.topic, point_cloud_type)
        subscriber.Init(on_cloud, self.queue_len)
        self._subscriber = subscriber
        logger.info("Subscribed to DDS topic %s on interface %s (domain %d)",
                    self.topic, interface or '<all>', self.domain_id)

    def stop(self) -> None:
        """Close the DDS subscriber. Safe to call when never started."""
        if self._subscriber is None:
            return
        try:
            self._subscriber.Close()
        except Exception as exc:
            logger.warning("Error closing DDS subscriber: %s", exc)
        finally:
            self._subscriber = None
