from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'go2_cslam_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chandler Justice',
    maintainer_email='chandler@todo.todo',
    description='Bridge between the Go2 ROS2 SDK and Swarm-SLAM (cslam)',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointcloud_bridge = go2_cslam_bridge.pointcloud_bridge:main',
        ],
    },
)
