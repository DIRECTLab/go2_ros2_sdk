from setuptools import setup, find_packages

package_name = 'autonomous_traversal'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Autonomous free-space traversal for the Unitree Go2 robot.',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'autonomous_traversal_node = autonomous_traversal.autonomous_traversal_node:main',
        ],
    },
)
