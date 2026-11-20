from setuptools import find_packages, setup
setup(
    name='depth_estimation', version='0.1.0', packages=find_packages(),
    data_files=[("share/ament_index/resource_index/packages", ["resource/depth_estimation"]), ("share/depth_estimation", ["package.xml"])],
    package_data={},
    include_package_data=True,
    install_requires=['setuptools'], zip_safe=False,
    maintainer='Research software maintainer', maintainer_email='maintainer@example.org',
    description='Depth Anything V2 ROS 2 relative depth estimation', license='Apache-2.0',
    entry_points={'console_scripts': [
            'depth_node = depth_estimation.depth_node:main',
    ]},
)
