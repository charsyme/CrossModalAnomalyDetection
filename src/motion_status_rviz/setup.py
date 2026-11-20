from setuptools import find_packages, setup
setup(
    name='motion_status_rviz', version='0.1.0', packages=find_packages(),
    data_files=[("share/ament_index/resource_index/packages", ["resource/motion_status_rviz"]), ("share/motion_status_rviz", ["package.xml"])],
    package_data={},
    include_package_data=True,
    install_requires=['setuptools'], zip_safe=False,
    maintainer='Research software maintainer', maintainer_email='maintainer@example.org',
    description='Odometry and anomaly status image for RViz', license='Apache-2.0',
    entry_points={'console_scripts': [
            'motion_status_rviz = motion_status_rviz.motion_status_rviz:main',
    ]},
)
