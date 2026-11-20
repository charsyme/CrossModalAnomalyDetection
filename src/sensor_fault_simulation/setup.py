from setuptools import find_packages, setup
setup(
    name='sensor_fault_simulation', version='0.1.0', packages=find_packages(),
    data_files=[("share/ament_index/resource_index/packages", ["resource/sensor_fault_simulation"]), ("share/sensor_fault_simulation", ["package.xml"])],
    package_data={'sensor_fault_simulation': ['camera_fault_assets/*.png']},
    include_package_data=True,
    install_requires=['setuptools'], zip_safe=False,
    maintainer='Research software maintainer', maintainer_email='maintainer@example.org',
    description='Optional safe RGB and LiDAR fault simulation for experiments', license='Apache-2.0',
    entry_points={'console_scripts': [
            'fault_injector = sensor_fault_simulation.fault_injector:main',
        'fault_controller = sensor_fault_simulation.fault_controller:main',
    ]},
)
