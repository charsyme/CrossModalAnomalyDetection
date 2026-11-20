from setuptools import find_packages, setup
setup(
    name='cross_modal_anomaly_detection', version='0.1.0', packages=find_packages(),
    data_files=[("share/ament_index/resource_index/packages", ["resource/cross_modal_anomaly_detection"]), ("share/cross_modal_anomaly_detection", ["package.xml"])]+[("share/cross_modal_anomaly_detection/config", ['config/detector.yaml'])],
    package_data={},
    include_package_data=True,
    install_requires=['setuptools'], zip_safe=False,
    maintainer='Research software maintainer', maintainer_email='maintainer@example.org',
    description='RGB and LiDAR cross-modal anomaly detection with spatial consistency', license='Apache-2.0',
    entry_points={'console_scripts': [
            'cross_modal_anomaly_detection = cross_modal_anomaly_detection.cross_modal_anomaly_detection:main',
    ]},
)
