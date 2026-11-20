from setuptools import find_packages, setup
setup(
    name='rgb_preprocessing', version='0.1.0', packages=find_packages(),
    data_files=[("share/ament_index/resource_index/packages", ["resource/rgb_preprocessing"]), ("share/rgb_preprocessing", ["package.xml"])]+[("share/rgb_preprocessing/launch", ['launch/rgb_preprocessing.launch.py'])],
    package_data={},
    include_package_data=True,
    install_requires=['setuptools'], zip_safe=False,
    maintainer='Research software maintainer', maintainer_email='maintainer@example.org',
    description='RGB rectification and border masking', license='Apache-2.0',
    entry_points={'console_scripts': [
            'rgb_border_cleaner = rgb_preprocessing.border_cleaner:main',
    ]},
)
