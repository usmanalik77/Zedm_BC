from setuptools import find_packages, setup

package_name = 'rgbd_converter'

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
    maintainer='usman',
    maintainer_email='usman@todo.todo',
    description='Getting RGB-D 4 dimensional image from ZED',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rgbd_node = rgbd_converter.rgbd_node:main',
        ],
    },
)
