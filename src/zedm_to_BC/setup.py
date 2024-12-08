from setuptools import find_packages, setup

package_name = 'zedm_to_BC'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'tensorflow',  # For the behavior cloning model
        'numpy',  # For numerical operations
    ],
    zip_safe=True,
    maintainer='usman',
    maintainer_email='usman@todo.todo',
    description='Behavior cloning package for ROS2 using a Zed Mini camera stream.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main = zedm_to_BC.main:main',
            'logger = zedm_to_BC.observation_logger.observation_logger:main',
            'trainer = zedm_to_BC.training.behavior_cloning_training:main',
            'scripts = zedm_to_BC.scripts.control_publisher:main',
        ],
    },
)
