from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'zedm_to_BC'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/zedm_to_BC']),
        ('share/zedm_to_BC', ['package.xml']),
        (os.path.join('share', 'zedm_to_BC', 'launch'), glob('launch/*.py')),
    ],

    install_requires=[
        'setuptools',
        'tensorflow',
        'numpy',
    ],
    zip_safe=True,
    maintainer='usman',
    maintainer_email='usman@todo.todo',
    description='Behavior cloning package for ROS2 using a Zed Mini camera stream.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'logger = zedm_to_BC.observation_logger.observation_logger:main',
            'trainer = zedm_to_BC.training.behavior_cloning_training:main',
            'scripts = zedm_to_BC.scripts.control_publisher:main',
            'inference = zedm_to_BC.scripts.inference_node:main',
            'combine_data = zedm_to_BC.scripts.combine_data:main', 
        ],
    },
)
