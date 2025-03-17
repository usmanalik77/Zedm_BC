from setuptools import setup, find_packages

package_name = 'fsm_project'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=['fsm_project', 'fsm_project.*']),
    install_requires=['setuptools', 'torch'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='FSM-based shape recognition with a PyTorch + Liquid Time Constant model.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'observation_logger = fsm_project.observation_logger.observation_logger:main',
            'user_key_node = fsm_project.scripts.user_key_node:main',
            'combine_data = fsm_project.scripts.combine_data:main',
            'training = fsm_project.training.behavior_cloning_training:main',
            'inference_decision = fsm_project.scripts.inference_decision_node:main', 
        ],
    },
)
