from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='zedm_to_BC',
            executable='logger',  # Runs observation_logger.py
            name='observation_logger',
            output='screen'
        ),
        Node(
            package='zedm_to_BC',
            executable='scripts',  # Runs control_publisher.py
            name='control_publisher',
            output='screen'
        )
    ])
