from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='fsm_project', executable='observation_logger', output='screen'),
        Node(package='fsm_project', executable='state_machine', output='screen'),
        Node(package='fsm_project', executable='control_publisher', output='screen'),
    ])
