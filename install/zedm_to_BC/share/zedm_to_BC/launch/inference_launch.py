import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    print("Launching Inference Node...")
    print("Executable: inference")
    print("Package: zedm_to_BC")
    
    return LaunchDescription([
        Node(
            package='zedm_to_BC',
            executable='inference',  # Match the executable name
            name='inference_node',  # Node name in ROS2
            output='screen',
        )
    ])
