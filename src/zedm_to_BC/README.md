# **Behavior Cloning Workflow**

## **Overview**
This project implements a behavior cloning pipeline for autonomous control. It uses RGBD images and control signals (steering, throttle) to train a neural network.

---

## Run ZED CAMERA
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedm

## **Setup**

### **Prerequisites**
- ROS 2 (Humble or Rolling), Python 3.8+
- Install dependencies:
  ```bash
  pip install tensorflow numpy

## Directory Structure

ros2_ws/
├── src/
│   ├── zedm_to_BC/
│   │   ├── zedm_to_BC/
│   │   │   ├── scripts/
│   │   │   │   ├── inference_node.py
│   │   │   │   ├── control_publisher.py
│   │   │   │   ├── combine_data.py
│   │   │   ├── training/
│   │   │   │   ├── behavior_cloning_training.py
│   │   │   ├── launch/
│   │   │   │   ├── inference_launch.py
│   │   │   ├── models/
│   │   │   │   ├── behavior_cloning_model.py
│   │   │   ├── observation_logger/
│   │   │   │   ├── observation_logger.py
│   │   ├── package.xml
│   │   ├── setup.py




### Build ROS 2 Package
colcon build --packages-select zedm_to_BC
source install/setup.bash
source ~/ros2_ws/install/setup.bash


### Executions


ros2 launch zedm_to_BC inference_launch.py



## **Workflow**
1. **Data Collection**: Collect images and controls using ROS 2 nodes.
2. **Data Preprocessing**: Combine data into a single `.npy` file.
3. **Model Training**: Train a CNN to predict controls from images.
4. **Model Deployment**: Deploy the trained model for inference.

---



### 1. Data Collection
Run the ROS 2 nodes:


ros2 run zedm_to_BC logger       # Logs RGBD images and controls

python3 /home/usman/ros2_ws/src/zedm_to_BC/zedm_to_BC/scripts/behavior_controls.py                            # Publishes example controls

### 2. Data Preprocessing

Generate training data:

python3 /home/usman/ros2_ws/src/zedm_to_BC/zedm_to_BC/scripts/combine_data.py


### 3. Model Training
ros2 run zedm_to_BC trainer


### 4. Model Deployment

Deploy the trained model for autonomous control. This can involve integrating the model into an inference pipeline or system.


# File Structure
## ROS Nodes:

### control_publisher.py: 
Publishes control commands to /behavior_controls.
### observation_logger.py: 
Logs images and controls to CSV and .npy.


## Scripts:
### combine_data.py: 
Combines collected data into a single .npy file for training.
### behavior_cloning_training.py:
Trains a CNN-based model using the .npy file.


## Output
### Training Data: 
Stored in /home/usman/collected_data/collected_data.npy.
### Trained Model: 
Saved to /home/usman/behavior_cloning_model.

## Running Inference
Run the inference node directly:

ros2 run zedm_to_BC inference

This will:

1. Subscribe to the RGB image stream from the ZED Mini camera.
2. Predict control commands (steering, throttle).
3. Log predictions to:

/home/usman/inference_logs/inference_outputs.csv

## Launch Inference
Use the launch file to start the inference node:

ros2 launch zedm_to_BC inference_launch.py

This simplifies the process by automatically configuring the inference node.



# Future Extensions
1. Enhance data augmentation for better generalization.
2. Improve inference integration with real-time systems.
3. Add visualization tools for debugging and evaluation.