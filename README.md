# Master Thesis

This complete pipeline utelises CosyPose to recieve 6D pose for a custom object. 

## Warning 
This version of CosyPose is modified to be CPU only. If you want to train a model using CosyPose it is recomended using the unmoded version:
```
git clone --recurse-submodules https://github.com/Simple-Robotics/cosypose.git
```

## Installation guide 
To deal with the environment dilemma, I opted to use micromamba. Which is a standalone version of Conda, better suited to create an environment with different packages.
Documentation can be found at: https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html 

Installing Mater files with:
```
git clone https://github.com/MRoesstad/Mater.git
```
## Dataset creation
A guide on to implemt and create datasat can be found in the README under Blenderproc_ws

## How activate the pipeline
Activate the webcame subscriber with the command:
```
ros2 run cosypose_live webcam
```
This nodes acts as the camera feed source
This command runs the webcam publisher which opens you webcam. (Remember to change port to the correct one). Reads frames at 30 fps.
Then converts each frame to a ROS sensor_msgs/Images using CVBridge. Here it publishes frames on the topic /webcam/image_raw.

If you are unsure of which channel your camera is you can test with
```
ros2 run cosypose_live vision
```
This node subscribes to the webcam and then livestreams it.

To activate the node that perfomrs the actual 6D pose estimation using cosypose use the command:
```
ros2 run cosypose_live pose
```
or
```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python ~/Mater/ros2_ws/src/cosypose_live/cosypose_live/detection_node.py
```
It subscribes to image_raw and converts each ROS image back to a
numpy opencv image. It then creates a camera intrinsics matrix and feed the image and intrinsics to your rigidobjectpredictor.
Finally it logs the 6D poses to the terminal.

