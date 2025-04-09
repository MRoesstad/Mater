# Master Thesis

This complete pipeline utelises CosyPose to recieve 6D pose for a custom object. 

## Installation guide 
Multiple points here 

## How to use it 
Activate the webcame subscriber with the command:
```
ros2 run cosypose_live webcam
```
This nodes acts as the camera feed source
This command runs the webcam publisher which opens you webcam. (Remember to change port to the correct one). Reads frames at 30 fps.
Then converts each frame to a ROS sensor_msgs/Images using CVBridge. Here it publishes frames on the topic /webcam/image_raw.

If you are unsure of which channel your camera is you can test with
```
ros2 run cosypose_live something
```
This node subscribes to the webcam and then livestreams it.

To activate the node that perfomrs the actual 6D pose estimation using cosypose use the command:
```
ros2 run cosypose_live pose
```
It subscribes to image_raw and converts each ROS image back to a
numpy opencv image. It then creates a camera intrinsics matrix and feed the image and intrinsics to your rigidobjectpredictor.
Finally it logs the 6D poses to the terminal.
