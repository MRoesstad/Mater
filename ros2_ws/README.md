This fra have I dowloaded everything and set up a ros2 workspace

We write nodes inside packages, packages allows us to organize the code in a better way and dependencies.
Packages are created in src following the command:
ros2 pkg create name_it_something --build-type ament_cmake/python --dependencies rclpy
Each package can contain many nodes.
Packages can also have dependencies, e.g. package A depends on package B which depends on package C.

after creating the packefes go into it.
touch name_program.py  #This creates another file
chmod +x name_program.py #Makes in executable 
 
Make publisher and subscriber

----------General setup--------------------
Node for camera capture
Node for cosypose?? --subscribing to camera
Node for displaying? --subscribing to cosypose
Node for robot

