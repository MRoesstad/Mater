#!/usr/bin/env python3
import rclpy #pythin libary for ROS2
from rclpy.node import Node

class MyNode(Node):#now inherents everything form rclpy. this is the node class
    def __init__(self):
        super().__init__("first_node")#this is the nodes name 
        self.create_timer(1.0,self.timer_callback)
    def timer_callback(self):
        self.get_logger().info("Hello you")


def main(args=None):
    rclpy.init(args=args)
    #creates nodes inside here. Nodes are used with object oriented programming
    node = MyNode()
    rclpy.spin(node) #keeps it alive 
    #
    rclpy.shutdown()

if __name__ == '__main__':
    main()