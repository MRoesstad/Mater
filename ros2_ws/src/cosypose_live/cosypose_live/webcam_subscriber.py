#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class WebcamViewer(Node):
    def __init__(self):
        super().__init__('webcam_viewer')
        self.subscription = self.create_subscription(
            Image,
            '/webcam/image_raw',
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()
        self.get_logger().info("Subscribed to /webcam/image_raw")

    def listener_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            #bgr8 - color image with blue-green-red color order 
            cv2.imshow('Webcam Live Feed', frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebcamViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
