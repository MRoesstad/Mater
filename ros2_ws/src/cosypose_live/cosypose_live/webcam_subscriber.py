#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class WebcamSubscriber(Node):
    def __init__(self):
        super().__init__('webcam_subscriber')
        self.publisher_ = self.create_publisher(Image, 'webcam/image_raw', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.03, self.timer_callback)  # ~30 FPS
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.get_logger().error('Could not open webcam.')
        else:
            self.get_logger().info('Webcam successfully opened.')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('Failed to grab frame from webcam.')
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)
        self.get_logger().debug('Published image frame')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebcamSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
