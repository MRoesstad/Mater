#!/home/magnus/anaconda3/envs/cosypose/bin/python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np

from cosypose.utils.tensor_collection import PandasTensorCollection
from utils.predictor import RigidObjectPredictor  # you'll implement or adapt this
from utils.camera import make_cameras             # you'll adapt this too
from cosypose.config import LOCAL_DATA_DIR

class CosyPoseSubscriber(Node):
    def __init__(self):
        super().__init__('cosypose_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            'webcam/image_raw',
            self.image_callback,
            10
        )

        self.predictor = RigidObjectPredictor(
            detector_run_id='detector-bop-icbin-pbr--947409',
            coarse_run_id='coarse-bop-icbin-pbr--915044',
            refiner_run_id='refiner-bop-icbin-pbr--841882'
        )
        self.get_logger().info('CosyPose Subscriber Initialized.')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img = frame[..., ::-1]  # BGR → RGB
            cameras, _ = make_cameras([{
                'cx': 325.5, 'cy': 253.5,
                'fx': 572.4114, 'fy': 573.57043,
                'resolution': (480, 640),
            }])

            predictions = self.predictor([img], cameras)

            for obj in predictions:
                self.get_logger().info(f"Detected: {obj['label']} at pose:\n{obj['pose']}")
        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CosyPoseSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()