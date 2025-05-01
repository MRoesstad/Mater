#!/home/magnus/anaconda3/envs/cosypose/bin/python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import traceback #midlertidig 

from cosypose.utils.tensor_collection import PandasTensorCollection
from utils.predictor import RigidObjectPredictor  
from utils.camera import make_cameras             
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
        ####################### Change these based on the model you want to apply ############################## 
        self.predictor = RigidObjectPredictor(
            detector_run_id='detector-bop-ycbv-synt+real--292971',
            coarse_run_id='coarse-bop-ycbv-synt+real--822463',
            refiner_run_id='refiner-bop-ycbv-synt+real--631598'
        )
        ########################################################################################################
        self.get_logger().info('CosyPose Subscriber Initialized.')

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format (BGR)
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img = frame[..., ::-1]  # Convert to RGB

            # Define camera intrinsics
            cameras = make_cameras(
                resolution=(480, 640),
                fx=572.4114, fy=573.57043,
                cx=325.5, cy=253.5
            )

            # Run prediction
            predictions = self.predictor([img], cameras)
            infos = predictions.infos

            # === Filtering ===
            scores = infos.get('score', None)
            score_mask = torch.ones(len(predictions), dtype=torch.bool)

            if scores is not None:
                if not torch.is_tensor(scores):
                    scores = torch.tensor(scores.values if hasattr(scores, 'values') else scores)
                score_mask = scores > 0.85

            z = predictions.poses[:, 2, 3]  # Z-depth of object center
            depth_mask = (z > 0.2) & (z < 1.5)

            final_mask = score_mask & depth_mask
            predictions = predictions[final_mask.numpy()]

            # === Log valid detections ===
            for i in range(len(predictions)):
                label = predictions.infos['label'][i]
                pose = predictions.poses[i]
                self.get_logger().info(f"Detected: {label} at pose:\n{pose}")

            # Show webcam frame
            cv2.imshow("CosyPose Detection", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in callback:\n{traceback.format_exc()}")





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
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()