import cv2
import torch
import numpy as np
from utils.predictor import RigidObjectPredictor
from utils.camera import make_cameras

def draw_axes(image, TCO, K, length=0.1):
    """Draw XYZ axes from TCO on the image using camera intrinsics K."""
    # Define axes in object coordinate system
    axes = np.array([
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length]
    ])
    axes_hom = np.hstack([axes, np.ones((4, 1))])
    
    # Transform to camera coordinates
    cam_points = (TCO @ axes_hom.T).T[:, :3]

    # Skip if any point is behind the camera
    if np.any(cam_points[:, 2] <= 0):
        return image

    # Project to 2D
    proj = (K @ cam_points.T).T
    proj /= proj[:, 2:3]
    pts = proj[:, :2].astype(int)

    # Draw the axes
    origin = tuple(pts[0])
    cv2.line(image, origin, tuple(pts[1]), (0, 0, 255), 2)   # X in red
    cv2.line(image, origin, tuple(pts[2]), (0, 255, 0), 2)   # Y in green
    cv2.line(image, origin, tuple(pts[3]), (255, 0, 0), 2)   # Z in blue
    return image

def main():
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 't' to take a snapshot and run CosyPose. Press 'q' to quit.")

    frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        cv2.imshow('Live Feed', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting.")
            break
        elif key == ord('t'):
            print("Snapshot taken.")
            break

    cap.release()
    cv2.destroyWindow('Live Feed')

    if frame is None:
        print("No frame captured.")
        return

    print("Running CosyPose on captured image...")
    rgb = frame[..., ::-1]

    cameras = make_cameras(
        resolution=(480, 640),
        fx=572.4114, fy=573.57043,
        cx=325.5, cy=253.5
    )

    predictor = RigidObjectPredictor(
            detector_run_id='detector-detector_bop-my_custom_bop-pbr--738712',
            coarse_run_id='bop-mycustombop-pbr-coarse-transnoise-zxyavg-197135',
            refiner_run_id='bop-mycustombop-pbr-refiner--470908'
        )    

    predictions = predictor([rgb], cameras)
    detections = getattr(predictor, 'detections', None)

    if len(predictions) == 0:
        print("No objects detected.")
        return

    scores_np = predictions.infos['score'].values
    z = predictions.poses[:, 2, 3].cpu().numpy()
    mask_np = (scores_np > 0.7) & (z > 0.2) & (z < 1.0)
    indices = np.where(mask_np)[0]
    predictions = predictions[indices]

    if len(predictions) == 0:
        print("No valid predictions after filtering.")
        return

    print(f"{len(predictions)} objects detected.")

    # Draw axes on a copy of the image
    K = torch.stack([cam.K for cam in cameras]).squeeze(0).cpu().numpy()
    result = frame.copy()
    for i in range(len(predictions)):
        label = predictions.infos['label'][i]
        TCO = predictions.poses[i].cpu().numpy()
        result = draw_axes(result, TCO, K)

        bbox = predictions.infos['bbox'][i]
        x1, y1, w, h = bbox
        x1, y1, x2, y2 = map(int, [x1, y1, x1 + w, y1 + h])
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, str(label), (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Optionally print pose
        print(f"\nDetected: {label}")
        print(TCO)

    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
