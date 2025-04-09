import torch
import pandas as pd
from cosypose.utils.tensor_collection import PandasTensorCollection

def make_cameras(resolution=(480, 640), fx=572.4114, fy=573.5704, cx=325.2611, cy=242.0489):
    """
    Creates a camera object compatible with CosyPose.

    Parameters:
        resolution: Tuple[int, int] - (height, width) of the image
        fx, fy, cx, cy: float - Intrinsic parameters of the camera

    Returns:
        cameras: PandasTensorCollection with camera intrinsics
    """
    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]).unsqueeze(0)  # Add batch dimension

    infos = pd.DataFrame([{
        'batch_im_id': 0,
        'resolution': resolution
    }])

    cameras = PandasTensorCollection(K=K, infos=infos)
    return cameras
