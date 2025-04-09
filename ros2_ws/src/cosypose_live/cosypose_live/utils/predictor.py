import torch
import numpy as np
import pandas as pd
import yaml  # For loading config files
import cosypose  # For accessing CosyPose modules

from cosypose.utils.tensor_collection import PandasTensorCollection
from cosypose.config import EXP_DIR
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.datasets.datasets_cfg import make_object_dataset
from cosypose.training.pose_models_cfg import create_model_pose
from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from cosypose.training.detector_models_cfg import create_model_detector
from cosypose.integrated.detector import Detector

def load_torch_model(run_id, *args, model_type='rigid'):
    """
    Load a Torch model (rigid or detector) from a training run ID.
    """
    run_dir = EXP_DIR / run_id
    cfg = (run_dir / 'config.yaml').read_text()
    cfg = yaml.safe_load(cfg)

    if model_type == 'rigid':
        model = create_model_pose(cfg, *args)
    elif model_type == 'detector':
        model = create_model_detector(cfg, len(cfg['label_to_category_id']))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ckpt = torch.load(run_dir / 'checkpoint.pth.tar')
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    return model

def load_pose_predictor(coarse_run_id, refiner_run_id, n_workers=1, preload_cache=False):
    """
    Load a full pose predictor (coarse + refiner) using CosyPose.
    """
    coarse_cfg = yaml.safe_load((EXP_DIR / coarse_run_id / 'config.yaml').read_text())

    renderer = BulletBatchRenderer(coarse_cfg['urdf_ds_name'], preload_cache=preload_cache, n_workers=n_workers)
    object_ds = make_object_dataset(coarse_cfg['object_ds_name'])
    mesh_db = cosypose.lib3d.rigid_mesh_database.MeshDataBase.from_object_ds(object_ds).batched().cuda()

    coarse_model = load_torch_model(coarse_run_id, renderer, mesh_db, model_type='rigid')
    refiner_model = load_torch_model(refiner_run_id, renderer, mesh_db, model_type='rigid')

    return CoarseRefinePosePredictor(coarse_model, refiner_model)

def load_detector(run_id):
    """
    Load a trained CosyPose detector model.
    """
    model = load_torch_model(run_id, model_type='detector')
    return Detector(model)

class RigidObjectPredictor:
    """
    Wrapper that combines the detector and pose predictor.
    """
    def __init__(self, detector_run_id, coarse_run_id, refiner_run_id):
        self.detector = load_detector(detector_run_id)
        self.pose_predictor = load_pose_predictor(coarse_run_id, refiner_run_id, preload_cache=True, n_workers=4)

    def __call__(self, images, cameras, pose_estimation_prior=None, detector_kwargs=None):
        if detector_kwargs is None:
            detector_kwargs = dict()

        # Convert images to torch tensor format
        images_tensor = torch.as_tensor(np.stack(images)).permute(0, 3, 1, 2).float().cuda() / 255.
        K = cameras.K.float().cuda()

        if pose_estimation_prior is None:
            # Run detector if no prior poses are given
            detections = self.detector(images_tensor, **detector_kwargs)
            if len(detections) > 0:
                poses, _ = self.pose_predictor.get_predictions(
                    images=images_tensor,
                    K=K,
                    n_coarse_iterations=1,
                    n_refiner_iterations=4,
                    detections=detections
                )
            else:
                poses = RigidObjectPredictor.empty_predictions()
        else:
            # Refine based on prior predictions
            poses, _ = self.pose_predictor.get_predictions(
                images=images_tensor,
                K=K,
                data_TCO_init=pose_estimation_prior,
                n_coarse_iterations=0,
                n_refiner_iterations=4
            )

        return poses

    @staticmethod
    def empty_predictions():
        return PandasTensorCollection(infos=pd.DataFrame(dict(label=[],)),
                                      poses=torch.empty((0, 4, 4)).float().cuda())
