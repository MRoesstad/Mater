import blenderproc as bproc
import numpy as np
import os
import argparse
import json
import mathutils
import bpy
import glob
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", help="Path to folder with .ply models")
parser.add_argument("output_dir", help="Output path for BOP dataset")
parser.add_argument("--object_name", required=True, help="Name of the object to render (e.g., obj_000001)")
parser.add_argument("--views_per_scene", type=int, default=50, help="How many views to render")
parser.add_argument("--start_view_id", type=int, default=0, help="Global view start index for chunked rendering")
parser.add_argument("--total_views", type=int, required=True, help="Total number of views across all chunks")
parser.add_argument("--distractor_dir", required=True, help="Path to distractor .ply models")
args = parser.parse_args()

# Load object
models_info_path = os.path.join(os.path.dirname(args.model_dir), "models_info.json")
with open(models_info_path, 'r') as f:
    models_info = json.load(f)

obj_path = os.path.join(args.model_dir, args.object_name + ".ply")
obj_id = int(args.object_name.split("_")[-1])

bproc.init()
#bproc.renderer.set_render_device(device_type="GPU")
obj = bproc.loader.load_obj(obj_path)[0]
obj.set_name(args.object_name)
obj.set_scale([0.001, 0.001, 0.001])
obj.set_cp("category_id", obj_id)
obj.set_location([0.0, 0.0, 0.0])

# Handle materials
materials = obj.get_materials()
if materials:
    try:
        materials[0].set_principled_shader_value("Roughness", np.random.uniform(0.0, 1.0))
        materials[0].set_principled_shader_value("Specular", np.random.uniform(0.2, 0.8))
    except Exception as e:
        print(f"[Warning] Failed to tweak shader: {e}")

# Add distractors
distractor_plys = sorted(glob.glob(os.path.join(args.distractor_dir, '*.ply')))
sampled_plys = np.random.choice(distractor_plys, size=min(15, len(distractor_plys)), replace=False)
distractors = []
for ply_path in sampled_plys:
    objs = bproc.loader.load_obj(ply_path)
    for d_obj in objs:
        for _ in range(100):
            loc = np.random.uniform([-0.6, -0.6, 0.0], [0.6, 0.6, 0.6])
            if np.linalg.norm(loc) > 0.25 and abs(loc[0]) > 0.1 and abs(loc[1]) > 0.1:
                break
        d_obj.set_location(loc)
        d_obj.set_rotation_euler(bproc.sampler.uniformSO3())
        d_obj.set_scale([0.001, 0.001, 0.001])
        d_obj.set_cp("category_id", 999)
        distractors.append(d_obj)

# Lighting
bpy.data.worlds['World'].use_nodes = False
bpy.data.worlds['World'].color = (0.05, 0.05, 0.05)

light1 = bproc.types.Light()
light1.set_type("POINT")
light1.set_location([2.5, -2.5, 3.0])
light1.set_energy(12.0)
light1.set_color([1.0, 0.98, 0.9])

light2 = bproc.types.Light()
light2.set_type("SUN")
light2.set_rotation_euler([np.pi/4, 0, np.pi/4])
light2.set_energy(4.0)

# Camera setup
bproc.camera.set_resolution(640, 480)
bproc.camera.set_intrinsics_from_K_matrix(
    K=np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]]),
    image_width=640,
    image_height=480
)

bbox = obj.get_bound_box()
bbox_min = np.min(bbox, axis=0)
bbox_max = np.max(bbox, axis=0)
bbox_center = (bbox_min + bbox_max) / 2
bbox_size = np.linalg.norm(bbox_max - bbox_min)
min_radius = bbox_size * 1.3
max_radius = bbox_size * 2.8

# Generate camera poses
for global_i in range(args.start_view_id, args.start_view_id + args.views_per_scene):
    radius = np.random.uniform(min_radius, max_radius)
    theta = np.pi * global_i / (args.total_views - 1)
    phi = 2 * np.pi * global_i / args.total_views
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    cam_location = bbox_center + np.array([x, y, z])
    forward_vec = bbox_center - cam_location
    rot_matrix = bproc.camera.rotation_from_forward_vec(forward_vec)
    cam2world = bproc.math.build_transformation_mat(cam_location, rot_matrix)
    bproc.camera.add_camera_pose(cam2world)

# Render
bproc.renderer.enable_depth_output(activate_antialiasing=False)
data = bproc.renderer.render()
max_depth = 1.5
clipped_depths = [np.clip(d, 0, max_depth) for d in data["depth"]]

# Save output to train_pbr/XXXXX
train_pbr_root = os.path.join(args.output_dir, 'train_pbr')
os.makedirs(train_pbr_root, exist_ok=True)
existing_scenes = glob.glob(os.path.join(train_pbr_root, '[0-9][0-9][0-9][0-9][0-9][0-9]'))
existing_ids = sorted([int(os.path.basename(s)) for s in existing_scenes])
next_scene_id = max(existing_ids) + 1 if existing_ids else 0
scene_dir = os.path.join(train_pbr_root, f"{next_scene_id:06d}")

bproc.writer.write_bop(
    output_dir=scene_dir,
    dataset="",
    depths=clipped_depths,
    colors=data["colors"],
    target_objects=[obj],
    color_file_format="JPEG",
    append_to_existing_output=False,
    calc_mask_info_coco=False
)

# Flatten nested folders
inner_scene_dir = os.path.join(scene_dir, 'train_pbr', '000000')
if os.path.exists(inner_scene_dir):
    for item in os.listdir(inner_scene_dir):
        shutil.move(os.path.join(inner_scene_dir, item), os.path.join(scene_dir, item))
    shutil.rmtree(os.path.join(scene_dir, 'train_pbr'))

# Move camera.json to train_pbr root if missing
camera_src = os.path.join(scene_dir, 'camera.json')
camera_dst = os.path.join(train_pbr_root, 'camera.json')
if os.path.exists(camera_src) and not os.path.exists(camera_dst):
    shutil.move(camera_src, camera_dst)
elif os.path.exists(camera_src):
    os.remove(camera_src)
