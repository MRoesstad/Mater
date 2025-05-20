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
parser.add_argument("--scenes_per_object", type=int, default=1, help="How many scenes to generate for the object")
parser.add_argument("--views_per_scene", type=int, default=50, help="How many views per scene")
parser.add_argument("--start_view_id", type=int, default=0, help="Global view start index for chunked rendering")
parser.add_argument("--total_views", type=int, required=True, help="Total number of views across all chunks")
args = parser.parse_args()


# Load category IDs from models_info.json
models_info_path = os.path.join(os.path.dirname(args.model_dir), "models_info.json")
with open(models_info_path, 'r') as f:
    models_info = json.load(f)

# Load only the selected object path
ply_file = args.object_name + ".ply"
obj_path = os.path.join(args.model_dir, ply_file)
obj_id = int(args.object_name.split("_")[-1])

for i in range(args.scenes_per_object):
    bproc.init()
    #bproc.renderer.set_render_device(['GPU'])  # Comment this out if not using GPU 
    

    # Load the object
    obj = bproc.loader.load_obj(obj_path)[0]
    obj.set_name(args.object_name)
    obj.set_scale([0.001, 0.001, 0.001])
    obj.set_cp("category_id", obj_id)

    materials = obj.get_materials()
    if len(materials) > 0:
        mat = materials[0]
        try:
            if mat.has_principled_shader():
                if "Roughness" in mat.get_principled_shader_inputs():
                    mat.set_principled_shader_value("Roughness", np.random.uniform(0.0, 1.0))
                if "Specular" in mat.get_principled_shader_inputs():
                    mat.set_principled_shader_value("Specular", np.random.uniform(0.2, 0.8))
        except Exception as e:
            print(f"[Warning] Failed to tweak shader for {obj.get_name()}: {e}")

    
    # Replaced walls with black background
    bpy.data.worlds['World'].use_nodes = False
    bpy.data.worlds['World'].color = (0.0, 0.0, 0.0)


    
    # Adding some Ambient light to create shadow
    ambient_light = bproc.types.Light()
    ambient_light.set_type("POINT")
    ambient_light.set_location([5, 5, 5])
    ambient_light.set_energy(4)


    # Set camera intrinsics
    bproc.camera.set_resolution(640, 480)
    bproc.camera.set_intrinsics_from_K_matrix(
        K=np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]]),
        image_width=640,
        image_height=480
    )

    # Place object in mid air
    obj.set_location([0.0, 0.0, 0.0])
   #obj.set_rotation_euler(bproc.sampler.uniformSO3())

    bbox = obj.get_bound_box()
    bbox_min = np.min(bbox, axis=0)
    bbox_max = np.max(bbox, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = np.linalg.norm(bbox_max - bbox_min)
    radius = bbox_size * 1.5

    # Generate camera poses in a hemisphere arc: top to bottom
    global_start = args.start_view_id
    global_end = args.start_view_id + args.views_per_scene

    for global_i in range(global_start, global_end):
        theta = np.pi * global_i / (args.total_views - 1)           # elevation: 0 to pi
        phi = 2 * np.pi * global_i / args.total_views               # azimuth: 0 to 2pi

        # Convert to cartesian coordinates
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        cam_location = bbox_center + np.array([x, y, z])
        forward_vec = bbox_center - cam_location
        rot_matrix = bproc.camera.rotation_from_forward_vec(forward_vec)
        cam2world = bproc.math.build_transformation_mat(cam_location, rot_matrix)
        bproc.camera.add_camera_pose(cam2world)

        # Add a light near the camera
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_energy(8)
        light_offset = 0.55
        light_position = cam_location - light_offset * forward_vec / np.linalg.norm(forward_vec)
        light.set_location(light_position)



    # Enable rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    data = bproc.renderer.render()

    # Want to reduce the max depth to solve issues with depth maps.
    max_depth = 1.5  # This is something to tweek on, if you are using large objects.
    clipped_depths = [np.clip(d, 0, max_depth) for d in data["depth"]]


    train_pbr_root = os.path.join(args.output_dir, 'train_pbr')
    os.makedirs(train_pbr_root, exist_ok=True)

    existing_scenes = glob.glob(os.path.join(train_pbr_root, '[0-9][0-9][0-9][0-9][0-9][0-9]'))
    existing_ids = sorted([int(os.path.basename(s)) for s in existing_scenes])
    next_scene_id = max(existing_ids) + 1 if existing_ids else 0

    
    scene_dir = os.path.join(train_pbr_root, f"{next_scene_id:06d}")


    bproc.writer.write_bop(
        output_dir=scene_dir,
        dataset="", 
        #depths=data["depth"],
        depths=clipped_depths,
        colors=data["colors"],
        target_objects=[obj],
        color_file_format="JPEG",
        append_to_existing_output=False,
        calc_mask_info_coco=False
    )
    

    # Fix for folder structure 
    inner_scene_dir = os.path.join(scene_dir, 'train_pbr', '000000')
    for item in os.listdir(inner_scene_dir):
        s = os.path.join(inner_scene_dir, item)
        d = os.path.join(scene_dir, item)
        shutil.move(s, d)

    
    shutil.rmtree(os.path.join(scene_dir, 'train_pbr'))
 
    camera_src = os.path.join(scene_dir, 'camera.json')
    camera_dst = os.path.join(train_pbr_root, 'camera.json')

    if os.path.exists(camera_src) and not os.path.exists(camera_dst):
        shutil.move(camera_src, camera_dst)

    elif os.path.exists(camera_src):
        os.remove(camera_src)




