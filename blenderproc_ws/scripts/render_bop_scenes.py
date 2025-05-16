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
    bproc.renderer.set_render_device('GPU')  # Comment this out if not using GPU 
    

    # Load the object
    obj = bproc.loader.load_obj(obj_path)[0]
    obj.set_name(args.object_name)
    obj.set_scale([0.001, 0.001, 0.001])
    obj.set_cp("category_id", obj_id)

    materials = obj.get_materials()
    if len(materials) > 0:
        mat = materials[0]
        mat.set_principled_shader_value("Roughness", np.random.uniform(0.0, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0.2, 0.8))


    """
    # Create floating black room
    room_planes = [
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.5708, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.5708, 0, 0+]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.5708, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.5708, 0])
    ]
    for plane in room_planes:
        plane.enable_rigidbody(False)
        plane.set_cp("category_id", 0)
        black = bproc.material.create("Black")
        black.set_principled_shader_value("Base Color", [0, 0, 0, 1])
        plane.replace_materials(black)
    
    """
    # Replaced walls with black background
    bpy.data.worlds['World'].use_nodes = False
    bpy.data.worlds['World'].color = (0.0, 0.0, 0.0)

    # Lighting
    
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
    obj.set_location([0.0, 0.0, 0.6])
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

    bbox = obj.get_bound_box()
    bbox_min = np.min(bbox, axis=0)
    bbox_max = np.max(bbox, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = np.linalg.norm(bbox_max - bbox_min)
    radius = bbox_size * 1.5

    # Generate camera poses in a hemisphere arc: top to bottom
    total_views = args.views_per_scene
    for i in range(total_views):
        theta = np.pi * i / (total_views - 1)
        phi = 2 * np.pi * i / total_views

        # Camera position
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        cam_location = bbox_center + np.array([x, y, z])
        forward_vec = bbox_center - cam_location
        rot_matrix = bproc.camera.rotation_from_forward_vec(forward_vec)
        cam2world = bproc.math.build_transformation_mat(cam_location, rot_matrix)
        bproc.camera.add_camera_pose(cam2world)

        # Add a headlight-style point light just behind the camera
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_energy(8)  # Front lighting strength, too high and we get overexposure.
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





  
"""
Older versions are down here, but render differntly 

import blenderproc as bproc
import numpy as np
import os
import argparse
import json
import mathutils




# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("model_dir", help="Path to folder with .ply models")
parser.add_argument("output_dir", help="Output path for BOP dataset")
args = parser.parse_args()

bproc.init()

# Load .ply models
ply_files = sorted([f for f in os.listdir(args.model_dir) if f.endswith(".ply")])

for ply_file in ply_files:
    obj_path = os.path.join(args.model_dir, ply_file)
    obj = bproc.loader.load_obj(obj_path)[0]
    obj.set_name(os.path.splitext(ply_file)[0])  
    obj.set_scale([0.001, 0.001, 0.001])  # mm to m, stabdard for BOP

    # Load category ID
    models_info_path = os.path.join(os.path.dirname(args.model_dir), "models_info.json")
    with open(models_info_path, 'r') as f:
        models_info = json.load(f)
    obj_id = int(obj.get_name().split("_")[-1])
    obj.set_cp("category_id", obj_id)

    # Adjust material
    materials = obj.get_materials()
    if len(materials) > 0:
        mat = materials[0]
        try:
            mat.set_principled_shader_value("Roughness", np.random.uniform(0.0, 1.0))
            mat.set_principled_shader_value("Specular", np.random.uniform(0.2, 0.8))
        except KeyError:
            print(f"Material on {obj.get_name()} is not a Principled BSDF or missing inputs. Skipping shader tweak.")

    # Create room (floor + 4 walls)
    room_planes = [
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.5708, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.5708, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.5708, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.5708, 0])
    ]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', friction=100.0, linear_damping=0.99, angular_damping=0.99)
        plane.set_cp("category_id", 0)

    # Add lighting
    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 5])
    light_plane_material = bproc.material.create('EmissiveLight')
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3, 6),
        emission_color=np.random.uniform([0.7,0.7,0.7,1.0], [1.0,1.0,1.0,1.0])
    )
    light_plane.replace_materials(light_plane_material)

    point_light = bproc.types.Light()
    point_light.set_energy(np.random.uniform(100, 300))
    point_light.set_location(bproc.sampler.shell(center=[0,0,0], radius_min=1, radius_max=2,
                                                  elevation_min=10, elevation_max=85))

    # Set camera intrinsics
    bproc.camera.set_resolution(640, 480)
    bproc.camera.set_intrinsics_from_K_matrix(
        K=np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]]),
        image_width=640,
        image_height=480
    )

    # Pose sampling for this single object
    obj.enable_rigidbody(True, collision_shape="CONVEX_HULL", mass=1.0)
    def sample_pose(obj):
        obj.set_location(np.random.uniform([-0.3, -0.3, 0.5], [0.3, 0.3, 1.0]))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())
    bproc.object.sample_poses([obj], sample_pose_func=sample_pose, max_tries=100)
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1, max_simulation_time=3)

    # Sample camera views
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects([obj] + room_planes)
    poses = 0
    while poses < 50:
        location = bproc.sampler.shell(center=[0, 0, 0], radius_min=0.4, radius_max=0.9,
                                       elevation_min=15, elevation_max=85)
        poi = bproc.object.compute_poi([obj])
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location,
                                                                 inplane_rot=np.random.uniform(-0.7854, 0.7854))
        cam2world = bproc.math.build_transformation_mat(location, rotation_matrix)
        if bproc.camera.perform_obstacle_in_view_check(cam2world, {"min": 0.3}, bop_bvh_tree):
            bproc.camera.add_camera_pose(cam2world)
            poses += 1

    # Render
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    data = bproc.renderer.render()

    # Export
    obj_output_dir = os.path.join(args.output_dir, obj.get_name())
    bproc.writer.write_bop(
        output_dir=obj_output_dir,
        dataset="my_custom_bop",
        depths=data["depth"],
        colors=data["colors"],
        target_objects=[obj],
        color_file_format="JPEG",
        append_to_existing_output=False,
        calc_mask_info_coco=False
    )








#============================================================
import blenderproc as bproc
import numpy as np
import os
import argparse
import json
import mathutils
import bpy

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("model_dir", help="Path to folder with .ply models")
parser.add_argument("output_dir", help="Output path for BOP dataset")
parser.add_argument("--scenes_per_object", type=int, default=1, help="How many scenes to generate per object")
args = parser.parse_args()

bproc.init()

# Enable depth output just once
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# Load .ply models
ply_files = sorted([f for f in os.listdir(args.model_dir) if f.endswith(".ply")])

# Load category IDs from models_info.json
models_info_path = os.path.join(os.path.dirname(args.model_dir), "models_info.json")
with open(models_info_path, 'r') as f:
    models_info = json.load(f)

scene_counter = 0
for ply_file in ply_files:
    obj_path = os.path.join(args.model_dir, ply_file)
    obj = bproc.loader.load_obj(obj_path)[0]
    obj.set_name(os.path.splitext(ply_file)[0])  # e.g., obj_000001
    obj.set_scale([0.001, 0.001, 0.001])  # mm to m

    obj_id = int(obj.get_name().split("_")[-1])
    obj.set_cp("category_id", obj_id)

    materials = obj.get_materials()
    if len(materials) > 0:
        mat = materials[0]
        try:
            mat.set_principled_shader_value("Roughness", np.random.uniform(0.0, 1.0))
            mat.set_principled_shader_value("Specular", np.random.uniform(0.2, 0.8))
        except KeyError:
            print(f"Material on {obj.get_name()} is not a Principled BSDF or missing inputs. Skipping shader tweak.")

    for i in range(args.scenes_per_object):
        for obj in bpy.context.scene.objects:
            obj.select_set(True)
        bpy.ops.object.delete()

        obj = bproc.loader.load_obj(obj_path)[0]
        obj.set_name(os.path.splitext(ply_file)[0])
        obj.set_scale([0.001, 0.001, 0.001])
        obj.set_cp("category_id", obj_id)

        materials = obj.get_materials()
        if len(materials) > 0:
            mat = materials[0]
            try:
                mat.set_principled_shader_value("Roughness", np.random.uniform(0.0, 1.0))
                mat.set_principled_shader_value("Specular", np.random.uniform(0.2, 0.8))
            except KeyError:
                print(f"Material on {obj.get_name()} is not a Principled BSDF or missing inputs. Skipping shader tweak.")

        # Create room (floor + 4 walls)
        room_planes = [
            bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
            bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.5708, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.5708, 0, 0]),
            bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.5708, 0]),
            bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.5708, 0])
        ]
        for plane in room_planes:
            plane.enable_rigidbody(False, collision_shape='BOX', friction=100.0, linear_damping=0.99, angular_damping=0.99)
            plane.set_cp("category_id", 0)
            mat = bproc.material.create("Black")
            mat.set_principled_shader_value("Base Color", [0.0, 0.0, 0.0, 1.0])
            plane.replace_materials(mat)

        # Add lighting
        light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 5])
        light_plane_material = bproc.material.create('EmissiveLight')
        light_plane_material.make_emissive(
            emission_strength=np.random.uniform(3, 6),
            emission_color=np.random.uniform([0.7,0.7,0.7,1.0], [1.0,1.0,1.0,1.0])
        )
        light_plane.replace_materials(light_plane_material)

        point_light = bproc.types.Light()
        point_light.set_energy(np.random.uniform(100, 300))
        point_light.set_location(bproc.sampler.shell(center=[0,0,0], radius_min=1, radius_max=2,
                                                      elevation_min=10, elevation_max=85))

        # Set camera intrinsics
        bproc.camera.set_resolution(640, 480)
        bproc.camera.set_intrinsics_from_K_matrix(
            K=np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]]),
            image_width=640,
            image_height=480
        )

        # Object placement
        obj.enable_rigidbody(True, collision_shape="CONVEX_HULL", mass=1.0)
        def sample_pose(obj):
            obj.set_location(np.random.uniform([-0.3, -0.3, 0.5], [0.3, 0.3, 1.0]))
            obj.set_rotation_euler(bproc.sampler.uniformSO3())
        bproc.object.sample_poses([obj], sample_pose_func=sample_pose, max_tries=100)
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1, max_simulation_time=3)

        # Camera views
        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects([obj] + room_planes)
        poses = 0
        while poses < 50:
            location = bproc.sampler.shell(center=[0, 0, 0], radius_min=0.4, radius_max=0.9,
                                           elevation_min=15, elevation_max=85)
            poi = bproc.object.compute_poi([obj])
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location,
                                                                     inplane_rot=np.random.uniform(-0.7854, 0.7854))
            cam2world = bproc.math.build_transformation_mat(location, rotation_matrix)
            if bproc.camera.perform_obstacle_in_view_check(cam2world, {"min": 0.3}, bop_bvh_tree):
                bproc.camera.add_camera_pose(cam2world)
                poses += 1

        # Render
        data = bproc.renderer.render()

        # Export
        scene_dir = os.path.join(args.output_dir, 'train_pbr', f"{scene_counter:06d}")
        os.makedirs(scene_dir, exist_ok=True)
        bproc.writer.write_bop(
            output_dir=scene_dir,
            dataset="my_custom_bop",
            depths=data["depth"],
            colors=data["colors"],
            target_objects=[obj],
            color_file_format="JPEG",
            append_to_existing_output=False,
            calc_mask_info_coco=False
        )
        scene_counter += 1






import blenderproc as bproc
import numpy as np
import os
import argparse
import json
import mathutils

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("model_dir", help="Path to folder with .ply models")
parser.add_argument("output_dir", help="Output path for BOP dataset")
args = parser.parse_args()

bproc.init()

# Load .ply models
ply_files = sorted([f for f in os.listdir(args.model_dir) if f.endswith(".ply")])
objects = []
for ply_file in ply_files:
    obj_path = os.path.join(args.model_dir, ply_file)
    obj = bproc.loader.load_obj(obj_path)[0]
    obj.set_name(os.path.splitext(ply_file)[0])  # e.g., obj_000001
    obj.set_scale([0.001, 0.001, 0.001])  # mm to m
    objects.append(obj)

# Load category IDs from models_info.json
models_info_path = os.path.join(os.path.dirname(args.model_dir), "models_info.json")
with open(models_info_path, 'r') as f:
    models_info = json.load(f)

for obj in objects:
    obj_id = int(obj.get_name().split("_")[-1])
    obj.set_cp("category_id", obj_id)
    materials = obj.get_materials()
    if len(materials) > 0:
        mat = materials[0]
        try:
            mat.set_principled_shader_value("Roughness", np.random.uniform(0.0, 1.0))
            mat.set_principled_shader_value("Specular", np.random.uniform(0.2, 0.8))
        except KeyError:
            print(f"Material on {obj.get_name()} is not a Principled BSDF or missing inputs. Skipping shader tweak.")

# Create room (floor + 4 walls)
room_planes = [
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.5708, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.5708, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.5708, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.5708, 0])
]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', friction=100.0, linear_damping=0.99, angular_damping=0.99)
    plane.set_cp("category_id", 0)

# Add light (top plane emitter + point)
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 5])
light_plane_material = bproc.material.create('EmissiveLight')
light_plane_material.make_emissive(
    emission_strength=np.random.uniform(3, 6),
    emission_color=np.random.uniform([0.7,0.7,0.7,1.0], [1.0,1.0,1.0,1.0])
)
light_plane.replace_materials(light_plane_material)

point_light = bproc.types.Light()
point_light.set_energy(np.random.uniform(100, 300))
point_light.set_location(bproc.sampler.shell(center=[0,0,0], radius_min=1, radius_max=2,
                                              elevation_min=10, elevation_max=85))

# Set camera intrinsics
bproc.camera.set_resolution(640, 480)
bproc.camera.set_intrinsics_from_K_matrix(
    K=np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]]),
    image_width=640,
    image_height=480
)

# Object pose sampling and physics
def sample_pose(obj):
    obj.set_location(np.random.uniform([-0.3, -0.3, 0.5], [0.3, 0.3, 1.0]))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

for obj in objects:
    obj.enable_rigidbody(True, collision_shape="CONVEX_HULL", mass=1.0)

bproc.object.sample_poses(objects, sample_pose_func=sample_pose, max_tries=100)
bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1, max_simulation_time=3)

# Camera pose sampling
bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(objects + room_planes)
poses = 0
while poses < 50:
    location = bproc.sampler.shell(center=[0, 0, 0], radius_min=1.0, radius_max=2.0,
                                   elevation_min=15, elevation_max=85)
    poi = bproc.object.compute_poi(np.random.choice(objects, size=len(objects)))
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location,
                                                             inplane_rot=np.random.uniform(-0.7854, 0.7854))
    cam2world = bproc.math.build_transformation_mat(location, rotation_matrix)
    if bproc.camera.perform_obstacle_in_view_check(cam2world, {"min": 0.3}, bop_bvh_tree):
        bproc.camera.add_camera_pose(cam2world)
        poses += 1

# Enable rendering (depth only; masks generated automatically)
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# Render RGB and depth
data = bproc.renderer.render()

bproc.writer.write_bop(
    output_dir=args.output_dir,
    dataset="my_custom_bop",
    depths=data["depth"],
    colors=data["colors"],
    target_objects=objects,
    color_file_format="JPEG",
    append_to_existing_output=False,
)

# The final thing that could be added which I haven't done is adding occlusions and clutter.

"""
