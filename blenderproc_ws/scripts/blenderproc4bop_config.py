import blenderproc as bproc
import numpy as np
import os, argparse

'''
Inittial atempt. Here i tried to create the entire dataset. But didn't work since it automatically uses bop_toolkit_lib internally through bproc.writer.write_bop(...). Perhaps could be modified to work and make the progress easier.
'''

parser = argparse.ArgumentParser()
parser.add_argument("bop_path", help="Path to custom BOP dataset directory")
parser.add_argument("dataset_name", help="Name of the dataset (e.g., my_custom_bop)")
parser.add_argument("output_dir", help="Output directory for generated data")
args = parser.parse_args()

bproc.init()  # Initialize BlenderProc

# 1. **Load Objects**: Load all models from the custom BOP dataset 
objects = bproc.loader.load_bop_objs(
    bop_dataset_path=os.path.join(args.bop_path, args.dataset_name),
    mm2m=True,
    model_type="eval"  # or "train" depending on folder name (typically use "eval")
)

# Ensure each object uses auto smooth shading for realism
for obj in objects:
    obj.set_shading_mode('auto')
    # Optionally randomize object material properties (color, roughness, specular)
    mat = obj.get_materials()[0]
    mat.set_principled_shader_value("Roughness", np.random.uniform(0.0, 1.0))
    mat.set_principled_shader_value("Specular", np.random.uniform(0.0, 1.0))

# 2. **Domain Randomization Setup**:
# Create a ground plane and surrounding walls (to simulate a room or backdrop)
room_planes = [
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),                         # floor
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], 
                                  rotation=[-1.5708, 0, 0]),                         # back wall
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], 
                                  rotation=[1.5708, 0, 0]),                          # front wall
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], 
                                  rotation=[0, -1.5708, 0]),                         # right wall
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], 
                                  rotation=[0, 1.5708, 0])                           # left wall
]
# Make the planes non-movable (rigid bodies for physics with high friction)
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', 
                           friction=100.0, linear_damping=0.99, angular_damping=0.99)

# Load some random floor/wall texture (optional, requires cc_textures dataset)
cc_materials = bproc.loader.load_ccmaterials("../resources/cctextures")  # adjust path
if cc_materials:
    random_tex = np.random.choice(cc_materials)
    for plane in room_planes:
        plane.replace_materials(random_tex)

# Random Light: Add an emitting plane light above
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 5])
light_plane_material = bproc.material.create('EmissiveLight')
light_plane_material.make_emissive(
    emission_strength=np.random.uniform(3, 6),
    emission_color=np.random.uniform([0.5,0.5,0.5,1.0], [1.0,1.0,1.0,1.0])
)
light_plane.replace_materials(light_plane_material)
# Additional random point light in scene
point_light = bproc.types.Light()
point_light.set_energy(np.random.uniform(100, 300))
point_light.set_color(np.random.uniform([0.5,0.5,0.5], [1,1,1]))
point_light.set_location(bproc.sampler.shell(
    center=[0,0,0], radius_min=1, radius_max=2,
    elevation_min=5, elevation_max=85
))

# 3. **Object Placement and Physics**:
# Randomly position objects in the air within a region, then drop them under gravity
def sample_pose(obj: bproc.types.MeshObject):
    # sample a random location above the ground near the origin
    obj.set_location(np.random.uniform([-0.3, -0.3, 0.5], [0.3, 0.3, 1.0]))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())  # random orientation

# Enable physics on objects
for obj in objects:
    obj.enable_rigidbody(True, collision_shape="CONVEX_HULL", mass=1.0)
# Sample initial poses for objects to avoid initial collisions
bproc.object.sample_poses(objects, sample_pose_func=sample_pose, max_tries=100)
# Run physics simulation to let objects drop onto the floor
bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1, max_simulation_time=3)

# 4. **Camera Sampling**:
bproc.camera.set_intrinsics_from_K_matrix(
    np.array([[600, 0, 320],[0, 600, 240],[0, 0, 1]]), 640, 480
)
# Randomly generate camera poses looking at the scene
poses = 0
bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(objects + room_planes)
while poses < 50:  # e.g., 50 images per scene
    # sample a random camera location on a sphere shell around scene center [0,0,0]
    location = bproc.sampler.shell(center=[0,0,0], radius_min=1.0, radius_max=2.0, 
                                   elevation_min=15, elevation_max=85)
    # Look at a random point on one of the target objects
    poi = bproc.object.compute_poi(np.random.choice(objects, size=len(objects)))
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, 
                                   inplane_rot=np.random.uniform(-0.7854, 0.7854))
    cam2world = bproc.math.build_transformation_mat(location, rotation_matrix)
    # Ensure no object is too close to camera and the view is not empty
    if bproc.camera.perform_obstacle_in_view_check(cam2world, {"min": 0.3}, bop_bvh_tree):
        bproc.camera.add_camera_pose(cam2world)
        poses += 1

# 5. **Rendering and Writing BOP Output**:
# Enable depth and segmentation outputs
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["instance", "class"])
# Render the scene from all added camera poses
data = bproc.renderer.render()
# Write out in BOP format
bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'), 
                       dataset=args.dataset_name, 
                       depths=data["depth"], 
                       colors=data["colors"], 
                       seg_masks=data["seg"], 
                       color_file_format="PNG")