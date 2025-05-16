import os
import json
import trimesh

models_dir = r"C:\Users\magnu\OneDrive\Documents\Skole\Master\blenderproc_ws\my_custom_bop\models"
output_path = os.path.join(os.path.dirname(models_dir), "models_info.json")

model_info = {}
for filename in os.listdir(models_dir):
    if filename.endswith(".ply") and filename.startswith("obj_"):
        try:
            obj_id = int(filename.split("_")[-1].split(".")[0])
        except ValueError:
            print(f"⚠️ Skipping {filename}: invalid object ID")
            continue

        mesh_path = os.path.join(models_dir, filename)
        mesh = trimesh.load_mesh(mesh_path)

        bbox = mesh.bounding_box.bounds
        size = mesh.extents

        # Try oriented bounding box for diameter; fallback to bounding box diagonal
        try:
            diameter = mesh.bounding_box_oriented.primitive.extents.max()
        except:
            print(f"⚠️ Fallback to AABB for diameter of object {obj_id}")
            diameter = float((size**2).sum()**0.5)  # Euclidean norm of extents

        model_info[obj_id] = {
            "diameter": float(diameter),
            "min_x": float(bbox[0][0]),
            "min_y": float(bbox[0][1]),
            "min_z": float(bbox[0][2]),
            "size_x": float(size[0]),
            "size_y": float(size[1]),
            "size_z": float(size[2]),
            "symmetries_continuous": [],
            "symmetries_discrete": []
        }

with open(output_path, 'w') as f:
    json.dump(model_info, f, indent=4)

print(f"✅ models_info.json created at: {output_path}")
