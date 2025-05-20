import os
import json
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("train_pbr_dir", help="Path to train_pbr folder")
parser.add_argument("--chunks", nargs="+", required=True, help="Scene folder names to merge (e.g. 000000 000001 000002)")
parser.add_argument("--offsets", nargs="+", type=int, required=True, help="Start index offsets for each chunk")
parser.add_argument("--output_scene", default="000001", help="Name of final merged folder")
args = parser.parse_args()

output_dir = os.path.join(args.train_pbr_dir, args.output_scene)
rgb_output = os.path.join(output_dir, "rgb")
depth_output = os.path.join(output_dir, "depth")
os.makedirs(rgb_output, exist_ok=True)
os.makedirs(depth_output, exist_ok=True)

json_files = ["scene_gt.json", "scene_gt_info.json", "scene_camera.json"]
merged_json = {name: {} for name in json_files}

for chunk, offset in zip(args.chunks, args.offsets):
    chunk_path = os.path.join(args.train_pbr_dir, chunk)
    print(f"🧩 Merging from {chunk} with offset {offset}")

    # JSONs
    for fname in json_files:
        fpath = os.path.join(chunk_path, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                data = json.load(f)
            for k, v in data.items():
                new_k = str(int(k) + offset)
                merged_json[fname][new_k] = v

    # RGB images
    rgb_dir = os.path.join(chunk_path, "rgb")
    if os.path.exists(rgb_dir):
        for fname in sorted(os.listdir(rgb_dir)):
            if fname.endswith(".jpg") and fname[:6].isdigit():
                old_idx = int(fname[:6])
                new_idx = old_idx + offset
                new_name = f"{new_idx:06d}.jpg"
                shutil.copy2(os.path.join(rgb_dir, fname), os.path.join(rgb_output, new_name))

    # Depth images
    depth_dir = os.path.join(chunk_path, "depth")
    if os.path.exists(depth_dir):
        for fname in sorted(os.listdir(depth_dir)):
            if fname.endswith(".png") and fname[:6].isdigit():
                old_idx = int(fname[:6])
                new_idx = old_idx + offset
                new_name = f"{new_idx:06d}.png"
                shutil.copy2(os.path.join(depth_dir, fname), os.path.join(depth_output, new_name))

# Write merged JSONs
for fname, content in merged_json.items():
    out_path = os.path.join(output_dir, fname)
    with open(out_path, 'w') as f:
        json.dump(content, f, indent=2)

print(f"✅ Finished merging chunks into: {output_dir}")
