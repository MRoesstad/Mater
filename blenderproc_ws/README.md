# Dataset Creation 
This workspace is created for dataset creation. It utilises BlenderProc and bop_toolkit. All you need to make it work is to supply a ply file. From there, the generate_json.py will create a JSON file.
This script looks for a specific object name. ojb_000001, obj_000002 ... Therefore, remember to change the name of the ply files. In addition to this change, the number of objects you are using in dataset_params.py in the bop_toolkit. The required dataset should have a structure similar to:
```
my_dataset/
├── models/
│   ├── obj_000001.ply
│   ├── obj_000002.ply
│   └── ...
├── models_info.json
├── train_pbr/
│   ├── 000000/
│   │   ├── rgb/
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   ├── scene_gt.json
│   │   ├── scene_gt_info.json
│   │   └── scene_camera.json
│   ├── 000001/
│   └── ...
└── test/
    ├── 000000/
    │   ├── rgb/
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    │   ├── scene_gt.json
    │   ├── scene_gt_info.json
    │   └── scene_camera.json
    ├── 000001/
    └── ...
```


## Environment
The environment is created for a Linux machine and can be found and created using the command:
```
conda env create -f blenderproc_env.yml
```
Then 
```
cd bop_toolkit
pip install -e .
```

## Commands 
Run render_bop_scenes.py from the terminal with the command:
(Just adjust the paths, and which model you want to train)

```
blenderproc run render_bop_scenes.py  /home/magnus/blenderproc_ws/my_custom_bop/models   /home/magnus/blenderproc_ws/my_custom_bop   --object_name obj_000001   --scenes_per_object 1 --views_per_scene 20

```
This script generates camera.json, scene_gt.json, scene_camera.json, depth and rgb folders. However, to get a complete dataset, do we need to utilise bop_toolkit as well? This workspace presents a modified version of bot_toolkit for enabling custom datasets.
To create test dataset use the command:
```
blenderproc run generate_test.py   /home/magnus/Mater/blenderproc_ws/my_custom_bop/models   /home/magnus/Mater/blenderproc_ws/my_custom_bop   --object_name obj_000001   --total_views 20   --distractor_dir /home/magnus/Downloads/tless_models/models_cad

```

Before utilising bop_toolkit run (NB! adjust it based on your path):
```
touch /home/magnus/blenderproc_ws/my_custom_bop/test_targets_bop19.json
echo "[]" > /home/magnus/blenderproc_ws/my_custom_bop/test_targets_bop19.json
```

First step is creating the calc_gt_masks
```
python scripts/calc_gt_masks.py \
  --dataset my_custom_bop \
  --dataset_split train_pbr \
  --datasets_path /home/magnus/blenderproc_ws \
  --renderer_type vispy \
  --use_all_gt
```
Next run:
```
python scripts/calc_gt_info.py \
  --dataset my_custom_bop \
  --dataset_split train_pbr \
  --datasets_path /home/magnus/blenderproc_ws

```
Then finally:

```
python scripts/calc_gt_coco.py \
  --dataset my_custom_bop \
  --dataset_split train_pbr \
  --datasets_path /home/magnus/blenderproc_ws \
  --use_all_gt

```
