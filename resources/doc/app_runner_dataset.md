# Running with Dataset

> **Note:** Before running DualMap with a dataset, please ensure the dataset is arranged in the correct directory structure.

## 📚 Table of Contents
- [Preparing Datasets](#preparing-datasets)
  - [Replica & ScanNet](#replica--scannet)
  - [HM3D Self-collected Data](#hm3d-self-collected-data)
  - [Dataset Structure](#dataset-structure)
- [Run](#run)
- [Key Configurations](#key-configurations)
- [Evaluation](#evaluation)
  - [Evaluate Single Scene](#evaluate-single-scene)
  - [Evaluate the Whole Dataset](#evaluate-the-whole-dataset)
- [Offline Query](#offline-query)
- [Troubleshooting](#troubleshooting)

## Preparing Datasets

### Replica & ScanNet

Please follow [this guide](./data_replica_scannet.md) to download and arrange the public Replica and ScanNet datasets for use with DualMap.


### HM3D Self-collected Data

We manually collected data in three HM3D scenes to support static and dynamic object navigation. Please follow [this guide](./data_hm3d_self_collected.md) to download and arrange the self-collected HM3D data. 

### Dataset Structure
We recommend placing the data in the `dataset` folder within this repository.
The final `dataset` structure should look like this:
```
dataset/
├── Replica/
│   ├── office0/
│   │   ├── results/              # RGB-D frames (depth + RGB)
│   │   └── traj.txt              # Trajectory file
│   ├── office1/
│   ├── ...
│   └── room2/
│
├── Replica-Dataset/
│   └── Replica_original/
│       ├── apartment_0/
│       ├── room_0/
│       │   └── habitat/
│       │       └── mesh_semantic.ply
│       └── ...
│
├── scannet/
│   └── exported/                  # exported ScanNet data
│       ├── scene0010_00/
│       │   ├── color/             # Exported color images
│       │   ├── depth/             # Exported depth maps
│       │   ├── intrinsic/         # Camera intrinsics
│       │   └── pose/              # Camera poses
│       ├── scene0050_00/
│       └── ...
│
├── scannet200/
│   ├── train/
│   └── val/
│       ├── scene0011_00.ply
│       └── ...
│
└── HM3D_collect/
```

## Run
First activate the conda environment
```
conda activate dualmap
```

Then, navigate to the repository root and run the application:
```
cd DualMap
python -m applications.runner_dataset
```

You will see the **Rerun** visualization window pop up, and the system will run with the Replica `room_0` sequence by default.
After the run finishes, you can check the `output` directory for the results, which are organized as follows:

```
output/                                 # Root output directory 
└── map_results/                        # Output folder assigned in base_config.yaml
    ├── log/                            # Running logs
    │   └── log_20250813_162449.log
    └── replica_room_0/                 # Processed dataset_scene_id
        ├── detections/                 # Saved detection results if save_detection is enabled
        ├── map/                        # Maps saved by DualMap
        │   ├── 0e5ea11d.pkl            # Object information
        │   ├── ...
        │   ├── 2cd13570.pkl
        │   ├── layout.pcd              # Layout point cloud
        ├── detector_time.csv           # Time breakdown for observation generation
        └── system_time.csv             # Time breakdown for overall system run
```

The terminal screenshot of finishing the running will be like this:
<p align="center">
    <img src="../image/app_dataset/runner_dataset_ok.png" width="80%">
</p>

Please checkout the `.log` file for all the running details. The **time decomposition** results are also included in the log file.

## Key Configurations

📁 `config/base_config.yaml`

```yaml
# Dataset name; determines which dataset loader is used
dataset_name: your_dataset_name

# ID of the scene to run
scene_id: scene_01

# Path to the dataset
dataset_path: /path/to/your/dataset

# Dataset config file (use as a template for customization)
dataset_conf_path: ./config/data_config/your_dataset_name.yaml

# Ground truth path used for evaluation
dataset_gt_path: /path/to/ground_truth

# Output directory for DualMap results
output_path: ./output/map_results
```
> **Note:** For `output_path`, each run will clear all existing files in output directory. If you plan to run the system multiple times, make sure to change this path accordingly.

📁 `config/system_config.yaml`
```yaml
# Path to the class list file used by YOLO
# For indoor datasets, use the following:
yolo:
  given_classes_path: ./config/class_list/gpt_indoor_general.txt
```
📁 `config/runner_dataset.yaml`

```yaml
# Enable detection visualization via Rerun (will slow down performance)
visualize_detection: true

# Use FastSAM for open-vocabulary segmentation
# Segments will be labeled as 'unknown'
use_fastsam: true

# Enable Rerun visualization (recommended unless benchmarking)
use_rerun: true

# Run only local mapping without abstraction (for segmentation evaluation)
run_local_mapping_only: true

# Save local map outputs for offline evaluation and query
save_local_map: true
```


## Evaluation

Running the evaluation to reproduce the results in Table II in the paper!

### Evaluate Single Scene
After running `runner_dataset` with the default settings, you will get the output concrete map in Replica `room_0`.

Next, without changing any configuration, run the following command to generate class colors for Replica dataset evaluation:
```
python -m applications.generate_replica_class_color
```
> **Note**: For ScanNet, you do not need to run this class color generation script.

Finally, run the evaluation script. It will evaluate the saved concrete map based on the settings in `base_config.yaml`:
```
python -m evaluation.sem_seg_eval
```
After successfully running the evaluation, you will see the output metrics. For `room_0`, **FmIoU** is around `0.75`, **mAcc** is around `0.54`, and **mIoU** is around `0.37`.

You can also find the results in `{output_path}/eval/results.json`.

> For ScanNet, update the settings in `base_config.yaml`. After successfully running the mapping, follow the same evaluation process as above.

### Evaluate the Whole Dataset
We provide scripts to run and evaluate the entire dataset for both Replica and ScanNet.

1. Set the correct dataset path and configuration file in `base_config.yaml`.  
2. Configure the parameters you want to test in `runner_dataset.yaml` and `system_config.yaml`.  
3. Run the corresponding script:

   - **Replica**:  
     ```bash
     bash scripts/dualmap_replica.sh
     ```

   - **ScanNet**:  
     ```bash
     bash scripts/dualmap_scannet.sh
     ```

After finishing the script run, all evaluation results from all scenes will be saved in `output_path`.

To get aggregated results, run:
```
python scripts/save_as_xlsx.py --dataset ${dataset_name} --eval_path ${output_path}/eval
```
Make sure to replace `${dataset_name}` and `${output_path}` with the actual parameter values you used. 

## Offline Query

After running and saving the concrete map, you can simply run the following command to perform an offline query.
```
python -m applications.offline_local_map_query
```

The map loaded during the query is determined by the settings in `base_config.yaml`.  
For more details, please refer to [this guide](./app_offline_query.md).


## Troubleshooting

#### Runner -> ERROR - [Detector][Init] Error loading CLIP model

> ERROR - [Detector][Init] Error loading CLIP model: Failed to download file (open_clip_pytorch_model.bin) for apple/MobileCLIP-S2-OpenCLIP. Last error: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443)

This error may also lead to `AttributeError: 'Detector' object has no attribute 'fastsam/yolo'`. The root cause is that the current machine cannot connect to `huggingface.co`. Since our CLIP implementation uses the OpenCLIP library and relies on weights downloaded from Hugging Face, the absence of a connection will trigger this error. 

To resolve it, enable a proxy so that the machine can connect to `huggingface.co` and download the required weights.

Further, you can also try to load local CLIP weights, please refer to [this link](https://github.com/mlfoundations/open_clip?tab=readme-ov-file#loading-models) for more information.


#### Evaluation-> Error: File not found: replica_room_0_id_names.json

> FileNotFoundError: Error: File not found: ./output/map_results/replica_room_0/classes_info/replica_room_0_id_names.json

This error occurs when running the Replica dataset evaluation without first generating the class color file.  
Make sure to run the following command **before** executing `python -m evaluation.sem_seg_eval`:
```
python -m applications.generate_replica_class_color
```

