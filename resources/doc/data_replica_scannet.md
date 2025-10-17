# Replica and ScanNet Dataset

## Replica
For the Replica dataset, we require two components:
1. **Scanned RGB-D trajectories** from [Nice-SLAM](https://github.com/cvg/nice-slam), used for mapping and fair comparison in our experiments.
2. Original Replica dataset, used for evaluation. If you've already downloaded this following the [Habitat Data Collector](https://github.com/Eku127/habitat-data-collector/blob/main/documents/dataset/dataset.md#replica-dataset), you can skip this step.

### Scanned RGB-D Data
Follow the script https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh in Nice slam and download your replica dataset in your directory. Or, follow the following commands to download:
```
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
```

The expected structure of the dataset will be like this:

```
<PATH TO Replica>/
├── cam_params.json
├── office0/
│   ├── results/              # RGB-D frames (depth + RGB)
│   └── traj.txt              # Trajectory file
├── office0_mesh.ply
├── office1/
│   ├── results/
│   └── traj.txt
├── office1_mesh.ply
├── ...
├── room2/
│   ├── results/
│   └── traj.txt
└── room2_mesh.ply
```

### Original Replica Data
To download the original Replica dataset, follow the official instructions provided in the [Replica-Dataset repository](https://github.com/facebookresearch/Replica-Dataset). Or, you can follow the simplified steps below to download:

```
git clone https://github.com/facebookresearch/Replica-Dataset.git
chmod +x Replica-Dataset/download.sh
./Replica-Dataset/download.sh Replica_original
```

The expected structure of the dataset will be like this:
```
<PATH TO Replica_original>/
├── apartment_0/
├── room_0/
│   └── habitat/
│       ├── mesh_semantic.ply
│       ├── info_semantic.json
│       └── replica_stage.stage_config.json
├── replica.scene_dataset_config.json
```

## ScanNet

Official release of the ScanNet dataset can be obtained from [ScanNet Github repository](https://github.com/ScanNet/ScanNet) or the [ScanNet website](http://www.scan-net.org/). In our experiments, we used the following sequences: `scene0011_00`, `scene0050_00`, `scene0231_00`, `scene0378_00`, and `scene0518_00`.

### Scanned RGB-D Data

The downloaded raw ScanNet data is packaged as `.sens` files, which can be exported into RGB-D frames with poses using the [SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python). Note that Python 2.7 is required for this process. After running SensReader, your directory structure should look like this:

```
<PATH TO ScanNet>/
└── exported/
│   ├── scene0010_00/
│   |   └── color/          # exported color images
│   │   ├── depth/          # exported depth maps
│   │   ├── intrinsic/      # camera intrinsics
│   │   └── pose/           # camera poses
│   └── scene0050_00/
├── scene0010_00/           # raw .sens files and labels
│   ├── scene0010_00.sens
│   ├── ...                 # other related .ply and .json files
│   └── scene0010_00_vh_clean_2.labels.ply
└── scene0050_00/
    └── ...
```

### Original ScanNet Data

To run evaluation on the ScanNet dataset, you need the ground truth semantic pointclouds. We support both original `ScanNet` (with 20 classes) and `ScanNet200` (expanded semantic label set with 200 classes).

If you are using ScanNet (20 classes), the semantic annotations are stored in files named `<scene_id>_vh_clean_2.labels.ply`, as shown above. In our experiments, we use ScanNet200. To preprocess ScanNet200, please follow this [Github repo](https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/ScanNet200). After preprocessing, your directory structure should look like this:

```
<PATH TO ScanNet200>/
├── train/
└── val/
    ├── scene0011_00.ply
    ├── scene0050_00.ply
    ├── scene0231_00.ply
    ├── scene0378_00.ply
    └── scene0050_00.ply
```
