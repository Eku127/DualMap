# Replica and ScanNet Dataset

## Replica
For the Replica dataset, we require two components:
1. **Scanned RGB-D trajectories** from [Nice-SLAM](https://github.com/cvg/nice-slam), used for mapping and fair comparison in our experiments.
2. Original Replica dataset, used for evaluation. If you've already downloaded this following the [Habitat Data Collector](https://github.com/Eku127/habitat-data-collector/blob/main/documents/dataset/dataset.md#replica-dataset), you can skip this step.

### Scanned RGB-D data
Follow the script https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh in Nice slam and download your replica dataset in your directory.

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
To download the original Replica dataset, follow the official instructions provided in the [Replica-Dataset repository](https://github.com/facebookresearch/Replica-Dataset).

The expected structure of the dataset will be like this:
```
<PATH TO Original Replica Dataset>/
├── apartment_0/
├── room_0/
│   └── habitat/
│       ├── mesh_semantic.ply
│       ├── info_semantic.json
│       └── replica_stage.stage_config.json
├── replica.scene_dataset_config.json
```

## ScanNet
