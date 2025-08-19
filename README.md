# DualMap
<h3>
  <a href="https://eku127.github.io/DualMap/">Project Page</a> |
  <a href="https://arxiv.org/abs/2506.01950">Arxiv</a> 
  <!-- <a href="https://youtu.be/ZmZDvhyXL_g">Video</a> -->
</h3>

<p align="center">
  <img src="resources/image/teaser.jpg" width="60%">
</p>


**DualMap** is an online open-vocabulary mapping system that enables robots to understand and navigate dynamic 3D environments using natural language.

The system supports multiple input sources, including offline datasets (**Dataset Mode**), ROS streams & rosbag files (**ROS Mode**), and iPhone video streams (**Record3d Mode**). We provide examples for each input type.

## News

**[2025.08]** Release [Online Mapping and Navigation Guide](#online-mapping-and-navigation-in-simulation) and [Run with iPhone Guide](#run-with-iphone).

**[2025.08]** Release [Run with Dataset Guide](#run-with-datasets) and [Run with ROS Guide](#run-with-ros).

**[2025.06]** Release [Offline Query Guide](resources/doc/app_offline_query.md) and examples.

## Installation

> ✅ Tested on **Ubuntu 22.04** with **ROS 2 Humble** and **Python 3.10**

### 1. Clone the Repository (with submodules)

```bash
git clone --branch main --single-branch --recurse-submodules git@github.com:Eku127/DualMap.git
cd DualMap
```
>  Make sure to use `--recurse-submodules` to get `mobileclip`.

### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate dualmap
```

### 3. Install MobileCLIP
```bash
cd 3rdparty/mobileclip
pip install -e . --no-deps
cd ../..
```

### (Optional) Setup ROS 2 Environment
Setting up ROS2 environment for ROS support and applications.
We recommend [ROS 2 Humble](https://docs.ros.org/en/humble/Installation.html).
Once installed, activate the environment:

```bash
source /opt/ros/humble/setup.bash
```

> DualMap’s navigation functionality and real-world integration are based on ROS 2. Installation is strongly recommended.

> **ROS1 noetic** is also supported, you can setup the ROS 1 in Ubuntu 22.04 by follow [this guide](resources/doc/ros_communication.md).

### (Optional) Setup Habitat Data Collector

[Habitat Data Collector](https://github.com/Eku127/habitat-data-collector) is a tool built on top of the [Habitat-sim](https://github.com/facebookresearch/habitat-sim). It supports agent control, object manipulation, dataset and ROS2 bag recording, as well as navigation through external ROS2 topics. DualMap subscribes to live ROS2 topics from the collector for real-time mapping and language-guided querying, and publishes navigation trajectories for the agent to follow.

> For the best DualMap experience (especially interactive mapping and navigation), **we strongly recommend setting up the Habitat Data Collector**. See [the repo](https://github.com/Eku127/habitat-data-collector) for installation and usage details.


## Applications
### Run with Datasets

DualMap supports running with **offline datasets**. Currently supported datasets include:
1. Replica Dataset  
2. ScanNet Dataset  
3. TUM RGB-D Dataset  
4. Self-collected data using [Habitat Data Collector](https://github.com/Eku127/habitat-data-collector)  

👉 Follow [Dataset Runner Guide](resources/doc/app_runner_dataset.md) to arrange datasets, run DualMap with these datasets and reproduce our offline mapping results in **Table II** in our paper.

### Run with ROS

DualMap supports input from both **ROS1** and **ROS2**.  
You can run the system with **offline rosbags** or in **online mode** with real robots.

👉 Follow the [ROS Runner Guide](resources/doc/app_runner_ros.md) to get started with running DualMap using ROS1/ROS2 rosbags or live ROS streams.

### Online Mapping and Navigation in Simulation

DualMap supports **online** interactive mapping and object navigation in simulation via the [Habitat Data Collector](https://github.com/Eku127/habitat-data-collector).

👉 Follow the [Online Mapping and Navigation Guide](resources/doc/app_simulation.md) to get started with running DualMap in interactive simulation scenes and to reproduce the navigation results in **Table III** in our paper.

### Run with iPhone

DualMap supports **real-time data streaming** from the **Record3D** app on iPhone.

👉 Follow the [iPhone Runner Guide](resources/doc/app_runner_record_3d.md) to get started with setting up Record3D, streaming data to DualMap, and mapping with your own iPhone!

### Offline Map Query

We provide two prebuilt map examples for offline querying: one from iPhone data and one from Replica Room 0.

👉 Follow [Offline Query Guide](resources/doc/app_offline_query.md) to run the query application.

### Visualization
<p align="center">
    <img src="resources/image/app_visual.jpg" width="100%">
</p>

The system supports both [Rerun](https://rerun.io) and [Rviz](http://wiki.ros.org/rviz) visualization. When running with ROS, you can switch the visualizaiton via `use_rerun` and `use_rviz` option in `config/runner_ros.yaml`


## Citation

If you find our work helpful, please consider starring this repo 🌟 and cite:

```bibtex
@article{jiang2025dualmap,
  title={DualMap: Online Open-Vocabulary Semantic Mapping for Natural Language Navigation in Dynamic Changing Scenes},
  author={Jiang, Jiajun and Zhu, Yiming and Wu, Zirui and Song, Jie},
  journal={arXiv preprint arXiv:2506.01950},
  year={2025}
}
```

## Contact
For technical questions, please create an issue. For other questions, please contact the first author: jjiang127 [at] connect.hkust-gz.edu.cn

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.

## Acknowledgment

We are grateful to the authors of [HOVSG](https://github.com/hovsg/HOV-SG) and [ConceptGraphs](https://github.com/concept-graphs/concept-graphs) for their contributions and inspiration.

Special thanks to @[TOM-Huang](https://github.com/Tom-Huang) for his valuable advice and support throughout the development of this project.

We also thank the developers of [MobileCLIP](https://github.com/apple/ml-mobileclip), [CLIP](https://github.com/openai/CLIP), [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), and [YOLO-World](https://github.com/AILab-CVC/YOLO-World) for their excellent open-source work, which provided strong technical foundations for this project.