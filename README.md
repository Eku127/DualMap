# DualMap
<h3>
  <a href="https://eku127.github.io/DualMap/">Project Page</a> |
  <a href="https://arxiv.org/abs/2103.xxxx">Paper</a> |
  <a href="https://youtu.be/ZmZDvhyXL_g">Video</a>
</h3>

<p align="center">
  <img src="media/teaser.jpg" width="60%">
</p>


**DualMap** is an online open-vocabulary mapping system that enables robots to understand and navigate dynamic 3D environments using natural language. 

The system supports multiple input sources, including offline datasets (**Dataset Mode**), ROS streams & rosbag files (**ROS Mode**), and iPhone video streams(**Record3d Mode**). We provide examples for each input type.

## Updates
**[2025.06]** Full code release is coming soon—stay tuned!

## Release Plan

- [ ] Environment setup & dataset links  
- [ ] Simulation tools for dynamic scenes  
- [ ] Full system code (mapping + querying)  
- [ ] Evaluation & benchmarking scripts
- [ ] [Examples] Offline query
- [ ] [Examples] Realworld guidance

## Installation

> ✅ Tested on **Ubuntu 22.04** with **ROS 2 Humble** and **Python 3.10+**

### 1. Clone the Repository (with submodules)

```bash
git clone --recurse-submodules git@github.com:Eku127/DualMap.git
cd DualMap
```

>  Cloning may take some time, especially for Habitat-related submodules. If you forget `--recurse-submodules`, you can initialize them manually:

```bash
git submodule update --init --recursive
```

### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate dualmap
```

### 3. Install MobileCLIP
```bash
cd 3rdparty/mobileclip
pip install -e . --no-deps
```

### (Optional) Setup ROS 2 Environment
Setting up ROS2 environment for ROS support and applications.
We recommend [ROS 2 Humble](https://docs.ros.org/en/humble/Installation.html).
Once installed, activate the environment:

```bash
source /opt/ros/humble/setup.bash
```

### (Optional) Setup Habitat Data Collector

[Habitat Data Collector](https://github.com/Eku127/habitat-data-collector) is a tool built on top of the Habitat simulator. It supports agent control, object manipulation, dataset and ROS bag recording, as well as navigation through external ROS topics. DualMap subscribes to live ROS topics from the collector for real-time mapping and language-guided querying, and publishes navigation trajectories for the agent to follow.

> For the best DualMap experience, we strongly recommend setting up the Habitat Data Collector. See [the repo](https://github.com/Eku127/habitat-data-collector) for installation and usage details.


## Dataset

## Examples