# Online Mapping and Navigation in Simulation

> **Note:** Before continuing, make sure you have correctly set up the [Habitat Data Collector](https://github.com/Eku127/habitat-data-collector).  
> All of the following functionalities rely on this simulation tool.

> **Note:** ROS2 is required for these features. It is recommended to complete the ROS2 usage in the [ROS Runner Guide](./app_runner_ros.md) before using the following features.


## Table of Content

## Online Interactive Mapping

Before running, make sure to configure the following YAML files in DualMap repo:

📁 `config/system_config.yaml`
```yaml
# Choose the appropriate class list depending on the scene:
# - For HM3D scenes, use hm3d300_classes_ycb
# - For Replica scenes, both gpt_indoor_general and scannet200_classes are ok

given_classes_path: ./config/class_list/hm3d300_classes_ycb.txt
```

📁 `config/runner_ros.yaml`
```yaml
# Set the ROS topic configuration
ros_stream_config_path: ./config/data_config/ros/self_collected.yaml

# Whether to use compressed image topics
use_compressed_topic: false
```
Then, open a terminal and start DualMap:

```bash
cd DualMap
conda activate dualmap

source /opt/ros/humble/setup.bash
python -m applications.runner_ros
```

Next, open a **new terminal** and start the Habitat Data Collector:
```bash
cd <Path to >/habitat-data-collector
conda activate habitat_data_collector

source source /opt/ros/humble/setup.bash
python -m habitat_data_collector.main
```

> You can start DualMap and the Habitat Data Collector in any order.

Now, you can move the agent with the Habitat Data Collector.  
DualMap will continue mapping in real time and display the results in **Rerun** according to your movements and observations.

<p align="center">
  <img src="../video/interactive_mapping.gif" width="60%">
</p>

## Preloading Pre-built Map
DualMap supports loading a previously built **global (abstract) map** from disk.   This functionality is essential for enabling navigation in both **static** and **dynamic** environments.

Below, we provide an example of a pre-built map from simulation.  Note that we can also preload pre-built maps from real-world data, enabling real-world navigation tests under object relocation scenarios.

First, check whether the `global_map` folder exists inside the downloaded `HM3D_collect` or `DualMap_ROS2_Bags` dataset.   Example path:
```
HM3D_collect/00829-QaLdnwvtxbs/global_map
```

Change the configs to enable **preloading** and ensure the **class list** is compatible with the one used to build the map.

📁 `config/system_config.yaml`
```yaml
# Use the class list that matches the pre-built map.
# For preloading HM3D pre-built maps, use:
given_classes_path: ./config/class_list/hm3d300_classes_ycb.txt
```

📁 `config/runner_ros.yaml`
```yaml
# Preload an existing global (abstract) map from disk
preload_global_map: true

# Preload layout and wall point clouds from disk
preload_layout: true

# Path to the pre-built map folder (example)
preload_path: "<path-to>/HM3D_collect/00829-QaLdnwvtxbs/global_map/hm3d"
# Or
preload_path: "<path-to>/DualMap_ROS2_Bags/00829/global_map/hm3d"
```
> **Note:** When preloading, you **must** use the same class list as the one used when building the map.  
> Using a different class list will cause category/ID mismatches and result in errors.

Then, open a terminal and start DualMap:

```bash
cd DualMap
conda activate dualmap

source /opt/ros/humble/setup.bash
python -m applications.runner_ros
```

Next, open a **new terminal** and start the Habitat Data Collector to trigger the start of the system:
```bash
cd <Path to >/habitat-data-collector
conda activate habitat_data_collector

source source /opt/ros/humble/setup.bash
python -m habitat_data_collector.main
```
You can also use a `rosbag` to trigger the mapping.

You will see the pre-built map is now preloaded form the disk.

<p align="center">
  <img src="../image/app_simulation/app_preload_00829.jpg" width="60%">
</p>

## Online Navigation
With the help of the **Habitat Data Collector** and the **Preloading Pre-built Map** functionality, we can now start navigation.

### Prerequisites
Here we use the `00829` scene as an example.  
Make sure the **preload path** in DualMap and the **scene** in Habitat Data Collector are set up correctly.

Follow the same process as before:  
- Open DualMap with the pre-built map.  
- Launch Habitat Data Collector with the corresponding scene.

**Terminal A** — Start DualMap
```bash
cd DualMap
conda activate dualmap

source /opt/ros/humble/setup.bash
python -m applications.runner_ros
```
**Terminal B** — Start Habitat Data Collector
```bash
cd <Path to >/habitat-data-collector
conda activate habitat_data_collector

source source /opt/ros/humble/setup.bash
python -m habitat_data_collector.main
```

### Configurations
Now open the `config/actions.yaml` file — this configuration controls:  
1. The navigation mode  
2. The navigation query sentence  
3. Whether navigation starts

📁 `config/actions.yaml`
```yaml
# Whether to start navigation, set to True to start
calculate_path: false

# Navigation mode:
# 1. random  — The system randomly selects a navigable goal point
# 2. click   — Shows the navigation map; click on a goal point to start navigation
# 3. inquiry — Navigates to the target object with the highest similarity to the inquiry_sentence
get_goal_mode: click

# Target description used when get_goal_mode is "inquiry"
inquiry_sentence: bowl

# If the navigation attempt fails, set this to true to trigger navigation to the next target
# It is very useful in dynamic object navigation
trigger_find_next: false
```

### Navigation Mode - 'random'

1. Set `get_goal_mode` to `random` in `config/actions.yaml`.  
2. Set `calculate_path` to `true`.

The system will automatically plan a path and start navigation.

https://github.com/user-attachments/assets/d338280c-9d95-4303-a6e0-d3d6520cea01

### Navigation Mode - 'click'
1. Set `get_goal_mode` to `click` in `config/actions.yaml`.  
2. Set `calculate_path` to `true`.

A navigation map will appear. You can use your mouse to click on a point to start navigation.  
- **White** areas indicate navigable regions.  
- **Black** areas indicate non-navigable regions.  

If you click on a black area, the system will automatically plan a path to the nearest navigable point to your click.

https://github.com/user-attachments/assets/c3a0f533-ebb7-4bb9-978f-a160f7f4ba5b

### Navigation Mode - 'inquiry'
1. Set `get_goal_mode` to `inquiry` in `config/actions.yaml`.  
2. Set `inquiry_sentence` to the object you want to navigate to. It can be a descriptive sentence like `place to sit`.  
3. Set `calculate_path` to `true`.

The system will automatically plan a global path (green) based on the current map and start navigation to the anchor object with the highest similarity to the `inquiry_sentence`.  

During the navigation process, DualMap keeps building the local (concrete) map with incoming observations.  

When the agent is near the target anchor object and the queried object is on the anchor object (or is the anchor itself), the system will automatically plan a local path (red) based on the local map. The agent will then follow the local path and reach the target queried object.

If no local path is planned, the navigation attempt is considered failed.  
You can set `trigger_find_next` to `true` in `config/actions.yaml` to start the next navigation attempt automatically.

https://github.com/user-attachments/assets/06f0b357-13c6-4d5a-a800-c4c531c8c1a8