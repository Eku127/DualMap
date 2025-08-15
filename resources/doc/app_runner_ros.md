# Running with ROS1/ROS2

## 📚 Table of Contents
- [ROS1 (Noetic) Guide](#ros1-noetic-guide)
    - [Environment Setup](#environment-setup)
    - [Download Rosbag Data](#download-rosbag-data)
    - [Run with ROS1](#run-with-ros1)
- [ROS2 (Humble) Guide](#ros2-humble-guide)
    - [Environment Setup](#environment-setup-1)
    - [Download Rosbag Data](#download-rosbag-data-1)
    - [Run with ROS2](#run-with-ros2)
- [ROS1 <--> ROS2 Bridge](#ros1----ros2-bridge)


## ROS1 (Noetic) Guide

### Environment Setup

- **If you're using Ubuntu 20.04**:  
  Follow the official ROS1 Noetic installation guide:   [ROS Noetic Installation - Official Guide](https://wiki.ros.org/noetic/Installation/Ubuntu)

- **If you're using Ubuntu 22.04**:  
  ROS1 Noetic is not officially supported on Ubuntu 22.04, but you can use a community-maintained source. You can quickly set up the Noetic environment using the following commands:

```bash
echo "deb [trusted=yes arch=amd64] http://deb.repo.autolabor.com.cn jammy main" | sudo tee /etc/apt/sources.list.d/autolabor.list
sudo apt update
sudo apt install ros-noetic-autolabor
```

### Download Rosbag Data

Download real-world collected rosbags recorded across **two platforms** and **four different scenes**.

[Download from OneDrive](https://hkustgz-my.sharepoint.com/:f:/g/personal/jjiang127_connect_hkust-gz_edu_cn/Eo1dlzBIS8ZMqXuFpAlUelIBP5w-yDmnN2IfDbplzNJ5cA?e=YNiHx5)

| Scene         | Platform   | Task     | Rosbag Size    |
|---------------|------------|----------|---------|
| [Apartment](https://hkustgz-my.sharepoint.com/:f:/g/personal/jjiang127_connect_hkust-gz_edu_cn/Eqp3gLDW5TlJkNDvYzn-al4Bo_iNIDPAZGKDWOmYu4BOhw?e=Dbfgnl)     | Wheeled    | Mapping  | 3.14 GB |
| [Meeting Room](https://hkustgz-my.sharepoint.com/:f:/g/personal/jjiang127_connect_hkust-gz_edu_cn/Ev0zHaF9TLlAt1SFYIQgExMBKrYP90XV3Ct7ZSb2_xfdCQ?e=tGyy6w)  | Wheeled    | Mapping  | 886 MB  |
| [Hallway](https://hkustgz-my.sharepoint.com/:f:/g/personal/jjiang127_connect_hkust-gz_edu_cn/Eh_9p-vd-XhIsaC3MDAneSkBBPs7RuJkprLf7mnLGO7b6w?e=MLJV0e)       | Quadruped  | Mapping  | 2.48 GB |
| [Outdoor](https://hkustgz-my.sharepoint.com/:f:/g/personal/jjiang127_connect_hkust-gz_edu_cn/ElxsqYjXgNNAgavNC6HeDrIBeOn0y0qR9utmchhoqOocCA?e=KXbalC)       | Quadruped  | Mapping  | 881 MB  |

### Run with ROS1
#### Configurations
Before running, make sure to configure the following YAML files:

📁 `config/base_config.yaml`

```yaml
# Set the desired output directory; mapping results will be saved here.
output_path: ./output/map_results
```
📁 `config/system_config.yaml`
```yaml
# Choose the appropriate class list depending on the scene:
# - For Meeting Room and Hallway: use gpt_indoor_office
# - For Apartment: use gpt_indoor_apartment
# - For Outdoor: use gpt_outdoor_general

given_classes_path: ./config/class_list/gpt_indoor_office.txt
```

📁 `config/support_config/demo_config.yaml`
```yaml
# World axis rotation adjustment
# Modify the default world origin to match lab device orientation
world_roll: 0.0
world_pitch: 20.0
world_yaw: 0.0
```
> 📝 **Note:**  If you are using our provided real-world ROS1 rosbags, this configuration is crucial for correct alignment!

📁 `config/runner_ros.yaml`
```yaml
# Set the ROS topic configuration
ros_stream_config_path: ./config/data_config/ros/lab_device_online_ros1.yaml

# Whether to use compressed image topics
use_compressed_topic: true
```

#### Start Running

> 📝 **Note:** Make sure your `ROS_MASTER_URI` is set to `http://localhost:11311`

**1. Start roscore**

In a terminal:

```bash
source /opt/ros/noetic/setup.bash
roscore
```

**2. Activate DualMap Environment**

In a new terminal:
```bash
cd DualMap
conda activate dualmap
```
**3. Source ROS1 and Run DualMap**

Still in the new terminal:

```bash
source /opt/ros/noetic/setup.bash
python -m applications.runner_ros
```
You will see output like the following in the terminal:
<p align="center">
    <img src="../image/app_ros/app_ros1_output.jpg" width="70%">
</p>

**4. Start ROS1 Data Stream**

In another **new terminal**, start playing the ROS1 rosbag. Here we use the **Meeting Room** rosbag as an example:

```bash
rosbag play path/to/rosbag/car_meetingroom_mapping.bag
```

The final results will be visualized in **Rerun**, as shown below.  
Meanwhile, the **global (abstract) map** will be saved in your configured `output_path`.


<p align="center">
    <img src="../image/app_ros/app_ros1_meetingroom.jpg" width="70%">
</p>

## ROS2 (Humble) Guide

### Environment Setup

- **If you're using Ubuntu 22.04**:  
  Follow the official ROS2 Humble installation guide:   [ROS Humble Installation - Official Guide](https://docs.ros.org/en/humble/Installation.html)


### Download Rosbag Data

Download simulation rosbags recorded in three HM3D scenes via [Habitat Data Collector](https://github.com/Eku127/habitat-data-collector).

[Download from OneDrive](https://hkustgz-my.sharepoint.com/:f:/g/personal/jjiang127_connect_hkust-gz_edu_cn/EqLzgeEJZZVJpttVbDWVDXYBDyiGTMoFB3qaktQONetS6A?e=RJelVv)

| Scene  | Task     | Rosbag Size |
|--------|----------|-------------|
| 00829  | Mapping  | 17.2 GB     |
| 00848  | Mapping  | 21.2 GB     |
| 00880  | Mapping  | 23.3 GB     |


### Run with ROS2
#### Configurations
Before running, make sure to configure the following YAML files:

📁 `config/base_config.yaml`

```yaml
# Set the desired output directory; mapping results will be saved here.
output_path: ./output/map_results
```
📁 `config/system_config.yaml`
```yaml
# Choose the appropriate class list depending on the scene:
# - For HM3D collected rosbag, use hm3d300_classes_ycb

given_classes_path: ./config/class_list/hm3d300_classes_ycb.txt
```

📁 `config/support_config/demo_config.yaml`
```yaml
# World axis rotation adjustment
# Modify to the default world origin
world_roll: 0.0
world_pitch: 0.0
world_yaw: 0.0
```

📁 `config/runner_ros.yaml`
```yaml
# Set the ROS topic configuration
ros_stream_config_path: ./config/data_config/ros/self_collected.yaml

# Whether to use compressed image topics
use_compressed_topic: false
```

#### Start Running

**1. Activate DualMap Environment**

```bash
cd DualMap
conda activate dualmap
```
**2. Source ROS2 and Run DualMap**


```bash
source /opt/ros/humble/setup.bash
python -m applications.runner_ros
```
You will see output like the following in the terminal:
<p align="center">
    <img src="../image/app_ros/app_ros2_output.jpg" width="70%">
</p>

**4. Start ROS2 Data Stream**

In another **new terminal**, start playing the ROS2 rosbag. Here we use the **00829** rosbag as an example:

```bash
ros2 bag play path/to/rosbag/rosbag2_odom
```

The mapping results will be visualized in **Rerun**, as shown below.  
Meanwhile, the **global (abstract) map** will be saved in your configured `output_path`.


<p align="center">
    <img src="../image/app_ros/app_ros2_00829.jpg" width="70%">
</p>

## ROS1 <--> ROS2 Bridge

If you need to run a **ROS1 rosbag** within a **ROS2 environment**, or a **ROS2 rosbag** within a **ROS1 environment**,   follow [this guide](./ros_communication.md) to set up a ROS bridge between ROS1 and ROS2.

