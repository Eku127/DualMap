# runner_ros.py
import importlib.util
import logging
import hydra
from omegaconf import DictConfig
from utils.logging_helper import setup_logging


def detect_ros_version():
    if importlib.util.find_spec("rclpy") is not None:
        return "ros2"
    elif importlib.util.find_spec("rospy") is not None:
        return "ros1"
    else:
        return None

@hydra.main(version_base=None, config_path="../config/", config_name="runner_ros")
def main(cfg: DictConfig):
    setup_logging(output_path=cfg.output_path, config_path=cfg.logging_config)
    logger = logging.getLogger(__name__)

    ros_version = detect_ros_version()

    if ros_version == "ros1":
        logger.warning("[runner_ros] Detected ROS1 environment. Running ROS1 runner.")
        from applications.utils.runner_ros1 import run_ros1
        run_ros1(cfg)

    elif ros_version == "ros2":
        logger.warning("[runner_ros] Detected ROS2 environment. Running ROS2 runner.")
        from applications.utils.runner_ros2 import run_ros2
        run_ros2(cfg)
    else:
        logger.error("[runner_ros] Could not detect ROS1 or ROS2 environment.")
        raise RuntimeError(
            "[runner_ros] Could not detect ROS1 or ROS2 environment. Please source your ROS workspace before running this script."
        )

if __name__ == "__main__":
    main()
