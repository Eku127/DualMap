# runner_ros1.py

import logging
import threading
import time

import numpy as np
import rospy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from omegaconf import OmegaConf
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from applications.utils.runner_ros_base import RunnerROSBase
from dualmap.core import Dualmap
from utils.logging_helper import setup_logging


class RunnerROS1(RunnerROSBase):
    """
    ROS1-specific runner, handles topic subscriptions and data flow using rospy.
    """

    def __init__(self, cfg):
        rospy.init_node("runner_ros", anonymous=True)
        setup_logging(output_path=cfg.output_path, config_path=cfg.logging_config)
        self.logger = logging.getLogger(__name__)
        self.logger.info("[Runner ROS1]")
        self.logger.info(OmegaConf.to_yaml(cfg))

        self.cfg = cfg
        self.dualmap = Dualmap(cfg)
        super().__init__(cfg, self.dualmap)

        self.bridge = CvBridge()
        self.dataset_cfg = OmegaConf.load(cfg.ros_stream_config_path)
        self.intrinsics = self.load_intrinsics(self.dataset_cfg)
        self.extrinsics = self.load_extrinsics(self.dataset_cfg)

        # Image and Odometry Subscribers
        if self.cfg.use_compressed_topic:
            self.logger.warning("[Main] Using compressed topics.")
            self.rgb_sub = Subscriber(self.dataset_cfg.ros_topics.rgb, CompressedImage)
            self.depth_sub = Subscriber(
                self.dataset_cfg.ros_topics.depth, CompressedImage
            )
        else:
            self.logger.warning("[Main] Using uncompressed topics.")
            self.rgb_sub = Subscriber(self.dataset_cfg.ros_topics.rgb, Image)
            self.depth_sub = Subscriber(self.dataset_cfg.ros_topics.depth, Image)

        self.odom_sub = Subscriber(self.dataset_cfg.ros_topics.odom, Odometry)

        # Sync RGB + Depth + Odometry
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.odom_sub],
            queue_size=10,
            slop=self.cfg.sync_threshold,
        )
        self.sync.registerCallback(self.synced_callback)

        # Fallback to camera_info topic if intrinsics not loaded
        rospy.Subscriber(
            self.dataset_cfg.ros_topics.camera_info,
            CameraInfo,
            self.camera_info_callback,
        )

    def synced_callback(self, rgb_msg, depth_msg, odom_msg):
        """Callback for synchronized RGB, Depth, and Odom messages."""
        timestamp = rgb_msg.header.stamp.to_sec()

        if self.cfg.use_compressed_topic:
            rgb_img = self.decompress_image(rgb_msg.data, is_depth=False)
            depth_img = self.decompress_image(depth_msg.data, is_depth=True)
        else:
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")

        depth_factor = getattr(self.dataset_cfg, 'depth_factor', 1000.0)
        depth_img = depth_img.astype(np.float32) / depth_factor
        depth_img = np.expand_dims(depth_img, axis=-1)

        translation = np.array(
            [
                odom_msg.pose.pose.position.x,
                odom_msg.pose.pose.position.y,
                odom_msg.pose.pose.position.z,
            ]
        )
        quaternion = np.array(
            [
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w,
            ]
        )

        pose_matrix = self.build_pose_matrix(translation, quaternion)
        self.push_data(rgb_img, depth_img, pose_matrix, timestamp)
        self.last_message_time = time.time()

    def camera_info_callback(self, msg):
        """Fallback callback to get intrinsics from CameraInfo if needed."""
        if self.intrinsics is None:
            self.intrinsics = np.array(msg.K).reshape(3, 3)
            self.logger.warning("[Main] Camera intrinsics received and stored.")

    def spin(self):
        """Main loop calling run_once() at configured ROS rate."""
        rate = rospy.Rate(self.cfg.ros_rate)
        while not rospy.is_shutdown() and not self.shutdown_requested:
            try:
                self.run_once(lambda: time.time())
            except Exception as e:
                self.logger.error(f"[RunnerROS1] Exception: {e}", exc_info=True)
            rate.sleep()


def run_ros1(cfg):
    """Launch the ROS1 runner in a background thread."""
    runner = RunnerROS1(cfg)
    runner.logger.warning("[Main] ROS1 Runner started. Waiting for data stream...")

    spin_thread = threading.Thread(target=runner.spin)
    spin_thread.start()

    try:
        while not rospy.is_shutdown() and not runner.shutdown_requested:
            time.sleep(0.1)
    except KeyboardInterrupt:
        runner.logger.warning("[Main] KeyboardInterrupt received.")
    finally:
        runner.shutdown_requested = True
        runner.logger.warning("[Main] Shutting down...")
        spin_thread.join(timeout=3.0)

        try:
            rospy.signal_shutdown("User requested shutdown")
        except Exception:
            pass

        runner.logger.warning("[Main] Exit complete.")

        import os

        os._exit(0)
