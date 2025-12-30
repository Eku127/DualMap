# runner_ros2.py

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from omegaconf import OmegaConf
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from applications.utils.ros_publisher import ROSPublisher
from applications.utils.runner_ros_base import RunnerROSBase
from dualmap.core import Dualmap
from utils.logging_helper import setup_logging


class RunnerROS2(Node, RunnerROSBase):
    """
    ROS2-specific runner. Uses rclpy and ROS2 message_filters for synchronization,
    subscription, and publishing.
    """

    def __init__(self, cfg):
        Node.__init__(self, "runner_ros")
        setup_logging(output_path=cfg.output_path, config_path=cfg.logging_config)
        self.logger = logging.getLogger(__name__)
        self.logger.info("[Runner ROS2]")
        self.logger.info(OmegaConf.to_yaml(cfg))

        self.cfg = cfg
        self.dualmap = Dualmap(cfg)
        RunnerROSBase.__init__(self, cfg, self.dualmap)

        self.bridge = CvBridge()
        self.dataset_cfg = OmegaConf.load(cfg.ros_stream_config_path)
        self.intrinsics = self.load_intrinsics(self.dataset_cfg)
        self.extrinsics = self.load_extrinsics(self.dataset_cfg)

        # Topic Subscribers
        if self.cfg.use_compressed_topic:
            self.logger.warning("[Main] Using compressed topics.")
            self.rgb_sub = Subscriber(
                self, CompressedImage, self.dataset_cfg.ros_topics.rgb
            )
            self.depth_sub = Subscriber(
                self, CompressedImage, self.dataset_cfg.ros_topics.depth
            )
        else:
            self.logger.warning("[Main] Using uncompressed topics.")
            self.rgb_sub = Subscriber(self, Image, self.dataset_cfg.ros_topics.rgb)
            self.depth_sub = Subscriber(self, Image, self.dataset_cfg.ros_topics.depth)

        self.odom_sub = Subscriber(self, Odometry, self.dataset_cfg.ros_topics.odom)

        # Sync messages
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.odom_sub],
            queue_size=10,
            slop=self.cfg.sync_threshold,
        )
        self.sync.registerCallback(self.synced_callback)

        # CameraInfo fallback
        self.create_subscription(
            CameraInfo,
            self.dataset_cfg.ros_topics.camera_info,
            self.camera_info_callback,
            10,
        )

        # Publisher and timer
        self.publisher = ROSPublisher(self, cfg)
        self.publish_executor = ThreadPoolExecutor(max_workers=2)

        timer_period = 1.0 / self.cfg.ros_rate
        self.timer = self.create_timer(timer_period, self.run)

    def synced_callback(self, rgb_msg, depth_msg, odom_msg):
        """Callback for synced RGB-D-Odom input."""
        timestamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9

        if self.cfg.use_compressed_topic:
            rgb_img = self.decompress_image(rgb_msg.data, is_depth=False)
            depth_img = self.decompress_image(depth_msg.data, is_depth=True)
        else:
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

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
        self.last_message_time = self.get_clock().now().nanoseconds / 1e9

    def camera_info_callback(self, msg):
        """Populate intrinsics from CameraInfo topic if not already loaded."""
        if self.intrinsics is None:
            self.intrinsics = np.array(msg.k).reshape(3, 3)
            self.logger.warning("[Main] Camera intrinsics received and stored.")

    def run(self):
        """Periodic processing loop triggered by ROS2 timer."""
        self.run_once(lambda: self.get_clock().now().nanoseconds / 1e9)
        self.publish_executor.submit(self.publisher.publish_all, self.dualmap)

    def shutdown_all_threads(self):
        """Clean up all threads and timers."""
        self.logger.warning("[Main] Shutting down all threads and timers.")
        try:
            self.timer.cancel()
        except Exception as e:
            self.logger.warning(f"[Main] Failed to cancel timer: {e}")
        self.publish_executor.shutdown(wait=True)

    def destroy_node(self):
        """Override base destroy_node with cleanup logic."""
        self.shutdown_all_threads()
        super().destroy_node()


def run_ros2(cfg):
    """Entry point for launching ROS2 runner."""
    rclpy.init()
    runner = RunnerROS2(cfg)
    runner.logger.warning("[Main] ROS2 Runner started. Waiting for data stream...")
    try:
        while rclpy.ok() and not runner.shutdown_requested:
            rclpy.spin_once(runner, timeout_sec=0.1)
    except KeyboardInterrupt:
        runner.logger.warning("[Main] KeyboardInterrupt received. Shutting down.")
    finally:
        runner.destroy_node()
        rclpy.shutdown()
        runner.logger.warning("[Main] Done.")
