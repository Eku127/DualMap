# runner_ros_base.py

import logging
import time
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.time_utils import timing_context
from utils.types import DataInput


class RunnerROSBase:
    """
    Base class for ROS1 and ROS2 runners.
    Handles shared logic such as intrinsics/extrinsics loading,
    image decompression, pose conversion, and keyframe processing.
    """

    def __init__(self, cfg, dualmap):
        self.cfg = cfg
        self.dualmap = dualmap
        self.logger = logging.getLogger(__name__)

        self.kf_idx = 0
        self.intrinsics = None
        self.extrinsics = None
        self.synced_data_queue = deque(maxlen=1)
        self.shutdown_requested = False
        self.last_message_time = None

    def load_intrinsics(self, dataset_cfg):
        """Load camera intrinsics from config file."""
        intrinsic_cfg = dataset_cfg.get("intrinsic", None)
        if intrinsic_cfg:
            fx, fy, cx, cy = (
                intrinsic_cfg["fx"],
                intrinsic_cfg["fy"],
                intrinsic_cfg["cx"],
                intrinsic_cfg["cy"],
            )
            self.logger.warning("[Main] Loaded intrinsics from config.")
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.logger.warning("[Main] No intrinsics provided.")
        return None

    def load_extrinsics(self, dataset_cfg):
        """Load camera extrinsics from config file."""
        extrinsic_cfg = dataset_cfg.get("extrinsics", None)
        if extrinsic_cfg:
            matrix = np.array(extrinsic_cfg)
            if matrix.shape == (4, 4):
                self.logger.warning("[Main] Loaded extrinsics from config.")
                return matrix
        self.logger.warning(
            "[Main] No valid extrinsics provided. Using identity matrix."
        )
        return np.eye(4)

    def create_world_transform(self):
        """Create world coordinate transformation from roll/pitch/yaw."""
        roll = np.radians(self.cfg.world_roll)
        pitch = np.radians(self.cfg.world_pitch)
        yaw = np.radians(self.cfg.world_yaw)

        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )
        Ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )
        Rz = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        R_combined = Rz @ Ry @ Rx
        T = np.eye(4)
        T[:3, :3] = R_combined
        return T

    def decompress_image(self, msg_data, is_depth=False):
        """Decode compressed image data (RGB or depth)."""
        msg_data = bytes(msg_data)
        if is_depth:
            depth_data = np.frombuffer(msg_data[12:], np.uint8)
            img = cv2.imdecode(depth_data, cv2.IMREAD_UNCHANGED)
        else:
            np_arr = np.frombuffer(msg_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def build_pose_matrix(self, translation, quaternion):
        """Construct 4x4 pose matrix from translation and quaternion."""
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation
        return transformation_matrix

    def push_data(self, rgb_img, depth_img, pose, timestamp):
        """Push synchronized input data into queue for processing."""
        transformed_pose = self.create_world_transform() @ (pose @ self.extrinsics)

        data_input = DataInput(
            idx=self.kf_idx,
            time_stamp=timestamp,
            color=rgb_img,
            depth=depth_img,
            color_name=str(timestamp),
            intrinsics=self.intrinsics,
            pose=transformed_pose,
        )
        self.synced_data_queue.append(data_input)
        return data_input

    def run_once(self, current_time_fn):
        """Check and process a keyframe if data is ready."""
        if not self.synced_data_queue:
            return

        data_input = self.synced_data_queue[-1]

        if not self.dualmap.calculate_path:
            current_time = current_time_fn()
            last_time = self.last_message_time
            if self.cfg.use_end_process and last_time is not None:
                if current_time - last_time > 20.0:
                    self.logger.warning(
                        "[Main] No new data received. Entering end process."
                    )
                    self.dualmap.end_process()
                    self.shutdown_requested = True
                    return

        if not self.dualmap.check_keyframe(data_input.time_stamp, data_input.pose):
            return

        data_input.idx = self.dualmap.get_keyframe_idx()

        self.logger.info(
            "[Main] ============================================================"
        )
        with timing_context("Time Per Frame", self.dualmap):
            if self.cfg.use_parallel:
                self.dualmap.parallel_process(data_input)
            else:
                self.dualmap.sequential_process(data_input)

        self.logger.info(
            f"[Main] Processing keyframe {data_input.idx} took {time.time() - data_input.time_stamp:.2f} seconds."
        )
