# runner_record_3d.py

import time
import threading
import cv2
import numpy as np
import hydra

from collections import deque
from threading import Event
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R

from record3d import Record3DStream
from dualmap.core import Dualmap
from utils.types import DataInput
from utils.time_utils import timing_context


class DemoApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

        self.kf_idx = 0
        self.synced_data_queue = deque(maxlen=1)
        self.processing_thread = None

        self.stop_count = 0
        self.prev_count = -1

    def on_new_frame(self):
        self.event.set()  # Notify main thread of new frame arrival

    def on_stream_stopped(self):
        print("Stream stopped")

    def connect_to_device(self, dev_idx=0):
        print("Searching for devices...")
        devs = Record3DStream.get_connected_devices()
        print(f"{len(devs)} device(s) found")
        for dev in devs:
            print(f"\tID: {dev.product_id}\n\tUDID: {dev.udid}\n")

        if len(devs) <= dev_idx:
            raise RuntimeError(f"Cannot connect to device #{dev_idx}, try a different index.")

        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(devs[dev_idx])

    def get_intrinsic_matrix(self, coeffs):
        return np.array([[coeffs.fx, 0, coeffs.tx], [0, coeffs.fy, coeffs.ty], [0, 0, 1]])

    def start_processing_stream(self):
        while True:
            self.event.wait()

            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            confidence = self.session.get_confidence_frame()
            intrinsics = self.get_intrinsic_matrix(self.session.get_intrinsic_mat())
            pose = self.session.get_camera_pose()

            translation = np.array([pose.tx, pose.ty, pose.tz])
            quaternion = np.array([pose.qx, pose.qy, pose.qz, pose.qw])
            rotation_matrix = R.from_quat(quaternion).as_matrix()

            T = np.eye(4)
            T[:3, :3] = rotation_matrix
            T[:3, 3] = translation

            # Flip world YZ axes to ROS convention
            T[:, 1:3] *= -1

            # Rotate world frame: +90 deg around X
            T_fix = np.eye(4)
            T_fix[:3, :3] = R.from_euler('x', 90, degrees=True).as_matrix()
            transformation_matrix = T_fix @ T

            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            depth_resized = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            timestamp = time.time()

            data_input = DataInput(
                idx=self.kf_idx,
                time_stamp=timestamp,
                color=rgb,
                depth=depth_resized,
                color_name=str(timestamp),
                intrinsics=intrinsics,
                pose=transformation_matrix
            )

            self.synced_data_queue.append(data_input)
            self.stop_count += 1
            self.event.clear()

    def start_processing_in_thread(self):
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self.start_processing_stream, daemon=True)
            self.processing_thread.start()
            print("Stream processing thread started")
        else:
            print("Stream processing already running")

    def get_synced_data_queue(self):
        return self.synced_data_queue


@hydra.main(version_base=None, config_path="../config/", config_name="runner_record_3d")
def main(cfg: DictConfig):
    dualmap = Dualmap(cfg)
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_in_thread()

    end_count = 0

    while True:
        time.sleep(0.1)
        print("Main thread running...")

        if app.stop_count == app.prev_count:
            end_count += 1
        else:
            end_count = 0

        app.prev_count = app.stop_count

        if cfg.use_end_process and end_count > 50:
            print("No new frames detected. Terminating...")
            dualmap.end_process()
            break

        synced_queue = app.get_synced_data_queue()
        if not synced_queue:
            continue

        data_input = synced_queue[-1]
        if data_input is None:
            continue

        if not dualmap.check_keyframe(data_input.time_stamp, data_input.pose):
            continue

        kf_idx = dualmap.get_keyframe_idx()
        data_input.idx = kf_idx

        with timing_context("Time Per Frame", dualmap):
            if cfg.use_parallel:
                dualmap.parallel_process(data_input)
            else:
                dualmap.sequential_process(data_input)


if __name__ == "__main__":
    main()
