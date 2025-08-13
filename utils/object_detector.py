import time
import os
import logging
import gzip
import pickle
import threading
from pathlib import Path

import torch
import numpy as np
import supervision as sv
import cv2
import open_clip
import open3d as o3d
from scipy.spatial.transform import Rotation as R, Slerp
from ultralytics import YOLO, SAM, FastSAM
from PIL import Image
from omegaconf import DictConfig
from sklearn.metrics.pairwise import cosine_similarity

from utils.types import DataInput, ObjectClasses, LocalObservation
from utils.pcd_utils import (
    mask_depth_to_points,
    refine_points_with_clustering,
    safe_create_bbox,
)
from utils.visualizer import ReRunVisualizer, visualize_result_rgb
from utils.time_utils import timing_context

import pdb

# Set up the module-level logger
logger = logging.getLogger(__name__)


class PoseLowPassFilter:
    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.initialized = False
        self.smoothed_translation = None
        self.smoothed_rotation = None  # Rotation object (scipy)

    def update(self, pose_mat: np.ndarray) -> np.ndarray:
        """
        input 4x4 pose matrix, output smoothed 4x4 pose matrix.
        """
        curr_translation = pose_mat[:3, 3]
        curr_rotation = R.from_matrix(pose_mat[:3, :3])

        if not self.initialized:
            self.smoothed_translation = curr_translation
            self.smoothed_rotation = curr_rotation
            self.initialized = True
        else:
            # translation filtering
            self.smoothed_translation = (
                self.alpha * self.smoothed_translation
                + (1 - self.alpha) * curr_translation
            )

            # rotation using slerp
            slerp = Slerp(
                [0, 1], R.concatenate([self.smoothed_rotation, curr_rotation])
            )
            self.smoothed_rotation = slerp(1 - self.alpha)

        T_smooth = np.eye(4)
        T_smooth[:3, :3] = self.smoothed_rotation.as_matrix()
        T_smooth[:3, 3] = self.smoothed_translation
        return T_smooth


class Detector:
    # Given input output detection

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        """
        Initialize the Detector class.

        Parameters:
        cfg (DictConfig): A configuration object containing paths, parameters, and settings for the detector.

        Returns:
        None
        """

        # Object classes
        classes_path = cfg.yolo.classes_path
        if cfg.yolo.use_given_classes:
            classes_path = cfg.yolo.given_classes_path
            logger.info(f"[Detector][Init] Using given classes, path:{classes_path}")

        self.obj_classes = ObjectClasses(
            classes_file_path=classes_path,
            bg_classes=cfg.yolo.bg_classes,
            skip_bg=cfg.yolo.skip_bg,
        )

        # get detection paths
        self.detection_path = Path(cfg.detection_path)
        self.detection_path.mkdir(parents=True, exist_ok=True)

        # Configs
        self.cfg = cfg
        # Detection results
        # NOTICE: Detection results are stored in Batch, it is not separated by objects
        self.curr_results = {}
        # Data input
        self.curr_data = DataInput()
        self.prev_data = None
        # KF for layout keyframe
        self.prev_kf_data = None

        # masked points and colors
        self.masked_points = []
        self.masked_colors = []
        # Observations, a list for each obj observation
        self.curr_observations = []

        # visualizer
        self.visualizer = ReRunVisualizer()
        self.annotated_image = None

        # Variables for FastSAM
        self.unknown_class_id = len(self.obj_classes.get_classes_arr()) - 1
        self.annotated_image_fs = None
        self.annotated_image_fs_after = None

        # Layout Pointcloud
        self.layout_pointcloud = o3d.geometry.PointCloud()
        self.layout_num = 0
        self.layout_time = 0.0
        # For thread processing
        self.layout_lock = (
            threading.Lock()
        )  # Thread lock for protecting layout_pointcloud
        self.data_thread = None  # Thread handle
        self.data_event = threading.Event()  # Thread notification event

        logger.info(f"[Detector][Init] Initilizating detection modules...")

        if cfg.run_detection:
            try:
                # CLIP module
                logger.info(
                    f"[Detector][Init] Loading CLIP model: {cfg.clip.model_name} with pretrained weights '{cfg.clip.pretrained}'"
                )

                self.clip_model, _, self.clip_preprocess = (
                    open_clip.create_model_and_transforms(
                        cfg.clip.model_name, pretrained=cfg.clip.pretrained
                    )
                )
                self.clip_model = self.clip_model.to(cfg.device)
                self.clip_model.eval()

                # Only reparameterize if the model is MobileCLIP
                if "MobileCLIP" in cfg.clip.model_name:
                    from mobileclip.modules.common.mobileone import reparameterize_model

                    self.clip_model = reparameterize_model(self.clip_model)

                self.clip_tokenizer = open_clip.get_tokenizer(cfg.clip.model_name)
            except Exception as e:
                logger.error(f"[Detector][Init] Error loading CLIP model: {e}")
                return

            try:
                # Detection module
                logger.info(
                    f"[Detector][Init] Loading YOLO model from\t{cfg.yolo.model_path}"
                )
                self.yolo = YOLO(cfg.yolo.model_path)
                self.yolo.set_classes(self.obj_classes.get_classes_arr())
            except Exception as e:
                logger.error(f"[Detector][Init] Error loading YOLO model: {e}")
                return

            try:
                # Segmentation module
                logger.info(
                    f"[Detector][Init] Loading SAM model from\t{cfg.sam.model_path}"
                )
                self.sam = SAM(cfg.sam.model_path)
            except Exception as e:
                logger.error(f"[Detector][Init] Error loading SAM model: {e}")
                return

            # Open fastsam for open vocabulary detection
            if cfg.use_fastsam:
                try:
                    logger.info(
                        f"[Detector][Init] Loading FastSAM model from\t{cfg.fastsam.model_path}"
                    )
                    self.fastsam = FastSAM(cfg.fastsam.model_path)
                except Exception as e:
                    logger.error(f"[Detector][Init] Error loading FASTSAM model: {e}")
                    return

            logger.info("[Detector][Init] Initializing high-low mobility classifier.")
            lm_examples = cfg.lm_examples
            hm_examples = cfg.hm_examples
            lm_descriptions = cfg.lm_descriptions
            num_examples = [len(lm_examples), len(hm_examples), len(lm_descriptions)]
            prototypes = lm_examples + hm_examples + lm_descriptions
            proto_feats = get_text_features(
                prototypes,
                self.clip_model,
                self.clip_tokenizer,
                device=cfg.device,
                clip_length=cfg.clip.clip_length,
            )
            self.num_examples = num_examples
            self.proto_feats = proto_feats

            # Get the text feats of all the classes
            class_feats = get_text_features(
                self.obj_classes.get_classes_arr(),
                self.clip_model,
                self.clip_tokenizer,
                device=cfg.device,
                clip_length=cfg.clip.clip_length,
            )
            self.class_feats = class_feats

            # Used for unknown class
            if cfg.use_avg_feat_for_unknown:
                class_feats_mean = np.mean(class_feats, axis=0)
                self.class_feats_mean = class_feats_mean / np.linalg.norm(
                    class_feats_mean
                )

            with timing_context("Detection Filter", self):
                self.filter = Filter(
                    classes=self.obj_classes,
                    small_mask_size=self.cfg.small_mask_th,
                    skip_refinement=self.cfg.skip_refinement,
                )
                self.filter.set_device(self.cfg.device)

        # for filtering the pose of follower camera for visualization
        self.pose_filter_follower = PoseLowPassFilter(alpha=0.95)

        logger.info(f"[Detector][Init] Finish Init.")

    def update_state(self) -> None:
        self.curr_results = {}
        self.curr_observations = []
        # self.prev_data = self.curr_data.copy()
        # self.curr_data.clear()

    def update_data(self) -> None:
        # self.curr_results = {}
        # self.curr_observations = []
        self.prev_data = self.curr_data.copy()

        # self.curr_data.clear()

    def set_data_input(self, curr_data: DataInput) -> None:
        self.curr_data = curr_data

        if not self.cfg.preload_layout:
            # If a thread is already running, wait for it to finish
            if self.data_thread and self.data_thread.is_alive():
                self.data_thread.join()

            # Create a new thread to process data input
            self.data_thread = threading.Thread(target=self._process_data_input_thread)
            self.data_thread.start()

    def _process_data_input_thread(self):
        """
        Logic executed in the background thread.
        """
        # Initialize prev_kf_data and layout_pointcloud
        if self.prev_kf_data is None:
            self.prev_kf_data = self.curr_data.copy()
            layout_pcd = self.depth_to_point_cloud(sample_rate=16)
            with self.layout_lock:  # Ensure thread safety for layout_pointcloud
                self.layout_pointcloud += layout_pcd.voxel_down_sample(
                    voxel_size=self.cfg.layout_voxel_size
                )
            logger.info(
                f"[Detector][Layout] Initialized layout pointcloud with {len(self.layout_pointcloud.points)} points."
            )
            return

        # Print current frame index
        logger.info(f"[Detector][Layout] Processing frame idx: {self.curr_data.idx}")

        # Check if layout_pointcloud needs to be updated
        if self.check_keyframe_for_layout_pcd():
            start_time = time.time()

            # Generate current frame point cloud
            current_pcd = self.depth_to_point_cloud(sample_rate=16)

            # Merge point clouds
            with self.layout_lock:
                self.layout_pointcloud += current_pcd
                logger.info(
                    f"[Detector][Layout] Points before downsample: {len(self.layout_pointcloud.points)}"
                )
                self.layout_pointcloud = self.layout_pointcloud.voxel_down_sample(
                    voxel_size=self.cfg.layout_voxel_size
                )
                logger.info(
                    f"[Detector][Layout] Points after downsample: {len(self.layout_pointcloud.points)}"
                )

            # Update prev_kf_data
            self.prev_kf_data = self.curr_data.copy()
            logger.info("[Detector][Layout] Updated layout pointcloud.")

            # Update time and count
            end_time = time.time()
            layout_time = end_time - start_time
            self.layout_time += layout_time
            self.layout_num += 1
            logger.info(
                f"[Detector][Layout] Layout update took {layout_time:.4f} seconds."
            )

    def get_layout_pointcloud(self):
        """
        Return the current layout_pointcloud.
        """
        with self.layout_lock:
            return self.layout_pointcloud

    def save_layout(self):
        if self.layout_pointcloud is not None:
            layout_pcd = self.get_layout_pointcloud()
            save_dir = self.cfg.map_save_path
            layout_pcd_path = save_dir + "/layout.pcd"
            o3d.io.write_point_cloud(layout_pcd_path, layout_pcd)
            logger.info(f"[Detector][Layout] Saving layout to: {layout_pcd_path}")

    def load_layout(self):
        """
        Load layout point cloud layout.pcd, prefer preload_path,
        if not exist, use map_save_path. Skip loading if path or file is missing.
        """
        # Prefer preload_layout_path, if not exist then use map_save_path
        if os.path.exists(self.cfg.preload_path):
            load_dir = self.cfg.preload_path
            logger.info(f"[Detector][Layout] Using preload layout path: {load_dir}")
        else:
            load_dir = self.cfg.map_save_path
            logger.info(
                f"[Detector][Layout] Preload layout path not found. Using default map save path: {load_dir}"
            )

        # Build layout point cloud file path
        layout_pcd_path = os.path.join(load_dir, "layout.pcd")

        # Check if layout point cloud file exists
        if not Path(layout_pcd_path).is_file():
            logger.info(
                f"[Detector][Layout] Layout file not found at: {layout_pcd_path}"
            )
            return None

        # Load layout point cloud
        layout_pcd = o3d.io.read_point_cloud(layout_pcd_path)
        logger.info(f"[Detector][Layout] Layout loaded from: {layout_pcd_path}")

        # Save to class attribute
        self.layout_pointcloud = layout_pcd

    def get_curr_data(
        self,
    ) -> DataInput:
        return self.curr_data

    def get_curr_observations(self) -> None:
        return self.curr_observations

    def check_keyframe_for_layout_pcd(self):
        """
        Check if the current frame should be selected as a keyframe based on
        time interval, pose difference (translation), and rotation difference.
        """
        curr_pose = self.curr_data.pose
        prev_kf_pose = self.prev_kf_data.pose

        # Translation check
        translation_diff = np.linalg.norm(
            curr_pose[:3, 3] - prev_kf_pose[:3, 3]
        )  # Translation difference
        if translation_diff >= 1.0:
            logger.info(
                f"[Detector][Layout] Candidate Frame for layout calculation -- translation: {translation_diff}"
            )
            return True

        # Rotation check
        curr_rotation = R.from_matrix(curr_pose[:3, :3])
        last_rotation = R.from_matrix(prev_kf_pose[:3, :3])
        rotation_diff = curr_rotation.inv() * last_rotation
        angle_diff = rotation_diff.magnitude() * (180 / np.pi)

        if angle_diff >= 20:
            logger.info(
                f"[Detector][Layout] Candidate Frame for layout calculation -- rotation: {angle_diff}"
            )
            return True

        return False

    def process_yolo_results(self, color, obj_classes):

        # Perform YOLO prediction
        results = self.yolo.predict(color, conf=0.2, verbose=False)

        # Extract confidence scores
        confidence_tensor = results[0].boxes.conf
        confidence_np = confidence_tensor.cpu().numpy()

        # Extract class IDs
        detection_class_id_tensor = results[0].boxes.cls
        detection_class_id_np = detection_class_id_tensor.cpu().numpy().astype(int)

        # Generate class labels
        detection_class_labels = [
            f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
            for class_idx, class_id in enumerate(detection_class_id_np)
        ]

        # Extract bounding box coordinates
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

        return confidence_np, detection_class_id_np, detection_class_labels, xyxy_np

    def process_fastsam_results(self, color):
        results = self.fastsam(
            color,
            device="cuda",
            retina_masks=True,
            imgsz=1024,
            conf=self.cfg.fastsam_confidence,
            iou=0.9,
            verbose=False,
        )
        # Extract confidence scores
        confidence_tensor = results[0].boxes.conf
        confidence_np = confidence_tensor.cpu().numpy()

        # Extract bounding box coordinates
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

        # Extract Masks with protection against None
        if results[0].masks is not None:
            masks_tensor = results[0].masks.data
            masks_np = masks_tensor.cpu().numpy().astype(bool)
        else:
            logging.warning(
                "[Detector] fastSAM did not return any masks, using empty mask array"
            )
            # If no mask is returned, create an empty array. Assume mask size matches input image's first two dims
            masks_np = np.empty((0,) + color.shape[:2], dtype=bool)

        # Extract class IDs (default all set to unknown_class_id)
        detection_class_id_tensor = results[0].boxes.cls
        detection_class_id_np = detection_class_id_tensor.cpu().numpy().astype(int)
        detection_class_id_np = np.full_like(
            detection_class_id_np, self.unknown_class_id
        )

        return confidence_np, detection_class_id_np, xyxy_np, masks_np

    def merge_detections(self, detections1, detections2):
        # Check if first detections is empty
        if len(detections1.xyxy) == 0:
            return detections2

        # Check if second detections is empty
        if len(detections2.xyxy) == 0:
            return detections1

        # Merge xyxy
        merged_xyxy = np.concatenate([detections1.xyxy, detections2.xyxy], axis=0)

        # Merge confidence
        merged_confidence = np.concatenate(
            [detections1.confidence, detections2.confidence], axis=0
        )

        # Merge class_id
        merged_class_id = np.concatenate(
            [detections1.class_id, detections2.class_id], axis=0
        )

        # Merge mask
        merged_masks = np.concatenate([detections1.mask, detections2.mask], axis=0)

        # Create new sv.Detections object
        merged_detections = sv.Detections(
            xyxy=merged_xyxy,
            confidence=merged_confidence,
            class_id=merged_class_id,
            mask=merged_masks,
        )

        return merged_detections

    def process_fastsam(self, color):

        with timing_context("FastSAM", self):
            fs_confidence_np, fs_class_id_np, fs_xyxy_np, fs_masks_np = (
                self.process_fastsam_results(color)
            )

        if len(fs_confidence_np) == 0:
            logger.warning("[Detector] No detections found in curr frame by FastSAM.")
            self.fastsam_detections = {}
            return

        # debug fastsam
        fs_detections = sv.Detections(
            xyxy=fs_xyxy_np,
            confidence=fs_confidence_np,
            class_id=fs_class_id_np,
            mask=fs_masks_np,
        )

        if self.cfg.visualize_detection and self.cfg.show_fastsam_debug:
            image_fs, _ = visualize_result_rgb(
                color, fs_detections, self.obj_classes.get_classes_arr()
            )

            self.annotated_image_fs = image_fs

        self.fastsam_detections = fs_detections

    def process_yolo_and_sam(self, color):
        with timing_context("YOLO", self):
            confidence, class_id, class_labels, xyxy = self.process_yolo_results(
                color, self.obj_classes
            )

        # if detection is empty, return
        if len(confidence) == 0:
            logger.warning("[Detector] No detections found in curr frame.")
            # set current results as empty dict
            self.curr_results = {}
            return
        with timing_context("Segmentation", self):
            sam_out = self.sam.predict(color, bboxes=xyxy, verbose=False)
            masks_tensor = sam_out[0].masks.data
            masks_np = masks_tensor.cpu().numpy()
            self.masks_np = masks_np

        curr_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            mask=masks_np,
        )

        self.curr_detections = curr_detections

    def filter_fs_detections_by_curr(
        self, fs_detections, curr_detections, iou_threshold=0.5, overlap_threshold=0.6
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert numpy arrays to torch tensors and move to GPU
        fs_masks = torch.tensor(
            fs_detections.mask, dtype=torch.bool, device=device
        )  # (N1, H, W)
        fs_xyxy = torch.tensor(
            fs_detections.xyxy, dtype=torch.float32, device=device
        )  # (N1, 4)
        fs_confidence = torch.tensor(
            fs_detections.confidence, dtype=torch.float32, device=device
        )
        fs_class_id = torch.tensor(
            fs_detections.class_id, dtype=torch.int64, device=device
        )

        curr_masks = torch.tensor(
            curr_detections.mask, dtype=torch.bool, device=device
        )  # (N2, H, W)

        # Get total number of pixels in masks
        num_fs = fs_masks.shape[0]  # N1
        num_curr = curr_masks.shape[0]  # N2

        # Flatten masks to (N, H * W) and convert to float32 for matrix multiplication
        fs_masks_flat = fs_masks.view(num_fs, -1).to(torch.float32)  # (N1, H * W)
        curr_masks_flat = curr_masks.view(num_curr, -1).to(torch.float32)  # (N2, H * W)

        # Compute intersection and union (using float operations)
        intersection = torch.matmul(fs_masks_flat, curr_masks_flat.T)  # (N1, N2)
        fs_area = fs_masks_flat.sum(dim=1, keepdim=True)  # (N1, 1)
        curr_area = curr_masks_flat.sum(dim=1).unsqueeze(0)  # (1, N2)
        union = fs_area + curr_area - intersection  # (N1, N2)

        # Compute IoU
        iou_matrix = intersection / torch.clamp(union, min=1e-7)  # (N1, N2)

        # Compute overlap ratio
        overlap_ratio_fs = intersection / torch.clamp(fs_area, min=1e-7)  # (N1, N2)
        overlap_ratio_curr = intersection / torch.clamp(curr_area, min=1e-7)  # (N1, N2)

        # Initialize keep mask, default is to keep all fs_masks
        keep_mask = torch.ones(num_fs, dtype=torch.bool, device=device)

        # Filter masks one by one
        for i in range(num_fs):
            # Check if current fs_mask overlaps with curr_mask
            overlap = (
                (iou_matrix[i] > iou_threshold)
                | (overlap_ratio_fs[i] > overlap_threshold)
                | (overlap_ratio_curr[i] > overlap_threshold)
            )

            # If overlap exists, mark as not keep
            if overlap.any():
                keep_mask[i] = False

        # Filter detections based on keep mask
        filtered_fs_detections = sv.Detections(
            xyxy=fs_xyxy[keep_mask].cpu().numpy(),
            confidence=fs_confidence[keep_mask].cpu().numpy(),
            class_id=fs_class_id[keep_mask].cpu().numpy(),
            mask=fs_masks[keep_mask].cpu().numpy(),
        )

        return filtered_fs_detections

    def add_extra_detections_from_fastsam(
        self, color, fastsam_detections, incoming_detections
    ):

        with timing_context("mask_filter", self):
            fs_after_detections = self.filter_fs_detections_by_curr(
                fastsam_detections, incoming_detections
            )

        if self.cfg.visualize_detection and self.cfg.show_fastsam_debug:
            image_fs_after, _ = visualize_result_rgb(
                color, fs_after_detections, self.obj_classes.get_classes_arr()
            )
            self.annotated_image_fs_after = image_fs_after

        # merge_detctions
        merged_detctions = self.merge_detections(
            fs_after_detections, incoming_detections
        )
        return merged_detctions

    def process_detections(self):

        color = self.curr_data.color.astype(np.uint8)

        with timing_context("YOLO+Segmentation+FastSAM", self):
            # Run FastSAM
            if self.cfg.use_fastsam:
                fastsam_thread = threading.Thread(
                    target=self.process_fastsam, args=(color,)
                )
                fastsam_thread.start()

            # Run YOLO and SAM
            self.process_yolo_and_sam(color)

            # Waiting for FastSAM to finish
            if self.cfg.use_fastsam:
                fastsam_thread.join()

        with timing_context("Detection Filter", self):
            self.filter.update_detections(self.curr_detections, color)
            filtered_detections = self.filter.run_filter()

        if self.filter.get_len() == 0:
            logger.warning(
                "[Detector] No valid detections in curr frame after filtering."
            )
            self.curr_results = {}
            return

        # add extra detections from FastSAM results
        # if no detection from fastsam, just skip
        if self.cfg.use_fastsam and self.fastsam_detections:
            filtered_detections = self.add_extra_detections_from_fastsam(
                color, self.fastsam_detections, filtered_detections
            )

        with timing_context("CLIP+Create Object Pointcloud", self):
            cluster_thread = threading.Thread(
                target=self.process_masks, args=(filtered_detections.mask,)
            )
            cluster_thread.start()

            with timing_context("CLIP", self):
                image_crops, image_feats, text_feats = (
                    self.compute_clip_features_batched(
                        color,
                        filtered_detections,
                        self.clip_model,
                        self.clip_tokenizer,
                        self.clip_preprocess,
                        self.cfg.device,
                        self.obj_classes.get_classes_arr(),
                    )
                )

            cluster_thread.join()

        results = {
            # SAM Info
            "xyxy": filtered_detections.xyxy,
            "confidence": filtered_detections.confidence,
            "class_id": filtered_detections.class_id,
            "masks": filtered_detections.mask,
            # CLIP info
            "image_feats": image_feats,
            "text_feats": text_feats,
        }

        if self.cfg.visualize_detection:
            with timing_context("Visualize Detection", self):
                annotated_image, _ = visualize_result_rgb(
                    color, filtered_detections, self.obj_classes.get_classes_arr()
                )
                self.annotated_image = annotated_image

        self.curr_results = results

    def process_masks(self, masks):
        """
        Processes the given masks to extract and refine 3D points and colors.

        Args:
            self: The object containing configuration and data attributes.
            masks: A NumPy array of shape (N, H, W), where N is the number of masks.

        Returns:
            refined_points_list: A list of refined 3D points for each mask.
            refined_colors_list: A list of refined colors corresponding to the points for each mask.
        """

        with timing_context("Create Object Pointcloud", self):
            N, _, _ = masks.shape

            # Convert input data to tensors
            depth_tensor = (
                torch.from_numpy(self.curr_data.depth)
                .to(self.cfg.device)
                .float()
                .squeeze()
            )
            masks_tensor = torch.from_numpy(masks).to(self.cfg.device).float()
            intrinsic_tensor = (
                torch.from_numpy(self.curr_data.intrinsics).to(self.cfg.device).float()
            )
            image_rgb_tensor = (
                torch.from_numpy(self.curr_data.color).to(self.cfg.device).float()
                / 255.0
            )

            # Generate 3D points and colors for the masks
            points_tensor, colors_tensor = mask_depth_to_points(
                depth_tensor,
                image_rgb_tensor,
                intrinsic_tensor,
                masks_tensor,
                self.cfg.device,
            )

            refined_points_list = []
            refined_colors_list = []

            # Process each mask
            for i in range(N):
                mask_points = points_tensor[i]
                mask_colors = colors_tensor[i]

                # Filter valid points based on Z-axis > 0
                valid_points_mask = mask_points[:, :, 2] > 0

                if torch.sum(valid_points_mask) < self.cfg.min_points_threshold:
                    refined_points_list.append(None)
                    refined_colors_list.append(None)
                    continue

                valid_points = mask_points[valid_points_mask]
                valid_colors = mask_colors[valid_points_mask]

                # Random sampling based on sample ratio
                sample_ratio = self.cfg.pcd_sample_ratio
                num_points = valid_points.shape[0]

                if sample_ratio < 1.0:
                    sample_count = int(num_points * sample_ratio)
                    sample_indices = torch.randperm(num_points)[:sample_count]
                    downsampled_points = valid_points[sample_indices]
                    downsampled_colors = valid_colors[sample_indices]
                else:
                    downsampled_points = valid_points
                    downsampled_colors = valid_colors

                # Refine points using clustering
                refined_points, refined_colors = refine_points_with_clustering(
                    downsampled_points,
                    downsampled_colors,
                    eps=self.cfg.dbscan_eps,
                    min_points=self.cfg.dbscan_min_points,
                )

                refined_points_list.append(refined_points)
                refined_colors_list.append(refined_colors)

        self.masked_points = refined_points_list
        self.masked_colors = refined_colors_list

    def compute_max_cos_sim(self, image_feats, class_feats):
        """
        Compute the cosine similarity between image_feats and class_feats, and return the class index with the maximum similarity for each image_feat.

        Args:
            image_feats (np.ndarray): CLIP features of all current images, shape (N, 512).
            class_feats (np.ndarray): CLIP features of classes, shape (C, 512).

        Returns:
            max_indices (np.ndarray): The class index with the maximum cosine similarity for each image_feat, shape (N,).
        """
        # Normalize the features to compute cosine similarity
        image_feats_norm = image_feats / np.linalg.norm(
            image_feats, axis=1, keepdims=True
        )  # Normalize image_feats
        class_feats_norm = class_feats / np.linalg.norm(
            class_feats, axis=1, keepdims=True
        )  # Normalize class_feats

        # Compute cosine similarity: (N, 512) @ (512, C) -> (N, C)
        cos_sim = np.dot(image_feats_norm, class_feats_norm.T)

        # Find the index of the maximum similarity for each image
        max_indices = np.argmax(cos_sim, axis=1)  # shape (N,)

        return max_indices

    def depth_to_point_cloud(self, sample_rate=1) -> o3d.geometry.PointCloud:
        """
        Convert depth image to a point cloud and transform it to world coordinates.

        Parameters:
        - sample_rate: The downsampling rate for pixel selection. (1 means all pixels, 2 means every other pixel)

        Returns:
        - point_cloud: The point cloud in world coordinates as an Open3D PointCloud object.
        """
        # Extract necessary data from curr_data
        depth = self.curr_data.depth.squeeze(
            -1
        )  # Remove the last dimension if depth is (H, W, 1)
        intrinsics = self.curr_data.intrinsics
        pose = self.curr_data.pose

        # Mask out invalid depth values (e.g., depth = 0 or NaN)
        valid_mask = (depth > 0) & (
            depth != np.inf
        )  # Create a mask for valid depth values
        depth = depth[valid_mask]  # Only keep valid depth values

        # Get the corresponding u, v coordinates for valid pixels
        height, width = self.curr_data.depth.shape[:2]
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u[valid_mask]
        v = v[valid_mask]

        # Downsample the points if needed (sampling every `sample_rate` pixels)
        u = u[::sample_rate]
        v = v[::sample_rate]
        depth = depth[::sample_rate]

        # Use the intrinsic matrix to convert from pixel coordinates to camera coordinates (X, Y, Z)
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Convert from pixel coordinates to normalized camera coordinates
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        # Stack the coordinates to form the point cloud in the camera coordinate system
        points_camera = np.vstack((x, y, z)).T

        # Convert points to homogeneous coordinates (4D) for transformation
        points_homogeneous = np.hstack(
            (points_camera, np.ones((points_camera.shape[0], 1)))
        )

        # Apply the pose transformation to move points to world coordinates
        points_world_homogeneous = (pose @ points_homogeneous.T).T

        # Discard the homogeneous coordinate (last column) to get the final 3D points in world coordinates
        points_world = points_world_homogeneous[:, :3]

        # Create a PointCloud object and set its points
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_world)

        return point_cloud

    def save_detection_results(
        self,
    ) -> None:

        if self.curr_results == {}:
            logger.error("[Detector] No detection, Nothing to save")
            return

        output_det_path = self.detection_path / self.curr_data.color_name
        output_det_path.mkdir(exist_ok=True, parents=True)

        # save results
        for key, value in self.curr_results.items():
            save_path = Path(output_det_path) / f"{key}"
            if isinstance(value, np.ndarray):
                # Save NumPy arrays using .npz for efficient storage
                np.savez_compressed(f"{save_path}.npz", value)
            else:
                # For other types, fall back to pickle
                with gzip.open(f"{save_path}.pkl.gz", "wb") as f:
                    pickle.dump(value, f)

        # save annotated images
        output_file_path = (
            self.detection_path / "vis" / (self.curr_data.color_name + "_annotated.jpg")
        )
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.cvtColor(self.curr_data.color, cv2.COLOR_BGR2RGB)

        detections = sv.Detections(
            xyxy=self.curr_results["xyxy"],
            confidence=self.curr_results["confidence"],
            class_id=self.curr_results["class_id"],
            mask=self.curr_results["masks"],
        )
        annotated_image, _ = visualize_result_rgb(
            image, detections, self.obj_classes.get_classes_arr()
        )

        self.annotated_image = annotated_image

        cv2.imwrite(str(output_file_path), annotated_image)

    def load_detection_results(
        self,
    ):
        det_path = self.detection_path / self.curr_data.color_name

        det_path = Path(det_path)

        # if current frame has no detection in disk, return with empty dict
        if not det_path.exists():
            logger.error(f"[Detector] No detection results found in {det_path}")
            self.curr_results = {}
            return

        # Load results from disk
        logger.info(f"[Detector] Loading detection results from {det_path}")

        loaded_detections = {}

        for file_path in det_path.iterdir():
            # handle the files with their extensions
            if file_path.suffix == ".gz" and file_path.suffixes[-2] == ".pkl":
                key = file_path.name.replace(".pkl.gz", "")
                with gzip.open(file_path, "rb") as f:
                    loaded_detections[key] = pickle.load(f)
            elif file_path.suffix == ".npz":
                loaded_detections[file_path.stem] = np.load(file_path)["arr_0"]
            elif file_path.suffix == ".jpg":
                continue
            else:
                raise ValueError(f"{file_path} is not a .pkl.gz or .npz file!")

        self.curr_results = loaded_detections

    def calculate_observations(
        self,
    ) -> None:
        # if no detection, just return
        if self.curr_results == {}:
            logger.warning("[Detector] No detection, Nothing to calculate observations")
            self.curr_observations = []
            return

        # Traverse all the detections
        N, _, _ = self.curr_results["masks"].shape

        # for debugging only
        # bbox_hl_mapping = []
        for i in range(N):

            if self.masked_points[i] is None:
                continue

            # Create pointcloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.masked_points[i])
            pcd.colors = o3d.utility.Vector3dVector(self.masked_colors[i])
            pcd.transform(self.curr_data.pose)

            # Get bbox
            bbox = safe_create_bbox(pcd)
            # bbox = pcd.get_axis_aligned_bounding_box()

            if self.cfg.filter_ceiling:
                z = bbox.get_center()[2]

                # check z is close to ceiling_height
                if abs(z - self.cfg.ceiling_height) < self.cfg.ceiling_threshold:  # 0.1
                    continue  # If z close ceiling_height， skip this observation

            # Get Mobility
            # class_name = self.obj_classes.get_classes_arr()[self.curr_results['class_id'][i]]
            # Get distance
            distance = self.get_distance(bbox, self.curr_data.pose)

            # Init observation
            curr_obs = LocalObservation()

            # Set observation info
            curr_obs.idx = self.curr_data.idx
            curr_obs.class_id = self.curr_results["class_id"][i]
            curr_obs.mask = self.curr_results["masks"][i]

            curr_obs.xyxy = self.curr_results["xyxy"][i]
            curr_obs.conf = self.curr_results["confidence"][i]

            if self.cfg.use_weighted_feature:
                curr_obs.clip_ft = self.get_weighted_feature(idx=i)
            else:
                curr_obs.clip_ft = self.curr_results["image_feats"][i]

            curr_obs.pcd = pcd
            curr_obs.bbox = bbox
            curr_obs.distance = distance

            # judge if low mobility according to the clip feature
            curr_obs.is_low_mobility = self.is_low_mobility(
                curr_obs.clip_ft
            )  # , hl_debug, hl_idx
            # for debugging only
            # bbox_hl_mapping.append([self.curr_results['xyxy'][i], hl_debug, hl_idx])

            # if curr_obs classid is desk set as low mobility
            if (
                self.obj_classes.get_classes_arr()[curr_obs.class_id]
                in self.cfg.lm_examples
            ):
                curr_obs.is_low_mobility = True

            if self.cfg.save_cropped:
                whole_image = self.curr_data.color

                # crop image by xyxy
                x1, y1, x2, y2 = map(int, curr_obs.xyxy)
                cropped_image = whole_image[y1:y2, x1:x2]
                cropped_mask = curr_obs.mask[y1:y2, x1:x2].astype(np.uint8) * 255

                masked_image = cv2.bitwise_and(
                    cropped_image, cropped_image, mask=cropped_mask
                )

                curr_obs.masked_image = masked_image
                curr_obs.cropped_image = cropped_image

            # Add observation to the list
            self.curr_observations.append(curr_obs)

        logger.info(
            f"[Detector] Current observations num: {len(self.curr_observations)}"
        )

    def get_weighted_feature(self, idx):
        image_feat = self.curr_results["image_feats"][idx]
        text_feat = self.curr_results["text_feats"][idx]

        w_image = self.cfg.image_weight
        w_text = 1 - w_image

        weighted_feature = w_image * image_feat + w_text * text_feat

        norm = np.linalg.norm(weighted_feature)
        if norm > 0:
            weighted_feature /= norm

        return weighted_feature

    def visualize_time(
        self,
        elapsed_time,
    ) -> None:
        logger.info(f"[Detector][Visualize] Elapsed time: {elapsed_time:.4f} seconds")
        self.visualizer.log(
            "plot_time/frame_elapsed_time",
            self.visualizer.Scalar(elapsed_time),
            self.visualizer.SeriesLine(width=2.5, color=[255, 0, 0]),  # Red color
        )

    def visualize_memory(
        self,
        memory_usage,
    ) -> None:
        logger.info(f"[Detector][Visualize] Memory usage: {memory_usage:.2f} MB")
        self.visualizer.log(
            "plot_memory/memory_usage",
            self.visualizer.Scalar(memory_usage),
            self.visualizer.SeriesLine(width=2.5, color=[0, 255, 0]),  # Green color
        )

    def visualize_detection(
        self,
    ) -> None:

        if self.annotated_image is not None:
            self.visualizer.log(
                "world/camera/rgb_image_annotated",
                self.visualizer.Image(self.annotated_image),
            )

        if self.cfg.show_local_entities:
            self.visualizer.log(
                "world/camera_raw/rgb_image",
                self.visualizer.Image(self.curr_data.color),
            )

            if self.annotated_image_fs is not None:
                self.visualizer.log(
                    "world/camera_fs/rgb_image_annotated",
                    self.visualizer.Image(self.annotated_image_fs),
                )

            if self.annotated_image_fs_after is not None:
                self.visualizer.log(
                    "world/camera_fs_after/rgb_image_annotated",
                    self.visualizer.Image(self.annotated_image_fs_after),
                )

        # Visualize camera traj
        focal_length = [
            self.curr_data.intrinsics[0, 0].item(),
            self.curr_data.intrinsics[1, 1].item(),
        ]
        principal_point = [
            self.curr_data.intrinsics[0, 2].item(),
            self.curr_data.intrinsics[1, 2].item(),
        ]
        height, width = self.curr_data.color.shape[:2]
        resolution = [width, height]
        self.visualizer.log(
            "world/camera",
            self.visualizer.Pinhole(
                resolution=resolution,
                focal_length=focal_length,
                principal_point=principal_point,
            ),
        )

        translation = self.curr_data.pose[:3, 3].tolist()

        # change the rotation mat to axis-angle
        axis, angle = self.visualizer.rotation_matrix_to_axis_angle(
            self.curr_data.pose[:3, :3]
        )
        self.visualizer.log(
            "world/camera",
            self.visualizer.Transform3D(
                translation=translation,
                rotation=self.visualizer.RotationAxisAngle(axis=axis, angle=angle),
                from_parent=False,
            ),
        )

        # follower camera for recording
        # Visualize camera traj
        f = [
            self.curr_data.intrinsics[0, 0].item(),
            self.curr_data.intrinsics[1, 1].item(),
        ]
        p = [
            self.curr_data.intrinsics[0, 2].item(),
            self.curr_data.intrinsics[1, 2].item(),
        ]
        h, w = self.curr_data.color.shape[:2]
        r = [w, h]
        self.visualizer.log(
            "world/follower_camera",
            self.visualizer.Pinhole(resolution=r, focal_length=f, principal_point=p),
        )
        self.visualizer.log(
            "world/follower_camera_2",
            self.visualizer.Pinhole(resolution=r, focal_length=f, principal_point=p),
        )

        pose_current = self.curr_data.pose
        cam2_to_cam1 = self.create_camera2_to_camera1_transform()
        cam2_to_cam1_2 = self.create_camera2_to_camera1_transform2()
        pose_new = pose_current @ cam2_to_cam1
        pose_new_2 = pose_current @ cam2_to_cam1_2

        translation = pose_new[:3, 3].tolist()
        # change the rotation mat to axis-angle
        axis, angle = self.visualizer.rotation_matrix_to_axis_angle(pose_new[:3, :3])
        self.visualizer.log(
            "world/follower_camera",
            self.visualizer.Transform3D(
                translation=translation,
                rotation=self.visualizer.RotationAxisAngle(axis=axis, angle=angle),
                from_parent=False,
            ),
        )

        # Using for visualization
        pose_smooth = self.pose_filter_follower.update(pose_new_2)
        translation = pose_smooth[:3, 3].tolist()
        axis, angle = self.visualizer.rotation_matrix_to_axis_angle(pose_smooth[:3, :3])

        self.visualizer.log(
            "world/follower_camera_2",
            self.visualizer.Transform3D(
                translation=translation,
                rotation=self.visualizer.RotationAxisAngle(axis=axis, angle=angle),
                from_parent=False,
            ),
        )

        if self.prev_data is not None:
            prev_translation = self.prev_data.pose[:3, 3].tolist()
            prev_quaternion = self.visualizer.rotation_matrix_to_quaternion(
                self.prev_data.pose[:3, :3]
            )

            # # Log a line strip from the previous to the current camera pose
            # self.visualizer.log(
            #     f"world/camera_trajectory/{self.curr_data.idx}",
            #     self.visualizer.LineStrips3D(
            #         [np.vstack([prev_translation, translation]).tolist()],
            #         colors=[[255, 0, 0]]  # Red color for the trajectory line
            #     )
            # )

        if self.cfg.show_debug_entities:
            layout_pointcloud = self.get_layout_pointcloud()
            positions = layout_pointcloud.points
            pcd_entity = "world/layout"
            self.visualizer.log(pcd_entity, self.visualizer.Points3D(positions))

    def create_camera2_to_camera1_transform(self):
        # Define translation vector: translation of camera 2 relative to camera 1
        translation = np.array(self.cfg.follower_translation)  # Up 0.2m, back -0.2m

        # Rotation angles in degrees
        angle_roll = self.cfg.follower_roll  # Rotation around X axis
        angle_pitch = self.cfg.follower_pitch  # Rotation around Y axis
        angle_yaw = self.cfg.follower_yaw  # Rotation around Z axis

        # Convert angles to radians
        angle_roll_rad = np.radians(angle_roll)
        angle_pitch_rad = np.radians(angle_pitch)
        angle_yaw_rad = np.radians(angle_yaw)

        # Rotation matrix around X axis (roll)
        rotation_roll = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_roll_rad), -np.sin(angle_roll_rad)],
                [0, np.sin(angle_roll_rad), np.cos(angle_roll_rad)],
            ]
        )

        # Rotation matrix around Y axis (pitch)
        rotation_pitch = np.array(
            [
                [np.cos(angle_pitch_rad), 0, np.sin(angle_pitch_rad)],
                [0, 1, 0],
                [-np.sin(angle_pitch_rad), 0, np.cos(angle_pitch_rad)],
            ]
        )

        # Rotation matrix around Z axis (yaw)
        rotation_yaw = np.array(
            [
                [np.cos(angle_yaw_rad), -np.sin(angle_yaw_rad), 0],
                [np.sin(angle_yaw_rad), np.cos(angle_yaw_rad), 0],
                [0, 0, 1],
            ]
        )

        # Combined rotation matrix: order is Z, Y, X
        rotation_matrix = rotation_yaw @ rotation_pitch @ rotation_roll

        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix  # Fill rotation matrix
        transform[:3, 3] = translation  # Fill translation vector

        return transform

    def create_camera2_to_camera1_transform2(self):
        # Define translation vector: translation of camera 2 relative to camera 1
        translation = np.array(self.cfg.follower_translation2)  # Up 0.2m, back -0.2m

        # Rotation angles in degrees
        angle_roll = self.cfg.follower_roll2  # Rotation around X axis
        angle_pitch = self.cfg.follower_pitch2  # Rotation around Y axis
        angle_yaw = self.cfg.follower_yaw2  # Rotation around Z axis

        # Convert angles to radians
        angle_roll_rad = np.radians(angle_roll)
        angle_pitch_rad = np.radians(angle_pitch)
        angle_yaw_rad = np.radians(angle_yaw)

        # Rotation matrix around X axis (roll)
        rotation_roll = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_roll_rad), -np.sin(angle_roll_rad)],
                [0, np.sin(angle_roll_rad), np.cos(angle_roll_rad)],
            ]
        )

        # Rotation matrix around Y axis (pitch)
        rotation_pitch = np.array(
            [
                [np.cos(angle_pitch_rad), 0, np.sin(angle_pitch_rad)],
                [0, 1, 0],
                [-np.sin(angle_pitch_rad), 0, np.cos(angle_pitch_rad)],
            ]
        )

        # Rotation matrix around Z axis (yaw)
        rotation_yaw = np.array(
            [
                [np.cos(angle_yaw_rad), -np.sin(angle_yaw_rad), 0],
                [np.sin(angle_yaw_rad), np.cos(angle_yaw_rad), 0],
                [0, 0, 1],
            ]
        )

        # Combined rotation matrix: order is Z, Y, X
        rotation_matrix = rotation_yaw @ rotation_pitch @ rotation_roll

        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix  # Fill rotation matrix
        transform[:3, 3] = translation  # Fill translation vector

        return transform

    def is_low_mobility(self, clip_feat) -> bool:
        # Calculate the cosine similarity between the clip feature and the prototypes
        clip_feat = clip_feat.reshape(1, -1)
        sim = cosine_similarity(clip_feat, self.proto_feats)
        sim = sim.reshape(-1)

        sim_lm = np.max(sim[: self.num_examples[0]])
        sim_hm = np.max(
            sim[self.num_examples[0] : (self.num_examples[0] + self.num_examples[1])]
        )
        sim_lm_des = np.max(sim[(self.num_examples[0] + self.num_examples[1]) :])

        # for debugging only
        # lm_idx = np.argmax(sim[:self.num_examples[0]])
        # hm_idx = np.argmax(sim[self.num_examples[0] : (self.num_examples[0]+self.num_examples[1])])
        # lm_des_idx = np.argmax(sim[(self.num_examples[0]+self.num_examples[1]):])
        # hl_debug = np.array([sim_lm, sim_hm, sim_lm_des])
        # hl_idx = np.array([lm_idx, hm_idx, lm_des_idx])

        # TODO: Magic number: threshold
        if sim_lm > sim_hm + 0.05:
            res = True
        elif sim_lm + 0.05 < sim_hm:
            res = False
        else:
            res = sim_lm_des > 0.45
        return res  # , hl_debug, hl_idx

    def get_distance(self, bbox, pose) -> float:
        # Get the center of the bounding box
        bbox_center = np.array(bbox.get_center())

        # Get the translation part of the pose (assuming it's a 4x4 matrix)
        pose_translation = np.array(pose[:3, 3])  # Extract translation (x, y, z)

        # Calculate the Euclidean distance between the pose translation and the bbox center
        distance = np.linalg.norm(bbox_center - pose_translation)

        return distance

    def compute_clip_features_batched(
        self,
        image,
        detections,
        clip_model,
        clip_tokenizer,
        clip_preprocess,
        device,
        classes,
    ):
        # Convert the image to a PIL Image
        image = Image.fromarray(image)

        # Set the padding for cropping
        padding = 20

        # Initialize lists to store the cropped images and features
        image_crops = []
        image_feats = []
        text_feats = []

        # Initialize lists to store preprocessed images and text tokens for batch processing
        preprocessed_images = []
        text_tokens = []

        # Prepare data for batch processing
        for idx in range(len(detections.xyxy)):
            x_min, y_min, x_max, y_max = detections.xyxy[idx]
            image_width, image_height = image.size

            # Calculate the padding for each side of the bounding box
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)

            # Adjust the bounding box coordinates based on the padding
            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding

            # Crop the image
            cropped_image = image.crop((x_min, y_min, x_max, y_max))

            # Preprocess the cropped image
            preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
            preprocessed_images.append(preprocessed_image)

            # Get the class id for the detection
            class_id = detections.class_id[idx]

            # Append the class name to the text tokens list
            text_tokens.append(classes[class_id])

            # Append the cropped image to the image crops list
            image_crops.append(cropped_image)

        # Convert lists to batches
        preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)
        text_tokens_batch = clip_tokenizer(text_tokens).to(device)

        # Perform batch inference
        with torch.no_grad():
            # Encode the images using the CLIP model
            image_features = clip_model.encode_image(preprocessed_images_batch)

            # Normalize the image features
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Encode the text tokens using the CLIP model
            text_features = clip_model.encode_text(text_tokens_batch)

            # Normalize the text features
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # Convert the image and text features to numpy arrays
        image_feats = image_features.cpu().numpy()
        text_feats = text_features.cpu().numpy()

        if self.cfg.use_avg_feat_for_unknown:
            count = 0
            for idx, class_id in enumerate(detections.class_id):
                if class_id == self.unknown_class_id:
                    count += 1
                    # Modify the text_feats for the unknown class
                    text_feats[idx] = (
                        self.class_feats_mean
                    )  # You can modify how you update the text_feats here

                    # random_feats = np.random.rand(*self.class_feats_mean.shape)
                    # random_feats /=     np.linalg.norm(random_feats)

                    # text_feats[idx] = random_feats

            logger.info(
                f"[Detector] Updated {count} unknown class text features to the mean value."
            )
        else:
            for idx, class_id in enumerate(detections.class_id):
                if class_id == self.unknown_class_id:
                    count += 1
                    # Modify the text_feats for the unknown class
                    # text_feats[idx] = self.class_feats_mean  # You can modify how you update the text_feats here

                    random_feats = np.random.rand(*self.class_feats_mean.shape)
                    random_feats /= np.linalg.norm(random_feats)

                    text_feats[idx] = random_feats

        # Return the cropped images, image features, and text features
        return image_crops, image_feats, text_feats


class Filter:
    def __init__(
        self,
        classes,
        iou_th: float = 0.80,
        proximity_th: float = 0.95,
        keep_larger: bool = True,
        small_mask_size: int = 200,
        skip_refinement: bool = False,
    ):

        self.confidence = None
        self.class_id = None
        self.xyxy = None
        self.masks = None
        self.color = None
        self.masks_size = None
        self.inter_np = None

        self.skip_refinement = skip_refinement
        self.classes = classes
        self.iou_th = iou_th
        self.proximity_th = proximity_th
        self.keep_larger = keep_larger
        self.small_mask_size = small_mask_size

        self.device = "cpu"

    def update_detections(self, detections: sv.Detections, color: np.array):
        with timing_context("update_detections", self):
            self.color = color

            self.confidence = detections.confidence
            self.class_id = detections.class_id
            self.xyxy = detections.xyxy
            self.masks = detections.mask

            self.masks_size = np.sum(self.masks, axis=(1, 2))

            # Compute intersection every time the detections are updated
            N = self.get_len()
            # Convert masks to PyTorch tensors to accelerate computation
            # Compute pairwise intersection using matrix operations
            device = self.device
            masks = torch.tensor(self.masks, dtype=torch.float32).to(device)
            intersection = torch.matmul(masks.view(N, -1), masks.view(N, -1).T)

            self.inter_np = intersection.cpu().numpy()

    def set_device(self, device):
        self.device = device

    def run_filter(self):
        original_num = self.get_len()
        if self.confidence is None or original_num == 0:
            logger.warning("[Detector][Filter] No detections to filter.")
            return

        keep = self.filter_by_mask_size()
        self.set_detections(keep)

        if not self.skip_refinement:
            keep = self.filter_by_iou()
            self.set_detections(keep)

            keep = self.filter_by_proximity()
            self.set_detections(keep)

            self.overlap_check()

        keep = self.filter_by_bg()
        self.set_detections(keep)

        keep = self.filter_by_mask_size()
        self.set_detections(keep)

        if self.get_len() == 0:
            logger.warning(
                "[Detector][Filter] After filtering, no detection result remains..."
            )
            return None
        logger.info(
            f"[Detector][Filter] Filtered {self.get_len()} out of {original_num}"
        )

        # create new detections object and return
        filtered_detections = sv.Detections(
            class_id=np.array(self.class_id, dtype=np.int64),
            confidence=np.array(self.confidence, dtype=np.float32),
            xyxy=np.array(self.xyxy, dtype=np.float32),
            mask=np.array(self.masks, dtype=np.bool_),
        )
        return filtered_detections

    def get_len(self):
        return len(self.confidence)

    def set_detections(self, keep):
        if len(keep) != self.get_len():
            logger.warning(
                "[Detector][Filter] The boolean list should be as long as the detections."
            )
            return

        self.confidence, self.class_id, self.xyxy, self.masks, self.masks_size = (
            self.confidence[keep],
            self.class_id[keep],
            self.xyxy[keep],
            self.masks[keep],
            self.masks_size[keep],
        )
        self.inter_np = self.inter_np[keep][:, keep]

    def filter_by_iou(self):
        N = self.get_len()
        # use a list of boolean to control
        if N == 0:
            return np.array([], dtype=bool)

        masks = self.masks
        masks_size = self.masks_size

        # Compute pairwise IoU matrix using matrix operations for acceleration
        intersection = self.inter_np
        area = masks.reshape(N, -1).sum(axis=1)
        union = area[:, None] + area[None, :] - intersection
        iou_matrix = intersection / union

        # Initialize keep mask
        keep = np.ones(N, dtype=bool)

        # Apply IoU threshold and keep larger/smaller masks
        for i in range(N):
            if not keep[i]:
                continue
            for j in range(i + 1, N):
                if iou_matrix[i, j] > self.iou_th:
                    if ((masks_size[i] > masks_size[j]) and self.keep_larger) or (
                        (masks_size[i] < masks_size[j]) and not self.keep_larger
                    ):
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        logger.info(
            f"[Detector][Filter] Original number of detections: {N}, after mask IoU filter: {np.sum(keep)}"
        )
        return keep

    def filter_by_proximity(self):
        if self.color is None:
            logger.warning("[Detector][Filter] No color image is provided.")
            return
        N = self.get_len()
        if N == 0:
            return np.array([], dtype=bool)
        # check if mask overlaps with each other
        overlap = self.inter_np
        overlap = overlap > 0
        np.fill_diagonal(overlap, False)

        N = self.get_len()
        # Initialize keep mask
        keep = np.ones(N, dtype=bool)
        masks_size = self.masks_size

        # Get all the cropped images first to accelerate computation
        cropped_images = []
        cropped_masks = []
        for i in range(N):
            x1, y1, x2, y2 = map(int, self.xyxy[i])
            cropped_image = self.color[y1:y2, x1:x2]
            cropped_images.append(cropped_image)
            cropped_mask = self.masks[i][y1:y2, x1:x2].astype(bool)
            cropped_masks.append(cropped_mask)

        for i in range(N):
            if not keep[i]:
                continue
            for j in range(i + 1, N):
                # if overlapped, crop the images and check if they have the same distribution
                if overlap[i, j]:
                    from_same_dis = if_same_distribution(
                        cropped_images[i],
                        cropped_images[j],
                        cropped_masks[i],
                        cropped_masks[j],
                        self.proximity_th,
                    )

                    if from_same_dis:
                        class_i = self.classes.get_classes_arr()[self.class_id[i]]
                        class_j = self.classes.get_classes_arr()[self.class_id[j]]
                        if ((masks_size[i] > masks_size[j]) and self.keep_larger) or (
                            (masks_size[i] < masks_size[j]) and not self.keep_larger
                        ):
                            keep[j] = False
                            self.merge_detections(j, i)
                            logger.info(
                                f"[Detector][Filter] Merging {class_j} into {class_i}"
                            )
                        else:
                            keep[i] = False
                            self.merge_detections(i, j)
                            logger.info(
                                f"[Detector][Filter] Merging {class_i} into {class_j}"
                            )

        logger.info(
            f"[Detector][Filter] Original number of detections: {N}, after proximity filter: {np.sum(keep)}"
        )
        return keep

    def overlap_check(self):
        N = self.get_len()
        if N == 0:
            return
        masks_size = self.masks_size

        # check if mask overlaps with each other
        overlap = self.inter_np
        overlap = overlap > 0
        np.fill_diagonal(overlap, False)

        for i in range(N):
            for j in range(i + 1, N):
                if overlap[i, j]:
                    if masks_size[i] > masks_size[j]:
                        self.masks[i] = self.masks[i] & (~self.masks[j])
                        self.xyxy[i] = update_bbox(self.masks[i])
                    else:
                        self.masks[j] = self.masks[j] & (~self.masks[i])
                        self.xyxy[j] = update_bbox(self.masks[j])

    def filter_by_bg(self):
        N = self.get_len()
        keep = np.ones(N, dtype=bool)

        for idx, class_id in enumerate(self.class_id):
            if self.classes.get_classes_arr()[class_id] in self.classes.bg_classes:
                logger.info(
                    f"[Detector][Filter] Removing {self.classes.get_classes_arr()[class_id]} because it is a background class."
                )
                keep[idx] = False
        return keep

    def filter_by_mask_size(self):
        keep = self.masks_size >= self.small_mask_size
        for idx, is_keep in enumerate(keep):
            if not is_keep:
                class_name = self.classes.get_classes_arr()[self.class_id[idx]]
                logger.info(
                    f"[Detector][Filter] Removing {class_name} because the mask size is too small."
                )
        return keep

    def merge_detections(self, det, target):
        # merge det into the target detection
        self.masks[target] = np.logical_or(self.masks[target], self.masks[det])
        x_i1, y_i1, x_i2, y_i2 = map(int, self.xyxy[target])
        x_j1, y_j1, x_j2, y_j2 = map(int, self.xyxy[det])
        y1 = min(y_i1, y_j1)
        y2 = max(y_i2, y_j2)
        x1 = min(x_i1, x_j1)
        x2 = max(x_i2, x_j2)
        self.xyxy[target, :] = x1, y1, x2, y2


def update_bbox(mask):
    y, x = np.nonzero(mask)
    return np.min(x), np.min(y), np.max(x), np.max(y)


def if_same_distribution(img1, img2, mask1, mask2, sim_threshold):
    # Separate the image into three channels
    b1, g1, r1 = cv2.split(img1)
    b2, g2, r2 = cv2.split(img2)

    b1, g1, r1 = b1[mask1], g1[mask1], r1[mask1]
    b2, g2, r2 = b2[mask2], g2[mask2], r2[mask2]

    # Compute histograms for each channel
    num_batches = 16
    hist_b1, _ = np.histogram(b1, bins=num_batches, range=(0, 256))
    hist_g1, _ = np.histogram(g1, bins=num_batches, range=(0, 256))
    hist_r1, _ = np.histogram(r1, bins=num_batches, range=(0, 256))
    hist_b2, _ = np.histogram(b2, bins=num_batches, range=(0, 256))
    hist_g2, _ = np.histogram(g2, bins=num_batches, range=(0, 256))
    hist_r2, _ = np.histogram(r2, bins=num_batches, range=(0, 256))

    # Normalize histograms
    hist_b1 = hist_b1 / np.linalg.norm(hist_b1)
    hist_g1 = hist_g1 / np.linalg.norm(hist_g1)
    hist_r1 = hist_r1 / np.linalg.norm(hist_r1)
    hist_b2 = hist_b2 / np.linalg.norm(hist_b2)
    hist_g2 = hist_g2 / np.linalg.norm(hist_g2)
    hist_r2 = hist_r2 / np.linalg.norm(hist_r2)

    # Concatenate histograms
    hist1 = np.concatenate([hist_b1, hist_g1, hist_r1])
    hist2 = np.concatenate([hist_b2, hist_g2, hist_r2])

    # Compute cosine similarity
    cos_sim = cosine_similarity([hist1], [hist2])[0][0]

    return cos_sim > sim_threshold


def get_text_features(
    class_names: list, clip_model, clip_tokenizer, device, clip_length, batch_size=64
) -> np.ndarray:

    multiple_templates = [
        "{}",
        "There is the {} in the scene.",
    ]

    # Get all the prompted sequences
    class_name_prompts = [
        x.format(lm) for lm in class_names for x in multiple_templates
    ]

    # Get tokens
    text_tokens = clip_tokenizer(class_name_prompts).to(device)
    # Get Output features
    text_feats = np.zeros((len(class_name_prompts), clip_length), dtype=np.float32)
    # Get the text feature batch by batch
    text_id = 0
    while text_id < len(class_name_prompts):
        # Get batch size
        batch_size = min(len(class_name_prompts) - text_id, batch_size)
        # Get text prompts based on batch size
        text_batch = text_tokens[text_id : text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()

        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        # move the calculated batch into the Ouput features
        text_feats[text_id : text_id + batch_size, :] = batch_feats
        # Move on and Move on
        text_id += batch_size

    # shrink the output text features into classes names size
    text_feats = text_feats.reshape((-1, len(multiple_templates), text_feats.shape[-1]))
    text_feats = np.mean(text_feats, axis=1)

    # TODO: Should we do normalization? Answer should be YES
    norms = np.linalg.norm(text_feats, axis=1, keepdims=True)
    text_feats /= norms

    return text_feats


def save_hilow_debug(bbox_hl_mapping, output_image, frame_idx):
    for item in bbox_hl_mapping:
        bbox = item[0]
        hl_debug = item[1]
        hl_idx = item[2]
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{hl_debug[0]:.3f}_{hl_idx[0]}, {hl_debug[1]:.3f}_{hl_idx[1]}, {hl_debug[2]:.3f}_{hl_idx[2]}"
        cv2.putText(
            output_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    output_image_path = f"./debug/{frame_idx}_bbox_hl_mapping.jpg"
    cv2.imwrite(output_image_path, output_image)
