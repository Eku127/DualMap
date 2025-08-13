import glob
import logging
import os
from pathlib import Path
from typing import Optional
from scipy.spatial.transform import Rotation as R

import cv2
import imageio
import numpy as np
import torch
import yaml
from kornia.geometry.linalg import compose_transformations, inverse_transformation
from natsort import natsorted
from omegaconf import DictConfig, OmegaConf

# Set up the module-level logger
logger = logging.getLogger(__name__)

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_dict,
        use_stride: bool = True,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        device="cuda:0",
        dtype=torch.float,
        relative_pose: bool = True,  # If True, the pose is relative to the first frame
        **kwargs,
    ):
        super().__init__()
        # Load self config from config_dict
        self.name = config_dict["dataset_name"]
        self.device = device

        self.png_depth_scale = config_dict["camera_params"]["png_depth_scale"]

        self.orig_height = config_dict["camera_params"]["image_height"]
        self.orig_width = config_dict["camera_params"]["image_width"]
        self.fx = config_dict["camera_params"].get("fx", None)
        self.fy = config_dict["camera_params"].get("fy", None)
        self.cx = config_dict["camera_params"].get("cx", None)
        self.cy = config_dict["camera_params"].get("cy", None)

        self.desired_height = desired_height
        self.desired_width = desired_width
        self.dtype = dtype

        self.h_downsample_ratio = float(
            self.desired_height) / float(self.orig_height)
        self.w_downsample_ratio = float(
            self.desired_width) / float(self.orig_width)
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.stride = stride
        self.start = start
        self.end = end

        self.relative_pose = relative_pose

        self.distortion = (
            np.array(config_dict["camera_params"]["distortion"]
                     ) if "distortion" in config_dict else None
        )

        # file paths
        # paths is a list of paths to color and depth images
        self.color_paths, self.depth_paths = self.get_filepaths()

        self.num_imgs = len(self.color_paths)

        # time_stamps is a list of time stamp for each image
        # poses is a list of poses for each image
        self.time_stamps = []
        self.poses = self.get_poses()

        if self.end == -1:
            self.end = self.num_imgs

        # upsample the paths by the stride
        logger.info(f"[Dataset] Number of images: {self.num_imgs}")

        if use_stride:
            logger.info("[Dataset] Use stride")
            self.color_paths = self.color_paths[self.start:self.end:self.stride]
            self.depth_paths = self.depth_paths[self.start:self.end:self.stride]
            self.poses = self.poses[self.start:self.end:self.stride]

        # update number of images
        self.num_imgs = len(self.color_paths)
        logger.info(f"[Dataset] Number of images: {self.num_imgs}")

        # process poses
        # stack poses in a single tensor
        self.poses = torch.stack(self.poses)
        if self.relative_pose:
            self.transformed_poses = self._preprocess_poses(self.poses)
        else:
            self.transformed_poses = self.poses

        logger.info("[Dataset] Dataset loading complete!")

    def get_filepaths(self):
        """Return paths to color images, depth images. Implement in subclass."""
        raise NotImplementedError

    def get_poses(self):
        """Return poses. Implement in subclass."""
        raise NotImplementedError

    def _preprocess_image(self, image):
        """
        Preprocess the input image according to the dataset configuration.

        Parameters:
        image (numpy.ndarray): The input image to preprocess.

        Returns:
        numpy.ndarray: The preprocessed image.
        """
        color = cv2.resize(image, (self.desired_width,
                           self.desired_height), interpolation=cv2.INTER_LINEAR)
        if self.normalize_color:
            color = color / 255.0
        if self.channels_first:
            color = color.transpose((2, 0, 1))
        return color

    def _preprocess_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """
        Transforms all the poses from camera to world coordinate system to camera to camera0 coordinate system.

        Parameters:
        poses (torch.Tensor): A tensor of poses in the shape (N, 4, 4), where N is the number of poses.
            The poses are assumed to be in the camera to world coordinate system.

        Returns:
        torch.Tensor: A tensor of transformed poses in the shape (N, 4, 4).
            The poses are in the camera to camera0 coordinate system.
        """
        # transform all the poses from c2w to c2c0
        return self.relative_trans(
            # augment the first element as the anchor pose
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal=True
        )

    def _preprocess_depth(self, depth):
        depth = cv2.resize(depth, (self.desired_width,
                           self.desired_height), interpolation=cv2.INTER_NEAREST)
        depth = np.expand_dims(depth, -1)
        # TODO: test the shape
        if self.channels_first:
            depth = depth.transpose((2, 0, 1))
        return depth / self.png_depth_scale

    def __len__(self):
        """Return the number of images."""
        return self.num_imgs

    def __getitem__(self, idx):
        # get color path
        color_path = self.color_paths[idx]
        depth_path = self.depth_paths[idx]
        # load color and depth images
        color = np.asarray(imageio.imread(str(color_path)), dtype=np.float32)
        if ".png" in depth_path:
            depth = np.asarray(imageio.imread(str(depth_path)), dtype=np.int64)
        else:
            raise ValueError(f"Depth path {depth_path} is not a png file")

        # preprocess color and depth images
        color = self._preprocess_image(color)
        depth = self._preprocess_depth(depth)

        # transfer numpy arrays to torch tensors
        color = torch.from_numpy(color).to(self.device)
        depth = torch.from_numpy(depth).to(self.device)

        # TODO: intrinsic params is FIXED, no need to load every time
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        # K mat from numpy to torch
        K = torch.from_numpy(K).to(self.device)

        # # scale the K with the downsample ratio
        # K[:3, :3] *= self.w_downsample_ratio
        # K[:3, 3] *= self.w_downsample_ratio
        # K[3, :3] *= self.h_downsample_ratio
        # K[3, 3] *= self.h_downsample_ratio

        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        # Transfrom poses to the relative pose
        pose = self.transformed_poses[idx]

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype)
        )

    def get_color(self, idx):

        # get color path
        color_path = self.color_paths[idx]

        # load color and depth images
        color = np.asarray(imageio.imread(str(color_path)), dtype=np.uint8)

        # preprocess color and depth images
        color = self._preprocess_image(color)

        color_name = os.path.splitext(os.path.basename(color_path))[0]

        return (
            color,
            color_name
        )

    def get_depth(self, idx):

        # get depth path
        depth_path = self.depth_paths[idx]

        # load color and depth images
        depth = np.asarray(imageio.imread(str(depth_path)), dtype=np.int64)

        # preprocess color and depth images
        depth = self._preprocess_depth(depth)

        return (
            depth
        )

    def get_intrinsics(self, idx):
        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        return K

    def relative_trans(
        self, trans_ab: torch.Tensor, trans_cb: torch.Tensor, orthogonal: bool = False
    ) -> torch.Tensor:
        # judge the input
        if not torch.is_tensor(trans_ab):
            raise TypeError("trans_ab must be a torch.Tensor")
        if not torch.is_tensor(trans_cb):
            raise TypeError("trans_cb must be a torch.Tensor")
        if not trans_ab.dim() == trans_cb.dim():
            raise ValueError("trans_ab and trans_cb must have the same dimension")

        # calculate trans_ba
        trans_ba = (
            inverse_transformation(trans_ab) if orthogonal else torch.inverse(trans_ab)
        )

        # calculate trans_ca
        trans_ca = (
            compose_transformations(trans_ba, trans_cb)
            if orthogonal
            else torch.matmul(trans_ba, trans_cb)
        )

        return trans_ca


class ReplicaDataset(BaseDataset):
    def __init__(
        self,
        config_dict,
        based_dir,
        scene_id,
        use_stride: bool = True,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        **kwargs,
    ):
        # remove "_" in scene id
        scene_id = scene_id.replace("_", "")
        
        self.input_path = os.path.join(based_dir, scene_id)
        self.pose_path = os.path.join(self.input_path, "traj.txt")
        super().__init__(
            config_dict,
            use_stride = use_stride,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(
            glob.glob(f"{self.input_path}/results/frame*.jpg"))
        depth_paths = natsorted(
            glob.glob(f"{self.input_path}/results/depth*.png"))
        
        logger.info(f"[Dataset] Number of color images: {len(color_paths)}")
        logger.info(f"[Dataset] Number of depth images: {len(depth_paths)}")
        
        return color_paths, depth_paths

    def get_poses(self):
        poses = []

        lines = []

        with open(self.pose_path, "r") as f:
            lines = f.readlines()

        for i in range(self.num_imgs):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # np -> torch
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)

        logger.info(f"[Dataset] Number of pose: {len(poses)}")

        # Set self.time_stamps to zeros
        self.time_stamps = [0.0] * self.num_imgs
        
        return poses

class ScanNetDataset(BaseDataset):
    def __init__(
        self,
        config_dict,
        based_dir,
        scene_id,
        use_stride: bool = True,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        **kwargs,
    ):
        # remove "_" in scene id
        
        self.input_path = os.path.join(based_dir, "exported", scene_id)
        self.pose_dir = os.path.join(self.input_path, "pose")
        super().__init__(
            config_dict,
            use_stride=use_stride,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            **kwargs,
        )
        intrinsics_path = os.path.join(self.input_path, "intrinsic", "intrinsic_depth.txt")
        intrinsics = np.loadtxt(intrinsics_path)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.cx = intrinsics[0, 2]
        self.cy = intrinsics[1, 2]

    def get_filepaths(self):
        color_paths = natsorted(
            glob.glob(f"{self.input_path}/color/*.jpg"))
        depth_paths = natsorted(
            glob.glob(f"{self.input_path}/depth/*.png"))
        
        logger.info(f"[Dataset] Number of color images: {len(color_paths)}")
        logger.info(f"[Dataset] Number of depth images: {len(depth_paths)}")
        
        return color_paths, depth_paths

    def get_poses(self):
        poses = []
        pose_path_list = natsorted(glob.glob(f"{self.pose_dir}/*.txt"))

        for pose_path in pose_path_list:
            pose = np.loadtxt(pose_path).reshape(4, 4)
            c2w = torch.from_numpy(pose).float()
            poses.append(c2w)
        
        logger.info(f"[Dataset] Number of pose: {len(poses)}")

        # Set self.time_stamps to zeros
        self.time_stamps = [0.0] * self.num_imgs
        
        return poses

class SelfCollectedDataset(BaseDataset):
    def __init__(
        self,
        config_dict,
        based_dir,
        scene_id,
        use_stride: bool = True,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        **kwargs,
    ):
        # remove "_" in scene id
        # scene_id = scene_id.replace("_", "")
        
        self.input_path = os.path.join(based_dir, scene_id)
        self.pose_path = os.path.join(self.input_path, "pose.txt")

        # debug: if exist, logger.info
        if os.path.exists(self.pose_path):
            logger.info(f"[Dataset] Pose file exists: {self.pose_path}")
        else:
            logger.info(f"[Dataset] Pose file does not exist: {self.pose_path}")

        super().__init__(
            config_dict,
            use_stride = use_stride,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(
            glob.glob(f"{self.input_path}/rgb/*.png"))
        depth_paths = natsorted(
            glob.glob(f"{self.input_path}/depth/*.png"))
        
        logger.info(f"[Dataset] Number of color images: {len(color_paths)}")
        logger.info(f"[Dataset] Number of depth images: {len(depth_paths)}")
        logger.info(color_paths[0])
        logger.info(depth_paths[0])
        
        return color_paths, depth_paths
    
    def get_poses(self):
        poses = []
        time_stamps = []

        lines = []

        with open(self.pose_path, "r") as f:
            lines = f.readlines()

        for i in range(self.num_imgs):
            line = lines[i]

            parts = line.strip().split()
            timestamp = float(parts[0])
            time_stamps.append(timestamp)

            pose = np.array([float(x) for x in parts[1:]])

            transformation_matrix = pose.reshape(4, 4)

            # np -> torch
            c2w = torch.from_numpy(transformation_matrix).float()
            poses.append(c2w)

        self.time_stamps = time_stamps

        logger.info(f"[Dataset] Number of pose: {len(poses)}")
        
        return poses

class TUMRGBDDataset(BaseDataset):
    def __init__(
        self,
        config_dict,
        based_dir,
        scene_id,
        use_stride: bool = True,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: int = 480,
        desired_width: int = 640,
        **kwargs,
    ):
        # remove "_" in scene id
        # scene_id = scene_id.replace("_", "")
        
        self.input_path = os.path.join(based_dir, scene_id)
        self.pose_path = os.path.join(self.input_path, "groundtruth.txt")

        # debug: if exist, logger.info
        if os.path.exists(self.pose_path):
            logger.info(f"[Dataset] Pose file exists: {self.pose_path}")
        else:
            logger.info(f"[Dataset] Pose file does not exist: {self.pose_path}")

        super().__init__(
            config_dict,
            use_stride = use_stride,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            **kwargs,
        )

    def extract_timestamp(self, path):
        basename = os.path.basename(path)
        timestamp_str = os.path.splitext(basename)[0]
        return float(timestamp_str)

    
    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_path}/rgb/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_path}/depth/*.png"))

        color_timestamps = [self.extract_timestamp(p) for p in color_paths]
        depth_timestamps = [self.extract_timestamp(p) for p in depth_paths]

        logger.info(f"[Dataset] Number of color images: {len(color_paths)}")
        logger.info(f"[Dataset] Number of depth images: {len(depth_paths)}")

        if len(color_paths) <= len(depth_paths):
            base_paths, base_ts = color_paths, color_timestamps
            target_paths, target_ts = depth_paths, depth_timestamps
            base_is_color = True
        else:
            base_paths, base_ts = depth_paths, depth_timestamps
            target_paths, target_ts = color_paths, color_timestamps
            base_is_color = False

        target_ts_sorted = sorted((ts, i) for i, ts in enumerate(target_ts))
        sorted_ts = [ts for ts, _ in target_ts_sorted]
        sorted_idx = [i for _, i in target_ts_sorted]

        used_target_idx = set()
        matched_base_paths = []
        matched_target_paths = []

        import bisect
        for i, ts in enumerate(base_ts):
            pos = bisect.bisect_left(sorted_ts, ts)

            candidates = []
            if pos < len(sorted_ts):
                candidates.append(pos)
            if pos > 0:
                candidates.append(pos - 1)

            best = None
            min_diff = float('inf')

            for c in candidates:
                idx = sorted_idx[c]
                if idx in used_target_idx:
                    continue
                diff = abs(sorted_ts[c] - ts)
                if diff < min_diff:
                    min_diff = diff
                    best = idx

            if best is not None:
                used_target_idx.add(best)
                matched_base_paths.append(base_paths[i])
                matched_target_paths.append(target_paths[best])

        if base_is_color:
            final_color_paths = matched_base_paths
            final_depth_paths = matched_target_paths
        else:
            final_color_paths = matched_target_paths
            final_depth_paths = matched_base_paths

        logger.info(f"[Dataset] First aligned color image: {final_color_paths[0]}")
        logger.info(f"[Dataset] First aligned depth image: {final_depth_paths[0]}")
        logger.info(f"[Dataset] Final aligned length: {len(final_color_paths)}")

        self.target_timestamps = [self.extract_timestamp(p) for p in final_depth_paths]

        return final_color_paths, final_depth_paths
    
    def get_poses(self):
        poses_all = []
        time_stamps_all = []

        with open(self.pose_path, "r") as f:
            lines = [line for line in f.readlines() if not line.startswith("#")]

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 8:
                continue

            timestamp = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:])

            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = [tx, ty, tz]



            c2w = torch.from_numpy(T).float()
            poses_all.append(c2w)
            time_stamps_all.append(timestamp)

        # -------- ----------
        target_ts = self.target_timestamps
        time_stamps_sorted = sorted((ts, i) for i, ts in enumerate(time_stamps_all))
        sorted_ts = [ts for ts, _ in time_stamps_sorted]
        sorted_idx = [i for _, i in time_stamps_sorted]

        used_pose_idx = set()
        aligned_poses = []
        not_found_ts = []

        import bisect
        for target in target_ts:
            pos = bisect.bisect_left(sorted_ts, target)

            candidates = []
            if pos < len(sorted_ts):
                candidates.append(pos)
            if pos > 0:
                candidates.append(pos - 1)

            best = None
            min_diff = float('inf')
            for c in candidates:
                idx = sorted_idx[c]
                if idx in used_pose_idx:
                    continue
                diff = abs(sorted_ts[c] - target)
                if diff < min_diff:
                    min_diff = diff
                    best = idx

            if best is not None:
                used_pose_idx.add(best)
                aligned_poses.append(poses_all[best])
            else:
                logger.warning(f"No matching pose found for timestamp {target}")
                not_found_ts.append(target)

        # ---------- ----------
        if not_found_ts:
            logger.warning(f"Removing {len(not_found_ts)} unmatched timestamps from color/depth/target lists.")

            bad_indices = [i for i, t in enumerate(self.target_timestamps) if t in not_found_ts]

            def remove_indices_from_list(lst, indices):
                return [item for i, item in enumerate(lst) if i not in indices]

            self.target_timestamps = remove_indices_from_list(self.target_timestamps, bad_indices)
            self.color_paths = remove_indices_from_list(self.color_paths, bad_indices)
            self.depth_paths = remove_indices_from_list(self.depth_paths, bad_indices)

        logger.info(f"[Dataset] Aligned poses: {len(aligned_poses)}")
        self.time_stamps = self.target_timestamps
        # import pdb; pdb.set_trace()
        return aligned_poses


def load_dataset_config(path, default_path=None):
    """
    Loads config file.

    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.

    Returns:
        cfg (dict): config dict.

    """
    # load configuration from file itself
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get("inherit_from")

    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_dataset_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def get_dataset(dataconfig, basedir, scene_id, **kwargs):
    config_dict = load_dataset_config(dataconfig)
    if config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, scene_id, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScanNetDataset(config_dict, basedir, scene_id, **kwargs)
    elif config_dict["dataset_name"].lower() in ["self_collected"]:
        return SelfCollectedDataset(config_dict, basedir, scene_id, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum_rgbd"]:
        return TUMRGBDDataset(config_dict, basedir, scene_id, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")

def dataset_initialization(
    cfg: DictConfig
) -> torch.utils.data.Dataset:
    
    cfg.dataset_path = Path(cfg.dataset_path)
    cfg.dataset_conf_path = Path(cfg.dataset_conf_path)
    dataset_cfg = OmegaConf.load(cfg.dataset_conf_path)
    cfg.image_height = dataset_cfg.camera_params.image_height
    cfg.image_width = dataset_cfg.camera_params.image_width
    
    dataset = get_dataset(
        dataconfig=cfg.dataset_conf_path,
        basedir=cfg.dataset_path,
        scene_id=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        use_stride = cfg.use_stride,
        stride=cfg.stride,
        start=cfg.start,
        end=cfg.end,
    )
    
    return dataset
