# Standard library imports
import copy
import os
import pickle
import pdb
import uuid
from collections import Counter, deque
from enum import Enum
from typing import List, Optional
import logging

# Third-party imports
import numpy as np
import open3d as o3d
from omegaconf import DictConfig

# Local module imports
from utils.type import Observation

# Set up the module-level logger
logger = logging.getLogger(__name__)

class LocalObjStatus(Enum):
    UPDATING = "updating"
    PENDING = "pending for updating"
    ELIMINATION = "elimination"
    LM_ELIMINATION = "elimination for low mobility"
    HM_ELIMINATION = "elimination for high mpbility"
    WAITING = "waiting for stable obj process"

class BaseObject:
    
    # Global variable for config
    _cfg = None
    
    def __init__(self):
        # id
        self.uid = uuid.uuid4()
        
        # obs info
        self.observed_num = 0
        self.observations: List[str] = []
        
        # Spatial primitives
        self.pcd: Optional[o3d.geometry.PointCloud] = o3d.geometry.PointCloud()
        self.bbox: Optional[o3d.geometry.AxisAlignedBoundingBox] = o3d.geometry.AxisAlignedBoundingBox()
        
        # high level feats
        self.clip_ft: Optional[np.ndarray] = np.empty(0, dtype=np.float32)
        
        # class id 
        self.class_id: Optional[int] = None
        
        # Initialize save_path
        self.save_path = self._initialize_save_path()

        # is navigation goal flag
        self.nav_goal = False

    def __getstate__(self):
        # Prepare the state dictionary for serialization
        state = {
            'uid': self.uid,
            'pcd_points': np.asarray(self.pcd.points).tolist(),  # Convert to list
            'pcd_colors': np.asarray(self.pcd.colors).tolist(),  # Convert to list
            'clip_ft': self.clip_ft.tolist(),
            'class_id': self.class_id,
            'nav_goal': self.nav_goal
        }
        return state
    
    def __setstate__(self, state):
        self.uid = state.get('uid')
        
        # Restore PointCloud from points & colors
        points = np.array(state.get('pcd_points'))
        colors = np.array(state.get('pcd_colors'))

        if (len(points) != 0) or (len(colors) != 0):
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
        
        self.clip_ft = np.array(state.get('clip_ft'))
        self.class_id = state.get('class_id')
        self.nav_goal = state.get('nav_goal')

        self.observed_num = 0
        self.observations: List[str] = []

        self.save_path = self._initialize_save_path()
    
    @classmethod
    def initialize_config(cls, config: DictConfig):
        cls._cfg = config

        classes_path = config.yolo.classes_path
        if config.yolo.use_given_classes:
            classes_path = config.yolo.given_classes_path
            logger.info(f"[BaseObject] Using given classes, path:{classes_path}")
        
        with open(classes_path, 'r') as file:
            lines = file.readlines()
            num_classes = len(lines)
        # set num_classes for bayesian class filter
        cls._cfg.yolo.num_classes = num_classes
    
    def _initialize_save_path(self):
        if self._cfg:
            # save dir construction
            save_dir = self._cfg.map_save_path
            # If not exist, then create
            os.makedirs(save_dir, exist_ok=True)
            return os.path.join(save_dir, f"{self.uid}.pkl")
        
        return None
    
    def copy(self):
        return copy.deepcopy(self)
    
    def save_to_disk(self):
        """Save the object to disk using pickle."""
        with open(self.save_path, 'wb') as f:
            pickle.dump(self, f)
        
        if self._cfg.save_cropped:
            # save the cropped image in the observation
            save_dir = self._cfg.map_save_path
            save_dir = os.path.join(save_dir, f"{self.class_id}_{self.uid}")
            cropped_save_dir = os.path.join(save_dir, "cropped")
            masked_save_dir = os.path.join(save_dir, "masked")
            os.makedirs(cropped_save_dir, exist_ok=True)
            os.makedirs(masked_save_dir, exist_ok=True)
            for obs in self.observations:
                obs_idx = obs.idx
                cropped_image = obs.cropped_image
                masked_image = obs.masked_image
                cropped_image_dir = os.path.join(cropped_save_dir, f"{obs_idx}.png")
                masked_image_dir = os.path.join(masked_save_dir, f"{obs_idx}.png")
                # both cropped and masked images are np.ndarray, so save as png
                import imageio
                imageio.imwrite(cropped_image_dir, cropped_image)
                imageio.imwrite(masked_image_dir, masked_image)

    @staticmethod
    def load_from_disk(filename: str):
        """Load the object from disk using pickle."""
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            return obj
    
    def voxel_downsample_2d(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float,
    ) -> o3d.geometry.PointCloud:
        # Input 3d point cloud, voxel size, return downsampled 2d point cloud
        # This function is to avoid the warning caused by o3d.geometry.voxel_down_sample
        # TODO: Color is not right

        # Get point cloud's points
        points_arr = np.asarray(pcd.points)
        colors_arr = np.asarray(pcd.colors)

        # Only retain X and Y coordinates
        points_2d = points_arr[:, :2]

        # 2D voxel downsample based on voxel size
        grid_indices = np.floor(points_2d / voxel_size).astype(np.int32)
        unique_indices, inverse_indices = np.unique(grid_indices, axis=0, return_inverse=True)

        downsampled_points_2d = np.zeros_like(unique_indices, dtype=np.float64)
        downsampled_colors = np.zeros((len(unique_indices), 3), dtype=np.float64)

        # calculate the mean of points in each voxel
        for i in range(len(unique_indices)):
            mask = (inverse_indices == i)
            downsampled_points_2d[i] = points_2d[mask].mean(axis=0)
            downsampled_colors[i] = colors_arr[mask].mean(axis=0)

        # restore the Z with given Z
        downsampled_points = np.zeros((len(downsampled_points_2d), 3))
        downsampled_points[:, :2] = downsampled_points_2d
        # TODO: Magic number -> floor height
        downsampled_points[:, 2] = self._cfg.floor_height

        # Generate the downsampled point cloud
        downsampled_pcd = o3d.geometry.PointCloud()
        downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
        downsampled_pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)

        return downsampled_pcd

class LocalObject(BaseObject):

    # Global variable for overall idx
    _curr_idx = 0

    def __init__(self):
        super().__init__()

        # lm sign
        self.is_low_mobility: Optional[bool] = False
        # major plane info, the z value of the major plane
        self.major_plane_info = None
        
        # Split Check info dict
        self.split_info: Optional[dict] = {}
        self.max_common: int = 0
        self.should_split: Optional[bool] = False
        # debug for split feat
        self.split_class_id_one: Optional[int] = 0
        self.split_class_id_two: Optional[int] = 0

        # Spatial Stablity Check Info List
        self.spatial_stable_info: Optional[list] = []

        # status
        self.status = LocalObjStatus.UPDATING
        self.is_stable = False
        self.pending_count = 0
        self.waiting_count = 0

        # # bayesian stable
        self.num_classes = self._cfg.yolo.num_classes
        # # init the prob
        self.class_probs = np.ones(self.num_classes) / self.num_classes
        self.class_probs_history: List[str] = []
        self.max_prob = 0.0
        self.entropy = 0.0
        self.change_rate = 0.0

        # for local map merging
        self.is_merged = False

        ################
        # Debug Variable
        ################
        self.downsample_num: int = 0

    @classmethod
    def set_curr_idx(cls, idx: int):
        cls._curr_idx = idx

    def add_observation(
        self,
        observation: Observation
    ) -> None:
        self.observations.append(observation)
        self.observed_num += 1

    def get_latest_observation(
        self
    ) -> Observation:
        return self.observations[-1] if self.observations else None
    
    def clear_info(
        self,
    ) -> None:
        self.observed_num = 0
        self.observations = []
        self.pcd = o3d.geometry.PointCloud()
        # self.bbox = o3d.geometry.OrientedBoundingBox()
        self.class_id = None
        self.split_info = None
        self.max_common = 0
        self.should_split = False
        self.split_class_id_one = None
        self.split_class_id_two = None
        self.spatial_stable_info = None