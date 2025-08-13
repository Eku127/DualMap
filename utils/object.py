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
from utils.types import Observation

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

        # Calculate average points for each voxel
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
    
    # Baye: Update posterior probability
    def update_class_probs(
        self,
        alpha = 0.6,
    ) -> None:
        
        latest_obs = self.get_latest_observation()

        distance = latest_obs.distance

        # Current observation's confidence and class
        class_id = latest_obs.class_id
        conf = latest_obs.conf

        # Create class probability distribution for observation
        observation_probs = np.zeros(self.num_classes)

        # Distribute remaining confidence evenly among other classes
        remaining_conf = 1 - conf

        # Find top 10 classes with highest scores excluding current class
        all_indices = np.arange(self.num_classes)
        other_indices = all_indices[all_indices != class_id]  # Indices excluding current class
        top_10_indices = np.argsort(self.class_probs[other_indices])[-10:]  # Get top 10 classes with highest scores
        top_10_real_indices = other_indices[top_10_indices]  # Get real class indices

        # Distribute remaining confidence evenly among these 3 classes
        for idx in top_10_real_indices:
            observation_probs[idx] = remaining_conf / 10
        
        # Confidence of the current observation class
        observation_probs[class_id] = conf
        observation_probs /= np.sum(observation_probs)
        
        # Sliding window smoothing
        window_size = 5

        self.class_probs_history.append(observation_probs)
        if len(self.class_probs_history) > window_size:
            self.class_probs_history.pop(0)
        
        smoothed_probs = np.mean(self.class_probs_history, axis=0)

        # calculate alpha
        max_distance=10.0
        k = 1.0
        alpha = np.exp(-k * distance / max_distance)

        # Bayesian update
        self.class_probs = (1 - alpha) * self.class_probs + alpha * smoothed_probs

        # Normalize
        self.class_probs /= np.sum(self.class_probs)

    def update_info(
        self
    ) -> None:
        # Local Obj
        # Integrate information from observation into external
        # This is the update part in the filter
        # TODO: Check how many times is appropriate for point cloud downsampling

        latest_obs = self.get_latest_observation()

        if self.observed_num == 0:
            logger.error("[LocalObject] No observation in this object")
            return

        if self.observed_num == 1:
            self.pcd = latest_obs.pcd
            self.bbox = latest_obs.bbox
            self.clip_ft = latest_obs.clip_ft
            self.class_id = latest_obs.class_id
            self.is_low_mobility = latest_obs.is_low_mobility
            return

        # Split dict update
        self.update_split_info(latest_obs)
        if self.should_split:
            return

        # Get outside infomations
        # Merge pcd
        # Simply add
        self.pcd += latest_obs.pcd

        # Get new bbox
        # self.bbox = self.pcd.get_oriented_bounding_box(robust=True)
        self.bbox = self.pcd.get_axis_aligned_bounding_box()
        
        # Merge clip_ft
        self.clip_ft = (self.clip_ft * (self.observed_num - 1) + latest_obs.clip_ft) * 1.0 / (self.observed_num)
        # Normalize the new clip_ft
        norm = np.linalg.norm(self.clip_ft)
        self.clip_ft = self.clip_ft / norm

        # Update spatial stable info list
        self.update_spatial_stable_info(latest_obs)

        # Get major class id
        # get class id list
        class_ids = [obs.class_id for obs in self.observations]
        obj_class_id_counter = Counter(class_ids)
        most_common_class_id = obj_class_id_counter.most_common(1)[0][0]
        self.class_id = most_common_class_id

        # Bayesian update
        self.update_class_probs()

        # Get low mobility info
        # get low mobility info list
        low_mobility_infos = [obs.is_low_mobility for obs in self.observations]
        num_true = sum(low_mobility_infos)
        num_false = len(low_mobility_infos) - num_true
        most_common_lm = True if num_true > num_false else False

        self.is_low_mobility = most_common_lm

        # Periodically downsample the pcd
        if self.observed_num % self._cfg.downsample_interval == 0:
            self.downsample_num += 1
            self.pcd = self.pcd.voxel_down_sample(voxel_size=self._cfg.downsample_voxel_size)

            # only after downsample, we calculate the major plane info
            # If the object is low mobility, do major plane info calculation
            # TODO: do not need to calculate major plane info from scratch, incrementally calculate z_value
            # The major plane info can be calculated in the Observation Generation
            # Merging using the point number as the weight process
            if self.is_low_mobility:
                self.major_plane_info = self.find_major_plane_info()
            else:
                self.major_plane_info = None

    
    def update_split_info(
        self,
        latest_obs: Observation
    ) -> None:
        # https://www.yuque.com/u21262689/fxzc7g/fmfk1gkv6fbemlus?singleDoc#
        # Boundary condition check
        if latest_obs is None:
            raise ValueError("latest_obs cannot be None")
        if not hasattr(latest_obs, 'class_id') or not hasattr(latest_obs, 'idx'):
            raise ValueError("latest_obs must have class_id and idx attributes")

        # Split dict update
        if latest_obs.class_id not in self.split_info:
            self.split_info[latest_obs.class_id] = deque()
        self.split_info[latest_obs.class_id].append(latest_obs.idx)

        # Make sure the idx in the active window
        # Current window size is 10
        for class_id, idx_deque in self.split_info.items():
            while idx_deque and idx_deque[0] <= latest_obs.idx - self._cfg.active_window_size:
                idx_deque.popleft()
            # # if empty, delete
            # if len(idx_deque) == 0:
            #     del self.split_info[class_id]
            if len(idx_deque) < self._cfg.active_window_size:
                continue

        # logger.info("current idx: ", latest_obs.idx)
        # for class_id, idx_deque in self.split_info.items():
        #     logger.info(f"Class ID: {class_id}, Observations: {list(idx_deque)}")

        # TODO: WHAT about three deque? currently only two considered
        split_tuple = self.find_max_common_elements(self.split_info)
        # logger.info slipt tupe
        # logger.info("Max common elements:", split_tuple)
        self.max_common = split_tuple[0]
        if split_tuple[0] != 0:
            self.split_class_id_one = split_tuple[1][0]
            self.split_class_id_two = split_tuple[1][1]
        else:
            self.split_class_id_one = 0
            self.split_class_id_two = 0

        if self.max_common > self._cfg.max_common_th:
            self.should_split = True

    def print_split_info(
        self
    ) -> None:
        # logger.info split info into a string
        for class_id, idx_deque in self.split_info.items():
            logger.info(f"[LocalObject] Class ID: {class_id}, Observations: {list(idx_deque)}")

    def print_split_info(
        self,
    ) -> str:
        split_info = ""
        for class_id, idx_deque in self.split_info.items():
            string_split = f"{class_id}" + "-" + f"{list(idx_deque)}"
            split_info = split_info + string_split + "||"
        return split_info

    def find_max_common_elements(
        self,
        data: dict,
    ) -> tuple:
        # Find max common elements in split info

        max_common_count = 0
        max_common_pair = (None, None)

        class_ids = list(data.keys())

        for i in range(len(class_ids)):
            for j in range(i + 1, len(class_ids)):
                class_id1 = class_ids[i]
                class_id2 = class_ids[j]

                deque1 = data[class_id1]
                deque2 = data[class_id2]

                # jump the empty deque
                if not deque1 or not deque2:
                    continue

                set1 = set(deque1)
                set2 = set(deque2)

                common_elements = set1 & set2
                common_count = len(common_elements)

                if common_count > max_common_count:
                    max_common_count = common_count
                    max_common_pair = (class_id1, class_id2)

        max_common_tuple = (max_common_count, max_common_pair)

        return max_common_tuple

    def update_info_from_observations(
        self,
    ) -> None:
        # This function will be used to update the object class info based on all observations
        # ALso can use iteration of the update_info function
        counter = 0
        for obs in self.observations:
            counter += 1
            if counter == 1:
                self.pcd = obs.pcd
                self.clip_ft = obs.clip_ft
                continue
            self.pcd += obs.pcd
            self.clip_ft += obs.clip_ft

        # downsample pcd
        self.pcd = self.pcd.voxel_down_sample(voxel_size=self._cfg.downsample_voxel_size)
        # Group to majority
        from utils.pcd_utils import init_pcd_denoise_dbscan
        self.pcd = init_pcd_denoise_dbscan(
            self.pcd, self._cfg.dbscan_eps, self._cfg.dbscan_min_points)

        # self.bbox = self.pcd.get_oriented_bounding_box(robust=True)
        self.bbox = self.pcd.get_axis_aligned_bounding_box()

        # norm feat
        self.clip_ft = (self.clip_ft) * 1.0 / (self.observed_num)
        # Normalize the new clip_ft
        norm = np.linalg.norm(self.clip_ft)
        self.clip_ft = self.clip_ft / norm

        # Get major class id
        # get class id list
        class_ids = [obs.class_id for obs in self.observations]
        obj_class_id_counter = Counter(class_ids)
        most_common_class_id = obj_class_id_counter.most_common(1)[0][0]
        self.class_id = most_common_class_id

        # Get low mobility info
        # get low mobility info list
        low_mobility_infos = [obs.is_low_mobility for obs in self.observations]
        num_true = sum(low_mobility_infos)
        num_false = len(low_mobility_infos) - num_true
        most_common_lm = True if num_true > num_false else False

        self.is_low_mobility = most_common_lm

        # get major plane info
        if self.is_low_mobility:
            self.major_plane_info = self.find_major_plane_info()

    def update_spatial_stable_info(
        self,
        latest_obs: Observation
    ) -> None:
        # Including two parts of the spatial check
        # 1. latest obs bbox( or other primitives ) compare with the overall bbox or pcd
        # 2. latest obs bbox with prev obs bbox
        pass

    def update_status(
        self,
    ) -> None:
        # Object life cycle
        # Updating the status of the object by stability and infos

        # last observation time
        last_obs = self.get_latest_observation()

        # 1. if the object is in inside the sliding window, status will be UPDATING
        # No matter what previous status is, the object will always be UPDATING if in the window
        if last_obs.idx <= self._curr_idx and last_obs.idx >= max(self._curr_idx - self._cfg.active_window_size, 0):
            self.status = LocalObjStatus.UPDATING
            self.pending_count = 0
            self.waiting_count = 0
            return

        # 2. if the object is out of the sliding window, check the stability first
        # and now the object is out of the window
        # Once the object is set as stable, it will always be stable
        self.stability_check()

        # if not stable, pending for next update, set status to PENDING
        if self.is_stable == False:
            self.status = LocalObjStatus.PENDING
            self.pending_count += 1

            # if pending count is large, status set as ELIMINATION
            if self.pending_count > self._cfg.max_pending_count:
                self.status = LocalObjStatus.ELIMINATION
                return

            return

        # 3. if the object is stable, waiting first
        # and now the obj is set as stable
        self.status = LocalObjStatus.WAITING
        self.waiting_count += 1

        if self.waiting_count < self._cfg.max_pending_count:
            # still in the waiting status, just return
            return
            
        # if waiting enough, then judge next step by lm status
        if self.is_low_mobility:
            self.status = LocalObjStatus.LM_ELIMINATION
            return
        else:
            self.status = LocalObjStatus.HM_ELIMINATION
            return

    def stability_check(
        self,
    ) -> None:
        # Check if the object is stable
        # the only function change the is_stable flag
        # Enhance the ways of stability check, here we only use label check
        # The initial check here is very simple, then goes bayesian stability check

        # 1. obs num should be large
        if self.observed_num < self._cfg.stable_num:
            self.is_stable = False
            return

        # 2. if the largest label over 1/2 of the observed num, just set as stable
        class_ids = [obs.class_id for obs in self.observations]
        obj_class_id_counter = Counter(class_ids)
        most_common_class_id, most_common_count = obj_class_id_counter.most_common(1)[0]

        if most_common_count > self.observed_num / 3:
            self.is_stable = True
            return
        
        # 3. if the object is stable by the filter, then set as stable
        if self.is_class_converged():
            self.is_stable = True
            return

        self.is_stable = False
    
    def is_class_converged(
        self,
        entropy_threshold=0.2,
        prob_threshold=0.50,
        change_rate_threshold=0.2,
        window_size=3,
    ) -> bool:
        
        # 1. Major class probability check
        max_prob = np.max(self.class_probs)
        self.max_prob = max_prob
        if max_prob > prob_threshold:
            return True

        # 2. Entropy check
        entropy = -np.sum(self.class_probs * np.log(self.class_probs + 1e-10))  # 防止 log(0)
        self.entropy = entropy
        if entropy < entropy_threshold:
            return True

        # 3. Change rate check
        if len(self.class_probs_history) >= window_size:
            recent_probs = np.array(self.class_probs_history[-window_size:])
            change_rate = np.mean(np.abs(recent_probs[1:] - recent_probs[:-1]), axis=0)
            self.change_rate = np.max(change_rate)
            if np.max(change_rate) < change_rate_threshold:
                return True

        return False

    def find_major_plane_info(
        self,
        bin_size = 0.02,
    ) -> float:
        # This function will find the major plane of the object
        # return the major plane z value
        # Get the pcd
        # get all z_zxis value
        z_axis = np.asarray(self.pcd.points)[:, 2]

        # Get the bin count
        bin_edges = np.arange(z_axis.min(), z_axis.max() + bin_size, bin_size)

        # Histogram calculation
        hist, bin_edges = np.histogram(z_axis, bins=bin_edges)

        # Optional: save the histogram

        # Find the peak of the histogram
        peak_index = np.argmax(hist)

        # Get the major plane z value
        major_plane_z = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2.0

        return major_plane_z

class GlobalObject(BaseObject):
    def __init__(self, observation=None):
        super().__init__()

        # Spatial primitives
        self.pcd_2d: Optional[o3d.geometry.PointCloud] = o3d.geometry.PointCloud()
        self.bbox_2d: Optional[o3d.geometry.AxisAlignedBoundingBox] = o3d.geometry.AxisAlignedBoundingBox()
        
        # Related objs
        # Current we "only save clip feats" <-- PAY Attention!
        self.related_objs: List[np.ndarray] = []

        # for visualization in rerun
        self.related_bbox = []
        self.related_color = []

        # If provide the LocalObject, then initialize the GlobalObject from it
        if observation is not None:
            self.init_from_global_obs(observation)
    
    def __getstate__(self):
        # Base Object getstate
        state = super().__getstate__()
        
        # serialize List[np.ndarray] 
        state['related_objs'] = [arr.tolist() for arr in self.related_objs]

        # serialize pcd_2d
        state['pcd_2d_points'] = np.asarray(self.pcd_2d.points).tolist()
        state['pcd_2d_colors'] = np.asarray(self.pcd_2d.colors).tolist()


        # Serialize related_bbox (convert AxisAlignedBoundingBox to dict)
        state['related_bbox'] = [
            {
                'min_bound': bbox.get_min_bound().tolist(),
                'max_bound': bbox.get_max_bound().tolist()
            }
            for bbox in self.related_bbox
        ]

        # Serialize related_color (class IDs list)
        state['related_color'] = self.related_color  # assuming it's a list of class IDs

        return state

    def __setstate__(self, state):
        # Base Object setstate
        super().__setstate__(state)

        # Restore related_objs as np.ndarray
        self.related_objs = [np.array(arr) for arr in state.get('related_objs', [])]

        # Restore pcd_2d (points and colors)
        points = np.array(state.get('pcd_2d_points'))
        colors = np.array(state.get('pcd_2d_colors'))
        
        self.pcd_2d = o3d.geometry.PointCloud()
        self.pcd_2d.points = o3d.utility.Vector3dVector(points)
        self.pcd_2d.colors = o3d.utility.Vector3dVector(colors)

        self.bbox_2d = self.pcd_2d.get_axis_aligned_bounding_box()

        self.related_bbox = [
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=np.array(bbox_dict['min_bound']),
                max_bound=np.array(bbox_dict['max_bound'])
            )
            for bbox_dict in state.get('related_bbox', [])
        ]

        # Restore related_color (assuming it's stored as a list of class IDs)
        self.related_color = state.get('related_color', [])

        # Set obs num to 1 to avoid global updating bug
        self.observed_num = 1

    def copy(self):
        return copy.deepcopy(self)
    
    def init_from_global_obs(self, observation):
        
        self.uid = observation.uid
        
        self.pcd = observation.pcd
        self.bbox = observation.bbox

        self.pcd_2d = observation.pcd_2d
        self.bbox_2d = observation.bbox_2d
                
        self.clip_ft = observation.clip_ft
        
        # Class ID
        self.class_id = observation.class_id

        # Related objs
        self.related_objs = observation.related_objs
        
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


    def update_info(
        self
    ) -> None:
        # Global Obj
        
        latest_obs = self.get_latest_observation()

        if self.observed_num == 0:
            logger.error("[GlobalObject] No observation in this object")
            return
        
        if self.observed_num == 1:
            self.uid = latest_obs.uid
            
            self.pcd = latest_obs.pcd
            self.bbox = latest_obs.bbox
            self.pcd_2d = latest_obs.pcd_2d
            self.bbox_2d = latest_obs.bbox_2d

            self.clip_ft = latest_obs.clip_ft
            self.class_id = latest_obs.class_id

            self.related_objs = latest_obs.related_objs
            
            # for visualization in rerun
            self.related_bbox = latest_obs.related_bbox
            self.related_color = latest_obs.related_color
            return
        
        # Update the information for outside
        
        # Merge pcd
        # Simply add
        self.pcd += latest_obs.pcd
        self.pcd = self.pcd.voxel_down_sample(voxel_size=self._cfg.downsample_voxel_size)
        
        # Get new bbox
        self.bbox = self.pcd.get_axis_aligned_bounding_box()

        # Merge pcd_2d
        # get the points num of the latest obs and current pcd_2d
        original_num = len(self.pcd_2d.points)
        incoming_num = len(latest_obs.pcd_2d.points)
        
        self.pcd_2d += latest_obs.pcd_2d
        self.pcd_2d = self.voxel_downsample_2d(pcd=self.pcd_2d, voxel_size=self._cfg.downsample_voxel_size)
        self.bbox_2d = self.pcd_2d.get_axis_aligned_bounding_box()

        
        # TODO: Should we merge clip feat? Any judgement?
        # Answer: Weighted merge
        
        # TODO: for class id, we choose the bigger one? Currently we choose the bigger one as the output
        # for class id, we choose the bigger one as the output
        # bigger or smaller depends on the pointcloud size of the pcd_2d
        if original_num < incoming_num:
            self.class_id = latest_obs.class_id
            self.clip_ft = latest_obs.clip_ft
        
        # TODO: Any other matching strategy on related objs?
        # Maintain the related objs, simply add the objs from the latest observation (Current Strategy)
        self.related_objs += latest_obs.related_objs

        # for visualization in rerun
        self.related_bbox += latest_obs.related_bbox
        self.related_color += latest_obs.related_color

        pass
