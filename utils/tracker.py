import logging
import pdb
from typing import List

import numpy as np
import faiss
import torch
import torch.nn.functional as F
import open3d as o3d
from omegaconf import DictConfig
from scipy.sparse.csgraph import connected_components

from utils.types import Observation
from utils.object import BaseObject
from utils.visualizer import plot_similarity_matrix

# Set up the module-level logger
logger = logging.getLogger(__name__)

class Tracker:
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:    
        # Construct different tracker types based on cfg.tracker classification
        # config
        self.cfg = cfg

        self.__is_global = False

        self.merge_info = None

    def set_ref_map(
        self,
        ref_map: List[BaseObject],
    ) -> None:
        self.ref_map = ref_map

    def set_ref_frame(
        self,
        ref_frame: List[Observation],
    ) -> None:
        self.ref_frame = ref_frame

    def set_current_frame(
        self,
        curr_frame
    ) -> None:
        # TODO: Deepcopy?
        self.curr_frame = curr_frame

    def get_current_frame(
        self,
    ):
        return self.curr_frame

    def get_merge_info(
        self
    ):
        return self.merge_info

    def set_global(
        self,
    ) -> None:
        self.__is_global = True
        return

    def matching_map(
        self,
        is_map_only: bool = False,
    ) -> None:
        # match current frame to the map
        # Find relationships between current observations and previous map

        if self.__is_global:
            # for global matching we use geometry only
            spatial_sim_mat = self.compute_global_spatial_sim()

            sim_mat = spatial_sim_mat.T

            self.update_global_obs_with_sim_mat(sim_mat)

        else:
            if is_map_only:
                spatial_sim_mat = self.compute_overlap_spatial_sim()

                graph = spatial_sim_mat > self.cfg.merge_sim_threshold
                n_components, component_labels = connected_components(graph)
                # get merge info
                self.merge_info = [np.where(component_labels == i)[0] for i in range(n_components)]

            # Spatial sim torch
            # M(map) x N(curr)
            spatial_sim_mat = self.compute_spatial_sim()
            # plot_similarity_matrix(spatial_sim_mat)

            # visual sim torch
            # M(map) x N(curr)
            visual_sim_mat = self.compute_visual_sim()
            # plot_similarity_matrix(visual_sim_mat)

            if (
                spatial_sim_mat is None or visual_sim_mat is None
                or spatial_sim_mat.numel() == 0 or visual_sim_mat.numel() == 0
                or spatial_sim_mat.shape != visual_sim_mat.shape
            ):
                logger.warning("[Tracker] Empty or mismatched similarity matrices, skipping match.")
                return

            # overall sim
            sim_mat = spatial_sim_mat + visual_sim_mat
            # switch (map, curr) to (curr, map)
            sim_mat = sim_mat.T

            if sim_mat.shape[0] == 0 or sim_mat.shape[1] == 0:
                logger.warning("[Tracker] sim_mat is empty after transpose, skipping update.")

                return

            self.update_obs_with_sim_mat(sim_mat)

    def compute_overlap_spatial_sim(
        self    
    ) -> np.ndarray:
        len_map = len(self.ref_map)
        len_curr = len(self.curr_frame)

        overlap_matrix = np.zeros((len_map, len_curr))

        # calculate iou first

        # Get stacked bboxes for iou calculation
        # from map
        map_bbox_values = []
        for obj in self.ref_map:
            obj_bbox = np.asarray(obj.bbox.get_box_points())
            obj_bbox = torch.from_numpy(obj_bbox)
            map_bbox_values.append(obj_bbox)
        map_bbox_torch = torch.stack(map_bbox_values, dim=0)

        # from curr obs
        curr_bbox_values = []
        for obj in self.ref_map:
            obj_bbox = np.asarray(obj.bbox.get_box_points())
            obj_bbox = torch.from_numpy(obj_bbox)
            curr_bbox_values.append(obj_bbox)
        curr_bbox_torch = torch.stack(curr_bbox_values, dim=0)

        # calculate iou
        iou = self.compute_3d_iou_batch(map_bbox_torch, curr_bbox_torch)

        for idx_a in range(len_map):
            for idx_b in range(idx_a + 1, len_curr):
                if iou[idx_a, idx_b] < 1e-6:
                    continue

                pcd_map = self.ref_map[idx_a].pcd
                pcd_curr = self.curr_frame[idx_b].pcd

                overlap_matrix[idx_a, idx_b] = self.find_overlapping_ratio_faiss(
                    pcd_map, pcd_curr, radius=0.02)

        return overlap_matrix

    def compute_spatial_sim(
        self
    ) -> np.ndarray:

        len_map = len(self.ref_map)
        len_curr = len(self.curr_frame)

        if len_map == 0 or len_curr == 0:
            return torch.zeros((len_map, len_curr))  # shape = (0, N) or (M, 0)

        overlap_matrix = np.zeros((len_map, len_curr))

        points_map = [np.asarray(obj.pcd.points, dtype=np.float32)
                       for obj in self.ref_map]  # m size
        indices_map = [faiss.IndexFlatL2(points_arr.shape[1])
                        for points_arr in points_map]  # m indices
        for idx, points_arr in zip(indices_map, points_map):
            idx.add(points_arr)

        points_curr = [np.asarray(obs.pcd.points, dtype=np.float32) for obs in self.curr_frame]

        # Get stacked bboxes for iou calculation
        # from map
        map_bbox_values = []
        for obj in self.ref_map:
            obj_bbox = np.asarray(obj.bbox.get_box_points())
            obj_bbox = torch.from_numpy(obj_bbox)
            map_bbox_values.append(obj_bbox)
        map_bbox_torch = torch.stack(map_bbox_values, dim=0)

        # from curr obs
        curr_bbox_values = []
        for obs in self.curr_frame:
            obs_bbox = np.asarray(obs.bbox.get_box_points())
            obs_bbox = torch.from_numpy(obs_bbox)
            curr_bbox_values.append(obs_bbox)
        curr_bbox_torch = torch.stack(curr_bbox_values, dim=0)

        # calculate iou
        iou = self.compute_3d_iou_batch(map_bbox_torch, curr_bbox_torch)

        counter = 0

        # compute the overlap info using pcd
        for idx_a in range(len_map):
            for idx_b in range(len_curr):

                if iou[idx_a, idx_b] < 1e-6:
                    counter += 1
                    continue

                D, I = indices_map[idx_a].search(points_curr[idx_b], 1) 
                overlap = (D < self.cfg.downsample_voxel_size ** 2).sum()
                # calculate the ratio of points within the threshold distance
                denom = len(points_curr[idx_b])
                if denom == 0:
                    overlap_matrix[idx_a, idx_b] = 0.0
                else:
                    overlap_matrix[idx_a, idx_b] = overlap / denom

        overlap_matrix = torch.from_numpy(overlap_matrix)

        return overlap_matrix

    def compute_global_spatial_sim(
        self,
    ) -> np.ndarray:
        # compute globally spatial sim
        # we only use 2d bbox to judge here
        len_map = len(self.ref_map)
        len_curr = len(self.curr_frame)

        logger.info(f"[Tracker][Global] Current obs num: {len_curr}, current map num: {len_map}")

        # Get stacked bboxes for iou calculation
        map_bbox_values = []

        for obj in self.ref_map:
            min_bound = obj.bbox_2d.get_min_bound()
            max_bound = obj.bbox_2d.get_max_bound()
            map_bbox_values.append(torch.tensor([min_bound[0], min_bound[1], max_bound[0], max_bound[1]]))
        map_bbox_torch = torch.stack(map_bbox_values, dim=0)

        # from curr obs
        curr_bbox_values = []
        for obs in self.curr_frame:
            min_bound = obs.bbox_2d.get_min_bound()
            max_bound = obs.bbox_2d.get_max_bound()
            curr_bbox_values.append(torch.tensor([min_bound[0], min_bound[1], max_bound[0], max_bound[1]]))
        curr_bbox_torch = torch.stack(curr_bbox_values, dim=0)

        ratio = self.compute_match_by_intersection_ratio(map_bbox_torch, curr_bbox_torch)

        return ratio

    def compute_match_by_intersection_ratio(
        self,
        bboxes1: torch.Tensor,
        bboxes2: torch.Tensor,
        threshold=0.8
    ) -> torch.Tensor:
        """
        Calculate match matrix based on the intersection ratio between bounding boxes.
        
        bboxes1: torch.Tensor of shape (N, 4), first set of bounding boxes (min_x, min_y, max_x, max_y)
        bboxes2: torch.Tensor of shape (M, 4), second set of bounding boxes (min_x, min_y, max_x, max_y)
        threshold: match threshold, default is 0.8
        
        Returns: torch.Tensor of shape (N, M), match matrix
        """
        # Extract coordinates of bounding boxes
        bboxes1_min_x, bboxes1_min_y, bboxes1_max_x, bboxes1_max_y = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]
        bboxes2_min_x, bboxes2_min_y, bboxes2_max_x, bboxes2_max_y = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]

        # Compute intersection coordinates
        inter_min_x = torch.max(bboxes1_min_x[:, None], bboxes2_min_x)  # top-left x of intersection
        inter_min_y = torch.max(bboxes1_min_y[:, None], bboxes2_min_y)  # top-left y of intersection
        inter_max_x = torch.min(bboxes1_max_x[:, None], bboxes2_max_x)  # bottom-right x of intersection
        inter_max_y = torch.min(bboxes1_max_y[:, None], bboxes2_max_y)  # bottom-right y of intersection

        # Calculate intersection width and height, ensuring non-negative values
        inter_width = (inter_max_x - inter_min_x).clamp(min=0)
        inter_height = (inter_max_y - inter_min_y).clamp(min=0)

        # Calculate intersection area
        inter_area = inter_width * inter_height

        # Calculate the area of each bounding box
        bboxes1_area = (bboxes1_max_x - bboxes1_min_x) * (bboxes1_max_y - bboxes1_min_y)
        bboxes2_area = (bboxes2_max_x - bboxes2_min_x) * (bboxes2_max_y - bboxes2_min_y)

        # Calculate the ratio of intersection area to each bounding box's area
        ratio1 = inter_area / bboxes1_area[:, None]  # ratio for bboxes1
        ratio2 = inter_area / bboxes2_area  # ratio for bboxes2

        # Determine matches if either ratio exceeds the threshold
        # match_matrix = (ratio1 >= threshold) | (ratio2 >= threshold)
        match_matrix = torch.max(ratio1, ratio2)

        return match_matrix

    def compute_visual_sim(
        self,
    ) -> np.ndarray:
        # Get stacked clip fts for calculation
        # from map
        map_feats_values = []
        for obj in self.ref_map:
            obj_feat = torch.from_numpy(obj.clip_ft)
            map_feats_values.append(obj_feat)

        if len(map_feats_values) == 0:
            return torch.zeros((0, 0))

        map_feats_torch = torch.stack(map_feats_values, dim=0) # (M, D)

        # from curr obs
        curr_feats_values = []
        for obs in self.curr_frame:
            obs_feat = torch.from_numpy(obs.clip_ft)
            curr_feats_values.append(obs_feat)
        curr_feats_torch = torch.stack(curr_feats_values, dim=0) # (N, D)

        map_fts = map_feats_torch.unsqueeze(-1) # (M, D, 1)
        curr_fts = curr_feats_torch.T.unsqueeze(0) # (1, D, N)

        visual_sim = F.cosine_similarity(map_fts, curr_fts, dim=1) # (M, N)

        return visual_sim

    def update_obs_with_sim_mat(
        self,
        sim_mat: torch.Tensor
    ) -> None:
        # update the obs in current frame with the matched map
        # IF no matches, then the current obs matched places will be None

        # get len of the curr obs
        len_curr_obs = len(self.curr_frame)

        add_new_obj = 0

        # update information into current observation
        for idx in range(len_curr_obs):
            max_sim_value = sim_mat[idx].max()
            if max_sim_value > self.cfg.sim_threshold:
                map_idx = sim_mat[idx].argmax().item()
                self.curr_frame[idx].matched_obj_uid = self.ref_map[map_idx].uid
                self.curr_frame[idx].matched_obj_score = max_sim_value
                self.curr_frame[idx].matched_obj_idx = map_idx
            else:
                self.curr_frame[idx].matched_obj_uid = None
                add_new_obj += 1

        logger.info(f"[Tracker] Added {add_new_obj} new objects, current detections: {len_curr_obs}")

    def update_global_obs_with_sim_mat(
        self,
        sim_mat: torch.Tensor
    ) -> None:
        len_curr_obs = len(self.curr_frame)

        add_new_obj = 0

        for obs_idx in range(len_curr_obs):
            max_sim_value = sim_mat[obs_idx].max()

            # TODO: Magic number -> thereshold
            if max_sim_value > 0.8:
                map_idx = sim_mat[obs_idx].argmax().item()

                # print obs_idx, map_idx, max_sim_value
                logger.info(f"[Tracker][Global] obs_idx: {obs_idx}, map_idx: {map_idx}, max_sim_value: {max_sim_value}  ")

                self.curr_frame[obs_idx].matched_obj_uid = self.ref_map[map_idx].uid
                self.curr_frame[obs_idx].matched_obj_score = sim_mat[obs_idx][map_idx].item()
                self.curr_frame[obs_idx].matched_obj_idx = map_idx

                logger.info(f"[Tracker][Global] Finding matching, obj from observation is: {self.curr_frame[obs_idx].class_id}, matched map obj is : {self.ref_map[map_idx].class_id}, score: {max_sim_value}")

            else:
                self.curr_frame[obs_idx].matched_obj_uid = None
                add_new_obj += 1

        logger.info(f"[Tracker][Global] Added {add_new_obj} new objects, current observations: {len_curr_obs}")

    def find_overlapping_ratio_faiss(self, pcd1, pcd2, radius=0.02):
        """
        Calculate the percentage of overlapping points between two point clouds using FAISS.

        Parameters:
        pcd1 (numpy.ndarray): Point cloud 1, shape (n1, 3).
        pcd2 (numpy.ndarray): Point cloud 2, shape (n2, 3).
        radius (float): Radius for KD-Tree query (adjust based on point density).

        Returns:
        float: Overlapping ratio between 0 and 1.
        """
        if type(pcd1) == o3d.geometry.PointCloud and type(pcd2) == o3d.geometry.PointCloud:
            pcd1 = np.asarray(pcd1.points)
            pcd2 = np.asarray(pcd2.points)

        if pcd1.shape[0] == 0 or pcd2.shape[0] == 0:
            return 0

        # Create the FAISS index for each point cloud
        index1 = faiss.IndexFlatL2(pcd1.shape[1])
        index2 = faiss.IndexFlatL2(pcd2.shape[1])
        index1.add(pcd1.astype(np.float32))
        index2.add(pcd2.astype(np.float32))

        # Query all points in pcd1 for nearby points in pcd2
        D1, I1 = index2.search(pcd1.astype(np.float32), k=1)
        D2, I2 = index1.search(pcd2.astype(np.float32), k=1)

        number_of_points_overlapping1 = np.sum(D1 < radius**2)
        number_of_points_overlapping2 = np.sum(D2 < radius**2)

        overlapping_ratio = np.max(
            [number_of_points_overlapping1 / pcd1.shape[0], number_of_points_overlapping2 / pcd2.shape[0]]
        )

        return overlapping_ratio

    def compute_box_volume_torch(self, box):
        # box shape is (M, 8, 3)
        edge1 = torch.norm(box[:, 1] - box[:, 0], dim=-1)
        edge2 = torch.norm(box[:, 3] - box[:, 0], dim=-1)
        edge3 = torch.norm(box[:, 4] - box[:, 0], dim=-1)
        return edge1 * edge2 * edge3

    def compute_intersection_volume_torch(self, bbox1, bbox2):
        # Calculate min/max corners for intersection computation
        min_corner1 = torch.min(bbox1, dim=1).values  # Shape (M, 3)
        max_corner1 = torch.max(bbox1, dim=1).values  # Shape (M, 3)
        min_corner2 = torch.min(bbox2, dim=1).values  # Shape (N, 3)
        max_corner2 = torch.max(bbox2, dim=1).values  # Shape (N, 3)

        # Broadcasting for pairwise intersection
        min_intersection = torch.maximum(
            min_corner1[:, None], min_corner2
        )  # Shape (M, N, 3)
        max_intersection = torch.minimum(
            max_corner1[:, None], max_corner2
        )  # Shape (M, N, 3)
        intersection_dims = torch.clamp(
            max_intersection - min_intersection, min=0
        )  # Shape (M, N, 3)
        return torch.prod(intersection_dims, dim=-1)  # Shape (M, N)

    def compute_3d_iou_batch(self, bbox1, bbox2):
        """
        Optimized IoU computation between two sets of axis-aligned 3D bounding boxes using PyTorch.

        bbox1: (M, 8, 3) tensor
        bbox2: (N, 8, 3) tensor

        returns: (M, N) tensor of IoU values
        """
        # Ensure inputs are torch tensors
        if not torch.is_tensor(bbox1):
            bbox1 = torch.tensor(bbox1, dtype=torch.float32)
        if not torch.is_tensor(bbox2):
            bbox2 = torch.tensor(bbox2, dtype=torch.float32)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bbox1 = bbox1.to(device)
        bbox2 = bbox2.to(device)

        # Compute volumes
        volume1 = self.compute_box_volume_torch(bbox1)  # Shape (M,)
        volume2 = self.compute_box_volume_torch(bbox2)  # Shape (N,)

        # Compute intersection volumes
        intersection_volume = self.compute_intersection_volume_torch(
            bbox1, bbox2
        )  # Shape (M, N)

        # Compute IoU
        iou = intersection_volume / (volume1[:, None] + volume2 - intersection_volume)
        return iou.cpu()
