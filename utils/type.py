from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import numpy as np
import open3d as o3d

import uuid
import copy
import json

@dataclass
class DataInput:
    idx: int = 0
    time_stamp: float = 0.0
    color: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 3), dtype=np.uint8))
    # Depth in H, W, 1
    depth: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    color_name: str = ""
    # Intrinsic in 3*3
    intrinsics: np.ndarray = field(default_factory=lambda: np.eye(3))
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    
    def clear(self) -> None:
        self.idx = 0
        self.time_stamp = 0.0
        self.color = np.empty((0, 0, 3), dtype=np.uint8)
        self.depth = np.empty((0, 0), dtype=np.float32)
        self.color_name = ""
        self.intrinsics = np.eye(3)
        self.pose = np.eye(4)
    
    def copy(self):
        return copy.deepcopy(self)

@dataclass
class Observation:
    class_id: int = 0
    pcd: o3d.geometry.PointCloud = field(default_factory=o3d.geometry.PointCloud)
    bbox: o3d.geometry.AxisAlignedBoundingBox = field(default_factory=o3d.geometry.AxisAlignedBoundingBox)
    clip_ft: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    
    # matching
    matched_obj_uid: None = None
    matched_obj_score: float = 0.0
    matched_obj_idx: int = -1

@dataclass
class LocalObservation(Observation):
    idx: int = 0
    mask: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.uint8))
    xyxy: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))
    conf: float = 0.0
    distance: float = 0.0
    
    is_low_mobility: bool = False

    # This property is for debugging
    masked_image: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 3), dtype=np.uint8))
    cropped_image: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 3), dtype=np.uint8))

@dataclass
class GlobalObservation(Observation):
    uid: uuid.UUID = field(default_factory=uuid.uuid4)
    pcd_2d: o3d.geometry.PointCloud = field(default_factory=o3d.geometry.PointCloud)
    bbox_2d: o3d.geometry.AxisAlignedBoundingBox = field(default_factory=o3d.geometry.AxisAlignedBoundingBox)
    # related objs
    # Current we "only save clip feats" <-- PAY Attention!
    related_objs: list = field(default_factory=list)
    # only for better demo
    related_bbox: list = field(default_factory=list)
    related_color: list = field(default_factory=list)