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


class ObjectClasses:
    '''
    Manages objects classes and their associated colors

    Create color map for a class file. Also manage the background type to show or not. Default bg clases are [floor, wall, ceiling]
    
    Attributes:
        classes_file_path (str): Path to the file containing class names, one per line.

    Usage:
        obj_classes = ObjectClasses(classes_file_path, skip_bg=True)
        model.set_classes(obj_classes.get_classes_arr())
        some_class_color = obj_classes.get_class_color(index or class_name)
    '''

    def __init__(self, classes_file_path, bg_classes, skip_bg):
        self.classes_file_path = Path(classes_file_path)
        self.bg_classes = bg_classes
        self.skip_bg = skip_bg
        self.classes, self.class_to_color = self._load_or_create_colors(selection_ratio = 1.0)

    def _load_or_create_colors(self, selection_ratio = 1.0):
        '''
        Key function for classes creation and color creation
        Need external class file for detection part
        '''
        with open(self.classes_file_path, "r") as f:
            all_class = [cls.strip() for cls in f.readlines()]

        # filter all classes with skip bg flag
        if self.skip_bg:
            classes = [cls for cls in all_class if cls not in self.bg_classes]
        else:
            classes = all_class
        
        # add color to each class
        # load color path
        color_file_path = self.classes_file_path.parent / f"{self.classes_file_path.stem}_colors.json"
        
        id_color_file_path = self.classes_file_path.parent / f"{self.classes_file_path.stem}_id_colors.json"

        if color_file_path.exists():
            with open(color_file_path, "r") as f:
                class_to_color = json.load(f)
            # construct a dict map {class, color}
            class_to_color = {cls : class_to_color[cls] for cls in classes if cls in class_to_color}
        else:
            class_to_color = {class_name: list(np.random.rand(3).tolist())for class_name in classes}
            # Generate the corresponding id_to_color mapping
            id_to_color = {str(i): class_to_color[cls] for i, cls in enumerate(classes)}
            # dump the new dict to json
            with open(color_file_path, "w") as f:
                json.dump(class_to_color, f)
            
            with open(id_color_file_path, "w") as f:
                json.dump(id_to_color, f)
        
        if selection_ratio == 1.0:
            return classes, class_to_color

        import random

        # Randomly select a portion of classes and their colors according to the ratio
        num_selected = max(1, int(len(classes) * selection_ratio))  # Select at least one
        selected_classes = random.sample(classes, num_selected)
        selected_class_to_color = {cls: class_to_color[cls] for cls in selected_classes}
                
        return selected_classes, selected_class_to_color

    def get_classes_arr(self):
        '''
        Return the list of classes names
        '''
        return self.classes
    
    def get_bg_classes_arr(self):
        '''
        Return the list of the skipped classes names
        '''
        return self.bg_classes
    
    def get_class_color(self, key):
        '''
        Get tge color associated with a given class name or index
        
        Args:
            key (str or int): The index or name of the class
        
        Returns:
            list: The color with RGB associated with the class
        '''
        if isinstance(key, int):
            if key < 0 or key >= len(self.classes):
                raise ValueError(f"Invalid class index out of range: {key}")
            return self.class_to_color[self.classes[key]]
        elif isinstance(key, str):
            if key not in self.classes:
                raise ValueError(f"Invalid class name: {key}")
            return self.class_to_color[key]
        else:
            raise TypeError(f"Invalid type for key: {type(key)}")
        
    def get_class_color_dict_by_index(self):
        '''
        Return a dictionary mapping class index to color, indexed by class index
        '''
        return {str(i): self.class_to_color[self.classes[i]] for i in range(len(self.classes))}
    
class GoalMode(Enum):
    RANDOM = "random"
    CLICK = "click"
    INQUIRY = "inquiry"