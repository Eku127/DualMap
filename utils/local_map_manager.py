import os
import shutil
import pdb
import logging
from collections import Counter
from typing import List

import numpy as np
import networkx as nx
from omegaconf import DictConfig

from utils.types import Observation, GlobalObservation, GoalMode
from utils.object import LocalObject, LocalObjStatus
from utils.base_map_manager import BaseMapManager
from utils.navigation_helper import NavigationGraph

# Set up the module-level logger
logger = logging.getLogger(__name__)

class LocalMapManager(BaseMapManager):
    def __init__(
        self,
        cfg: DictConfig,
        ) -> None:
        # Construct function
        super().__init__(cfg)

        # idx
        self.curr_idx = 0

        # global observations, all are low mobility objects
        self.global_observations = []
        

        # objects list
        self.local_map = []
        self.global_map = []

        self.graph = nx.Graph()  # Use undirected graph to manage object relationships
        self.current_relations = set()

        # objects need to be eliminated in local map and graph
        self.to_be_eliminated = set()

        # Local Object Config Init
        LocalObject.initialize_config(cfg)

        # For navigation
        self.nav_graph = None
        self.inquiry = ''
        # If in Click mode, the goal by clicked
        self.click_goal = None
        # If in Inquiry mode, the goal by bbox
        self.global_bbox = None
        self.global_score = 0.0

    def set_click_goal(self, goal):
        self.click_goal = goal

    def set_global_bbox(self, bbox):
        self.global_bbox = bbox

    def set_global_score(self, score):
        self.global_score = score

    def set_global_map(self, global_map):
        self.global_map = global_map

    def has_local_map(self) -> bool:
        return len(self.local_map) > 0

    def set_curr_idx(
        self,
        idx: int,
    ) -> None:
        self.curr_idx = idx
        LocalObject.set_curr_idx(idx)

    def get_global_observations(
        self
    ) -> list:
        return self.global_observations.copy()

    def clear_global_observations(
        self,
    ) -> None:
        self.global_observations.clear()

    # TODO: Using list, finding object by UID requires traversal, which is inefficient
    def get_object(
        self,
        uid
    ) -> LocalObject:
        """Helper function to get an object by UID from the list."""
        for obj in self.local_map:
            if obj.uid == uid:
                return obj

    def set_relation(
        self,
        obj1_uid,
        obj2_uid
    ):
        """Set a relation between two objects based on their UIDs."""
        # Check if both objects exist
        if self.get_object(obj1_uid) and self.get_object(obj2_uid):
            self.graph.add_edge(obj1_uid, obj2_uid)

    def has_relation(
        self,
        obj1_uid,
        obj2_uid
    ) -> bool:
        """Check if two objects have a relation based on their UIDs."""
        return self.graph.has_edge(obj1_uid, obj2_uid)

    def remove_relation(
        self,
        obj1_uid,
        obj2_uid
    ):
        """Remove a relation between two objects based on their UIDs."""
        if self.graph.has_edge(obj1_uid, obj2_uid):
            self.graph.remove_edge(obj1_uid, obj2_uid) 

    def get_related_objects(
        self, 
        obj_uid
    ) -> list:
        """Return a list of UIDs of objects that are related to the given object UID."""
        if obj_uid in self.graph:
            related_uids = list(self.graph.neighbors(obj_uid))
            # Convert these UIDs to LocalObject objects and return
            related_objects = [self.get_object(uid) for uid in related_uids if self.get_object(uid)]
            return related_objects
        else:
            return []  # If the UID is not in the graph, return an empty list

    # Main entry point
    # Update local map using observations
    def process_observations(
        self,
        curr_observations: List[Observation],
    ) -> None:

        # if first, then just insert
        if self.is_initialized == False:
            # Init the local map
            logger.info("[LocalMap] Init Local Map by first observation")

            if len(curr_observations) == 0:
                logger.warning("[LocalMap] No observation in this frame")
                return

            self.init_from_observation(curr_observations)
            self.is_initialized = True
            return

        if len(curr_observations) == 0:
            logger.warning("[LocalMap] No observation in this frame")
            self.update_local_map(curr_observations)
            return

        # if not first, then do the matching
        logger.info("[LocalMap] Matching")
        self.tracker.set_current_frame(curr_observations)

        # Set tracker reference
        self.tracker.set_ref_map(self.local_map)
        self.tracker.matching_map()

        # After matching map, current frame information will be updated
        curr_observations = self.tracker.get_current_frame()

        # Update local map
        self.update_local_map(curr_observations)

    def init_from_observation(
        self,
        observations: List[Observation],
    ) -> list:

        for obs in observations:
            # Init
            local_obj = LocalObject()
            local_obj.add_observation(obs)
            local_obj.update_info()

            self.local_map.append(local_obj)
            self.graph.add_node(local_obj.uid)

    def update_local_map(
        self,
        curr_obs: List[Observation]
    ) -> None:
        # update the local map with the lateset observation
        for obs in curr_obs:
            if obs.matched_obj_idx == -1:
                # Add new local object
                local_obj = LocalObject()
                local_obj.add_observation(obs)
                local_obj.update_info()

                self.local_map.append(local_obj)
                self.graph.add_node(local_obj.uid)
                # logger.info("[LMM] Add new local object!, current local map objs num: ", len(self.local_map))
            else:
                # Update existing local object
                matched_obj = self.local_map[obs.matched_obj_idx]
                matched_obj.add_observation(obs)
                matched_obj.update_info()

        # traverse through the local map
        # split the local obj with the split marker
        # Solve couch + pillow problem
        for obj in self.local_map:
            if obj.should_split:
                self.split_local_object(obj)

        # update the graph and map for insertion and elimination
        self.update_map_and_graph()

        # traverse through the local map
        # 1. check stability and update status
        # 2. do actions based on objects status
        for obj in self.local_map:
            # Update the status of the current local object
            obj.update_status()

            # do actions based on objects status
            self.status_actions(obj)

        # update the graph and map for insertion and elimination
        self.update_map_and_graph()

        logger.info("[LocalMap] Current we have Global Observations num: " + str(len(self.global_observations)))

        if self.cfg.use_rerun:
            self.visualize_local_map()

    def end_process(
        self,
    ) -> None:

        for obj in self.local_map:
            # Update the status of the current local object
            obj.update_status()

            # do actions based on objects status
            self.status_actions(obj)

        # update the graph and map for insertion and elimination
        self.update_map_and_graph()

        if self.cfg.use_rerun:
            self.visualize_local_map()

    def update_map_and_graph(
        self,
    ) -> None:
        # Manage the map and graph
        # 1. Update the graph with current relations
        # 2. Insertion and Elimination of local objects in local_map
        # 3. Insertion and Elimination of nodes in graph

        # 1. Update the graph with current relations FIRST
        if self.current_relations:
            for obj1_uid, obj2_uid in list(self.graph.edges):
                if (obj1_uid, obj2_uid) not in self.current_relations and (
                    obj2_uid, obj1_uid) not in self.current_relations:
                    self.remove_relation(obj1_uid, obj2_uid)

            # clear the current relations
            self.current_relations.clear()

        # 2. Elimination of local objects in local_map and graph
        if self.to_be_eliminated:
            # local map deletion
            self.local_map = [obj for obj in self.local_map if obj.uid not in self.to_be_eliminated]

            # graph deletion
            for uid in self.to_be_eliminated:
                if self.graph.has_node(uid):
                    self.graph.remove_node(uid)

            # clear the to_be_eliminated
            self.to_be_eliminated.clear()

    def status_actions(
        self,
        obj: LocalObject
    ) -> None:
        # do actions based on the object status

        # Set Relations
        # if the object is stable, then no matter in what status,
        # it will search and check all the objs with major plane info in the map
        # to check if it has the on relation
        if obj.is_stable == True:
            # logger.info uid for debug
            # logger.info(f"Checking relations for {obj.uid}")

            # traverse all the other objects and check the relation with the current object
            for other_obj in self.local_map:
                # If the other obj meets the on relation with the current object
                # and not the obj itself
                if other_obj.uid != obj.uid and self.on_relation_check(obj, other_obj):
                    # set the relation in the graph
                    self.set_relation(obj.uid, other_obj.uid)
                    # save the current valid relations
                    self.current_relations.add((min(obj.uid, other_obj.uid), max(obj.uid, other_obj.uid)))
                    # logger.info the relation for debug
                    # logger.info(f"Relation: {obj.uid} - {other_obj.uid}")

        # ELIMINATION Actions
        if obj.status == LocalObjStatus.ELIMINATION:
            self.to_be_eliminated.add(obj.uid)
            return

        # HM_ELIMINATION Actions
        if obj.status == LocalObjStatus.HM_ELIMINATION:

            # if we run local mapping only, we keep the HM/LM_ELIMINATION objects
            if self.cfg.run_local_mapping_only:
                return

            # Get all the related objects in the graph
            related_objs = self.get_related_objects(obj.uid)

            # if no related objects, delete the current object and return
            if len(related_objs) == 0:
                self.to_be_eliminated.add(obj.uid)
                return

            # else return, waiting for the LM actions
            return

        # LM_ELIMINATION Actions
        if obj.status == LocalObjStatus.LM_ELIMINATION:

            # if we run local mapping only, we keep the HM/LM_ELIMINATION objects
            if self.cfg.run_local_mapping_only:
                return

            # Get all the related objects in the graph
            related_objs = self.get_related_objects(obj.uid)

            # if no related objects, delete in local map and ready for global obs
            if len(related_objs) == 0:

                class_name = self.visualizer.obj_classes.get_classes_arr()[obj.class_id]

                # restrict unknown
                if self.cfg.restrict_unknown_labels and class_name == "unknown":
                    self.to_be_eliminated.add(obj.uid)
                    return

                # generate global observation and insert to global obs list
                global_obs = self.create_global_observation(obj)
                self.global_observations.append(global_obs)

                self.to_be_eliminated.add(obj.uid)

                return

            # If it has related objects
            # Check all the related objects status
            is_related_obj_ready = True
            for related_obj in related_objs:
                # if all the related object is not ready, than the LM will wait
                if (related_obj.status == LocalObjStatus.UPDATING or
                   related_obj.status == LocalObjStatus.PENDING
                   ):
                    is_related_obj_ready = False
                    break

            # If all the related object is ready, then generate global observation
            # And delete the current object and all related objects
            if is_related_obj_ready:

                class_name = self.visualizer.obj_classes.get_classes_arr()[obj.class_id]

                # restrict unknown
                if self.cfg.restrict_unknown_labels and class_name == "unknown":
                    self.to_be_eliminated.add(obj.uid)
                    for related_obj in related_objs:
                        self.to_be_eliminated.add(related_obj.uid)
                    return

                # generate global observation and insert to global obs list
                global_obs = self.create_global_observation(obj, related_objs=related_objs)
                self.global_observations.append(global_obs)

                self.to_be_eliminated.add(obj.uid)

                # Delete all the related objs
                for related_obj in related_objs:
                    self.to_be_eliminated.add(related_obj.uid)

            return

    def on_relation_check(
        self,
        base_obj: LocalObject,
        test_obj: LocalObject
    ) -> bool:
        # test whether the test_obj is related to the base_obj (with "on" relation)
        # return True if related, False otherwise

        # If no bbox, return False
        if base_obj.bbox is None or test_obj.bbox is None:
            return False

        # If both objs have no major plane info, return False
        if base_obj.major_plane_info is None and test_obj.major_plane_info is None:
            return False

        # if both objs have major plane info, also return False
        if base_obj.major_plane_info is not None and test_obj.major_plane_info is not None:
            return False

        base_center = base_obj.bbox.get_center()
        test_center = test_obj.bbox.get_center()
        if np.all(base_center == 0) or np.all(test_center == 0):
            return False

        # Here we have one obj has major plane info and the other has no major plane info
        # check which one has major plane info, and set that obj as base_obj
        if base_obj.major_plane_info is None:
            # swap base_obj and test_obj
            base_obj, test_obj = test_obj, base_obj

        base_aabb = base_obj.bbox
        test_aabb = test_obj.bbox

        # Get base_obj and test_obj AABB bounds
        base_min_bound = base_aabb.get_min_bound()  # return [x_min, y_min, z_min]
        base_max_bound = base_aabb.get_max_bound()  # return [x_max, y_max, z_max]

        test_min_bound = test_aabb.get_min_bound()  # return [x_min, y_min, z_min]
        test_max_bound = test_aabb.get_max_bound()  # return [x_max, y_max, z_max]

        # Get AABB xy range
        base_x_min, base_y_min = base_min_bound[0], base_min_bound[1]
        base_x_max, base_y_max = base_max_bound[0], base_max_bound[1]

        test_x_min, test_y_min = test_min_bound[0], test_min_bound[1]
        test_x_max, test_y_max = test_max_bound[0], test_max_bound[1]

        # calculate AABB overlap area in xy plane
        overlap_x = max(0, min(base_x_max, test_x_max) - max(base_x_min, test_x_min))
        overlap_y = max(0, min(base_y_max, test_y_max) - max(base_y_min, test_y_min))

        # Calculate test_obj AABB area size in xy plane
        test_area = (test_x_max - test_x_min) * (test_y_max - test_y_min)

        # get overlap area size
        overlap_area_size = overlap_x * overlap_y

        # calculate the ratio of overlap area to test_obj AABB area size
        # ensure the ratio is a float
        overlap_ratio = overlap_area_size / test_area

        # if the ratio is lower than the threshold, return False
        # TODO: Magic number -> threshold
        if overlap_ratio < 0.8:
            return False

        # Check if the down z value of test_obj is near the base_obj's major plane
        # TODO: Magic number -> threshold of near, currently set as 0.1
        if not (
            test_min_bound[2] - 0.1 <= base_obj.major_plane_info
            and base_obj.major_plane_info <= test_min_bound[2] + 0.2
        ):
            return False

        return True

    def create_global_observation(
        self,
        obj: LocalObject,
        related_objs: List[LocalObject] = []
    ) -> Observation:
        # generate observations for global mapping

        curr_obs = GlobalObservation()

        # set info for global observation
        curr_obs.uid = obj.uid
        curr_obs.class_id = obj.class_id
        curr_obs.pcd = obj.pcd
        curr_obs.bbox = obj.pcd.get_axis_aligned_bounding_box()
        curr_obs.clip_ft = obj.clip_ft
        # TODO: Magic number -> downsample voxel size
        pcd_2d = obj.voxel_downsample_2d(obj.pcd, 0.02)
        curr_obs.pcd_2d = pcd_2d
        curr_obs.bbox_2d = pcd_2d.get_axis_aligned_bounding_box()

        if related_objs:
            for related_obj in related_objs:
                curr_obs.related_objs.append(related_obj.clip_ft)
                # for visualization in rerun
                curr_obs.related_bbox.append(related_obj.bbox)
                curr_obs.related_color.append(related_obj.class_id)
        else:
            curr_obs.related_objs = []
            curr_obs.related_bbox = []
            curr_obs.related_color = []

        return curr_obs 

    def split_local_object(
        self,
        obj: LocalObject,
    ) -> None:
        for class_id, deque in obj.split_info.items():
            new_obj = LocalObject()
            # traverse through the observation in the obj
            for obs in list(obj.observations):
                if obs.class_id == class_id:
                    new_obj.add_observation(obs)
                    # delete the obs in observation
                    # TODO: ALso we actually no need to delete
                    obj.observations = [ob for ob in obj.observations if ob.class_id != class_id or ob.idx != obs.idx]
            # new obj update
            new_obj.update_info_from_observations()

            # add to local map
            self.local_map.append(new_obj)
            # add to graph
            self.graph.add_node(new_obj.uid)

        split_info = obj.print_split_info()
        logger.info("[LocalMap][Split] Split Local Object, splitted info: %s" % split_info)

        # find the obj in local map by using uid
        # remove the obj in the local map list
        self.to_be_eliminated.add(obj.uid)

    def save_map(
        self
    ) -> None:
        # get the directory
        save_dir = self.cfg.map_save_path

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            logger.info(f"[LocalMap] Cleared the directory: {save_dir}")
        os.makedirs(save_dir)

        for i, obj in enumerate(self.local_map):
            if obj.save_path is not None:
                logger.info(f"[LocalMap] Saving No.{i} obj: {obj.save_path}")
                obj.save_to_disk()
            else:
                logger.warning("[LocalMap] No save path for local object")
                continue

    def merge_local_map(
        self
    ) -> None:

        # Use tracker for matching
        self.tracker.set_current_frame(self.local_map)

        # Set tracker reference
        self.tracker.set_ref_map(self.local_map)
        self.tracker.matching_map(is_map_only=True)

        # After matching map, current map information will be updated
        self.merge_info = self.tracker.get_merge_info()

        # Traverse the map, merge based on merge info
        new_local_map = []
        # Traverse merge info
        for indices in self.merge_info:
            # If not merged, just add the object
            if len(indices) == 1:
                new_local_map.append(self.local_map[indices[0]])
            else:
                new_local_map.append(self.merge_local_object([self.local_map[i] for i in indices]))
        # New local map is generated

        logger.info(f"[LocalMap][Merge] Map size before merge: {len(self.local_map)}, after merge: {len(new_local_map)}")

        self.local_map = new_local_map

        if self.cfg.use_rerun:
            self.visualize_local_map()

    def merge_local_object(
        self,
        obj_list: List[LocalObject],
    ) -> LocalObject:
        # This function merges a list of objects
        # Two approaches: 1) Update with all observations, 2) Use object info only
        # Using the first method for code simplicity
        new_obj = LocalObject()
        # Traverse objects in the given list
        for obj in obj_list:
            for obs in obj.observations:
                new_obj.add_observation(obs)

        # Update the info of the new object
        new_obj.update_info_from_observations()
        new_obj.is_merged = True
        return new_obj

    def visualize_local_map(
        self,
    ) -> None:
        # Show all local objects in the local map

        new_logged_entities = set()

        # Temp lists for 3d bbox overlapping drawing
        obj_names = []
        obj_colors = []
        obj_bboxes = []

        for local_obj in self.local_map:
            # Newly added objects are not considered
            if local_obj.observed_num <= 2:
                continue

            obj_name = self.visualizer.obj_classes.get_classes_arr()[local_obj.class_id]

            # Ignore ceiling wall
            if obj_name == "ceiling wall" or obj_name == "carpet" or obj_name == "rug" or obj_name == "ceiling_molding":
                continue

            obj_label = f"{local_obj.observed_num}_{obj_name}"
            obj_label = obj_label.replace(" ", "_")

            base_entity_path = "world/objects"
            entity_path = f"world/objects/{obj_label}"

            positions = np.asarray(local_obj.pcd.points)
            colors = np.asarray(local_obj.pcd.colors) * 255
            colors = colors.astype(np.uint8)
            curr_obj_color = self.visualizer.obj_classes.get_class_color(obj_name)

            obj_names.append(obj_name)
            obj_colors.append(curr_obj_color)

            if self.cfg.show_local_entities:

                # Log pcd data
                rgb_pcd_entity = base_entity_path + "/rgb_pcd" + f"/{local_obj.uid}"
                self.visualizer.log(
                    rgb_pcd_entity,
                    # entity_path + "/pcd",
                    self.visualizer.Points3D(
                        positions,
                        colors=colors,
                        # radii=0.01
                        # labels=[obj_label],
                    ),
                    self.visualizer.AnyValues(
                        uuid=str(local_obj.uid),
                    )
                )

                # Log pcd data
                sem_pcd_entity = base_entity_path + "/sem_pcd" + f"/{local_obj.uid}"
                self.visualizer.log(
                    sem_pcd_entity,
                    # entity_path + "/pcd",
                    self.visualizer.Points3D(
                        positions,
                        colors=curr_obj_color,
                        # radii=0.01
                        # labels=[obj_label],
                    ),
                    self.visualizer.AnyValues(
                        uuid=str(local_obj.uid),
                    )
                )

                target_bbox_entity = None

                if local_obj.nav_goal:
                    bbox = local_obj.bbox
                    centers = [bbox.get_center()]
                    half_sizes = [bbox.get_extent() / 2]
                    target_bbox_entity = base_entity_path + "/bbox_target" + f"/{local_obj.uid}"
                    curr_obj_color = (255, 0, 0)

                    self.visualizer.log(
                        target_bbox_entity,
                        # entity_path + "/bbox",
                        self.visualizer.Boxes3D(
                            centers=centers,
                            half_sizes=half_sizes,
                            # rotations=bbox_quaternion,
                            colors=[curr_obj_color],
                        ),
                        self.visualizer.AnyValues(
                            uuid=str(local_obj.uid),
                        )
                    )

                bbox = local_obj.bbox
                centers = [bbox.get_center()]
                half_sizes = [bbox.get_extent() / 2]
                # Convert rotation matrix to quaternion
                # bbox_quaternion = [self.visualizer.rotation_matrix_to_quaternion(bbox.R)]

                bbox_entity = base_entity_path + "/bbox" + f"/{local_obj.uid}"

                obj_bboxes.append(bbox)

                if local_obj.nav_goal:
                    # Set red
                    curr_obj_color = (255, 0, 0)

                self.visualizer.log(
                    bbox_entity,
                    # entity_path + "/bbox",
                    self.visualizer.Boxes3D(
                        centers=centers,
                        half_sizes=half_sizes,
                        # labels=[f"{obj_label}" + "_" + f"{local_obj.downsample_num}"],
                        labels=[f"{obj_label}"],
                        colors=[curr_obj_color],
                    ),
                    self.visualizer.AnyValues(
                        uuid=str(local_obj.uid),
                    )
                )

            if self.cfg.show_debug_entities:

                self.visualizer.log(
                    "strips",
                    self.visualizer.LineStrips3D(
                        [
                            [
                                [0, 0, 0],
                                [1, 0, 0],
                            ],
                            [
                                [0, 0, 0],
                                [0, 1, 0],
                            ],
                            [
                                [0, 0, 0],
                                [0, 0, 1],
                            ],
                        ],
                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                        radii=[0.025, 0.025, 0.025],
                        labels=["x_w", "y_w", "z_w"],
                    ),
                )

                # Class info
                class_ids = [obs.class_id for obs in local_obj.observations]
                obj_class_id_counter = Counter(class_ids)
                class_labels = ""

                for idx, data in enumerate(obj_class_id_counter.most_common()):
                    class_labels = class_labels + \
                        str(data[0]) + "_(" + str(data[1]) + ")||"

                # Split info
                split_info = ""
                for class_id, idx_deque in local_obj.split_info.items():
                    string_split = f"{class_id}" + "-" + f"{list(idx_deque)}"
                    split_info = split_info + string_split + "||"

                # Debug, should split will be red
                if local_obj.should_split:
                    curr_obj_color = [255, 0, 0]

                if local_obj.status == LocalObjStatus.WAITING:
                    curr_obj_color = [0, 0, 255]

                if local_obj.status == LocalObjStatus.PENDING:
                    curr_obj_color = [255, 0, 0]

                if local_obj.status == LocalObjStatus.HM_ELIMINATION:
                    curr_obj_color = [0, 0, 0]

                if local_obj.status == LocalObjStatus.LM_ELIMINATION:
                    curr_obj_color = [255, 255, 255]

                if local_obj.is_merged:
                    curr_obj_color = [0, 255, 0]

                # Debug for relations
                # Get all related objects in the graph
                related_objs = self.get_related_objects(local_obj.uid)
                related_num = len(related_objs)

                aabb_bbox = local_obj.pcd.get_axis_aligned_bounding_box()
                aabb_centers = [aabb_bbox.get_center()]
                aabb_half_sizes = [aabb_bbox.get_extent() / 2]

                # Baye debug
                # Information from the most recent observation
                latest_obs = local_obj.get_latest_observation()
                lateset_class_id = latest_obs.class_id
                lateset_class_conf = latest_obs.conf
                lateset_class_distance = latest_obs.distance
                # Maximum common of the current object
                prob_class_id = np.argmax(local_obj.class_probs)
                prob_class_conf = local_obj.class_probs[prob_class_id]

                bbox_entity_debug = base_entity_path + "/bbox_debug" + f"/{local_obj.uid}"
                self.visualizer.log(
                    bbox_entity_debug,
                    # entity_path + "/bbox",
                    self.visualizer.Boxes3D(
                        centers=aabb_centers,
                        half_sizes=aabb_half_sizes,
                        labels=[f"{class_labels}"],
                        colors=[curr_obj_color],
                    ),
                    self.visualizer.AnyValues(
                        uuid=str(local_obj.uid),
                        max_num = int(local_obj.max_common),
                        max_class_one = int(local_obj.split_class_id_one),
                        max_class_two = int(local_obj.split_class_id_two),
                        is_stable = bool(local_obj.is_stable),
                        is_low_mobility = bool(local_obj.is_low_mobility),
                        status = str(local_obj.status),
                        status_pending_count = int(local_obj.pending_count),
                        status_waiting_count = int(local_obj.waiting_count),
                        last_seen_idx = int(local_obj.get_latest_observation().idx),
                        related_num = int(related_num),
                        late_class_id = int(lateset_class_id),
                        late_class_conf = float(lateset_class_conf),
                        prob_class_id = int(prob_class_id),
                        prob_class_conf = float(prob_class_conf),
                        lateset_class_distance = float(lateset_class_distance),
                        entropy = float(local_obj.entropy),
                        change_rate = float(local_obj.change_rate),
                        max_prob = float(local_obj.max_prob),
                    )
                )

                # Major plane visualization
                major_plane_entity = base_entity_path + "/major_plane" + f"/{local_obj.uid}"
                if local_obj.is_low_mobility:
                    # Change z axis of half_sizes to 0
                    plane_half_sizes = np.copy(aabb_half_sizes)
                    plane_half_sizes[0][2] = 0
                    # Change plane center z value to the z_value
                    plane_center = np.copy(aabb_centers)
                    plane_center[0][2] = local_obj.major_plane_info

                    self.visualizer.log(
                        major_plane_entity,
                        self.visualizer.Boxes3D(
                            centers=plane_center,
                            half_sizes=plane_half_sizes,
                            fill_mode="solid",
                            colors=[curr_obj_color],
                        )
                    )

                # On relation visualization
                relation_entity = base_entity_path + "/relation" + f"/{local_obj.uid}"
                # Get all related objects in the graph
                related_objs = self.get_related_objects(local_obj.uid)
                if local_obj.is_low_mobility and len(related_objs) > 0:

                    # Get current obj bbox
                    local_obj_center = local_obj.bbox.get_center()

                    # All line information
                    all_lines = []

                    # Traverse all related objects
                    for related_obj in related_objs:

                        # Get the related object's bounding box
                        related_obj_center = related_obj.bbox.get_center()

                        all_lines.append(np.vstack([local_obj_center, related_obj_center]).tolist())

                    self.visualizer.log(
                        relation_entity,
                        self.visualizer.LineStrips3D(
                            all_lines,
                            colors=[[255, 255, 255]] * len(all_lines)
                        ),
                        self.visualizer.AnyValues(
                            relate=str(related_obj_center),
                        )
                    )

            if self.cfg.show_local_entities:
                new_logged_entities.add(rgb_pcd_entity)
                new_logged_entities.add(sem_pcd_entity)
                new_logged_entities.add(bbox_entity)

                if target_bbox_entity is not None:
                    new_logged_entities.add(target_bbox_entity)

            if self.cfg.show_debug_entities:
                new_logged_entities.add(bbox_entity_debug)
                new_logged_entities.add(major_plane_entity)
                new_logged_entities.add(relation_entity)

        local_path_entity = "world/local_path"
        if self.nav_graph is not None and self.nav_graph.pos_path is not None:
            # Create a list of 3D points from the pos_path
            path_points = np.array(self.nav_graph.pos_path)

            # Log the navigation path as a line strip (connecting consecutive points)
            self.visualizer.log(
                local_path_entity,
                self.visualizer.LineStrips3D(
                    [path_points.tolist()],  # Convert the list of points to the required format
                    colors=[[0, 128, 255]]  # Green color for the path
                )
            )
            new_logged_entities.add(local_path_entity)

        if len(self.prev_entities) != 0:
            for entity_path in self.prev_entities:
                if entity_path not in new_logged_entities:
                    # logger.info(f"Clearing {entity_path}")
                    self.visualizer.log(
                        entity_path,
                        self.visualizer.Clear(recursive=True)
                    )
        self.prev_entities = new_logged_entities

        # Visualize 3d bbox overlapping
        if self.cfg.show_3d_bbox_overlapped:
            self.visualizer.visualize_3d_bbox_overlapping(
                obj_names, obj_colors, obj_bboxes
            )

    def calculate_local_path(
        self, curr_pose, goal_mode=GoalMode.RANDOM, resolution=0.03
    ):
        import open3d as o3d

        # Get all pcd from local map
        total_pcd = o3d.geometry.PointCloud()
        for obj in self.local_map:
            if obj.observed_num <= 3:
                continue
            obj_name = self.visualizer.obj_classes.get_classes_arr()[obj.class_id]
            # Ignore ceiling wall
            if obj_name == "ceiling wall" or obj_name == "carpet" or obj_name == "rug" or obj_name == "unknown":
                continue
            total_pcd += obj.pcd

        for obj in self.global_map:
            total_pcd += obj.pcd_2d

        # curr_point_coords = curr_pose[:3, 3]
        # curr_point = o3d.geometry.PointCloud()
        # curr_point.points = o3d.utility.Vector3dVector([curr_point_coords])
        # total_pcd += curr_point

        pcd_points = np.array(total_pcd.points)

        if len(pcd_points) == 0:
            logger.error("[LocalMap][Path] No points in the point cloud!")
            return None

        # Step 1: Constructing 2D occupancy map
        nav_graph = NavigationGraph(self.cfg, total_pcd, resolution)
        self.nav_graph = nav_graph

        nav_graph.get_occ_map()

        # Get start and goal position
        curr_position = curr_pose[:3, 3]
        start_position = nav_graph.calculate_pos_2d(curr_position)

        goal_position = self.get_goal_position(nav_graph, start_position, goal_mode)

        if goal_position is None:
            logger.warning("[LocalMap][Path] No goal position found!")
            return None

        # Step 2: Calculate path
        rrt_path = nav_graph.find_rrt_path(start_position, goal_position)

        if len(rrt_path) == 0:
            logger.warning("[LocalMap][Path] No path found!")
            return None
        else:
            return rrt_path

    def get_goal_position(self, nav_graph, start_position, goal_mode):
        if goal_mode == GoalMode.CLICK:
            logger.info("[LocalMap][Path] Local Goal mode: CLICK")
            return nav_graph.calculate_pos_2d(self.click_goal)

        if goal_mode == GoalMode.INQUIRY:
            logger.info("[LocalMap][Path] Local Goal mode: INQUIRY")
            # Step 1, find local objects within the global best candidate
            candidate_objects = self.filter_objects_in_global_bbox(expand_ratio=0.1)

            if len(candidate_objects) == 0:
                logger.warning("[LocalMap][Path] No local objects found within the global best candidate!")
                return None

            # Step 2. Within filtered objects, find the best score object
            local_goal_candidate, candidate_score = self.find_best_candidate_with_inquiry(candidate_objects)

            # If the score is very far from the global score, return None
            # TODO: Magic number, using given threshold to judge whether the local score is okay or not
            diff_score = abs(candidate_score - self.global_score)
            if diff_score > 0.1:
                logger.warning("[LocalMap][Path] The local score is too far from the global score: ", diff_score)
                return None

            goal_3d = local_goal_candidate.bbox.get_center()
            goal_2d = nav_graph.calculate_pos_2d(goal_3d)

            # TODO: Check whether to use the global map or the local map
            if not nav_graph.free_space_check(goal_2d) is False:

                snapped_goal = self.nav_graph.snap_to_free_space(goal_2d, self.nav_graph.free_space)

                goal_2d = np.array(snapped_goal)

            return goal_2d

    def filter_objects_in_global_bbox(self, expand_ratio=0.1):
        """
        Find local objects that fall within the expanded global_bbox in the xy-plane.

        Parameters:
        - global_bbox: o3d.geometry.AxisAlignedBoundingBox, the global bounding box.
        - local_map: List of objects, each containing an `bbox` attribute of type o3d.geometry.AxisAlignedBoundingBox.
        - expand_ratio: float, the ratio to expand the global_bbox in the xy-plane.

        Returns:
        - candidate_objects: List of local objects whose bboxes fall within the expanded global_bbox in the xy-plane.
        """
        # Get the min and max bounds of the global_bbox
        global_min = np.array(self.global_bbox.min_bound)
        global_max = np.array(self.global_bbox.max_bound)

        # Expand the bbox in the xy-plane
        expand_vector = np.array([(global_max[0] - global_min[0]) * expand_ratio,  # Expand x
                                (global_max[1] - global_min[1]) * expand_ratio,  # Expand y
                                0])  # No expansion in z
        expanded_min_xy = global_min[:2] - expand_vector[:2]
        expanded_max_xy = global_max[:2] + expand_vector[:2]

        # Filter local objects
        candidate_objects = []
        for obj in self.local_map:
            if obj.observed_num <= 2:
                continue
            obj_bbox = obj.bbox

            # Project the object's bbox onto the xy-plane (ignore z)
            obj_min_xy = np.array([obj_bbox.min_bound[0], obj_bbox.min_bound[1]])
            obj_max_xy = np.array([obj_bbox.max_bound[0], obj_bbox.max_bound[1]])

            # Check if the object's bbox intersects with the expanded global bbox in the xy-plane
            if (obj_min_xy[0] <= expanded_max_xy[0] and obj_max_xy[0] >= expanded_min_xy[0] and
                obj_min_xy[1] <= expanded_max_xy[1] and obj_max_xy[1] >= expanded_min_xy[1]):
                candidate_objects.append(obj)
                obj.nav_goal = True

        return candidate_objects

    def find_best_candidate_with_inquiry(self, candidates):
        import torch
        import torch.nn.functional as F

        text_query_ft = self.inquiry

        cos_sim = []

        # Loop through each object in the global map to calculate cosine similarity
        for obj in candidates:
            if obj.observed_num <= 2:
                continue
            obj.nav_goal = False
            obj_feat = torch.from_numpy(obj.clip_ft).to("cuda")
            max_sim = F.cosine_similarity(text_query_ft.unsqueeze(0), obj_feat.unsqueeze(0), dim=-1).item()
            obj_name = self.visualizer.obj_classes.get_classes_arr()[obj.class_id]
            logger.info(f"[LocalMap][Inquiry] =========={obj_name}==============")
            logger.info(f"[LocalMap][Inquiry] Itself: \t{max_sim:.3f}")

            # Store the maximum similarity for this object
            cos_sim.append((obj, max_sim))

        # Now we have a list of tuples [(obj, max_sim), (obj, max_sim), ...]
        # Sort the objects by similarity, from highest to lowest
        sorted_candidates = sorted(cos_sim, key=lambda x: x[1], reverse=True)

        # Get the best candidate (highest cosine similarity)
        best_candidate, best_similarity = sorted_candidates[0]

        # Output the best candidate and its similarity
        best_candidate_name = self.visualizer.obj_classes.get_classes_arr()[best_candidate.class_id]
        logger.info(f"[LocalMap][Inquiry] Best Candidate: '{best_candidate_name}' with similarity: {best_similarity:.3f}")

        # Set flag to the best candidate for visualization
        best_candidate.nav_goal = True

        logger.info(f"[LocalMap][Inquiry] global score: {self.global_score:.3f} ")

        return best_candidate, best_similarity

    def compute_pose_difference(self, curr_pose, prev_pose):
        """
        Calculate the translation and rotation difference between current and previous poses.

        Parameters:
            curr_pose (np.ndarray): Current pose 4x4 homogeneous transformation matrix.
            prev_pose (np.ndarray): Previous frame pose 4x4 homogeneous transformation matrix.

        Returns:
            tuple: (translation difference norm, rotation difference norm)
        """
        if prev_pose is not None:
            # Extract translation vector
            curr_pos = curr_pose[:3, 3]
            prev_pos = prev_pose[:3, 3]

            # Calculate translation difference norm
            delta_translation = np.linalg.norm(curr_pos - prev_pos)

            # Extract rotation matrix
            curr_rot = curr_pose[:3, :3]
            prev_rot = prev_pose[:3, :3]

            # Calculate rotation matrix difference
            delta_rotation_matrix = curr_rot @ prev_rot.T  # Current rotation matrix multiplied by previous frame rotation matrix transpose
            angle = np.arccos(np.clip((np.trace(delta_rotation_matrix) - 1) / 2, -1.0, 1.0))  # Rotation angle (radians)
            delta_rotation = np.degrees(angle)  # Convert to degrees for easier observation

            return delta_translation, delta_rotation
        else:
            return None, None
