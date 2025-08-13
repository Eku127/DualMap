import os
import shutil
import json
import pdb
import logging
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d
from omegaconf import DictConfig

from utils.object import GlobalObject
from utils.types import Observation, GoalMode
from utils.base_map_manager import BaseMapManager
from utils.navigation_helper import NavigationGraph, LayoutMap

# Set up the module-level logger
logger = logging.getLogger(__name__)

class GlobalMapManager(BaseMapManager):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__(cfg)

        # global objects list
        self.global_map = []

        # set global flag in tracker
        self.tracker.set_global()

        GlobalObject.initialize_config(cfg)

        # For navigation --> NavigationGraph
        self.nav_graph = None
        self.inquiry = ''
        self.action_path = None
        self.has_action_path = False
        self.lost_and_found = False
        # layout information --> LayoutMap
        layout_resolution = self.cfg.layout_voxel_size * 2
        self.layout_map = LayoutMap(cfg, resolution=layout_resolution, percentile=90, min_area=5, kernel_size=3)

        # pass to the local map manager for inquiry
        self.global_candidate_bbox = None
        self.global_candidate_score = 0.0
        # for lost and found
        # save the previous queried global obj uids
        self.ignore_global_obj_list = []

        self.best_candidate_name = None
        
        self.preload_path_ok = False
        self.vis_preload_wall_ok = False

    def has_global_map(self) -> bool:
        return len(self.global_map) > 0

    def set_layout_info(self, layout_pcd):
        self.layout_map.set_layout_pcd(layout_pcd)

        if self.layout_map.wall_pcd is None:
            self.layout_map.extract_wall_pcd(num_samples_per_grid=10, z_value=self.cfg.floor_height)

    def process_observations(
        self,
        curr_observations: List[Observation]
    ) -> None:

        # for debug, show the preload global map
        if len(self.global_map) > 0 and self.cfg.use_rerun:
            self.visualize_global_map()

        if len(curr_observations) == 0:
            logger.info("[GlobalMap] No global observation update this time, return")
            return

        if self.is_initialized == False:
            # Init the global map
            logger.info("[GlobalMap] Init Global Map by first Local Map input")
            self.global_map = self.init_from_observation(curr_observations)
            self.is_initialized = True
            return

        # The test part, no matching just adding
        if self.cfg.no_update:
            logger.info("[GlobalMap] No update mode, simply adding")
            for obs in curr_observations:
                self.global_map.append(GlobalObject(obs))

            if self.cfg.use_rerun:
                self.visualize_global_map()
            
            return

        # if not the first, then do the global matching
        logger.info("[GlobalMap] Matching")
        self.tracker.set_current_frame(curr_observations)

        # Set tracker reference
        self.tracker.set_ref_map(self.global_map)
        self.tracker.matching_map()

        # After matching map, current frame information will be updated
        curr_observations = self.tracker.get_current_frame()

        # Update global map
        self.update_global_map(curr_observations)
        # visualize the global map


        if self.cfg.use_rerun:
            self.visualize_global_map()

    def init_from_observation(
        self,
        curr_observations: List[Observation]
    ) -> List[GlobalObject]:

        # global_map = []

        # for each local object, generate a global object and add to global_map
        for global_obs in curr_observations:

            global_obj = GlobalObject()
            global_obj.add_observation(global_obs)
            global_obj.update_info()

            self.global_map.append(global_obj)

        return self.global_map

    def update_global_map(
        self,
        curr_observations: List[Observation]
    ) -> None:
        # update the local map with the lateset observation
        for obs in curr_observations:
            if obs.matched_obj_idx == -1:
                # Add new global object
                global_obj = GlobalObject()
                global_obj.add_observation(obs)
                global_obj.update_info()
                self.global_map.append(global_obj)
            else:
                # Update existed global object
                matched_obj_idx = obs.matched_obj_idx
                matched_obj = self.global_map[matched_obj_idx]
                matched_obj.add_observation(obs)
                matched_obj.update_info()

        pass

    def save_map(
        self
    ) -> None:
        # get the directory
        save_dir = self.cfg.map_save_path

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            logger.info(f"[GlobalMap] Cleared the directory: {save_dir}")
        os.makedirs(save_dir)
        for i, obj in enumerate(self.global_map):
            if obj.save_path is not None:
                obj.save_path = obj._initialize_save_path()
                logger.info(f"[GlobalMap] Saving No.{i} obj: {obj.save_path}")
                obj.save_to_disk()
            else:
                logger.info("[GlobalMap] No save path for local object")
                continue

    def load_map(self) -> None:
        """
        Load saved global map objects. If preload path exists, use it first;
        otherwise use default save path. If directory doesn't exist or is empty, do nothing.
        """
        # Use preload_global_map_path first, if not exists then use map_save_path
        if os.path.exists(self.cfg.preload_path):
            load_dir = self.cfg.preload_path
            logger.info(f"[GlobalMap] Using preload global map path: {load_dir}")
        else:
            load_dir = self.cfg.map_save_path
            logger.info(f"[GlobalMap] Preload path not found. Using default map save path: {load_dir}")

        # Check if directory exists
        if not os.path.exists(load_dir):
            logger.warning(f"[GlobalMap] Directory {load_dir} does not exist. Skipping map loading.")
            return

        # Get .pkl files in directory
        pkl_files = [file for file in os.listdir(load_dir) if file.endswith(".pkl")]

        # Skip loading if no .pkl files
        if not pkl_files:
            logger.warning(f"[GlobalMap] No .pkl files found in {load_dir}. Skipping map loading.")
            return

        # Load .pkl files into global map
        for file in pkl_files:
            obj_results_path = os.path.join(load_dir, file)
            loaded_obj = GlobalObject.load_from_disk(obj_results_path)

            import open3d as o3d

            if self.cfg.floor_height:
                # Convert the open3d Vector3dVector to a numpy array
                points = np.asarray(loaded_obj.pcd_2d.points)

                # Now you can modify the Z values (third column)
                points[:, 2] = self.cfg.floor_height

                # After modifying, convert it back to open3d Vector3dVector if needed
                loaded_obj.pcd_2d.points = o3d.utility.Vector3dVector(points)
                loaded_obj.bbox_2d = loaded_obj.pcd_2d.get_axis_aligned_bounding_box()

            self.global_map.append(loaded_obj)

        logger.info(f"[GlobalMap] Successfully preloaded {len(self.global_map)} objects")
        self.is_initialized = True

    def load_wall(self) -> None:
        if os.path.exists(self.cfg.preload_path):
            load_dir = self.cfg.preload_path
            logger.info(f"[GlobalMap] Using preload wall path: {load_dir}")
        else:
            load_dir = self.cfg.map_save_path
            logger.info(f"[GlobalMap] Preload wall path not found. Using default map save path: {load_dir}")

        wall_pcd_path = os.path.join(load_dir, "wall.pcd")

        if not Path(wall_pcd_path).is_file():
            logger.warning(f"[GlobalMap] wall file not found at: {wall_pcd_path}")
            return None

        # load wall pcd
        wall_pcd = o3d.io.read_point_cloud(wall_pcd_path)
        # Convert to numpy array
        points = np.asarray(wall_pcd.points)

        # Set all z values to 22
        points[:, 2] = self.cfg.floor_height

        # Assign the modified points back to the point cloud
        wall_pcd.points = o3d.utility.Vector3dVector(points)
        logger.info(f"[GlobalMap] Wall loaded from: {wall_pcd_path}")

        # Save to class attribute
        self.layout_map.wall_pcd = wall_pcd
    
    def read_json_files(self, directory):
        data_records = {}

        # Get all JSON files in directory
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

        for json_file in json_files:
            file_path = os.path.join(directory, json_file)
            
            # Read JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Store data
            data_records[json_file] = data

        return data_records

    def visualize_global_map(
        self
    ) -> None:
        new_logged_entities = set()

        path_radii = self.cfg.path_radii

        # show the preload wall pcd
        if self.vis_preload_wall_ok is False:
            if self.layout_map.wall_pcd is not None:
                self.visualizer.log(
                    "world/preload_wall",
                    self.visualizer.Points3D(
                        np.asarray(self.layout_map.wall_pcd.points),
                        colors=[(169, 169, 169)],
                        radii = [0.01]
                    )
                )
                self.vis_preload_wall_ok = True


        if self.preload_path_ok is False and self.cfg.use_given_path:
            json_data = self.read_json_files(self.cfg.given_path_dir)
            # traverse all the json data
            for key, value in json_data.items():
                idx = key.split('.')[0]
                path_points = np.array(value)

                path_color = self.cfg.global_path_color

                # if idx == '3':
                #     path_color = (169, 220, 169)
                #     path_radii = 0.02

                # if idx == '5':
                #     path_color = self.cfg.action_path_color

                preload_path_entity = f"world/preload_path/{idx}"

                # Log the navigation path as a line strip (connecting consecutive points)
                self.visualizer.log(
                    preload_path_entity,
                    self.visualizer.LineStrips3D(
                        [path_points.tolist()],  # Convert the list of points to the required format
                        colors=[path_color],  # Green color for the path
                        radii=[path_radii]
                    )
                )
                # new_logged_entities.add(global_path_entity)
            
            self.preload_path_ok = True

        for global_obj in self.global_map:
            base_entity_path = "global/objects"

            obj_name = self.visualizer.obj_classes.get_classes_arr()[global_obj.class_id]
            positions = np.asarray(global_obj.pcd_2d.points)
            colors = np.asarray(global_obj.pcd_2d.colors) * 255
            colors = colors.astype(np.uint8)
            curr_obj_color = self.visualizer.obj_classes.get_class_color(obj_name)

            if global_obj.nav_goal:
                # set red
                curr_obj_color = (255, 0, 0)

            if self.nav_graph is not None and self.nav_graph.pos_path is not None:
                if global_obj.nav_goal:
                    # set red
                    curr_obj_color = (255, 0, 0)
                else:
                    curr_obj_color = (169, 169, 169)

            # Get current related num
            related_num = len(global_obj.related_objs)

            # log pcd data
            rgb_pcd_entity = base_entity_path + "/rgb_pcd" + f"/{global_obj.uid}"
            self.visualizer.log(
                rgb_pcd_entity,
                # entity_path + "/pcd",
                self.visualizer.Points3D(
                    positions,
                    colors=[curr_obj_color],
                    # labels=[obj_label],
                ),
                self.visualizer.AnyValues(
                    uuid=str(global_obj.uid),
                )
            )

            bbox_2d = global_obj.bbox_2d
            centers = [bbox_2d.get_center()]
            half_sizes = [bbox_2d.get_extent() / 2]

            bbox_entity = base_entity_path + "/bbox" + f"/{global_obj.uid}"

            self.visualizer.log(
                bbox_entity,
                # entity_path + "/bbox",
                self.visualizer.Boxes3D(
                    centers=centers,
                    half_sizes=half_sizes,
                    labels=[f"{obj_name}"],
                    colors=[curr_obj_color],
                ),
                self.visualizer.AnyValues(
                    uuid=str(global_obj.uid),
                    related_num=related_num,
                )
            )

            # log related bbox
            s = 0.04
            fixed_half_size = np.array([s, s, s])
            for i, related_bbox in enumerate(global_obj.related_bbox):
                center = related_bbox.get_center()

                center_changed = [center[0], center[1], self.cfg.related_height]

                # centers = [related_bbox.get_center()]
                centers = [center_changed]

                half_sizes = [fixed_half_size]

                # get color
                class_id = global_obj.related_color[i]
                obj_name = self.visualizer.obj_classes.get_classes_arr()[class_id]
                obj_color = self.visualizer.obj_classes.get_class_color(obj_name)

                related_bbox_entity = base_entity_path + "/related_bbox" + f"/{global_obj.uid}_{i}"
                self.visualizer.log(
                    related_bbox_entity,
                    # entity_path + "/bbox",
                    self.visualizer.Boxes3D(
                        centers=centers,
                        half_sizes=half_sizes,
                        colors=[obj_color],
                        fill_mode="solid",
                    ),
                    self.visualizer.AnyValues(
                        uuid=str(global_obj.uid)
                    )
                )

                center_tilted = [center[0], center[1], self.cfg.related_height + 0.1]  # Increase Z axis value
                s = 0.01
                title_half_size = np.array([s, s, s])

                related_title_entity = base_entity_path + "/related_title" + f"/{global_obj.uid}_{i}"
                self.visualizer.log(
                    related_title_entity,
                    # entity_path + "/bbox",
                    self.visualizer.Boxes3D(
                        centers=[center_tilted],
                        half_sizes=title_half_size,
                        colors=[obj_color],
                        labels=[f"{obj_name}"],
                        fill_mode="solid",
                    ),
                    self.visualizer.AnyValues(
                        uuid=str(global_obj.uid)
                    )
                )

                related_line_entity = base_entity_path + "/related_line" + f"/{global_obj.uid}_{i}"

                self.visualizer.log(
                    related_line_entity,
                    # entity_path + "/bbox",
                    self.visualizer.LineStrips3D(
                        [
                            [
                                [center[0], center[1], self.cfg.floor_height],
                                [center[0], center[1], self.cfg.related_height],
                            ],
                        ],
                        colors=[obj_color],
                        radii=[0.01],
                    )
                )

                new_logged_entities.add(related_bbox_entity)
                new_logged_entities.add(related_line_entity)
                new_logged_entities.add(related_title_entity)

            if self.cfg.show_global_map_3d_bbox:
                bbox_3d = global_obj.pcd.get_axis_aligned_bounding_box()
                centers = [bbox_3d.get_center()]
                half_sizes = [bbox_3d.get_extent() / 2]

                bbox_3d_entity = base_entity_path + "/bbox_3d" + f"/{global_obj.uid}"

                self.visualizer.log(
                    bbox_3d_entity,
                    # entity_path + "/bbox",
                    self.visualizer.Boxes3D(
                        centers=centers,
                        half_sizes=half_sizes,
                        colors=[curr_obj_color],
                    ),
                    self.visualizer.AnyValues(
                        uuid=str(global_obj.uid)
                    )
                )
                new_logged_entities.add(bbox_3d_entity)

            new_logged_entities.add(rgb_pcd_entity)
            new_logged_entities.add(bbox_entity)

        # Draw the global path if available
        global_path_entity = "world/global_path"
        if self.nav_graph is not None and self.nav_graph.pos_path is not None:
            # Create a list of 3D points from the pos_path
            path_points = np.array(self.nav_graph.pos_path)

            path_color = self.cfg.global_path_color

            if self.has_action_path:
                path_color = (169, 220, 169)
                path_radii = 0.02

            # Log the navigation path as a line strip (connecting consecutive points)
            self.visualizer.log(
                global_path_entity,
                self.visualizer.LineStrips3D(
                    [path_points.tolist()],  # Convert the list of points to the required format
                    colors=[path_color],  # Green color for the path
                    radii=[path_radii]
                )
            )
            new_logged_entities.add(global_path_entity)

        action_path_entity = "world/action_path"
        if self.action_path is not None:
            # Create a list of 3D points from the pos_path
            path_points = np.array(self.action_path)

            # Log the navigation path as a line strip (connecting consecutive points)
            self.visualizer.log(
                action_path_entity,
                self.visualizer.LineStrips3D(
                    [path_points.tolist()],  # Convert the list of points to the required format
                    colors=[self.cfg.action_path_color],
                    radii=[path_radii]
                )
            )
            new_logged_entities.add(action_path_entity)

        if len(self.prev_entities) != 0:
            for entity_path in self.prev_entities:
                if entity_path not in new_logged_entities:
                    # logger.info(f"Clearing {entity_path}")
                    self.visualizer.log(
                        entity_path,
                        self.visualizer.Clear(recursive=True)
                    )
        self.prev_entities = new_logged_entities

        pass

    def calculate_global_path(
        self, curr_pose, goal_mode=GoalMode.RANDOM, resolution=0.03
    ):
        # calculate global path

        import open3d as o3d

        # Get all pcd from global map
        total_pcd = o3d.geometry.PointCloud()
        for obj in self.global_map:
            total_pcd += obj.pcd_2d

        # Add current pose into the total pcd
        curr_point_coords = curr_pose[:3, 3]
        curr_point = o3d.geometry.PointCloud()
        curr_point.points = o3d.utility.Vector3dVector([curr_point_coords])
        total_pcd += curr_point

        # Add layout wall inforin to the total pcd
        total_pcd += self.layout_map.wall_pcd

        # Step 1: Constucting 2D occupancy map
        nav_graph = NavigationGraph(self.cfg, total_pcd, resolution)
        self.nav_graph = nav_graph
        nav_graph.get_graph()

        # Step 2: Set Start point and goal point
        # transform curr pose to 2d coordinate
        curr_position = curr_pose[:3, 3]
        start_position = nav_graph.calculate_pos_2d(curr_position)

        # Select and process goal based on mode
        goal_position = self.get_goal_position(nav_graph, start_position, goal_mode)

        # Find shortest path
        if goal_position is not None:
            path = nav_graph.find_shortest_path(start_position, goal_position)
            if path:
                logger.info("[GlobalMap][Path] Path successfully generated.")
                return nav_graph.pos_path
            else:
                logger.info("[GlobalMap][Path] Failed to generate a valid path.")
                return None
        else:
            logger.info("[GlobalMap][Path] No valid goal position provided.")
            return None

    def get_goal_position(self, nav_graph, start_position, goal_mode):
        """
        Get the goal position based on the specified mode.

        Parameters:
        - nav_graph: An instance of NavigationGraph.
        - start_position: The starting position in the graph.
        - goal_mode: The mode to select the goal.

        Returns:
        - goal_position: The selected goal position.
        """
        if goal_mode == GoalMode.RANDOM:
            logger.info("[GlobalMap][Path] Goal mode: RANDOM")
            return nav_graph.sample_random_point()

        if goal_mode == GoalMode.CLICK:
            logger.info("[GlobalMap][Path] Goal mode: CLICK")
            return nav_graph.visualize_start_and_select_goal(start_position)

        if goal_mode == GoalMode.INQUIRY:
            logger.info("[GlobalMap][Path] Goal mode: INQUIRY")

            # Step 1, from gloabl map find the best candidate

            global_goal_candidate = None

            global_goal_candidate, score = self.find_best_candidate_with_inquiry()

            # Get center of the best candidate
            goal_3d = global_goal_candidate.bbox_2d.get_center()

            # pass to the local map manager
            self.global_candidate_bbox = global_goal_candidate.bbox_2d
            if self.global_candidate_score == 0.0:
                self.global_candidate_score = score

            # save the queried obj into ignore list
            self.ignore_global_obj_list.append(global_goal_candidate.uid)

            # step 2, using navgraph to switch best candidate into specific goal point
            goal_2d = nav_graph.calculate_pos_2d(goal_3d)
            # check if is in free space, actually the goal is not
            if not nav_graph.free_space_check(goal_2d) is False:
                snapped_goal = nav_graph.snap_to_free_space_directional(goal_2d, start_position, nav_graph.free_space)
                # nearest_node = nav_graph.find_nearest_node(goal_2d)

                if self.cfg.use_directional_path:
                    nearest_node = nav_graph.find_nearest_node(snapped_goal, start_position)
                else:
                    nearest_node = nav_graph.find_nearest_node(goal_2d)

                goal_2d = np.array(nearest_node)

            logger.info(f"[GlobalMap][Path] Nearest node: {goal_2d}")

            return goal_2d

        logger.warning(f"[GlobalMap][Path] Invalid goal mode: {goal_mode}")
        return None

    def find_best_candidate_with_inquiry(self):
        """
        This function finds the best candidate in the global map based on cosine similarity.
        It compares the input query with all objects in the global map and selects the object with the highest similarity.
        """

        import torch
        import torch.nn.functional as F

        text_query_ft = self.inquiry

        cos_sim = []

        obj_list = []   

        # Loop through each object in the global map to calculate cosine similarity
        for obj in self.global_map:

            if obj.uid in self.ignore_global_obj_list:
                obj.nav_goal = False
                continue

            obj.nav_goal = False
            obj_feat = torch.from_numpy(obj.clip_ft).to("cuda")
            max_sim = F.cosine_similarity(text_query_ft.unsqueeze(0), obj_feat.unsqueeze(0), dim=-1).item()
            obj_name = self.visualizer.obj_classes.get_classes_arr()[obj.class_id]
            logger.info(f"[GlobalMap][Inquiry] =========={obj_name}==============")
            logger.info(f"[GlobalMap][Inquiry] Itself: \t{max_sim:.3f}")

            # Check if there are related objects, if so calculate cosine similarity with related_objs
            if obj.related_objs:
                related_sims = []
                for related_obj_ft in obj.related_objs:
                    related_obj_ft_tensor = torch.from_numpy(related_obj_ft).to("cuda")
                    sim = F.cosine_similarity(text_query_ft.unsqueeze(0), related_obj_ft_tensor.unsqueeze(0), dim=-1).item()
                    related_sims.append(sim)
                    logger.info(f"[GlobalMap][Inquiry] Related: \t{sim:.3f}")

                # Update max_sim with the largest similarity from related_objs
                max_sim = max(max_sim, max(related_sims))

            # Store the maximum similarity for this object
            cos_sim.append((obj, max_sim))

        # Now we have a list of tuples [(obj, max_sim), (obj, max_sim), ...]
        # Sort the objects by similarity, from highest to lowest
        sorted_candidates = sorted(cos_sim, key=lambda x: x[1], reverse=True)

        # Get the best candidate (highest cosine similarity)
        best_candidate, best_similarity = sorted_candidates[0]

        if (
            self.global_candidate_score != 0.0
            and abs(best_similarity - self.global_candidate_score) > 0.1
        ):
            # now we find the same ancher object
            # find all the objects with the same name
            for obj in self.global_map:
                if obj.uid in self.ignore_global_obj_list:
                    continue
                obj_name = self.visualizer.obj_classes.get_classes_arr()[obj.class_id]
                if obj_name == self.best_candidate_name:
                    obj_list.append(obj)
        
        if len(obj_list) != 0:
            best_candidate = obj_list[0]

        # Output the best candidate and its similarity
        best_candidate_name = self.visualizer.obj_classes.get_classes_arr()[best_candidate.class_id]

        logger.info(f"[GlobalMap][Inquiry] We ignore {len(self.ignore_global_obj_list)} objects in this global query.")
        logger.info(f"[GlobalMap][Inquiry] Best Candidate: '{best_candidate_name}' with similarity: {best_similarity:.3f}")

        if self.best_candidate_name is None:
            self.best_candidate_name = best_candidate_name

        logger.info(f"[GlobalMap][Inquiry] Memory Best Candidate '{self.best_candidate_name}'")
        print(f"[GlobalMap][Inquiry] Memory Best Candidate '{self.best_candidate_name}'")

        # Set flag to the best candidate for visualization
        best_candidate.nav_goal = True

        # input("Press any key to continue...")

        return best_candidate, best_similarity
