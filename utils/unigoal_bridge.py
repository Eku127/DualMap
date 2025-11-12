"""
DualMap-UniGoal Bridge Module

This module provides the integration layer between DualMap's semantic mapping
system and UniGoal's zero-shot embodied navigation capabilities.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from utils.types import GlobalObject, LocalObject
from utils.navigation_helper import NavigationGraph, LayoutMap


logger = logging.getLogger(__name__)


@dataclass
class SubGoal:
    """Represents a sub-goal in the navigation task"""
    description: str
    target_object: Optional[str] = None
    target_location: Optional[np.ndarray] = None
    completed: bool = False


@dataclass
class SemanticObject:
    """Unified semantic object representation for sharing between systems"""
    uid: str
    class_name: str
    position: np.ndarray  # 3D position (x, y, z)
    bbox: np.ndarray  # Bounding box
    clip_feature: np.ndarray  # CLIP feature vector
    pcd: Optional[np.ndarray] = None  # Point cloud
    confidence: float = 1.0


class DualMapInterface:
    """Interface to access DualMap's mapping data"""

    def __init__(self, dualmap_core):
        """
        Args:
            dualmap_core: Instance of Dualmap class
        """
        self.dualmap = dualmap_core

    def get_global_objects(self) -> List[SemanticObject]:
        """Get all global objects from DualMap"""
        semantic_objects = []

        if self.dualmap.global_map_manager is None:
            logger.warning("Global map manager not initialized")
            return semantic_objects

        global_map = self.dualmap.global_map_manager.global_map

        for uid, global_obj in global_map.items():
            if global_obj.bbox is None or global_obj.clip_ft is None:
                continue

            # Extract center position from bbox
            position = global_obj.bbox[:3]  # Use min corner or compute center

            semantic_obj = SemanticObject(
                uid=uid,
                class_name=global_obj.class_name,
                position=position,
                bbox=global_obj.bbox,
                clip_feature=global_obj.clip_ft,
                pcd=global_obj.pcd if hasattr(global_obj, 'pcd') else None,
                confidence=getattr(global_obj, 'confidence', 1.0)
            )
            semantic_objects.append(semantic_obj)

        logger.info(f"Retrieved {len(semantic_objects)} global objects from DualMap")
        return semantic_objects

    def get_occupancy_map(self) -> Optional[np.ndarray]:
        """Get the occupancy map from DualMap's layout map"""
        if self.dualmap.layout_map is None:
            logger.warning("Layout map not available")
            return None

        # Get the occupancy map from LayoutMap
        occ_map = self.dualmap.layout_map.occupancy_map
        return occ_map

    def get_object_features(self, obj_id: str) -> Optional[np.ndarray]:
        """Get CLIP features for a specific object"""
        if self.dualmap.global_map_manager is None:
            return None

        global_obj = self.dualmap.global_map_manager.global_map.get(obj_id)
        if global_obj is not None:
            return global_obj.clip_ft

        return None

    def query_object_by_text(self, text: str, top_k: int = 1) -> List[Tuple[SemanticObject, float]]:
        """
        Query objects using text description

        Args:
            text: Natural language description
            top_k: Number of top matches to return

        Returns:
            List of (SemanticObject, similarity_score) tuples
        """
        if self.dualmap.object_detector is None:
            logger.warning("Object detector not initialized")
            return []

        # Get text feature using CLIP
        import torch
        text_feature = self.dualmap.object_detector.clip_processor.encode_text([text])

        if isinstance(text_feature, torch.Tensor):
            text_feature = text_feature.cpu().numpy()

        text_feature = text_feature.flatten()

        # Get all global objects
        semantic_objects = self.get_global_objects()

        # Compute similarities
        similarities = []
        for obj in semantic_objects:
            if obj.clip_feature is None:
                continue

            # Cosine similarity
            similarity = np.dot(text_feature, obj.clip_feature) / (
                np.linalg.norm(text_feature) * np.linalg.norm(obj.clip_feature)
            )
            similarities.append((obj, float(similarity)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_navigation_graph(self) -> Optional[nx.Graph]:
        """Get the navigation graph from DualMap"""
        if hasattr(self.dualmap, 'navigation_graph') and self.dualmap.navigation_graph is not None:
            return self.dualmap.navigation_graph.graph
        return None


class UniGoalInterface:
    """Interface to UniGoal's navigation system"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: UniGoal configuration dictionary
        """
        self.config = config
        self.scene_objects = []
        self.occupancy_map = None
        self.llm_backend = config.get('llm_backend', 'ollama')
        self.llm_model = config.get('llm_model', 'llama3')

        logger.info(f"UniGoal interface initialized with LLM backend: {self.llm_backend}")

    def set_scene_objects(self, objects: List[SemanticObject]):
        """Set scene objects for navigation planning"""
        self.scene_objects = objects
        logger.info(f"Updated scene with {len(objects)} objects")

    def set_occupancy_map(self, occ_map: np.ndarray):
        """Set occupancy map for path planning"""
        self.occupancy_map = occ_map
        logger.info(f"Updated occupancy map: shape={occ_map.shape}")

    def decompose_goal(self, goal: str) -> List[SubGoal]:
        """
        Decompose a high-level goal into sub-goals using LLM

        Args:
            goal: Natural language goal description

        Returns:
            List of SubGoal objects
        """
        if not self.config.get('use_llm_decomposition', False):
            # Simple decomposition: treat as single goal
            return [SubGoal(description=goal, target_object=goal)]

        # Use LLM to decompose goal
        sub_goals = self._llm_decompose_goal(goal)
        return sub_goals

    def _llm_decompose_goal(self, goal: str) -> List[SubGoal]:
        """
        Use LLM to decompose goal into sub-goals

        This is a placeholder for actual LLM integration.
        In practice, you would call Ollama or OpenAI API here.
        """
        # Example: Simple rule-based decomposition
        # In production, replace with actual LLM call

        sub_goals = []

        # Check if goal contains multiple steps
        if "then" in goal.lower() or "and then" in goal.lower():
            # Split by "then"
            parts = goal.lower().split("then")
            for i, part in enumerate(parts):
                part = part.strip().replace("and", "").strip()
                sub_goals.append(SubGoal(description=part, target_object=part))
        else:
            # Single step goal
            sub_goals.append(SubGoal(description=goal, target_object=goal))

        logger.info(f"Decomposed goal '{goal}' into {len(sub_goals)} sub-goals")
        return sub_goals

    def plan_path_to_object(self, target_obj: SemanticObject, start_pos: np.ndarray) -> Optional[np.ndarray]:
        """
        Plan a path from start position to target object

        Args:
            target_obj: Target semantic object
            start_pos: Starting position (x, y)

        Returns:
            Path as numpy array of waypoints, or None if planning fails
        """
        if self.occupancy_map is None:
            logger.error("Occupancy map not set")
            return None

        # Use Fast Marching Method or A* for path planning
        # This is a simplified version - integrate with UniGoal's actual planner
        target_pos = target_obj.position[:2]  # Use x, y only

        # Placeholder: return straight line path
        path = np.linspace(start_pos, target_pos, num=10)
        logger.info(f"Planned path from {start_pos} to {target_pos} with {len(path)} waypoints")

        return path

    def get_next_action(self, current_pos: np.ndarray, path: np.ndarray, current_idx: int) -> Dict[str, Any]:
        """
        Get next navigation action

        Args:
            current_pos: Current position (x, y)
            path: Planned path
            current_idx: Current waypoint index

        Returns:
            Action dictionary with 'linear' and 'angular' velocities
        """
        if current_idx >= len(path):
            return {'linear': 0.0, 'angular': 0.0, 'done': True}

        target_waypoint = path[current_idx]

        # Compute direction vector
        direction = target_waypoint - current_pos[:2]
        distance = np.linalg.norm(direction)

        if distance < 0.1:  # Close enough to waypoint
            return {'linear': 0.0, 'angular': 0.0, 'waypoint_reached': True}

        # Simple proportional controller
        linear_vel = min(0.5, distance * 0.5)  # Cap at 0.5 m/s
        angular_vel = 0.0  # Simplified - compute from heading error in practice

        return {'linear': linear_vel, 'angular': angular_vel, 'done': False}


class DualMapUniGoalBridge:
    """
    Main bridge class that integrates DualMap and UniGoal

    This class coordinates between DualMap's semantic mapping and UniGoal's
    navigation planning to enable intelligent embodied navigation.
    """

    def __init__(self, dualmap_core, unigoal_config: Dict[str, Any]):
        """
        Args:
            dualmap_core: Instance of Dualmap class
            unigoal_config: Configuration dictionary for UniGoal
        """
        self.dualmap_interface = DualMapInterface(dualmap_core)
        self.unigoal_interface = UniGoalInterface(unigoal_config)

        self.current_goal = None
        self.current_sub_goals = []
        self.current_path = None
        self.current_waypoint_idx = 0

        logger.info("DualMapUniGoalBridge initialized")

    def set_navigation_goal(self, goal: str) -> bool:
        """
        Set a new navigation goal

        Args:
            goal: Natural language description of the goal

        Returns:
            True if goal was successfully set, False otherwise
        """
        self.current_goal = goal

        # Decompose goal into sub-goals
        self.current_sub_goals = self.unigoal_interface.decompose_goal(goal)

        logger.info(f"Navigation goal set: '{goal}' with {len(self.current_sub_goals)} sub-goals")
        return True

    def update_map_data(self):
        """Update UniGoal with latest map data from DualMap"""
        # Get semantic objects
        objects = self.dualmap_interface.get_global_objects()
        self.unigoal_interface.set_scene_objects(objects)

        # Get occupancy map
        occ_map = self.dualmap_interface.get_occupancy_map()
        if occ_map is not None:
            self.unigoal_interface.set_occupancy_map(occ_map)

        logger.info("Map data updated in UniGoal interface")

    def plan_to_goal(self, sub_goal: SubGoal, start_pos: np.ndarray) -> Optional[np.ndarray]:
        """
        Plan a path to a sub-goal

        Args:
            sub_goal: SubGoal to navigate to
            start_pos: Starting position

        Returns:
            Planned path or None
        """
        # Query DualMap for objects matching the sub-goal
        matches = self.dualmap_interface.query_object_by_text(sub_goal.description, top_k=1)

        if not matches:
            logger.warning(f"No objects found for sub-goal: {sub_goal.description}")
            return None

        target_obj, similarity = matches[0]
        logger.info(f"Found target object '{target_obj.class_name}' with similarity {similarity:.3f}")

        # Plan path to target object
        path = self.unigoal_interface.plan_path_to_object(target_obj, start_pos)

        return path

    def execute_navigation(self, current_pos: np.ndarray, current_heading: float = 0.0) -> Dict[str, Any]:
        """
        Execute one step of navigation

        Args:
            current_pos: Current robot position (x, y, z)
            current_heading: Current robot heading in radians

        Returns:
            Action dictionary with navigation commands
        """
        if not self.current_sub_goals:
            return {'done': True, 'message': 'No active navigation goal'}

        # Update map data
        self.update_map_data()

        # Get current sub-goal
        current_sub_goal = self.current_sub_goals[0]

        # Plan path if not already planned
        if self.current_path is None:
            self.current_path = self.plan_to_goal(current_sub_goal, current_pos[:2])
            self.current_waypoint_idx = 0

            if self.current_path is None:
                return {'done': True, 'message': f'Failed to plan path for: {current_sub_goal.description}'}

        # Get next action
        action = self.unigoal_interface.get_next_action(
            current_pos, self.current_path, self.current_waypoint_idx
        )

        # Check if waypoint reached
        if action.get('waypoint_reached', False):
            self.current_waypoint_idx += 1

        # Check if sub-goal completed
        if action.get('done', False):
            logger.info(f"Sub-goal completed: {current_sub_goal.description}")
            current_sub_goal.completed = True
            self.current_sub_goals.pop(0)
            self.current_path = None

            if not self.current_sub_goals:
                return {'done': True, 'message': 'All sub-goals completed!'}

        return action

    def get_status(self) -> Dict[str, Any]:
        """Get current navigation status"""
        return {
            'goal': self.current_goal,
            'total_sub_goals': len(self.current_sub_goals) + sum(1 for sg in self.current_sub_goals if sg.completed),
            'completed_sub_goals': sum(1 for sg in self.current_sub_goals if sg.completed),
            'current_sub_goal': self.current_sub_goals[0].description if self.current_sub_goals else None,
            'has_path': self.current_path is not None,
            'waypoint_idx': self.current_waypoint_idx,
        }
