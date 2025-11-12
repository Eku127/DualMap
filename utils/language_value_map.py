"""
Language-Grounded Value Map Module

This module implements VLFM-style language-grounded value maps
integrated with DualMap's dual-level architecture.

Key innovation: Combine VLFM's semantic value scoring with DualMap's
real-time dual-level mapping for efficient semantic navigation.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LanguageValue:
    """Language-grounded value for a location/object"""
    position: np.ndarray  # (x, y) or (x, y, z)
    value: float  # 0-1, similarity to goal
    confidence: float  # 0-1, confidence in this value
    source: str  # 'direct_observation' or 'predicted' or 'propagated'


class LanguageGroundedValueMap:
    """
    Computes language-grounded value maps similar to VLFM,
    but integrated with DualMap's dual-level structure.

    Innovation: Unlike VLFM which recomputes values each time,
    we maintain persistent value maps that update incrementally.
    """

    def __init__(self, clip_model, grid_resolution=0.05):
        """
        Args:
            clip_model: CLIP model for encoding text and images
            grid_resolution: Resolution of value map in meters
        """
        self.clip_model = clip_model
        self.grid_resolution = grid_resolution

        # Value maps for different goals (cached)
        self.value_maps = {}  # goal_text -> 2D numpy array

        # Goal embeddings (cached)
        self.goal_embeddings = {}  # goal_text -> CLIP embedding

        # Statistics
        self.update_count = 0

    def compute_goal_embedding(self, goal_text: str) -> np.ndarray:
        """Compute and cache CLIP embedding for goal"""
        if goal_text not in self.goal_embeddings:
            with torch.no_grad():
                embedding = self.clip_model.encode_text([goal_text])
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy().flatten()
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                self.goal_embeddings[goal_text] = embedding

        return self.goal_embeddings[goal_text]

    def compute_value_from_objects(
        self,
        goal_text: str,
        objects: List,
        grid_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Compute language-grounded value map from detected objects.

        This is the core VLFM-style computation, adapted for real-time use.

        Args:
            goal_text: Natural language goal
            objects: List of detected objects with positions and CLIP features
            grid_size: (height, width) of the value map

        Returns:
            2D value map where each cell contains similarity to goal
        """
        goal_embedding = self.compute_goal_embedding(goal_text)

        # Initialize value map
        value_map = np.zeros(grid_size, dtype=np.float32)
        confidence_map = np.zeros(grid_size, dtype=np.float32)

        # For each object, compute its contribution to nearby cells
        for obj in objects:
            if obj.clip_feature is None:
                continue

            # Compute semantic similarity
            obj_feature = obj.clip_feature / np.linalg.norm(obj.clip_feature)
            similarity = np.dot(goal_embedding, obj_feature)
            similarity = (similarity + 1) / 2  # Map from [-1,1] to [0,1]

            # Get object position in grid coordinates
            grid_pos = self.world_to_grid(obj.position, grid_size)

            # Spread value around object position (Gaussian)
            self._add_gaussian_value(
                value_map,
                confidence_map,
                grid_pos,
                similarity,
                obj.confidence if hasattr(obj, 'confidence') else 1.0,
                sigma=5  # cells
            )

        # Normalize by confidence
        value_map = np.divide(
            value_map,
            confidence_map + 1e-6,
            out=value_map,
            where=confidence_map > 0
        )

        return value_map

    def compute_value_with_prediction(
        self,
        goal_text: str,
        current_map,
        frontier_positions: List[np.ndarray]
    ) -> Dict[Tuple[float, float], LanguageValue]:
        """
        Compute value for frontier positions with prediction.

        Innovation: Combine direct observation + predicted content.

        Args:
            goal_text: Navigation goal
            current_map: Current DualMap (local + global)
            frontier_positions: List of frontier positions to evaluate

        Returns:
            Dictionary mapping frontier position to LanguageValue
        """
        goal_embedding = self.compute_goal_embedding(goal_text)
        frontier_values = {}

        for frontier_pos in frontier_positions:
            # Component 1: Value from visible objects
            visible_value = self._compute_visible_value(
                frontier_pos,
                goal_embedding,
                current_map
            )

            # Component 2: Predicted value from unseen regions
            predicted_value = self._compute_predicted_value(
                frontier_pos,
                goal_text,
                current_map
            )

            # Component 3: Spatial prior from layout
            spatial_prior = self._compute_spatial_prior(
                frontier_pos,
                goal_text,
                current_map
            )

            # Combine components
            combined_value = (
                0.5 * visible_value +
                0.3 * predicted_value +
                0.2 * spatial_prior
            )

            frontier_values[tuple(frontier_pos[:2])] = LanguageValue(
                position=frontier_pos,
                value=combined_value,
                confidence=self._compute_confidence(frontier_pos, current_map),
                source='combined'
            )

        return frontier_values

    def _compute_visible_value(
        self,
        position: np.ndarray,
        goal_embedding: np.ndarray,
        current_map
    ) -> float:
        """Compute value based on currently visible objects"""
        # Get nearby objects
        nearby_objects = current_map.get_objects_near(position, radius=2.0)

        if not nearby_objects:
            return 0.0

        # Compute similarities
        similarities = []
        for obj in nearby_objects:
            if obj.clip_feature is not None:
                obj_feature = obj.clip_feature / np.linalg.norm(obj.clip_feature)
                sim = np.dot(goal_embedding, obj_feature)
                sim = (sim + 1) / 2  # [0, 1]
                similarities.append(sim)

        if similarities:
            # Use max similarity (most relevant object)
            return max(similarities)
        else:
            return 0.0

    def _compute_predicted_value(
        self,
        position: np.ndarray,
        goal_text: str,
        current_map
    ) -> float:
        """
        Predict value of unseen regions beyond frontier.

        Innovation: Use context and commonsense to predict unseen content.
        """
        # Get context: nearby objects and room type
        context_objects = current_map.get_objects_near(position, radius=3.0)
        room_type = self._estimate_room_type(context_objects)

        # Use commonsense: some objects are more likely in certain rooms
        goal_lower = goal_text.lower()

        # Simple heuristic (can be replaced with LLM)
        room_priors = {
            'kitchen': {
                'coffee machine': 0.8, 'refrigerator': 0.9, 'stove': 0.9,
                'microwave': 0.7, 'dining table': 0.6
            },
            'bedroom': {
                'bed': 0.95, 'nightstand': 0.8, 'closet': 0.7,
                'desk': 0.5, 'chair': 0.6
            },
            'bathroom': {
                'toilet': 0.95, 'sink': 0.9, 'shower': 0.8, 'mirror': 0.7
            },
            'living_room': {
                'sofa': 0.9, 'tv': 0.8, 'coffee table': 0.7, 'chair': 0.6
            }
        }

        if room_type in room_priors:
            for obj_name, prior in room_priors[room_type].items():
                if obj_name in goal_lower:
                    return prior

        return 0.3  # Default moderate prior

    def _compute_spatial_prior(
        self,
        position: np.ndarray,
        goal_text: str,
        current_map
    ) -> float:
        """Compute spatial prior (e.g., from layout)"""
        # Simple prior: prefer frontiers that lead to larger unexplored areas
        unexplored_area = self._estimate_unexplored_area(position, current_map)

        # Normalize to [0, 1]
        return min(unexplored_area / 20.0, 1.0)  # 20 m² as reference

    def _estimate_room_type(self, objects: List) -> str:
        """Estimate room type from objects"""
        if not objects:
            return 'unknown'

        # Count object categories
        object_types = [obj.class_name.lower() for obj in objects]

        # Simple classification
        kitchen_objects = ['refrigerator', 'stove', 'microwave', 'sink', 'oven']
        bedroom_objects = ['bed', 'nightstand', 'closet', 'dresser']
        bathroom_objects = ['toilet', 'sink', 'shower', 'bathtub']
        living_objects = ['sofa', 'tv', 'couch']

        scores = {
            'kitchen': sum(1 for o in object_types if any(k in o for k in kitchen_objects)),
            'bedroom': sum(1 for o in object_types if any(k in o for k in bedroom_objects)),
            'bathroom': sum(1 for o in object_types if any(k in o for k in bathroom_objects)),
            'living_room': sum(1 for o in object_types if any(k in o for k in living_objects))
        }

        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return 'unknown'

    def _add_gaussian_value(
        self,
        value_map: np.ndarray,
        confidence_map: np.ndarray,
        center: Tuple[int, int],
        value: float,
        confidence: float,
        sigma: float
    ):
        """Add Gaussian-distributed value around a point"""
        h, w = value_map.shape
        y, x = center

        # Create gaussian kernel
        kernel_size = int(3 * sigma)
        y_range = range(max(0, y - kernel_size), min(h, y + kernel_size + 1))
        x_range = range(max(0, x - kernel_size), min(w, x + kernel_size + 1))

        for yi in y_range:
            for xi in x_range:
                dist_sq = (yi - y) ** 2 + (xi - x) ** 2
                weight = np.exp(-dist_sq / (2 * sigma ** 2))

                value_map[yi, xi] += value * weight * confidence
                confidence_map[yi, xi] += weight * confidence

    def world_to_grid(
        self,
        world_pos: np.ndarray,
        grid_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        # Assuming world origin at grid center
        h, w = grid_size
        x = int(world_pos[0] / self.grid_resolution + w / 2)
        y = int(world_pos[1] / self.grid_resolution + h / 2)
        return (max(0, min(h - 1, y)), max(0, min(w - 1, x)))

    def _compute_confidence(self, position: np.ndarray, current_map) -> float:
        """Compute confidence in the value estimate"""
        # Higher confidence if:
        # 1. More observations nearby
        # 2. More recent observations
        # 3. Higher quality observations

        nearby_objects = current_map.get_objects_near(position, radius=2.0)

        if not nearby_objects:
            return 0.3  # Low confidence without observations

        # Confidence from number of objects (more = higher confidence)
        num_confidence = min(len(nearby_objects) / 5.0, 1.0)

        # Average detection confidence
        avg_detection_conf = np.mean([
            obj.confidence if hasattr(obj, 'confidence') else 0.8
            for obj in nearby_objects
        ])

        return 0.5 * num_confidence + 0.5 * avg_detection_conf

    def _estimate_unexplored_area(self, position: np.ndarray, current_map) -> float:
        """Estimate size of unexplored area visible from position"""
        # Simplified: count unknown cells in a cone from position
        # In practice, use raycast or visibility polygon

        # Placeholder implementation
        return 10.0  # m²

    def visualize_value_map(
        self,
        value_map: np.ndarray,
        title: str = "Language Value Map"
    ):
        """Visualize the value map for debugging"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        plt.imshow(value_map, cmap='hot', origin='lower')
        plt.colorbar(label='Language Value (0-1)')
        plt.title(title)
        plt.xlabel('X (grid)')
        plt.ylabel('Y (grid)')
        plt.show()


class HierarchicalLanguageGrounding:
    """
    Extension: Hierarchical language grounding

    Grounds language at multiple levels:
    - Object level: "coffee machine"
    - Room level: "kitchen"
    - Building level: "second floor"
    """

    def __init__(self, clip_model):
        self.clip_model = clip_model
        self.object_grounding = LanguageGroundedValueMap(clip_model)

    def ground_hierarchical_goal(self, goal_text: str, current_map):
        """
        Ground a hierarchical goal like "coffee machine in the kitchen"

        Returns:
            Dictionary with grounded elements at each level
        """
        # Parse hierarchy (simple version, can use LLM)
        grounding = {
            'object': None,
            'room': None,
            'floor': None
        }

        # Extract components
        goal_lower = goal_text.lower()

        # Room indicators
        room_types = ['kitchen', 'bedroom', 'bathroom', 'living room', 'office']
        for room in room_types:
            if room in goal_lower:
                grounding['room'] = room

        # Object (remaining part)
        if grounding['room']:
            # Remove room from goal to get object
            grounding['object'] = goal_lower.replace(grounding['room'], '').strip()
            grounding['object'] = grounding['object'].replace('in', '').replace('the', '').strip()
        else:
            grounding['object'] = goal_lower

        return grounding

    def compute_hierarchical_value(self, goal_text: str, current_map, position: np.ndarray):
        """Compute value considering hierarchy"""
        grounding = self.ground_hierarchical_goal(goal_text, current_map)

        # If room is specified, first check if we're in/near that room
        if grounding['room']:
            room_value = self._compute_room_value(position, grounding['room'], current_map)

            if room_value < 0.5:
                # Not in target room, prioritize finding the room
                return room_value
            else:
                # In target room, now find object
                object_value = self.object_grounding._compute_visible_value(
                    position,
                    self.object_grounding.compute_goal_embedding(grounding['object']),
                    current_map
                )
                return 0.5 * room_value + 0.5 * object_value
        else:
            # No room specified, just find object
            return self.object_grounding._compute_visible_value(
                position,
                self.object_grounding.compute_goal_embedding(grounding['object']),
                current_map
            )

    def _compute_room_value(self, position: np.ndarray, target_room: str, current_map) -> float:
        """Compute how likely the position is in the target room"""
        nearby_objects = current_map.get_objects_near(position, radius=5.0)

        if not nearby_objects:
            return 0.3  # Unknown

        # Estimate current room type
        estimated_room = self.object_grounding._estimate_room_type(nearby_objects)

        if estimated_room == target_room:
            return 0.9  # High confidence we're in target room
        elif estimated_room == 'unknown':
            return 0.5  # Not sure
        else:
            return 0.2  # Probably in wrong room


# Example usage
if __name__ == "__main__":
    # This would be integrated with DualMap
    # See applications/runner_unigoal.py for actual integration

    print("Language-Grounded Value Map Module")
    print("This implements VLFM-style value maps with DualMap integration")
