"""
Scene-Goal Graph Module

Implements UniGoal-style scene-goal graph representation
with DualMap's online dynamic updates.

Key innovation: Online graph construction and matching, supporting dynamic scenes.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the graph"""
    OBJECT = "object"
    ROOM = "room"
    LOCATION = "location"
    ABSTRACT = "abstract"  # For high-level concepts


class RelationType(Enum):
    """Types of relations between nodes"""
    SPATIAL_NEAR = "near"
    SPATIAL_IN = "in"
    SPATIAL_ON = "on"
    TEMPORAL_BEFORE = "before"
    TEMPORAL_AFTER = "after"
    SEMANTIC_SIMILAR = "similar"
    FUNCTIONAL = "used_for"


@dataclass
class GraphNode:
    """Node in scene or goal graph"""
    id: str
    type: NodeType
    description: str
    attributes: Dict = field(default_factory=dict)
    position: Optional[np.ndarray] = None  # For scene graph
    clip_feature: Optional[np.ndarray] = None
    confidence: float = 1.0
    timestamp: float = 0.0  # For tracking updates


@dataclass
class GraphEdge:
    """Edge in scene or goal graph"""
    source: str
    target: str
    relation: RelationType
    weight: float = 1.0
    attributes: Dict = field(default_factory=dict)


class SceneGraph:
    """
    Online scene graph that updates dynamically.

    Unlike static scene graphs, this one:
    1. Updates in real-time as new objects are observed
    2. Handles object movement (dynamic nodes)
    3. Maintains temporal consistency
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_mapping = {}  # object_id -> node_id
        self.dynamic_nodes = set()  # Nodes that move
        self.update_count = 0

    def add_object(self, obj, timestamp: float = 0.0):
        """Add or update an object in the scene graph"""
        node_id = obj.uid

        # Create node data
        node_data = GraphNode(
            id=node_id,
            type=NodeType.OBJECT,
            description=obj.class_name,
            position=obj.bbox[:3] if hasattr(obj, 'bbox') else None,
            clip_feature=obj.clip_ft if hasattr(obj, 'clip_ft') else None,
            confidence=obj.confidence if hasattr(obj, 'confidence') else 1.0,
            timestamp=timestamp
        )

        if node_id in self.graph:
            # Update existing node
            old_pos = self.graph.nodes[node_id]['data'].position

            # Check if moved (dynamic object)
            if old_pos is not None and node_data.position is not None:
                distance = np.linalg.norm(node_data.position - old_pos)
                if distance > 0.5:  # Moved more than 0.5m
                    self.dynamic_nodes.add(node_id)
                    logger.info(f"Object {obj.class_name} moved {distance:.2f}m - marked as dynamic")

            self.graph.nodes[node_id]['data'] = node_data
        else:
            # Add new node
            self.graph.add_node(node_id, data=node_data)

        self.node_mapping[obj.uid] = node_id
        self.update_count += 1

        # Update edges
        self._update_edges(node_id)

    def _update_edges(self, node_id: str):
        """Update edges for a node based on spatial relations"""
        node_data = self.graph.nodes[node_id]['data']

        if node_data.position is None:
            return

        # Find nearby nodes
        for other_id in self.graph.nodes():
            if other_id == node_id:
                continue

            other_data = self.graph.nodes[other_id]['data']

            if other_data.position is None:
                continue

            # Compute distance
            distance = np.linalg.norm(node_data.position - other_data.position)

            # Add spatial relations
            if distance < 1.0:  # Within 1m
                self._add_or_update_edge(
                    node_id,
                    other_id,
                    RelationType.SPATIAL_NEAR,
                    weight=1.0 / (distance + 0.1)
                )

            # Check "on" relation (vertical stacking)
            if abs(node_data.position[2] - other_data.position[2]) > 0.5:
                if node_data.position[2] > other_data.position[2]:
                    self._add_or_update_edge(
                        node_id,
                        other_id,
                        RelationType.SPATIAL_ON,
                        weight=0.8
                    )

    def _add_or_update_edge(
        self,
        source: str,
        target: str,
        relation: RelationType,
        weight: float
    ):
        """Add or update an edge"""
        if self.graph.has_edge(source, target):
            # Update weight
            edge_data = self.graph[source][target]
            if edge_data['relation'] == relation:
                edge_data['weight'] = weight
        else:
            self.graph.add_edge(
                source,
                target,
                relation=relation,
                weight=weight
            )

    def add_room(self, room_id: str, room_type: str, objects: List):
        """Add a room node and connect objects to it"""
        room_node = GraphNode(
            id=room_id,
            type=NodeType.ROOM,
            description=room_type,
            attributes={'object_count': len(objects)}
        )

        self.graph.add_node(room_id, data=room_node)

        # Connect objects to room
        for obj in objects:
            if obj.uid in self.node_mapping:
                self._add_or_update_edge(
                    self.node_mapping[obj.uid],
                    room_id,
                    RelationType.SPATIAL_IN,
                    weight=1.0
                )

    def query(self, query_text: str, clip_model) -> List[str]:
        """Query the scene graph with natural language"""
        # Encode query
        query_embedding = clip_model.encode_text([query_text])
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy().flatten()
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Score all nodes
        scores = []
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]['data']

            if node_data.clip_feature is not None:
                feature = node_data.clip_feature / np.linalg.norm(node_data.clip_feature)
                similarity = np.dot(query_embedding, feature)
                scores.append((node_id, similarity, node_data))

        # Sort by score
        scores.sort(key=lambda x: -x[1])

        return [node_id for node_id, _, _ in scores[:5]]

    def get_subgraph(self, center_node: str, radius: int = 2) -> nx.DiGraph:
        """Get subgraph around a node (for efficient reasoning)"""
        # BFS to find nodes within radius
        nodes_in_radius = set([center_node])
        frontier = {center_node}

        for _ in range(radius):
            new_frontier = set()
            for node in frontier:
                neighbors = set(self.graph.neighbors(node)) | set(self.graph.predecessors(node))
                new_frontier |= neighbors
            nodes_in_radius |= new_frontier
            frontier = new_frontier

        return self.graph.subgraph(nodes_in_radius)


class GoalGraph:
    """
    Goal graph decomposed from high-level task.

    Example: "Find laptop in bedroom, then bring it to living room"
    Creates a graph with sub-goals and their relations.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.root_goal = None

    def build_from_llm(self, task: str, llm):
        """Build goal graph from task description using LLM"""
        prompt = f"""
        Decompose this navigation task into a goal graph:
        Task: "{task}"

        Output a JSON structure:
        {{
            "nodes": [
                {{"id": "g1", "type": "object/location", "description": "...", "priority": 1}},
                {{"id": "g2", "type": "object/location", "description": "...", "priority": 2}},
                ...
            ],
            "edges": [
                {{"from": "g1", "to": "g2", "relation": "before", "reason": "..."}},
                ...
            ]
        }}

        Guidelines:
        1. Break complex tasks into atomic sub-goals
        2. Specify spatial and temporal relations
        3. Order by priority (what needs to be done first)
        """

        response = llm.query(prompt)
        self._parse_and_build(response)

    def _parse_and_build(self, llm_response):
        """Parse LLM response and build graph"""
        import json

        try:
            data = json.loads(llm_response)

            # Add nodes
            for node_data in data.get('nodes', []):
                node = GraphNode(
                    id=node_data['id'],
                    type=NodeType[node_data['type'].upper()],
                    description=node_data['description'],
                    attributes={'priority': node_data.get('priority', 1)}
                )
                self.graph.add_node(node.id, data=node)

                if node_data.get('priority') == 1:
                    self.root_goal = node.id

            # Add edges
            for edge_data in data.get('edges', []):
                relation = RelationType[edge_data['relation'].upper()]
                self.graph.add_edge(
                    edge_data['from'],
                    edge_data['to'],
                    relation=relation,
                    reason=edge_data.get('reason', '')
                )

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Fallback: create simple single-node goal
            self._create_simple_goal(llm_response)

    def _create_simple_goal(self, task: str):
        """Create a simple single-node goal (fallback)"""
        node = GraphNode(
            id="g1",
            type=NodeType.OBJECT,
            description=task,
            attributes={'priority': 1}
        )
        self.graph.add_node(node.id, data=node)
        self.root_goal = node.id

    def get_next_subgoal(self, completed: Set[str]) -> Optional[str]:
        """Get next sub-goal to pursue"""
        # Find uncompleted goals
        uncompleted = set(self.graph.nodes()) - completed

        if not uncompleted:
            return None

        # Among uncompleted, find those whose predecessors are completed
        ready = []
        for node_id in uncompleted:
            predecessors = set(self.graph.predecessors(node_id))
            if predecessors.issubset(completed):
                priority = self.graph.nodes[node_id]['data'].attributes.get('priority', 999)
                ready.append((node_id, priority))

        if ready:
            # Return highest priority
            ready.sort(key=lambda x: x[1])
            return ready[0][0]
        else:
            # No ready goals (dependency issue), return any uncompleted
            return list(uncompleted)[0]


class SceneGoalMatcher:
    """
    Matches scene graph to goal graph.

    This is the core reasoning component that:
    1. Finds correspondences between scene and goal
    2. Determines navigation strategy
    3. Handles partial matches
    """

    def __init__(self, clip_model):
        self.clip_model = clip_model

    def match(
        self,
        scene_graph: SceneGraph,
        goal_graph: GoalGraph
    ) -> Dict[str, Optional[str]]:
        """
        Match goal nodes to scene nodes.

        Returns:
            Dictionary: goal_node_id -> best_matching_scene_node_id (or None)
        """
        matching = {}

        for goal_node_id in goal_graph.graph.nodes():
            goal_node = goal_graph.graph.nodes[goal_node_id]['data']

            # Find best match in scene
            best_scene_node = self._find_best_match(goal_node, scene_graph)

            matching[goal_node_id] = best_scene_node

            if best_scene_node:
                logger.info(f"Matched goal '{goal_node.description}' to scene node {best_scene_node}")
            else:
                logger.info(f"No match found for goal '{goal_node.description}'")

        return matching

    def _find_best_match(
        self,
        goal_node: GraphNode,
        scene_graph: SceneGraph
    ) -> Optional[str]:
        """Find best matching scene node for a goal node"""
        # Encode goal description
        import torch
        goal_embedding = self.clip_model.encode_text([goal_node.description])
        if isinstance(goal_embedding, torch.Tensor):
            goal_embedding = goal_embedding.cpu().numpy().flatten()
        goal_embedding = goal_embedding / np.linalg.norm(goal_embedding)

        # Score all scene nodes
        best_match = None
        best_score = -1.0

        for scene_node_id in scene_graph.graph.nodes():
            scene_node = scene_graph.graph.nodes[scene_node_id]['data']

            # Only match same type
            if scene_node.type != goal_node.type:
                continue

            # Compute similarity
            if scene_node.clip_feature is not None:
                scene_feature = scene_node.clip_feature / np.linalg.norm(scene_node.clip_feature)
                similarity = np.dot(goal_embedding, scene_feature)

                # Consider confidence
                score = similarity * scene_node.confidence

                if score > best_score:
                    best_score = score
                    best_match = scene_node_id

        # Return match if above threshold
        if best_score > 0.6:  # Threshold
            return best_match
        else:
            return None

    def compute_structural_similarity(
        self,
        goal_subgraph: nx.DiGraph,
        scene_subgraph: nx.DiGraph
    ) -> float:
        """
        Compute structural similarity between subgraphs.

        Uses graph edit distance or graph kernels.
        """
        # Simplified: compare node and edge counts
        goal_nodes = len(goal_subgraph.nodes())
        goal_edges = len(goal_subgraph.edges())
        scene_nodes = len(scene_subgraph.nodes())
        scene_edges = len(scene_subgraph.edges())

        node_sim = 1.0 - abs(goal_nodes - scene_nodes) / max(goal_nodes, scene_nodes)
        edge_sim = 1.0 - abs(goal_edges - scene_edges) / max(goal_edges, scene_edges)

        return 0.5 * node_sim + 0.5 * edge_sim


class OnlineSceneGoalNavigator:
    """
    Main class integrating scene graph, goal graph, and navigation.

    This is designed to work with DualMap's real-time system.
    """

    def __init__(self, clip_model, llm):
        self.scene_graph = SceneGraph()
        self.goal_graph = None
        self.matcher = SceneGoalMatcher(clip_model)
        self.llm = llm

        self.completed_goals = set()
        self.current_matching = {}

    def set_task(self, task: str):
        """Set navigation task"""
        logger.info(f"Setting task: {task}")

        # Build goal graph from task
        self.goal_graph = GoalGraph()
        self.goal_graph.build_from_llm(task, self.llm)

        # Reset completion state
        self.completed_goals = set()
        self.current_matching = {}

    def update_scene(self, observations, timestamp: float):
        """Update scene graph with new observations"""
        for obs in observations:
            self.scene_graph.add_object(obs, timestamp)

    def get_next_navigation_goal(self) -> Optional[Tuple[str, np.ndarray]]:
        """
        Get next navigation goal based on graph matching.

        Returns:
            (goal_description, target_position) or None if task complete
        """
        if self.goal_graph is None:
            return None

        # Get next sub-goal
        next_subgoal_id = self.goal_graph.get_next_subgoal(self.completed_goals)

        if next_subgoal_id is None:
            logger.info("All sub-goals completed!")
            return None

        # Match scene to goal
        self.current_matching = self.matcher.match(self.scene_graph, self.goal_graph)

        # Check if next sub-goal is matched
        matched_scene_node = self.current_matching.get(next_subgoal_id)

        if matched_scene_node:
            # Goal found in scene, navigate to it
            scene_node = self.scene_graph.graph.nodes[matched_scene_node]['data']
            logger.info(f"Goal found: {scene_node.description} at {scene_node.position}")

            return (scene_node.description, scene_node.position)
        else:
            # Goal not found, need to explore
            goal_node = self.goal_graph.graph.nodes[next_subgoal_id]['data']
            logger.info(f"Goal not found, need to explore for: {goal_node.description}")

            return (goal_node.description, None)  # None means explore

    def mark_goal_reached(self, goal_id: str):
        """Mark a sub-goal as reached"""
        self.completed_goals.add(goal_id)
        logger.info(f"Goal {goal_id} completed. Total: {len(self.completed_goals)}")


# Example usage
if __name__ == "__main__":
    print("Scene-Goal Graph Module")
    print("Implements UniGoal-style graph representation with online updates")
