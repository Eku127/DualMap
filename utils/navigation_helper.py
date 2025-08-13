import cv2
import random

import numpy as np
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, KDTree
from scipy.ndimage import binary_erosion
from scipy.spatial import KDTree
from omegaconf import DictConfig

class LayoutMap:
    def __init__(self, cfg, resolution=0.1, percentile=90, min_area=5, kernel_size=3):
        """
        Initialize the LayoutMap class.

        Args:
            resolution: Size of each grid cell (in meters).
            percentile: Percentile threshold for binarization.
            min_area: Minimum area for removing small connected components.
            kernel_size: Kernel size for morphological operations.
        """
        self.cfg = cfg
        self.resolution = resolution
        self.percentile = percentile
        self.min_area = min_area
        self.kernel_size = kernel_size

        self.point_cloud = None
        self.occ_map = None
        self.x_edges = None
        self.y_edges = None
        self.wall_pcd = None  # Store extracted wall point cloud

    def set_layout_pcd(self, layout_pcd):
        """
        Load point cloud and generate Occupancy Map.

        Args:
            layout_pcd: Point cloud data.
        """
        self.point_cloud = layout_pcd
        self.occ_map, self.x_edges, self.y_edges = self.create_occupancy_map()
        print("Occupancy Map created.")

    def create_occupancy_map(self):
        """
        Create Occupancy Map from point cloud data.
        """
        points = np.asarray(self.point_cloud.points)
        xy_points = points[:, :2]
        x_min, y_min = np.min(xy_points, axis=0)
        x_max, y_max = np.max(xy_points, axis=0)

        occ_map, x_edges, y_edges = np.histogram2d(
            xy_points[:, 0],
            xy_points[:, 1],
            bins=(int((x_max - x_min) / self.resolution), int((y_max - y_min) / self.resolution))
        )
        return occ_map, x_edges, y_edges

    def calculate_threshold(self, method="percentile"):
        """
        Calculate threshold based on Occupancy Map.

        Args:
            method: Threshold calculation method, options: "mean", "median", or "percentile".
        """
        non_zero_values = self.occ_map[self.occ_map > 0]
        if method == "mean":
            return np.mean(non_zero_values)
        elif method == "median":
            return np.median(non_zero_values)
        elif method == "percentile":
            return np.percentile(non_zero_values, self.percentile)
        else:
            raise ValueError("Unsupported threshold calculation method.")

    def process_binary_map(self):
        """
        Process binary map with connected component filtering and morphological operations.
        """
        # Binarization
        threshold = self.calculate_threshold()
        binary_map = (self.occ_map > threshold).astype(np.uint8)

        # Remove small connected components
        cleaned_map = self.remove_small_components(binary_map)

        # Morphological operation (closing)
        processed_map = self.apply_morphological_operations(cleaned_map)

        if self.cfg.edit_wall:
            processed_map = self.visualize_and_edit_map(processed_map)

        return processed_map

    def visualize_and_edit_map(self, processed_map):
        edited_map = processed_map.copy()
        cell_size = 3  # Adjust the size of each cell for better visualization
        drawing = False
        
        # Function to handle mouse events
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing
            grid_x = x // cell_size
            grid_y = y // cell_size
            if 0 <= grid_x < edited_map.shape[1] and 0 <= grid_y < edited_map.shape[0]:
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    edited_map[grid_y, grid_x] = 1 - edited_map[grid_y, grid_x]
                    update_display()
                elif event == cv2.EVENT_MOUSEMOVE and drawing:
                    if edited_map[grid_y, grid_x] == 0:
                        edited_map[grid_y, grid_x] = 1
                    update_display()
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False

        # Function to update display after modification
        def update_display():
            display_map = cv2.resize(edited_map * 255, (edited_map.shape[1] * cell_size, edited_map.shape[0] * cell_size), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Editable Map", display_map)
        
        # Create window and set mouse callback
        cv2.namedWindow("Editable Map")
        cv2.setMouseCallback("Editable Map", mouse_callback)
        update_display()

        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):  # Press 'q' to finish
                break

        cv2.destroyAllWindows()
        return edited_map

    def remove_small_components(self, binary_map):
        """
        Remove small connected components.
        """
        cleaned_map = np.zeros_like(binary_map, dtype=np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                cleaned_map[labels == label] = 1

        return cleaned_map

    def apply_morphological_operations(self, binary_map):
        """
        Apply morphological operations to the binary image.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
        return cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)

    def convert_binary_map_to_3d_points(self, binary_map, num_samples_per_grid=10, z_value=0.0):
        """
        Convert wall grid cells in the binary map to 3D point cloud.
        """
        wall_points_3d = []
        x_centers = (self.x_edges[:-1] + self.x_edges[1:]) / 2
        y_centers = (self.y_edges[:-1] + self.y_edges[1:]) / 2

        for i in range(binary_map.shape[0]):
            for j in range(binary_map.shape[1]):
                if binary_map[i, j] == 1:
                    x_samples = np.random.uniform(
                        self.x_edges[i], self.x_edges[i + 1], num_samples_per_grid
                    )
                    y_samples = np.random.uniform(
                        self.y_edges[j], self.y_edges[j + 1], num_samples_per_grid
                    )
                    z_samples = np.full_like(x_samples, z_value)
                    grid_points = np.stack((x_samples, y_samples, z_samples), axis=1)
                    wall_points_3d.append(grid_points)

        if wall_points_3d:
            wall_points_3d = np.vstack(wall_points_3d)
        else:
            wall_points_3d = np.empty((0, 3))

        return wall_points_3d

    def extract_wall_pcd(self, num_samples_per_grid=10, z_value=0.0):
        """
        Extract wall points and save to self.wall_pcd.
        """
        binary_map = self.process_binary_map()
        wall_points = self.convert_binary_map_to_3d_points(
            binary_map, num_samples_per_grid=num_samples_per_grid, z_value=self.cfg.floor_height
        )
        self.wall_pcd = o3d.geometry.PointCloud()
        self.wall_pcd.points = o3d.utility.Vector3dVector(wall_points)
        # self.visualize_wall_pcd()
        save_dir = self.cfg.map_save_path
        layout_pcd_path = save_dir + "/wall.pcd"
        self.save_wall_pcd(layout_pcd_path)

        print(f"Extracted wall point cloud with {len(self.wall_pcd.points)} points.")
    
    def save_wall_pcd(self, output_path="wall_points.pcd"):
        """
        Save wall point cloud to disk.
        """
        if self.wall_pcd is None or len(self.wall_pcd.points) == 0:
            print("No wall points to save.")
            return
        o3d.io.write_point_cloud(output_path, self.wall_pcd)
        print(f"Wall point cloud saved to {output_path}")

    def visualize_wall_pcd(self):
        """
        Visualize wall point cloud.
        """
        if self.wall_pcd is None or len(self.wall_pcd.points) == 0:
            print("No wall points to visualize.")
            return
        o3d.visualization.draw_geometries([self.wall_pcd])

class RRT:
    def __init__(self, algorithm="rrt", max_iter=1000, steer_length=5, search_radius=10, goal_sample_rate=0.1):
        """
        Initialize the RRT planner with different algorithm choices.

        :param occupancy_grid_map: 2D numpy array representing the occupancy grid map (1 for free, 0 for occupied).
        :param start: Tuple (x, y) representing the start point.
        :param goal: Tuple (x, y) representing the goal point.
        :param algorithm: The algorithm to use: "rrt", "rrt_sharp", or "rrt_star".
        :param max_iter: Maximum number of iterations.
        :param steer_length: Step size for tree expansion.
        :param search_radius: Radius to rewire nearby nodes (only for RRT* and RRT-Sharp).
        :param goal_sample_rate: Probability of sampling near the goal.
        """

        self.algorithm = algorithm.lower()
        self.max_iter = max_iter
        self.steer_length = steer_length
        self.search_radius = search_radius
        self.goal_sample_rate = goal_sample_rate

    def set_occ_map(self, occupancy_grid_map):
        """Set the occupancy grid map."""
        self.occupancy_grid_map = occupancy_grid_map
    
    def set_start_goal(self, start, goal):
        """Set the start point."""
        self.start = start
        self.goal = goal

        self.tree_nodes = [start]
        self.tree_parents = {tuple(start): None}
        self.tree_costs = {tuple(start): 0}
        self.tree_heuristics = {tuple(start): np.linalg.norm(np.array(start) - np.array(goal))}
        self.kdtree = KDTree(self.tree_nodes)


    def is_free(self, x, y):
        """Check if the given point is free (not occupied)."""
        return 0 <= x < self.occupancy_grid_map.shape[1] and 0 <= y < self.occupancy_grid_map.shape[0] and self.occupancy_grid_map[int(y), int(x)] == 1

    def steer(self, p1, p2):
        """Steer from point p1 to point p2 with a defined step length."""
        diff = np.array(p2) - np.array(p1)
        dist = np.linalg.norm(diff)
        if dist <= self.steer_length:
            return tuple(p2)
        direction = diff / dist
        return tuple(np.array(p1) + direction * self.steer_length)

    def rewire(self, new_node, search_radius=10):
        """
        Rewires nearby nodes to improve the path by checking if any nearby node
        can be reconnected to the new node for a lower cost.

        :param new_node: The new node that was added to the tree.
        :param search_radius: The radius within which to rewire nearby nodes.
        """
        # Query nearby nodes using KDTree
        indices = self.kdtree.query_ball_point(new_node, search_radius)
        
        # For each nearby node, check if the path cost can be improved by connecting to new_node
        for idx in indices:
            node = self.tree_nodes[idx]
            
            # Check if the node is already in the tree_costs (this should always be the case)
            if node not in self.tree_costs:
                self.tree_costs[node] = float('inf')  # Initialize with a very high cost
            
            potential_cost = self.tree_costs[new_node] + np.linalg.norm(np.array(new_node) - np.array(node))
            
            # If the potential cost is lower than the current cost, update the parent and cost
            if potential_cost < self.tree_costs[node]:
                self.tree_parents[node] = new_node
                self.tree_costs[node] = potential_cost

    def heuristic_sampling(self):
        """Sample near the goal with a higher probability."""
        if random.random() < self.goal_sample_rate:
            offset = np.random.normal(scale=5, size=2)
            return tuple(np.array(self.goal) + offset)
        else:
            return (random.uniform(0, self.occupancy_grid_map.shape[1]), random.uniform(0, self.occupancy_grid_map.shape[0]))

    def plan(self):
        """Execute the chosen RRT algorithm."""
        if self.algorithm == "rrt":
            return self.rrt_plan()
        elif self.algorithm == "rrt_sharp":
            return self.rrt_sharp_plan()
        elif self.algorithm == "rrt_star":
            return self.rrt_star_plan()
        else:
            raise ValueError(f"Unknown algorithm {self.algorithm}. Choose 'rrt', 'rrt_sharp', or 'rrt_star'.")

    def rrt_plan(self):
        """Standard RRT path planning."""
        for _ in range(self.max_iter):
            rand_point = self.heuristic_sampling()
            _, nearest_idx = self.kdtree.query(rand_point)
            nearest_node = self.tree_nodes[nearest_idx]

            new_node = self.steer(nearest_node, rand_point)
            if self.is_free(*new_node):
                self.tree_nodes.append(new_node)
                self.tree_parents[new_node] = nearest_node
                self.kdtree = KDTree(self.tree_nodes)

                # Check if the goal is reached
                if np.linalg.norm(np.array(new_node) - np.array(self.goal)) <= self.steer_length:
                    self.tree_nodes.append(self.goal)
                    self.tree_parents[self.goal] = new_node
                    break

        return self._reconstruct_path()

    def rrt_sharp_plan(self):
        """RRT-Sharp path planning with rewire and cost optimization."""
        for _ in range(self.max_iter):
            rand_point = self.heuristic_sampling()
            _, nearest_idx = self.kdtree.query(rand_point)
            nearest_node = self.tree_nodes[nearest_idx]

            new_node = self.steer(nearest_node, rand_point)
            if self.is_free(*new_node):
                cost = self.tree_costs[nearest_node] + np.linalg.norm(np.array(nearest_node) - np.array(new_node))
                self.tree_nodes.append(new_node)
                self.tree_parents[new_node] = nearest_node
                self.tree_costs[new_node] = cost
                self.kdtree = KDTree(self.tree_nodes)

                # Rewire nearby nodes
                self.rewire(new_node)

                # Check if the goal is reached
                if np.linalg.norm(np.array(new_node) - np.array(self.goal)) <= self.steer_length:
                    self.tree_nodes.append(self.goal)
                    self.tree_parents[self.goal] = new_node
                    break

        return self._reconstruct_path()

    def rrt_star_plan(self):
        """RRT* path planning with optimal rewiring."""
        for _ in range(self.max_iter):
            rand_point = self.heuristic_sampling()
            _, nearest_idx = self.kdtree.query(rand_point)
            nearest_node = self.tree_nodes[nearest_idx]

            new_node = self.steer(nearest_node, rand_point)
            if self.is_free(*new_node):
                self.tree_nodes.append(new_node)
                self.tree_parents[new_node] = nearest_node
                self.kdtree = KDTree(self.tree_nodes)

                # Rewire nearby nodes
                self.rewire(new_node)

                # Check if the goal is reached
                if np.linalg.norm(np.array(new_node) - np.array(self.goal)) <= self.steer_length:
                    self.tree_nodes.append(self.goal)
                    self.tree_parents[self.goal] = new_node
                    break

        return self._reconstruct_path()

    def _reconstruct_path(self):
        """Reconstruct the path from the goal to the start."""
        path = []
        current = tuple(self.goal)
        while current is not None:
            path.append(current)
            current = self.tree_parents.get(current)

        return path[::-1] if path and path[-1] == tuple(self.start) else []

    @staticmethod
    def visualize_occupancy_map(occupancy_grid_map, path=None):
        """
        Visualize the occupancy grid map and the planned path using matplotlib.

        :param occupancy_grid_map: 2D numpy array representing the occupancy grid map (1 for free, 0 for occupied).
        :param path: List of points representing the path [(x1, y1), (x2, y2), ...].
        """
        plt.figure(figsize=(8, 8))

        # Draw the occupancy grid map
        plt.imshow(occupancy_grid_map, cmap="Greys", origin="lower")

        # Draw the path if provided
        if path:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], color='blue', linewidth=2, marker='o')

        plt.title("Occupancy Grid Map and Path")
        plt.xlabel("X Cells")
        plt.ylabel("Y Cells")
        plt.grid(True)
        plt.show()

class NavigationGraph:
    def __init__(self, cfg: DictConfig, pcd: o3d.geometry.PointCloud, cell_size: int):
        """Initialization of the NavigationGraph class.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud of the floor.
            cell_size (int): the resolution of the cell (m/cell)
        """
        self.cfg = cfg

        self.pcd_min = np.min(np.array(pcd.points), axis=0)
        self.pcd_max = np.max(np.array(pcd.points), axis=0)
        self.grid_size = np.ceil((self.pcd_max - self.pcd_min) / cell_size + 1).astype(
            np.int32
        )
        self.grid_size = self.grid_size[[0, 1]]
        print(self.grid_size)
        self.cell_size = cell_size
        self.pcd = pcd

        self.pos_path=None

        self.snapped_goal = None

        self.rrt = RRT(
            algorithm="rrt_sharp",
            max_iter=500,
            steer_length=5,
            search_radius=5,
            goal_sample_rate=0.2,
        )

    # here we only use voronoi graph now
    def get_graph(self):
        free_space = self.get_occupancy_map()
        voronoi = self.get_voronoi_graph()
        self.graph = voronoi

    def get_occ_map(self):
        free_space = self.get_occupancy_map()
        # self.visualize_occupancy_map(free_space)

    def get_occupancy_map(self):
        # Initialize an empty grid map with all cells marked as unoccupied (0)
        occupancy_grid_map = np.zeros(self.grid_size[::-1], dtype=int)

        # Handle point cloud with negative values
        point_cloud = np.asarray(self.pcd.points)  # Get the point cloud as a numpy array

        # Adjust only x and y coordinates to positive values
        point_cloud[:, 0] -= self.pcd_min[0]  # Adjust x-coordinate
        point_cloud[:, 1] -= self.pcd_min[1]  # Adjust y-coordinate

        # Iterate through the point cloud and mark the corresponding cells as occupied (1)
        x_cells = np.floor(point_cloud[:, 0] / self.cell_size).astype(int)
        y_cells = np.floor(point_cloud[:, 1] / self.cell_size).astype(int)

        # Mark occupied cells
        occupancy_grid_map[y_cells, x_cells] = 1

        # self.visualize_occupancy_map(occupancy_grid_map)

        # Make the occupancy places with a dilation operation
        dilation_radius = 10

        if dilation_radius > 0:
            occupancy_grid_map = cv2.dilate(
                occupancy_grid_map.astype(np.uint8),
                np.ones((dilation_radius, dilation_radius)),
                iterations=1,
            )

        # self.visualize_occupancy_map(occupancy_grid_map)

        self.occupancy_grid_map = occupancy_grid_map

        # get largest free space
        # Now, we need to find the largest free space (value 0) using connected components
        # Invert the map so that free space is marked with 1 and occupied space with 0
        free_space_map = (occupancy_grid_map == 0).astype(np.uint8)

        # Find all connected components (regions of free space)
        num_labels, labels = cv2.connectedComponents(free_space_map)

        # Find the largest connected component (ignore the background label 0)
        largest_component = 0
        largest_size = 0

        for label in range(1, num_labels):  # Start from 1, as 0 is the background
            component_size = np.sum(labels == label)
            if component_size > largest_size:
                largest_size = component_size
                largest_component = label

        # Create a mask for the largest component
        largest_component_mask = (labels == largest_component).astype(np.uint8)

        # Optionally, visualize the largest free space
        # self.visualize_occupancy_map(largest_component_mask)

        self.free_space = largest_component_mask

        return largest_component_mask

    def is_in_bounds(self, point):
        x, y = point
        return (0 <= x < self.free_space.shape[0]) and (0 <= y < self.free_space.shape[1])

    def visualize_occupancy_map(self, occupancy_grid_map):
        """Visualize the occupancy grid map using matplotlib."""
        plt.figure(figsize=(8, 8))
        # Use cmap='gray' instead of 'gray_r'
        plt.imshow(occupancy_grid_map, cmap="gray", origin="lower", interpolation="nearest")
        plt.colorbar(label="Occupancy")
        plt.title("Occupancy Grid Map")
        plt.xlabel("X Cells")
        plt.ylabel("Y Cells")
        plt.show()

    def visualize_occupancy_map_with_point(self, occupancy_grid_map, start=None, end=None):
        """
        Visualize the occupancy grid map using matplotlib and optionally visualize the start and end points.

        Parameters:
        - occupancy_grid_map: 2D numpy array representing the occupancy grid map (1 for occupied, 0 for free).
        - start: Tuple (x, y) representing the start point in grid coordinates.
        - end: Tuple (x, y) representing the end point in grid coordinates.
        """
        plt.figure(figsize=(8, 8))

        # Display the occupancy grid map
        plt.imshow(occupancy_grid_map, cmap="gray", origin="lower", interpolation="nearest")

        # Add the start point if provided
        if start is not None:
            # Check if the start point is within the grid bounds
            if (0 <= start[0] < occupancy_grid_map.shape[1]) and (0 <= start[1] < occupancy_grid_map.shape[0]):
                plt.scatter(start[0], start[1], color='blue', label="Start", s=50, marker='o')

        # Add the end point if provided and if it is within the grid bounds
        if end is not None:
            # Check if the end point is within the grid bounds
            if (0 <= end[0] < occupancy_grid_map.shape[1]) and (0 <= end[1] < occupancy_grid_map.shape[0]):
                plt.scatter(end[0], end[1], color='red', label="End", s=50, marker='x')

        # Add a colorbar, title, and labels
        plt.colorbar(label="Occupancy")
        plt.title("Occupancy Grid Map")
        plt.xlabel("X Cells")
        plt.ylabel("Y Cells")

        # Display the legend if there are start/end points
        if start is not None or end is not None:
            plt.legend()

        # Show the plot
        plt.show()

    def snap_to_free_space(self, point, free_space_map):
        """
        Snap a given 2D point to the nearest free space if it falls on an obstacle.

        Parameters:
        - point: Tuple (row, col) representing the 2D point (y, x).
        - free_space_map: 2D numpy array where 1 indicates free space, 0 indicates obstacle.

        Returns:
        - Tuple (row, col): The nearest free space point.
        """
        y, x = int(point[0]), int(point[1])  # Unpack input point as (row, col)

        # Boundary check
        if not (0 <= y < free_space_map.shape[0] and 0 <= x < free_space_map.shape[1]):
            print(f"Point ({y}, {x}) is out of bounds!")
            return None

        # If the point is already in free space, return directly
        if free_space_map[y, x] == 1:
            return (y, x)

        # Get all free space points coordinates
        free_space_indices = np.argwhere(free_space_map == 1)

        # Calculate Euclidean distances to all free space points
        distances = np.linalg.norm(free_space_indices - np.array([y, x]), axis=1)

        # Find the point with the minimum distance
        nearest_idx = np.argmin(distances)
        nearest_point = free_space_indices[nearest_idx]

        return tuple(nearest_point)  # Return the nearest free space point (row, col)
    
    def snap_to_free_space_directional(self, point, start_point, free_space_map, search_radius=50):
        """
        Snap from the target point to the nearest free space along the direction towards the start point.

        Args:
            point: Tuple (row, col), target point (y, x).
            start_point: Tuple (row, col), start point (y, x).
            free_space_map: 2D numpy array, free space=1, obstacle=0.
            search_radius: int, number of steps to search along the direction (default 10).

        Returns:
            Tuple (row, col): Snapped free space point.
        """
        y, x = int(point[0]), int(point[1])

        # Boundary check
        if not (0 <= y < free_space_map.shape[0] and 0 <= x < free_space_map.shape[1]):
            print(f"Point ({y}, {x}) is out of bounds!")
            return None

        # If the target point is already in free space, return directly
        if free_space_map[y, x] == 1:
            return (y, x)

        # Calculate direction vector from target point to start point (normalized)
        direction_vector = np.array(start_point) - np.array(point)
        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            direction_vector = np.array([0, 0])
        else:
            direction_vector = direction_vector / norm

        # Search from target point along direction vector
        for step in range(1, search_radius + 1):
            offset = np.round(direction_vector * step).astype(int)
            new_y, new_x = y + offset[0], x + offset[1]

            # Boundary check
            if 0 <= new_y < free_space_map.shape[0] and 0 <= new_x < free_space_map.shape[1]:
                if free_space_map[new_y, new_x] == 1:
                    return (new_y, new_x)

        # Search along direction failed -> global snap
        free_space_indices = np.argwhere(free_space_map == 1)
        distances = np.linalg.norm(free_space_indices - np.array([y, x]), axis=1)
        nearest_idx = np.argmin(distances)
        nearest_point = free_space_indices[nearest_idx]

        return tuple(nearest_point)

    def calculate_pos_3d(self, row, col):
        """Helper function to calculate the position in the 3D space"""
        ## Z is defined as 0.0
        z = self.cfg.robot_height
        return (
            col * self.cell_size + self.pcd_min[0],
            row * self.cell_size + self.pcd_min[1],
            z
        )

    def calculate_pos_2d(self, point_3d):
        """Convert a 3D point to a 2D grid coordinate.

        Args:
            point_3d (tuple or np.ndarray): The 3D point (x, y, z).

        Returns:
            tuple: The 2D grid coordinate (row, col).
        """
        # Ensure the point is a numpy array
        point_3d = np.array(point_3d)

        # Adjust x and y to the grid's local coordinates
        x_adjusted = point_3d[0] - self.pcd_min[0]
        y_adjusted = point_3d[1] - self.pcd_min[1]

        # Convert to grid indices
        col = np.floor(x_adjusted / self.cell_size).astype(int)
        row = np.floor(y_adjusted / self.cell_size).astype(int)

        return (row, col)

    def get_voronoi_graph(self):
        # deep copy the free space
        free_space_map = np.copy(self.free_space)

        boundary_map = binary_erosion(free_space_map, iterations=1).astype(np.uint8)

        # get the boundary points
        boundary_map = free_space_map - boundary_map

        # self.visualize_occupancy_map(boundary_map)

        rows, cols = np.where(boundary_map == 1)
        boundaries = np.array(list(zip(rows, cols)))
        voronoi = Voronoi(boundaries)

        voronoi_graph = nx.Graph()

        for simplex in voronoi.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.any(simplex < 0):
                continue
            src, tar = voronoi.vertices[simplex]
            # check on the image
            if (
                src[0] < 0
                or src[0] >= self.free_space.shape[0]
                or src[1] < 0
                or src[1] >= self.free_space.shape[1]
                or tar[0] < 0
                or tar[0] >= self.free_space.shape[0]
                or tar[1] < 0
                or tar[1] >= self.free_space.shape[1]
            ):
                continue
            # check on the free space
            if (
                self.free_space[int(src[0]), int(src[1])] == 0
                or self.free_space[int(tar[0]), int(tar[1])] == 0
            ):
                continue

            # check if src and tar already exist in the graph
            if (src[0], src[1]) not in voronoi_graph.nodes:
                voronoi_graph.add_node(
                    (src[0], src[1]),
                    # Note: pay attention to xyz for pos here
                    pos=(
                        src[1] * self.cell_size + self.pcd_min[0],
                        src[0] * self.cell_size + self.pcd_min[1],
                    )
                )
            if (tar[0], tar[1]) not in voronoi_graph.nodes:
                voronoi_graph.add_node(
                    (tar[0], tar[1]),
                    pos=(
                        tar[1] * self.cell_size + self.pcd_min[0],
                        tar[0] * self.cell_size + self.pcd_min[1],
                    )
                )
            # check if the edge already exists
            if not voronoi_graph.has_edge((src[0], src[1]), (tar[0], tar[1])):
                voronoi_graph.add_edge(
                    (src[0], src[1]),
                    (tar[0], tar[1]),
                    dist=np.linalg.norm(src - tar),
                )

        # self.visualize_graph(voronoi_graph,"Test")

        self.remove_degree_2_nodes_and_reconnect(voronoi_graph)

        # self.visualize_graph(voronoi_graph,"remove_degree_2_nodes_and_reconnect", show_nodes=False)
        # self.visualize_graph(voronoi_graph,"remove_degree_2_nodes_and_reconnect", show_nodes=True)

        return voronoi_graph
    
    def visualize_graph(self, graph, title="Voronoi Graph", show_nodes=True, show_edges=True):
        """
        Function to visualize a graph over the free space map.
        
        Parameters:
            graph (networkx.Graph): The graph to visualize.
            title (str): The title of the plot.
            show_nodes (bool): Whether to visualize nodes.
            show_edges (bool): Whether to visualize edges.
        """
        # Create a copy of the free space to work with
        fig_free = self.free_space.copy().astype(np.uint8) * 255
        fig_free = cv2.cvtColor(fig_free, cv2.COLOR_GRAY2BGR)

        # Visualize edges if required
        if show_edges:
            for edge in graph.edges:
                v1, v2 = edge
                cv2.line(
                    fig_free,
                    tuple(np.int32(v1[::-1])),
                    tuple(np.int32(v2[::-1])),
                    (0, 0, 255),
                    1,
                )

        # Visualize nodes if required
        if show_nodes:
            for node in graph.nodes:
                node_pos = tuple(np.int32(node[::-1]))
                cv2.circle(fig_free, node_pos, 2, (255, 0, 0), -1)

        # Use matplotlib to display the image
        plt.figure(figsize=(10, 10))
        plt.imshow(fig_free)  # Show image
        plt.title(title)  # Add title
        plt.axis('off')  # Disable axis
        plt.show()


    def remove_degree_2_nodes_and_reconnect(self, graph):
        # Find nodes with degree 2
        nodes_to_remove = [node for node, degree in graph.degree() if degree == 2]

        for node in nodes_to_remove:
            # Find the neighbors of this node
            neighbors = list(graph.neighbors(node))

            # If the node has two neighbors, connect them
            if len(neighbors) == 2:
                n1, n2 = neighbors
                # Add a new edge to connect the two neighbors
                dist = np.linalg.norm(np.array(graph.nodes[n1]['pos']) - np.array(graph.nodes[n2]['pos']))
                graph.add_edge(n1, n2, dist=dist)

                # Remove the original edges connected to the degree-2 node
                graph.remove_edge(n1, node)
                graph.remove_edge(n2, node)

            # Remove the degree-2 node
            graph.remove_node(node)

    def find_nearest_node(self, point, goal=None):
        """
        Find the nearest node in the graph to the given point, considering direction and excluding nodes with degree 1.
        
        Args:
            point: Target point (x, y)
            goal: Optional, target point (x, y), used to determine direction from start to goal
        
        Returns:
            nearest_node: The nearest node to the target point
        """
        nearest_node = None
        min_dist = float('inf')

        # If there is a goal, calculate the direction vector from start to goal
        if goal is not None:
            direction_vector = np.array(goal) - np.array(point)

        for node in self.graph.nodes:
            # Skip nodes with degree 1
            if self.graph.degree(node) == 1:
                continue

            dist = np.linalg.norm(np.array(node) - np.array(point))  # Calculate Euclidean distance
            if dist < min_dist:
                # If there is a goal, further check if the node is in the correct direction
                if goal is not None:
                    node_vector = np.array(node) - np.array(point)
                    # Calculate the dot product between direction vectors
                    dot_product = np.dot(direction_vector, node_vector)
                    # If the dot product is positive, the node is in the same direction
                    if dot_product > 0:
                        min_dist = dist
                        nearest_node = node
                else:
                    # If there is no goal, only consider the distance
                    min_dist = dist
                    nearest_node = node

        return nearest_node

    def smooth_path(self, path, smoothing_factor=0.8):
        """
        Simple path smoothing: connect nodes in the path by linear interpolation.

        Args:
            path: Original path, list of (x1, y1), (x2, y2), ...
            smoothing_factor: Factor controlling the degree of smoothing, closer to 1 means smoother path

        Returns:
            smooth_path: Smoothed path
        """
        smoothed_path = [path[0]]  # Start from the first point
        for i in range(1, len(path) - 1):
            prev_point = np.array(path[i-1])
            current_point = np.array(path[i])
            next_point = np.array(path[i+1])

            # Smooth the current node by weighted average
            smoothed_point = smoothing_factor * current_point + (1 - smoothing_factor) * (prev_point + next_point) / 2
            smoothed_path.append(tuple(smoothed_point))

        smoothed_path.append(path[-1])  # End point
        return smoothed_path

    def angle_between_points(self, p1, p2, p3):
        """
        Calculate the angle between three points.
        Args:
            p1, p2, p3: Coordinates of the points (x, y)
        Returns:
            angle: The angle between the three points (in degrees)
        """
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)

        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        cos_angle = dot_product / (norm_v1 * norm_v2)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def remove_sharp_turns(self, path, angle_threshold=30):
        """
        Remove sharp turns in the path; only keep path segments with angles less than angle_threshold.
        Args:
            path: Original path, list of (x1, y1), (x2, y2), ...
            angle_threshold: Angle threshold; turns greater than this are considered sharp turns
        Returns:
            filtered_path: Path after removing sharp turns
        """
        filtered_path = [path[0]]  # Keep the start point
        for i in range(1, len(path) - 1):
            p1 = path[i - 1]
            p2 = path[i]
            p3 = path[i + 1]

            angle = self.angle_between_points(p1, p2, p3)

            # If the angle is less than the threshold, consider it a valid path and keep the current point
            if angle < angle_threshold:
                filtered_path.append(p2)

        filtered_path.append(path[-1])  # Keep the end point
        return filtered_path

    def find_shortest_path(self, start, goal):
        """
        Find the shortest path from start to goal in the Voronoi graph using Dijkstra's algorithm.
        
        Args:
            start: Start point coordinates (x, y)
            goal: Goal point coordinates (x, y)
        
        Returns:
            path: List of nodes representing the shortest path from start to goal
        """
        # Find the nearest nodes to start and goal
        nearest_start_node = self.find_nearest_node(start, goal)
        nearest_goal_node = self.find_nearest_node(goal, start)

        # Use Dijkstra's algorithm to find the shortest path
        try:
            path = nx.dijkstra_path(self.graph, source=nearest_start_node, target=nearest_goal_node, weight='dist')

            full_path = [start] + path + [goal]

            # self.visualize_path_on_map(full_path)

            path = self.remove_sharp_turns(full_path)

            # self.visualize_path_on_map(path)

            path = self.smooth_path(path)  # Smooth the path

            # self.visualize_path_on_map(path)

            # Convert the path to pos and store in a new list
            pos_path = [self.calculate_pos_3d(x, y) for x, y in path]

            # Save path and ready to send off
            self.pos_path = pos_path

            return path

        except nx.NetworkXNoPath:
            print(f"No path found between {nearest_start_node} and {nearest_goal_node}")
            return None

    def free_space_check(self, point):
        row, col = point

        # Check if the point is within the map boundaries
        if 0 <= row < self.free_space.shape[0] and 0 <= col < self.free_space.shape[1]:
            # Check if the point is in free space (value 1 indicates free space)
            return self.free_space[row, col] == 1
        else:
            # If the point is out of bounds, return False
            return False

    def visualize_start_and_select_goal(self, start):
        """
        Visualize the occupancy map and start point, and select a goal point by mouse click.
        If the goal point is in free space, directly perform path planning.
        If not in free space, find the nearest node as the goal for path planning.

        Args:
            start: Start point coordinates, format (row, col).

        Returns:
            path: Generated path, list of nodes [(x1, y1), (x2, y2), ...].
        """
        # Create a copy of the visualization image
        fig_free = self.free_space.copy().astype(np.uint8) * 255
        fig_free = cv2.cvtColor(fig_free, cv2.COLOR_GRAY2BGR)

        # Draw the start point
        cv2.circle(fig_free, tuple(np.int32(start[::-1])), 2, (255, 0, 0), -1)  # Red dot represents start

        # Set up matplotlib image
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(fig_free, origin='lower')
        ax.set_title("Click to select Goal Points (Press Q to finish)")
        ax.axis('off')  # Hide axes

        goal = [None]  # Container to store the goal point
        path = [None]  # Container to store the path

        def onclick(event):
            """Handle mouse click event"""
            nonlocal path

            # Check if the click is within the image boundaries
            if event.xdata is None or event.ydata is None:
                return

            # Get the goal point coordinates
            goal_x, goal_y = int(event.xdata), int(event.ydata)
            goal[0] = (int(event.ydata), int(event.xdata))  # Convert to (row, col)

            # Clear previously drawn image content
            ax.clear()
            ax.imshow(fig_free, origin='lower')
            ax.set_title("Click to select Goal Points (Press Q to finish)")
            ax.axis('off')

            # Check if the goal point is in free space
            if self.free_space_check(goal[0]):
                print(f"Goal Point Selected: {goal[0]} (In Free Space)")
                path[0] = self.find_shortest_path(start, goal[0])
            else:
                print(f"Goal Point Selected: {goal[0]} (Not in Free Space)")

                # Find snapped goal point
                # snapped_goal = self.snap_to_free_space(goal[0], self.free_space)
                snapped_goal = self.snap_to_free_space_directional(goal[0], start, self.free_space)
                snapped_goal_2d = snapped_goal
                print(f"Snapped Goal Point: {snapped_goal}, Goal Point: {goal[0]}")
                ax.plot(
                    snapped_goal[1],
                    snapped_goal[0],
                    "yo",
                    markersize=10,
                    label="Snapped Goal",
                )

                snapped_goal = self.calculate_pos_3d(snapped_goal[0], snapped_goal[1])

                self.snapped_goal = snapped_goal

                # TODO: Need to set start in this func?
                # Use the direction to finetune the nearest goal
                # nearest_node = self.find_nearest_node(goal[0], start)
                nearest_node = self.find_nearest_node(snapped_goal_2d, start)
                print(f"Nearest Node: {nearest_node}")
                path[0] = self.find_shortest_path(start, nearest_node)

            # Draw the goal point and path
            ax.plot(goal_x, goal_y, 'bo', markersize=10, label="Goal")  # Blue dot represents goal
            if not self.free_space_check(goal[0]):  # If the goal point is not in free space
                ax.plot(nearest_node[1], nearest_node[0], 'go', markersize=10, label="Nearest Node")  # Green dot represents nearest node

            if path[0]:  # If the path is successfully generated, draw the path
                for i in range(len(path[0]) - 1):
                    src, tar = path[0][i], path[0][i + 1]
                    ax.plot([src[1], tar[1]], [src[0], tar[0]], 'g-', linewidth=1)  # Green line represents the path

            # Update the image
            fig.canvas.draw()
            print(f"Path Generated with length: {len(path[0])}")

        # Connect mouse click event
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        # Display the image and wait for user key press to end
        plt.show()

        if path[0]:
            return path[0][-1]
        else:
            print("No path generated. Returning None.")
            return None

    def save_pose_path_to_disk(self, pos_path, filename="pose_path.json"):
        """
        Save the pose_path as a JSON file.
        
        Args:
            pos_path: List of paths to save (3D coordinate list)
            filename: Filename to store, default is "pose_path.json"
        """
        import json
        # Convert pos_path to a list format to avoid serialization issues with NumPy arrays
        pos_path_list = [list(pos) for pos in pos_path]

        # Save to disk
        with open(filename, 'w') as f:
            json.dump(pos_path_list, f, indent=4)
        print(f"Pose path saved to {filename}")

    def visualize_path_on_map(self, path):
        """
        Visualize the path, showing path nodes and connected edges.

        Args:
            path: Path from start to goal, list of nodes [(x1, y1), (x2, y2), ...]
            fig_free: Base image for displaying the path (free space map)

        Returns:
            fig_free_with_path: Image with the path drawn on the original image
        """
        fig_free = self.free_space.copy().astype(np.uint8) * 255
        fig_free = cv2.cvtColor(fig_free, cv2.COLOR_GRAY2BGR)

        # Iterate through the nodes on the path, draw the path
        for i in range(len(path) - 1):
            src = path[i]
            tar = path[i + 1]

            # Draw the edge between nodes (line connecting the path)
            cv2.line(fig_free, tuple(np.int32(src[::-1])), tuple(np.int32(tar[::-1])), (0, 255, 0), 1)

        # Iterate through path nodes, draw the nodes
        for node in path:
            cv2.circle(fig_free, tuple(np.int32(node[::-1])), 1, (0, 0, 255), -1)

        # Display the path image
        plt.figure(figsize=(10, 10))
        plt.imshow(fig_free, origin='lower')
        plt.title("Path Visualization")  # Add title
        plt.axis('off')  # Turn off axis display
        plt.show()

        return fig_free

    def sample_random_point(self):
        """
        Randomly sample a point on the free space image.

        Returns:
            random_point: Randomly sampled point coordinates (x, y)
        """
        # Get all free space indices (pixels with value 1)
        free_space_indices = np.argwhere(self.free_space == 1)

        # Randomly select one from all free space indices
        random_index = np.random.choice(len(free_space_indices))

        # Get the randomly selected point coordinates
        random_point = tuple(free_space_indices[random_index])

        return random_point

    def find_rrt_path(self, start, goal):
        print(f"{start}, {goal}")
        print(f"x :{self.free_space.shape[1]}, y: {self.free_space.shape[0]}")

        # TODO: judge whether the end point is in the occ map
        if not self.is_in_bounds(start):
            print("Start point is out of bounds!")
            self.pos_path = None
            return []
        
        if not self.is_in_bounds(goal):
            print("Goal point is out of bounds!")
            self.pos_path = None
            return []

        self.rrt.set_occ_map(self.free_space)

        # self.rrt.visualize_occupancy_map(self.free_space)

        start = (start[1], start[0])  # Swap x and y of start
        goal = (goal[1], goal[0])

        self.rrt.set_start_goal(start, goal)

        path_rrt = self.rrt.plan()

        if len(path_rrt) == 0:
            return []

        flipped_path = [(y, x) for (x, y) in path_rrt]

        flipped_path = self.remove_sharp_turns(flipped_path)

        flipped_path = self.smooth_path(flipped_path)

        # Change 2D path to 3D
        pos_path = [self.calculate_pos_3d(x, y) for x, y in flipped_path]

        self.pos_path = pos_path

        return pos_path


# functions used in core for path refine
def remaining_path(path, current_pose):
    """
    Calculate the remaining path from the current pose along the global path.

    Parameters:
    - global_path: List of 3D points [(x1, y1, z1), (x2, y2, z2), ...].
    - current_pose: 4x4 numpy array representing the current transformation matrix.

    Returns:
    - remaining_path: List of remaining 3D points [(x, y, z), ...].
    """
    # Extract current position (translation part) from the transformation matrix
    current_position = current_pose[:3, 3]
    current_xy = current_position[:2]  # Ignore Z component

    # Find the closest point in the global path to the current position (based on XY distance only)
    distances = [np.linalg.norm(np.array(point[:2]) - current_xy) for point in path]
    closest_idx = np.argmin(distances)

    # Ensure we do not go backwards in the path
    remaining_path = path[closest_idx:]
    return remaining_path

def angle_between_points_3d(p1, p2, p3):
    """
    Calculate the angle between three 3D points on the XY plane (in degrees).

    Args:
        p1, p2, p3: 3D points (x, y, z).

    Returns:
        angle: The angle between the three points (in degrees, 0-180).
    """
    # Extract XY coordinates, ignoring Z
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    # Calculate the angle between vectors
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

    # Avoid division by zero
    if norm_product == 0:
        return 0

    # Calculate the angle (range: 0-180 degrees)
    cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))

    return angle

def remove_sharp_turns_3d(path, angle_threshold=60):
    """
    Remove sharp turns in the 3D path on the XY plane, recursively processing until all angles meet the threshold.

    Args:
        path: List of 3D points [(x1, y1, z1), (x2, y2, z2), ...].
        angle_threshold: Angle threshold; turns greater than this are considered sharp turns.

    Returns:
        filtered_path: Path after removing sharp turns.
    """

    def filter_once(path):
        """Remove sharp turns in a single pass."""
        filtered_path = [path[0]]  # Keep the start point
        for i in range(1, len(path) - 1):
            p1 = path[i - 1]
            p2 = path[i]
            p3 = path[i + 1]

            angle = angle_between_points_3d(p1, p2, p3)

            # Keep the point if the angle is less than the threshold
            if angle < angle_threshold:
                filtered_path.append(p2)
        filtered_path.append(path[-1])  # Keep the end point
        return filtered_path

    # Recursively remove sharp turns until the path no longer changes
    previous_path = []
    filtered_path = path
    while filtered_path != previous_path:
        previous_path = filtered_path
        filtered_path = filter_once(filtered_path)

    return filtered_path


