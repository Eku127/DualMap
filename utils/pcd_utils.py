import numpy as np
import open3d as o3d
import torch

from collections import Counter

def mask_depth_to_points(
    depth: torch.Tensor,
    image: torch.Tensor,
    cam_K: torch.Tensor,
    masks: torch.Tensor,
    device: str = 'cuda'
):
    N, H, W = masks.shape
    
    fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
    
    # x, y = (H, W)
    y, x = torch.meshgrid(torch.arange(0, H, device=device), torch.arange(0, W, device=device), indexing='ij')
    
    # z = (N, H, W)
    z = depth.repeat(N, 1, 1) * masks
    
    # (N, H, W)
    valid = (z > 0).float()
    
    # (N, H, W)    
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    # (N, H, W, 3)
    points = torch.stack((x, y, z), dim=-1) * valid.unsqueeze(-1)
    
    if image is not None:
        rgb = image.repeat(N, 1, 1, 1) * masks.unsqueeze(-1)
        colors = rgb * valid.unsqueeze(-1)
    else:
        print("No RGB image provided, assigning random colors to objects")
        # Generate a random color for each mask
        random_colors = torch.randint(0, 256, (N, 3), device=device, dtype=torch.float32) / 255.0  # RGB colors in [0, 1]
        # Expand dims to match (N, H, W, 3) and apply to valid points
        colors = random_colors.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1) * valid.unsqueeze(-1)
    
    return points, colors


def init_pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ## Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan( # inint
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
        
    return pcd

def refine_points_with_clustering(points, colors, eps=0.05, min_points=10):
    """
    Cluster the point cloud using Open3D's DBSCAN and extract the largest cluster.

    Args:
    - points: Point cloud coordinates (torch.Tensor, Nx3).
    - colors: Point cloud colors (torch.Tensor, Nx3).
    - eps: DBSCAN neighborhood radius.
    - min_points: Minimum number of points for DBSCAN.

    Returns:
    - refined_points: Filtered point cloud coordinates (numpy.ndarray).
    - refined_colors: Filtered point cloud colors (numpy.ndarray).
    """
    # Convert to numpy format
    points_np = points.cpu().numpy()
    colors_np = colors.cpu().numpy()

    # If there are no points, return empty arrays to avoid further processing
    if points_np.shape[0] == 0:
        # print("No points found in the input point cloud.")
        # FIXED: [KDTreeFlann::SetRawData] Failed due to no data warning
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    # Use Open3D's DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    # Get the size of each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Remove noise points (-1)
    if -1 in unique_labels:
        mask_noise = labels != -1
        labels = labels[mask_noise]
        points_np = points_np[mask_noise]
        colors_np = colors_np[mask_noise]
        unique_labels, counts = np.unique(labels, return_counts=True)  # Recalculate clustering information

    # Check if there are still clusters
    if len(unique_labels) == 0:
        # print("No valid clusters found after removing noise.")
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    # Find the largest cluster
    max_label = unique_labels[np.argmax(counts)]

    # Select points in the largest cluster
    mask = labels == max_label
    refined_points_np = points_np[mask]
    refined_colors_np = colors_np[mask]

    # Return as numpy arrays
    return refined_points_np, refined_colors_np

def pcd_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ## Remove noise via clustering
    # Convert point cloud to numpy arrays
    points_np = np.asarray(pcd.points)
    colors_np = np.asarray(pcd.colors) if len(pcd.colors) > 0 else None

    # DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    if len(labels) == 0:
        print("No clusters found!")
        return o3d.geometry.PointCloud()  # Return empty point cloud

    # Get cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Find largest cluster
    max_label = unique_labels[np.argmax(counts)]

    # Print cluster info
    print(f"Found {len(unique_labels)} clusters, selecting cluster with label {max_label} (size: {counts.max()})")

    # Select points in largest cluster
    mask = labels == max_label
    refined_points = points_np[mask]

    if colors_np is not None:
        refined_colors = colors_np[mask]
    else:
        refined_colors = None

    # Create new point cloud
    refined_pcd = o3d.geometry.PointCloud()
    refined_pcd.points = o3d.utility.Vector3dVector(refined_points)
    if refined_colors is not None:
        refined_pcd.colors = o3d.utility.Vector3dVector(refined_colors)

    return refined_pcd

def safe_create_bbox(pcd: o3d.geometry.PointCloud) -> o3d.geometry.AxisAlignedBoundingBox:
    """
    Safely compute the axis-aligned bounding box of a point cloud.
    If the point cloud is empty, min_bound and max_bound are both [0,0,0].
    """
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        # Return empty bounding box
        return o3d.geometry.AxisAlignedBoundingBox(np.array([0, 0, 0]), np.array([0, 0, 0]))
    else:
        bbox = pcd.get_axis_aligned_bounding_box()
        return bbox