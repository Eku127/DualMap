import json
import plyfile
import torch

import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

from sklearn.neighbors import BallTree
from utils.eval.scannet200_constants import *


def load_replica_ply(ply_path: str, semantic_info_path: str):
    print(f"Loading GT from: {ply_path}")
    # Load ply file
    plydata = plyfile.PlyData.read(ply_path)

    # Read semantic info
    with open(semantic_info_path) as f:
        semantic_info = json.load(f)
    # Extract a mapping from object id to class id
    object_class_mapping = {obj["id"]: obj["class_id"]
                            for obj in semantic_info["objects"]}
    unique_class_ids = np.unique(list(object_class_mapping.values()))

    # Extract vertex data
    vertices = np.vstack(
        [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    # Extract vertex index tuple and object id for each face
    face_vertices = plydata["face"]["vertex_indices"]  # (num_faces, 4)
    object_ids = plydata["face"]["object_id"] # (num_faces,)

    # Store every face related 4 points xyz --> 4 x 3 shape
    vertices_per_face = []
    # Store every face related 4 points obj_id --> 4 shape
    object_ids_per_face = []

    # From face level to vertex level
    for i, face in enumerate(face_vertices):
        vertices_per_face.append(vertices[face])
        object_ids_per_face.append(np.repeat(object_ids[i], len(face)))
    vertices_face = np.vstack(vertices_per_face)  # (num_faces * 4, 3)
    object_ids_face = np.hstack(object_ids_per_face)  # (num_faces * 4,)

    # Lists to store the ordered pcd and class id for objects/points
    gt_objs = []
    gt_obj_ids = []
    gt_pts = []
    gt_pt_ids = []

    # Traverse through all the object ids
    unique_object_ids = np.unique(object_ids)
    
    # Object level GT
    for obj_id in unique_object_ids:
        if obj_id in object_class_mapping.keys():
            # Get points for the current object
            points = vertices_face[object_ids_face == obj_id, :]
            # Create a point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            gt_objs.append(pcd)
        if obj_id in object_class_mapping.keys():
            gt_obj_ids.append(object_class_mapping[obj_id])
        # else:
        #     gt_obj_ids.append(0)
    # Set undefined situation as background
    # gt_obj_ids = [0 if class_id == -1 else class_id for class_id in gt_obj_ids]
    gt_obj_ids = np.array(gt_obj_ids)

    # Point level GT
    for i, obj_id in enumerate(object_ids_face):
        if obj_id in object_class_mapping.keys():
            # if object_class_mapping[obj_id] in ignore:
            #     continue
            gt_pts.append(vertices_face[i])
            gt_pt_ids.append(object_class_mapping[obj_id])
        # else:
        #     gt_pt_ids.append(0)
    # Set undefined situation as background
    # gt_pt_ids = [0 if class_id == -1 else class_id for class_id in gt_pt_ids]
    gt_pt_ids = np.array(gt_pt_ids)
    gt_pts = np.vstack(gt_pts)

    class_colors = np.zeros((len(gt_pt_ids), 3))
    unique_class_colors = np.random.rand(len(unique_class_ids), 3)
    for i, class_id in enumerate(unique_class_ids):
        class_colors[gt_pt_ids == class_id] = unique_class_colors[i]

    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(gt_pts)
    full_pcd.colors = o3d.utility.Vector3dVector(class_colors)

    # pcd_instance = o3d.geometry.PointCloud()
    # pcd_instance.points = o3d.utility.Vector3dVector(vertices_face)
    # pcd_instance.colors = o3d.utility.Vector3dVector(instance_colors)

    return full_pcd, gt_pt_ids, gt_objs, gt_obj_ids


def load_scannet_ply(ply_path: str, use_scannet200: bool = False):
    print(f"Loading GT from: {ply_path}")
    # Load ply file
    plydata = plyfile.PlyData.read(ply_path)
    # Extract vertex xyz
    vertices = np.vstack(
        [plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    # print(vertices.shape)
    # Extract vertex colors
    vertex_colors = np.vstack(
        [plydata["vertex"]["red"], plydata["vertex"]["green"], plydata["vertex"]["blue"]]).T
    labels = plydata["vertex"]["label"]
    instance_ids = plydata["vertex"]["instance_id"]

    # Accept part of vertices because only the 20 classes are fully annotated
    valid_class_ids = list(VALID_CLASS_IDS_20)
    if use_scannet200:
        valid_class_ids = list(VALID_CLASS_IDS_200)
    
    valid_mask = np.isin(labels, valid_class_ids)
    valid_vertices = vertices[valid_mask, :]
    valid_instance_ids = instance_ids[valid_mask]
    gt_pt_ids = labels[valid_mask]

    valid_colors = vertex_colors[valid_mask, :]
    if use_scannet200:
        for class_id in valid_class_ids:
            color = SCANNET_COLOR_MAP_200[class_id]
            valid_colors[gt_pt_ids == class_id] = color
    
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(valid_vertices)
    full_pcd.colors = o3d.utility.Vector3dVector(valid_colors / 255.0)

    if use_scannet200:
        # Lists to store the ordered pcd and class id for objects/points
        gt_objs = []
        gt_obj_ids = []

        # Traverse through all the object ids
        unique_object_ids = np.unique(valid_instance_ids)

        # Object level GT
        for obj_id in unique_object_ids:
            points = valid_vertices[valid_instance_ids == obj_id, :]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            gt_objs.append(pcd)
            obj_labels = gt_pt_ids[valid_instance_ids == obj_id]
            if np.all(obj_labels == obj_labels[0]):
                gt_obj_ids.append(obj_labels[0])
            else:
                print("[Warning] Multiple classes in one object")
                gt_obj_ids.append(np.bincount(obj_labels).argmax())
        gt_obj_ids = np.array(gt_obj_ids)

        return full_pcd, gt_pt_ids, gt_objs, gt_obj_ids

    else:
        return full_pcd, gt_pt_ids, None, None


def pairwise_iou_calculate(
    bbox1: o3d.geometry.AxisAlignedBoundingBox, 
    bbox2: o3d.geometry.AxisAlignedBoundingBox
) -> float:

    v1 = bbox1.volume()
    v2 = bbox2.volume()

    max1 = bbox1.get_max_bound()
    min1 = bbox1.get_min_bound()
    max2 = bbox2.get_max_bound()
    min2 = bbox2.get_min_bound()

    int_range = np.minimum(max1, max2) - np.maximum(min1, min2)
    int_range = np.maximum(int_range, 0)
    inter = np.prod(int_range)
    union = v1 + v2 - inter

    return np.float32(inter / union) if union != 0 else 0.0


def calculate_avg_prec(iou_matrix: np.array, obj_idx, gt_idx):
    acc_values = list()
    prec_values = list()
    rec_values = list()
    for thresh in np.linspace(0.0, 1.0, 11, endpoint=True):
        TP, TN, FP, FN = 0, 0, 0, 0
        TP = np.sum(iou_matrix[obj_idx, gt_idx] > thresh)
        FP = iou_matrix.shape[0] - TP
        FN = iou_matrix.shape[1] - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        acc_values.append(accuracy)
        prec_values.append(precision)
        rec_values.append(recall)
    rec_values.sort()
    # print('Recall: ', rec_values)
    # print('Precision: ', prec_values)
    avg_prec = np.trapz(prec_values, rec_values)
    return avg_prec
    

def compute_auc(top_k: list, labels: np.array, sim_mat: np.ndarray, class_ids: np.array):
    success_k = {k : 0 for k in top_k}
    num_gt_classes = sim_mat.shape[1]

    for idx, sim_row in enumerate(sim_mat):
        sorted_sim_idx = np.argsort(sim_row)[::-1]
        for k in top_k:
            sim_top_k = sorted_sim_idx[:k]
            top_k_ids = class_ids[sim_top_k]
            if labels[idx] in top_k_ids:
                success_k[k] += 1
    top_k_acc = {k : v / len(labels) for k, v in success_k.items()}
    
    y = np.array(list(top_k_acc.values()))
    x = np.array(list(top_k_acc.keys())) / num_gt_classes
    y = np.hstack([0, y, 1])
    x = np.hstack([0, x, 1])
    auc = np.trapz(y, x)

    return top_k_acc, auc

def draw_auc(auc_path, top_k_acc, class_names):
    num_gt_classes = len(class_names)

    y = np.array(list(top_k_acc.values()))
    x = np.array(list(top_k_acc.keys())) / num_gt_classes
    y = np.hstack([0, y, 1])
    x = np.hstack([0, x, 1])

    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel('% of ranked categories considered')
    plt.ylabel('Accuracy')
    plt.title('AUC_top_k')
    plt.grid(True)

    plt.savefig(auc_path, dpi=300)
    print(f"AUC chart saved to {auc_path}")

    # plt.show()

def knn_interpolation(cumulated_pc: np.ndarray, full_sized_data: np.ndarray, k):
    """
    Using k-nn interpolation to find labels of points of the full sized pointcloud
    :param cumulated_pc: cumulated pointcloud results after running the network
    :param full_sized_data: full sized point cloud
    :param k: k for k nearest neighbor interpolation
    :return: pointcloud with predicted labels in last column and ground truth labels in last but one column
    """

    labeled = cumulated_pc[cumulated_pc[:, -1] != -1]
    to_be_predicted = full_sized_data.copy()

    ball_tree = BallTree(labeled[:, :3], metric="minkowski")

    knn_classes = labeled[ball_tree.query(to_be_predicted[:, :3], k=k)[
        1]][:, :, -1].astype(int)

    interpolated = np.zeros(knn_classes.shape[0])

    for i in range(knn_classes.shape[0]):
        interpolated[i] = np.bincount(knn_classes[i]).argmax()

    output = np.zeros((to_be_predicted.shape[0], to_be_predicted.shape[1] + 1))

    output[:, :-1] = to_be_predicted

    output[:, -1] = interpolated

    assert output.shape[0] == full_sized_data.shape[0]

    return output


def draw_bar_chart(data_dict, save_path):
    """
    Generate and save a bar chart sorted by values in descending order.

    Args:
        data_dict: Input dictionary where keys are class names and values are IOU or other metrics.
        save_path: Path to save the bar chart.
    """
    # Sort the dictionary by values in descending order
    sorted_items = sorted(
        data_dict.items(), key=lambda item: item[1], reverse=True)

    # Unpack keys and values
    labels, values = zip(*sorted_items)

    # Create the bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, values, color='skyblue')

    # Add title and labels
    plt.title("Class IOU from High to Low")
    plt.xlabel("Class Name")
    plt.ylabel("IOU")

    # Add value labels on top of each bar, rounded to two decimal places
    for bar, value in zip(bars, values):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{value:.2f}',
                 ha='center', va='bottom', fontsize=10)

    # Rotate x-axis labels to prevent overlap
    plt.xticks(rotation=45, ha="right")

    # Adjust layout to fit labels
    plt.tight_layout()

    # Save the image
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Bar chart saved to {save_path}")


def draw_detailed_bar_chart(
    data_dict,
    class_id_names,
    unique_to_class_gt,
    unique_to_class_pred,
    save_path
):
    """
    Generate and save a detailed bar chart with color-coded bars based on class membership.

    Args:
        data_dict: Input dictionary where keys are class names and values are IOU or other metrics.
        class_id_names: Dictionary mapping class IDs to class names.
        unique_to_class_gt: List of class IDs unique to ground truth.
        unique_to_class_pred: List of class IDs unique to predictions.
        save_path: Path to save the bar chart.
    """
    unique_to_class_gt = [class_id_names[class_id]
                          for class_id in unique_to_class_gt]
    unique_to_class_pred = [class_id_names[class_id]
                            for class_id in unique_to_class_pred]

    # Sort the dictionary by values in descending order
    sorted_items = sorted(
        data_dict.items(), key=lambda item: item[1], reverse=True)

    # Unpack keys and values
    labels, values = zip(*sorted_items)

    # Create a color list based on class membership
    colors = []
    for class_id in labels:
        if np.isin(class_id, unique_to_class_gt):
            colors.append('blue')
        elif np.isin(class_id, unique_to_class_pred):
            colors.append('red')
        else:
            colors.append('gray')  # Default color

    # Create the bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(labels, values, color='skyblue')

    # Add title and labels
    plt.title("Class IOU from High to Low")
    plt.xlabel("Class Name")
    plt.ylabel("IOU")

    # Add value labels on top of each bar, rounded to two decimal places
    for bar, value in zip(bars, values):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{value:.2f}',
                 ha='center', va='bottom', fontsize=10)

    # Get current x-axis labels
    ax = plt.gca()
    xticks = ax.get_xticklabels()

    # Set x-axis label colors based on class membership
    for xtick, label in zip(xticks, labels):
        if np.isin(label, unique_to_class_gt):
            xtick.set_color('blue')
        elif np.isin(label, unique_to_class_pred):
            xtick.set_color('red')
        else:
            xtick.set_color('black')  # Default color

    # Rotate x-axis labels to prevent overlap
    plt.xticks(rotation=45, ha="right")

    # Adjust layout to fit labels
    plt.tight_layout()

    # Save the image
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Bar chart saved to {save_path}")


def get_text_features(
    clip_length: int,
    class_names: list,
    clip_model,
    clip_tokenizer,
    batch_size=64
) -> np.ndarray:
    
    multiple_templates = [
        "{}",
        "There is the {} in the scene.",
    ]
    
    # Get all the prompted sequences
    class_name_prompts = [x.format(lm) for lm in class_names for x in multiple_templates]
    
    # Get tokens
    text_tokens = clip_tokenizer(class_name_prompts).to('cuda')
    # Get Output features
    text_feats = np.zeros((len(class_name_prompts), clip_length), dtype=np.float32)
    # Get the text feature batch by batch
    text_id = 0
    while text_id < len(class_name_prompts):
        # Get batch size
        batch_size = min(len(class_name_prompts) - text_id, batch_size)
        # Get text prompts based on batch size
        text_batch = text_tokens[text_id : text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()
        
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        # move the calculated batch into the Ouput features
        text_feats[text_id : text_id + batch_size, :] = batch_feats
        # Move on and Move on
        text_id += batch_size
    
    # shrink the output text features into classes names size
    text_feats = text_feats.reshape((-1, len(multiple_templates), text_feats.shape[-1]))
    text_feats = np.mean(text_feats, axis=1)
    
    # Normalize the text features
    norms = np.linalg.norm(text_feats, axis=1, keepdims=True)
    text_feats /= norms
    
    return text_feats
