import json
import os
import sys

import hydra
import logging
import yaml
import torch
import open_clip

import numpy as np
import open3d as o3d

from omegaconf import DictConfig, OmegaConf
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity

from utils.eval.eval_utils import *
from utils.eval.metric import *
from utils.object import BaseObject
from utils.time_utils import measure_time_block
from scipy.optimize import linear_sum_assignment


def setup_logging(log_file_path: str) -> None:
    """
    This function sets up logging configuration using a YAML file.

    Parameters:
    log_file_path (str): The path to the YAML configuration file for logging. The file should contain a dictionary with configuration settings.

    Returns:
    None
    """
    # Load YAML configuration
    with open('config/support_config/logging_config.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        # Set the log file path for the file handler
        config['handlers']['file']['filename'] = log_file_path
        # Apply configuration
        logging.config.dictConfig(config)


@measure_time_block("Evaluation Time")
@hydra.main(version_base=None, config_path="../config/", config_name="seg_evaluation")
def main(cfg: DictConfig):

    # Create saving dir
    # results saving dir
    os.makedirs(os.path.dirname(cfg.save_dir), exist_ok=True)
    output_dir = os.path.join(cfg.save_dir, str(
        cfg.dataset_name) + "_" + str(cfg.scene_id))
    os.makedirs(output_dir, exist_ok=True)

    # log saving dir
    log_dir = os.path.join(output_dir, "logfile.log")
    setup_logging(log_dir)

    logging.info(
        f"################ {cfg.dataset_name}_{cfg.scene_id} ################")
    logging.info("This is the application for evaluation of segmentation")
    logging.info(f"Results output path: {output_dir}")
    logging.info(f"Log output path: {log_dir}")

    # Get classes and color info
    # Dict: class id --> name
    class_id_names_path = os.path.join(
        cfg.config_path, f"{cfg.dataset_name}_{cfg.scene_id}_id_names.json")
    logging.info(
        "Loading classes id --> names  from: {}".format(class_id_names_path))

    if not os.path.exists(class_id_names_path):
        raise FileNotFoundError(
            f"Error: File not found: {class_id_names_path}")

    class_id_names = {}
    with open(class_id_names_path, 'r') as file:
        class_id_names = json.load(file)
    class_id_names = {int(key): value for key, value in class_id_names.items()}

    # Dict: class id --> color
    class_id_colors_path = os.path.join(
        cfg.config_path, f"{cfg.dataset_name}_{cfg.scene_id}_id_colors.json")
    logging.info(
        "Loading classes id --> colors from: {}".format(class_id_colors_path))

    if not os.path.exists(class_id_colors_path):
        raise FileNotFoundError(
            f"Error: File not found: {class_id_colors_path}")

    class_id_colors = {}
    with open(class_id_colors_path, 'r') as file:
        class_id_colors = json.load(file)
    class_id_colors = {int(key): value for key,
                       value in class_id_colors.items()}

    # if use given classes, load given classes id color
    given_class_id_colors = {}
    if cfg.use_given_classes:
        class_id_colors_path = cfg.given_classes_id_color
        logging.info(
            "Using Given Classes, Loading classes id --> colors from: {}".format(class_id_colors_path))
        with open(class_id_colors_path, 'r') as file:
            given_class_id_colors = json.load(file)
        given_class_id_colors = {int(key): value for key,
                        value in given_class_id_colors.items()}

    # Get Gt information path
    ply_path = os.path.join(cfg.dataset_gt_path,
                            cfg.scene_id, "habitat", "mesh_semantic.ply")
    semantic_info_path = os.path.join(
        cfg.dataset_gt_path, cfg.scene_id, "habitat", "info_semantic.json")

    if not os.path.exists(ply_path) or not os.path.exists(semantic_info_path):
        missing_files = []
        if not os.path.exists(ply_path):
            missing_files.append(ply_path)
        if not os.path.exists(semantic_info_path):
            missing_files.append(semantic_info_path)
        raise FileNotFoundError(
            f"Error: The following file(s) were not found: {', '.join(missing_files)}")

    # Load GT pcd and related class_ids
    gt_pcd, gt_class_ids, gt_objs, gt_obj_ids = load_replica_ply(ply_path, semantic_info_path)

    if cfg.is_debug:
        # sem full pcd
        o3d.io.write_point_cloud(os.path.join(
            output_dir, "gt_sem_pcd_full.pcd"), gt_pcd)

    # Get ignoring classes
    ignore = [-1]
    if cfg.filter_gt_with_ignore:
        ignore.extend([id for id, name in class_id_names.items() if any(
            keyword in name for keyword in cfg.ignore_classes)])
    
    gt_points = np.asarray(gt_pcd.points)

    mask = np.isin(gt_class_ids, ignore, invert=True)
    gt_points = gt_points[mask]
    gt_class_ids = gt_class_ids[mask]

    mask_obj = np.isin(gt_obj_ids, ignore, invert=True)
    gt_objs_masked = []
    for idx, m in enumerate(mask_obj):
        if m:
            gt_objs_masked.append(gt_objs[idx])
    gt_obj_ids = gt_obj_ids[mask_obj]

    # assign the color from color config to GT pointcloud
    gt_colors = np.zeros((len(gt_class_ids), 3))
    for i, idx in enumerate(gt_class_ids):
        gt_colors[i] = class_id_colors[idx]

    gt_pcd.points = o3d.utility.Vector3dVector(np.asarray(gt_points))
    gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors)
    
    if cfg.is_debug:
        o3d.io.write_point_cloud(os.path.join(
            output_dir, "gt_sem_pcd_filtered.pcd"), gt_pcd)

    # Loading Saved Running Results
    # check whether cfg.load_dir exists
    if not os.path.exists(cfg.load_dir):
        logging.info(f"Error: {cfg.load_dir} does not exist.")
        sys.exit(1)

    logging.info("Loading saved obj results from: {}".format(cfg.load_dir))

    # traverse the .pkl in the directory to get constructed maps
    obj_map = []
    for file in os.listdir(cfg.load_dir):
        if file.endswith(".pkl"):
            obj_results_path = os.path.join(cfg.load_dir, file)
            # object construction
            loaded_obj = BaseObject.load_from_disk(obj_results_path)
            obj_map.append(loaded_obj)
    logging.info(f"Successfully loaded {len(obj_map)} objects")

    # Get the constructed objects map
    rgb_full_pcd = o3d.geometry.PointCloud()
    seg_full_pcd = o3d.geometry.PointCloud()
    for obj in obj_map:
        # rgb full pcd
        rgb_full_pcd += obj.pcd

        # seg full pcd
        if cfg.use_given_classes:
            color = given_class_id_colors[obj.class_id]
        else:
            color = class_id_colors[obj.class_id]
        seg_full_pcd += obj.pcd.paint_uniform_color(color)

    if cfg.is_debug:
        o3d.io.write_point_cloud(os.path.join(
            output_dir, "pred_rgb_pcd_full.pcd"), rgb_full_pcd)
        o3d.io.write_point_cloud(os.path.join(
            output_dir, "pred_seg_pcd_full.pcd"), seg_full_pcd)

    # From Here change the original class id in the obj into clip generated one
    # If we choose to use the CLIP to generate the labels for objects in the obj map
    if cfg.use_clip_for_labels:
        print(f"[Detector] Loading CLIP model: {cfg.clip.model_name} with pretrained weights '{cfg.clip.pretrained}'")
                
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            cfg.clip.model_name, pretrained=cfg.clip.pretrained
        )
        clip_model = clip_model.to(cfg.device)
        clip_model.eval()
        
        # Only reparameterize if the model is MobileCLIP
        if "MobileCLIP" in cfg.clip.model_name:
            from mobileclip.modules.common.mobileone import reparameterize_model
            clip_model = reparameterize_model(clip_model)
        
        clip_tokenizer = open_clip.get_tokenizer(cfg.clip.model_name)

        
        print("Start CLIP Label Generation and Evaluation")
        
        # get all the GT class names
        class_names = [class_id_names[id] for id in class_id_names]
        # prompt the class names into a sequence
        # Then generate the clip feat for each GT class name
        class_names_feats = get_text_features(cfg, class_names, clip_model, clip_tokenizer)
        
        # Get Top-N matched laebls for each object in obj map
        # Calculate cosine similarity mat, N(obj num) x M(GT num)
        sim_mat = np.zeros((len(obj_map), len(class_names)), dtype=np.float32)
        # stack all the clip feats from obj map into a np array
        obj_feats = np.vstack([obj.clip_ft for obj in obj_map])
        
        # calculate the cosine similarity between obj_feats and class_names_feats
        sim_mat = cosine_similarity(obj_feats, class_names_feats)
        
        # Determine the top-N matches for each object in obj_map
        top_n = cfg.top_n
        top_n_indices = np.argsort(-sim_mat, axis=1)[:, :top_n]  # Get the indices of the top-N matches
        
        # Optionally, convert indices to class names and return them
        # top_n_matches = [[class_names[i] for i in row] for row in top_n_indices]
        
        for obj, indices in zip(obj_map, top_n_indices):
            obj.clip_class_ids = indices
            
        # for idx, obj in enumerate(obj_map, start=1):
        #     print(f"Object {idx}: {class_names[obj.class_id]}: {', '.join(obj.clip_class_ids)}")

        # Currently we only support Top-1 results evaluation,
        # so we only change the class id with Top-1 clip matching result
        # Change the matching id in the further results evaluation
        for obj in obj_map:
            obj.class_id = obj.clip_class_ids[0]  # Change the class id with Top-1 clip matching result
        
        # save the debug information
        if cfg.is_debug:
            seg_full_pcd = o3d.geometry.PointCloud()
            for obj in obj_map:
                # seg full pcd
                color = class_id_colors[obj.class_id]
                seg_full_pcd += obj.pcd.paint_uniform_color(color)
            o3d.io.write_point_cloud(os.path.join(
                output_dir, "pred_seg_clip_pcd_full.pcd"), seg_full_pcd)
            
        # Object-level evaluation starts here...
        # ODJECT-LEVEL IoU and GT matching
        gt_bbox = []
        obj_bbox = []
        for gt_pcds in gt_objs_masked:
            gt_bbox.append(gt_pcds.get_axis_aligned_bounding_box())
        for obj in obj_map:
            obj_bbox.append(obj.pcd.get_axis_aligned_bounding_box())  
        # Calculate the IoU between GT and obj_map
        iou_matrix = np.zeros((len(obj_bbox), len(gt_bbox)))
        for i in range(len(obj_bbox)):
            for j in range(len(gt_bbox)):
                iou_matrix[i, j] = pairwise_iou_calculate(obj_bbox[i], gt_bbox[j])
        # Assign the GT label to each object
        obj_idx, gt_idx = linear_sum_assignment(iou_matrix, maximize=True)

        # obj_ap = calculate_avg_prec(iou_matrix, obj_idx, gt_idx)
        # print(f"Object-level AP: {obj_ap}")

        # Extract the assigned objects and labels
        map_assigned = []
        class_id_assigned = []
        for i in range(len(obj_idx)):
            # Only consider the matched pairs with IoU > 0.5
            if iou_matrix[obj_idx[i], gt_idx[i]] > 0:
                map_assigned.append(obj_map[obj_idx[i]])
                class_id_assigned.append(gt_obj_ids[gt_idx[i]])
        class_id_assigned = np.array(class_id_assigned)
        print(f"Assigned {len(map_assigned)} objects with GT labels")

        # Stack all the clip feats from obj map into a np array
        obj_feats_assigned = np.vstack([obj.clip_ft for obj in map_assigned])

        # Calculate cosine similarity matrix
        # obj_feats and class_names_feats are normalized before
        sim_mat_assigned = cosine_similarity(obj_feats_assigned, class_names_feats)

        # Top-k results evaluation
        # Calculate accuracy with different top-k
        top_k = cfg.replica.top_k_list

        top_k_acc, auc = compute_auc(top_k, class_id_assigned, sim_mat_assigned)

        # save the plot into the disk
        if cfg.is_debug:
            auc_draw_path = os.path.join(output_dir, "auc_top_k.png")
            draw_auc(auc_draw_path, top_k_acc, class_names)
            
        print(f"Top-k accuracy: {top_k_acc}")
        print(f"AUC: {auc}")
        # Object-level evaluation ends...

    # Process and matching gt <--> predicted
    # point level labels
    pred_class_ids = []
    for obj in obj_map:
        pred_class_ids.append(np.repeat(obj.class_id, len(obj.pcd.points)))
    pred_class_ids = np.hstack(pred_class_ids)

    pred_class_ids = pred_class_ids.reshape(-1, 1)
    gt_class_ids = gt_class_ids.reshape(-1, 1)

    # concat coords and labels for predicied pcd
    pred_xyz_ids = np.zeros((len(seg_full_pcd.points), 4))
    pred_xyz_ids[:, :3] = np.asarray(seg_full_pcd.points)
    pred_xyz_ids[:, -1] = pred_class_ids[:, 0]

    # concat coords and labels for gt pcd
    gt_xyz_ids = np.zeros((len(gt_pcd.points), 4))
    gt_xyz_ids[:, :3] = np.asarray(gt_pcd.points)
    gt_xyz_ids[:, -1] = gt_class_ids[:, 0]

    gt_xyz_ids_matchids = knn_interpolation(pred_xyz_ids, gt_xyz_ids, k=5)
    pred_class_ids = gt_xyz_ids_matchids[:, -1].reshape(-1, 1)

    class_ids_gt = gt_class_ids
    class_ids_pred = pred_class_ids

    if cfg.is_debug:
        gt_pred_pcd = o3d.geometry.PointCloud()
        gt_pred_pcd.points = o3d.utility.Vector3dVector(
            np.asarray(gt_pcd.points))
        # get colors based on the class_ids_pred
        gt_pred_pcd.colors = o3d.utility.Vector3dVector(
            np.array([class_id_colors[idx]
                     for idx in class_ids_pred.reshape(-1)])
        )
        o3d.io.write_point_cloud(os.path.join(
            output_dir, "gt_mapped_pcd_full.pcd"), gt_pred_pcd)

    # Calculation
    class_gt = np.unique(class_ids_gt)
    class_pred = np.unique(class_ids_pred)
    unique_to_class_gt = np.setdiff1d(class_gt, class_pred)
    unique_to_class_pred = np.setdiff1d(class_pred, class_gt)

    # IoU per class
    iou_per_class, iou_classes = compute_per_class_IoU(
        class_ids_pred, class_ids_gt, ignore=ignore)

    # Mean IoU
    mean_iou = np.mean(iou_per_class)

    # Frequency Weighted Intersection over Union
    fmiou = compute_FmIoU(class_ids_pred, class_ids_gt, ignore=ignore)

    # Acc per class
    acc_per_class, acc_classes = compute_per_class_accuracy(
        class_ids_pred, class_ids_gt, ignore=ignore)

    # Mean Acc
    macc = compute_mAcc(class_ids_pred, class_ids_gt, ignore=ignore)

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    results_dict = {
        "miou": mean_iou,
        "fmiou": fmiou,
        "macc": macc,
    }
    with open(results_path, 'w') as file:
        json.dump(results_dict, file, indent=4)

    pprint(results_dict)

    # debug info
    if cfg.is_debug:
        # save per class iou information to disk
        iou_path = os.path.join(output_dir, "classes_iou.json")
        # Generate classname -> iou dictionary
        iou_dict = {class_id_names[class_id]: iou for class_id, iou in zip(
            iou_classes, iou_per_class)}
        with open(iou_path, 'w') as file:
            json.dump(iou_dict, file, indent=4)

        iou_draw_path = os.path.join(output_dir, "classes_iou.png")

        class_gt = np.unique(class_ids_gt)
        class_pred = np.unique(class_ids_pred)
        unique_to_class_gt = np.setdiff1d(class_gt, class_pred)
        unique_to_class_pred = np.setdiff1d(class_pred, class_gt)

        draw_detailed_bar_chart(
            iou_dict, class_id_names, unique_to_class_gt, unique_to_class_pred, iou_draw_path)

        # save per class acc information to disk
        acc_path = os.path.join(output_dir, "classes_acc.json")
        # Generate classname -> acc dictionary
        acc_dict = {class_id_names[class_id]: iou for class_id, iou in zip(
            acc_classes, acc_per_class)}
        with open(acc_path, 'w') as file:
            json.dump(acc_dict, file, indent=4)

        acc_draw_path = os.path.join(output_dir, "classes_acc.png")

        draw_bar_chart(
            acc_dict, acc_draw_path)


def get_text_features(
    cfg: DictConfig,
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
    text_feats = np.zeros((len(class_name_prompts), cfg.clip.clip_length), dtype=np.float32)
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
    
    # TODO: Should we do normalization? Answer should be YES
    norms = np.linalg.norm(text_feats, axis=1, keepdims=True)
    text_feats /= norms
    
    return text_feats







if __name__ == "__main__":
    main()
