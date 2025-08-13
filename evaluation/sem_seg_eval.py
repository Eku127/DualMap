import json
import os
import sys

import hydra
import logging
import yaml
import open_clip
import copy

import numpy as np
import open3d as o3d

from omegaconf import DictConfig, OmegaConf
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity

from utils.eval.eval_utils import *
from utils.eval.metric import *
from utils.eval.scannet200_constants import *
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
        # Set file handler's log file path
        config['handlers']['file']['filename'] = log_file_path
        # Apply configuration
        logging.config.dictConfig(config)


class Evaluator:
    def __init__(self, cfg: DictConfig):
        print("[Evaluator] Initializing...")
        
        # Dataset and Scene Information
        self.dataset_name = cfg.dataset_name.lower()
        self.scene_id = cfg.scene_id
        
        # Configuration Paths
        self.config_path = cfg.config_path
        self.dataset_gt_path = cfg.dataset_gt_path
        self.load_dir = cfg.load_dir
        self.output_dir = os.path.join(cfg.save_dir, f"{self.dataset_name}_{self.scene_id}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Logging Setup
        log_dir = os.path.join(self.output_dir, "logfile.log")
        setup_logging(log_dir)
        logging.info(f"################ {cfg.dataset_name}_{cfg.scene_id} ################")
        logging.info("This is the application for evaluation of segmentation")
        logging.info(f"Results output path: {self.output_dir}")
        logging.info(f"Log output path: {log_dir}")
        
        # Evaluation Mode
        self.mode = "clip" if cfg.use_clip_for_labels else "detector"
        self.is_debug = cfg.is_debug
        
        # Dataset Specific Configurations
        self.if_calc_auc = cfg.if_calc_auc
        if self.dataset_name == "replica":        
            self.top_k_list = cfg.replica.top_k_list
        elif self.dataset_name == "scannet":
            self.top_k_list = cfg.scannet.top_k_list
            self.use_scannet200 = cfg.scannet.use_scannet200
        else:
            raise ValueError(f"Error: Unknown dataset name: {self.dataset_name}")
        
        # Ignore Classes Configuration
        self.filter_gt_with_ignore = cfg.filter_gt_with_ignore
        self.ignore_classes = cfg.ignore_classes
        self.ignore_ids = []

        # Class id-Color Mapping Configuration
        self.use_given_classes = cfg.use_given_classes
        self.given_classes_id_color = cfg.given_classes_id_color
        
        # CLIP Related Configuration
        self.clip_length = cfg.clip.clip_length
        self.top_n = cfg.top_n
        
        # Initialize Class Properties
        self.class_id_names = {}
        self.class_id_colors = {}
        self.given_class_id_colors = {}
        self.gt_pcd = None
        self.gt_pt_ids = None
        self.obj_map = []
        self.clip_model = None
        self.clip_tokenizer = None
        self.gt_objs = []
        self.gt_obj_ids = []
        self.class_names_feats = None
        self.iou_acc_dict = {}

        # Evaluation Results
        self.auc = 0
        self.top_k_acc = {}
        
        # Load CLIP Model if in CLIP Mode
        if self.mode == "clip":
            self.load_clip_model(cfg.clip.model_name, cfg.clip.pretrained, cfg.device)
    
    def set_ignore(self):
        # Get ignore classes of GT during evaluation
        ignore = [-1]
        if self.filter_gt_with_ignore:
            ignore.extend([id for id, name in self.class_id_names.items() if any(
                keyword in name for keyword in self.ignore_classes)])
        self.ignore_ids = ignore

    def replica_set_id_name_color_mapping(self):
        # Create class_id to name mapping
        class_id_names_path = os.path.join(
            self.config_path, f"{self.dataset_name}_{self.scene_id}_id_names.json")
        logging.info(
            "Loading class ids -> names from: {}".format(class_id_names_path))
        if not os.path.exists(class_id_names_path):
            raise FileNotFoundError(
                f"Error: File not found: {class_id_names_path}")
        
        self.class_id_names = {}
        with open(class_id_names_path, 'r') as file:
            class_id_names = json.load(file)
        self.class_id_names = {int(key): value 
                               for key, value in class_id_names.items()}

        # Create class id to color mapping
        class_id_colors_path = os.path.join(
            self.config_path, f"{self.dataset_name}_{self.scene_id}_id_colors.json")
        logging.info(
            "Loading classes id --> colors from: {}".format(class_id_colors_path))
        if not os.path.exists(class_id_colors_path):
            raise FileNotFoundError(
                f"Error: File not found: {class_id_colors_path}")
        
        self.class_id_colors = {}
        with open(class_id_colors_path, 'r') as file:
            class_id_colors = json.load(file)
        self.class_id_colors = {int(key): value 
                                for key, value in class_id_colors.items()}
        
    def scannet_set_id_name_color_mapping(self):
        logging.info(
            "Loading class ids -> names from: utils.eval.scannet200_constants")
        if self.use_scannet200:
            class_ids, class_names, class_id_colors = VALID_CLASS_IDS_200, CLASS_LABELS_200, SCANNET_COLOR_MAP_200
        else:
            class_ids, class_names, class_id_colors = VALID_CLASS_IDS_20, CLASS_LABELS_20, SCANNET_COLOR_MAP_20

        self.class_id_names ={class_ids[i]: class_names[i] for i in range(len(class_ids))}
        self.class_id_colors = class_id_colors
        
    def set_given_id_color(self):
        id_color_path = self.given_classes_id_color
        logging.info(
            "Using Given Classes, Loading classes id --> colors from: {}".format(id_color_path))
        with open(id_color_path, 'r') as file:
            given_class_id_colors = json.load(file)
        self.given_class_id_colors = {int(key): value 
            for key, value in given_class_id_colors.items()}

    def load_gt_replica(self):
        assert self.dataset_name == "replica"
        # Get GT information path
        ply_path = os.path.join(self.dataset_gt_path,
                                self.scene_id, "habitat", "mesh_semantic.ply")       
        semantic_info_path = os.path.join(
            self.dataset_gt_path, self.scene_id, "habitat", "info_semantic.json")

        # Check if the files exist
        if not os.path.exists(ply_path) or not os.path.exists(semantic_info_path):
            missing_files = []
            if not os.path.exists(ply_path):
                missing_files.append(ply_path)
            if not os.path.exists(semantic_info_path):
                missing_files.append(semantic_info_path)
            raise FileNotFoundError(
                f"Error: The following file(s) were not found: {', '.join(missing_files)}")

        if self.if_calc_auc:
            # Load point-level and object-level GT
            gt_pcd, gt_class_ids, gt_objs, gt_obj_ids = load_replica_ply(ply_path, semantic_info_path)
        else:
            # Load only point-level GT
            gt_pcd, gt_class_ids, _, _ = load_replica_ply(ply_path, semantic_info_path)
            
        if self.is_debug:
            # Save the sem full pcd
            o3d.io.write_point_cloud(os.path.join(
                self.output_dir, "gt_sem_pcd_full.pcd"), gt_pcd)
        
        # Filter GT pts with ignore classes
        gt_points = np.asarray(gt_pcd.points)
        mask = np.isin(gt_class_ids, self.ignore_ids, invert=True)
        gt_points = gt_points[mask]
        gt_class_ids = gt_class_ids[mask]

        # Filter GT objects with ignore classes
        if self.if_calc_auc:
            mask_obj = np.isin(gt_obj_ids, self.ignore_ids, invert=True)
            gt_objs_masked = []
            for idx, if_save in enumerate(mask_obj):
                if if_save:
                    gt_objs_masked.append(gt_objs[idx])
            self.gt_objs = gt_objs_masked
            self.gt_obj_ids = gt_obj_ids[mask_obj]

        # Assign uniform color for pts of same class
        gt_colors = np.zeros((len(gt_class_ids), 3))
        for i, idx in enumerate(gt_class_ids):
            gt_colors[i] = self.class_id_colors[idx]

        # Save the filtered and painted GT pcd
        self.gt_pcd = o3d.geometry.PointCloud()
        self.gt_pcd.points = o3d.utility.Vector3dVector(np.asarray(gt_points))
        self.gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors)
        self.gt_pt_ids = gt_class_ids
        
        if self.is_debug:
            o3d.io.write_point_cloud(os.path.join(
                self.output_dir, "gt_sem_pcd_filtered.pcd"), self.gt_pcd)

    def load_gt_scannet(self):
        ply_path = os.path.join(self.dataset_gt_path, self.scene_id, 
                                f"{self.scene_id}_vh_clean_2.labels.ply")
        if self.use_scannet200:
            ply_path = os.path.join(self.dataset_gt_path, f"{self.scene_id}.ply")
        missing_files = []
        if not os.path.exists(ply_path):
            missing_files.append(ply_path)
            raise FileNotFoundError(
                f"Error: The following file(s) were not found: {', '.join(missing_files)}")

        # Load GT pcd and related class_ids
        if self.if_calc_auc:
            # Load point-level and object-level GT
            gt_pcd, gt_class_ids, gt_objs, gt_obj_ids = load_scannet_ply(
                ply_path, self.use_scannet200)
        else:
            # Load only point-level GT
            gt_pcd, gt_class_ids, _, _ = load_scannet_ply(ply_path, self.use_scannet200)

        if gt_objs is None or gt_obj_ids is None:
            raise ValueError("Error: AUC calculation is not supported for ScanNet20, set if_calc_auc to False.")

        if self.is_debug:
            # sem full pcd
            o3d.io.write_point_cloud(
                os.path.join(self.output_dir, "gt_sem_pcd_full.pcd"), gt_pcd)
        
        # Filter GT pts with ignore classes
        gt_points = np.asarray(gt_pcd.points)
        mask = np.isin(gt_class_ids, self.ignore_ids, invert=True)
        gt_points = gt_points[mask]
        gt_class_ids = gt_class_ids[mask]

        # Filter GT objects with ignore classes
        if self.if_calc_auc:
            mask_obj = np.isin(gt_obj_ids, self.ignore_ids, invert=True)
            gt_objs_masked = []
            for idx, if_save in enumerate(mask_obj):
                if if_save:
                    gt_objs_masked.append(gt_objs[idx])
            self.gt_objs = gt_objs_masked
            self.gt_obj_ids = gt_obj_ids[mask_obj]

        # Assign uniform color for pts of same class
        gt_colors = np.zeros((len(gt_class_ids), 3))
        for i, idx in enumerate(gt_class_ids):
            gt_colors[i] = self.class_id_colors[idx]

        # Save the filtered and painted GT pcd
        # Note: The colors in Scannet config file are in [0, 255] range
        self.gt_pcd = o3d.geometry.PointCloud()
        self.gt_pcd.points = o3d.utility.Vector3dVector(np.asarray(gt_points))
        self.gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors / 255.0)
        self.gt_pt_ids = gt_class_ids

        if self.is_debug:
            o3d.io.write_point_cloud(os.path.join(
                self.output_dir, "gt_sem_pcd_filtered.pcd"), self.gt_pcd)
            
    def load_map(self):
        # Loading Saved Running Results of Dualmap
        # check whether cfg.load_dir exists
        if not os.path.exists(self.load_dir):
            logging.info(f"Error: {self.load_dir} does not exist.")
            sys.exit(1)

        logging.info("Loading saved obj results from: {}".format(self.load_dir))

        # traverse the .pkl in the directory to get constructed maps
        obj_map = []
        for file in os.listdir(self.load_dir):
            if file.endswith(".pkl"):
                obj_results_path = os.path.join(self.load_dir, file)
                # object construction
                loaded_obj = BaseObject.load_from_disk(obj_results_path)
                if not hasattr(loaded_obj, 'pcd'):
                    continue
                obj_map.append(loaded_obj)
        
        self.obj_map = copy.deepcopy(obj_map)
        logging.info(f"Successfully loaded {len(obj_map)} objects")

        if self.is_debug:
            # Get the constructed objects map
            rgb_full_pcd = o3d.geometry.PointCloud()
            seg_full_pcd = o3d.geometry.PointCloud()
    
            for obj in obj_map:
                # raw rgb full pcd
                rgb_full_pcd += obj.pcd

                # segmented full pcd (labeled by color)
                if self.use_given_classes:
                    color = self.given_class_id_colors[obj.class_id]
                else:
                    color = self.class_id_colors[obj.class_id]
                    if self.dataset_name == "scannet":
                        color = np.array(color) / 255.0
                seg_full_pcd += obj.pcd.paint_uniform_color(color)

            o3d.io.write_point_cloud(os.path.join(
                self.output_dir, "pred_rgb_pcd_full.pcd"), rgb_full_pcd)
            o3d.io.write_point_cloud(os.path.join(
                self.output_dir, "pred_seg_pcd_full.pcd"), seg_full_pcd)

    def calc_clip_labels(self):
        # From Here change the original class id in the obj into clip generated one
        # If we choose to use the CLIP to generate the labels for objects in the obj map        
        print("Start CLIP Label Generation...")
        
        # get all the GT class names
        class_names = [self.class_id_names[id] for id in self.class_id_names]
        # prompt the class names into a sequence
        # Then generate the clip feat for each GT class name
        self.class_names_feats = get_text_features(
            self.clip_length, class_names, self.clip_model, self.clip_tokenizer)
        
        # Get Top-N matched laebls for each object in obj map
        # Calculate cosine similarity mat, N(obj num) x M(GT num)
        sim_mat = np.zeros((len(self.obj_map), len(class_names)), dtype=np.float32)
        # stack all the clip feats from obj map into a np array
        obj_feats = np.vstack([obj.clip_ft for obj in self.obj_map])
        
        # calculate the cosine similarity between obj_feats and class_names_feats
        sim_mat = cosine_similarity(obj_feats, self.class_names_feats)
        
        # Determine the top-N matches for each object in obj_map
        top_n = self.top_n
        top_n_indices = np.argsort(-sim_mat, axis=1)[:, :top_n]  # Get the indices of the top-N matches
        
        # Optionally, convert indices to class names and return them
        # top_n_matches = [[class_names[i] for i in row] for row in top_n_indices]

        class_ids = np.array(list(self.class_id_names.keys()))
        for obj, indices in zip(self.obj_map, top_n_indices):
            obj.clip_class_ids = [class_ids[int(idx)] for idx in indices]

        # Currently we only support Top-1 results evaluation,
        # so we only change the class id with Top-1 clip matching result
        for obj in self.obj_map:
            obj.class_id = obj.clip_class_ids[0]  # Change the class id with Top-1 clip matching result
        
        # save the debug information
        if self.is_debug:
            seg_full_pcd = o3d.geometry.PointCloud()
            for obj in self.obj_map:
                # seg full pcd
                color = self.class_id_colors[obj.class_id]
                if self.dataset_name == "scannet":
                    color = np.array(color) / 255.0
                seg_full_pcd += obj.pcd.paint_uniform_color(color)
            o3d.io.write_point_cloud(os.path.join(
                self.output_dir, "pred_seg_clip_pcd_full.pcd"), seg_full_pcd)

    def calc_auc(self, iou_th=None):
        print("Calculating semantic Top-k AUC...")
        if iou_th is None:
            raise ValueError("Error: iou_th should be provided for AUC calculation")
        
        # ODJECT-LEVEL IoU and GT matching
        gt_bbox = []
        obj_bbox = []
        for gt_pcds in self.gt_objs:
            gt_bbox.append(gt_pcds.get_axis_aligned_bounding_box())
        for obj in self.obj_map:
            obj_bbox.append(obj.pcd.get_axis_aligned_bounding_box())
        
        # Calculate the IoU between GT and obj_map
        iou_matrix = np.zeros((len(obj_bbox), len(gt_bbox)))
        for i in range(len(obj_bbox)):
            for j in range(len(gt_bbox)):
                iou_matrix[i, j] = pairwise_iou_calculate(obj_bbox[i], gt_bbox[j])
        
        # Assign the GT label to each object
        obj_idx, gt_idx = linear_sum_assignment(iou_matrix, maximize=True)

        # Extract the assigned objects and labels
        map_assigned = []
        class_id_assigned = []
        for o_i, g_i in zip(obj_idx, gt_idx):
            # Only consider the matched pairs with IoU > 0.5
            if iou_matrix[o_i, g_i] > iou_th:
                map_assigned.append(self.obj_map[o_i])
                class_id_assigned.append(self.gt_obj_ids[g_i])
        class_id_assigned = np.array(class_id_assigned)
        print(f"Assigned {len(map_assigned)} objects with GT labels")

        # Stack all the clip feats from obj map into a np array
        obj_feats_assigned = np.vstack([obj.clip_ft for obj in map_assigned])

        # Calculate cosine similarity matrix
        # obj_feats and class_names_feats are normalized before
        sim_mat_assigned = cosine_similarity(obj_feats_assigned, self.class_names_feats)

        # Top-k results evaluation
        # Calculate accuracy with different top-k
        top_k = self.top_k_list
        class_ids = np.array(list(self.class_id_names.keys()))

        self.top_k_acc, self.auc = compute_auc(
            top_k, class_id_assigned, sim_mat_assigned, class_ids)

        # save the plot into the disk
        if self.is_debug:
            class_names = [self.class_id_names[id] for id in class_ids]
            auc_draw_path = os.path.join(self.output_dir, "auc_top_k.png")
            draw_auc(auc_draw_path, self.top_k_acc, class_names)
            
        print(f"Top-k Accuracy: {self.top_k_acc}")
        print(f"Semantic Top-k AUC: {self.auc}")

    def calc_iou_acc(self):
        # Process and matching gt <--> predicted
        # point level labels
        pred_class_ids = []
        for obj in self.obj_map:
            # if use CLIP, obj.class_id has been changed in calc_clip_labels()
            pred_class_ids.append(np.repeat(obj.class_id, len(obj.pcd.points)))
        pred_class_ids = np.hstack(pred_class_ids)

        pred_class_ids = pred_class_ids.reshape(-1, 1)
        gt_class_ids = self.gt_pt_ids.reshape(-1, 1)

        seg_full_pcd = o3d.geometry.PointCloud()
        for obj in self.obj_map:
            if self.mode == "clip":
                # Evaluation with GT class list
                color = self.class_id_colors[obj.class_id]
                if self.dataset_name == "scannet":
                    color = np.array(color) / 255.0
            elif self.use_given_classes:
                # Eval uses the given class lists
                color = self.given_class_id_colors[obj.class_id]
            else:
                # Eval uses the GT class lists
                color = self.class_id_colors[obj.class_id]
                if self.dataset_name == "scannet":
                    color = np.array(color) / 255.0
            seg_full_pcd += obj.pcd.paint_uniform_color(color)

        # concat coords and labels for predicied pcd
        pred_xyz_ids = np.zeros((len(seg_full_pcd.points), 4))
        pred_xyz_ids[:, :3] = np.asarray(seg_full_pcd.points)
        pred_xyz_ids[:, -1] = pred_class_ids[:, 0]

        # concat coords and labels for gt pcd
        gt_xyz_ids = np.zeros((len(self.gt_pcd.points), 4))
        gt_xyz_ids[:, :3] = np.asarray(self.gt_pcd.points)
        gt_xyz_ids[:, -1] = gt_class_ids[:, 0]

        gt_xyz_ids_matchids = knn_interpolation(pred_xyz_ids, gt_xyz_ids, k=5)
        pred_class_ids = gt_xyz_ids_matchids[:, -1].reshape(-1, 1)

        class_ids_gt = gt_class_ids
        class_ids_pred = pred_class_ids

        if self.is_debug:
            gt_pred_pcd = o3d.geometry.PointCloud()
            gt_pred_pcd.points = o3d.utility.Vector3dVector(
                np.asarray(self.gt_pcd.points))
            # get colors based on the class_ids_pred
            gt_pred_pcd.colors = o3d.utility.Vector3dVector(
                np.array([self.class_id_colors[idx]
                        for idx in class_ids_pred.reshape(-1)])
            )
            o3d.io.write_point_cloud(os.path.join(
                self.output_dir, "gt_mapped_pcd_full.pcd"), gt_pred_pcd)

        # Calculation
        class_gt = np.unique(class_ids_gt)
        class_pred = np.unique(class_ids_pred)
        unique_to_class_gt = np.setdiff1d(class_gt, class_pred)
        unique_to_class_pred = np.setdiff1d(class_pred, class_gt)

        # IoU per class
        iou_per_class, iou_classes = compute_per_class_IoU(
            class_ids_pred, class_ids_gt, ignore=self.ignore_ids)

        # Mean IoU
        mean_iou = np.mean(iou_per_class)

        # Frequency Weighted Intersection over Union
        fmiou = compute_FmIoU(class_ids_pred, class_ids_gt, ignore=self.ignore_ids)

        # Acc per class
        acc_per_class, acc_classes = compute_per_class_accuracy(
            class_ids_pred, class_ids_gt, ignore=self.ignore_ids)

        # Mean Acc
        macc = compute_mAcc(class_ids_pred, class_ids_gt, ignore=self.ignore_ids)
        
        self.iou_acc_dict = {
            "miou": mean_iou,
            "fmiou": fmiou,
            "macc": macc,
        }
        pprint(self.iou_acc_dict)

        # debug info
        if self.is_debug:
            # save per class iou information to disk
            iou_path = os.path.join(self.output_dir, "classes_iou.json")
            # get classname -> iou dict
            iou_dict = {self.class_id_names[class_id]: iou for class_id, iou in zip(
                iou_classes, iou_per_class)}
            with open(iou_path, 'w') as file:
                json.dump(iou_dict, file, indent=4)

            iou_draw_path = os.path.join(self.output_dir, "classes_iou.png")

            class_gt = np.unique(class_ids_gt)
            class_pred = np.unique(class_ids_pred)
            unique_to_class_gt = np.setdiff1d(class_gt, class_pred)
            unique_to_class_pred = np.setdiff1d(class_pred, class_gt)

            draw_detailed_bar_chart(
                iou_dict, self.class_id_names, unique_to_class_gt, unique_to_class_pred, iou_draw_path)

            # save per class acc information to disk
            acc_path = os.path.join(self.output_dir, "classes_acc.json")
            # get classname -> acc dict
            acc_dict = {self.class_id_names[class_id]: iou for class_id, iou in zip(
                acc_classes, acc_per_class)}
            with open(acc_path, 'w') as file:
                json.dump(acc_dict, file, indent=4)

            acc_draw_path = os.path.join(self.output_dir, "classes_acc.png")
            draw_bar_chart(acc_dict, acc_draw_path)

    def get_dataset_name(self):
        return self.dataset_name

    def load_clip_model(self, model_name: str, pretrained: str, device: str):
        print(f"[Evaluator] Loading CLIP model: {model_name} with pretrained weights '{pretrained}'")                 
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.clip_model = clip_model.to(device)
        self.clip_model.eval()
        
        # Only reparameterize if the model is MobileCLIP
        if "MobileCLIP" in model_name:
            from mobileclip.modules.common.mobileone import reparameterize_model
            self.clip_model = reparameterize_model(self.clip_model)
        
        self.clip_tokenizer = open_clip.get_tokenizer(model_name)

    def save_results(self):
        results_dict = {}
        results_dict["miou"] = self.iou_acc_dict["miou"]
        results_dict["fmiou"] = self.iou_acc_dict["fmiou"]
        results_dict["macc"] = self.iou_acc_dict["macc"]
        results_dict["top_k_acc"] = self.top_k_acc
        results_dict["auc"] = self.auc
        results_dict["obj_num"] = len(self.obj_map)
        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, 'w') as file:
            json.dump(results_dict, file, indent=4)


@measure_time_block("Evaluation Time")
@hydra.main(version_base=None, config_path="../config/", config_name="seg_evaluation")
def main(cfg: DictConfig):
    evaluator = Evaluator(cfg)
    dataset_name = evaluator.get_dataset_name()
    # scene_id = evaluator.get_scene_id()

    if dataset_name == "replica":
        evaluator.replica_set_id_name_color_mapping()
        evaluator.set_ignore()
        evaluator.load_gt_replica()
    elif dataset_name == "scannet":
        evaluator.scannet_set_id_name_color_mapping()
        evaluator.set_ignore()
        evaluator.load_gt_scannet()

    if evaluator.is_debug:
        evaluator.set_given_id_color()

    evaluator.load_map()

    if evaluator.mode == "clip":
        evaluator.calc_clip_labels()

    if evaluator.if_calc_auc:
        evaluator.calc_auc(iou_th=0)

    evaluator.calc_iou_acc()
    evaluator.save_results()


if __name__ == "__main__":
    main()
