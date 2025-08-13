import os
import sys

import hydra
import logging
import torch
import copy

import numpy as np
import open3d as o3d

from natsort import natsorted
from omegaconf import DictConfig
from sklearn.metrics.pairwise import cosine_similarity
from plyfile import PlyData

from utils.eval.eval_utils import *
from utils.eval.metric import *
from utils.eval.scannet200_constants import *
from utils.time_utils import measure_time_block
from evaluation.sem_seg_eval import Evaluator


class LoadedObject:
    def __init__(self, pcd, clip_ft, class_id):
        self.pcd = pcd
        self.class_id = class_id
        self.clip_ft = clip_ft


class HOVEvaluator(Evaluator):
    def __init__(self, cfg):
        super(HOVEvaluator, self).__init__(cfg)
        self.hov_result_path = cfg.hov_result_path
    
    def load_map(self):
        # Read map built by HOV-SG
        # check whether load_dir exists
        if not os.path.exists(self.hov_result_path):
            logging.info(f"Error: {self.hov_result_path} does not exist.")
            sys.exit(1)

        logging.info("Loading saved obj results from: {}".format(self.hov_result_path))

        ply_dir = os.path.join(self.hov_result_path, "objects")
        logging.info("Loading saved obj results from: {}".format(ply_dir))
        ply_path_list = os.listdir(ply_dir)
        ply_path_list = natsorted(ply_path_list)
        
        feat_path = os.path.join(self.hov_result_path, "mask_feats.pt")
        feat_tensor = torch.load(feat_path)
        feats = feat_tensor.numpy()

        # For HOV-SG, we have no class_id, use CLIP to generate top-1 class_id
        # get all the GT class names
        class_names = [self.class_id_names[id] for id in self.class_id_names]
        class_ids = [id for id in self.class_id_names]
        # prompt the class names into a sequence
        # Then generate the clip feat for each GT class name
        class_names_feats = get_text_features(self.clip_length, class_names, self.clip_model, self.clip_tokenizer)

        obj_map = []
        for ply_path in ply_path_list:
            plydata = PlyData.read(os.path.join(ply_dir, ply_path))
            points = np.vstack([plydata['vertex'][dim] for dim in ('x', 'y', 'z')]).T
            colors = np.vstack([plydata['vertex'][dim] for dim in ('red', 'green', 'blue')]).T / 255.0
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            clip_feat = feats[int(ply_path.split('_')[1].split('.')[0]), :]
            # clip_feat = clip_feat / np.linalg.norm(clip_feat)
            clip_feat = clip_feat.reshape(1, -1)
            # calculate the cosine similarity between obj_feats and class_names_feats
            sim = cosine_similarity(clip_feat, class_names_feats)
            # Determine the top-N matches for each object in obj_map
            # for scannet, need refactoring
            class_id = class_ids[np.argmax(sim)]
            if class_id in self.ignore_ids:
                continue
            # print(self.class_id_names[class_id])
            obj_map.append(LoadedObject(pcd, clip_feat, class_id))
        
        logging.info(f"Successfully loaded {len(obj_map)} objects")
        self.obj_map = copy.deepcopy(obj_map)

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


@measure_time_block("Evaluation Time")
@hydra.main(version_base=None, config_path="../config/", config_name="seg_evaluation")
def main(cfg: DictConfig):
    evaluator = HOVEvaluator(cfg)
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
