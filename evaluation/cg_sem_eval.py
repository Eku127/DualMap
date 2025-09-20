import copy
import logging
import os
import pickle
import sys

import hydra
import numpy as np
import open3d as o3d
from omegaconf import DictConfig

from evaluation.sem_seg_eval import Evaluator
from utils.eval.eval_utils import *
from utils.eval.metric import *
from utils.eval.scannet200_constants import *
from utils.time_utils import measure_time_block


class LoadedObject:
    def __init__(self, pcd, clip_ft, class_id):
        self.pcd = pcd
        self.class_id = class_id
        self.clip_ft = clip_ft


class CGEvaluator(Evaluator):
    def __init__(self, cfg):
        super(CGEvaluator, self).__init__(cfg)
        self.cg_result_path = cfg.cg_result_path

    def load_map(self):
        # Read map built by ConceptGraphs
        # check whether load_dir exists
        if not os.path.exists(self.cg_result_path):
            logging.info(f"Error: {self.cg_result_path} does not exist.")
            sys.exit(1)

        logging.info("Loading saved obj results from: {}".format(self.cg_result_path))

        pkl_path = self.cg_result_path

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        objects = data["objects"]
        # print(data['class_names'])
        obj_map = []

        for obj in objects:
            pts = obj["pcd_np"]
            colors = obj["pcd_color_np"]
            # print("color shape:", colors.shape)
            class_id_list = obj["class_id"]
            class_id = max(class_id_list, key=class_id_list.count)
            # print("class_id:", class_id)
            clip_feat = obj["clip_ft"].reshape(1, -1)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)

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

            o3d.io.write_point_cloud(
                os.path.join(self.output_dir, "pred_rgb_pcd_full.pcd"), rgb_full_pcd
            )
            o3d.io.write_point_cloud(
                os.path.join(self.output_dir, "pred_seg_pcd_full.pcd"), seg_full_pcd
            )


@measure_time_block("Evaluation Time")
@hydra.main(version_base=None, config_path="../config/", config_name="seg_evaluation")
def main(cfg: DictConfig):
    evaluator = CGEvaluator(cfg)
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
