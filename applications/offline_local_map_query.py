import copy
import json
import os
import sys

import hydra
import matplotlib
import numpy as np
import open3d as o3d
import open_clip
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from utils.object import BaseObject


@hydra.main(version_base=None, config_path="../config/", config_name="query_config")
def main(cfg: DictConfig):

    ### Loading Color map: class id --> color Dict
    if cfg.yolo.use_given_classes:
        given_classes_path = cfg.yolo.given_classes_path
        dir_path = os.path.dirname(given_classes_path)  # './model'
        base_name = os.path.basename(given_classes_path)  # 'gpt_indoor_table.txt'
        file_root, _ = os.path.splitext(base_name)  # 'gpt_indoor_table'

        class_id_colors_path = os.path.join(dir_path, file_root + "_id_colors.json")

    else:
        class_id_colors_path = os.path.join(
            cfg.output_path,
            f"{cfg.dataset_name}_{cfg.scene_id}",
            "classes_info",
            f"{cfg.dataset_name}_{cfg.scene_id}_id_colors.json",
        )

    print("Loading classes id --> colors from: {}".format(class_id_colors_path))

    if not os.path.exists(class_id_colors_path):
        raise FileNotFoundError(f"Error: File not found: {class_id_colors_path}")

    class_id_colors = {}
    with open(class_id_colors_path, "r") as file:
        class_id_colors = json.load(file)
    class_id_colors = {int(key): value for key, value in class_id_colors.items()}

    # Dict: class id --> name
    class_id_names = {}

    if cfg.yolo.use_given_classes:
        class_id_names_path = cfg.yolo.given_classes_path
        # Load class names from txt
        with open(class_id_names_path, "r") as f:
            class_list = [line.strip() for line in f if line.strip()]
        class_id_names = {i: name for i, name in enumerate(class_list)}
    else:
        class_id_names_path = os.path.join(
            cfg.output_path,
            f"{cfg.dataset_name}_{cfg.scene_id}",
            "classes_info",
            f"{cfg.dataset_name}_{cfg.scene_id}_id_names.json",
        )

        with open(class_id_names_path, "r") as file:
            class_id_names = json.load(file)
        class_id_names = {int(key): value for key, value in class_id_names.items()}

    print("Loading classes id --> names  from: {}".format(class_id_names_path))

    if not os.path.exists(class_id_names_path):
        raise FileNotFoundError(f"Error: File not found: {class_id_names_path}")

    ### Loading saved results
    # if map_dir is not provided, use the default path
    load_dir = None
    if os.path.exists(cfg.map_dir):
        load_dir = cfg.map_dir
    else:
        load_dir = os.path.join(
            cfg.output_path, f"{cfg.dataset_name}_{cfg.scene_id}", "map"
        )

    if not os.path.exists(load_dir):
        print(f"Error: {load_dir} does not exist.")
        sys.exit(1)

    print(("Loading saved obj results from: {}".format(load_dir)))

    ### Loading viewpoint
    viewpoint_path = os.path.join(load_dir, "viewpoint.json")
    print(f"Loading viewpoint from: {viewpoint_path}")

    # traverse the .pkl in the directory to get constructed maps
    obj_map = []
    for file in os.listdir(load_dir):
        if file.endswith(".pkl"):
            obj_results_path = os.path.join(load_dir, file)
            # object construction
            loaded_obj = BaseObject.load_from_disk(obj_results_path)
            obj_map.append(loaded_obj)
    print(f"Successfully loaded {len(obj_map)} objects")

    ### Init of CLIP
    print("Loading CLIP model")
    # clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    #     "ViT-H-14", "laion2b_s32b_b79k"
    # )
    # clip_model = clip_model.to("cuda")
    # clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

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

    print("Done initializing CLIP model.")

    cmap = matplotlib.colormaps.get_cmap("turbo")

    ### Set the visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    # Create window
    vis.create_window(window_name=f"Offline Visualization", width=1920, height=1920)

    for obj in obj_map:
        vis.add_geometry(obj.pcd)

    print(f"Obj Map length: %d" % len(obj_map))

    view_param = None
    if os.path.exists(viewpoint_path):
        print(f"Loading saved viewpoint from {viewpoint_path}")
        view_param = o3d.io.read_pinhole_camera_parameters(viewpoint_path)
        vis.get_view_control().convert_from_pinhole_camera_parameters(view_param)

    # variables for query sim color map
    queried_color_objs = []
    highlighted_objs = []

    def pcd_sem_color_callback(vis):
        print("Show the Pointcloud with semantic colors")
        vis.clear_geometries()
        for obj in obj_map:
            sem_pcd = copy.deepcopy(obj.pcd)
            color = class_id_colors[obj.class_id]
            vis.add_geometry(sem_pcd.paint_uniform_color(color))
        reset_view()

    def pcd_rgb_color_callback(vis):
        print("Show the Pointcloud with RGB colors")
        vis.clear_geometries()
        for obj in obj_map:
            vis.add_geometry(obj.pcd)
        reset_view()

    ### Visualization exit
    def vis_exit_callback(vis):
        vis.destroy_window()
        sys.exit(0)

    def query_callback(vis):
        text_query = input("Enter your query: ")

        # exit the querying
        if text_query == "exit" or text_query == "quit":
            vis.destroy_window()
            sys.exit(0)

        text_queries = [text_query]

        text_queries_tokenized = clip_tokenizer(text_queries).to("cuda")
        text_query_ft = clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()

        ## Get stacked clip feats from the map
        values = []
        for obj in obj_map:
            values.append(torch.from_numpy(obj.clip_ft))
        map_clip_fts = torch.stack(values, dim=0).to("cuda")

        ## claculate the cos sim between text clip and map clips
        cos_sim = F.cosine_similarity(text_query_ft.unsqueeze(0), map_clip_fts, dim=-1)

        ## Get top k candidates
        top_k = 1
        top_k_cos_sim, top_k_idx = torch.topk(cos_sim, top_k, dim=0)
        print("Top 5 similar objects:")
        for i, (cos_val, idx) in enumerate(
            zip(top_k_cos_sim.tolist(), top_k_idx.tolist())
        ):
            print(
                f"{i+1}. No. {idx} {class_id_names[obj_map[idx].class_id]}: {cos_val:.3f}"
            )

        ## Save the highlighted objects with top5 in red color
        global highlighted_objs
        highlighted_objs = []
        for idx, obj in enumerate(obj_map):
            temp_obj = copy.deepcopy(obj)
            if idx in top_k_idx.tolist():
                color = [1.0, 0.0, 0.0]  # Red color
                temp_obj.pcd.paint_uniform_color(color)
            highlighted_objs.append(temp_obj)

        max_value = cos_sim.max()
        min_value = cos_sim.min()
        normalized_similarities = (cos_sim - min_value) / (max_value - min_value)
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[
            ..., :3
        ]

        ## Save the colored objects
        global queried_color_objs
        queried_color_objs = []
        for idx, obj in enumerate(obj_map):
            temp_obj = copy.deepcopy(obj)
            # change the color in temp_obj
            temp_obj.pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    [
                        similarity_colors[idx, 0].item(),
                        similarity_colors[idx, 1].item(),
                        similarity_colors[idx, 2].item(),
                    ],
                    (len(temp_obj.pcd.points), 1),
                )
            )
            queried_color_objs.append(temp_obj)

        ### visualization
        vis.clear_geometries()
        for obj in highlighted_objs:
            vis.add_geometry(obj.pcd)

        reset_view()

    def highlight_objs_callback(vis):
        global highlighted_objs
        vis.clear_geometries()
        for obj in highlighted_objs:
            vis.add_geometry(obj.pcd)
        reset_view()

    def queried_color_objs_callback(vis):
        global queried_color_objs
        vis.clear_geometries()
        for obj in queried_color_objs:
            vis.add_geometry(obj.pcd)
        reset_view()

    def help_callback(vis):
        help_info = """
        Keybindings:
        Q - Quit the application
        R - Display the point cloud with RGB colors
        C - Display the point cloud with semantic colors
        F - Enter a query to find top similarity objects
        H - Display this help message
        N - Highlight objects based on previous query results
        M - Colored objects based on previous query results
        S - Save the current viewpoint

        Press the corresponding key to perform the action.
        """
        print(help_info)

    def save_view_callback(vis):
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(viewpoint_path, param)
        print(f"Viewpoint saved to {viewpoint_path}")

    def reset_view():
        if view_param is not None:
            vis.get_view_control().convert_from_pinhole_camera_parameters(view_param)

    vis.register_key_callback(ord("Q"), vis_exit_callback)
    vis.register_key_callback(ord("R"), pcd_rgb_color_callback)
    vis.register_key_callback(ord("C"), pcd_sem_color_callback)
    vis.register_key_callback(ord("F"), query_callback)
    vis.register_key_callback(ord("N"), highlight_objs_callback)
    vis.register_key_callback(ord("M"), queried_color_objs_callback)
    vis.register_key_callback(ord("H"), help_callback)
    vis.register_key_callback(ord("S"), save_view_callback)

    help_callback(vis)

    vis.run()


if __name__ == "__main__":
    main()
