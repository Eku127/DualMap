import json
import os
import pdb
import sys

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config/", config_name="base_config")
def main(cfg: DictConfig):

    print(
        "This app is for classes and color file generation. It is for replica dataset"
    )

    # Generate output path
    classes_info_path = os.path.join(
        cfg.output_path, str(cfg.dataset_name) + "_" + str(cfg.scene_id), "classes_info"
    )
    os.makedirs(classes_info_path, exist_ok=True)

    ### Load Replica Semantics Info
    # Path to GT semantic info
    semantic_info_path = os.path.join(
        cfg.dataset_gt_path, cfg.scene_id, "habitat", "info_semantic.json"
    )

    with open(semantic_info_path) as f:
        semantic_info = json.load(f)

    # Get Dict: class id --> names
    class_id_names = {obj["id"]: obj["name"] for obj in semantic_info["classes"]}
    class_id_names = {0: "background", **class_id_names}

    # Get Dict: class id --> colors
    unique_class_ids = np.unique(list(class_id_names.keys()))
    unique_colors = np.random.rand(len(unique_class_ids), 3)
    class_id_colors = {
        int(class_id): unique_colors[i].tolist()
        for i, class_id in enumerate(unique_class_ids)
    }

    # Get Dict: names --> colors
    names_colors = {
        class_id_names[int(class_id)]: color
        for class_id, color in class_id_colors.items()
    }

    # Save names .txt
    # get path
    names_path = os.path.join(
        classes_info_path, f"{cfg.dataset_name}_{cfg.scene_id}_names.txt"
    )
    print(names_path)
    with open(names_path, "w") as f:
        for class_name in class_id_names.values():
            f.write(f"{class_name}\n")

    # Save names --> colors .json
    # get path
    names_colors_path = os.path.join(
        classes_info_path, f"{cfg.dataset_name}_{cfg.scene_id}_names_colors.json"
    )
    print(names_colors_path)
    with open(names_colors_path, "w") as f:
        json.dump(names_colors, f)

    # Save class idx --> colors .json
    class_id_colors_path = os.path.join(
        classes_info_path, f"{cfg.dataset_name}_{cfg.scene_id}_id_colors.json"
    )
    print(class_id_colors_path)
    with open(class_id_colors_path, "w") as f:
        json.dump(class_id_colors, f)

    # Save class idx --> names .json
    class_id_names_path = os.path.join(
        classes_info_path, f"{cfg.dataset_name}_{cfg.scene_id}_id_names.json"
    )
    print(class_id_names_path)
    with open(class_id_names_path, "w") as f:
        json.dump(class_id_names, f)

    pass


if __name__ == "__main__":
    main()
