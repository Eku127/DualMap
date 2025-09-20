# runner_dataset.py

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from dualmap.core import Dualmap
from utils.dataset import dataset_initialization
from utils.logging_helper import setup_logging
from utils.time_utils import timing_context
from utils.types import DataInput


@hydra.main(version_base=None, config_path="../config/", config_name="runner_dataset")
def main(cfg: DictConfig):
    # Set up the logging system
    setup_logging(output_path=cfg.output_path, config_path=cfg.logging_config)
    logger = logging.getLogger(__name__)

    logger.warning("[Runner Dataset]")
    logger.info(OmegaConf.to_yaml(cfg))

    # Initialize Dualmap instance
    dualmap = Dualmap(cfg)

    # Dataset Object Initialization
    dataset = dataset_initialization(cfg)

    # Traverse and process dataset
    for image_idx in trange(len(dataset), desc="Processing"):
        time_stamp = dataset.time_stamps[image_idx]
        curr_pose = dataset.poses[image_idx].cpu().numpy()

        if cfg.use_stride:
            is_keyframe = True
            kf_idx = image_idx
        else:
            is_keyframe = dualmap.check_keyframe(time_stamp, curr_pose)
            kf_idx = dualmap.get_keyframe_idx()

        if not is_keyframe:
            continue

        logger.info(
            "[Main] ============================================================"
        )
        logger.info(
            f"[Main] Keyframe idx: {kf_idx}, image idx: {image_idx}, Time: {time_stamp}"
        )

        color, color_name = dataset.get_color(image_idx)
        depth = dataset.get_depth(image_idx)
        intrinsics = dataset.get_intrinsics(image_idx)

        curr_data = DataInput(
            idx=kf_idx,
            color=color,
            depth=depth,
            color_name=color_name,
            intrinsics=intrinsics,
            pose=curr_pose,
        )

        # Process data using Dualmap instance
        with timing_context("Time Per Frame", dualmap):
            if cfg.use_parallel:
                dualmap.parallel_process(curr_data)
            else:
                dualmap.sequential_process(curr_data)

    # End-process stage if configured
    if cfg.use_end_process:
        logger.info("[Main][EndProcess] End Processing Start.")
        dualmap.end_process()
        print("Done")


if __name__ == "__main__":
    main()
