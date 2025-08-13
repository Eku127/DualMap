#!/bin/bash

# Define scene ID list
scene_ids=("scene0011_00" "scene0050_00" "scene0231_00" "scene0378_00" "scene0518_00")
hov_result_dir="./output/sem_seg_new_scannet_mobileclip"

for scene_id in "${scene_ids[@]}"; do
    echo "Running scripts for scene_id=${scene_id}"
    hov_result_path="$hov_result_dir/${scene_id}/scannet"

    python -m evaluation.hov_sem_eval scene_id=$scene_id hov_result_path=$hov_result_path
    if [ $? -ne 0 ]; then
        echo "Error occurred while running hov_eval.py for scene_id=${scene_id}"
        exit 1
    fi

    echo "Finished processing scene_id=${scene_id}"
done

echo "All scripts executed successfully for all scene_ids!"