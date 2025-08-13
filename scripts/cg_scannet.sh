#!/bin/bash

# Define scene ID list
scene_ids=("scene0011_00" "scene0050_00" "scene0231_00" "scene0378_00" "scene0518_00")
cg_result_dir="./output/scannet_cg_vit"

# Iterate through each scene_id
for scene_id in "${scene_ids[@]}"; do
    echo "Running scripts for scene_id=${scene_id}"
    cg_result_path="$cg_result_dir/${scene_id}/${scene_id}.pkl"

    # Execute the third script
    python -m evaluation.cg_sem_eval scene_id=$scene_id cg_result_path=$cg_result_path
    if [ $? -ne 0 ]; then
        echo "Error occurred while running cg_sem_eval.py for scene_id=${scene_id}"
        exit 1
    fi

    echo "Finished processing scene_id=${scene_id}"
done

echo "All scripts executed successfully for all scene_ids!"