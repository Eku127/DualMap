#!/bin/bash

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <output_path> <image_weight>"
    exit 1
fi

# Get input arguments
output_path=$1
image_weight=$2

# Define scene ID list
# scene_ids=("office_0" "office_1" "office_2" "office_3" "office_4" "room_0" "room_1" "room_2")
scene_ids=("scene0011_00" "scene0050_00" "scene0231_00" "scene0378_00" "scene0518_00")

# Iterate through each scene_id
for scene_id in "${scene_ids[@]}"; do
    echo "Running scripts for scene_id=${scene_id}"
    
    # # Execute the first script
    # python applications/generate_replica_class_color.py scene_id=$scene_id output_path=$output_path
    # if [ $? -ne 0 ]; then
    #     echo "Error occurred while running generate_replica_class_color.py for scene_id=${scene_id}"
    #     exit 1
    # fi

    # Execute the second script
    python -m applications.runner_dataset scene_id=$scene_id use_rerun=false output_path=$output_path image_weight=$image_weight
    if [ $? -ne 0 ]; then
        echo "Error occurred while running runner_dataset for scene_id=${scene_id}"
        exit 1
    fi

    # Execute the third script
    python -m evaluation.sem_seg_eval scene_id=$scene_id output_path=$output_path
    if [ $? -ne 0 ]; then
        echo "Error occurred while running sem_seg_eval for scene_id=${scene_id}"
        exit 1
    fi

    echo "Finished processing scene_id=${scene_id}"
done

echo "All scripts executed successfully for all scene_ids!"
