#!/bin/bash

# Define scene ID list
scene_ids=("office_0" "office_1" "office_2" "office_3" "office_4" "room_0" "room_1" "room_2")
hov_result_dir="./output/sem_seg_new_mobileclip"

# Iterate through each scene_id
for scene_id in "${scene_ids[@]}"; do
    echo "Running scripts for scene_id=${scene_id}"
    hov_result_path="$hov_result_dir/${scene_id//_/}/replica"
    
    # Execute the first script
    python applications/generate_replica_class_color.py scene_id=$scene_id
    if [ $? -ne 0 ]; then
        echo "Error occurred while running generate_replica_class_color.py for scene_id=${scene_id}"
        exit 1
    fi

    # Execute the third script
    python -m evaluation.hov_sem_eval scene_id=$scene_id hov_result_path=$hov_result_path
    if [ $? -ne 0 ]; then
        echo "Error occurred while running hov_eval.py for scene_id=${scene_id}"
        exit 1
    fi

    echo "Finished processing scene_id=${scene_id}"
done

echo "All scripts executed successfully for all scene_ids!"