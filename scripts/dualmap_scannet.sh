#!/bin/bash

# Define scene ID list
scene_ids=("scene0011_00" "scene0050_00" "scene0231_00" "scene0378_00" "scene0518_00")

for scene_id in "${scene_ids[@]}"; do
    echo "Running scripts for scene_id=${scene_id}"
    
    # Execute the second script
    python -m applications.runner_dataset scene_id=$scene_id use_rerun=false
    if [ $? -ne 0 ]; then
        echo "Error occurred while running run_mapping.py for scene_id=${scene_id}"
        exit 1
    fi

    # Execute the third script
    python -m evaluation.sem_seg_eval scene_id=$scene_id
    if [ $? -ne 0 ]; then
        echo "Error occurred while running sem_seg_eval.py for scene_id=${scene_id}"
        exit 1
    fi

    echo "Finished processing scene_id=${scene_id}"
done

echo "All scripts executed successfully for all scene_ids!"