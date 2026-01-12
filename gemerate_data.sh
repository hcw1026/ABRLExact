#!/bin/bash

CONFIG_FILES=(
    "configs/config_deepsea_5x5_det.yaml"
)

BASE_DIR="./data"
N_JOBS=16
BATCH_SIZE=1024

EPSILONS=(0.01 0.1 1.0 5.0)

for config in "${CONFIG_FILES[@]}"; do
    for eps in "${EPSILONS[@]}"; do
        echo "----------------------------------------------------------------"
        echo "Running Experiment: Config=$config, Epsilon=$eps"
        echo "----------------------------------------------------------------"
        
        python generate_data.py "$config" \
            --base_dir "$BASE_DIR" \
            --n_jobs "$N_JOBS" \
            --epsilon "$eps" \
            --batch_size "$BATCH_SIZE"
    done
done