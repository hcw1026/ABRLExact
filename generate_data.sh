#!/bin/bash

CONFIG_FILES=(
    "configs/config_deepsea_5x5_det.yaml"
)

BASE_DIR="./data"
N_JOBS=10
BATCH_SIZE=1024
NUM_EXPERIMENTS=10

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
            --batch_size "$BATCH_SIZE"\
            --num_experiments "$NUM_EXPERIMENTS"
    done
done

HMC_CONFIG="configs/config_deepsea_5x5_det_hmc.yaml"
STEPSIZES=(0.0001 0.005 0.05 0.1)

for i in "${!EPSILONS[@]}"; do
    eps="${EPSILONS[$i]}"
    step="${STEPSIZES[$i]}"

    echo "----------------------------------------------------------------"
    echo "Running HMC Experiment: Config=$HMC_CONFIG, Epsilon=$eps, StepSize=$step"
    echo "----------------------------------------------------------------"
    
    python generate_data.py "$HMC_CONFIG" \
        --base_dir "$BASE_DIR" \
        --n_jobs "$N_JOBS" \
        --epsilon "$eps" \
        --step_size "$step" \
        --batch_size "$BATCH_SIZE" \
        --num_experiments "$NUM_EXPERIMENTS"
done