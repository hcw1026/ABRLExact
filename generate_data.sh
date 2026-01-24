#!/bin/bash

CONFIG_FILES=(
    "configs/config_deepsea_5x5_det.yaml"
    "configs/config_deepsea_5x5_det_bs.yaml"
)

BASE_DIR="./data"
N_JOBS=10
BATCH_SIZE=1024
NUM_EXPERIMENTS=10

EPSILONS=(0.01 0.1 0.2 0.5 1.0 5.0)

# for config in "${CONFIG_FILES[@]}"; do
#     for eps in "${EPSILONS[@]}"; do
#         echo "----------------------------------------------------------------"
#         echo "Running Experiment: Config=$config, Epsilon=$eps"
#         echo "----------------------------------------------------------------"
        
#         python generate_data.py "$config" \
#             --base_dir "$BASE_DIR" \
#             --n_jobs "$N_JOBS" \
#             --epsilon "$eps" \
#             --batch_size "$BATCH_SIZE" \
#             --num_experiments "$NUM_EXPERIMENTS"
#     done
# done



# HMC_CONFIG="configs/config_deepsea_5x5_det_hmc.yaml"
# STEPSIZES=(0.0001 0.005 0.01 0.03 0.05 0.1)

# for i in "${!EPSILONS[@]}"; do
#     eps="${EPSILONS[$i]}"
#     step="${STEPSIZES[$i]}"

#     echo "----------------------------------------------------------------"
#     echo "Running HMC Experiment: Config=$HMC_CONFIG, Epsilon=$eps, StepSize=$step"
#     echo "----------------------------------------------------------------"
    
#     python generate_data.py "$HMC_CONFIG" \
#         --base_dir "$BASE_DIR" \
#         --n_jobs "$N_JOBS" \
#         --epsilon "$eps" \
#         --step_size "$step" \
#         --num_experiments "$NUM_EXPERIMENTS" \
#         --disable_hmc_progbar
# done

# STEPSIZES=(0.0001 0.005)
# HMC_CONFIG_ADDS=(
#     "configs/config_deepsea_5x5_det_hmc_001.yaml"
#     "configs/config_deepsea_5x5_det_hmc_01.yaml"
# )

# for i in "${!HMC_CONFIG_ADDS[@]}"; do
#     config="${HMC_CONFIG_ADDS[$i]}"
#     step="${STEPSIZES[$i]}"

#     echo "----------------------------------------------------------------"
#     echo "Running Experiment: Config=$config"
#     echo "----------------------------------------------------------------"
    
#     python generate_data.py "$config" \
#         --base_dir "$BASE_DIR" \
#         --n_jobs "$N_JOBS" \
#         --step_size "$step" \
#         --num_experiments "$NUM_EXPERIMENTS" \
#         --disable_hmc_progbar
# done


# DEEPSEAPYRAMID_CONFIG="configs/config_deepseapyramid_5x5_det.yaml"

# echo "----------------------------------------------------------------"
# echo "Running Experiment: Config=$DEEPSEAPYRAMID_CONFIG"
# echo "----------------------------------------------------------------"

# python generate_data.py "$DEEPSEAPYRAMID_CONFIG" \
#     --base_dir "$BASE_DIR" \
#     --n_jobs "$N_JOBS" \
#     --batch_size "$BATCH_SIZE" \
#     --num_experiments "$NUM_EXPERIMENTS"


# DEEPSEASWIRL_CONFIG="configs/config_deepseaswirl_5x5_det.yaml"

# echo "----------------------------------------------------------------"
# echo "Running Experiment: Config=$DEEPSEASWIRL_CONFIG"
# echo "----------------------------------------------------------------"

# python generate_data.py "$DEEPSEASWIRL_CONFIG" \
#     --base_dir "$BASE_DIR" \
#     --n_jobs "$N_JOBS" \
#     --batch_size "$BATCH_SIZE" \
#     --num_experiments "$NUM_EXPERIMENTS"


# DEEPSEASTO_CONFIG="configs/config_deepsea_4x4_sto.yaml"

# echo "----------------------------------------------------------------"
# echo "Running Experiment: Config=$DEEPSEASTO_CONFIG"
# echo "----------------------------------------------------------------"

# python generate_data.py "$DEEPSEASTO_CONFIG" \
#     --base_dir "$BASE_DIR" \
#     --n_jobs "$N_JOBS" \
#     --batch_size "$BATCH_SIZE" \
#     --num_experiments "$NUM_EXPERIMENTS"

