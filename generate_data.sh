#!/bin/bash

# Bayes-BR, Bayes-TD-based
CONFIG_FILES=(
    "configs/config_deepsea_5x5_det.yaml"
    "configs/config_deepsea_5x5_det_bs0.yaml"
    "configs/config_deepsea_5x5_det_bs1.yaml"
    "configs/config_deepsea_5x5_det_bs2_100.yaml"
    "configs/config_deepseapyramid_5x5_det.yaml"
    "configs/config_deepseapyramid_5x5_det_bs0.yaml"
    "configs/config_deepseapyramid_5x5_det_bs1.yaml"
    "configs/config_deepseapyramid_5x5_det_bs2_100.yaml"
    "configs/config_deepseaswirl_5x5_det.yaml"
    "configs/config_deepseaswirl_5x5_det_bs0.yaml"
    "configs/config_deepseaswirl_5x5_det_bs1.yaml"
    "configs/config_deepseaswirl_5x5_det_bs2_100.yaml"
    "configs/config_deepseaswirl_5x5_det_sigma1.yaml"
    "configs/config_deepseaswirl_5x5_det_sigma1_bs0.yaml"
    "configs/config_deepseaswirl_5x5_det_sigma1_bs1.yaml"
    "configs/config_deepseaswirl_5x5_det_sigma1_bs2_100.yaml"
    "configs/config_deepsea_4x4_sto.yaml"
    "configs/config_deepsea_4x4_sto_bs0.yaml"
    "configs/config_deepsea_4x4_sto_bs1.yaml"
    "configs/config_deepsea_4x4_sto_bs2_10.yaml"
)

BASE_DIR="./data"
N_JOBS=10
BATCH_SIZE=1024
NUM_EXPERIMENTS=10

EPSILONS=(0.01 0.1 0.2 0.5 1.0 5.0)

for config in "${CONFIG_FILES[@]}"; do
    for eps in "${EPSILONS[@]}"; do
        echo "----------------------------------------------------------------"
        echo "Running Experiment: Config=$config, Epsilon=$eps"
        echo "----------------------------------------------------------------"
        
        python generate_data.py "$config" \
            --base_dir "$BASE_DIR" \
            --n_jobs "$N_JOBS" \
            --epsilon "$eps" \
            --batch_size "$BATCH_SIZE" \
            --num_experiments "$NUM_EXPERIMENTS"
    done
done


# HMC for sigma=10
STEPSIZES=(0.0001 0.005 0.01 0.03 0.05 0.1)

HMC_CONFIG_FILES=(
    "configs/config_deepsea_5x5_det_hmc.yaml"
    "configs/config_deepseapyramid_5x5_det_hmc.yaml"
    "configs/config_deepseaswirl_5x5_det_hmc.yaml"
    "configs/config_deepsea_4x4_sto_hmc.yaml"
)

for config in "${HMC_CONFIG_FILES[@]}"; do

    for i in "${!EPSILONS[@]}"; do
        eps="${EPSILONS[$i]}"
        step="${STEPSIZES[$i]}"

        echo "----------------------------------------------------------------"
        echo "Running Experiment: Config=$config, StepSize=$step"
        echo "----------------------------------------------------------------"
        
        python generate_data.py "$config" \
            --base_dir "$BASE_DIR" \
            --n_jobs "$N_JOBS" \
            --epsilon "$eps" \
            --step_size "$step" \
            --num_experiments "$NUM_EXPERIMENTS" \
            --disable_hmc_progbar
    done
done


# HMC for sigma=10 with alternative tuning
STEPSIZES=(0.0001 0.005)
HMC_CONFIG_ADDS=(
    "configs/config_deepsea_5x5_det_hmc_001.yaml"
    "configs/config_deepsea_5x5_det_hmc_01.yaml"
)

for i in "${!HMC_CONFIG_ADDS[@]}"; do
    config="${HMC_CONFIG_ADDS[$i]}"
    step="${STEPSIZES[$i]}"

    echo "----------------------------------------------------------------"
    echo "Running Experiment: Config=$config"
    echo "----------------------------------------------------------------"
    
    python generate_data.py "$config" \
        --base_dir "$BASE_DIR" \
        --n_jobs "$N_JOBS" \
        --step_size "$step" \
        --num_experiments "$NUM_EXPERIMENTS" \
        --disable_hmc_progbar
done


# HMC for sigma=1
STEPSIZES=(0.001 0.05 0.1 0.3 0.5 1.0)
HMC_CONFIG_ADDS=(
    "configs/config_deepseaswirl_5x5_det_sigma1_hmc.yaml"
)

for config in "${HMC_CONFIG_ADDS[@]}"; do
    for i in "${!EPSILONS[@]}"; do
        eps="${EPSILONS[$i]}"
        step="${STEPSIZES[$i]}"

        echo "----------------------------------------------------------------"
        echo "Running HMC Experiment: Config=$config, Epsilon=$eps, StepSize=$step"
        echo "----------------------------------------------------------------"
        
        python generate_data.py "$config" \
            --base_dir "$BASE_DIR" \
            --n_jobs "$N_JOBS" \
            --epsilon "$eps" \
            --step_size "$step" \
            --num_experiments "$NUM_EXPERIMENTS" \
            --disable_hmc_progbar
    done
done
