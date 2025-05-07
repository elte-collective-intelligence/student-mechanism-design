#!/bin/bash

cd "$(dirname "$0")"

EXPERIMENTS_DIR="./experiments"
CONTAINER="scyard.sif"

export CUDA_LAUNCH_BLOCKING=1

for EXP_DIR in "$EXPERIMENTS_DIR"/*/; do
    EXP_NAME=$(basename "$EXP_DIR")

    echo "Running experiment: $EXP_NAME"

    CONFIG_FILE="${EXP_DIR}/config.yml"
    LOG_DIR="${EXP_DIR}/logs"
    SAVE_PATH="${EXP_DIR}/maml_policy.pth"

    mkdir -p "$LOG_DIR"

    apptainer run \
        --nv \
        --bind ./:/app \
        --bind "$EXP_DIR:/app/src/experiments" \
        "$CONTAINER" \
        python /app/src/main.py "$@" \
        --config "/app/src/experiments/config.yml" \
        --log_dir "/app/src/experiments/logs" \
        --wandb_api_key "$WANDB_API_KEY" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --wandb_run_name "$EXP_NAME" 
        # --wandb_resume


    echo "Experiment $EXP_NAME completed."
    echo "Logs and model saved in $EXP_DIR"
    echo "---------------------------------------"
done
