#!/bin/bash

cd "$(dirname "$0")"

EXPERIMENTS_DIR="./experiments"
CONTAINER="scyard.sif"

for EXP_DIR in "$EXPERIMENTS_DIR"/*/; do
    EXP_NAME=$(basename "$EXP_DIR")

    echo "Running experiment: $EXP_NAME"

    CONFIG_FILE="${EXP_DIR}/config.yml"
    LOG_DIR="${EXP_DIR}/logs"
    SAVE_PATH="${EXP_DIR}/maml_policy.pth"

    mkdir -p "$LOG_DIR"

    apptainer run \
        --bind ./:/app \
        --bind "$EXP_DIR:/app/experiment" \
        "$CONTAINER" \
        python /app/main.py "$@" \
        --config "/app/experiment/config.yml" \
        --log_dir "/app/experiment/logs" \
        --save_path "/app/experiment/maml_policy.pth" \
        --wandb_api_key "$WANDB_API_KEY" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_entity "$WANDB_ENTITY" \
        --wandb_run_name "$EXP_NAME" \
        --wandb_resume


    echo "Experiment $EXP_NAME completed."
    echo "Logs and model saved in $EXP_DIR"
    echo "---------------------------------------"
done
