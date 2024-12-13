#!/bin/bash

cd "$(dirname "$0")"

EXPERIMENTS_DIR="./experiments"
CONTAINER="scyard.sif"

usage() {
    echo "Usage: $0 <experiment_name> [additional_arguments]"
    echo "Example: $0 experiment1 --learning_rate 0.01"
    exit 1
}

if [ $# -lt 1 ]; then
    echo "Error: No experiment name provided."
    usage
fi

EXP_NAME="$1"
shift  
EXP_DIR="${EXPERIMENTS_DIR}/${EXP_NAME}"

if [ ! -d "$EXP_DIR" ]; then
    echo "Error: Experiment directory '$EXP_DIR' does not exist."
    exit 1
fi

echo "Running experiment: $EXP_NAME"

CONFIG_FILE="${EXP_DIR}/config.yml"
LOG_DIR="${EXP_DIR}/logs"
SAVE_PATH="${EXP_DIR}/maml_policy.pth"

mkdir -p "$LOG_DIR"

apptainer run \
    --nv \
    --bind ./:/app \
    --bind "$EXP_DIR:/app/experiment" \
    "$CONTAINER" \
    python /app/main.py "$@" \
    --config "/app/experiment/config.yml" \
    --log_dir "/app/experiment/logs" \
    --wandb_api_key "$WANDB_API_KEY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_run_name "$EXP_NAME" 
    # --wandb_resume

if [ $? -eq 0 ]; then
    echo "Experiment '$EXP_NAME' completed successfully."
    echo "Logs and model saved in '$EXP_DIR'."
else
    echo "Experiment '$EXP_NAME' failed. Check the logs for details."
    exit 1
fi

echo "---------------------------------------"
