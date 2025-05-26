#!/bin/bash

cd "$(dirname "$0")"

EXPERIMENTS_DIR="./src/experiments"

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
echo "Configuration file: $CONFIG_FILE"
echo "Log directory: $LOG_DIR"
python ./src/torch_rl_main.py "$@" \
    --config "$CONFIG_FILE" \
    --log_dir "$LOG_DIR"

if [ $? -eq 0 ]; then
    echo "Experiment '$EXP_NAME' completed successfully."
    echo "Logs and model saved in '$EXP_DIR'."
else
    echo "Experiment '$EXP_NAME' failed. Check the logs for details."
    exit 1
fi

echo "---------------------------------------"
