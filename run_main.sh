#!/bin/bash
cd "$(dirname "$0")"
if [ $# -lt 1 ]; then
    echo "Error: No experiment name provided!!!"
    exit 1
fi
if [ "$1" == "--unit_test" ]; then
    pytest
    exit 0
fi
ROOT_EXP_DIR="/app/src/experiments"
EXP_NAME="$1"
shift  
EXP_DIR="${ROOT_EXP_DIR}/${EXP_NAME}"
if [ ! -d "$EXP_DIR" ]; then
    echo "Error: Experiment directory '$EXP_DIR' does not exist."
    exit 1
fi
echo "Running experiment: $EXP_NAME"

CONFIG_FILE="${EXP_DIR}/config.yml"
LOG_DIR="${EXP_DIR}/logs"
mkdir -p "$LOG_DIR"
if [ -f ./src/wandb_data.json ]; then
    WANDB_API_KEY=$(jq -r .wandb_api_key src/wandb_data.json)
    WANDB_PROJECT=$(jq -r .wandb_project src/wandb_data.json)
    WANDB_ENTITY=$(jq -r .wandb_entity src/wandb_data.json)
    if [ "$WANDB_API_KEY" == "null" ]; then
        echo "Running without WANDB integration!"
    else
        echo "WANDB credentials loaded succesfully!"
    fi
else
    echo "Running without WANDB integration!"
    WANDB_API_KEY=null
    WANDB_PROJECT=null
    WANDB_ENTITY=null
fi
python /app/src/main.py "$@" \
    --config "$CONFIG_FILE" \
    --exp_dir "$EXP_DIR" \
    --wandb_api_key $WANDB_API_KEY \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --wandb_run_name "$EXP_NAME" 
if [ $? -eq 0 ]; then
    echo "Experiment '$EXP_NAME' completed successfully."
    echo "Logs and model saved in '$EXP_DIR'."
else
    echo "Experiment '$EXP_NAME' failed. Check the logs for details."
    exit 1
fi
echo "========================================"
