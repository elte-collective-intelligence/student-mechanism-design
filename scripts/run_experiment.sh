#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Detect if running inside Docker or locally
if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    ROOT_DIR="/app"
    IN_DOCKER=true
else
    ROOT_DIR="$PROJECT_ROOT"
    IN_DOCKER=false
fi

SRC_DIR="${ROOT_DIR}/src"
ROOT_EXP_DIR="${SRC_DIR}/configs/experiments"

if [ $# -lt 1 ]; then
    echo "Error: No experiment name provided!"
    echo "Usage: $0 <experiment_name> [additional_args...]"
    echo ""
    echo "Available experiments:"
    for exp in "$ROOT_EXP_DIR"/*/; do
        [ -d "$exp" ] && echo "  - $(basename "$exp")"
    done
    exit 1
fi

if [ "$1" == "--unit_test" ]; then
    cd "$SRC_DIR"
    pytest
    exit 0
fi

EXP_NAME="$1"
shift
EXP_DIR="${ROOT_EXP_DIR}/${EXP_NAME}"

if [ ! -d "$EXP_DIR" ]; then
    echo "Error: Experiment directory '$EXP_DIR' does not exist."
    echo ""
    echo "Available experiments:"
    for exp in "$ROOT_EXP_DIR"/*/; do
        [ -d "$exp" ] && echo "  - $(basename "$exp")"
    done
    exit 1
fi

echo "Running experiment: $EXP_NAME"
echo "  Environment: $([ "$IN_DOCKER" = true ] && echo "Docker" || echo "Local")"

CONFIG_FILE="${EXP_DIR}/config.yml"
LOG_DIR="${EXP_DIR}/logs"
mkdir -p "$LOG_DIR"

# Load WANDB credentials if available (using Python instead of jq for portability)
WANDB_DATA="${SRC_DIR}/wandb_data.json"
if [ -f "$WANDB_DATA" ]; then
    WANDB_API_KEY=$(python -c "import json; data=json.load(open('$WANDB_DATA')); print(data.get('wandb_api_key', '') or '')" 2>/dev/null)
    WANDB_PROJECT=$(python -c "import json; data=json.load(open('$WANDB_DATA')); print(data.get('wandb_project', '') or '')" 2>/dev/null)
    WANDB_ENTITY=$(python -c "import json; data=json.load(open('$WANDB_DATA')); print(data.get('wandb_entity', '') or '')" 2>/dev/null)
    if [ -z "$WANDB_API_KEY" ]; then
        echo "  WANDB: disabled"
        WANDB_API_KEY=null
        WANDB_PROJECT=null
        WANDB_ENTITY=null
    else
        echo "  WANDB: enabled"
    fi
else
    echo "  WANDB: disabled (no credentials file)"
    WANDB_API_KEY=null
    WANDB_PROJECT=null
    WANDB_ENTITY=null
fi

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"

python "${SRC_DIR}/main.py" "$@" \
    --config "$CONFIG_FILE" \
    --wandb_api_key "$WANDB_API_KEY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_run_name "$EXP_NAME"

if [ $? -eq 0 ]; then
    echo ""
    echo "Experiment '$EXP_NAME' completed successfully."
    echo "Logs saved in: $LOG_DIR"
else
    echo ""
    echo "Experiment '$EXP_NAME' failed. Check the logs for details."
    exit 1
fi
