#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

EXPERIMENTS_DIR="./src/configs/experiments"
DOCKER_IMAGE="student_mechanism_design"
USE_DOCKER=false
USE_GPU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Evaluate all experiments that have evaluate: True in their config."
            echo "NOTE: Requires trained models in src/artifacts/ or experiment logs/"
            echo ""
            echo "Options:"
            echo "  --docker    Run experiments inside Docker container"
            echo "  --gpu       Enable GPU support (requires --docker and nvidia-docker)"
            echo "  --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Evaluate locally with Python"
            echo "  $0 --docker           # Evaluate in Docker container"
            echo "  $0 --docker --gpu     # Evaluate in Docker with GPU support"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Check if Docker image exists when using Docker mode
if [ "$USE_DOCKER" = true ]; then
    if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
        echo "Docker image '$DOCKER_IMAGE' not found."
        echo "Building Docker image first..."
        docker build --progress plain -f ./docker/Dockerfile -t "$DOCKER_IMAGE" .
    fi
fi

# Find all evaluation experiments (evaluate: True or evaluate: true)
EXPERIMENTS=()
for exp_dir in "$EXPERIMENTS_DIR"/*/; do
    if [ -d "$exp_dir" ] && [ -f "${exp_dir}/config.yml" ]; then
        exp_name="$(basename "$exp_dir")"
        # Check if this is an evaluation experiment
        if grep -qE "^evaluate:\s*(True|true)" "${exp_dir}/config.yml"; then
            EXPERIMENTS+=("$exp_name")
        fi
    fi
done

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    echo "No evaluation experiments found in $EXPERIMENTS_DIR"
    echo "Evaluation experiments should have 'evaluate: True' in their config.yml"
    exit 1
fi

echo "========================================"
echo "Found ${#EXPERIMENTS[@]} evaluation experiment(s):"
for exp in "${EXPERIMENTS[@]}"; do
    echo "  - $exp"
done
echo "========================================"
echo ""

# Run each experiment
FAILED=()
for exp_name in "${EXPERIMENTS[@]}"; do
    echo "========================================"
    echo "Evaluating: $exp_name"
    echo "========================================"
    
    if [ "$USE_DOCKER" = true ]; then
        GPU_FLAG=""
        if [ "$USE_GPU" = true ]; then
            GPU_FLAG="--gpus=all"
        fi
        
        if docker run --rm $GPU_FLAG \
            --mount type=bind,src="$SCRIPT_DIR",dst=/app \
            "$DOCKER_IMAGE" "$exp_name"; then
            echo "✓ Evaluation '$exp_name' completed successfully."
        else
            echo "✗ Evaluation '$exp_name' failed."
            FAILED+=("$exp_name")
        fi
    else
        if ./run_experiment.sh "$exp_name"; then
            echo "✓ Evaluation '$exp_name' completed successfully."
        else
            echo "✗ Evaluation '$exp_name' failed."
            FAILED+=("$exp_name")
        fi
    fi
    echo ""
done

# Summary
echo "========================================"
echo "EVALUATION SUMMARY"
echo "========================================"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "Successful: $((${#EXPERIMENTS[@]} - ${#FAILED[@]}))"
echo "Failed: ${#FAILED[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed experiments:"
    for exp in "${FAILED[@]}"; do
        echo "  - $exp"
    done
    exit 1
fi

echo ""
echo "All evaluation experiments completed successfully!"
