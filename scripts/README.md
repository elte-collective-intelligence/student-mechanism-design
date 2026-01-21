# Shell Scripts

This directory contains convenience scripts for running common tasks like training, evaluation, and batch experiments.

## Files

### `run_experiment.sh`
**Run a single experiment by name.** This is the primary script for executing experiments.

**Usage:**
```bash
./scripts/run_experiment.sh EXPERIMENT_NAME
```

**Examples:**
```bash
# Quick training test
./scripts/run_experiment.sh smoke_train

# Full training run
./scripts/run_experiment.sh full_train

# Evaluation with visualization
./scripts/run_experiment.sh smoke_eval_vis
```

**What it does:**
1. Validates experiment directory exists
2. Checks for config file at `src/configs/experiments/{NAME}/config.yml`
3. Runs: `python src/main.py --config {path_to_config}`
4. Logs output to console and file
5. Creates timestamped log file in experiment directory

**Error Handling:**
- Missing experiment → Shows available experiments
- Missing config.yml → Error message with expected path
- Python errors → Captured in log file

**Script Logic:**
```bash
#!/bin/bash
EXPERIMENT=$1

if [ -z "$EXPERIMENT" ]; then
    echo "Usage: ./scripts/run_experiment.sh EXPERIMENT_NAME"
    exit 1
fi

CONFIG="src/configs/experiments/${EXPERIMENT}/config.yml"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config not found at $CONFIG"
    exit 1
fi

python src/main.py --config $CONFIG
```

### `train_all.sh`
**Run multiple training experiments sequentially.** Useful for training different configurations overnight.

**Usage:**
```bash
./scripts/train_all.sh
```

**What it does:**
1. Defines list of experiments to run
2. Runs each experiment sequentially
3. Logs each experiment separately
4. Continues even if one experiment fails
5. Reports summary at the end

**Default Experiments:**
```bash
EXPERIMENTS=(
    "smoke_train"
    "small_train"
    "medium_train"
    "full_train"
)
```

**Customize:**
Edit the script to add/remove experiments:
```bash
# Edit train_all.sh
EXPERIMENTS=(
    "my_experiment_1"
    "my_experiment_2"
    "my_experiment_3"
)
```

**Output:**
```
========================================
Running experiment: smoke_train
========================================
[training output...]
Experiment smoke_train completed.

========================================
Running experiment: full_train
========================================
[training output...]
Experiment full_train completed.

========================================
All experiments completed!
========================================
```

**Use Cases:**
- Run multiple training configurations overnight
- Hyperparameter search (different configs)
- Reproduce paper results
- Systematic comparison studies

**Tips:**
- Start with quick experiments first (if one fails, you know early)
- Use tmux/screen for long-running jobs
- Check disk space before starting (models can be large)

### `eval_all.sh`
**Evaluate all trained models.** Runs evaluation for experiments with saved checkpoints.

**Usage:**
```bash
./scripts/eval_all.sh
```

**What it does:**
1. Searches for experiments with saved models (*.pt files)
2. For each experiment, creates/runs evaluation config
3. Generates evaluation reports
4. Compiles aggregate statistics

**Evaluation Process:**
```bash
# For each experiment directory with models:
for exp in src/configs/experiments/*/logs/*.pt; do
    exp_name=$(dirname $(dirname $exp))
    eval_config="${exp_name}_eval/config.yml"
    
    if [ -f "$eval_config" ]; then
        ./scripts/run_experiment.sh "${exp_name}_eval"
    fi
done
```

**Requirements:**
- Trained models must exist (*.pt files)
- Corresponding eval configs must exist (*_eval/config.yml)
- Eval configs must reference correct model paths

**Output:**
- Evaluation logs for each experiment
- Aggregate metrics file: `evaluation_summary.txt`
- Comparison table across experiments

**Example Output:**
```
Evaluating: smoke_train
  MrX Win Rate: 45%
  Police Win Rate: 55%
  Avg Episode Length: 42.3 steps

Evaluating: full_train
  MrX Win Rate: 51%
  Police Win Rate: 49%
  Avg Episode Length: 68.7 steps

Summary saved to: evaluation_summary.txt
```

## Running Scripts in Docker

All scripts can be run inside Docker:

**Single Experiment:**
```bash
docker run --rm --gpus=all \
    --mount type=bind,src=$PWD,dst=/app \
    student_mechanism_design \
    bash -c "./scripts/run_experiment.sh smoke_train"
```

**Multiple Experiments:**
```bash
docker run --rm --gpus=all \
    --mount type=bind,src=$PWD,dst=/app \
    student_mechanism_design \
    bash -c "./scripts/train_all.sh"
```

**Interactive Shell:**
```bash
docker run --rm -it --gpus=all \
    --mount type=bind,src=$PWD,dst=/app \
    student_mechanism_design
# Then inside container:
./scripts/run_experiment.sh smoke_train
```

## Script Best Practices

**1. Make Scripts Executable**
```bash
chmod +x scripts/*.sh
```

**2. Run from Project Root**
```bash
# Good (from project root)
./scripts/run_experiment.sh smoke_train

# Bad (from scripts directory)
cd scripts/
./run_experiment.sh smoke_train  # Wrong paths!
```

**3. Check for Errors**
```bash
# Run script and check exit code
./scripts/run_experiment.sh smoke_train
if [ $? -eq 0 ]; then
    echo "Success!"
else
    echo "Failed!"
fi
```

**4. Redirect Output**
```bash
# Save output to file
./scripts/train_all.sh > training_log.txt 2>&1

# Save and display simultaneously
./scripts/train_all.sh 2>&1 | tee training_log.txt
```

## Creating Custom Scripts

### Template for New Script

```bash
#!/bin/bash
# Script name: my_script.sh
# Description: What this script does

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
EXPERIMENTS=(
    "exp1"
    "exp2"
)

# Main logic
for EXP in "${EXPERIMENTS[@]}"; do
    echo "Processing: $EXP"
    
    # Your commands here
    ./scripts/run_experiment.sh $EXP
    
    echo "Completed: $EXP"
done

echo "All done!"
```

### Make it Executable
```bash
chmod +x scripts/my_script.sh
```

### Test it
```bash
./scripts/my_script.sh
```

## Common Script Tasks

### Task 1: Run Hyperparameter Search
```bash
#!/bin/bash
# hyperparam_search.sh

LEARNING_RATES=(0.0001 0.001 0.01)
HIDDEN_DIMS=(64 128 256)

for LR in "${LEARNING_RATES[@]}"; do
    for DIM in "${HIDDEN_DIMS[@]}"; do
        EXP_NAME="gnn_lr${LR}_dim${DIM}"
        echo "Running: $EXP_NAME"
        
        # Create config on-the-fly
        # Or use pre-created configs
        ./scripts/run_experiment.sh $EXP_NAME
    done
done
```

### Task 2: Evaluate Best Checkpoint
```bash
#!/bin/bash
# eval_best.sh

# Find best model by win rate
BEST_MODEL=$(find src/configs/experiments/*/logs/*.pt -type f | head -1)
BEST_EXP=$(dirname $(dirname $BEST_MODEL))

echo "Best model found in: $BEST_EXP"

# Run evaluation
./scripts/run_experiment.sh "${BEST_EXP}_eval"
```

### Task 3: Clean Old Logs
```bash
#!/bin/bash
# clean_logs.sh

# Remove logs older than 7 days
find src/configs/experiments/*/logs/ -name "*.log" -mtime +7 -delete

# Remove large checkpoint files (keep only best)
# Implementation depends on your checkpoint strategy
```

## Troubleshooting

### Script Won't Run
```bash
# Check permissions
ls -l scripts/run_experiment.sh

# Make executable if needed
chmod +x scripts/run_experiment.sh
```

### Wrong Working Directory
```bash
# Scripts assume running from project root
# Check current directory
pwd

# Should be: .../student-mechanism-design
# Not: .../student-mechanism-design/scripts
```

### Missing Dependencies
```bash
# If script uses jq, dos2unix, etc.
sudo apt-get install jq dos2unix  # Linux
brew install jq dos2unix          # macOS
```

### Script Fails Mid-Run
```bash
# Add error handling
set -e  # Exit on any error
set -x  # Print commands before executing (debug mode)

# Or handle errors manually
if ! ./scripts/run_experiment.sh exp1; then
    echo "exp1 failed, continuing..."
fi
```

## Tips for Students

1. **Read scripts before running**: Understand what they do
2. **Start with small tests**: Use smoke tests before full runs
3. **Monitor progress**: Check logs while scripts run
4. **Use tmux/screen**: For long-running scripts (don't lose progress if SSH disconnects)
5. **Save outputs**: Redirect to log files for later analysis
6. **Test in Docker**: Ensure reproducibility
7. **Modify conservatively**: Make small changes and test
8. **Version control scripts**: Commit scripts alongside code
9. **Document changes**: Add comments explaining modifications
10. **Share successful configs**: Help classmates by sharing working scripts

## Advanced Usage

### Parallel Execution
```bash
#!/bin/bash
# Run experiments in parallel (if you have multiple GPUs)

./scripts/run_experiment.sh smoke_train &
./scripts/run_experiment.sh small_train &
wait  # Wait for all background jobs to complete
echo "All experiments finished"
```

### Conditional Execution
```bash
#!/bin/bash
# Only evaluate if training succeeded

if ./scripts/run_experiment.sh smoke_train; then
    echo "Training succeeded, running evaluation"
    ./scripts/run_experiment.sh smoke_train_eval
else
    echo "Training failed, skipping evaluation"
fi
```

### Resource Monitoring
```bash
#!/bin/bash
# Monitor GPU usage during training

watch -n 5 nvidia-smi &
WATCH_PID=$!

./scripts/run_experiment.sh full_train

kill $WATCH_PID
```
