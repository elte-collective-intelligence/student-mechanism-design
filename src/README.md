# Source Code Directory

This directory contains the main implementation of the multi-agent reinforcement learning system for the Scotland Yard game.

## Files

### `main.py`
**Entry point for the entire application.** This file:
- Parses command-line arguments to load experiment configurations
- Merges configuration files (experiment config, agent config, logger config, visualization config)
- Routes execution to either training or evaluation mode based on the `evaluate` flag
- Handles WandB (Weights & Biases) integration for experiment tracking
- Loads and validates all configurations before starting training/evaluation

**Usage:**
```bash
python src/main.py --config src/configs/experiments/smoke_train/config.yml
```

### `logger.py`
**Unified logging and model management system.** This file:
- Provides a `Logger` class that handles:
  - Console and file logging with configurable verbosity levels
  - Model saving and loading (checkpoint management)
  - Metrics tracking and reporting
- Creates timestamped log directories for each experiment run
- Supports both INFO and DEBUG level logging
- Integrates with WandB for remote experiment tracking

**Key Methods:**
- `log(message, level)`: Log messages to console and file
- `save_model(model, name)`: Save PyTorch model checkpoints
- `load_model(name)`: Load previously saved models
- `log_metrics(metrics_dict)`: Record metrics for analysis

### `reward_net.py`
**Reward weight network for adaptive reward shaping.** This file:
- Implements `RewardWeightNet`: a neural network that learns optimal reward weights
- Used to balance different reward components dynamically during training
- Takes agent state information as input and outputs reward scaling weights
- Helps adapt the reward function based on the current learning stage
- Critical for curriculum learning and achieving balanced win rates between MrX and Police

**Architecture:**
- Input: State features from the environment
- Hidden layers: Fully connected layers with ReLU activation
- Output: Weights for different reward components (e.g., distance penalties, capture rewards)

### `wandb_data.json`
**Configuration file for Weights & Biases integration.** Contains:
- API key for WandB authentication
- Project name for organizing experiments
- Entity (username/organization) for WandB tracking
- Default run name template

**Note:** This file should not be committed to version control if it contains sensitive API keys.

## Subdirectories

### `agent/`
Contains all agent implementations (GNN-based, MAPPO, Random, Base classes).
See [agent/README.md](agent/README.md) for details.

### `configs/`
Contains all configuration files for experiments, agents, logging, and visualization.
See [configs/README.md](configs/README.md) for details.

### `environment/`
Contains the game environment implementation including graph generation, reward calculation, and visualization.
See [environment/README.md](environment/README.md) for details.

### `eval/`
Contains evaluation scripts for measuring agent performance and analyzing training results.
See [eval/README.md](eval/README.md) for details.

### `training/`
Contains training loops for different agent types (GNN, MAPPO).
See [training/README.md](training/README.md) for details.

## Quick Start

1. **Training a new model:**
   ```bash
   python src/main.py --config src/configs/experiments/smoke_train/config.yml
   ```

2. **Evaluating a trained model:**
   ```bash
   python src/main.py --config src/configs/experiments/smoke_train_eval/config.yml
   ```

3. **Running with visualization:**
   ```bash
   python src/main.py --config src/configs/experiments/smoke_eval_vis/config.yml
   ```

## Architecture Overview

The codebase follows a modular design:
- **Environment**: Defines the Scotland Yard game mechanics
- **Agents**: Implement different strategies (GNN, MAPPO, Random)
- **Training**: Orchestrates the learning process with curriculum learning
- **Evaluation**: Measures and analyzes agent performance
- **Configuration**: Separates hyperparameters from code for easy experimentation
