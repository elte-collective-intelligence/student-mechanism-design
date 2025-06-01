# RL Meta-Learning with GNN

## Introduction

This repository presents a meta-learning approach for reinforcement learning (RL) environments, leveraging Multi Agent Proximal Policy Optimization (MAPPO) (and Graph Neural Networks (GNNs)) to enable dynamic adaptability. The project emphasizes multi-agent setups where agents collaboratively learn optimal policies, focusing on flexibility, shared information, and environment-aware strategies.

## Overview

The project aims to equip RL agents with the ability to adapt to varying task difficulties and dynamic interactions using meta-learning techniques and MAPPO (with backward compatibility for Graph Neural Networks (GNNs)). Agents interact in a simulated city-like grid, taking on distinct objectives and utilizing shared information for optimized decision-making.

### Key Features

- **Meta-Learning**: Dynamically balances success and failure rates to achieve a 50/50 outcome.
- **Graph Neural Networks**: Models agent relationships, enabling enhanced real-time adaptability.
- **MAPPO**:
- **Multi-Agent Policies**: Develops specialized strategies for distinct roles.
- **Dynamic Environment**: Adjusts parameters like agent count and resources to ensure evolving difficulty.
- **Shared Policemen Policy**: Unifies strategies across agents for improved coordination.

## Architecture

### Environment

The simulation involves a grid-based city environment where:

- **MrX**: Operates as the target agent, focusing on evasion.
- **Policemen**: Cooperatively work to track and capture MrX.
- **Difficulty Parameter**: Modifies agent capabilities and resources to fine-tune task complexity.

### Meta-Learning Framework

The outer loop adjusts task difficulty through:

1. Collecting and analyzing performance data from multiple episodes.
2. Balancing success and failure rates to maintain a stable learning environment.
3. Embedding difficulty adjustments as a learnable parameter directly into the environment.

### GNN Integration

GNNs enhance the system by:

- **Spatial and Temporal Encoding**: Capturing dynamic relationships among agents.
- **State Sharing**: Facilitating coordinated strategies across multiple agents.
- **Policy Adaptability**: Supporting flexible decision-making through graph-based message passing.

### MAPPO Integration



### Policies

1. **MrX Policy**: Optimized to maximize evasion success.
2. **Policemen Policy**: Shared across agents to promote efficient collaboration and coordination.

## Code Structure Overview

- **main.py**  
  - Contains the main entry point (train and evaluate functions) and the central training loop.  
  - Sets up the command-line arguments, loads configurations, initializes the logger, environment, and agents.  
  - Implements the logic for either training or evaluating the RL agents based on arguments.  

- **logger.py**  
  - Defines the Logger class for handling logging to console, file, TensorBoard, and Weights & Biases.  
  - Manages logging metrics, weights, and model artifacts.  

- **Enviroment/base_env.py**  
  - Declares an abstract base class (BaseEnvironment) for custom environments using PettingZoo’s ParallelEnv.  

- **Enviroment/graph_layout.py**  
  - Contains a custom ConnectedGraph class for creating random connected graphs with optional extra edges and weights.  
  - Provides graph sampling logic (e.g., Prim’s algorithm to ensure connectivity).  

- **Enviroment/yard.py**  
  - Implements CustomEnvironment, which inherits from BaseEnvironment.  
  - Manages environment reset, step logic, agent positions, reward calculations, rendering, and graph observations.  

- **RLAgent/base_agent.py**  
  - Declares an abstract BaseAgent class defining the interface (select_action, update, etc.) for all RL agents.  

- **RLAgent/gnn_agent.py**  
  - Defines GNNAgent, a DQN-like agent using a GNN (GNNModel) to compute Q-values for graph nodes.  
  - Handles experience replay, epsilon-greedy action selection, and network updates.  
- **RLAgent/mappo_agent.py**
  
### Main Training Loop (in main.py, train function)

1. **Initialize** logger, network(s), optimizers, and hyperparameters.  
2. **For each epoch**:  
   - Randomly choose environment config (number of agents, money, etc.).  
   - Forward pass through the RewardWeightNet to compute reward weights for the environment.  
   - **Inside loop**: for each episode:
     - Reset environment, get initial state.  
     - While not done:  
       - Build GNN input (create_graph_data), pick actions (MrX and Police) using the GNN agents.
       - MAPPO train loop.
       - env.step(actions), compute rewards/terminations, update agents.  
   - Evaluate performance (num_eval_episodes), compute target difficulty, backpropagate loss in RewardWeightNet.  
   - Log metrics and proceed to the next epoch.  

## Installation

Clone the repository and install the required dependencies using [Docker](https://www.docker.com/):

```bash
git clone https://github.com/elte-collective-intelligence/student-mechanism-design.git
cd student-mechanism-design
```
Build base image:
```bash
docker build --progress plain -f ./docker/BaseDockerfile -t student_mechanism_design_base .
```
Build main image:
```bash
docker build --progress plain -f ./docker/Dockerfile -t student_mechanism_design .
```
This should build the Docker images. 

## Usage

### Docker support

Run experiment:
```bash
docker run --rm --gpus=all --mount type=bind,src=$PWD/src/,dst=/app/src student_mechanism_design <experiment> <flags>
```

Run unit tests:
```bash
docker run --rm --gpus=all --mount type=bind,src=$PWD,dst=/app student_mechanism_design --unit_test
```

### Wandb config

If you want to use wandb to log your experiments, dont forget to set the credentials in wandb_data.json, leave them as "null" to disable Wandb logging:
```json
{
    "wandb_api_key":"<api-key>",
    "wandb_project":"<project-name>",
    "wandb_entity":"<entity-name>"
}
```
### Experiment config

1. In the experiment folder, create a folder with the name of you experiment.
2. Add a config.yml file to it, with the required configurations (there are examples)

### Config flags

1. --agent_configs=mappo
2. --log_configs=verbose
3. --vis_configs=default

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with detailed descriptions of changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


