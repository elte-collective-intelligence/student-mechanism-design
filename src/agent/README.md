# Agent Implementations

This directory contains all agent implementations for the Scotland Yard game, including neural network-based agents (GNN and MAPPO) and baseline agents.

## Files

### `base_agent.py`
**Abstract base class for all agents.** This file:
- Defines the common interface that all agents must implement
- Provides a template for action selection (`select_action`)
- Handles agent initialization and configuration
- Ensures consistency across different agent types

**Key Methods:**
- `select_action(observation, action_mask)`: Choose an action based on the current observation
  - `observation`: Current state of the game (graph-based representation)
  - `action_mask`: Binary mask indicating which actions are valid (respects budget constraints)
  - Returns: Selected action index

**Usage Pattern:**
```python
class MyAgent(BaseAgent):
    def select_action(self, observation, action_mask):
        # Implement custom action selection logic
        return action_index
```

### `gnn_agent.py`
**Graph Neural Network (GNN) based agent.** This file:
- Implements an agent using PyTorch Geometric for graph-based reasoning
- Uses message passing to aggregate information from neighboring nodes
- Processes the graph structure of the Scotland Yard map
- Learns spatial relationships and strategic movement patterns

**Architecture:**
- **Input**: Graph observation with node features (positions, budgets, beliefs)
- **GNN Layers**: Multiple graph convolution layers for information propagation
- **Output Layer**: Fully connected layer outputting action probabilities

**Key Features:**
- Handles variable graph sizes (different numbers of nodes/edges)
- Uses attention mechanisms to focus on important nodes
- Respects action masks for valid moves only
- Supports epsilon-greedy exploration during training

**Parameters:**
- `node_feature_size`: Number of features per node (e.g., position, money, belief)
- `hidden_dim`: Size of hidden layers in the GNN
- `num_layers`: Number of graph convolution layers
- `action_size`: Number of possible actions (depends on graph size)

### `mappo_agent.py`
**Multi-Agent Proximal Policy Optimization (MAPPO) agent.** This file:
- Implements an actor-critic architecture for multi-agent learning
- Uses centralized training with decentralized execution (CTDE)
- Shares a global critic among all police agents
- Individual actors for each agent (MrX and each police officer)

**Architecture:**
- **Actor Network**: Outputs action probabilities for each agent
  - Input: Local observations (agent's own state and nearby information)
  - Output: Probability distribution over actions
- **Critic Network**: Estimates value function using global state
  - Input: Concatenated observations from all agents
  - Output: State value estimate

**Key Features:**
- PPO clipping for stable policy updates
- Value function baseline to reduce variance
- Entropy regularization for exploration
- Supports parameter sharing among police agents

**Training Process:**
1. Collect trajectories using current policy
2. Compute advantages using GAE (Generalized Advantage Estimation)
3. Update policy using PPO objective
4. Update value function using MSE loss

### `random_agent.py`
**Random baseline agent.** This file:
- Selects actions uniformly at random from valid actions
- Useful as a baseline for comparing learned policies
- Fast and simple implementation
- Respects action masks (only chooses valid moves)

**Purpose:**
- Sanity check for environment implementation
- Baseline performance metric
- Debugging tool for testing environment mechanics

## Agent Selection Guide

### When to use GNN Agent?
- **Pros:** 
  - Naturally handles graph structure
  - Learns spatial reasoning
  - Generalizes to different graph sizes
  - Good for strategy games with explicit graph topology
- **Cons:**
  - Requires graph representation
  - More computationally intensive
  - May need more training data

### When to use MAPPO Agent?
- **Pros:**
  - Strong multi-agent coordination
  - Sample efficient (PPO updates)
  - Stable training with PPO clipping
  - Good for cooperative tasks
- **Cons:**
  - Requires centralized training setup
  - More complex implementation
  - Higher memory requirements (stores trajectories)

### When to use Random Agent?
- **Use for:**
  - Testing environment correctness
  - Baseline comparisons
  - Quick prototyping
  - Sanity checks

## Configuration

Each agent type has a corresponding configuration file in `src/configs/agent/`:
- `gnn.yaml`: GNN agent hyperparameters
- `mappo.yaml`: MAPPO agent hyperparameters
- `random.yaml`: Random agent settings

**Example GNN config:**
```yaml
node_feature_size: 10
hidden_dim: 128
num_layers: 3
learning_rate: 0.001
epsilon: 0.1  # Exploration rate
```

## Training Flow

1. **Initialization**: Agent is created based on config file
2. **Action Selection**: Agent observes state and selects action
   - During training: Epsilon-greedy or stochastic policy
   - During evaluation: Greedy (argmax) policy
3. **Learning**: Agent updates its parameters based on rewards
   - GNN: Q-learning style updates
   - MAPPO: PPO policy gradient updates
4. **Model Saving**: Trained agent parameters saved to disk

## Model Naming Convention

Models are saved with names indicating the agent type and configuration:
- **GNN agents**: `MrX_{node_feature_size}_agents.pt`, `Police_{node_feature_size}_agents.pt`
- **MAPPO agents**: `MAPPO_MrX_0.pt`, `MAPPO_Police_0.pt`
- **Reward network**: `RewardWeightNet.pt` (shared across agents)

## Tips for Students

1. **Start with Random Agent**: Understand the interface before implementing learning
2. **Study GNN Agent**: Good example of graph-based learning
3. **Experiment with MAPPO**: Learn multi-agent coordination techniques
4. **Modify Architectures**: Try different network designs in `gnn_agent.py`
5. **Compare Performance**: Run experiments with different agent types
