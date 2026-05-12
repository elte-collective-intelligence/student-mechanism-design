# Agent Implementations

This directory contains all agent implementations for the Scotland Yard game, including neural network agents (GNN, GAT, and Transformer), MAPPO, and baseline agents.

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

### `graph_dqn_agent.py`
**Deep Q-Network (DQN) agent for graph environments.** This file:
- Implements a flexible agent that uses DQN for learning optimal strategies on graphs
- Acts as an orchestrator for multiple underlying model architectures (GNN, GAT, Transformer)
- Handles experience replay and epsilon-greedy exploration
- Manages device placement (CPU/GPU) for graph data

**Key Features:**
- **Architecture Agnostic**: Can swap between GNN, GAT, and Transformer models via configuration
- **Experience Replay**: Stores and samples past experiences for stable learning
- **Batch Processing**: Uses PyTorch Geometric's `Batch` for efficient parallel graph processing
- **Device Management**: Automatically moves graphs and models to the configured device (e.g., CUDA)

**Parameters:**
- `node_feature_size`: Number of features per node (e.g., position, money, belief)
- `gamma`: Discount factor for rewards (default: 0.99)
- `lr`: Learning rate for the Adam optimizer (default: 0.001)
- `batch_size`: Number of samples per training batch (default: 64)
- `buffer_size`: Maximum size of the experience replay buffer (default: 10000)
- `epsilon`: Initial exploration rate for epsilon-greedy policy
- `epsilon_decay`: Rate at which epsilon decreases after each update
- `epsilon_min`: Minimum exploration rate
- `agent_type`: Underlying model architecture (`gnn`, `gat`, or `transformer`)

### `gnn_model.py`
**Baseline Graph Neural Network model.**
- Uses standard message passing layers to aggregate local node information
- Provides a simple yet effective baseline for spatial reasoning

### `gat_model.py`
**Graph Attention Network (GAT) model.**
- Implements multi-head attention over neighboring nodes
- Allows the agent to focus on specific, more important neighbors during decision making
- Supports attention weight extraction for visualization and analysis

### `transformer_model.py`
**Graph Transformer model.**
- Uses global attention mechanisms to capture long-range dependencies across the graph
- Well-suited for complex coordination tasks where distant nodes may be relevant
- Includes positional encodings or graph-based structural features

### `gat_agent.py`
**Graph Attention Network (GAT) based agent.** This file:
- Implements a Q-learning style agent with PyTorch Geometric GAT layers
- Uses multi-head attention to weigh neighboring nodes differently
- Supports attention inspection for debugging and visualization
- Respects action masks and epsilon-greedy exploration during training

**Architecture:**
- **Input**: Graph observation with node features
- **GAT Layers**: Stacked attention convolution layers with multi-head aggregation
- **Output Layer**: Node-level Q-values used for action selection

**Key Features:**
- Returns per-layer attention weights for analysis if requested
- Handles graph-structured inputs without fixed node counts
- Uses experience replay for training updates
- Works well when edge importance matters more than uniform message passing

**Parameters:**
- `node_feature_size`: Number of features per node
- `hidden_dim`: Hidden width for the attention layers
- `heads`: Number of attention heads in the intermediate layers
- `num_layers`: Number of GAT layers

### `transformer_agent.py`
**Graph Transformer based agent.** This file:
- Implements an attention-based agent using Transformer convolutions
- Computes Laplacian positional encodings before inference and training
- Combines structural position information with node features
- Supports attention extraction for inspection and debugging

**Architecture:**
- **Input**: Graph observation plus positional encodings
- **Transformer Layers**: Stacked TransformerConv layers for graph reasoning
- **Output Layer**: Node-level Q-values for masked action selection

**Key Features:**
- Adds Laplacian positional encodings to improve structural awareness
- Supports multi-head attention over graph neighborhoods
- Returns attention tensors for downstream analysis if requested
- Respects action masks during action selection

**Parameters:**
- `node_feature_size`: Number of features per node
- `pos_dim`: Dimensionality of the positional encoding
- `hidden_dim`: Hidden width of the transformer layers
- `heads`: Number of attention heads
- `num_layers`: Number of transformer layers

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

### When to use Graph DQN Agent?
- **Pros:** 
  - Flexible: choose between **GNN**, **GAT**, or **Transformer** models
  - Naturally handles graph structure and spatial reasoning
  - Sample efficient through Experience Replay
  - Generalizes well to different graph sizes
- **Cons:**
  - Requires tuning of buffer size and exploration parameters
  - Can be computationally intensive with large graphs

### When to use GAT Agent?
- **Pros:**
  - Learns which neighboring nodes matter most
  - Exposes attention weights for interpretability
  - Handles graph structure directly
  - Useful when local neighborhood importance is uneven
- **Cons:**
  - More expensive than simple graph convolutions
  - Attention heads add tuning complexity

### When to use Transformer Agent?
- **Pros:**
  - Adds positional encodings for structural context
  - Uses attention to model long-range interactions
  - Suitable for richer graph reasoning tasks
  - Provides attention outputs for analysis
- **Cons:**
  - More complex than the GNN or GAT variants
  - Attention heads add tuning complexity

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
- `gat.yaml`: GAT agent hyperparameters
- `gnn.yaml`: GNN agent hyperparameters
- `transformer.yaml`: Transformer agent hyperparameters
- `mappo.yaml`: MAPPO agent hyperparameters
- `random.yaml`: Random agent settings

**Example Graph DQN (GAT) config:**
```yaml
agent_type: 'gat'
node_feature_size: 10
hidden_dim: 128
num_layers: 3
num_heads: 4             # GAT-specific: attention heads
learning_rate: 0.001
epsilon: 0.1  # Exploration rate
```

## Training Flow

1. **Initialization**: Agent and model are created based on the YAML config
2. **Action Selection**: Agent observes state and selects action
   - During training: Epsilon-greedy (explores random moves with probability ε)
   - During evaluation: Greedy (always chooses the move with highest Q-value)
3. **Learning**: Agent updates its parameters based on rewards
  - GNN: Q-learning style updates
  - GAT: Q-learning style updates with attention-based message passing
  - Transformer: Q-learning style updates with positional encodings
  - MAPPO: PPO policy gradient updates
4. **Model Saving**: Trained agent parameters saved to disk

## Model Naming Convention

Models are saved with names indicating the agent type and configuration:
- **GNN agents**: `MrX_{node_feature_size}_agents.pt`, `Police_{node_feature_size}_agents.pt`
- **GAT agents**: model-specific checkpoints saved by the training pipeline
- **Transformer agents**: model-specific checkpoints saved by the training pipeline
- **MAPPO agents**: `MAPPO_MrX_0.pt`, `MAPPO_Police_0.pt`
- **Reward network**: `RewardWeightNet.pt` (shared across agents)

## Tips for Students

1. **Start with Random Agent**: Understand the interface before implementing learning
2. **Study GNN Agent**: Good example of graph-based learning
3. **Experiment with MAPPO**: Learn multi-agent coordination techniques
4. **Modify Architectures**: Try different network designs in `gnn_model.py`, `gat_model.py`, or `transformer_model.py`
5. **Compare Performance**: Run experiments with different agent types
