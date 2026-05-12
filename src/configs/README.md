# Configuration Files

This directory contains all configuration files for the project, organized by type. Configurations use YAML format for easy editing and version control.

## Directory Structure

```
configs/
├── agent/           # Agent-specific configurations
├── experiments/     # Complete experiment configurations
├── logger/          # Logging configurations
└── visualization/   # Visualization settings
```

## Configuration Hierarchy

Experiments load multiple config files:
1. **Experiment Config** (defines the experiment)
2. **Agent Config** (agent architecture and hyperparameters)
3. **Logger Config** (logging verbosity and options)
4. **Visualization Config** (rendering and GIF settings)

**Priority:** Experiment config overrides default values from other configs.

## Agent Configurations (`agent/`)

### `default.yaml`
**Default agent configuration template.** Contains:
- Basic agent parameters
- Common hyperparameters
- Fallback values if specific config missing

### `gnn.yaml`, `gat.yaml`, `transformer.yaml`
**Graph DQN agent configurations.** These files specify the architecture and learning parameters for the Graph DQN agent, which supports multiple underlying models.

#### Common Parameters (All Graph DQN Agents)
- `agent_type`: The architecture to use (`gnn`, `gat`, or `transformer`).
- `learning_rate`: Step size for optimization (typically 0.001).
- `gamma`: Discount factor for future rewards (typically 0.99).
- `batch_size`: Number of transitions sampled from replay buffer (e.g., 64).
- `buffer_size`: Capacity of the experience replay memory (e.g., 10000).
- `epsilon`, `epsilon_decay`, `epsilon_min`: Epsilon-greedy exploration parameters.

#### GAT and Transformer Specific
These parameters are passed as `model_kwargs` to the respective models:
- `hidden_dim`: Capacity of hidden layers (Standard: 64-256).
- `num_layers`: Depth of the network (Standard: 2-5).
- `dropout`: Dropout probability for regularization.

#### GAT Specific (`gat.yaml`)
- `heads`: Number of multi-head attention mechanisms (Standard: 1-8).
- `edge_dim`: Dimension of edge features.

#### Transformer Specific (`transformer.yaml`)
- `num_heads`: Number of attention heads (Standard: 1-8).
- `use_positional_encoding`: Boolean to enable Laplacian Positional Encoding (required for capturing graph topology).
- `pe_dim`: Dimension of the positional encoding (e.g., 8).

**Note on GNN**: The baseline `gnn` model use a fixed architecture of 2 AntiSymmetricConv layers and does not currently use the `hidden_dim` or `num_layers` parameters.

**Key Parameters for Ablation Studies:**
As per Assignment 2 requirements, students should vary `num_layers` (2-5), `hidden_dim` (64-256), and attention heads (1-8) in the `gat` and `transformer` configs to study their impact on coordination and sample efficiency.

### `gat.yaml`
**Graph Attention Network agent configuration.** Specifies:
```yaml
# Architecture
node_feature_size: 10        # Features per node
hidden_dim: 64              # Hidden layer size
heads: 4                    # Attention heads
num_layers: 3               # Number of GAT layers

# Learning
learning_rate: 0.001         # Adam optimizer learning rate
epsilon: 0.1                 # Exploration rate (epsilon-greedy)
gamma: 0.99                  # Discount factor for rewards
```

**Key Parameters:**
- `heads`: Controls how many attention patterns the model can learn
- `hidden_dim`: Width of the intermediate graph representations
- `num_layers`: Depth of attention-based message passing

### `transformer.yaml`
**Graph Transformer agent configuration.** Specifies:
```yaml
# Architecture
node_feature_size: 10        # Features per node
pos_dim: 8                   # Laplacian positional encoding dimension
hidden_dim: 128             # Hidden layer size
heads: 4                    # Attention heads
num_layers: 3               # Number of Transformer layers

# Learning
learning_rate: 0.001         # Adam optimizer learning rate
epsilon: 0.1                 # Exploration rate (epsilon-greedy)
gamma: 0.99                  # Discount factor for rewards
```

**Key Parameters:**
- `pos_dim`: Size of the positional encoding added to the graph
- `heads`: Controls multi-head attention capacity
- `num_layers`: Depth of transformer-based graph reasoning

### `mappo.yaml`
**Multi-Agent PPO configuration.** Specifies:
```yaml
# Architecture
actor_hidden_dim: 128        # Actor network hidden size
critic_hidden_dim: 256       # Critic network hidden size (typically larger)

# PPO Parameters
ppo_clip: 0.2                # Clipping epsilon for PPO objective
value_loss_coef: 0.5         # Weight for value function loss
entropy_coef: 0.01           # Weight for entropy regularization

# Training
batch_size: 128              # Timesteps per update
num_ppo_epochs: 4            # Update iterations per batch
learning_rate: 0.0003        # Lower than GNN (PPO more stable)
gamma: 0.99                  # Discount factor
gae_lambda: 0.95             # GAE parameter for advantage estimation
```

**Key Parameters:**
- `ppo_clip`: Controls update magnitude
  - Standard: 0.1-0.3
  - Lower → more conservative updates
- `num_ppo_epochs`: Data reuse
  - More epochs → better sample efficiency
  - Too many → overfitting to batch
- `gae_lambda`: Bias-variance tradeoff
  - 1.0 → high variance, no bias
  - 0.0 → low variance, high bias
  - 0.95 typical sweet spot

### `random.yaml`
**Random baseline agent.** Minimal configuration:
```yaml
# No learning parameters needed
# Just needs to know action space
action_size: 50
```

## Experiment Configurations (`experiments/`)

Each experiment has its own subdirectory containing:
- `config.yml`: Main experiment configuration
- `logs/`: Training logs and saved models
- `vis/`: Visualization outputs (if enabled)

### `smoke_train/config.yml`
**Quick training test for development.** Settings:
```yaml
agent_configs: 'gnn'         # Use GNN agent
log_configs: 'default'       # Standard logging
vis_configs: 'none'          # No visualization (faster)
evaluate: False              # Training mode

# Training
epochs: 1                    # Single epoch for testing
episodes_per_epoch: 2        # Just 2 episodes
max_steps: 30               # Short episodes

# Environment
graph_nodes: 10             # Small graph
graph_edges: 10             # Minimal complexity

# Agent Configurations (train on diverse scenarios)
agent_configurations:
  - num_police_agents: 2
    agent_money: 16
  - num_police_agents: 3
    agent_money: 14
  # ... more configurations
```

**Purpose:** Fast iteration during development
- Run in seconds
- Verify code changes work
- Test new features quickly

### `smoke_train_mappo/config.yml`
**MAPPO training test.** Similar to `smoke_train` but:
```yaml
agent_configs: 'mappo'       # Use MAPPO instead of GNN
# ... rest similar to smoke_train
```

### `smoke_train_eval/config.yml`
**Evaluation of trained models.** Settings:
```yaml
evaluate: True               # Evaluation mode (NOT training)
num_eval_episodes: 2         # Episodes per configuration
agent_configs: 'gnn'         # Match training agent type

# Must match training configurations
agent_configurations:
  - num_police_agents: 3
    agent_money: 16
  # ... only configs with saved models
```

**Important:** Only include configurations that have trained models!
### `smoke_eval_vis/config.yml`
**Evaluation with visualization.** Settings:
```yaml
evaluate: True
vis_configs: 'full'          # Enable all visualization
num_eval_episodes: 3         # Fewer episodes (GIFs are large)

# Single configuration for focused visualization
agent_configurations:
  - num_police_agents: 4
    agent_money: 16
```

### `gat_experiment/` and `transformer_experiment/`
**Architecture-specific training runs.**
- These experiments use the `gat` and `transformer` agent configurations respectively.
- Used to validate that attention-based agents can successfully learn coordination strategies on the Scotland Yard graph.

### Other Experiments
**Output:** Generates GIFs in `logs/vis/`:
- `run_epoch_0-episode_1.gif`: Game visualization
- `heatmap_epoch_0-episode_1.gif`: Belief heatmap

### `all_{agent_type}/config.yml`
**Full training experiments for a specific agent type.** These experiments follow the `all_{agent_type}` naming convention and train the specified model.

**Current variants:**
- `all_gnn/config.yml`: Full training of the GNN agent
- `all_gat/config.yml`: Full training of the GAT agent
- `all_transformer/config.yml`: Full training of the Transformer agent

**Purpose:**
- Compare agent families under the same environmentsettings
- Keep training experiments organized by agent type
- Make it easy to add new agent-specific training experiments later

### Other Experiments

You can create custom experiments:
```
experiments/my_experiment/
├── config.yml
└── logs/
    ├── *.pt (model checkpoints)
    └── events.out.tfevents.* (TensorBoard logs)
```

## Ablation Configurations (`ablation/`)

### `architecture.yaml`
**Architecture ablation sweep.** Defines agent variants for comparing graph model families and depth/width settings:
- Baseline GNN reference
- Several GAT variants with different layer counts, widths, and heads
- Several Transformer variants with different layer counts, widths, and heads

**Purpose:**
- Compare model families under a common evaluation harness
- Measure the effect of depth and attention heads
- Provide a shared source of architecture variants for ablation runs

### `belief.yaml`
**Belief-model ablation sweep.** Defines variants for comparing belief updates and learned belief encoders.

### `mechanism.yaml`
**Mechanism-design ablation sweep.** Defines variants for comparing tolls, budgets, reveal schedules, and meta-learned mechanism parameters.

## Logger Configurations (`logger/`)

### `default.yaml`
**Standard logging configuration.** Settings:
```yaml
log_level: 'INFO'            # Log level (DEBUG, INFO, WARNING, ERROR)
log_to_file: True            # Save logs to file
log_to_console: True         # Print to console
log_metrics: True            # Log numerical metrics
log_dir: 'logs'              # Directory for log files
```

### `verbose.yaml`
**Detailed logging for debugging.** Settings:
```yaml
log_level: 'DEBUG'           # Show all debug messages
log_to_file: True
log_to_console: True
log_metrics: True
log_rewards: True            # Log reward components
log_beliefs: True            # Log belief updates
log_actions: True            # Log action selection details
```

**Use when:**
- Debugging agent behavior
- Understanding reward function
- Tracking belief updates
- Analyzing action patterns

**Warning:** Creates large log files!

## Visualization Configurations (`visualization/`)

### `attention.yaml`
**Attention visualization settings.** Settings:
```yaml
visualize_game: False
visualize_heatmap: False
visualize_attention: True
save_visualization: True
save_dir: 'logs/vis'
attention_layer: -1
attention_head_reduction: 'mean'
attention_cmap: 'viridis'
```

**Use for:**
- Rendering attention overlays or summaries during evaluation
- Inspecting the final attention layer by default
- Saving attention-focused visualization outputs to disk

**Notes:**
- `attention_layer: -1` selects the last layer
- `attention_head_reduction: 'mean'` reduces across attention heads using simple averaging

### `none.yaml`
**No visualization (fastest).** Settings:
```yaml
visualize_game: False        # No game rendering
visualize_heatmap: False     # No heatmap
save_visualization: False    # No GIF generation
```

**Use for:**
- Training (don't need visuals)
- Fast experimentation
- Headless servers

### `default.yaml`
**Basic visualization.** Settings:
```yaml
visualize_game: True         # Show game state
visualize_heatmap: False     # Skip heatmap (faster)
save_visualization: False    # Don't save GIFs
fps: 2                       # Frames per second (if live viewing)
```

### `full.yaml`
**Complete visualization with GIF generation.** Settings:
```yaml
visualize_game: True         # Render game
visualize_heatmap: True      # Show belief heatmap
save_visualization: True     # Generate GIF files
save_dir: 'logs/vis/'        # Output directory
fps: 2                       # Animation speed
dpi: 100                     # Image quality
```

**Output:** Creates two GIF types:
1. **Game GIF**: Agent positions and movements
2. **Heatmap GIF**: Police belief evolution

**Use for:**
- Evaluation visualization
- Presentation/demonstration
- Understanding learned strategies
- Debugging spatial reasoning

## Creating Custom Configurations

### New Experiment
1. Create directory: `experiments/my_experiment/`
2. Create `config.yml`:
```yaml
agent_configs: 'gnn'
log_configs: 'default'
vis_configs: 'none'
evaluate: False

epochs: 50
episodes_per_epoch: 100
graph_nodes: 50
graph_edges: 110

agent_configurations:
  - num_police_agents: 4
    agent_money: 20
```
3. Run: `python src/main.py --config src/configs/experiments/my_experiment/config.yml`

### New Agent Config
1. Create `agent/my_agent.yaml`
2. Define hyperparameters
3. Reference in experiment: `agent_configs: 'my_agent'`

## Configuration Best Practices

**For Students:**
1. **Start with smoke tests**: Verify code works before long training
2. **Copy existing configs**: Don't create from scratch
3. **Document changes**: Add comments explaining non-standard values
4. **Version control**: Commit configs with code changes
5. **Match eval to training**: Use same agent_configs in evaluation

**Naming Conventions:**
- `smoke_*`: Quick tests (< 1 minute)
- `small_*`: Small-scale experiments (< 10 minutes)
- `full_*`: Complete training runs (hours)
- `*_eval`: Evaluation configurations
- `*_vis`: Visualization-enabled

**Parameter Tuning:**
- **Learning Rate**: Start with 0.001 (GNN) or 0.0003 (MAPPO)
- **Batch Size**: 32-128 for MAPPO, N/A for GNN
- **Hidden Dim**: 64-256 depending on problem complexity
- **Epsilon**: 0.1 for 10% exploration
- **Epochs**: 50-100 for meaningful training

## Common Configuration Patterns

**Quick Development Test:**
```yaml
epochs: 1
episodes_per_epoch: 2
graph_nodes: 10
vis_configs: 'none'
```

**Full Training Run:**
```yaml
epochs: 50
episodes_per_epoch: 100
graph_nodes: 50
log_configs: 'default'
vis_configs: 'none'
```

**Evaluation with Visualization:**
```yaml
evaluate: True
num_eval_episodes: 10
vis_configs: 'full'
agent_configurations: [...]  # Only configs with models
```

## Tips for Students

1. **Read before running**: Understand what each config does
2. **Start small**: Use smoke tests before full training
3. **Match configurations**: Eval configs must match training
4. **Check model files**: Ensure models exist before evaluation
5. **Monitor logs**: Watch win rates and losses during training
6. **Experiment systematically**: Change one parameter at a time
7. **Save good configs**: Keep configs that produce good results
8. **Use visualization sparingly**: GIFs are large and slow to generate
