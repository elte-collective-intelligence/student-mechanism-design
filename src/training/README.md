# Training Modules

This directory contains the training loops and utilities for different agent types, implementing curriculum learning and comprehensive logging.

## Files

### `gnn_trainer.py` (318 lines)
**Training loop for Graph Neural Network (GNN) agents.** This file:
- Orchestrates the complete training process for GNN-based agents
- Implements curriculum learning with progressive difficulty increase
- Handles model checkpointing and performance tracking
- Supports multi-configuration training (different agent counts and budgets)

**Training Process:**
1. **Curriculum Initialization**: Create schedules for graph complexity
2. **Epoch Loop**: Iterate through training epochs
3. **Configuration Selection**: Choose random agent configuration
4. **Episode Generation**: Run episodes with current policy
5. **Model Update**: Update agent networks based on collected experience
6. **Evaluation**: Track win rates and adjust reward weights
7. **Model Saving**: Checkpoint models for each configuration

**Key Features:**
- **Curriculum Learning**: Gradually increase graph size and complexity
  - Start with small graphs (easier to learn)
  - Progressively add more nodes and edges
  - Helps agents learn basic strategies before facing full complexity
  
- **Adaptive Reward Shaping**: Use `RewardWeightNet` to balance game
  - Monitors MrX vs Police win rates
  - Adjusts rewards to maintain competitive balance
  - Prevents one side from dominating too much
  
- **Multi-Configuration Training**: Train on diverse scenarios
  - Different numbers of police agents
  - Variable budget allocations
  - Ensures generalization across game conditions
  
- **Model Management**: Intelligent checkpoint system
  - Saves models for each configuration separately
  - Maintains best models based on performance
  - Enables continuation of interrupted training

**Parameters:**
- `args.epochs`: Total number of training epochs
- `args.episodes_per_epoch`: Episodes per epoch
- `args.agent_configurations`: List of (num_police, budget) tuples
- `args.graph_nodes`: Number of nodes in graph
- `args.graph_edges`: Number of edges in graph

**Logging:**
- Episode-level: Winner, steps, rewards
- Epoch-level: Win rates, loss values, model performance
- Configuration-level: Per-config statistics

### `mappo_trainer.py` (319 lines)
**Training loop for Multi-Agent Proximal Policy Optimization (MAPPO).** This file:
- Implements PPO algorithm for multi-agent learning
- Uses centralized training with decentralized execution (CTDE)
- Collects trajectories and performs batch updates
- Supports parameter sharing among police agents

**Training Process:**
1. **Environment Initialization**: Create game environment
2. **Trajectory Collection**: Gather experiences using current policy
   - Run episodes with stochastic policy
   - Store states, actions, rewards, next states
   - Collect from all agents simultaneously
3. **Advantage Computation**: Calculate GAE (Generalized Advantage Estimation)
   - Use critic to estimate state values
   - Compute advantage = reward + γ*V(next) - V(current)
4. **Policy Update**: Optimize actor networks with PPO loss
   - Clip probability ratios for stability
   - Multiple epochs over collected batch
   - Separate updates for MrX and Police actors
5. **Value Update**: Train critic with MSE loss
   - Predict state values using global observation
   - Minimize squared error from true returns
6. **Entropy Regularization**: Encourage exploration
   - Add entropy bonus to loss
   - Prevents premature convergence to deterministic policy

**Key Features:**
- **PPO Clipping**: Prevents large policy updates
  - Clip ratio: `clip(ratio, 1-ε, 1+ε)` where ε=0.2
  - Maintains stable learning
  - Avoids catastrophic forgetting
  
- **Centralized Critic**: Global value function
  - Uses information from all agents
  - Reduces variance in advantage estimates
  - Improves credit assignment
  
- **Experience Replay**: Batch-based learning
  - Collects multiple episodes before updating
  - Sample efficient (multiple epochs per batch)
  - Stable gradients from large batches
  
- **Multi-Agent Coordination**: 
  - Shared critic for police agents
  - Individual actors maintain agent identity
  - Learns coordinated strategies

**MAPPO-Specific Parameters:**
- `batch_size`: Number of timesteps per update (default: 128)
- `num_ppo_epochs`: Update iterations per batch (default: 4)
- `ppo_clip`: Clipping parameter ε (default: 0.2)
- `value_loss_coef`: Weight for value loss (default: 0.5)
- `entropy_coef`: Weight for entropy bonus (default: 0.01)
- `gamma`: Discount factor (default: 0.99)
- `gae_lambda`: GAE lambda parameter (default: 0.95)

**Advantages of MAPPO:**
- Sample efficient (reuses data multiple times)
- Stable learning (clipping prevents large updates)
- Strong multi-agent coordination
- Well-suited for cooperative tasks

### `evaluator.py` (242 lines)
**Evaluation system for trained agents.** This file:
- Loads trained models and evaluates their performance
- Runs episodes without exploration (greedy policy)
- Computes aggregate statistics across multiple episodes
- Generates visualizations of evaluation runs

**Evaluation Process:**
1. **Model Loading**: Load trained agent models from disk
   - Automatically finds models based on configuration
   - Handles missing models gracefully (uses untrained agents)
2. **Episode Execution**: Run episodes with greedy policy
   - No exploration (epsilon = 0)
   - Deterministic action selection
   - Multiple episodes per configuration
3. **Statistics Collection**: Track performance metrics
   - Win rates for MrX and Police
   - Average episode length
   - Reward distributions
4. **Visualization**: Generate GIFs if configured
   - Shows agent strategies visually
   - Helps identify learned behaviors

**Key Methods:**
- `evaluate_agent(env, agent, num_episodes)`: Run evaluation episodes
  - Returns win rates and statistics
  - Logs detailed episode information
- `load_trained_models(config, agent_type)`: Load checkpoint files
  - Searches for models matching configuration
  - Falls back to untrained agents if models missing

**Evaluation Metrics:**
- **Win Rate**: Percentage of episodes won by each side
- **Average Steps**: Mean episode length across runs
- **Reward Statistics**: Mean, std, min, max rewards
- **Success Rate**: Episodes where MrX captured vs escaped

**Use Cases:**
- **Performance Assessment**: Measure agent skill level
- **Comparison**: Compare different training runs
- **Debugging**: Identify failure modes visually
- **Demonstration**: Show trained agent behavior

### `utils.py`
**Utility functions for training.** This file:
- Provides helper functions used across training modules
- Implements common operations to avoid code duplication
- May include functions for:
  - Curriculum generation
  - State preprocessing
  - Action sampling
  - Metric computation

**Common Utilities:**
- `create_curriculum(epochs, max_nodes, max_edges, max_money, increase_rate)`:
  - Generates progressive schedules for curriculum learning
  - Returns lists of increasing complexity over epochs
  - Supports multiple curriculum dimensions simultaneously
  
- `extract_step_info(state, agents)`:
  - Extracts rewards, terminations, truncations from TensorDict
  - Handles PettingZoo format conversions
  - Used in both training and evaluation
  
- `is_episode_done(terminations, truncations)`:
  - Checks if episode has ended
  - Combines termination and truncation signals
  - Returns True if any agent is done

**Why Utilities?**
- Reduce code duplication
- Centralize common logic
- Make training loops cleaner
- Easier to maintain and test

## Training Architecture

```
Training Flow:
┌─────────────────────┐
│   Configuration     │
│   (YAML files)      │
└──────┬──────────────┘
       │
       v
┌─────────────────────┐
│   main.py           │
│   (Entry Point)     │
└──────┬──────────────┘
       │
       ├──→ GNN Agent?  → gnn_trainer.py
       │                  ├── Curriculum Learning
       │                  ├── Experience Collection
       │                  ├── Model Updates
       │                  └── Model Saving
       │
       ├──→ MAPPO Agent? → mappo_trainer.py
       │                   ├── Trajectory Collection
       │                   ├── GAE Computation
       │                   ├── PPO Updates
       │                   └── Value Updates
       │
       └──→ Evaluate?    → evaluator.py
                           ├── Model Loading
                           ├── Greedy Evaluation
                           ├── Statistics
                           └── Visualization
```

## Curriculum Learning Strategy

Both trainers implement curriculum learning:

**Stage 1 (Early Training):**
- Small graphs (10-20 nodes)
- Few police agents (2-3)
- High initial budgets
- Learn basic movement and objectives

**Stage 2 (Mid Training):**
- Medium graphs (30-40 nodes)
- More police agents (4-5)
- Moderate budgets
- Learn coordination and strategy

**Stage 3 (Late Training):**
- Large graphs (50+ nodes)
- Many police agents (6-8)
- Realistic budgets
- Master complex scenarios

## Model Checkpointing

**Checkpoint Strategy:**
- Save models after each epoch
- Separate files per configuration
- Naming: `{AgentType}_{node_feature_size}_agents.pt`
- Best models retained based on win rate

**Loading Checkpoints:**
- Automatic discovery by configuration match
- Graceful fallback to untrained agents
- Supports continuing interrupted training

## Logging and Metrics

**Episode-Level Logging:**
```
Episode X finished. Winner: Police, Steps: 42, MrX Reward: 12.5, Police Reward: 85.3
```

**Epoch-Level Logging:**
```
============================================================
Epoch 5/50 Summary:
  MrX Wins: 12/20 (60.0%)
  Police Wins: 8/20 (40.0%)
  Target: 50% (balanced)
============================================================
Epoch 5: Loss: 0.123, Win Ratio: 0.60
```

**Model Saving:**
```
Models saved for epoch 5.
MrX_5_agents.pt: 4.8 KB
Police_5_agents.pt: 4.9 KB
```

## Tips for Students

1. **Start with GNN Trainer**: Simpler algorithm, easier to understand
2. **Study Curriculum Implementation**: See how difficulty progression helps learning
3. **Compare GNN vs MAPPO**: Different learning paradigms, different strengths
4. **Monitor Win Rates**: Balanced training is key to good agents
5. **Experiment with Hyperparameters**: Learning rate, batch size, PPO clip
6. **Visualize Evaluation**: Use evaluator.py to see what agents learned
7. **Read Training Logs**: Understand learning progress from metrics
8. **Checkpoint Management**: Learn when to save/load models

## Common Issues and Solutions

**Problem: One side dominates (100% win rate)**
- Solution: Adjust reward weights in `reward_calculator.py`
- Use adaptive reward shaping with `RewardWeightNet`
- May need to balance reward magnitudes

**Problem: Training unstable (loss diverges)**
- Solution: Reduce learning rate
- For MAPPO: Decrease PPO clip value
- Check for NaN values in gradients

**Problem: Slow convergence**
- Solution: Adjust curriculum (start easier)
- Increase batch size (MAPPO)
- Check reward signal strength

**Problem: Agent strategies look random**
- Solution: Need more training epochs
- Verify reward function aligns with objectives
- Check if action masks are correct
