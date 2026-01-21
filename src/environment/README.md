# Environment Implementation

This directory contains the Scotland Yard game environment implementation, including graph generation, game mechanics, reward calculation, pathfinding, and visualization.

## Files

### `yard.py` (605 lines)
**Main environment class implementing the Scotland Yard game.** This is the core environment file that:
- Implements the PettingZoo multi-agent environment interface
- Manages game state (agent positions, budgets, beliefs, timesteps)
- Coordinates between different subsystems (rewards, pathfinding, visualization)
- Handles agent observations and action processing
- Enforces game rules and termination conditions

**Key Components:**
- **State Management**: Tracks all agent positions, money, beliefs about MrX location
- **Action Processing**: Converts agent actions to graph movements with toll payments
- **Observation Generation**: Creates graph-based observations for each agent
- **Episode Management**: Handles reset, step, and termination logic
- **Delegation**: Uses helper modules for specialized tasks:
  - `RewardCalculator` for reward computation
  - `Pathfinder` for shortest path calculations
  - `GameVisualizer` for rendering and GIF generation

**Main Methods:**
- `reset()`: Initialize new episode with random starting positions
- `step(actions)`: Process actions and advance game state by one timestep
- `observe(agent)`: Generate observation for specific agent
- `_is_mrx_caught()`: Check if police caught MrX (within capture distance)

**Game Rules Enforced:**
- Police agents must stay within budget when moving
- MrX has unlimited budget
- Police have imperfect information (belief states)
- Episode ends when MrX is caught or max timesteps reached

### `reward_calculator.py` (270 lines)
**Dedicated reward calculation system.** This file:
- Implements `RewardCalculator` class with comprehensive reward shaping
- Provides different reward components for MrX and Police agents
- Uses the reward weight network to adapt reward scaling during training
- Logs detailed reward breakdowns for analysis

**Reward Components:**

**For MrX (Fugitive):**
- **Distance Reward**: Positive reward for staying far from police
- **Capture Penalty**: Large negative reward when caught (-10.0)
- **Survival Bonus**: Small positive reward each timestep (encourages evasion)
- **Budget Penalty**: Negative reward if police run low on money (helps MrX)

**For Police (Pursuers):**
- **Pursuit Reward**: Positive reward for getting closer to MrX
- **Grouping Reward**: Bonus for multiple police near MrX (encourages coordination)
- **Coverage Reward**: Reward for spreading out to cover more area
- **Capture Reward**: Large positive reward when catching MrX (+10.0)
- **Overlap Penalty**: Negative reward if police are too close to each other
- **Budget Penalty**: Penalty for running out of money

**Key Methods:**
- `calculate_rewards(state, actions, next_state)`: Compute rewards for all agents
- `calculate_rewards_and_terminations(...)`: Combined reward + termination logic
- `_calculate_mrx_reward(...)`: MrX-specific reward calculation
- `_calculate_police_rewards(...)`: Police-specific reward calculation

**Adaptive Rewards:**
The system uses `RewardWeightNet` to dynamically adjust reward weights based on:
- Current win rate imbalance
- Training progress
- Agent performance trends

### `pathfinding.py` (138 lines)
**Efficient shortest path computation using Dijkstra's algorithm.** This file:
- Implements `Pathfinder` class with graph-based pathfinding
- Uses priority queue for efficient shortest path computation
- Handles weighted graphs (edges have toll costs)
- Caches graph structure for performance

**Key Methods:**
- `get_distance(start, end)`: Compute shortest weighted path distance
  - Accounts for edge weights (tolls)
  - Returns infinity if no path exists
  - Used by reward calculator for distance-based rewards

**Algorithm:**
- Classic Dijkstra's algorithm with binary heap priority queue
- Time complexity: O((V + E) log V) where V=nodes, E=edges
- Optimized for repeated queries on the same graph

**Usage in Rewards:**
- Calculate distance from each police officer to MrX
- Determine if police are getting closer or farther
- Compute coverage metrics (how spread out police are)

### `visualization.py` (316 lines)
**Game rendering and GIF generation system.** This file:
- Implements `GameVisualizer` class for creating visual representations
- Generates animated GIFs showing game progression
- Creates heatmaps of police belief distributions
- Uses matplotlib with non-interactive Agg backend

**Visualization Modes:**
1. **Game View**: Shows agent positions on graph
   - MrX in red, Police in blue
   - Edge weights (tolls) displayed
   - Agent budgets shown
2. **Heatmap View**: Shows police belief about MrX location
   - Color intensity indicates belief probability
   - Helps visualize uncertainty and information gathering

**Key Methods:**
- `render(state, timestep, agent_infos)`: Create single frame
  - Draws graph with NetworkX spring layout
  - Positions agents on nodes
  - Shows relevant information (budgets, beliefs)
- `save_visualizations(epoch, episode)`: Generate GIF files
  - Compiles all frames into animated GIF
  - Creates separate GIFs for game view and heatmap
  - Saves to configured directory (e.g., `logs/vis/`)

**Configuration Options:**
- `visualize_game`: Enable/disable real-time game rendering
- `visualize_heatmap`: Enable/disable belief heatmap
- `save_visualization`: Save frames to GIF files
- `save_dir`: Directory for saving GIF files

### `graph_generator.py`
**Procedural graph generation for the Scotland Yard map.** This file:
- Creates random connected graphs for game scenarios
- Assigns random edge weights (tolls) within specified range
- Ensures graph connectivity (all nodes reachable)
- Supports curriculum learning (variable graph complexity)

**Key Functions:**
- `generate_graph(num_nodes, num_edges)`: Create random graph
  - Uses NetworkX graph generation algorithms
  - Ensures strong connectivity
  - Returns adjacency matrix and edge weights

**Curriculum Learning Support:**
- Start training on small graphs (easier)
- Gradually increase graph size (harder)
- Helps agents learn basic strategies before facing complexity

### `graph_layout.py`
**Layout computation for consistent graph visualization.** This file:
- Computes node positions for visualization
- Ensures consistent layout across frames (no jumping nodes)
- Uses spring layout or other NetworkX algorithms
- Caches layouts for performance

**Purpose:**
- Consistent visualization in GIF animations
- Aesthetic graph rendering
- Spatial clarity for human interpretation

### `belief_module.py`
**Belief state tracking for imperfect information.** This file:
- Implements belief update logic for police agents
- Models uncertainty about MrX's location
- Updates beliefs based on observations and time
- Implements belief propagation on graph structure

**Belief Update Rules:**
- **Diffusion**: Beliefs spread to neighboring nodes over time
- **Observation**: Direct observations override beliefs
- **Decay**: Beliefs gradually become more uncertain
- **Normalization**: Belief probabilities sum to 1.0

**Key Methods:**
- `update_beliefs(current_beliefs, observation, graph)`: Update belief state
- `reveal_mrx_location(beliefs, mrx_position)`: Set belief to certainty when MrX seen

**Why Beliefs Matter:**
Police agents have imperfect information, making the game more realistic and challenging. They must:
- Infer MrX's likely location from indirect evidence
- Coordinate search patterns
- Balance exploration vs. exploitation

### `action_mask.py`
**Action masking for valid move generation.** This file:
- Computes which actions are valid given current state
- Enforces budget constraints (police can't afford expensive moves)
- Respects graph structure (can't move to non-adjacent nodes)
- Used during action selection to prevent invalid actions

**Key Functions:**
- `get_action_mask(agent_position, budget, graph, edge_weights)`: Generate binary mask
  - Returns array: 1 for valid actions, 0 for invalid
  - Considers money constraints
  - Accounts for staying in place (action 0)

**Why Action Masks?**
- Prevent agents from learning invalid actions
- Speed up learning (don't waste time on impossible moves)
- Ensure game rules are always followed
- Improve sample efficiency

### `base_env.py`
**Base environment class with common functionality.** This file:
- Provides shared utilities for environment implementations
- Defines common interfaces for PettingZoo compatibility
- May contain helper functions used by `yard.py`
- Abstract class that `Yard` inherits from

## Architecture Overview

The environment is designed with clear separation of concerns:

```
yard.py (Main Environment)
├── reward_calculator.py (Rewards)
│   └── Uses pathfinding.py for distances
├── pathfinding.py (Shortest Paths)
├── visualization.py (Rendering)
├── belief_module.py (Uncertainty)
├── action_mask.py (Valid Moves)
└── graph_generator.py (Map Creation)
```

## Refactoring History

**Original Design**: All functionality in `yard.py` (1018 lines) - difficult to maintain

**Refactored Design**: Split into specialized modules (41% reduction in main file)
- Better organization and readability
- Easier testing of individual components
- Clearer separation of concerns
- Maintained full functionality while improving code quality

## Tips for Students

1. **Start with `yard.py`**: Understand the main game loop and state management
2. **Study `reward_calculator.py`**: Learn reward shaping techniques for multi-agent RL
3. **Explore `pathfinding.py`**: See efficient graph algorithms in action
4. **Experiment with `visualization.py`**: Create custom visualizations
5. **Modify `belief_module.py`**: Try different belief update strategies
6. **Test with `action_mask.py`**: Understand constraint handling in RL

## Common Tasks

**Adding a new reward component:**
1. Open `reward_calculator.py`
2. Add reward calculation in `_calculate_mrx_reward` or `_calculate_police_rewards`
3. Update reward weights if needed
4. Test with a training run

**Changing graph complexity:**
1. Modify `graph_generator.py` parameters
2. Update curriculum schedules in training config
3. Adjust agent architectures if needed (larger graphs → larger networks)

**Creating custom visualizations:**
1. Extend `GameVisualizer` class in `visualization.py`
2. Override `render()` method with custom drawing logic
3. Configure in `vis_configs` YAML file
