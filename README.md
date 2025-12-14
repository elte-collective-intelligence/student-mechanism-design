# Mechanism Design (Scotland Yard): Multi-Agent Reinforcement Learning (TorchRL)

[![CI](https://github.com/elte-collective-intelligence/student-mechanism-design/actions/workflows/ci.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-mechanism-design/actions/workflows/ci.yml)
[![Docker](https://github.com/elte-collective-intelligence/student-mechanism-design/actions/workflows/docker.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-mechanism-design/actions/workflows/docker.yml)
[![codecov](https://codecov.io/gh/elte-collective-intelligence/student-mechanism-design/branch/main/graph/badge.svg)](https://codecov.io/gh/elte-collective-intelligence/student-mechanism-design)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-blue.svg)](LICENSE)

## Quick Start

### Prerequisites

- Docker (recommended) or Python 3.10+
- CUDA-capable GPU (optional, for faster training)

### Installation & Running

```bash
# Clone the repository
git clone https://github.com/elte-collective-intelligence/student-mechanism-design.git
cd student-mechanism-design

# Build Docker images
docker build --progress plain -f ./docker/BaseDockerfile -t student_mechanism_design_base .
docker build --progress plain -f ./docker/Dockerfile -t student_mechanism_design .

# Run training experiment
docker run --rm --gpus=all --mount type=bind,src=$PWD,dst=/app student_mechanism_design all

# Run unit tests
docker run --rm --mount type=bind,src=$PWD,dst=/app student_mechanism_design --unit_test

# Run ablation studies
docker run --rm --mount type=bind,src=$PWD,dst=/app student_mechanism_design python src/eval/run_ablations.py --ablation all
```

### Local Development (without Docker)

```bash
pip install -r requirements.txt
cd src
python main.py all --agent_configs=mappo --log_configs=verbose
```

---

## Project Overview

This project implements a **mechanism design** approach for the Scotland Yard pursuit-evasion game using multi-agent reinforcement learning. Key features include:

- **Partial Observability**: MrX is hidden from police with configurable reveal schedules
- **Belief Tracking**: Particle filter and learned belief encoders for police
- **Mechanism Design**: Configurable tolls, budgets, and reveal policies
- **Meta-Learning**: Automatic tuning of mechanism parameters toward 50% win rate
- **Population-Based Self-Play**: Policy pools with ELO-style scoring
- **MAPPO & GNN Agents**: State-of-the-art multi-agent RL algorithms

---

## Experiment Matrix

| Experiment    | Agents | Graph Size  | Budget | Reveal | Description            |
| ------------- | ------ | ----------- | ------ | ------ | ---------------------- |
| `smoke_train` | 2      | 15 nodes    | 10     | R=5    | Quick sanity check     |
| `singular`    | 2-3    | 15 nodes    | 8-12   | R=5    | Single config training |
| `all`         | 2-6    | 15-20 nodes | 4-18   | R=5    | Full sweep             |
| `big_graph`   | 3-4    | 25+ nodes   | 10-15  | R=5    | Large graph evaluation |
| `test`        | 2      | 12 nodes    | 10     | R=5    | Development testing    |

### Running Experiments

```bash
# Run specific experiment
docker run --rm --gpus=all --mount type=bind,src=$PWD,dst=/app student_mechanism_design <experiment_name>

# Examples:
docker run ... student_mechanism_design smoke_train
docker run ... student_mechanism_design all
docker run ... student_mechanism_design big_graph
```

---

## Environment Specification

### Observation Space

Each agent receives:

| Field              | Type      | Description                              |
| ------------------ | --------- | ---------------------------------------- |
| `adjacency_matrix` | NxN float | Binary graph connectivity                |
| `node_features`    | NxK float | Agent positions encoded as one-hot       |
| `edge_index`       | 2xE int   | Edge list for GNN                        |
| `edge_features`    | E float   | Edge weights/costs                       |
| `action_mask`      | N bool    | Valid actions (fixed index→node mapping) |
| `valid_actions`    | list[int] | Affordable neighbor nodes                |
| `belief_map`       | N float   | MrX location distribution (Police only)  |
| `agent_position`   | int       | Current node                             |
| `agent_budget`     | float     | Remaining money                          |

### Action Space

- **Type**: `Discrete(N)` where N = number of nodes
- **Masking**: Actions masked by budget and topology
- **Mapping**: Fixed identity mapping (action i → node i)

### Action Mask Implementation

```python
# Fixed index→node mapping ensures consistency
mask[node] = True if (adjacent[current, node] and cost <= budget)
index_to_node = {i: i for i in range(num_nodes)}  # Identity mapping
```

### Mechanism Parameters

| Parameter          | Config Key           | Default | Description               |
| ------------------ | -------------------- | ------- | ------------------------- |
| Police Budget      | `police_budget`      | 10      | Initial money for police  |
| Reveal Interval    | `reveal_interval`    | 5       | Steps between MrX reveals |
| Reveal Probability | `reveal_probability` | 0.0     | Stochastic reveal chance  |
| Toll               | `tolls`              | 0.0     | Per-edge movement cost    |
| Ticket Price       | `ticket_price`       | 1.0     | Base movement cost        |
| Target Win Rate    | `target_win_rate`    | 0.5     | Meta-learning objective   |

---

## Metrics (3 Required)

We report the following three metrics as required by the assignment:

### 📊 Metric 1: Balance (Win Rate)

**Definition**: Fraction of episodes won by MrX

```
Win Rate = MrX Wins / Total Episodes
Target: 0.50 ± 0.05
```

**Implementation**: `src/eval/metrics.py::compute_win_rate()`

**Why this metric**: Measures game balance—the primary goal of mechanism design. A win rate of 50% indicates fair gameplay where neither side has a systematic advantage.

### 📊 Metric 2: Belief Quality (Cross-Entropy)

**Definition**: Cross-entropy between police belief distribution and true MrX position at reveal times.

```
CE = -log(belief[true_mrx_position])
Lower is better (more accurate belief)
```

**Implementation**: `src/eval/metrics.py::belief_cross_entropy()`

**Why this metric**: Measures how well police can track MrX under partial observability. Lower cross-entropy means the belief distribution assigns higher probability to MrX's true location.

### 📊 Metric 3: Time-to-Catch / Survival Time

**Definition**: Average episode length, split by winner.

- **Time-to-Catch**: Mean steps when Police wins
- **Survival Time**: Mean steps when MrX wins

**Implementation**: `src/eval/metrics.py::compute_time_metrics()`

**Why this metric**: Captures game dynamics—shorter catch times indicate effective police coordination, while longer survival times indicate successful evasion strategies.

---

## Ablation Studies

### Ablation 1: Belief Tracking

**Config**: `src/configs/ablation/belief.yaml`

Compares belief tracking methods under partial observability:

| Variant           | Reveal | Belief Method   | Expected Effect                   |
| ----------------- | ------ | --------------- | --------------------------------- |
| `no_belief`       | R=0    | None            | Police severely disadvantaged     |
| `particle_filter` | R=5    | Particle Filter | Baseline tracking                 |
| `learned_encoder` | R=5    | Neural Encoder  | Potentially better generalization |

**Run**:

```bash
python src/eval/run_ablations.py --ablation belief --num_episodes 100 --seeds 42 123 456
```

**Expected Results**:

- `no_belief`: MrX win rate ~70-80% (Police cannot track)
- `particle_filter`: MrX win rate ~50-55% (Baseline)
- `learned_encoder`: MrX win rate ~45-55% (Comparable or better)

### Ablation 2: Mechanism Design

**Config**: `src/configs/ablation/mechanism.yaml`

Compares mechanism configurations:

| Variant           | Tolls   | Budget  | Reveal  | Expected Win Rate     |
| ----------------- | ------- | ------- | ------- | --------------------- |
| `no_mechanism`    | 0       | ∞       | R=0     | ~70% MrX (unbalanced) |
| `fixed_mechanism` | 1.0     | 15      | R=5     | ~45% MrX (hand-tuned) |
| `meta_learned`    | learned | learned | learned | ~50% MrX (target)     |

**Run**:

```bash
python src/eval/run_ablations.py --ablation mechanism --num_episodes 100 --seeds 42 123 456
```

**Expected Results**:

- `no_mechanism`: Demonstrates need for mechanism design
- `fixed_mechanism`: Shows improvement over baseline
- `meta_learned`: Achieves target balance through optimization

### Running All Ablations

```bash
python src/eval/run_ablations.py --ablation all --num_episodes 100 --output_dir logs/ablations
```

### Ablation Results Location

Results are saved to `logs/ablations/`:

- `belief_results.json`: Raw metrics data
- `belief_report.txt`: Formatted comparison report
- `mechanism_results.json`: Raw metrics data
- `mechanism_report.txt`: Formatted comparison report

---

## Failure Analysis

### Known Limitations

1. **Belief Collapse**: Particle filter can collapse to incorrect modes when reveals are sparse (R > 10)

   - _Mitigation_: Noise injection, increased particle count, or use learned encoder

2. **Budget Exhaustion**: Police may run out of budget before catching MrX on large graphs

   - _Mitigation_: Meta-learning adjusts budget based on observed win rate

3. **Graph Topology Sensitivity**: Performance varies significantly with graph structure (degree distribution, diameter)

   - _Mitigation_: Curriculum learning over diverse graph distributions

4. **Action Mask Edge Cases**: When no moves are affordable, agent stays in place

   - _Handled_: Environment returns current position as default action

5. **Reward Hacking**: Agents may exploit reward shaping rather than achieving true objectives
   - _Mitigation_: Use terminal rewards primarily, validate with win rate metric

### Debugging Tips

```bash
# Enable verbose logging
docker run ... student_mechanism_design all --log_configs=verbose

# Visualize episodes (generates GIFs)
docker run ... student_mechanism_design smoke_train --vis_configs=full

# Run unit tests to verify components
docker run ... student_mechanism_design --unit_test

# Check specific test
pytest test/test_action_mask.py -v
```

---

## Code Structure

```
src/
├── main.py                     # Training entry point
├── logger.py                   # Logging utilities (WandB, TensorBoard)
├── reward_net.py               # RewardWeightNet for meta-learning
├── configs/
│   ├── ablation/
│   │   ├── belief.yaml         # Belief ablation variants
│   │   └── mechanism.yaml      # Mechanism ablation variants
│   ├── agent/                  # Agent configurations
│   ├── mechanism/default.yaml  # Mechanism parameters
│   └── ...
├── Enviroment/
│   ├── yard.py                 # Main environment (CustomEnvironment)
│   ├── action_mask.py          # Action masking with fixed index→node mapping
│   ├── belief_module.py        # ParticleBeliefTracker, LearnedBeliefEncoder
│   ├── partial_obs.py          # PartialObservationWrapper
│   ├── graph_generator.py      # GraphGenerator with seed saving
│   └── graph_layout.py         # ConnectedGraph sampling
├── RLAgent/
│   ├── mappo_agent.py          # MAPPO implementation
│   ├── gnn_agent.py            # GNN-based DQN agent
│   ├── random_agent.py         # Random baseline
│   └── base_agent.py           # Abstract base class
├── selfplay/
│   ├── population_manager.py   # Population-based training with ELO
│   ├── opponent_modeling.py    # Opponent behavior modeling
│   └── best_response.py        # Best response utilities
├── mechanism/
│   ├── mechanism_config.py     # MechanismConfig dataclass
│   ├── meta_learning_loop.py   # MetaLearner for mechanism optimization
│   └── reward_weight_integration.py
├── eval/
│   ├── metrics.py              # Core metrics (win rate, belief CE, time)
│   ├── run_ablations.py        # Ablation study runner
│   ├── ood_eval.py             # OOD & robustness evaluation
│   ├── belief_quality.py       # Belief cross-entropy
│   └── exploitability.py       # Exploitability proxy
├── experiments/
│   ├── all/config.yml
│   ├── smoke_train/config.yml
│   ├── singular/config.yml
│   └── ...
└── artifacts/                  # Saved model checkpoints
test/
├── test_action_mask.py         # Action mask unit tests
├── test_belief_update.py       # Belief tracking tests
├── env_test.py                 # Environment smoke tests
└── smoke_test.py               # Basic sanity check
```

---

## Configuration

### Hydra-Style Configs

All parameters are configurable via YAML:

```yaml
# src/configs/mechanism/default.yaml
police_budget: 10
reveal_interval: 5
reveal_probability: 0.0
ticket_price: 1.0
target_win_rate: 0.5
secondary_weight: 0.1
```

```yaml
# src/experiments/all/config.yml
agent_configurations:
  - num_police_agents: 2
    agent_money: 10
  - num_police_agents: 3
    agent_money: 8
  # ...
num_episodes: 70
epochs: 200
random_seed: 42
```

### WandB Integration

Set credentials in `src/wandb_data.json`:

```json
{
  "wandb_api_key": "<your-api-key>",
  "wandb_project": "scotland-yard",
  "wandb_entity": "<your-entity>"
}
```

Leave as `"null"` to disable WandB logging.

---

## Tests

### Unit Tests

```bash
# Run all tests
pytest test/

# Run specific tests
pytest test/test_action_mask.py -v
pytest test/test_belief_update.py -v
pytest test/env_test.py -v
```

### Test Coverage

| Test File               | Description             | Key Assertions                               |
| ----------------------- | ----------------------- | -------------------------------------------- |
| `test_action_mask.py`   | Action mask correctness | Fixed index→node mapping, budget constraints |
| `test_belief_update.py` | Belief tracking         | Distribution normalization, reveal collapse  |
| `env_test.py`           | Environment smoke test  | Reset/step don't throw exceptions            |

### Required Tests (Assignment)

1. ✅ **Action mask correctness**: `test_action_mask.py::test_action_mask_fixed_index_node_mapping`
2. ✅ **Belief update step**: `test_belief_update.py::test_belief_updates_and_reveals`

---

## Task Division

_Document team task division here (if applicable)_

| Task                               | Assignee | Status      |
| ---------------------------------- | -------- | ----------- |
| Task 1: Core Functionality         | -        | ✅ Complete |
| Task 2: Mechanism Design           | -        | ✅ Complete |
| Task 3: Scenarios (Self-play, OOD) | -        | ✅ Complete |
| Task 4: Metrics & Evaluation       | -        | ✅ Complete |
| Task 5: Ablation Studies           | -        | ✅ Complete |
| Task 6: Reproducibility Pack       | -        | ✅ Complete |
| Task 7: Documentation              | -        | ✅ Complete |

---

## References

- [MAPPO: The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955)
- [Scotland Yard Board Game](<https://en.wikipedia.org/wiki/Scotland_Yard_(board_game)>)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [TorchRL Documentation](https://pytorch.org/rl/)
- [Mechanism Design Theory](https://en.wikipedia.org/wiki/Mechanism_design)

---

## License

This project is licensed under CC BY-NC-ND 4.0. See the `LICENSE` file for details.
