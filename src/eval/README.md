# Evaluation and Analysis Tools

This directory contains scripts for evaluating trained agents, analyzing performance, and running ablation studies.

## Files

### `metrics.py`
**Core metrics computation for agent evaluation.** This file:
- Defines standard evaluation metrics for multi-agent RL
- Computes aggregate statistics across multiple episodes
- Provides visualization-ready metric formatting
- Used by other evaluation scripts

**Key Metrics:**

**Performance Metrics:**
- **Win Rate**: Percentage of episodes won by each side
  - Formula: `wins / total_episodes`
  - Range: [0, 1], higher is better
  - Balanced game should have ≈0.5 win rate
  
- **Average Episode Length**: Mean number of timesteps
  - Formula: `sum(episode_lengths) / num_episodes`
  - Longer episodes → harder for police to catch MrX
  - Shorter episodes → police catching quickly or poor MrX strategy
  
- **Reward Statistics**: Distributional analysis
  - Mean, standard deviation, min, max
  - Per agent type (MrX vs Police)
  - Helps identify reward function effectiveness

**Efficiency Metrics:**
- **Time to Capture**: Steps until MrX caught (for Police wins)
- **Escape Duration**: Steps survived (for MrX wins)
- **Budget Utilization**: Average money spent per episode

**Strategic Metrics:**
- **Coverage**: How well police spread across graph
- **Pursuit Efficiency**: Distance reduction per timestep
- **Belief Accuracy**: How close beliefs are to true MrX position

**Functions:**
- `compute_win_rate(results)`: Calculate win percentage
- `compute_episode_stats(results)`: Aggregate episode statistics
- `compute_reward_distribution(results)`: Analyze reward patterns
- `format_metrics(metrics_dict)`: Pretty-print for reports

### `exploitability.py`
**Measures strategy robustness and exploitability.** This file:
- Implements game-theoretic analysis of learned policies
- Tests agent performance against different opponents
- Identifies weaknesses in learned strategies
- Useful for understanding policy quality

**Key Concepts:**

**Exploitability**: How much an adversary can exploit a policy
- Low exploitability → robust strategy
- High exploitability → predictable, exploitable behavior
- Computed by training best-response opponent

**Evaluation Methods:**
- **Cross-play**: Test agent A vs agent B
  - Different training runs
  - Different algorithms (GNN vs MAPPO)
  - Identifies dataset overfitting
  
- **Self-play Convergence**: 
  - Checkpoint at different epochs
  - Test early vs late training
  - Measures improvement over time
  
- **Adversarial Testing**:
  - Train opponent specifically to exploit agent
  - Reveals blind spots in strategy
  - Tests worst-case performance

**Functions:**
- `compute_exploitability(agent, test_opponents)`: Measure exploit potential
- `cross_play_evaluation(agent1, agent2, num_games)`: Test different agents
- `best_response_training(target_agent)`: Train exploiter

**Use Cases:**
- Verify policy robustness
- Compare different training methods
- Identify failure modes
- Guide further training

### `belief_quality.py`
**Evaluates police belief system accuracy.** This file:
- Measures how accurate police beliefs are about MrX location
- Analyzes belief update effectiveness
- Tests information gathering strategies
- Critical for understanding partial observability handling

**Belief Quality Metrics:**

**Accuracy Metrics:**
- **KL Divergence**: Distance between belief and true distribution
  - Formula: `D_KL(P||Q) = Σ P(i) log(P(i)/Q(i))`
  - Lower is better (closer beliefs to truth)
  - 0 = perfect belief accuracy
  
- **True Location Probability**: Belief mass on correct node
  - Formula: `belief[true_mrx_position]`
  - Higher is better
  - 1.0 = certainty, 1/n = uniform (no info)
  
- **Entropy**: Uncertainty in belief distribution
  - Formula: `H = -Σ P(i) log P(i)`
  - Lower = more certain
  - Higher = more uncertain
  - Max entropy = uniform distribution

**Belief Evolution:**
- Track how beliefs change over time
- Measure convergence to true position
- Identify effective search patterns
- Analyze belief propagation on graph

**Functions:**
- `compute_belief_accuracy(beliefs, true_positions)`: Overall accuracy
- `belief_kl_divergence(belief, true_pos)`: KL divergence metric
- `belief_entropy(belief_dist)`: Compute uncertainty
- `track_belief_evolution(episode_beliefs)`: Temporal analysis

**Use Cases:**
- Debug belief update logic
- Optimize belief propagation parameters
- Understand information gathering
- Compare different belief models

### `ood_eval.py`
**Out-of-Distribution (OOD) evaluation.** This file:
- Tests agent generalization to unseen scenarios
- Evaluates robustness to distribution shift
- Measures adaptation capability
- Critical for real-world deployment

**OOD Scenarios:**

**Graph Structure Changes:**
- Different graph sizes (larger/smaller than training)
- Different connectivity patterns
- Different edge weight distributions
- Tests spatial reasoning generalization

**Agent Configuration Changes:**
- More/fewer police agents
- Different budget allocations
- Asymmetric starting positions
- Tests strategic adaptation

**Dynamics Changes:**
- Different capture distances
- Modified reward structures
- Altered observation noise
- Tests robustness to rule variations

**Evaluation Methods:**
- **Zero-shot Transfer**: No adaptation, direct testing
  - Measures inherent generalization
  - Quick evaluation
  - Lower bound on performance
  
- **Few-shot Adaptation**: Brief fine-tuning on new domain
  - Measures adaptation speed
  - More realistic scenario
  - Tests learning flexibility
  
- **Gradual Shift**: Progressive difficulty increase
  - Simulates curriculum in reverse
  - Identifies breaking points
  - Tests robustness limits

**Functions:**
- `generate_ood_scenarios(base_config)`: Create test cases
- `evaluate_zero_shot(agent, ood_envs)`: Direct testing
- `evaluate_adaptation(agent, ood_envs, adapt_episodes)`: Fine-tuning test
- `analyze_failure_modes(results)`: Identify weaknesses

**Use Cases:**
- Validate generalization claims
- Identify training biases
- Test deployment readiness
- Guide data collection

### `run_ablations.py`
**Automated ablation study runner.** This file:
- Systematically tests importance of different components
- Removes/modifies features one at a time
- Compares performance to full system
- Generates comparison reports

**Ablation Studies:**

**Architecture Ablations:**
- Remove GNN layers (test depth importance)
- Change hidden dimensions (capacity)
- Disable attention mechanisms
- Modify activation functions

**Reward Component Ablations:**
- Remove pursuit reward
- Disable grouping bonus
- Remove budget penalties
- Test each component's contribution

**Training Feature Ablations:**
- Disable curriculum learning
- Remove adaptive reward shaping
- Skip belief updates
- Test training methodology

**Ablation Process:**
1. **Baseline**: Train full system
2. **Ablate**: Remove one component
3. **Train**: Run training with ablation
4. **Compare**: Measure performance difference
5. **Repeat**: Test each component

**Functions:**
- `define_ablation_configs(base_config)`: Create variant configs
- `run_ablation_suite(configs)`: Execute all ablations
- `compare_results(baseline, ablations)`: Statistical comparison
- `generate_ablation_report(results)`: Format findings

**Output:** Identifies critical vs non-critical components

### `plot_ablations.py`
**Visualization of ablation study results.** This file:
- Creates comparison plots for ablation studies
- Generates statistical significance tests
- Produces publication-ready figures
- Helps interpret ablation results

**Visualization Types:**

**Performance Comparisons:**
- Bar charts: Win rates across ablations
- Line plots: Learning curves over training
- Box plots: Performance distributions
- Heatmaps: Configuration × metric matrices

**Statistical Analysis:**
- Confidence intervals (95%)
- Significance stars (*, **, ***)
- Effect size measurements
- Multiple comparison corrections

**Plot Examples:**
```python
# Win rate comparison
plot_win_rates(baseline, ablations)
# Shows which components matter most

# Learning curves
plot_training_curves(experiments)
# Compares convergence speed

# Component importance
plot_feature_importance(ablation_results)
# Ranks components by impact
```

**Functions:**
- `plot_win_rate_comparison(results)`: Bar chart with error bars
- `plot_learning_curves(training_logs)`: Time series comparison
- `plot_feature_importance(ablations)`: Ranked importance chart
- `generate_latex_table(results)`: Publication table

**Use Cases:**
- Understand what makes system work
- Justify design choices
- Identify opportunities for simplification
- Prepare research presentations

## Evaluation Workflow

**Standard Evaluation Process:**

1. **Train Agents**: Complete training runs
   ```bash
   python src/main.py --config configs/experiments/full_train/config.yml
   ```

2. **Basic Evaluation**: Measure performance
   ```bash
   python src/main.py --config configs/experiments/full_eval/config.yml
   ```

3. **Compute Metrics**: Analyze results (uses `metrics.py`)
   - Win rates, episode lengths, rewards
   
4. **Test Robustness**: Run OOD evaluation (`ood_eval.py`)
   - Different graphs, configurations
   
5. **Measure Exploitability**: Test against opponents (`exploitability.py`)
   - Cross-play, adversarial testing
   
6. **Ablation Studies**: Understand contributions (`run_ablations.py`)
   - Remove components, measure impact
   
7. **Visualization**: Create plots (`plot_ablations.py`)
   - Publication figures, reports

## Running Evaluations

### Basic Performance Evaluation
```bash
# Evaluate trained models
python src/main.py --config src/configs/experiments/smoke_train_eval/config.yml
```

### Out-of-Distribution Testing
```python
from src.eval.ood_eval import evaluate_zero_shot, generate_ood_scenarios

# Generate OOD test cases
ood_scenarios = generate_ood_scenarios(base_config)

# Test without adaptation
results = evaluate_zero_shot(trained_agent, ood_scenarios)
print(f"OOD Performance: {results['win_rate']:.2%}")
```

### Exploitability Analysis
```python
from src.eval.exploitability import compute_exploitability

# Test against different opponents
exploit_score = compute_exploitability(agent, test_opponents)
print(f"Exploitability: {exploit_score:.3f} (lower is better)")
```

### Belief Quality Assessment
```python
from src.eval.belief_quality import compute_belief_accuracy

# Analyze belief accuracy
accuracy = compute_belief_accuracy(recorded_beliefs, true_positions)
print(f"Mean belief KL divergence: {accuracy['kl_div']:.3f}")
```

### Complete Ablation Study
```bash
# Run full ablation suite
python src/eval/run_ablations.py --base_config configs/experiments/full_train/config.yml

# Generate comparison plots
python src/eval/plot_ablations.py --results ablation_results.json
```

## Tips for Students

1. **Start with metrics.py**: Understand standard evaluation metrics
2. **Use evaluation configs**: Don't modify training configs for eval
3. **Run multiple seeds**: Single runs can be misleading (use 5+ seeds)
4. **Test generalization**: OOD evaluation reveals overfitting
5. **Visualize results**: Plots are easier to interpret than tables
6. **Statistical testing**: Use significance tests, not just means
7. **Document findings**: Keep notes on what you discover
8. **Compare to baselines**: Random agent, heuristic strategies

## Common Evaluation Tasks

**Task 1: Compare GNN vs MAPPO**
1. Train both agent types with same configurations
2. Evaluate on same test episodes
3. Compute metrics for both
4. Plot learning curves
5. Test cross-play (GNN vs MAPPO)

**Task 2: Measure Generalization**
1. Train on small graphs (20 nodes)
2. Evaluate on large graphs (100 nodes)
3. Compare performance drop
4. Identify failure modes
5. Retrain with diverse data if needed

**Task 3: Ablate Reward Components**
1. Define ablations (remove each reward type)
2. Train with each ablation
3. Evaluate all variants
4. Plot component importance
5. Simplify reward function if possible

## Interpreting Results

**Good Performance Indicators:**
- Win rate near 50% (balanced game)
- Stable learning curves (no collapse)
- Low exploitability scores
- Good OOD generalization
- Interpretable strategies (from visualization)

**Warning Signs:**
- 100% win rate for one side (unbalanced)
- High variance in performance
- Poor OOD performance (overfitting)
- High exploitability (weak strategy)
- Random-looking behavior

**Next Steps Based on Results:**
- Poor performance → Adjust rewards, hyperparameters
- Overfitting → Add regularization, diverse training
- Unbalanced → Adaptive reward shaping
- Exploitable → More diverse opponents, adversarial training
