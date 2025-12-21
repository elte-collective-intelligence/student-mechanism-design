# Assignment 2

## Scotland Yard: Mechanism Design

## Overview

This assignment iterates a Scotland Yard-style pursuit-evasion project into a research-grade study in mechanism design, meta-learning, and generalization on graph environments. The current codebase already features PettingZoo + TorchRL wrappers, a connected weighted-graph world, MAPPO and GNN agents, and a RewardWeightNet for meta-learning toward a target balance (e.g., 50% MrX win rate). Your task is to turn this into a principled investigation: design mechanisms (budgets, pricing, reveal schedules), introduce partial observability and belief tracking, improve self-play and population training, and produce strong generalization and robustness results.

## What is already provided

- Graph world env (connected, undirected, weighted); PettingZoo-compatible; TorchRL wrapper via PettingZooWrapper.
- Agents: MrX (evader) vs K Police (cooperative team); synchronous moves.
- Learning: MAPPO baseline; GNN-based agents; RewardWeightNet (meta-learner) targeting balance via win ratio.
- Pipeline: Hydra-style configs, logging, basic curriculum hooks; visualization (matplotlib GIFs).
- Docker and CTDE setup.

## What you will build this semester

1. Introduce partial observability (hidden MrX with reveal schedule) and belief tracking (e.g., particle filter or learned belief encoder).
2. Design and study mechanisms: budget/ticket pricing, edge tolls, and reveal policies; meta-learn mechanism parameters to achieve target equilibria (e.g., 50/50 win rate) with secondary objectives (e.g., cost).
3. Upgrade self-play: opponent modeling, population-based training (PBT), and exploitability-style evaluation against diverse policies.
4. Strong generalization: train on a distribution of graphs and evaluate out-of-distribution (OOD) settings; add robustness tests (noise, missing edges).

## Task Description

Agents play on a procedurally generated connected weighted graph. Police pay edge costs; MrX aims to evade capture until timeout. Your contributions:
(i) mechanism design (tolls, budgets, reveal frequency) optimized by meta-learning to meet a target win-rate and secondary criteria;
(ii) partial observability and belief-based policies;
(iii) self-play with opponent modeling;
(iv) evaluation on unseen graphs with rigorous metrics.

## What to Keep

- MAPPO / GNN agents, CTDE; TorchRL + PettingZoo wrapper.
- Hydra configs, Docker, logging (W&B / TensorBoard).
- RewardWeightNet (or equivalent meta-learner), curriculum hooks.

## Environment & Scenarios

- Procedural graphs: variable nodes/edges/degree; weighted edges (costs).
- Partial observability: MrX hidden by default, revealed every R steps or probabilistically; Police receive noisy MrX location cues.
- Mechanisms:

  - pricing: per-edge tolls
  - budgets: initial Police money
  - reveal policy: fixed or learned schedule
  - tickets: transport-specific costs
  - fairness constraints: bound the fraction of dead-ends or traps

- Self-play populations: keep pools of MrX and Police policies with periodic best-response training.

## Observation and Action

- Graph observation: adjacency, edge weights, per-node features; hide MrX node except at reveal; include last-seen time or belief map as features.
- Belief module: particle filter or learned encoder producing a per-node belief over MrX; feed belief (or top-k nodes) to Police policy.
- Action space: masked Discrete over nodes (valid neighbors under budget). Fix index→node mapping and provide action masks in the wrapper.

## Mechanism Design and Meta-Learning

- Bilevel view:
  Upper-level chooses mechanism parameters θ (tolls, budgets, reveal R);
  Lower-level runs self-play to equilibrium policies π\*(θ).
- Objective: match target win rate (e.g., 50%) and minimize a secondary cost (e.g., sum of tolls, time-to-catch).
- Training: meta-gradients through reward weights / mechanism parameters; or black-box bandit/ES for θ; curriculum on graph difficulty.

## Metrics

- Balance: MrX win rate (target 50%), variance across seeds/graphs.
- Exploitability proxy: performance vs held-out opponents (scripted heuristics, older checkpoints).
- Belief quality: cross-entropy vs ground-truth MrX position at reveal times.
- Mechanism cost: mean tolls paid, mean steps-to-catch, mean budget spent.
- Generalization & robustness: in/out-of-distribution (OOD) graphs; edge/cost noise; missing reveals.

## Ablations

- No-belief vs PF-belief vs learned-belief.
- No-mechanism vs fixed-mechanism vs meta-learned mechanism.
- Self-play variants: independent SP, population SP, opponent modeling on/off.
- Observation variants: no edge weights vs with weights; reveal frequency R.

## Reproducibility Pack

- Configs for: partial observability; mechanism parameters; populations; evaluation suites.
- Dockerfile and make targets (train/eval/self-play/meta).
- README with quick start; experiment matrix; environment spec (spaces, masks).
- Tests: action mask correctness (index↔node), belief update step, env reset/step smoke test.

## Grading Rubric

**Total: 70 points + up to 10 bonus**
Keep everything configurable via Hydra. Report seeds/configs. Use Docker for the runs you submit.

### Task 1: Core Functionality — 20 pts

- Action masking & fixed index→node mapping; masks exported in the wrapper — 8 pts
- Configurable graph generator (nodes/edges/degree ranges) with saved seeds — 4 pts
- Partial observability basics: hidden MrX + reveal schedule R (or probabilistic) — 4 pts
- Renderer/visualization updated and success criterion defined — 4 pts

### Task 2: Mechanism Design — 10 pts

- Mechanism parameters in config: tolls, budgets, reveal policy (single source of truth) — 4 pts
- Meta-learning loop toward 50% win rate with a secondary cost objective — 6 pts

### Task 3: Scenarios — 10 pts

- Population-based self-play (pools; periodic best response) — 5 pts
- Generalization/robustness scenario (OOD graph distribution or edge/cost noise or missing reveals) — 5 pts

### Task 4: Metrics & Evaluation — 10 pts

Choose exactly three metrics from the list below and report them with plots/tables:
Balance (win rate), Exploitability proxy, Belief quality at reveal times, Mechanism cost/efficiency, Time-to-catch/survive
Scoring: correct implementation & clear reporting of any three — 10 pts

### Task 5: Ablations — 10 pts

Run exactly two ablations with fixed seeds and clear plots (pick from the Ablations list):

- Ablation 1 with short analysis — 5 pts
- Ablation 2 with short analysis — 5 pts

### Task 6: Reproducibility Pack — 6 pts

- Hydra configs for experiments/sweeps — 2 pts
- Dockerfile builds and runs training and eval — 2 pts
- Exactly two unit/smoke tests (mask correctness; belief update or env step) — 2 pts

### Task 7: Reporting Quality — 4 pts

- README with quick start, experiment matrix, the three chosen metrics, ablations, and a brief failure analysis — 4 pts

### Bonus up to +10 pts

- Information design: learn reveal schedules (policy over reveals) and compare to fixed R — +5 pts
- Strong baseline for exploitability: MCTS/heuristic MrX or coordinated Police — +5 pts

## Presentation

Optional; skipping caps the maximum at 42. Keep slides concise: questions, mechanisms, partial observability, self-play, metrics, key plots, and failure cases. Include a brief live/recorded GIF.

## Submission

- Upload a ZIP to Canvas with code, configs, plots, and slides (PDF if presenting). Exclude large checkpoints.
- Solo or teams; if a team, document task division in the README.
- Use branches/PRs as preferred. Include logs/ or W&B links.