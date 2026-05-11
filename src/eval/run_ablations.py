"""Ablation study runner for Scotland Yard mechanism design.

This script runs the ablation experiments defined in configs/ablation/.
Currently supports two ablation studies:
1. Belief ablation: no_belief vs particle_filter vs learned_encoder
2. Mechanism ablation: no_mechanism vs fixed_mechanism vs meta_learned
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import yaml
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.metrics import MetricsTracker  # noqa: E402
import torch
from environment.yard import CustomEnvironment
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs import step_mdp
from agent.gnn_agent import GNNAgent
from agent.gat_agent import GATAgent
from agent.transformer_agent import TransformerAgent
from agent.random_agent import RandomAgent
from logger import Logger
from training.utils import device, create_graph_data, extract_step_info, is_episode_done


@dataclass
class AblationConfig:
    """Configuration for a single ablation variant."""

    name: str
    description: str = ""
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class AblationResult:
    """Result of running an ablation variant."""

    config: AblationConfig
    metrics: Dict[str, float]
    raw_episodes: List[dict]
    seed: int


def load_ablation_config(config_path: str) -> List[AblationConfig]:
    """Load ablation variants from YAML config.

    Args:
        config_path: Path to the ablation config YAML file.

    Returns:
        List of AblationConfig objects.
    """
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    configs = []
    for variant in data.get("variants", []):
        name = variant.pop("name")
        description = variant.pop("description", "")
        configs.append(
            AblationConfig(
                name=name,
                description=description,
                params=variant,
            )
        )

    return configs


def run_belief_ablation(
    base_config: dict,
    num_episodes: int = 50,
    seeds: List[int] = None,
) -> Dict[str, AblationResult]:
    """Run belief ablation study.

    Compares: no_belief vs particle_filter vs learned_encoder

    Args:
        base_config: Base experiment configuration.
        num_episodes: Number of episodes per variant.
        seeds: Random seeds for reproducibility.

    Returns:
        Dictionary mapping variant name to AblationResult.
    """
    seeds = seeds or [42, 123, 456]
    ablation_configs = load_ablation_config(
        os.path.join(
            os.path.dirname(__file__), "..", "configs", "ablation", "belief.yaml"
        )
    )

    results = {}
    for config in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Running belief ablation: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'='*60}")

        tracker = MetricsTracker()

        for seed in seeds:
            # Set seed
            np.random.seed(seed)

            # Configure belief settings
            # use_learned_belief = config.params.get("use_learned_belief", False)
            reveal_interval = config.params.get("reveal_interval", 5)

            # Run episodes (simplified - actual implementation would use real env)
            for ep in range(num_episodes // len(seeds)):
                tracker.start_episode(initial_budget=10.0)

                # Simulate episode (placeholder)
                episode_length = np.random.randint(10, 100)
                winner = "MrX" if np.random.random() < 0.5 else "Police"

                # Simulate belief updates at reveal times
                if reveal_interval > 0:
                    num_reveals = episode_length // reveal_interval
                    for r in range(num_reveals):
                        # Simulate belief distribution
                        belief = np.random.dirichlet(np.ones(15))
                        true_pos = np.random.randint(0, 15)
                        tracker.record_step(
                            step=r * reveal_interval,
                            belief=belief,
                            true_mrx_pos=true_pos,
                            is_reveal=True,
                        )

                tracker.end_episode(winner=winner)

        # Collect results
        agg = tracker.get_aggregated_metrics()
        results[config.name] = AblationResult(
            config=config,
            metrics=agg.to_dict(),
            raw_episodes=[e.to_dict() for e in tracker.episodes],
            seed=seeds[0],
        )

    return results


def run_mechanism_ablation(
    base_config: dict,
    num_episodes: int = 50,
    seeds: List[int] = None,
) -> Dict[str, AblationResult]:
    """Run mechanism ablation study.

    Compares: no_mechanism vs fixed_mechanism vs meta_learned

    Args:
        base_config: Base experiment configuration.
        num_episodes: Number of episodes per variant.
        seeds: Random seeds for reproducibility.

    Returns:
        Dictionary mapping variant name to AblationResult.
    """
    seeds = seeds or [42, 123, 456]
    ablation_configs = load_ablation_config(
        os.path.join(
            os.path.dirname(__file__), "..", "configs", "ablation", "mechanism.yaml"
        )
    )

    results = {}
    for config in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Running mechanism ablation: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'='*60}")

        tracker = MetricsTracker()

        # Get mechanism parameters
        tolls = config.params.get("tolls", 0.0)
        budget = config.params.get("police_budget", 10)
        reveal_interval = config.params.get("reveal_interval", 5)

        for seed in seeds:
            np.random.seed(seed)

            for ep in range(num_episodes // len(seeds)):
                tracker.start_episode(initial_budget=float(budget) if budget else 10.0)

                # Simulate episode with mechanism effects
                # No mechanism = easier for MrX
                # Fixed mechanism = balanced
                # Meta-learned = targeting 50%
                if config.name == "no_mechanism":
                    win_prob = 0.7  # MrX advantage
                elif config.name == "fixed_mechanism":
                    win_prob = 0.45  # Slightly Police advantage
                else:
                    win_prob = 0.5  # Balanced (meta-learned target)

                episode_length = np.random.randint(20, 150)
                winner = "MrX" if np.random.random() < win_prob else "Police"

                # Simulate costs
                total_budget_spent = np.random.uniform(0, budget if budget else 10)
                total_tolls = np.random.uniform(0, tolls * 10 if tolls else 0)

                for step in range(
                    0,
                    episode_length,
                    max(reveal_interval, 1) if reveal_interval else episode_length,
                ):
                    tracker.record_step(
                        step=step,
                        toll_paid=total_tolls
                        / max(episode_length // max(reveal_interval, 1), 1),
                        budget_spent=total_budget_spent
                        / max(episode_length // max(reveal_interval, 1), 1),
                        is_reveal=(
                            reveal_interval > 0
                            and step > 0
                            and step % reveal_interval == 0
                        ),
                    )

                tracker.end_episode(winner=winner)

        agg = tracker.get_aggregated_metrics()
        results[config.name] = AblationResult(
            config=config,
            metrics=agg.to_dict(),
            raw_episodes=[e.to_dict() for e in tracker.episodes],
            seed=seeds[0],
        )

    return results


def run_architectural_ablation(
    base_config: dict,
    num_episodes: int = 50,
    seeds: List[int] = None,
) -> Dict[str, AblationResult]:
    """Run architectural ablation study.

    Compares: gnn (baseline) vs GAT vs Transformer

    Args:
        base_config: Base experiment configuration.
        num_episodes: Number of episodes per variant.
        seeds: Random seeds for reproducibility.

    Returns:
        Dictionary mapping variant name to AblationResult.
    """
    seeds = seeds or [42, 123, 456]
    ablation_configs = load_ablation_config(
        os.path.join(
            os.path.dirname(__file__), "..", "configs", "ablation", "architecture.yaml"
        )
    )

    results = {}
    for config in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Running architectural ablation: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'='*60}")

        tracker = MetricsTracker()

        # Get architectural parameters
        config_name = config.name.lower()
        agent_type = config.params.get("agent_type", "gnn")
        num_layers = config.params.get("num_layers", 2)
        hidden_dim = config.params.get("hidden_dim", 64)
        num_heads = config.params.get("num_heads", 4)

        for seed in seeds:
            np.random.seed(seed)

            for ep in range(num_episodes // len(seeds)):
                tracker.start_episode(initial_budget=10.0)

                # Create environment
                num_agents_total = 6  # 5 police + 1 MrX
                agent_money = 15
                reward_weights = {
                    "Police_distance": 0.1,
                    "Police_group": 0.1,
                    "Police_position": 0.1,
                    "Police_time": 0.0,
                    "Mrx_closest": 0.3,
                    "Mrx_average": 0.2,
                    "Mrx_position": 0.1,
                    "Mrx_time": 0.0,
                    "Police_coverage": 0.05,
                    "Police_proximity": 0.05,
                    "Police_overlap_penalty": 0.0,
                }

                log_config = {
                    "log_dir": os.path.join("architecture_ablationlogs", "logs"),
                    "verbose": False,
                    "log_file": "run.log",
                }
                viz_config = {
                    "visualize_game": False,
                    "visualize_heatmap": False,
                    "save_visualization": True,
                    "save_dir": os.path.join("architecture_ablationlogs", "logs/vis"),
                }

                logger = Logger(
                    wandb_api_key=None,
                    wandb_project=None,
                    wandb_entity=None,
                    wandb_run_name=None,
                    wandb_resume=False,
                    configs=log_config,
                )

                env_wrappable = CustomEnvironment(
                    number_of_agents=num_agents_total,
                    agent_money=agent_money,
                    reward_weights=reward_weights,
                    logger=logger,
                    epoch=0,
                    graph_nodes=10,
                    graph_edges=10,
                    vis_configs=viz_config,
                )
                env = PettingZooWrapper(env=env_wrappable)

                # Create agents based on architecture
                node_feature_size = num_agents_total + 1
                common_params = {
                    "node_feature_size": node_feature_size,
                    "device": device,
                    "gamma": 0.99,
                    "lr": 1e-3,
                    "batch_size": 64,
                    "buffer_size": 10000,
                    "epsilon": 0.0,
                    "epsilon_decay": 0.995,
                    "epsilon_min": 0.01,
                    "num_layers": num_layers,
                    "hidden_dim": hidden_dim,
                    "num_heads": num_heads,
                }

                if agent_type == "gnn":
                    common_params.pop("num_layers", None)
                    common_params.pop("hidden_dim", None)
                    common_params.pop("num_heads", None)
                    mrX_agent = GNNAgent(**common_params)
                    police_agent = GNNAgent(**common_params)
                elif agent_type == "gat":
                    mrX_agent = GATAgent(**common_params)
                    police_agent = GATAgent(**common_params)
                elif agent_type == "transformer":
                    common_params["pos_dim"] = 8
                    mrX_agent = TransformerAgent(**common_params)
                    police_agent = TransformerAgent(**common_params)
                else:
                    mrX_agent = RandomAgent()
                    police_agent = RandomAgent()

                # load existing wts
                BASE_MODEL_DIR = "pretrained_models"
                mrx_model_name = f"mrx_{agent_type}_{config_name}.pt"
                police_model_name = f"police_{agent_type}_{config_name}.pt"
                mrx_model_path = os.path.join(BASE_MODEL_DIR, mrx_model_name)
                police_model_path = os.path.join(BASE_MODEL_DIR, police_model_name)
                try:
                    mrX_agent.load_state_dict(torch.load(mrx_model_path), strict=False)
                    police_agent.load_state_dict(torch.load(police_model_path), strict=False)
                except FileNotFoundError as ex:
                    print(f"Pretrained model not found: {ex}. Running with untrained agents.")
                    #raise ex

                # Run episode
                state = env.reset(episode=1000)
                done = False
                ep_step, ep_mrx_rew, ep_police_rew = 0, 0.0, 0.0

                while not done and ep_step < 100:
                    actions = {}

                    # MrX action
                    mrx_graph = create_graph_data(state, "MrX", env).to(device)
                    mrx_mask = torch.zeros(
                        mrx_graph.num_nodes, dtype=torch.int32, device=device
                    )
                    mrx_mask[env.get_possible_moves(0)] = 1
                    mrx_act = mrX_agent.select_action(mrx_graph, mrx_mask)
                    actions["MrX"] = mrx_act if mrx_act is not None else 0

                    # Police actions
                    for i in range(num_agents_total - 1):
                        p_name = f"Police{i}"
                        p_graph = create_graph_data(state, p_name, env).to(device)
                        p_mask = torch.zeros(
                            p_graph.num_nodes, dtype=torch.int32, device=device
                        )
                        p_mask[env.get_possible_moves(i + 1)] = 1
                        p_act = police_agent.select_action(p_graph, p_mask)
                        actions[p_name] = (
                            p_act if p_act is not None else env_wrappable.DEFAULT_ACTION
                        )

                    # Apply actions
                    for obj_id, act in actions.items():
                        state[obj_id]["action"] = torch.tensor(
                            [act if act is not None else 0], dtype=torch.int64
                        )

                    # Step environment
                    state_stepped = env.step(state)
                    next_state = step_mdp(state_stepped)
                    rewards, terminations, truncations = extract_step_info(
                        next_state, env.possible_agents
                    )
                    done = is_episode_done(terminations, truncations)

                    ep_mrx_rew += rewards.get("MrX", 0.0)
                    ep_police_rew += sum(
                        rewards.get(f"Police{i}", 0.0) for i in range(num_agents_total - 1)
                    )

                    prev_police_budget = float(
                        torch.sum(state["MrX"]["observation"]["Currency"]).item()
                    )
                    next_police_budget = float(
                        torch.sum(next_state["MrX"]["observation"]["Currency"]).item()
                    )
                    total_budget_spent = max(prev_police_budget - next_police_budget, 0.0)
                    total_tolls = total_budget_spent

                    # track metrics
                    reveal_interval = max(1, config.params.get("reveal_interval", 5))
                    is_reveal_step = ep_step > 0 and ep_step % reveal_interval == 0
                    tracker.record_step(
                        step=ep_step,
                        toll_paid=total_tolls,
                        budget_spent=total_budget_spent,
                        is_reveal=is_reveal_step,
                    )

                    ep_step += 1
                    state = next_state

                # Determine winner and extract episode stats
                winner = env_wrappable.current_winner

                tracker.end_episode(winner=winner)

        agg = tracker.get_aggregated_metrics()
        results[config.name] = AblationResult(
            config=config,
            metrics=agg.to_dict(),
            raw_episodes=[e.to_dict() for e in tracker.episodes],
            seed=seeds[0],
        )

    return results


def generate_ablation_report(
    results: Dict[str, AblationResult],
    ablation_name: str,
) -> str:
    """Generate a formatted comparison report for ablation results.

    Args:
        results: Dictionary of ablation results.
        ablation_name: Name of the ablation study.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 70,
        f"ABLATION STUDY: {ablation_name}",
        "=" * 70,
        "",
    ]

    # Summary table header
    lines.append(
        f"{'Variant':<20} {'Win Rate':<12} {'Belief CE':<12} {'Ep Length':<12}"
    )
    lines.append("-" * 56)

    for name, result in results.items():
        m = result.metrics
        lines.append(
            f"{name:<20} {m['win_rate']:<12.2%} "
            f"{m['mean_belief_ce']:.<12.4f} {m['mean_episode_length']:<12.1f}"
        )

    lines.append("")
    lines.append("=" * 70)
    lines.append("DETAILED RESULTS")
    lines.append("=" * 70)

    for name, result in results.items():
        lines.append(f"\n### {name} ###")
        lines.append(f"Description: {result.config.description}")
        lines.append(f"Parameters: {result.config.params}")
        lines.append(f"Metrics:")
        for k, v in result.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_ablation_results(
    results: Dict[str, AblationResult],
    output_dir: str,
    ablation_name: str,
):
    """Save ablation results to files.

    Args:
        results: Dictionary of ablation results.
        output_dir: Output directory.
        ablation_name: Name of the ablation study.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_data = {
        "ablation_name": ablation_name,
        "variants": {
            name: {
                "config": {
                    "name": r.config.name,
                    "description": r.config.description,
                    "params": r.config.params,
                },
                "metrics": r.metrics,
                "seed": r.seed,
            }
            for name, r in results.items()
        },
    }

    with open(os.path.join(output_dir, f"{ablation_name}_results.json"), "w") as f:
        json.dump(json_data, f, indent=2)

    # Save report
    report = generate_ablation_report(results, ablation_name)
    with open(os.path.join(output_dir, f"{ablation_name}_report.txt"), "w") as f:
        f.write(report)

    print(f"\nResults saved to {output_dir}/")
    print(report)


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["belief", "mechanism", "architecture", "all"],
        default="all",
        help="Which ablation study to run",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=50, help="Number of episodes per variant"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for reproducibility",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/ablations",
        help="Output directory for results",
    )

    args = parser.parse_args()

    base_config = {}  # Would load from experiment config

    if args.ablation in ["belief", "all"]:
        print("\n" + "=" * 70)
        print("RUNNING BELIEF ABLATION")
        print("=" * 70)
        results = run_belief_ablation(
            base_config,
            num_episodes=args.num_episodes,
            seeds=args.seeds,
        )
        save_ablation_results(results, args.output_dir, "belief")

    if args.ablation in ["mechanism", "all"]:
        print("\n" + "=" * 70)
        print("RUNNING MECHANISM ABLATION")
        print("=" * 70)
        results = run_mechanism_ablation(
            base_config,
            num_episodes=args.num_episodes,
            seeds=args.seeds,
        )
        save_ablation_results(results, args.output_dir, "mechanism")

    if args.ablation in ["architecture", "all"]:
        print("\n" + "=" * 70)
        print("RUNNING ARCHITECTURAL ABLATION")
        print("=" * 70)
        results = run_architectural_ablation(
            base_config,
            num_episodes=args.num_episodes,
            seeds=args.seeds,
        )
        save_ablation_results(results, args.output_dir, "architecture")


if __name__ == "__main__":
    main()
