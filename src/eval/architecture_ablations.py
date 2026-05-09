"""Architecture ablation study for Scotland Yard.

Evaluates pre-trained GNN, GAT, and Transformer checkpoints across a
hyperparameter grid and saves one results row per
(arch, n_layers, hidden_dim, n_heads, seed, graph_size) to a CSV.

Usage:
    python src/eval/architecture_ablations.py
    python src/eval/architecture_ablations.py --config src/configs/eval/ablation.yaml
    python src/eval/architecture_ablations.py --arch gat transformer
    python src/eval/architecture_ablations.py --dry_run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from torchrl.envs import step_mdp
from torchrl.envs.libs.pettingzoo import PettingZooWrapper

from agent.graph_dqn_agent import GraphDQNAgent
from environment.yard import CustomEnvironment
from training.utils import (
    create_graph_data,
    device,
    extract_step_info,
    is_episode_done,
)

class _SilentLogger:
    """Minimal logger stub for evaluation — suppresses all output."""
    def log(self, *a, **kw): pass
    def log_scalar(self, *a, **kw): pass
    def log_model(self, *a, **kw): pass
    def model_exists(self, *a, **kw): return False
    def close(self): pass


_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "configs", "eval", "ablation.yaml"
)

# Fixed DQN hyper-params shared across all architectures during evaluation.
# These are not swept — only architecture-specific params vary.
_DQN_EVAL_DEFAULTS = dict(
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    buffer_size=1,   # minimal buffer; no training during eval
    epsilon=0.0,
    epsilon_decay=1.0,
    epsilon_min=0.0,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def checkpoint_dir_for(
    checkpoint_root: str,
    arch: str,
    n_layers: int,
    hidden_dim: int,
    n_heads: int,
    seed: int,
    graph_size: str,
) -> Path:
    run_name = f"{arch}_L{n_layers}_H{hidden_dim}_h{n_heads}_s{seed}_{graph_size}"
    return Path(checkpoint_root) / run_name


def _build_model_kwargs(arch: str, n_layers: int, hidden_dim: int, n_heads: int) -> dict:
    if arch == "gat":
        return dict(
            hidden_dim=hidden_dim,
            num_layers=n_layers,
            heads=n_heads,
            dropout=0.2,
            edge_dim=1,
        )
    if arch == "transformer":
        return dict(
            hidden_dim=hidden_dim,
            num_layers=n_layers,
            num_heads=n_heads,
            dropout=0.1,
            edge_dim=1,
            use_positional_encoding=True,
            pe_dim=8,
        )
    # gnn — model ignores extra kwargs
    return {}


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def count_parameters(agent: GraphDQNAgent) -> int:
    return sum(p.numel() for p in agent.model.parameters() if p.requires_grad)


def action_entropy(q_values: torch.Tensor, action_mask: torch.Tensor) -> float:
    """Shannon entropy of the softmax policy over valid actions."""
    valid = action_mask.bool()
    if valid.sum() == 0:
        return 0.0
    probs = torch.softmax(q_values[valid].float(), dim=0)
    ent = -(probs * torch.log(probs + 1e-8)).sum().item()
    return float(ent)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _try_load_checkpoint(path: Path, agent: GraphDQNAgent) -> bool:
    if not path.exists():
        return False
    state_dict = torch.load(str(path), map_location=device, weights_only=True)
    agent.load_state_dict(state_dict, strict=False)
    return True


def build_agent(
    arch: str,
    n_layers: int,
    hidden_dim: int,
    n_heads: int,
    node_feature_size: int,
) -> GraphDQNAgent:
    model_kwargs = _build_model_kwargs(arch, n_layers, hidden_dim, n_heads)
    return GraphDQNAgent(
        node_feature_size=node_feature_size,
        device=device,
        agent_type=arch,
        model_kwargs=model_kwargs,
        **_DQN_EVAL_DEFAULTS,
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episodes(
    mrx_agent: GraphDQNAgent,
    police_agent: GraphDQNAgent,
    env_wrappable: CustomEnvironment,
    env: PettingZooWrapper,
    n_episodes: int,
    seed_offset: int,
    seed: int,
) -> Dict[str, float]:
    mrx_wins = 0
    episode_lengths = []
    entropies = []

    n_police = env_wrappable.number_of_agents  # police count (MrX is separate)

    for ep in range(n_episodes):
        torch.manual_seed(seed_offset + seed * 1000 + ep)
        np.random.seed(seed_offset + seed * 1000 + ep)

        state = env.reset(episode=ep)
        done = False
        steps = 0

        while not done:
            actions = {}

            # MrX
            mrx_graph = create_graph_data(state, "MrX", env).to(device)
            mrx_possible = env.get_possible_moves(0)
            mrx_mask = torch.zeros(mrx_graph.num_nodes, dtype=torch.int32, device=device)
            mrx_mask[mrx_possible] = 1

            with torch.no_grad():
                if mrx_agent.agent_type in ["gat", "transformer"]:
                    q_mrx, _ = mrx_agent.model(mrx_graph, return_attention=True)
                else:
                    q_mrx = mrx_agent.model(mrx_graph)
            entropies.append(action_entropy(q_mrx, mrx_mask))

            actions["MrX"] = mrx_agent.select_action(mrx_graph, mrx_mask)

            # Police
            for i in range(n_police):
                pname = f"Police{i}"
                pg = create_graph_data(state, pname, env).to(device)
                pm = env.get_possible_moves(i + 1)
                pmask = torch.zeros(pg.num_nodes, dtype=torch.int32, device=device)
                pmask[pm] = 1
                pa = police_agent.select_action(pg, pmask)
                actions[pname] = pa if pa is not None else env_wrappable.DEFAULT_ACTION

            for obj_id, act in actions.items():
                state[obj_id]["action"] = torch.tensor(
                    [act if act is not None else env_wrappable.DEFAULT_ACTION],
                    dtype=torch.int64,
                )

            state_stepped = env.step(state)
            next_state = step_mdp(state_stepped)
            rewards, terminations, truncations = extract_step_info(
                next_state, env.possible_agents
            )
            done = is_episode_done(terminations, truncations)
            state = next_state
            steps += 1

        if env_wrappable.current_winner == "MrX":
            mrx_wins += 1
        episode_lengths.append(steps)

    total = n_episodes
    return {
        "win_rate": mrx_wins / total if total > 0 else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
    }


# ---------------------------------------------------------------------------
# Single configuration evaluation
# ---------------------------------------------------------------------------

_NO_VIS_CONFIG = {
    "visualize_game": False,
    "visualize_heatmap": False,
    "save_visualization": False,
    "save_dir": "logs/vis",
}

_NULL_REWARD_WEIGHTS = {
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


def run_single_eval(
    arch: str,
    n_layers: int,
    hidden_dim: int,
    n_heads: int,
    seed: int,
    size_name: str,
    size_cfg: dict,
    cfg: dict,
    dry_run: bool = False,
) -> Optional[dict]:
    checkpoint_root = cfg["checkpoint_dir"]
    ckpt_dir = checkpoint_dir_for(
        checkpoint_root, arch, n_layers, hidden_dim, n_heads, seed, size_name
    )
    mrx_path = ckpt_dir / "MrX.pt"
    police_path = ckpt_dir / "Police.pt"

    if dry_run:
        print(f"  [dry_run] would load: {ckpt_dir}")
        return None

    num_police = size_cfg["num_police_agents"]
    num_agents = num_police + 1   # +1 for MrX
    node_feature_size = num_agents + 1

    mrx_agent = build_agent(arch, n_layers, hidden_dim, n_heads, node_feature_size)
    police_agent = build_agent(arch, n_layers, hidden_dim, n_heads, node_feature_size)

    mrx_loaded = _try_load_checkpoint(mrx_path, mrx_agent)
    police_loaded = _try_load_checkpoint(police_path, police_agent)

    if not mrx_loaded and not police_loaded:
        print(f"  [skip] checkpoint not found: {ckpt_dir}")
        return None

    if not mrx_loaded:
        print(f"  [warn] MrX checkpoint missing, using untrained: {mrx_path}")
    if not police_loaded:
        print(f"  [warn] Police checkpoint missing, using untrained: {police_path}")

    env_wrappable = CustomEnvironment(
        number_of_agents=num_agents,
        agent_money=size_cfg["agent_money"],
        reward_weights=_NULL_REWARD_WEIGHTS,
        logger=_SilentLogger(),
        epoch=0,
        graph_nodes=size_cfg["graph_nodes"],
        graph_edges=size_cfg["graph_edges"],
        vis_configs=_NO_VIS_CONFIG,
    )
    env = PettingZooWrapper(env=env_wrappable)

    metrics = _run_episodes(
        mrx_agent=mrx_agent,
        police_agent=police_agent,
        env_wrappable=env_wrappable,
        env=env,
        n_episodes=cfg["n_eval_episodes"],
        seed_offset=cfg["env"]["seed_offset"],
        seed=seed,
    )
    metrics["n_params"] = count_parameters(mrx_agent)
    return metrics


# ---------------------------------------------------------------------------
# Main grid sweep
# ---------------------------------------------------------------------------

def run_ablations(cfg: dict, arch_filter=None, dry_run: bool = False) -> pd.DataFrame:
    results = []
    architectures = cfg["architectures"]
    if arch_filter:
        architectures = [a for a in architectures if a in arch_filter]

    arch_heads = cfg.get("arch_heads", {})

    def _heads_for(arch):
        if arch in arch_heads:
            return arch_heads[arch]
        return cfg["sweep"]["n_heads"] if arch != "gnn" else [1]

    total = sum(
        len(cfg["sweep"]["n_layers"])
        * len(cfg["sweep"]["hidden_dims"])
        * len(_heads_for(arch))
        * len(cfg["seeds"])
        * len(cfg["graph_sizes"])
        for arch in architectures
    )
    done = 0

    for arch in architectures:
        for n_layers in cfg["sweep"]["n_layers"]:
            for hidden_dim in cfg["sweep"]["hidden_dims"]:
                heads_list = _heads_for(arch)
                for n_heads in heads_list:
                    for seed in cfg["seeds"]:
                        for size_name, size_cfg in cfg["graph_sizes"].items():
                            done += 1
                            print(
                                f"[{done}/{total}] arch={arch} layers={n_layers} "
                                f"dim={hidden_dim} heads={n_heads} "
                                f"seed={seed} size={size_name}"
                            )
                            metrics = run_single_eval(
                                arch, n_layers, hidden_dim, n_heads,
                                seed, size_name, size_cfg, cfg,
                                dry_run=dry_run,
                            )
                            if metrics is None:
                                continue
                            results.append({
                                "arch": arch,
                                "n_layers": n_layers,
                                "hidden_dim": hidden_dim,
                                "n_heads": n_heads,
                                "seed": seed,
                                "graph_size": size_name,
                                **metrics,
                            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run architecture ablation studies for Scotland Yard agents."
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="Path to ablation YAML config (default: src/configs/eval/ablation.yaml)",
    )
    parser.add_argument(
        "--arch",
        nargs="+",
        help="Restrict evaluation to specific architectures, e.g. --arch gat transformer",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the run grid without loading any checkpoints",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output CSV path (default: {output_dir}/ablation_results.csv)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config loaded from: {args.config}")
    print(f"Architectures: {cfg['architectures']}")
    print(f"Seeds: {cfg['seeds']}")
    print(f"Graph sizes: {list(cfg['graph_sizes'].keys())}")
    print(f"Device: {device}")

    df = run_ablations(cfg, arch_filter=args.arch, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[dry_run] No results to save.")
        return

    if df.empty:
        print("\nNo checkpoints found — nothing to save.")
        print(
            "Train models and save them following the convention in "
            "src/configs/eval/README.md, then re-run this script."
        )
        return

    out_path = args.output or str(
        Path(cfg["output_dir"]) / "ablation_results.csv"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}  ({len(df)} rows)")
    print(df.groupby("arch")[["win_rate", "mean_episode_length", "n_params"]].mean().to_string())


if __name__ == "__main__":
    main()
