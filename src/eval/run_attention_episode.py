"""Attention-rollout runner for Scotland Yard mechanism design.

Loads a GAT or Transformer agent, rolls out one or more episodes in a small
environment, extracts MrX's attention weights at each step, renders an
attention-overlay GIF, and writes a sidecar JSON of strategic events
(reveal steps, capture step, MrX/police positions) so downstream plots can
correlate attention with gameplay.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402
from environment.yard import CustomEnvironment  # noqa: E402
from torchrl.envs.libs.pettingzoo import PettingZooWrapper  # noqa: E402
from torchrl.envs import step_mdp  # noqa: E402
from agent.gat_agent import GATAgent  # noqa: E402
from agent.transformer_agent import TransformerAgent  # noqa: E402
from logger import Logger  # noqa: E402
from training.utils import (  # noqa: E402
    device,
    create_graph_data,
    extract_step_info,
    is_episode_done,
)
from eval.attention_viz import (  # noqa: E402
    compute_attention_summary,
    render_attention_frame,
    save_attention_gif,
)


@dataclass
class EpisodeEvents:
    """Strategic events logged during a single attention-rollout episode."""

    steps: List[Dict[str, Any]] = field(default_factory=list)
    winner: Optional[str] = None
    capture_step: Optional[int] = None
    reveal_steps: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "winner": self.winner,
            "capture_step": self.capture_step,
            "reveal_steps": self.reveal_steps,
            "steps": self.steps,
        }


def load_runner_config(config_path: str) -> dict:
    """Load the attention-runner YAML config.

    Args:
        config_path: Path to the runner config YAML file.

    Returns:
        Parsed config dict.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _build_agent(agent_type: str, common_params: Dict[str, Any]):
    """Construct a GAT or Transformer agent from common params."""
    if agent_type == "gat":
        return GATAgent(**common_params)
    if agent_type == "transformer":
        common_params = dict(common_params)
        common_params.setdefault("pos_dim", 8)
        return TransformerAgent(**common_params)
    raise ValueError(f"Unsupported agent_type for attention rollout: {agent_type}")


def _try_load_checkpoint(agent, path: Optional[str], label: str) -> None:
    """Load a checkpoint if path is set and the file exists, else fall back."""
    if not path:
        print(f"No checkpoint for {label}, running with untrained weights.")
        return
    try:
        agent.load_state_dict(torch.load(path, map_location=device), strict=False)
        print(f"Loaded {label} checkpoint from {path}.")
    except FileNotFoundError:
        print(f"Checkpoint not found for {label}: {path}. Using untrained weights.")


def run_attention_episode(
    config: dict,
    seed: int = 42,
    output_dir: str = "logs/attention",
) -> EpisodeEvents:
    """Run a single attention-rollout episode and write GIF + event JSON.

    Args:
        config: Runner config (agent_type, num_layers, hidden_dim, heads,
            checkpoint_path_mrx, checkpoint_path_police, env params, etc.).
        seed: Random seed for reproducibility.
        output_dir: Directory for GIF + JSON outputs.

    Returns:
        EpisodeEvents capturing the strategic event log.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    agent_type = config.get("agent_type", "gat")
    num_layers = config.get("num_layers", 2)
    hidden_dim = config.get("hidden_dim", 64)
    heads = config.get("heads", 4)
    num_police = config.get("num_police", 5)
    agent_money = config.get("agent_money", 15)
    graph_nodes = config.get("graph_nodes", 10)
    graph_edges = config.get("graph_edges", 10)
    max_steps = config.get("max_steps", 100)
    reveal_interval = max(1, config.get("reveal_interval", 5))

    reward_weights = config.get(
        "reward_weights",
        {
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
        },
    )

    log_config = {
        "log_dir": os.path.join(output_dir, "logs"),
        "verbose": False,
        "log_file": "attention_run.log",
    }
    viz_config = {
        "visualize_game": False,
        "visualize_heatmap": False,
        "save_visualization": False,
        "save_dir": os.path.join(output_dir, "vis"),
    }
    os.makedirs(log_config["log_dir"], exist_ok=True)

    logger = Logger(
        wandb_api_key="",
        wandb_project="",
        wandb_entity="",
        wandb_run_name="",
        wandb_resume=False,
        configs=log_config,
    )

    env_wrappable = CustomEnvironment(
        number_of_agents=num_police,
        agent_money=agent_money,
        reward_weights=reward_weights,
        logger=logger,
        epoch=0,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        vis_configs=viz_config,
    )
    env = PettingZooWrapper(env=env_wrappable)

    node_feature_size = num_police + 1
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
        "heads": heads,
    }

    mrX_agent = _build_agent(agent_type, common_params)
    police_agent = _build_agent(agent_type, common_params)

    _try_load_checkpoint(mrX_agent, config.get("checkpoint_path_mrx"), "MrX")
    _try_load_checkpoint(police_agent, config.get("checkpoint_path_police"), "Police")

    events = EpisodeEvents()
    frames: List[np.ndarray] = []

    state = env.reset(episode=config.get("episode_id", 1000))
    done = False
    ep_step = 0

    while not done and ep_step < max_steps:
        actions: Dict[str, Any] = {}

        mrx_graph = create_graph_data(state, "MrX", env).to(device)
        mrx_mask = torch.zeros(mrx_graph.num_nodes, dtype=torch.int32, device=device)
        mrx_mask[env.get_possible_moves(0)] = 1

        # Capture MrX attention before acting.
        _, attention = mrX_agent.get_attention(mrx_graph)
        edge_index, alpha = attention[-1]
        is_reveal_step = ep_step > 0 and ep_step % reveal_interval == 0
        title = (
            f"step={ep_step}  reveal={is_reveal_step}  "
            f"agent={agent_type}  layer=last"
        )
        frame = render_attention_frame(
            board=env_wrappable.board,
            mrx_pos=env_wrappable.MrX_pos[0],
            police_positions=list(env_wrappable.police_positions),
            edge_index=edge_index,
            alpha=alpha,
            title=title,
        )
        frames.append(frame)

        mrx_act = mrX_agent.select_action(mrx_graph, mrx_mask)
        actions["MrX"] = mrx_act if mrx_act is not None else 0

        for i in range(num_police):
            p_name = f"Police{i}"
            p_graph = create_graph_data(state, p_name, env).to(device)
            p_mask = torch.zeros(p_graph.num_nodes, dtype=torch.int32, device=device)
            p_mask[env.get_possible_moves(i + 1)] = 1
            p_act = police_agent.select_action(p_graph, p_mask)
            actions[p_name] = (
                p_act if p_act is not None else env_wrappable.DEFAULT_ACTION
            )

        for obj_id, act in actions.items():
            state[obj_id]["action"] = torch.tensor(
                [act if act is not None else 0], dtype=torch.int64
            )

        state_stepped = env.step(state)
        next_state = step_mdp(state_stepped)
        _, terminations, truncations = extract_step_info(
            next_state, env.possible_agents
        )
        done = is_episode_done(terminations, truncations)

        summary = compute_attention_summary(
            edge_index=edge_index,
            alpha=alpha,
            mrx_pos=env_wrappable.MrX_pos[0],
            police_positions=list(env_wrappable.police_positions),
            num_nodes=env_wrappable.board.nodes.shape[0],
        )
        events.steps.append(
            {
                "step": ep_step,
                "mrx_pos": int(env_wrappable.MrX_pos[0]),
                "police_positions": [int(p) for p in env_wrappable.police_positions],
                "mrx_action": int(actions["MrX"]),
                "is_reveal": is_reveal_step,
                "attention_summary": summary,
            }
        )
        if is_reveal_step:
            events.reveal_steps.append(ep_step)

        ep_step += 1
        state = next_state

    events.winner = env_wrappable.current_winner
    if events.winner == "Police":
        events.capture_step = max(ep_step - 1, 0)

    gif_path = os.path.join(output_dir, f"attention_seed{seed}.gif")
    json_path = os.path.join(output_dir, f"attention_seed{seed}.json")
    save_attention_gif(frames, gif_path)
    with open(json_path, "w") as f:
        json.dump(events.to_dict(), f, indent=2)
    print(f"Wrote {gif_path} and {json_path} ({len(frames)} frames).")

    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "experiments",
            "attention_episode",
            "config.yaml",
        ),
        help="Path to the attention-runner YAML config.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="logs/attention")
    args = parser.parse_args()

    cfg = load_runner_config(args.config)
    run_attention_episode(cfg, seed=args.seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
