"""Attention Correlation Analysis for Scotland Yard.

This script quantifies how agent attention correlates with strategic events
like the proximity of MrX or capture events.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from agent.graph_dqn_agent import GraphDQNAgent
from environment.yard import CustomEnvironment
from training.utils import create_graph_data, device

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _SilentLogger:
    def log(self, *a, **kw):
        pass

    def log_scalar(self, *a, **kw):
        pass

    def close(self):
        pass


def compute_focus_metrics(edge_index, alpha, mrx_pos):
    """
    Compute metrics for attention focus on MrX.

    Returns:
        mrx_attention: Mean attention on edges pointing to MrX.
        avg_attention: Mean attention on all other edges.
        focus_ratio: mrx_attention / avg_attention
    """
    # alpha is [E, heads], average over heads
    weights = alpha.mean(dim=-1).detach().cpu().numpy()
    dst_nodes = edge_index[1].detach().cpu().numpy()

    mrx_indices = np.where(dst_nodes == mrx_pos)[0]
    other_indices = np.where(dst_nodes != mrx_pos)[0]

    mrx_att = np.mean(weights[mrx_indices]) if len(mrx_indices) > 0 else 0.0
    other_att = np.mean(weights[other_indices]) if len(other_indices) > 0 else 1e-8

    return mrx_att, other_att, mrx_att / (other_att + 1e-10)


def main():
    parser = argparse.ArgumentParser(
        description="Correlate attention with strategic events."
    )
    parser.add_argument("--arch", choices=["gat", "transformer"], default="gat")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to MrX.pt or Police.pt"
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out_dir", default="artifacts/semester_contribution/analysis")
    args = parser.parse_args()

    # Load agent (placeholder for simplicity, usually loaded via config)
    # Using default eval params
    agent = GraphDQNAgent(
        node_feature_size=4,  # Assuming 3 police + 1 mrx
        device=device,
        agent_type=args.arch,
        model_kwargs=(
            {"hidden_dim": 128, "num_layers": 3, "heads": 4}
            if args.arch == "gat"
            else {"hidden_dim": 128, "num_layers": 3, "num_heads": 8}
        ),
    )
    agent.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True),
        strict=False,
    )
    agent.model.eval()

    env_wrappable = CustomEnvironment(
        number_of_agents=4,
        agent_money=20,
        reward_weights={},
        logger=_SilentLogger(),
        graph_nodes=30,
        graph_edges=60,
        vis_configs={
            "visualize_game": False,
            "visualize_heatmap": False,
            "save_visualization": False,
            "save_dir": "logs/vis",
        },
    )
    env = PettingZooWrapper(env=env_wrappable)

    distances = []
    focus_ratios = []
    capture_events = []

    print(f"Running {args.episodes} episodes for correlation analysis...")

    for ep in range(args.episodes):
        state = env.reset(episode=ep)
        done = False
        while not done:
            mrx_pos = env_wrappable.MrX_pos[0]
            # Assume we are analyzing Police0's attention
            pname = "Police0"
            pg = create_graph_data(state, pname, env).to(device)
            p_mask = torch.ones(pg.num_nodes, dtype=torch.int32, device=device)

            # Distance Police0 -> MrX
            p_pos = env_wrappable.police_positions[0]
            dist = env_wrappable.pathfinder.get_distance(p_pos, mrx_pos)

            with torch.no_grad():
                q_vals, attentions = agent.model(pg, return_attention=True)

            # Use last layer attention
            edge_index, alpha = attentions[-1]
            mrx_att, other_att, focus = compute_focus_metrics(
                edge_index, alpha, mrx_pos
            )

            distances.append(dist)
            focus_ratios.append(focus)

            # Step env
            state["Police0"]["action"] = torch.tensor(
                [agent.select_action(pg, p_mask)], dtype=torch.int64
            )
            # Fill other agents with random or stay
            for a in env.possible_agents:
                if a != "Police0":
                    state[a]["action"] = torch.tensor([0], dtype=torch.int64)

            state_stepped = env.step(state)
            state = state_stepped  # Simplified

            if env_wrappable.current_winner == "Police":
                capture_events.append(len(distances) - 1)

            if "terminations" in state_stepped:  # Simplified done check
                done = True

    # Compute correlation
    corr = np.corrcoef(distances, focus_ratios)[0, 1]
    print(f"\nCorrelation (Distance vs Attention Focus): {corr:.4f}")

    # Plotting
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.scatter(distances, focus_ratios, alpha=0.5)
    plt.xlabel("Distance to MrX (Shortest Path)")
    plt.ylabel("Attention Focus Ratio (MrX node / Other nodes)")
    plt.title(f"Attention Interpretability: {args.arch.upper()} Focus vs Proximity")
    plt.savefig(f"{args.out_dir}/attention_correlation_{args.arch}.png")
    print(f"Plot saved to {args.out_dir}")


if __name__ == "__main__":
    main()
