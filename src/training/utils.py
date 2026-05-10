"""Training utilities and helper functions.

This module contains shared utilities used by training and evaluation functions
to avoid code duplication in main.py.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

from torch_geometric.data import Data

from reward_net import REWARD_WEIGHT_NAMES

# Device setup - use GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def create_curriculum(
    num_epochs: int,
    base_graph_nodes: int,
    base_graph_edges: int,
    base_money: float,
    curriculum_range: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create curriculum arrays for progressive difficulty scaling.

    The curriculum gradually increases graph complexity while decreasing
    agent money, making the game progressively harder.

    Args:
        num_epochs: Total number of training epochs.
        base_graph_nodes: Target number of graph nodes.
        base_graph_edges: Target number of graph edges.
        base_money: Target agent money.
        curriculum_range: Fraction of base values to vary (default 0.5 = ±50%).

    Returns:
        Tuple of (node_curriculum, edge_curriculum, money_curriculum) arrays.
        Each array has length num_epochs.
    """
    if num_epochs <= 1:
        return (
            np.asarray([base_graph_nodes]),
            np.asarray([base_graph_edges]),
            np.asarray([base_money]),
        )

    # Nodes and edges increase over training
    node_curriculum = np.arange(
        base_graph_nodes - curriculum_range * base_graph_nodes,
        base_graph_nodes + curriculum_range * base_graph_nodes + 1,
        (2 * curriculum_range * base_graph_nodes) / max(num_epochs - 1, 1),
    )

    edge_curriculum = np.arange(
        base_graph_edges - curriculum_range * base_graph_edges,
        base_graph_edges + curriculum_range * base_graph_edges + 1,
        (2 * curriculum_range * base_graph_edges) / max(num_epochs - 1, 1),
    )

    # Money decreases over training (harder for agents)
    money_curriculum = np.arange(
        base_money + curriculum_range * base_money,
        base_money - curriculum_range * base_money - 1,
        -(2 * curriculum_range * base_money) / max(num_epochs - 1, 1),
    )

    return node_curriculum, edge_curriculum, money_curriculum


def modify_curriculum(
    win_ratio: float,
    node_curriculum: np.ndarray,
    edge_curriculum: np.ndarray,
    money_curriculum: np.ndarray,
    modification_rate: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adjust curriculum based on agent performance.

    If MrX wins too often, difficulty increases. If Police wins too often,
    difficulty decreases.

    Args:
        win_ratio: Fraction of games won by MrX (0.0 to 1.0).
        node_curriculum: Current node curriculum array.
        edge_curriculum: Current edge curriculum array.
        money_curriculum: Current money curriculum array.
        modification_rate: How aggressively to adjust (default 0.1 = 10%).

    Returns:
        Modified (node_curriculum, edge_curriculum, money_curriculum) arrays.
    """
    # win_ratio=0.5 -> no change, win_ratio=1.0 -> increase, win_ratio=0.0 -> decrease
    modification_percentage = (
        1.0 + (2.0 * modification_rate) * win_ratio - modification_rate
    )

    return (
        node_curriculum * modification_percentage,
        edge_curriculum * modification_percentage,
        money_curriculum * modification_percentage,
    )


def compute_target_difficulty(win_ratio: float, target_balance: float = 0.5) -> float:
    """Compute target difficulty for the meta-learner.

    The goal is balanced gameplay where both MrX and Police have equal
    chances of winning.

    Args:
        win_ratio: Current win ratio (not used in simple version).
        target_balance: Desired win ratio (default 0.5 = 50% each).

    Returns:
        Target difficulty value for the loss function.
    """
    return target_balance


def predict_reward_weights(
    reward_weight_net: nn.Module,
    num_agents: int,
    agent_money: float,
    graph_nodes: int,
    graph_edges: int,
) -> Dict[str, torch.Tensor]:
    """Predict reward weights using the meta-learning network.

    Args:
        reward_weight_net: The trained RewardWeightNet model.
        num_agents: Number of agents in the game.
        agent_money: Money available to agents.
        graph_nodes: Number of nodes in the graph.
        graph_edges: Number of edges in the graph.

    Returns:
        Dictionary mapping reward weight names to predicted values.
    """
    inputs = torch.FloatTensor(
        [[num_agents, agent_money, graph_nodes, graph_edges]]
    ).to(device)

    predicted_weight = reward_weight_net(inputs)

    return {name: predicted_weight[0, i] for i, name in enumerate(REWARD_WEIGHT_NAMES)}


def create_graph_data(state: Dict, env) -> Data:
    """Create a shared PyTorch Geometric Data object for all agents.

    Uses pre-moved device tensors for edge structure and vectorized node
    feature assignment for speed.
    """
    env_inner = getattr(env, "_env", env)

    # Retrieve pre-cached device tensors (moved in yard.py)
    edge_index = env_inner._edge_index_tensor
    edge_features = env_inner._edge_features_tensor

    num_nodes = env.board.nodes.shape[0]
    num_agents_total = env.number_of_agents + 1

    # Vectorized node feature initialization
    node_features = torch.zeros(
        num_nodes, num_agents_total, dtype=torch.float32, device=device
    )

    # Fast vectorized position encoding
    agent_names = ["MrX"] + [f"Police{i}" for i in range(env.number_of_agents)]
    for i, name in enumerate(agent_names):
        agent_data = state.get(name, {}).get("observation", {})
        pos = agent_data.get("agent_position")
        if pos is not None:
            # Handle tensors with potential batch/extra dims
            if isinstance(pos, torch.Tensor):
                idx = int(pos.flatten()[0].item())
            else:
                idx = int(pos)
            node_features[idx, i] = 1.0

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)


def extract_step_info(
    next_state: Dict, possible_agents: List[str]
) -> Tuple[Dict, Dict, Dict]:
    """Extract rewards, terminations, and truncations from stepped state.

    Args:
        next_state: State dictionary after environment step.
        possible_agents: List of agent IDs.

    Returns:
        Tuple of (rewards, terminations, truncations) dictionaries.
    """
    rewards = {
        agent_id: next_state[agent_id]["reward"].squeeze()
        for agent_id in possible_agents
    }
    terminations = {
        agent_id: next_state[agent_id]["terminated"].squeeze()
        for agent_id in possible_agents
    }
    truncations = {
        agent_id: next_state[agent_id]["truncated"].squeeze()
        for agent_id in possible_agents
    }
    return rewards, terminations, truncations


def is_episode_done(terminations: Dict, truncations: Dict) -> bool:
    """Check if the episode has ended.

    Args:
        terminations: Dictionary of agent termination flags.
        truncations: Dictionary of agent truncation flags.

    Returns:
        True if episode is done, False otherwise.
    """
    return any(terminations.values()) or all(truncations.values())


def create_action_mask(
    num_actions: int, possible_moves: List[int], dtype=torch.float32
) -> torch.Tensor:
    """Create an action mask tensor using vectorized indexing.

    Args:
        num_actions: Total number of possible actions (mask size).
        possible_moves: List of valid action indices.
        dtype: Tensor dtype (default torch.float32).

    Returns:
        Tensor with 1s at valid action indices, 0s elsewhere.
    """
    mask = torch.zeros(num_actions, dtype=dtype, device=device)
    if len(possible_moves) > 0:
        moves = torch.tensor(possible_moves, dtype=torch.long, device=device)
        moves = moves[moves < num_actions]
        if len(moves) > 0:
            mask[moves] = 1.0
    return mask
