"""Training utilities and helper functions.

This module contains shared utilities used by training and evaluation functions
to avoid code duplication in main.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple

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


def create_graph_data(state: Dict, agent_id: str, env) -> Data:
    """Create a PyTorch Geometric Data object from the environment state.

    This converts the environment state into a graph representation suitable
    for Graph Neural Network agents.

    Args:
        state: Current environment state dictionary.
        agent_id: ID of the agent (e.g., "MrX", "Police0").
        env: The wrapped environment instance.

    Returns:
        PyTorch Geometric Data object with node features, edge indices,
        and edge attributes.
    """
    logger = env.logger
    logger.log(f"Creating graph data for agent {agent_id}.", level="debug")

    # Get graph structure from environment
    edge_index = torch.tensor(env.board.edge_links.T, dtype=torch.long)
    edge_features = torch.tensor(env.board.edges, dtype=torch.float)

    num_nodes = env.board.nodes.shape[0]
    num_features = env.number_of_agents + 1  # One-hot encoding for agent positions

    # Initialize node features
    node_features = np.zeros((num_nodes, num_features), dtype=np.float32)

    # Encode MrX position (feature index 0)
    mrX_pos = state.get("MrX", {}).get("observation", None)
    if mrX_pos is not None:
        mrX_pos = mrX_pos.get("MrX_pos", None)
        if mrX_pos is not None:
            node_features[mrX_pos, 0] = 1
            logger.log(
                f"Agent {agent_id}: MrX position encoded at node {mrX_pos}.",
                level="debug",
            )

    # Encode Police positions (feature indices 1+)
    for i in range(env.number_of_agents - 1):
        police_obs = state.get(f"Police{i}", {}).get("observation", None)
        if police_obs is not None:
            police_pos = police_obs.get("Polices_pos", None)
            if police_pos is not None and len(police_pos) > 0:
                node_features[police_pos[0], i + 1] = 1
                logger.log(
                    f"Agent {agent_id}: Police{i} position encoded at node {police_pos[0]}.",
                    level="debug",
                )

    # Convert to tensors and move to device
    node_features = torch.tensor(node_features, dtype=torch.float32).to(device)
    edge_index = edge_index.to(device)
    edge_features = edge_features.to(device)

    # Create PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    logger.log(f"Graph data for agent {agent_id} created.", level="debug")

    return data


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
    return terminations.get("Police0", False) or all(truncations.values())


def create_action_mask(
    num_actions: int, possible_moves: List[int], dtype=torch.float32
) -> torch.Tensor:
    """Create an action mask tensor.

    Args:
        num_actions: Total number of possible actions (mask size).
        possible_moves: List of valid action indices.
        dtype: Tensor dtype (default torch.float32).

    Returns:
        Tensor with 1s at valid action indices, 0s elsewhere.
    """
    mask = torch.zeros(num_actions, dtype=dtype, device=device)
    for move in possible_moves:
        if move < num_actions:
            mask[move] = 1.0
    return mask
