"""Action masking utilities for budget- and topology-constrained moves."""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class ActionMaskResult:
    """Result of action mask computation.

    Attributes:
        mask: Boolean array of size num_nodes where mask[node] = True if the agent
              can move to that node.
        index_to_node: Fixed mapping from action index to node ID (identity mapping).
        valid_actions: List of node indices that are valid actions.
        node_to_index: Reverse mapping from node ID to action index.
    """

    mask: np.ndarray
    index_to_node: Dict[int, int]
    valid_actions: List[int]
    node_to_index: Dict[int, int]

    @property
    def num_valid_actions(self) -> int:
        return len(self.valid_actions)


def compute_action_mask(
    adjacency: np.ndarray,
    current_node: int,
    budget: float,
    tolls: np.ndarray | float | None = None,
    edge_weights: np.ndarray | None = None,
) -> ActionMaskResult:
    """Compute an action mask and mapping for the given node.

    The mask uses a FIXED index→node mapping where action index i corresponds to node i.
    This ensures consistency across all agents and timesteps.

    Args:
        adjacency: Binary adjacency matrix describing the graph.
        current_node: Node where the agent is located.
        budget: Remaining budget for the agent.
        tolls: Either a scalar toll, a vector per node, or a matrix aligning with
            the adjacency matrix.
        edge_weights: Optional edge weight matrix for cost calculation. If provided,
            the cost to move is edge_weight + toll.

    Returns:
        ActionMaskResult with mask, mappings, and valid action list.
    """
    num_nodes = adjacency.shape[0]
    mask = np.zeros(num_nodes, dtype=bool)
    valid_actions = []

    # Fixed identity mapping: action index i = node i
    index_to_node = {i: i for i in range(num_nodes)}
    node_to_index = {i: i for i in range(num_nodes)}

    toll_matrix = _normalize_tolls(tolls, num_nodes)
    weight_matrix = _normalize_weights(edge_weights, adjacency, num_nodes)

    for node in range(num_nodes):
        if node == current_node:
            continue
        if adjacency[current_node, node] == 0:
            continue

        # Total cost = edge weight + toll
        cost = weight_matrix[current_node, node] + toll_matrix[current_node, node]

        if cost <= budget:
            mask[node] = True  # Fixed: use node index directly
            valid_actions.append(node)

    return ActionMaskResult(
        mask=mask,
        index_to_node=index_to_node,
        valid_actions=valid_actions,
        node_to_index=node_to_index,
    )


def _normalize_tolls(tolls, num_nodes: int) -> np.ndarray:
    if tolls is None:
        return np.zeros((num_nodes, num_nodes))
    if np.isscalar(tolls):
        return np.full((num_nodes, num_nodes), float(tolls))
    tolls_array = np.asarray(tolls, dtype=float)
    if tolls_array.ndim == 1:
        # interpret as per-destination tolls
        tiled = np.tile(tolls_array.reshape(1, -1), (num_nodes, 1))
        return tiled
    return tolls_array


def _normalize_weights(weights, adjacency: np.ndarray, num_nodes: int) -> np.ndarray:
    """Normalize edge weights to a matrix format.

    Args:
        weights: Edge weight matrix, or None.
        adjacency: Adjacency matrix to use for default weights.
        num_nodes: Number of nodes in the graph.

    Returns:
        Edge weight matrix where weights[i,j] is the cost to traverse from i to j.
    """
    if weights is None:
        # Default: use adjacency as unit weights (cost 1 for each edge)
        return adjacency.astype(float)
    return np.asarray(weights, dtype=float)


def get_action_mask_for_agent(
    adjacency: np.ndarray,
    edge_weights: np.ndarray,
    agent_position: int,
    agent_budget: float,
    tolls: np.ndarray | float | None = None,
) -> ActionMaskResult:
    """Convenience function to get action mask for an agent.

    This is the primary interface for the environment wrapper.

    Args:
        adjacency: Binary adjacency matrix.
        edge_weights: Matrix of edge weights/costs.
        agent_position: Current node of the agent.
        agent_budget: Remaining budget.
        tolls: Optional toll configuration.

    Returns:
        ActionMaskResult with the valid action mask.
    """
    return compute_action_mask(
        adjacency=adjacency,
        current_node=agent_position,
        budget=agent_budget,
        tolls=tolls,
        edge_weights=edge_weights,
    )
