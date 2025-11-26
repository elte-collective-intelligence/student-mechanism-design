"""Action masking utilities for budget- and topology-constrained moves."""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class ActionMaskResult:
    mask: np.ndarray
    index_to_node: Dict[int, int]


def compute_action_mask(adjacency: np.ndarray, current_node: int, budget: float, tolls: np.ndarray | float | None = None) -> ActionMaskResult:
    """Compute an action mask and mapping for the given node.

    Args:
        adjacency: binary adjacency matrix describing the graph.
        current_node: node where the agent is located.
        budget: remaining budget for the agent.
        tolls: either a scalar toll, a vector per node, or a matrix aligning with
            the adjacency matrix.
    """

    num_nodes = adjacency.shape[0]
    mask = np.zeros(num_nodes, dtype=bool)
    index_to_node = {}

    toll_matrix = _normalize_tolls(tolls, num_nodes)
    for node in range(num_nodes):
        if node == current_node:
            continue
        if adjacency[current_node, node] == 0:
            continue
        toll = toll_matrix[current_node, node]
        if toll <= budget:
            idx = len(index_to_node)
            index_to_node[idx] = node
            mask[idx] = True
    return ActionMaskResult(mask=mask, index_to_node=index_to_node)


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
