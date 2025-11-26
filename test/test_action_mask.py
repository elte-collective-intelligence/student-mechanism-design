import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from Enviroment.action_mask import compute_action_mask


def test_action_mask_respects_budget_and_mapping():
    adjacency = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ])
    tolls = np.array([
        [0, 2, 4],
        [2, 0, 1],
        [4, 1, 0],
    ])
    result = compute_action_mask(adjacency, current_node=0, budget=3, tolls=tolls)
    # Only node 1 should be affordable
    assert result.mask.sum() == 1
    mapped_nodes = list(result.index_to_node.values())
    assert mapped_nodes == [1]


def test_action_mask_with_scalar_toll():
    adjacency = np.ones((2, 2)) - np.eye(2)
    result = compute_action_mask(adjacency, current_node=0, budget=0.5, tolls=0.25)
    assert result.mask.sum() == 1
    assert result.index_to_node[0] == 1
