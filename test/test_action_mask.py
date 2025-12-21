import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from Enviroment.action_mask import compute_action_mask  # noqa: E402


def test_action_mask_respects_budget_and_mapping():
    """Test that action mask correctly respects budget constraints and uses fixed index→node mapping."""
    adjacency = np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ]
    )
    # Edge weights: node 0->1 costs 2, node 0->2 costs 4
    edge_weights = np.array(
        [
            [0, 2, 4],
            [2, 0, 0],
            [4, 0, 0],
        ]
    )
    result = compute_action_mask(
        adjacency, current_node=0, budget=3, edge_weights=edge_weights
    )

    # Only node 1 should be affordable (cost 2 <= budget 3)
    # Node 2 costs 4 > budget 3, so not affordable
    assert result.mask[1] == True, "Node 1 should be in mask (cost 2 <= budget 3)"
    assert result.mask[2] == False, "Node 2 should NOT be in mask (cost 4 > budget 3)"
    assert result.mask[0] == False, "Current node 0 should NOT be in mask"

    # Check valid_actions list
    assert 1 in result.valid_actions, "Node 1 should be in valid_actions"
    assert 2 not in result.valid_actions, "Node 2 should NOT be in valid_actions"

    # Check fixed index→node mapping (identity mapping)
    assert result.index_to_node[0] == 0, "index_to_node should be identity mapping"
    assert result.index_to_node[1] == 1, "index_to_node should be identity mapping"
    assert result.index_to_node[2] == 2, "index_to_node should be identity mapping"


def test_action_mask_with_scalar_toll():
    """Test action mask with scalar toll added to edge weights."""
    adjacency = np.ones((2, 2)) - np.eye(2)
    # With toll of 0.25 and default edge weight of 1, total cost is 1.25
    # Budget of 0.5 is not enough
    result = compute_action_mask(adjacency, current_node=0, budget=0.5, tolls=0.25)
    assert (
        result.mask.sum() == 0
    ), "No moves should be affordable with budget 0.5 and cost 1.25"

    # With budget of 1.5, move should be affordable
    result = compute_action_mask(adjacency, current_node=0, budget=1.5, tolls=0.25)
    assert result.mask[1] == True, "Node 1 should be affordable with budget 1.5"
    assert 1 in result.valid_actions


def test_action_mask_fixed_index_node_mapping():
    """Test that index→node mapping is always the identity mapping."""
    adjacency = np.array(
        [
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
    )

    result = compute_action_mask(adjacency, current_node=0, budget=100)

    # index_to_node should always be identity
    for i in range(4):
        assert result.index_to_node[i] == i, f"index_to_node[{i}] should be {i}"
        assert result.node_to_index[i] == i, f"node_to_index[{i}] should be {i}"

    # mask[node] should be True for valid neighbors
    assert result.mask[0] is False  # Current node
    assert result.mask[1] is True
    assert result.mask[2] is True
    assert result.mask[3] is True


def test_action_mask_no_valid_moves():
    """Test action mask when no moves are affordable."""
    adjacency = np.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    edge_weights = np.array(
        [
            [0, 100],
            [100, 0],
        ]
    )

    result = compute_action_mask(
        adjacency, current_node=0, budget=1, edge_weights=edge_weights
    )

    assert result.mask.sum() == 0, "No moves should be affordable"
    assert len(result.valid_actions) == 0, "valid_actions should be empty"
    assert result.num_valid_actions == 0


def test_action_mask_isolated_node():
    """Test action mask for a node with no neighbors."""
    adjacency = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ]
    )

    result = compute_action_mask(adjacency, current_node=0, budget=100)

    assert result.mask.sum() == 0, "Isolated node should have no valid moves"
    assert len(result.valid_actions) == 0
