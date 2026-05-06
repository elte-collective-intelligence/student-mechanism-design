import os
import sys
import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from agent.gat_agent import GATAgent  # noqa: E402
from agent.transformer_agent import TransformerAgent  # noqa: E402


@pytest.fixture
def dummy_graph():
    """Creates a small dummy graph for testing."""
    node_features = 16  # Matches typical setup
    num_nodes = 5

    # 5 nodes with random features
    x = torch.randn((num_nodes, node_features))

    # Simple edge index (line graph: 0-1-2-3-4)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
    )

    return Data(x=x, edge_index=edge_index)


@pytest.mark.parametrize("agent_class", [GATAgent, TransformerAgent])
def test_agent_full_cycle(agent_class, dummy_graph):
    """
    Tests initialization, action selection, and the update mechanism
    to ensure maximum code coverage.
    """
    node_feature_size = 16
    device = torch.device("cpu")

    # 1. Initialization
    agent = agent_class(
        node_feature_size=node_feature_size,
        batch_size=2,  # Small batch for quick testing
        device=device,
    )

    # 2. Test select_action (Forward Pass)
    # Mask: only nodes 0 and 1 are valid moves
    action_mask = torch.tensor([1, 1, 0, 0, 0], dtype=torch.int32)
    action = agent.select_action(dummy_graph, action_mask)

    assert action is not None
    assert action in [0, 1], f"Agent picked action {action} not in mask"
    assert isinstance(action, int)

    # 3. Test update (Training Pass)
    # We need to fill the buffer to trigger the actual training logic
    for _ in range(3):  # buffer size is larger than batch_size
        agent.update([dummy_graph], [action], [1.0], [dummy_graph], [False])

    # Check if epsilon decayed (meaning the optimizer step was reached)
    assert agent.epsilon < 1.0 or agent.epsilon == agent.epsilon_min


def test_transformer_specific_pe(dummy_graph):
    """Specifically tests the Positional Encoding logic in the Transformer."""
    agent = TransformerAgent(node_feature_size=16, pos_dim=4)

    # Test the internal PE computation
    pe = agent.compute_pos_enc(dummy_graph)
    assert pe.x_pe.shape == (5, 4), "Laplacian PE shape is incorrect"
    assert not torch.isnan(pe).any(), "PE contains NaNs"
