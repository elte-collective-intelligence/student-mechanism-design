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


@pytest.mark.parametrize("agent_class", [GATAgent, TransformerAgent])
def test_agent_returns_attention(agent_class, dummy_graph):
    """Tests that get_attention returns per-layer (edge_index, alpha)."""
    num_layers = 3
    heads = 4
    agent = agent_class(
        node_feature_size=16,
        num_layers=num_layers,
        heads=heads,
        device=torch.device("cpu"),
    )

    q, attention = agent.get_attention(dummy_graph)

    assert q.shape == (dummy_graph.num_nodes,)
    assert len(attention) == num_layers

    for layer_idx, (ei, alpha) in enumerate(attention):
        assert ei.dim() == 2 and ei.shape[0] == 2
        assert alpha.dim() == 2
        assert alpha.shape[0] == ei.shape[1]
        expected_heads = 1 if layer_idx == num_layers - 1 else heads
        assert alpha.shape[1] == expected_heads
        assert not torch.isnan(alpha).any()


def test_transformer_specific_pe(dummy_graph):
    """Specifically tests the Positional Encoding logic in the Transformer."""
    agent = TransformerAgent(node_feature_size=16, pos_dim=4)

    # Run the computation
    graph_with_pe = agent.compute_pos_enc(dummy_graph)

    # Match the attribute name in your compute_pos_enc code: .pos_enc
    assert hasattr(graph_with_pe, "pos_enc"), "Graph is missing pos_enc attribute"
    assert graph_with_pe.pos_enc.shape == (5, 4), "Laplacian PE shape is incorrect"
    assert not torch.isnan(graph_with_pe.pos_enc).any(), "PE contains NaNs"
