import torch
from src.agent.gat_model import GATModel
from torch_geometric.data import Data


def test_gat_output_shape():
    """Ensure output is [num_nodes] for Q-value indexing."""
    num_nodes = 10
    model = GATModel(node_feature_size=8, hidden_dim=16, edge_dim=3)
    data = Data(
        x=torch.randn(num_nodes, 8),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.randn(2, 3),
    )

    out = model(data)
    assert out.shape == (num_nodes,)


def test_gat_attention_integrity():
    """Verify attention weights match edge structure."""
    model = GATModel(node_feature_size=4, heads=2)
    data = Data(
        x=torch.randn(5, 4),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
    )

    out, atts = model(data, return_attention=True)
    # weights shape: [num_edges_with_self_loops, heads]
    _, weights = atts[0]
    assert weights.shape[1] == 2  # heads
    assert weights.shape[0] >= 3  # Original edges


def test_gat_parameter_matching():
    """Verify that multi-head hidden layers scale correctly."""
    hidden = 32
    heads = 4
    model = GATModel(node_feature_size=8, hidden_dim=hidden, heads=heads, num_layers=2)

    # Check that parameters exist and have correct dimensionality
    # GATv2Conv internal parameters are usually named lin_l, lin_r, etc.
    final_params = [p for n, p in model.layers[-1].named_parameters() if "lin" in n]
    # The input dimension to the final layer should be hidden * heads
    assert any(p.shape[-1] == hidden * heads for p in final_params)
