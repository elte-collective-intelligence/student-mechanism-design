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


def test_gat_gradient_flow():
    """Verify gradients flow through the model."""
    model = GATModel(node_feature_size=4, hidden_dim=16, heads=2)
    data = Data(
        x=torch.randn(5, 4, requires_grad=True),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
    )

    out = model(data)
    loss = out.sum()
    loss.backward()

    assert data.x.grad is not None
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_gat_edge_attr_consistency():
    """Verify edge attributes are properly used when provided."""
    model = GATModel(node_feature_size=4, hidden_dim=16, heads=2, edge_dim=3)

    # With edge attributes
    data_with_edge = Data(
        x=torch.randn(4, 4),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        edge_attr=torch.randn(3, 3),
    )

    # Without edge attributes
    data_without_edge = Data(
        x=data_with_edge.x.clone(),
        edge_index=data_with_edge.edge_index.clone(),
    )

    out_with = model(data_with_edge)
    out_without = model(data_without_edge)

    # Outputs should differ when edge attrs are provided
    assert out_with.shape == out_without.shape
    assert not torch.allclose(out_with, out_without)
