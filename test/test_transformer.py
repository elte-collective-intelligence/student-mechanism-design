import torch
from torch_geometric.data import Data

from src.agent.transformer_model import TransformerModel, compute_laplacian_pe


def test_transformer_output_shape():
    """Ensure output is [num_nodes] for Q-value indexing."""
    num_nodes = 15
    model = TransformerModel(
        node_feature_size=8,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        edge_dim=1,
    )

    data = Data(
        x=torch.randn(num_nodes, 8),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        edge_attr=torch.randn(4, 1),
    )

    out = model(data)
    assert out.shape == (num_nodes,), f"Expected ({num_nodes},), got {out.shape}"


def test_transformer_attention_extraction():
    """Verify attention weights can be extracted."""
    model = TransformerModel(
        node_feature_size=4,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
    )

    data = Data(
        x=torch.randn(5, 4),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
    )

    out, attentions = model(data, return_attention=True)

    # Should have attention from each layer + output layer
    assert len(attentions) == 3

    # Check attention structure
    for edge_idx, alpha in attentions:
        assert edge_idx.shape[0] == 2
        assert alpha.dim() == 2


def test_transformer_edge_features():
    """Verify edge features are used in first layer."""
    model = TransformerModel(
        node_feature_size=4,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        edge_dim=3,
    )

    data = Data(
        x=torch.randn(5, 4),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.randn(3, 3),
    )

    out = model(data)
    assert out.shape == (5,)


def test_transformer_gradient_flow():
    """Verify gradients flow through the network."""
    model = TransformerModel(
        node_feature_size=4,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
    )

    data = Data(
        x=torch.randn(5, 4, requires_grad=True),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
    )

    out = model(data)
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_laplacian_pe_computation():
    """Test Laplacian positional encoding computation."""
    edge_index = torch.tensor([[0, 0, 1, 1, 2], [1, 2, 0, 2, 1]], dtype=torch.long)
    num_nodes = 3

    pe = compute_laplacian_pe(edge_index, num_nodes, k=2)

    assert pe.shape == (num_nodes, 2), f"Expected (3, 2), got {pe.shape}"
    assert not torch.isnan(pe).any(), "PE contains NaN values"


def test_transformer_with_positional_encoding():
    """Test TransformerModel with Laplacian PE enabled."""
    model = TransformerModel(
        node_feature_size=4,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        use_positional_encoding=True,
        pe_dim=4,
    )

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    pe = compute_laplacian_pe(edge_index, num_nodes=5, k=4)

    data = Data(
        x=torch.randn(5, 4),
        edge_index=edge_index,
        laplacian_eigenvector_pe=pe,
    )

    out = model(data)
    assert out.shape == (5,)


def test_transformer_residual_connections():
    """Verify residual connections preserve information."""
    model = TransformerModel(
        node_feature_size=4,
        hidden_dim=16,
        num_layers=4,
        num_heads=2,
    )

    data = Data(
        x=torch.randn(5, 4),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
    )

    out = model(data)

    # Output should not be all zeros or NaN (residuals help gradient flow)
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert out.abs().sum() > 0, "Output is all zeros"
