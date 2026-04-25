import torch
import torch.nn as nn
from src.agent.gnn_agent import GNNModel
from torch_geometric.data import Data


def test_gnn_baseline_output():
    """Verify that the original GNNModel returns scalar Q-values per node."""
    num_nodes = 20
    node_feat_size = 8
    model = GNNModel(node_feature_size=node_feat_size)
    
    data = Data(x=torch.randn(num_nodes, node_feat_size),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    
    out = model(data)
    assert out.shape == (num_nodes,)
    assert not torch.isnan(out).any()

def test_gnn_baseline_permutation_invariance():
    """Verify that GNNModel is invariant to node ordering (Permutation Invariance)."""
    node_feat_size = 4
    model = GNNModel(node_feature_size=node_feat_size)
    model.eval() # Fix weights
    
    # Original Graph
    x = torch.randn(3, node_feat_size)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    
    with torch.no_grad():
        out_orig = model(data)
    
    # Permuted Graph (Swap node 0 and 2)
    p = [2, 1, 0]
    x_p = x[p]
    # Update edge_index for permutation: 0->2, 1->1, 2->0
    # Original: 0-1, 1-2
    # Permuted: 2-1, 1-0
    edge_index_p = torch.tensor([[2, 1], [1, 0]], dtype=torch.long)
    data_p = Data(x=x_p, edge_index=edge_index_p)
    
    with torch.no_grad():
        out_p = model(data_p)
    
    # Output for node 0 (orig) should match node 2 (permuted)
    assert torch.allclose(out_orig[0], out_p[2], atol=1e-5)
    assert torch.allclose(out_orig[1], out_p[1], atol=1e-5)

def test_gnn_baseline_gradient_flow():
    """Verify that parameters are updateable (no vanishing gradients)."""
    model = GNNModel(node_feature_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    data = Data(x=torch.randn(5, 4),
                edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long))
    
    # Forward & Backward pass
    out = model(data)
    loss = out.mean()
    loss.backward()
    
    # Check if any gradients exist and are non-zero
    has_gradients = any(p.grad is not None and torch.abs(p.grad).sum() > 0 for p in model.parameters())
    assert has_gradients, "No gradients detected in the model parameters."
    
    # Perform optimization step and ensure parameters actually change
    initial_params = [p.clone() for p in model.parameters()]
    optimizer.step()
    
    param_changed = any(not torch.allclose(p_init, p_curr) for p_init, p_curr in zip(initial_params, model.parameters()))
    assert param_changed, "Model parameters did not update after optimizer step."
