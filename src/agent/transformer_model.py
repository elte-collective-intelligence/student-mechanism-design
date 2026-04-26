import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class TransformerModel(nn.Module):
    """
    Graph Transformer Agent for Scotland Yard pursuit-evasion game.

    Uses TransformerConv with scaled dot-product attention and
    optional Laplacian positional encodings.

    Architecture:
        Input projection → [TransformerConv + LayerNorm + Residual] x N → Output head

    Args:
        node_feature_size: Input node feature dimension
        hidden_dim: Hidden layer dimension (divided by num_heads per head)
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        edge_dim: Edge feature dimension (for transportation costs)
        use_positional_encoding: Whether to use Laplacian PE
        pe_dim: Positional encoding dimension
    """

    def __init__(
        self,
        node_feature_size,
        hidden_dim=128,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
        edge_dim=None,
        use_positional_encoding=False,
        pe_dim=8,
        **kwargs,
    ):
        super(TransformerModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_pe = use_positional_encoding
        self.pe_dim = pe_dim

        # Adjust input dimension if using positional encoding
        input_dim = node_feature_size + (pe_dim if use_positional_encoding else 0)

        # Input projection to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # TransformerConv: only first layer uses edge features
            self.layers.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,  # Per-head dimension
                    heads=num_heads,
                    concat=True,  # Concatenate heads: output = hidden_dim
                    dropout=dropout,
                    edge_dim=edge_dim if i == 0 else None,
                    root_weight=True,
                    beta=False,  # No gating mechanism
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Output layer for Q-values (single head, averaged)
        self.output_layer = TransformerConv(
            in_channels=hidden_dim,
            out_channels=1,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=None,
            root_weight=True,
        )

    def forward(self, data, return_attention=False):
        """
        Forward pass.

        Args:
            data: PyTorch Geometric Data object with x, edge_index, edge_attr
            return_attention: If True, return attention weights from each layer

        Returns:
            q_values: Shape [num_nodes] - Q-value per node
            all_attentions: (optional) List of (edge_index, alpha) tuples per layer
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        all_attentions = []

        # Optional: Add Laplacian positional encoding
        if self.use_pe and hasattr(data, "laplacian_eigenvector_pe"):
            pe = data.laplacian_eigenvector_pe[:, : self.pe_dim]
            x = torch.cat([x, pe], dim=-1)

        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # Transformer layers with residual connections
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # Store residual
            residual = x

            # Apply transformer (only first layer uses edge_attr)
            edge_attr_i = edge_attr if i == 0 else None

            if return_attention:
                x_new, (att_edge_index, alpha) = layer(
                    x, edge_index, edge_attr=edge_attr_i, return_attention_weights=True
                )
                all_attentions.append((att_edge_index, alpha))
            else:
                x_new = layer(x, edge_index, edge_attr=edge_attr_i)

            # LayerNorm + Residual
            x = norm(x_new + residual)

            # Activation and dropout (except last transformer layer)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer for Q-values
        if return_attention:
            out, (att_edge_index, alpha) = self.output_layer(
                x, edge_index, return_attention_weights=True
            )
            all_attentions.append((att_edge_index, alpha))
        else:
            out = self.output_layer(x, edge_index)

        # Squeeze to [num_nodes] to match GNNAgent expectation
        q_values = out.squeeze(-1)

        if return_attention:
            return q_values, all_attentions
        return q_values


def compute_laplacian_pe(edge_index, num_nodes, k=8):
    """
    Compute Laplacian eigenvector positional encoding.

    Args:
        edge_index: Edge connectivity [2, num_edges]
        num_nodes: Number of nodes
        k: Number of eigenvectors to use

    Returns:
        pe: Positional encoding [num_nodes, k]
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    # Build adjacency matrix
    edge_index_np = edge_index.cpu().numpy()
    row, col = edge_index_np[0], edge_index_np[1]

    # For undirected graph, ensure symmetry
    A = csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    A = A + A.T
    A.data = np.clip(A.data, 0, 1)  # Remove duplicate entries

    # Degree matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    D = csr_matrix(np.diag(degrees))

    # Laplacian: L = D - A
    L = D - A

    # Compute k smallest eigenvectors (skip the constant eigenvector with eigenvalue 0)
    try:
        eigenvalues, eigenvectors = eigsh(
            L.astype(np.float32),
            k=min(k + 1, num_nodes - 1),
            which="SM",
            return_eigenvectors=True,
        )
        # Remove the first eigenvector (constant)
        k_plus_1 = k + 1
        pe = torch.from_numpy(eigenvectors[:, 1:k_plus_1]).float()
    except Exception:
        # Fallback to zeros if eigendecomposition fails
        pe = torch.zeros(num_nodes, k)

    # Pad if we got fewer eigenvectors than requested
    if pe.shape[1] < k:
        padding = torch.zeros(num_nodes, k - pe.shape[1])
        pe = torch.cat([pe, padding], dim=1)

    return pe
