import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATModel(nn.Module):
    def __init__(
        self,
        node_feature_size,
        hidden_dim=128,
        num_layers=3,
        heads=4,
        dropout=0.2,
        edge_dim=None,
        **kwargs,
    ):
        super(GATModel, self).__init__()
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(
            GATv2Conv(
                node_feature_size,
                hidden_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GATv2Conv(
                    hidden_dim * heads,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )

        # Final layer (Consensus averaging)
        self.layers.append(
            GATv2Conv(
                hidden_dim * heads,
                1,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,
            )
        )

    def forward(self, data, return_attention=False):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        # Ensure edge_attr has the correct shape [num_edges, edge_dim]
        if edge_attr is not None and getattr(self, "edge_dim", None) is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)

        all_attentions = []

        for i, layer in enumerate(self.layers):
            if return_attention:
                x, (att_edge_index, alpha) = layer(
                    x, edge_index, edge_attr=edge_attr, return_attention_weights=True
                )
                all_attentions.append((att_edge_index, alpha))
            else:
                x = layer(x, edge_index, edge_attr=edge_attr)

            if i < len(self.layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Squeeze to [num_nodes] to match GNNAgent expectation
        out = x.squeeze(-1)

        if return_attention:
            return out, all_attentions
        return out
