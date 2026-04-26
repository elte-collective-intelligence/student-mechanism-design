import torch
import torch.nn as nn
from torch_geometric.nn import AntiSymmetricConv


class GNNModel(nn.Module):
    def __init__(self, node_feature_size):
        super(GNNModel, self).__init__()
        self.conv1 = AntiSymmetricConv(
            in_channels=node_feature_size,
            num_iters=1,
            epsilon=0.1,
            gamma=0.1,
            act="tanh",
        )
        self.conv2 = AntiSymmetricConv(
            in_channels=node_feature_size,
            num_iters=1,
            epsilon=0.1,
            gamma=0.1,
            act="tanh",
        )
        # Output layer to get scalar Q-value per node
        self.output_layer = nn.Linear(node_feature_size, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.output_layer(x)  # Shape: [num_nodes, 1]
        return x.squeeze(-1)  # Shape: [num_nodes]
