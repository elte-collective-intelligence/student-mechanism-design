import torch.optim as optim
import random
import numpy as np
from collections import deque
from torch_geometric.data import Batch
from torch_geometric.utils import get_laplacian, to_dense_adj
from base_agent import BaseAgent
import torch
import torch.nn as nn

from torch_geometric.nn import TransformerConv


class TransformerAgent(BaseAgent):
    def __init__(
        self,
        node_feature_size,
        pos_dim=8,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=10000,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        device=torch.device("cpu"),
    ):
        self.device = device
        self.pos_dim = pos_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.memory = deque(maxlen=buffer_size)

        self.model = GraphTransformerModel(node_feature_size, pos_edge_dim=pos_dim).to(
            self.device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def compute_pos_enc(self, data):
        if hasattr(data, "pos_enc"):
            return data

        edge_index = data.edge_index
        num_nodes = data.num_nodes

        L_index, L_weight = get_laplacian(
            edge_index, num_nodes=num_nodes, normalization="sym"
        )
        L_dense = to_dense_adj(L_index, edge_attr=L_weight).squeeze(0).cpu().numpy()

        eigvals, eigvecs = np.linalg.eigh(L_dense)
        idx = eigvals.argsort()
        eigvecs = eigvecs[:, idx]

        pos_enc = (
            torch.from_numpy(eigvecs[:, 1: self.pos_dim + 1]).float().to(self.device)
        )

        if pos_enc.shape[1] < self.pos_dim:
            pad = torch.zeros((num_nodes, self.pos_dim - pos_enc.shape[1])).to(
                self.device
            )
            pos_enc = torch.cat([pos_enc, pad], dim=1)

        data.pos_enc = pos_enc
        return data

    def select_action(self, graph, action_mask):
        self.model.eval()
        with torch.no_grad():
            graph = self.compute_pos_enc(graph).to(self.device)
            q_values = self.model(graph).cpu().numpy()

        mask_np = (
            action_mask.cpu().numpy() if torch.is_tensor(action_mask) else action_mask
        )
        valid_actions = np.where(mask_np == 1)[0]

        if len(valid_actions) == 0:
            return None

        if np.random.rand() <= self.epsilon:
            return int(np.random.choice(valid_actions))
        else:
            return int(valid_actions[np.argmax(q_values[valid_actions])])

    def update(self, graphs, actions, rewards, next_graphs, dones):
        if actions is None:
            return

        if not isinstance(graphs, (list, tuple)):
            graphs = [graphs]
        if not isinstance(next_graphs, (list, tuple)):
            next_graphs = [next_graphs]

        for g, a, r, ng, d in zip(graphs, actions, rewards, next_graphs, dones):
            g_pe = self.compute_pos_enc(g).to(self.device)
            ng_pe = self.compute_pos_enc(ng).to(self.device)

            self.memory.append(
                (
                    g_pe,
                    torch.tensor(a, dtype=torch.long, device=self.device),
                    torch.tensor(r, dtype=torch.float, device=self.device),
                    ng_pe,
                    torch.tensor(d, dtype=torch.float, device=self.device),
                )
            )

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        b_graphs, b_actions, b_rewards, b_next_graphs, b_dones = zip(*batch)

        bg = Batch.from_data_list(b_graphs).to(self.device)
        nbg = Batch.from_data_list(b_next_graphs).to(self.device)

        ba = torch.stack(b_actions)
        br = torch.stack(b_rewards)
        bd = torch.stack(b_dones)

        self.model.train()
        q_out = self.model(bg)

        current_indices = bg.ptr[:-1] + ba
        current_q = q_out[current_indices]

        with torch.no_grad():
            next_q_out = self.model(nbg)
            max_next_q = []
            for i in range(self.batch_size):
                start, end = nbg.ptr[i], nbg.ptr[i + 1]
                max_next_q.append(next_q_out[start:end].max())
            max_next_q = torch.stack(max_next_q)

        target_q = br + (self.gamma * max_next_q * (1 - bd))

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))

    def state_dict(self):
        """
        Returns the model parameters.
        """
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """
        Loads the model parameters.
        """
        self.model.load_state_dict(state_dict, strict=strict)


class GraphTransformerModel(nn.Module):
    def __init__(self, node_feature_size, pos_edge_dim=8, hidden_channels=64, heads=4):
        super(GraphTransformerModel, self).__init__()

        self.node_emb = nn.Linear(node_feature_size + pos_edge_dim, hidden_channels)

        self.trans1 = TransformerConv(
            hidden_channels, hidden_channels, heads=heads, dropout=0.1
        )
        self.trans2 = TransformerConv(
            hidden_channels * heads, hidden_channels, heads=heads, dropout=0.1
        )
        self.trans3 = TransformerConv(
            hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.1
        )

        self.output_layer = nn.Linear(hidden_channels, 1)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if hasattr(data, "pos_enc"):
            x = torch.cat([x, data.pos_enc], dim=-1)

        x = self.node_emb(x)
        x = self.leaky_relu(x)

        x = self.trans1(x, edge_index)
        x = self.leaky_relu(x)

        x = self.trans2(x, edge_index)
        x = self.leaky_relu(x)

        x = self.trans3(x, edge_index)
        x = self.leaky_relu(x)

        x = self.output_layer(x)
        return x.squeeze(-1)
