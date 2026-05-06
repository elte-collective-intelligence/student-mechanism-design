import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from torch_geometric.nn import GATConv
from torch_geometric.data import Batch
from base_agent import BaseAgent


class GATAgent(BaseAgent):
    def __init__(
        self,
        node_feature_size,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=10000,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        device=torch.device("cpu"),
    ):
        """
        Graph Attention Network (GAT) Agent with Experience Replay.
        """
        self.node_feature_size = node_feature_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = device

        self.memory = deque(maxlen=buffer_size)

        self.model = GATModel(node_feature_size).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def select_action(self, graph, action_mask):
        """Selects action using epsilon-greedy policy with attention-based Q-values."""

        self.model.eval()
        with torch.no_grad():
            graph = graph.to(self.device)
            q_values = self.model(graph)
            q_values = q_values.cpu().numpy()

        mask_np = (
            action_mask.cpu().numpy() if torch.is_tensor(action_mask) else action_mask
        )
        valid_actions = np.where(mask_np == 1)[0]

        if len(valid_actions) == 0:
            return None

        if np.random.rand() <= self.epsilon:
            selected_action = np.random.choice(valid_actions)
        else:
            valid_q_values = q_values[valid_actions]
            selected_action = valid_actions[np.argmax(valid_q_values)]

        return int(selected_action)

    def update(self, graphs, actions, rewards, next_graphs, dones):
        if actions is None:
            return

        if not isinstance(graphs, (list, tuple)):
            graphs = [graphs]
        if not isinstance(actions, (list, tuple)):
            actions = [actions]
        if not isinstance(rewards, (list, tuple)):
            rewards = [rewards]
        if not isinstance(next_graphs, (list, tuple)):
            next_graphs = [next_graphs]
        if not isinstance(dones, (list, tuple)):
            dones = [dones]

        for g, a, r, ng, d in zip(graphs, actions, rewards, next_graphs, dones):
            r_t = torch.tensor(r, dtype=torch.float) if not torch.is_tensor(r) else r
            d_t = torch.tensor(d, dtype=torch.float) if not torch.is_tensor(d) else d
            a_t = torch.tensor(a, dtype=torch.long) if not torch.is_tensor(a) else a

            self.memory.append(
                (
                    g.to(self.device),
                    a_t.to(self.device),
                    r_t.to(self.device),
                    ng.to(self.device),
                    d_t.to(self.device),
                )
            )

        if len(self.memory) < self.batch_size:
            return

        mini_batch = random.sample(self.memory, self.batch_size)
        b_graphs, b_actions, b_rewards, b_next_graphs, b_dones = zip(*mini_batch)

        batch_graph = Batch.from_data_list(b_graphs).to(self.device)
        next_batch_graph = Batch.from_data_list(b_next_graphs).to(self.device)

        b_actions = torch.stack(b_actions)
        b_rewards = torch.stack(b_rewards)
        b_dones = torch.stack(b_dones)

        self.model.train()
        q_values = self.model(batch_graph)
        node_indices = batch_graph.ptr[:-1] + b_actions
        current_q_values = q_values[node_indices]

        with torch.no_grad():
            next_q_all = self.model(next_batch_graph)
            max_next_q = []
            for i in range(len(b_next_graphs)):
                start, end = next_batch_graph.ptr[i], next_batch_graph.ptr[i + 1]
                max_next_q.append(next_q_all[start:end].max())
            max_next_q = torch.stack(max_next_q)

        target_q_values = b_rewards + (self.gamma * max_next_q * (1 - b_dones))

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)

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


class GATModel(nn.Module):
    def __init__(self, node_feature_size, hidden_channels=32, heads=8):
        super(GATModel, self).__init__()

        self.conv1 = GATConv(node_feature_size, hidden_channels, heads=heads)

        self.conv2 = GATConv(
            hidden_channels * heads, hidden_channels, heads=1, concat=False
        )

        self.output_layer = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = torch.leaky_relu(x)

        x = self.output_layer(x)
        return x.squeeze(-1)
