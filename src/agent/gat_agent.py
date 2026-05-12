import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from torch_geometric.nn import GATConv
from torch_geometric.data import Batch

try:
    from .base_agent import BaseAgent
except ImportError:
    from agent.base_agent import BaseAgent


class GATAgent(BaseAgent):
    def __init__(
        self,
        node_feature_size,
        hidden_dim=32,
        heads=8,
        num_layers=2,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=10000,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        device=torch.device("cpu"),
    ):
        self.node_feature_size = node_feature_size
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.num_layers = num_layers
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = device

        self.memory = deque(maxlen=buffer_size)

        self.model = GATModel(
            node_feature_size=self.node_feature_size,
            hidden_channels=self.hidden_dim,
            heads=self.heads,
            num_layers=self.num_layers,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def select_action(self, graph, action_mask):
        self.model.eval()
        with torch.no_grad():
            graph = graph.to(self.device)
            q_values = self.model(graph)
            q_values = q_values.cpu().numpy()

        if action_mask.size(0) != graph.num_nodes:
            raise ValueError(
                f"action_mask length ({action_mask.size(0)}) does not match "
                f"number of nodes in graph ({graph.num_nodes})."
            )

        valid_actions = np.where(action_mask.cpu().numpy() == 1)[0]
        if len(valid_actions) == 0:
            return None

        if np.random.rand() <= self.epsilon:
            selected_action = np.random.choice(valid_actions)
        else:
            valid_q_values = q_values[valid_actions]
            selected_action = valid_actions[np.argmax(valid_q_values)]

        num_nodes = graph.num_nodes
        if not (0 <= selected_action < num_nodes):
            raise ValueError(
                f"Selected action {selected_action} is invalid for graph with {num_nodes} nodes."
            )

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

        graphs = [g.to(self.device) for g in graphs]
        next_graphs = [ng.to(self.device) for ng in next_graphs]
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        for graph, action, reward, next_graph, done in zip(
            graphs, actions, rewards, next_graphs, dones
        ):
            if action >= graph.num_nodes:
                print(
                    f"Attempting to store invalid action: {action.item()} for graph with {graph.num_nodes} nodes. Skipping."
                )
                continue
            self.memory.append((graph, action, reward, next_graph, done))

        if len(self.memory) < self.batch_size:
            return

        mini_batch = random.sample(self.memory, self.batch_size)
        batch_graphs, batch_actions, batch_rewards, batch_next_graphs, batch_dones = (
            zip(*mini_batch)
        )

        batch_actions = torch.stack(batch_actions)
        batch_rewards = torch.stack(batch_rewards)
        batch_dones = torch.stack(batch_dones)

        batch_graph_num_nodes = torch.tensor(
            [g.num_nodes for g in batch_graphs], device=self.device
        )
        if not torch.all(batch_actions < batch_graph_num_nodes):
            invalid_indices = (batch_actions >= batch_graph_num_nodes).nonzero(
                as_tuple=True
            )[0]
            for idx in invalid_indices:
                print(
                    f"Invalid action: {batch_actions[idx].item()} for graph with {batch_graph_num_nodes[idx].item()} nodes."
                )
            raise ValueError(
                "Some actions exceed the number of nodes in their respective graphs."
            )

        batch_graph = Batch.from_data_list(batch_graphs).to(self.device)
        next_batch_graph = Batch.from_data_list(batch_next_graphs).to(self.device)

        self.model.train()
        q_values = self.model(batch_graph)
        node_indices = batch_graph.ptr[:-1] + batch_actions
        assert torch.all(
            node_indices < q_values.size(0)
        ), "node_indices exceed q_values size."
        current_q_values = q_values[node_indices]

        with torch.no_grad():
            next_q_all = self.model(next_batch_graph)
            max_next_q = []
            for i in range(self.batch_size):
                start = next_batch_graph.ptr[i]
                end = next_batch_graph.ptr[i + 1]
                max_next_q.append(next_q_all[start:end].max())
            max_next_q = torch.stack(max_next_q).to(self.device)

        target_q_values = batch_rewards + (self.gamma * max_next_q * (1 - batch_dones))

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_attention(self, graph):
        self.model.eval()
        with torch.no_grad():
            graph = graph.to(self.device)
            q, attention = self.model(graph, return_attention=True)
        return q, attention

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)


class GATModel(nn.Module):
    def __init__(self, node_feature_size, hidden_channels=32, heads=8, num_layers=2):
        super(GATModel, self).__init__()

        if num_layers < 2 or num_layers > 5:
            raise ValueError("Number of layers must be bewteen 2 and 5.")

        self.node_feature_size = node_feature_size
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers

        self.convs = nn.ModuleList()

        self.convs.append(
            GATConv(self.node_feature_size, self.hidden_channels, heads=self.heads)
        )

        for _ in range(self.num_layers - 2):
            self.convs.append(
                GATConv(
                    self.hidden_channels * self.heads,
                    self.hidden_channels,
                    heads=self.heads,
                )
            )

        self.convs.append(
            GATConv(
                self.hidden_channels * self.heads,
                self.hidden_channels,
                heads=1,
                concat=False,
            )
        )

        self.output_layer = nn.Linear(self.hidden_channels, 1)

    def forward(self, data, return_attention=False):
        x, edge_index = data.x, data.edge_index

        attention = [] if return_attention else None

        for conv in self.convs[:-1]:
            if return_attention:
                x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
                attention.append((ei.detach(), alpha.detach()))
            else:
                x = conv(x, edge_index)
            x = torch.nn.functional.leaky_relu(x)

        if return_attention:
            x, (ei, alpha) = self.convs[-1](
                x, edge_index, return_attention_weights=True
            )
            attention.append((ei.detach(), alpha.detach()))
        else:
            x = self.convs[-1](x, edge_index)
        x = torch.nn.functional.leaky_relu(x)

        x = self.output_layer(x)
        q = x.squeeze(-1)
        if return_attention:
            return q, attention
        return q
