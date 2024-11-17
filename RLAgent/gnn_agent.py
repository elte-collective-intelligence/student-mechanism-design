import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from torch_geometric.nn import AntiSymmetricConv
from torch_geometric.data import Data, Batch
from itertools import chain  # For flattening lists

class GNNAgent:
    def __init__(
        self,
        node_feature_size,
        hidden_size=64,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=10000,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        device=torch.device('cpu')  # Default device
    ):
        """
        Graph Neural Network Agent with Experience Replay and Epsilon-Greedy policy.
        Now supports GPU by accepting a device parameter.
        """
        self.node_feature_size = node_feature_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = device  # Store device

        # Experience replay buffer
        self.memory = deque(maxlen=buffer_size)

        # GNN Model
        self.model = GNNModel(node_feature_size).to(self.device)  # Move model to device

        # Optimizer and Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def select_action(self, graph, action_mask):
        """
        Selects an action using epsilon-greedy policy.
        Ensures that all inputs are on the correct device.
        """
        self.model.eval()
        with torch.no_grad():
            graph = graph.to(self.device)  # Move graph to device
            q_values = self.model(graph)  # Shape: [num_nodes]
            q_values = q_values.cpu().numpy()  # Move to CPU for numpy operations

        valid_actions = np.where(action_mask.cpu().numpy() == 1)[0]  # Ensure action_mask is on CPU
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available.")

        if np.random.rand() <= self.epsilon:
            # Explore
            selected_action = np.random.choice(valid_actions)
        else:
            # Exploit
            valid_q_values = q_values[valid_actions]
            selected_action = valid_actions[np.argmax(valid_q_values)]
        return selected_action

    def update(self, graphs, actions, rewards, next_graphs, dones):
        """
        Stores experience in replay memory and updates the GNN model.
        Ensures that all tensors are on the correct device.
        Automatically wraps single experiences in lists if necessary.
        """
        # Ensure all inputs are lists
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

        # Move all graphs to device
        graphs = [g.to(self.device) for g in graphs]
        next_graphs = [ng.to(self.device) for ng in next_graphs]
        actions = torch.LongTensor(actions).to(self.device)          # Shape: [batch_size]
        rewards = torch.FloatTensor(rewards).to(self.device)        # Shape: [batch_size]
        dones = torch.FloatTensor(dones).to(self.device)            # Shape: [batch_size]

        # Add experience to memory
        self.memory.append((graphs, actions, rewards, next_graphs, dones))

        # Start learning if memory has enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample a mini-batch of experiences from memory
        mini_batch = random.sample(self.memory, self.batch_size)
        batch_graphs, batch_actions, batch_rewards, batch_next_graphs, batch_dones = zip(*mini_batch)

        # Flatten the lists of graphs
        batch_graphs = list(chain.from_iterable(batch_graphs))           # Flat list of Data objects
        batch_next_graphs = list(chain.from_iterable(batch_next_graphs)) # Flat list of Data objects

        # Concatenate actions, rewards, and dones
        batch_actions = torch.cat(batch_actions)  # Shape: [batch_size]
        batch_rewards = torch.cat(batch_rewards)  # Shape: [batch_size]
        batch_dones = torch.cat(batch_dones)      # Shape: [batch_size]

        # Batch the graphs using PyTorch Geometric's Batch
        batch_graph = Batch.from_data_list(batch_graphs).to(self.device)
        next_batch_graph = Batch.from_data_list(batch_next_graphs).to(self.device)

        # Forward pass for current states
        q_values = self.model(batch_graph)          # Shape: [total_nodes_in_batch]
        
        # Map actions to global node indices in the batch
        # batch_graph.ptr is a tensor indicating the start of each graph in the batch
        node_indices = batch_graph.ptr[:-1] + batch_actions  # Shape: [batch_size]
        current_q_values = q_values[node_indices]            # Shape: [batch_size]

        # Forward pass for next states
        with torch.no_grad():
            next_q_values = self.model(next_batch_graph)     # Shape: [total_nodes_in_batch]
            max_next_q_values = []
            for i in range(len(batch_next_graphs)):
                node_start = next_batch_graph.ptr[i]
                node_end = next_batch_graph.ptr[i + 1]
                graph_q_values = next_q_values[node_start:node_end]
                max_q_value = graph_q_values.max()
                max_next_q_values.append(max_q_value)
            max_next_q_values = torch.stack(max_next_q_values).to(self.device)  # Shape: [batch_size]

        # Compute target Q-values
        target_q_values = batch_rewards + self.gamma * max_next_q_values * (1 - batch_dones)  # Shape: [batch_size]

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        """
        Saves the model parameters to the specified file.
        """
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        """
        Loads the model parameters from the specified file.
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)  # Ensure the model is on the correct device

class GNNModel(nn.Module):
    def __init__(self, node_feature_size):
        super(GNNModel, self).__init__()
        self.conv1 = AntiSymmetricConv(
            in_channels=node_feature_size,
            num_iters=1,
            epsilon=0.1,
            gamma=0.1,
            act="tanh"
        )
        self.conv2 = AntiSymmetricConv(
            in_channels=node_feature_size,
            num_iters=1,
            epsilon=0.1,
            gamma=0.1,
            act="tanh"
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
        return x.squeeze(-1)      # Shape: [num_nodes]