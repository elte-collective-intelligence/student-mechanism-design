import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from torch_geometric.data import Batch
from torch_geometric.utils import scatter
from .gat_model import GATModel
from .transformer_model import TransformerModel
from .gnn_model import GNNModel


class GraphDQNAgent:
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
        device=torch.device("cpu"),  # Default device
        agent_type="gnn",
        model_kwargs=None,
    ):
        """
        Graph Neural Network Agent with Experience Replay and Epsilon-Greedy policy.
        """
        self.node_feature_size = node_feature_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = device  # Store device
        self.agent_type = agent_type

        # Experience replay buffer
        self.memory = deque(maxlen=buffer_size)

        if self.agent_type == "gat":
            self.model = GATModel(node_feature_size, **(model_kwargs or {})).to(
                self.device
            )
        elif self.agent_type == "transformer":
            self.model = TransformerModel(node_feature_size, **(model_kwargs or {})).to(
                self.device
            )
        elif self.agent_type == "gnn":
            self.model = GNNModel(node_feature_size).to(self.device)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

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
            # Support attention extraction for logging/viz
            if self.agent_type in ["gat", "transformer"]:
                q_values, self.last_attention = self.model(graph, return_attention=True)
            else:
                q_values = self.model(graph)

        action_mask = action_mask.to(self.device)

        if action_mask.size(0) != graph.num_nodes:
            raise ValueError(
                f"action_mask length ({action_mask.size(0)}) does not match "
                f"number of nodes in graph ({graph.num_nodes})."
            )

        valid_actions = torch.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            return None

        if torch.rand(1, device=self.device).item() <= self.epsilon:
            # Explore: random valid action
            idx = torch.randint(len(valid_actions), (1,), device=self.device)
            selected_action = valid_actions[idx].item()
        else:
            # Exploit: best Q-value among valid actions
            valid_q_values = q_values[valid_actions]
            best_idx = torch.argmax(valid_q_values)
            selected_action = valid_actions[best_idx].item()

        num_nodes = graph.num_nodes
        if not (0 <= selected_action < num_nodes):
            raise ValueError(
                f"Selected action {selected_action} is invalid for graph with {num_nodes} nodes."
            )

        return selected_action

    def update(self, graphs, actions, rewards, next_graphs, dones):
        """
        Stores individual experiences in replay memory and updates the GNN model.
        Ensures that all tensors are on the correct device.
        """
        if actions is None:
            return
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

        # Move all graphs to CPU for storage (saves GPU memory in replay buffer)
        graphs_cpu = [g.cpu() for g in graphs]
        next_graphs_cpu = [ng.cpu() for ng in next_graphs]
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Store each experience individually (on CPU)
        for graph, action, reward, next_graph, done in zip(
            graphs_cpu, actions, rewards, next_graphs_cpu, dones
        ):
            if action >= graph.num_nodes:
                print(
                    f"Attempting to store invalid action: {action.item()} for graph with {graph.num_nodes} nodes. Skipping."
                )
                continue  # Skip storing this invalid experience
            self.memory.append((graph, action, reward, next_graph, done))

        # Start learning if memory has enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample a mini-batch of experiences from memory
        mini_batch = random.sample(self.memory, self.batch_size)
        batch_graphs, batch_actions, batch_rewards, batch_next_graphs, batch_dones = (
            zip(*mini_batch)
        )

        # Move actions, rewards, and dones to device for training
        batch_actions = torch.stack(batch_actions).to(self.device)
        batch_rewards = torch.stack(batch_rewards).to(self.device)
        batch_dones = torch.stack(batch_dones).to(self.device)

        # Validate actions against their respective graph sizes
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

        # Batch the graphs using PyTorch Geometric's Batch
        batch_graph = Batch.from_data_list(batch_graphs).to(self.device)
        next_batch_graph = Batch.from_data_list(batch_next_graphs).to(self.device)

        # Forward pass for current states
        q_values = self.model(batch_graph)

        # Map actions to global node indices in the batch
        node_indices = batch_graph.ptr[:-1] + batch_actions

        # Ensure node_indices are within bounds
        assert torch.all(
            node_indices < q_values.size(0)
        ), "node_indices exceed q_values size."

        current_q_values = q_values[node_indices]

        # Forward pass for next states
        with torch.no_grad():
            next_q_values = self.model(next_batch_graph)
            # Vectorized max-next-Q using scatter (replaces Python loop)
            max_next_q_values = scatter(
                next_q_values, next_batch_graph.batch, reduce="max"
            )

        # Compute target Q-values
        target_q_values = batch_rewards + self.gamma * max_next_q_values * (
            1 - batch_dones
        )

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad(set_to_none=True)
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
