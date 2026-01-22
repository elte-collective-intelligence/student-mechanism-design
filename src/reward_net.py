import torch
import torch.nn as nn

# Reward weight names used throughout the codebase
REWARD_WEIGHT_NAMES = [
    "Police_distance",
    "Police_group",
    "Police_position",
    "Police_time",
    "Mrx_closest",
    "Mrx_average",
    "Mrx_position",
    "Mrx_time",
    "Police_coverage",
    "Police_proximity",
    "Police_overlap_penalty",
]

NUM_REWARD_WEIGHTS = len(REWARD_WEIGHT_NAMES)  # 11


class RewardWeightNet(nn.Module):
    """Neural network that predicts reward weights based on game configuration.

    Input: [num_agents, agent_money, graph_nodes, graph_edges]
    Output: Sigmoid-activated weights for each reward component (11 values)
    """

    def __init__(self, input_size=4, hidden_size=32, output_size=NUM_REWARD_WEIGHTS):
        super(RewardWeightNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

    def predict_weights(
        self,
        num_agents: int,
        agent_money: float,
        graph_nodes: int,
        graph_edges: int,
        device=None,
    ) -> dict:
        """Convenience method to get named reward weights.

        Returns:
            Dictionary mapping reward weight names to their predicted values.
        """
        if device is None:
            device = next(self.parameters()).device

        inputs = torch.FloatTensor(
            [[num_agents, agent_money, graph_nodes, graph_edges]]
        ).to(device)
        predicted = self(inputs)

        return {name: predicted[0, i] for i, name in enumerate(REWARD_WEIGHT_NAMES)}
