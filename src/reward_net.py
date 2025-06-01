import torch
import torch.nn as nn

class RewardWeightNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, output_size=8):
        super(RewardWeightNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x) 