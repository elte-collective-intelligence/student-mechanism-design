import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from RLAgent.base_agent import BaseAgent

class DQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, hidden_size=64, gamma=0.99, lr=1e-3, batch_size=64, buffer_size=10000, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Deep Q-Network Agent with Experience Replay and Epsilon-Greedy policy.
        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            hidden_size (int): Number of units in the hidden layers of the Q-network.
            gamma (float): Discount factor.
            lr (float): Learning rate.
            batch_size (int): Mini-batch size for training.
            buffer_size (int): Experience replay buffer size.
            epsilon (float): Starting value of epsilon for epsilon-greedy policy.
            epsilon_decay (float): Decay rate for epsilon.
            epsilon_min (float): Minimum value of epsilon.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # Experience replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Q-Network
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def select_action(self, observation, action_mask):
        """
        Selects an action using epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:
            # Explore: select a random action from valid actions
            valid_actions = np.where(action_mask)[0]
            return random.choice(valid_actions)
        else:
            observation = torch.FloatTensor([observation]).view(1, 1) 
            q_values = self.model(observation).detach().numpy().flatten()  
            valid_actions = np.where(action_mask == 1)[0]
            if len(valid_actions) == 0:
                raise ValueError("No valid actions available.")
            valid_q_values = q_values[valid_actions]
            max_idx = np.argmax(valid_q_values)
            selected_action = valid_actions[max_idx]
            return selected_action


    def update(self, state, action, reward, next_state, done):
        """
        Stores experience in replay memory and updates Q-network.
        """
        # Add experience to memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Start learning if memory has enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample a mini-batch of experiences from memory
        mini_batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Convert to tensors
        states = torch.FloatTensor(states).view(self.batch_size, self.state_size)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states).view(self.batch_size, self.state_size)
        dones = torch.FloatTensor(dones)

        # Q values for current states
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = self.criterion(q_values, target_q_values.detach())

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
        self.model.load_state_dict(torch.load(filepath))
