import torch
import torch.nn as nn
import torch.optim as optim

class AgentPolicy(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return torch.softmax(self.actor(x), dim=-1)


class CentralCritic(nn.Module):
    def __init__(self, global_obs_size, hidden_size):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(global_obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.critic(x)


class MappoAgent:
    def __init__(self, n_agents, obs_size, global_obs_size, action_size, hidden_size, device='cpu', gamma=0.99, lr=3e-4, batch_size=64, buffer_size=10000, epsilon=0.2):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.gamma = gamma
        self.epsilon_clip = epsilon
        self.batch_size = batch_size
        self.action_size = action_size

        self.policies = [AgentPolicy(obs_size, action_size, hidden_size).to(self.device) for _ in range(n_agents)]
        self.critic = CentralCritic(global_obs_size, hidden_size).to(self.device)

        self.optimizers = [optim.Adam(policy.parameters(), lr=lr) for policy in self.policies]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = []

    def select_action(self, agent_id, obs, action_mask=None):
        obs_tensor = obs.float().to(self.device).unsqueeze(0)
        probs = self.policies[agent_id](obs_tensor).squeeze(0)

        if action_mask is not None:
            mask = action_mask.float().to(self.device)
            probs = probs * mask
            probs /= probs.sum() + 1e-8

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), probs.detach()

    def store(self, obs, global_obs, actions, rewards, log_probs, dones):
        self.memory.append((obs, global_obs, actions, rewards, log_probs, dones))

    def ppo_update(self):
        obs_b, global_obs_b, actions_b, rewards_b, log_probs_b, dones_b = zip(*self.memory)

        # Convert to tensors
        obs_b = [[o.to(self.device) for o in obs_t] for obs_t in obs_b]
        global_obs_b = torch.stack(global_obs_b).to(self.device)
        actions_b = torch.tensor(actions_b).long().to(self.device)
        rewards_b = torch.tensor(rewards_b).float().to(self.device)
        log_probs_b = torch.stack(log_probs_b).to(self.device)
        dones_b = torch.tensor(dones_b).float().to(self.device)

        # Compute returns and advantages
        returns = []
        G = 0
        for r, d in zip(reversed(rewards_b), reversed(dones_b)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.tensor(returns).float().to(self.device)
        values = self.critic(global_obs_b).squeeze()
        advantages = returns - values.detach()

        for _ in range(4):
            new_values = self.critic(global_obs_b).squeeze()
            critic_loss = nn.MSELoss()(new_values, returns)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            for agent_id in range(self.n_agents):
                agent_obs = torch.stack([step[agent_id] for step in obs_b])
                probs = self.policies[agent_id](agent_obs)
                dists = torch.distributions.Categorical(probs)
                new_log_probs = dists.log_prob(actions_b[:, agent_id])

                ratio = (new_log_probs - log_probs_b[:, agent_id]).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                self.optimizers[agent_id].zero_grad()
                actor_loss.backward()
                self.optimizers[agent_id].step()

        self.memory.clear()