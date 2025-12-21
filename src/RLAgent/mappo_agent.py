import torch
import torch.nn as nn
import torch.optim as optim


class AgentPolicy(nn.Module):
    """Actor network for individual agents in MAPPO."""

    def __init__(self, obs_size, action_size, hidden_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.obs_size = obs_size
        self.action_size = action_size

    def forward(self, x):
        # Validate input dimensionality
        if x.ndim == 2 and x.size(-1) != self.obs_size:
            raise ValueError(
                f"Input features dim {x.size(-1)} doesn't match expected obs_size {self.obs_size}"
            )
        elif x.ndim == 1 and x.size(0) != self.obs_size:
            raise ValueError(
                f"Input features dim {x.size(0)} doesn't match expected obs_size {self.obs_size} for 1D input"
            )
        return torch.softmax(self.actor(x), dim=-1)


class CentralCritic(nn.Module):
    """Shared critic that evaluates global state for all agents."""

    def __init__(self, global_obs_size, hidden_size):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(global_obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.critic(x)


class MappoAgent:
    """Multi-Agent Proximal Policy Optimization (MAPPO) agent class."""

    def __init__(
        self,
        n_agents,
        obs_size,
        global_obs_size,
        action_size,
        hidden_size,
        device="cpu",
        gamma=0.99,
        lr=3e-4,
        batch_size=64,
        buffer_size=5000,
        epsilon=0.2,
        ppo_epochs=1,
    ):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.gamma = gamma
        self.epsilon_clip = epsilon
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.ppo_epochs = ppo_epochs

        self.policies = [
            AgentPolicy(obs_size, self.action_size, hidden_size).to(self.device)
            for _ in range(n_agents)
        ]
        self.critic = CentralCritic(global_obs_size, hidden_size).to(self.device)

        self.optimizers = [
            optim.Adam(policy.parameters(), lr=lr) for policy in self.policies
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = []

    def select_action(self, agent_id, obs, action_mask=None):
        # Ensure obs is shaped properly for batch input to policy
        obs_tensor = obs.float().to(self.device)
        if obs_tensor.ndim < 2:
            while obs_tensor.ndim < 2:
                obs_tensor = obs_tensor.unsqueeze(0)
        elif obs_tensor.ndim > 2:
            obs_tensor = obs_tensor.view(1, -1)
        elif obs_tensor.shape[0] != 1:
            obs_tensor = obs_tensor[0].unsqueeze(0)

        policy_output_raw = self.policies[agent_id](obs_tensor)

        if policy_output_raw.ndim != 2 or policy_output_raw.shape[0] != 1:
            raise ValueError(
                f"Policy output has unexpected shape. Got {policy_output_raw.shape}"
            )
        current_probs = policy_output_raw.squeeze(0)
        # Apply action mask if provided to zero out illegal actions
        if action_mask is not None:
            if current_probs.size(0) != action_mask.size(0):

                new_probs_target_size = action_mask.size(0)
                new_probs = torch.zeros(
                    new_probs_target_size, dtype=current_probs.dtype, device=self.device
                )
                min_s = min(current_probs.size(0), new_probs_target_size)

                if min_s > 0:
                    slice_to_assign = current_probs[:min_s]
                    new_probs[:min_s] = slice_to_assign

                current_probs = new_probs

            mask = action_mask.float().to(self.device)
            current_probs = current_probs * mask
            # Normalize probabilities
            probs_sum = current_probs.sum()
            if probs_sum <= 1e-8:
                mask_sum = mask.sum()
                if mask_sum > 1e-8:
                    current_probs = mask / mask_sum
                else:
                    current_probs = torch.ones_like(
                        mask, device=self.device
                    ) / mask.size(0)
            else:
                current_probs = current_probs / (probs_sum + 1e-8)

        dist = torch.distributions.Categorical(probs=current_probs)
        action = dist.sample()
        return (
            action.item(),
            dist.log_prob(action),
            current_probs.detach(),
        )  # Return detached probs for logging

    def store(self, obs, global_obs, actions, rewards, log_probs, dones):
        """Stores a transition tuple in memory."""
        self.memory.append((obs, global_obs, actions, rewards, log_probs, dones))

    def memory_full(self):
        """Checks if the memory is full."""
        return len(self.memory) >= self.buffer_size

    def clear_memory(self):
        """Clears the memory buffer."""
        self.memory = []

    def ppo_update(self):
        """Performs a PPO update using the collected experience."""
        # Wait until memory buffer is full before updating
        if len(self.memory) < self.buffer_size:
            return

        memory_buffer = self.memory[-self.buffer_size :]

        obs_b, global_obs_b, actions_b, rewards_b, log_probs_b, dones_b = zip(
            *memory_buffer
        )
        # Convert to tensors and handle edge cases
        try:
            obs_b = [
                [
                    (
                        o.to(self.device)
                        if isinstance(o, torch.Tensor)
                        else torch.tensor(o, device=self.device)
                    )
                    for o in obs_t
                ]
                for obs_t in obs_b
            ]

            stacked_global_obs = torch.stack(
                [
                    (
                        g.to(self.device)
                        if isinstance(g, torch.Tensor)
                        else torch.tensor(g, device=self.device)
                    )
                    for g in global_obs_b
                ]
            )
            # Ensure global_obs matches critic input shape
            expected_input_size = self.critic.critic[0].in_features
            if (
                stacked_global_obs.dim() > 2
                or stacked_global_obs.size(-1) != expected_input_size
            ):
                batch_size = stacked_global_obs.size(0)
                global_obs_b = stacked_global_obs.reshape(batch_size, -1)
                if global_obs_b.size(1) != expected_input_size:
                    global_obs_b = torch.nn.functional.pad(
                        global_obs_b[
                            :, : min(global_obs_b.size(1), expected_input_size)
                        ],
                        (0, max(0, expected_input_size - global_obs_b.size(1))),
                    )
            else:
                global_obs_b = stacked_global_obs
        except Exception as e:
            print(f"Error processing observations: {e}")
            batch_size = len(global_obs_b)
            expected_input_size = self.critic.critic[0].in_features
            global_obs_b = torch.zeros(
                (batch_size, expected_input_size), device=self.device
            )

        actions_b = torch.tensor(
            [a[0] if isinstance(a, list) and len(a) > 0 else a for a in actions_b],
            dtype=torch.long,
            device=self.device,
        )
        rewards_b = torch.tensor(
            [r[0] if isinstance(r, list) and len(r) > 0 else r for r in rewards_b],
            dtype=torch.float,
            device=self.device,
        )
        dones_b = torch.tensor(
            [d[0] if isinstance(d, list) and len(d) > 0 else d for d in dones_b],
            dtype=torch.float,
            device=self.device,
        )

        log_probs_b = torch.tensor(
            [
                (
                    lp[0]
                    if isinstance(lp, list) and len(lp) > 0
                    else lp.item() if isinstance(lp, torch.Tensor) else lp
                )
                for lp in log_probs_b
            ],
            dtype=torch.float,
            device=self.device,
        )
        # Estimate value function
        with torch.no_grad():
            values = self.critic(global_obs_b).squeeze()
        # Compute returns using reverse discounted reward sum
        returns = torch.zeros_like(rewards_b)
        discounted_reward = 0
        for i in reversed(range(len(rewards_b))):
            discounted_reward = rewards_b[i] + self.gamma * discounted_reward * (
                1 - dones_b[i]
            )
            returns[i] = discounted_reward
        # Compute advantage (standardized GAE-style)
        advantages = returns - values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Update critic
        for _ in range(self.ppo_epochs):
            self.critic_optimizer.zero_grad()
            new_values = self.critic(global_obs_b).squeeze()
            critic_loss = nn.MSELoss()(new_values, returns)
            critic_loss.backward()
            self.critic_optimizer.step()
        # Update each agent's policy
        for agent_id in range(self.n_agents):
            try:
                agent_obs_indices = [
                    min(agent_id, len(step_obs) - 1) for step_obs in obs_b
                ]
                agent_obs = torch.stack(
                    [obs_b[i][idx] for i, idx in enumerate(agent_obs_indices)]
                )

                self.optimizers[agent_id].zero_grad()
                probs_dist = self.policies[agent_id](agent_obs)

                valid_actions = actions_b.clamp(0, probs_dist.size(1) - 1)

                dist = torch.distributions.Categorical(probs_dist)
                new_log_probs = dist.log_prob(valid_actions)
                # PPO clipped surrogate loss
                ratio = torch.exp(new_log_probs - log_probs_b)
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
                    * advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                actor_loss.backward()
                self.optimizers[agent_id].step()

            except Exception as e:
                print(f"Error updating agent {agent_id}: {e}")

        self.memory.clear()

    def save(self, filepath):
        """Saves model and optimizer states to file."""
        state_dict = {
            "policies": [policy.state_dict() for policy in self.policies],
            "critic": self.critic.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        torch.save(state_dict, filepath)

    def load(self, filepath):
        """Loads model and optimizer states from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        for i, policy in enumerate(self.policies):
            policy.load_state_dict(checkpoint["policies"][i])
            policy.to(self.device)
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic.to(self.device)
        for i, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(checkpoint["optimizers"][i])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def state_dict(self):
        """Returns the model state for checkpointing."""
        state_dict = {
            "policies": [policy.state_dict() for policy in self.policies],
            "critic": self.critic.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """Loads model state."""
        for i, policy in enumerate(self.policies):
            policy.load_state_dict(state_dict["policies"][i], strict=strict)
        self.critic.load_state_dict(state_dict["critic"], strict=strict)
