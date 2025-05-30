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
        # Store dimensions for debugging
        self.obs_size = obs_size
        self.action_size = action_size

    def forward(self, x):
        # Ensure input dimensions match the network
        if x.ndim == 2 and x.size(-1) != self.obs_size:  # check only if x is 2D
            raise ValueError(f"Input features dim {x.size(-1)} doesn't match expected obs_size {self.obs_size}")
        elif x.ndim == 1 and x.size(0) != self.obs_size:  # check for 1D input, e.g. during direct call with single obs
            raise ValueError(
                f"Input features dim {x.size(0)} doesn't match expected obs_size {self.obs_size} for 1D input")
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
    def __init__(self, n_agents, obs_size, global_obs_size, action_size, hidden_size, device='cpu', gamma=0.99, lr=3e-4,
                 batch_size=64, buffer_size=5000, epsilon=0.2, ppo_epochs=1):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.gamma = gamma
        self.epsilon_clip = epsilon
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.action_size = action_size  # This is the action_size the policy network is configured with
        self.ppo_epochs = ppo_epochs

        self.policies = [AgentPolicy(obs_size, self.action_size, hidden_size).to(self.device) for _ in range(n_agents)]
        self.critic = CentralCritic(global_obs_size, hidden_size).to(self.device)

        self.optimizers = [optim.Adam(policy.parameters(), lr=lr) for policy in self.policies]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = []

    def select_action(self, agent_id, obs, action_mask=None):
        #print(f"DEBUG: select_action for agent_id: {agent_id}")
        #print(f"DEBUG: Initial obs shape: {obs.shape if isinstance(obs, torch.Tensor) else type(obs)}")
        #print(f"DEBUG: self.action_size (policy's configured output_dim): {self.action_size}")
        #if action_mask is not None:
            #print(f"DEBUG: action_mask shape: {action_mask.shape}")

        obs_tensor = obs.float().to(self.device)
        #print(f"DEBUG: obs_tensor shape (after .float().to(device)): {obs_tensor.shape}")

        # Ensure obs_tensor is 2D [1, features] for the policy network
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)  # Convert [features] to [1, features]
            #print(f"DEBUG: obs_tensor shape (after unsqueeze for 1D input): {obs_tensor.shape}")
        elif obs_tensor.ndim == 2 and obs_tensor.shape[0] != 1:
            # This case should ideally not happen in select_action for a single agent.
            # If it does, take the first observation from the batch.
            print(f"WARNING: obs_tensor has batch dimension {obs_tensor.shape[0]} but expected 1. Using only the first element.")
            obs_tensor = obs_tensor[0].unsqueeze(0)
            #print(f"DEBUG: obs_tensor shape (after taking first element from batch): {obs_tensor.shape}")
        elif obs_tensor.ndim != 2:
            raise ValueError(f"obs_tensor has unexpected ndim {obs_tensor.ndim}. Expected 2D tensor [1, features]. Shape: {obs_tensor.shape}")

        # At this point, obs_tensor must be [1, features]
        #print(f"DEBUG: obs_tensor shape (input to policy): {obs_tensor.shape}")

        policy_output_raw = self.policies[agent_id](obs_tensor)  # Expected Shape: [1, self.action_size]
        #print(f"DEBUG: policy_output_raw shape: {policy_output_raw.shape}")

        # Ensure policy output batch size is 1, then squeeze to get 1D probs for this single state.
        if policy_output_raw.ndim != 2 or policy_output_raw.shape[0] != 1:
            raise ValueError(
                f"Policy output has unexpected shape. Expected [1, action_size], got {policy_output_raw.shape}. "
                f"This might happen if obs_tensor input to policy (shape: {obs_tensor.shape}) was not [1, features]."
            )
        current_probs = policy_output_raw.squeeze(0)  # Shape: [self.action_size]
        #print(f"DEBUG: current_probs shape after squeeze(0) (expected 1D): {current_probs.shape}")

        if action_mask is not None:
            if current_probs.size(0) != action_mask.size(0):
                #print(f"DEBUG: Size mismatch: current_probs.size(0)={current_probs.size(0)} vs action_mask.size(0)={action_mask.size(0)}")

                new_probs_target_size = action_mask.size(0)
                new_probs = torch.zeros(new_probs_target_size, dtype=current_probs.dtype, device=self.device)
                #print(f"DEBUG: new_probs created with shape: {new_probs.shape}")

                min_s = min(current_probs.size(0), new_probs_target_size)
                #print(f"DEBUG: min_s for copy: {min_s}")

                if min_s > 0:
                    slice_to_assign = current_probs[:min_s]
                    #print(f"DEBUG: slice_to_assign shape: {slice_to_assign.shape}")
                    new_probs[:min_s] = slice_to_assign
                    #print(f"DEBUG: Assignment new_probs[:min_s] = slice_to_assign completed.")

                current_probs = new_probs
                #print(f"DEBUG: current_probs updated due to size mismatch. New shape: {current_probs.shape}")

            mask = action_mask.float().to(self.device)
            current_probs = current_probs * mask

            probs_sum = current_probs.sum()
            if probs_sum <= 1e-8:  # Use a small epsilon for floating point comparison
                mask_sum = mask.sum()
                if mask_sum > 1e-8:
                    current_probs = mask / mask_sum  # Distribute probability uniformly over valid actions
                else:
                    current_probs = torch.ones_like(mask, device=self.device) / mask.size(0)  # Uniform over all
            else:
                current_probs = current_probs / (probs_sum + 1e-8)  # Normalize, add epsilon to avoid division by zero

        dist = torch.distributions.Categorical(probs=current_probs)  # Use probs= for clarity
        action = dist.sample()
        return action.item(), dist.log_prob(action), current_probs.detach()  # Return detached probs for logging

    def store(self, obs, global_obs, actions, rewards, log_probs, dones):
        self.memory.append((obs, global_obs, actions, rewards, log_probs, dones))

    def memory_full(self):
        return len(self.memory) >= self.buffer_size

    def clear_memory(self):
        self.memory = []

    def ppo_update(self):
        if len(self.memory) < self.buffer_size and len(self.memory) != 0:
            return

        memory_buffer = self.memory[-self.buffer_size:]

        obs_b, global_obs_b, actions_b, rewards_b, log_probs_b, dones_b = zip(*memory_buffer)

        try:
            # Process observations
            obs_b = [[o.to(self.device) if isinstance(o, torch.Tensor) else torch.tensor(o, device=self.device)
                      for o in obs_t] for obs_t in obs_b]

            # Process global observations (reshape once)
            stacked_global_obs = torch.stack([g.to(self.device) if isinstance(g, torch.Tensor)
                                              else torch.tensor(g, device=self.device) for g in global_obs_b])

            # Reshape only if needed
            expected_input_size = self.critic.critic[0].in_features
            if stacked_global_obs.dim() > 2 or stacked_global_obs.size(-1) != expected_input_size:
                batch_size = stacked_global_obs.size(0)
                global_obs_b = stacked_global_obs.reshape(batch_size, -1)
                # Make sure the size matches expected critic input size
                if global_obs_b.size(1) != expected_input_size:
                    global_obs_b = torch.nn.functional.pad(
                        global_obs_b[:, :min(global_obs_b.size(1), expected_input_size)],
                        (0, max(0, expected_input_size - global_obs_b.size(1)))
                    )
            else:
                global_obs_b = stacked_global_obs
        except Exception as e:
            print(f"Error processing observations: {e}")
            batch_size = len(global_obs_b)
            expected_input_size = self.critic.critic[0].in_features
            global_obs_b = torch.zeros((batch_size, expected_input_size), device=self.device)

        # Process actions, rewards, log_probs, dones efficiently
        actions_b = torch.tensor([a[0] if isinstance(a, list) and len(a) > 0 else a for a in actions_b],
                                 dtype=torch.long, device=self.device)
        rewards_b = torch.tensor([r[0] if isinstance(r, list) and len(r) > 0 else r for r in rewards_b],
                                 dtype=torch.float, device=self.device)
        dones_b = torch.tensor([d[0] if isinstance(d, list) and len(d) > 0 else d for d in dones_b],
                               dtype=torch.float, device=self.device)

        # Process log probs efficiently
        log_probs_b = torch.tensor([lp[0] if isinstance(lp, list) and len(lp) > 0 else
                                    lp.item() if isinstance(lp, torch.Tensor) else lp
                                    for lp in log_probs_b], dtype=torch.float, device=self.device)

        # Compute values and advantages once
        with torch.no_grad():
            values = self.critic(global_obs_b).squeeze()

        # Compute returns (single pass)
        returns = torch.zeros_like(rewards_b)
        discounted_reward = 0
        for i in reversed(range(len(rewards_b))):
            discounted_reward = rewards_b[i] + self.gamma * discounted_reward * (1 - dones_b[i])
            returns[i] = discounted_reward

        # Compute advantages (single operation)
        advantages = returns - values.detach()

        # Normalize advantages for stable training
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            self.critic_optimizer.zero_grad()
            new_values = self.critic(global_obs_b).squeeze()
            critic_loss = nn.MSELoss()(new_values, returns)
            critic_loss.backward()
            self.critic_optimizer.step()

        # 2. Update actors once per agent with independent graphs
        for agent_id in range(self.n_agents):
            try:
                # Gather observations for this agent
                agent_obs_indices = [min(agent_id, len(step_obs) - 1) for step_obs in obs_b]
                agent_obs = torch.stack([obs_b[i][idx] for i, idx in enumerate(agent_obs_indices)])

                # Forward pass
                self.optimizers[agent_id].zero_grad()
                probs_dist = self.policies[agent_id](agent_obs)

                # Get valid actions
                valid_actions = actions_b.clamp(0, probs_dist.size(1) - 1)

                # Get new log probs
                dist = torch.distributions.Categorical(probs_dist)
                new_log_probs = dist.log_prob(valid_actions)

                # Compute ratio and PPO objective (vectorized)
                ratio = torch.exp(new_log_probs - log_probs_b)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Backward pass
                actor_loss.backward()
                self.optimizers[agent_id].step()

            except Exception as e:
                print(f"Error updating agent {agent_id}: {e}")

        # Clear memory after update
        self.memory.clear()

    def save(self, filepath):
        state_dict = {
            'policies': [policy.state_dict() for policy in self.policies],
            'critic': self.critic.state_dict(),
            'optimizers': [optimizer.state_dict() for optimizer in self.optimizers],
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
        torch.save(state_dict, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        for i, policy in enumerate(self.policies):
            policy.load_state_dict(checkpoint['policies'][i])
            policy.to(self.device)
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic.to(self.device)
        for i, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict(checkpoint['optimizers'][i])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    def state_dict(self):
        state_dict = {
            'policies': [policy.state_dict() for policy in self.policies],
            'critic': self.critic.state_dict()
        }
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        for i, policy in enumerate(self.policies):
            policy.load_state_dict(state_dict['policies'][i], strict=strict)
        self.critic.load_state_dict(state_dict['critic'], strict=strict)