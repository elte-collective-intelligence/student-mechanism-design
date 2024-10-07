import learn2learn as l2l
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from logging import Logger

from MetaLearning.base_meta import BaseMetaLearningSystem


# Define the MAML meta-learning system for RL agents
class MAMLMetaLearningSystem(BaseMetaLearningSystem):
    """MAML system for RL agents catching a criminal on a weighted graph."""
    
    def __init__(self, policy, lr_inner=0.01, lr_outer=0.001, inner_steps=1, logger=None):
        super(MAMLMetaLearningSystem, self).__init__()
        self.policy = policy
        self.maml = l2l.algorithms.MAML(policy, lr=lr_inner, first_order=False)
        self.optimizer = optim.Adam(self.maml.parameters(), lr=lr_outer)
        self.inner_steps = inner_steps
        self.logger = logger if logger else Logger()
        self.logger.log(f"Initialized MAMLMetaLearningSystem with lr_inner: {lr_inner}, lr_outer: {lr_outer}, inner_steps: {inner_steps}")
    
    def adapt(self, task_env):
        learner = self.maml.clone()
        adaptation_loss = 0.0

        for step in range(self.inner_steps):
            trajectories = self.collect_trajectories(learner, task_env)
            loss = self.compute_loss(trajectories)
            learner.adapt(loss)
            adaptation_loss += loss.item()
            self.logger.log(f"Inner Step {step+1}/{self.inner_steps}: Adaptation Loss: {loss.item():.4f}")
            self.logger.log_scalar("adaptation_loss", loss.item(), step)
        evaluation_loss = self.evaluate(learner, task_env)
        self.logger.log(f"Evaluation Loss after adaptation: {evaluation_loss.item():.4f}")
        self.logger.log_scalar("evaluation_loss", evaluation_loss.item(), self.inner_steps)

        return learner, adaptation_loss, evaluation_loss

    def meta_train(self, meta_train_tasks):
        for epoch in range(len(meta_train_tasks)):
            self.logger.log(f"Starting epoch {epoch+1}/{len(meta_train_tasks)}")
            task_env = meta_train_tasks[epoch]
            learner, adaptation_loss, evaluation_loss = self.adapt(task_env)

            # Meta-update
            self.optimizer.zero_grad()
            evaluation_loss.backward()
            self.optimizer.step()

            self.logger.log(f"Epoch {epoch+1}: Adaptation Loss: {adaptation_loss:.4f}, Evaluation Loss: {evaluation_loss.item():.4f}")
            self.logger.log_metrics({"epoch_adaptation_loss": adaptation_loss, "epoch_evaluation_loss": evaluation_loss.item()}, epoch+1)
    
    def meta_evaluate(self, meta_test_tasks):
        total_evaluation_loss = 0.0
        num_tasks = len(meta_test_tasks)

        self.logger.log(f"Starting meta-evaluation on {num_tasks} tasks")
        for i, task_env in enumerate(meta_test_tasks):
            self.logger.log(f"Evaluating task {i+1}/{num_tasks}")
            _, _, evaluation_loss = self.adapt(task_env)
            total_evaluation_loss += evaluation_loss.item()
            self.logger.log_scalar("task_evaluation_loss", evaluation_loss.item(), i+1)

        avg_evaluation_loss = total_evaluation_loss / num_tasks
        self.logger.log(f"Meta Evaluation: Average Evaluation Loss: {avg_evaluation_loss:.4f}")
        self.logger.log_scalar("average_evaluation_loss", avg_evaluation_loss, num_tasks)
        return avg_evaluation_loss

    def collect_trajectories(self, policy, env, num_episodes=5):
        trajectories = []
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            episode = []
            while not done:
                obs_tensor = self.process_observation(obs)
                action_logits, _ = policy(obs_tensor)
                actions = []
                for logits in action_logits:
                    action_prob = F.softmax(logits, dim=-1)
                    action = action_prob.multinomial(num_samples=1).detach()
                    actions.append(action.item())
                obs_next, rewards, done, _ = env.step(actions)
                episode.append((obs, actions, rewards))
                obs = obs_next
            trajectories.append(episode)
        return trajectories

    def compute_loss(self, trajectories):
        loss = 0.0
        for episode in trajectories:
            G = np.zeros(self.policy.num_agents)
            for obs, actions, rewards in reversed(episode):
                G = rewards + 0.99 * G
                obs_tensor = self.process_observation(obs)
                action_logits, _ = self.policy(obs_tensor)
                for i, logits in enumerate(action_logits):
                    log_prob = F.log_softmax(logits, dim=-1)
                    action = torch.tensor([actions[i]])
                    selected_log_prob = log_prob.gather(1, action.unsqueeze(0))
                    loss -= selected_log_prob * G[i]  # Negative for gradient ascent
        loss /= len(trajectories)
        return loss

    def evaluate(self, policy, env, num_episodes=5):
        total_reward = 0.0
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            while not done:
                obs_tensor = self.process_observation(obs)
                action_logits, _ = policy(obs_tensor)
                actions = []
                for logits in action_logits:
                    action_prob = F.softmax(logits, dim=-1)
                    action = action_prob.multinomial(num_samples=1).detach()
                    actions.append(action.item())
                obs_next, rewards, done, _ = env.step(actions)
                total_reward += np.sum(rewards)
                obs = obs_next
        average_reward = total_reward / num_episodes
        evaluation_loss = -average_reward  # Negative for gradient descent
        return torch.tensor(evaluation_loss, requires_grad=True)

    def process_observation(self, obs):
        obs_list = []
        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, np.ndarray):
                obs_list.append(torch.from_numpy(value.flatten()).float())
            else:
                obs_list.append(torch.tensor([value]).float())
        obs_tensor = torch.cat(obs_list).unsqueeze(0)  # Batch dimension
        return obs_tensor

    def save_meta_model(self, filepath):
        torch.save(self.maml.module.state_dict(), filepath)
        self.logger.log(f"Meta-learned policy saved to {filepath}")

    def load_meta_model(self, filepath):
        self.maml.module.load_state_dict(torch.load(filepath))
        self.logger.log(f"Meta-learned policy loaded from {filepath}")
