"""Reinforcement Learning agents for Scotland Yard.

This module provides various RL agent implementations:

Available Agents:
    - RandomAgent: Simple baseline that selects random valid actions
    - GNNAgent: Graph Neural Network agent using DQN with experience replay
    - MappoAgent: Multi-Agent PPO with centralized critic and per-agent policies

Usage:
    from agent import GNNAgent, MappoAgent, RandomAgent
    
    # For GNN-based training
    agent = GNNAgent(node_feature_size=3, device=device)
    action = agent.select_action(graph_data, action_mask)
    
    # For MAPPO training
    agent = MappoAgent(n_agents=2, obs_size=10, global_obs_size=20, ...)
    action, log_prob, value = agent.select_action(agent_idx, observation, mask)
    
    # For baseline/testing
    agent = RandomAgent()
    action = agent.select_action(observation, action_mask)

Note:
    GNNAgent and MappoAgent do not inherit from BaseAgent because they have
    different interfaces and require PyTorch nn.Module functionality.
    See base_agent.py for detailed design rationale.
"""

from agent.base_agent import BaseAgent
from agent.gnn_agent import GNNAgent
from agent.mappo_agent import MappoAgent
from agent.random_agent import RandomAgent

__all__ = [
    "BaseAgent",
    "GNNAgent",
    "MappoAgent",
    "RandomAgent",
]
