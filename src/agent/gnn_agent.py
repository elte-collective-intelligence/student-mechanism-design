"""Graph Neural Network (GNN) agent wrapper for Scotland Yard.

This module provides a compliant GNNAgent class that inherits from BaseAgent,
satisfying the assignment's architecture requirements while delegating to
the optimized GraphDQNAgent implementation.
"""

from .base_agent import BaseAgent
from .graph_dqn_agent import GraphDQNAgent


class GNNAgent(GraphDQNAgent, BaseAgent):
    """
    Graph Neural Network (GNN) agent.

    This agent uses standard Graph Convolutional Layers.
    """

    def __init__(self, **kwargs):
        """
        Initialize the GNN agent.

        Extracts model-specific kwargs into model_kwargs and forces agent_type to 'gnn'.
        """
        # Define expected GraphDQNAgent constructor args
        dqn_args = [
            "node_feature_size",
            "gamma",
            "lr",
            "batch_size",
            "buffer_size",
            "epsilon",
            "epsilon_decay",
            "epsilon_min",
            "device",
        ]

        dqn_kwargs = {k: v for k, v in kwargs.items() if k in dqn_args}
        model_kwargs = {k: v for k, v in kwargs.items() if k not in dqn_args}

        dqn_kwargs["agent_type"] = "gnn"
        dqn_kwargs["model_kwargs"] = model_kwargs

        super().__init__(**dqn_kwargs)

    def select_action(self, observation, action_mask):
        """
        Select an action based on observation and mask.

        Note: If observation is already a torch_geometric Data object,
        it is passed directly. Otherwise, standard select_action is used.
        """
        return super().select_action(observation, action_mask)

    def update(self, *args, **kwargs):
        """Delegate update to GraphDQNAgent."""
        return super().update(*args, **kwargs)

    def save(self, filepath):
        """Delegate save to GraphDQNAgent."""
        return super().save(filepath)

    def load(self, filepath):
        """Delegate load to GraphDQNAgent."""
        return super().load(filepath)
