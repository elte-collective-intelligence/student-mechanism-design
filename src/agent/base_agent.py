"""Base agent interface for Scotland Yard RL agents.

This module defines the abstract base class for simple agents.
Note that GNNAgent and MappoAgent do NOT inherit from BaseAgent because they
have fundamentally different interfaces and require PyTorch module functionality.

Agent Types:
    - RandomAgent: Inherits from BaseAgent, simple random action selection
    - GNNAgent: Standalone class with graph neural network, uses state_dict for saving
    - MappoAgent: Standalone class with PPO, has per-agent policy management

Design Decision:
    GNNAgent and MappoAgent are not subclasses of BaseAgent because:
    1. They have different method signatures (e.g., select_action takes different arguments)
    2. They need torch.nn.Module functionality for save/load via state_dict
    3. Forcing inheritance would require awkward adapters or break existing code
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for simple RL agents.

    This base class is suitable for agents that don't require neural networks
    or complex state management. For neural network agents, see GNNAgent
    and MappoAgent which have their own interfaces.
    """

    @abstractmethod
    def select_action(self, observation, action_mask):
        """Select an action based on the current observation and action mask.

        Args:
            observation (Any): The current observation from the environment.
            action_mask (np.ndarray): Mask indicating valid actions.

        Returns:
            int: Selected action.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update the agent's knowledge based on experiences."""
        pass

    @abstractmethod
    def save(self, filepath):
        """Save the agent's state to a file."""
        pass

    @abstractmethod
    def load(self, filepath):
        """Load the agent's state from a file."""
        pass
