from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for RL agents."""

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
