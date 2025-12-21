import numpy as np
import random

from RLAgent.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """A simple agent that selects actions randomly based on the action mask."""

    def select_action(self, observation, action_mask):
        """Select a random valid action."""
        valid_actions = np.where(action_mask == 1)[0]
        return random.choice(valid_actions) if len(valid_actions) > 0 else 0

    def update(self, *args, **kwargs):
        """RandomAgent does not learn."""
        pass

    def save(self, filepath):
        """RandomAgent does not have state to save."""
        pass

    def load(self, filepath):
        """RandomAgent does not have state to load."""
        pass
