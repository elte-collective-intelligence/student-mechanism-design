from abc import ABC, abstractmethod
from pettingzoo import ParallelEnv


class BaseEnvironment(ParallelEnv, ABC):
    """Abstract base class for custom environments."""

    metadata = {"render_modes": ["human"], "name": "base_environment"}

    @abstractmethod
    def reset(self, seed=None, options=None):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def action_space(self, agent):
        pass
