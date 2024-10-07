from abc import ABC, abstractmethod


class BaseMetaLearningSystem(ABC):
    """Abstract base class for meta-learning systems."""

    @abstractmethod
    def meta_train(self, tasks, agents, environment):
        """Perform meta-training over a set of tasks.

        Args:
            tasks (Iterable): A collection of tasks for meta-training.
            agents (List[BaseAgent]): List of agents to be meta-trained.
            environment (BaseEnvironment): The environment to train on.
        """
        pass

    @abstractmethod
    def meta_evaluate(self, tasks, agents, environment):
        """Evaluate the meta-trained agents on new tasks.

        Args:
            tasks (Iterable): A collection of tasks for evaluation.
            agents (List[BaseAgent]): List of agents to evaluate.
            environment (BaseEnvironment): The environment to evaluate on.
        """
        pass

    @abstractmethod
    def save_meta_model(self, filepath):
        """Save the meta-learned model."""
        pass

    @abstractmethod
    def load_meta_model(self, filepath):
        """Load the meta-learned model."""
        pass