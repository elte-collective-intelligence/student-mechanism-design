"""Meta-learning loop skeleton for adjusting mechanism parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np

from mechanism.mechanism_config import MechanismConfig


@dataclass
class MetaLearningState:
    mechanism: MechanismConfig
    history: list


class MetaLearner:
    """Simple bandit-style meta-learner.

    The learner observes win-rate feedback and nudges reveal probability/tolls
    toward the desired target.  It is intentionally light-weight so it can run
    in unit tests and offline sweeps.
    """

    def __init__(self, mechanism: MechanismConfig, learning_rate: float = 0.1):
        self.state = MetaLearningState(mechanism=mechanism, history=[])
        self.learning_rate = learning_rate

    def step(self, metrics: Dict[str, float]):
        win_rate = metrics.get("win_rate", 0.0)
        cost = metrics.get("mechanism_cost", 0.0)
        target = self.state.mechanism.target_win_rate
        error = win_rate - target
        # Update reveal probability and ticket price as proxy parameters.
        self.state.mechanism.reveal_probability = float(
            np.clip(
                self.state.mechanism.reveal_probability - self.learning_rate * error,
                0.0,
                1.0,
            )
        )
        self.state.mechanism.ticket_price = float(
            max(0.0, self.state.mechanism.ticket_price + self.learning_rate * cost)
        )
        self.state.history.append(
            {
                "win_rate": win_rate,
                "cost": cost,
                "reveal_probability": self.state.mechanism.reveal_probability,
            }
        )
        return self.state.mechanism

    def reset_history(self):
        self.state.history.clear()


def run_meta_loop(
    initial_mechanism: MechanismConfig,
    eval_fn: Callable[[MechanismConfig], Dict[str, float]],
    iterations: int = 5,
    lr: float = 0.1,
) -> MechanismConfig:
    learner = MetaLearner(initial_mechanism, learning_rate=lr)
    mech = initial_mechanism
    for _ in range(iterations):
        metrics = eval_fn(mech)
        mech = learner.step(metrics)
    return mech
