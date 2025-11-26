"""Lightweight opponent behaviour modeling."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List
import numpy as np


class OpponentModel:
    def __init__(self):
        self.transition_counts: Dict[tuple, int] = defaultdict(int)

    def update(self, previous_action: int, action: int):
        key = (previous_action, action)
        self.transition_counts[key] += 1

    def predict(self, previous_action: int) -> List[int]:
        candidates = {k[1]: v for k, v in self.transition_counts.items() if k[0] == previous_action}
        if not candidates:
            return []
        sorted_actions = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
        return [a for a, _ in sorted_actions]

    def policy_prior(self, num_actions: int) -> np.ndarray:
        counts = np.zeros(num_actions)
        for (_, action), c in self.transition_counts.items():
            counts[action] += c
        if counts.sum() == 0:
            return np.ones(num_actions) / num_actions
        return counts / counts.sum()
