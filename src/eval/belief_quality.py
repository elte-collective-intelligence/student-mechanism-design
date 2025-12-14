"""Belief accuracy metrics."""

from __future__ import annotations

import numpy as np


def belief_cross_entropy(belief: np.ndarray, true_index: int) -> float:
    belief = np.clip(belief, 1e-8, 1.0)
    belief = belief / belief.sum()
    return float(-np.log(belief[true_index]))
