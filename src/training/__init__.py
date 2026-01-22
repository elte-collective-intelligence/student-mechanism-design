"""Training module for Scotland Yard RL agents.

This module contains reusable training utilities and helper functions
extracted from main.py to reduce code duplication and improve maintainability.

The main entry point remains main.py, which uses these utilities.
"""

from training.utils import (
    device,
    create_curriculum,
    modify_curriculum,
    compute_target_difficulty,
    predict_reward_weights,
    create_graph_data,
    extract_step_info,
    is_episode_done,
    create_action_mask,
)

__all__ = [
    "device",
    "create_curriculum",
    "modify_curriculum",
    "compute_target_difficulty",
    "predict_reward_weights",
    "create_graph_data",
    "extract_step_info",
    "is_episode_done",
    "create_action_mask",
]
