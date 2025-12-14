"""Utility for computing simple best responses within the population."""

from __future__ import annotations

from typing import Callable


def best_response_evaluate(policy, opponent_policy, eval_fn: Callable):
    """Evaluate a candidate policy against a fixed opponent."""
    return eval_fn(policy, opponent_policy)


def update_best_response_score(manager, role: str, policy_name: str, score_delta: float):
    manager.update_score(role, policy_name, score_delta)
