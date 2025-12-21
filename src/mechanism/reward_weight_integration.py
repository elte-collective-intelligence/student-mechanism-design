"""Helpers for wiring mechanism design parameters into the environment."""

from __future__ import annotations

from typing import Dict
import numpy as np

from mechanism.mechanism_config import MechanismConfig


def apply_mechanism_to_env(env, mechanism: MechanismConfig):
    """Attach mechanism parameters to an existing environment instance."""
    if hasattr(env, "agents_money"):
        env.agents_money = [env.agents_money[0]] + [
            mechanism.police_budget for _ in env.police_positions
        ]
    env.mechanism_config = mechanism
    return env


def mechanism_observation_tokens(mechanism: MechanismConfig) -> Dict[str, float]:
    """Expose mechanism parameters so policies can condition on them."""
    toll_mean = 0.0
    if mechanism.tolls is not None:
        toll_array = np.asarray(mechanism.tolls, dtype=float)
        toll_mean = float(toll_array.mean())
    return {
        "reveal_interval": mechanism.reveal_interval,
        "reveal_probability": mechanism.reveal_probability,
        "ticket_price": mechanism.ticket_price,
        "toll_mean": toll_mean,
        "police_budget": mechanism.police_budget,
    }
