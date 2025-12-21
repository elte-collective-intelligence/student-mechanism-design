"""Centralised mechanism configuration dataclasses."""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class MechanismConfig:
    tolls: float | np.ndarray | None = None
    police_budget: int = 10
    reveal_interval: int = 5
    reveal_probability: float = 0.0
    ticket_price: float = 1.0
    target_win_rate: float = 0.5
    secondary_weight: float = 0.1

    def to_dict(self) -> Dict[str, float]:
        return {
            "tolls": self.tolls,
            "police_budget": self.police_budget,
            "reveal_interval": self.reveal_interval,
            "reveal_probability": self.reveal_probability,
            "ticket_price": self.ticket_price,
            "target_win_rate": self.target_win_rate,
            "secondary_weight": self.secondary_weight,
        }
