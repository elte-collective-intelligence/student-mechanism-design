"""Population-based self-play utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Callable
import random


@dataclass
class PolicyEntry:
    name: str
    policy: object
    score: float = 0.0


@dataclass
class Population:
    agents: Dict[str, List[PolicyEntry]] = field(default_factory=lambda: {"MrX": [], "Police": []})


class PopulationManager:
    def __init__(self, population: Population | None = None):
        self.population = population or Population()

    def add_policy(self, role: str, name: str, policy: object, score: float = 0.0):
        self.population.agents[role].append(PolicyEntry(name=name, policy=policy, score=score))

    def sample_opponents(self, role: str, k: int = 1) -> List[PolicyEntry]:
        pool = self.population.agents.get(role, [])
        if not pool:
            return []
        return random.sample(pool, min(k, len(pool)))

    def best_response_target(self, role: str) -> PolicyEntry | None:
        pool = self.population.agents.get(role, [])
        if not pool:
            return None
        return max(pool, key=lambda p: p.score)

    def update_score(self, role: str, name: str, delta: float):
        for entry in self.population.agents.get(role, []):
            if entry.name == name:
                entry.score += delta
                break
