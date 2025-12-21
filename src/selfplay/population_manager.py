"""Population-based self-play utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any
import random
import os
import json


@dataclass
class PolicyEntry:
    """A single policy in the population pool."""

    name: str
    policy: object
    score: float = 0.0
    wins: int = 0
    losses: int = 0
    games_played: int = 0
    generation: int = 0

    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.5
        return self.wins / self.games_played

    def record_game(self, won: bool):
        self.games_played += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": self.score,
            "wins": self.wins,
            "losses": self.losses,
            "games_played": self.games_played,
            "generation": self.generation,
            "win_rate": self.win_rate,
        }


@dataclass
class Population:
    """Population pool containing MrX and Police policies."""

    agents: Dict[str, List[PolicyEntry]] = field(
        default_factory=lambda: {"MrX": [], "Police": []}
    )
    max_size: int = 10

    def get_pool_stats(self) -> dict:
        stats = {}
        for role, entries in self.agents.items():
            if entries:
                stats[role] = {
                    "size": len(entries),
                    "avg_score": sum(e.score for e in entries) / len(entries),
                    "avg_win_rate": sum(e.win_rate for e in entries) / len(entries),
                    "best_score": max(e.score for e in entries),
                }
            else:
                stats[role] = {
                    "size": 0,
                    "avg_score": 0,
                    "avg_win_rate": 0,
                    "best_score": 0,
                }
        return stats


class PopulationManager:
    """Manages population-based self-play training.

    Implements:
    - Policy pool management for MrX and Police
    - Best response computation
    - Periodic refresh of populations
    - ELO-style scoring
    """

    def __init__(
        self,
        population: Population | None = None,
        max_population_size: int = 10,
        refresh_interval: int = 5,
        elo_k: float = 32.0,
    ):
        self.population = population or Population(max_size=max_population_size)
        self.refresh_interval = refresh_interval
        self.elo_k = elo_k
        self.generation = 0
        self._match_history: List[dict] = []

    def add_policy(self, role: str, name: str, policy: object, score: float = 1000.0):
        """Add a new policy to the population pool.

        Args:
            role: "MrX" or "Police"
            name: Unique identifier for the policy
            policy: The policy object (agent)
            score: Initial ELO score
        """
        entry = PolicyEntry(
            name=name, policy=policy, score=score, generation=self.generation
        )

        pool = self.population.agents[role]

        # Check if policy with same name exists
        existing_idx = next((i for i, e in enumerate(pool) if e.name == name), None)
        if existing_idx is not None:
            pool[existing_idx] = entry
        else:
            pool.append(entry)

        # Prune if over max size (remove lowest scoring)
        if len(pool) > self.population.max_size:
            pool.sort(key=lambda e: e.score, reverse=True)
            self.population.agents[role] = pool[: self.population.max_size]

    def sample_opponents(
        self, role: str, k: int = 1, strategy: str = "weighted"
    ) -> List[PolicyEntry]:
        """Sample opponents from the population pool.

        Args:
            role: Role to sample from ("MrX" or "Police")
            k: Number of opponents to sample
            strategy: Sampling strategy ("uniform", "weighted", "best")

        Returns:
            List of sampled PolicyEntry objects
        """
        pool = self.population.agents.get(role, [])
        if not pool:
            return []

        k = min(k, len(pool))

        if strategy == "uniform":
            return random.sample(pool, k)
        elif strategy == "best":
            sorted_pool = sorted(pool, key=lambda e: e.score, reverse=True)
            return sorted_pool[:k]
        elif strategy == "weighted":
            # Weight by score (softmax-style)
            scores = [e.score for e in pool]
            min_score = min(scores) - 1
            weights = [s - min_score for s in scores]
            total = sum(weights)
            weights = [w / total for w in weights]
            return random.choices(pool, weights=weights, k=k)
        else:
            return random.sample(pool, k)

    def best_response_target(self, role: str) -> PolicyEntry | None:
        """Get the best policy to compute best response against.

        Args:
            role: Role to find best response target for

        Returns:
            The highest scoring PolicyEntry or None
        """
        pool = self.population.agents.get(role, [])
        if not pool:
            return None
        return max(pool, key=lambda p: p.score)

    def update_elo(
        self, winner_role: str, winner_name: str, loser_role: str, loser_name: str
    ):
        """Update ELO scores after a match.

        Args:
            winner_role: Role of the winner
            winner_name: Name of the winning policy
            loser_role: Role of the loser
            loser_name: Name of the losing policy
        """
        winner_entry = self._find_entry(winner_role, winner_name)
        loser_entry = self._find_entry(loser_role, loser_name)

        if winner_entry and loser_entry:
            # ELO calculation
            expected_winner = 1 / (
                1 + 10 ** ((loser_entry.score - winner_entry.score) / 400)
            )
            expected_loser = 1 - expected_winner

            winner_entry.score += self.elo_k * (1 - expected_winner)
            loser_entry.score += self.elo_k * (0 - expected_loser)

            winner_entry.record_game(won=True)
            loser_entry.record_game(won=False)

            self._match_history.append(
                {
                    "winner": {"role": winner_role, "name": winner_name},
                    "loser": {"role": loser_role, "name": loser_name},
                    "generation": self.generation,
                }
            )

    def update_score(self, role: str, name: str, delta: float):
        """Update score for a policy.

        Args:
            role: Policy role
            name: Policy name
            delta: Score change
        """
        entry = self._find_entry(role, name)
        if entry:
            entry.score += delta

    def _find_entry(self, role: str, name: str) -> PolicyEntry | None:
        for entry in self.population.agents.get(role, []):
            if entry.name == name:
                return entry
        return None

    def should_refresh(self, epoch: int) -> bool:
        """Check if population should be refreshed."""
        return epoch > 0 and epoch % self.refresh_interval == 0

    def step_generation(self):
        """Advance to next generation."""
        self.generation += 1

    def get_stats(self) -> dict:
        """Get population statistics."""
        return {
            "generation": self.generation,
            "pool_stats": self.population.get_pool_stats(),
            "total_matches": len(self._match_history),
        }

    def save_population(self, filepath: str):
        """Save population state to file."""
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )
        data = {
            "generation": self.generation,
            "agents": {
                role: [e.to_dict() for e in entries]
                for role, entries in self.population.agents.items()
            },
            "match_history": self._match_history[-100:],  # Keep last 100 matches
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_population_stats(self, filepath: str) -> dict:
        """Load population statistics from file."""
        with open(filepath, "r") as f:
            return json.load(f)


def run_population_self_play(
    population_manager: PopulationManager,
    train_fn: Callable,
    eval_fn: Callable,
    num_epochs: int = 100,
    games_per_epoch: int = 10,
    logger: Any = None,
) -> PopulationManager:
    """Run population-based self-play training loop.

    Args:
        population_manager: The PopulationManager instance
        train_fn: Function to train a policy against opponents
        eval_fn: Function to evaluate a match between two policies
        num_epochs: Number of training epochs
        games_per_epoch: Number of games per epoch
        logger: Optional logger

    Returns:
        Updated PopulationManager
    """
    for epoch in range(num_epochs):
        # Sample opponents for training
        mrx_opponents = population_manager.sample_opponents("MrX", k=3)
        police_opponents = population_manager.sample_opponents("Police", k=3)

        # Train new policies (best response)
        if mrx_opponents:
            new_mrx = train_fn("MrX", police_opponents)
            population_manager.add_policy("MrX", f"MrX_gen{epoch}", new_mrx)

        if police_opponents:
            new_police = train_fn("Police", mrx_opponents)
            population_manager.add_policy("Police", f"Police_gen{epoch}", new_police)

        # Run evaluation games
        for _ in range(games_per_epoch):
            mrx_entry = population_manager.sample_opponents(
                "MrX", k=1, strategy="weighted"
            )
            police_entry = population_manager.sample_opponents(
                "Police", k=1, strategy="weighted"
            )

            if mrx_entry and police_entry:
                winner = eval_fn(mrx_entry[0].policy, police_entry[0].policy)
                if winner == "MrX":
                    population_manager.update_elo(
                        "MrX", mrx_entry[0].name, "Police", police_entry[0].name
                    )
                else:
                    population_manager.update_elo(
                        "Police", police_entry[0].name, "MrX", mrx_entry[0].name
                    )

        # Step generation
        if population_manager.should_refresh(epoch):
            population_manager.step_generation()

        if logger:
            stats = population_manager.get_stats()
            logger.log(f"Epoch {epoch}: {stats}", level="info")

    return population_manager
