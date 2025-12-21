"""Evaluation metrics for Scotland Yard mechanism design.

This module implements the three required metrics:
1. Balance (win rate) - MrX win rate targeting 50%
2. Belief quality - Cross-entropy of belief vs true MrX position
3. Time-to-catch/survive - Average episode length

Additional metrics:
- Mechanism cost/efficiency
- Exploitability proxy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import json
import os


@dataclass
class EpisodeMetrics:
    """Metrics collected during a single episode."""

    winner: str  # "MrX" or "Police"
    episode_length: int
    mrx_survived: bool

    # Belief tracking
    belief_cross_entropies: List[float] = field(default_factory=list)
    reveal_steps: List[int] = field(default_factory=list)

    # Mechanism costs
    total_tolls_paid: float = 0.0
    total_budget_spent: float = 0.0
    initial_budget: float = 0.0

    # Additional info
    mrx_final_position: int = -1
    police_final_positions: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "winner": self.winner,
            "episode_length": self.episode_length,
            "mrx_survived": self.mrx_survived,
            "mean_belief_ce": (
                np.mean(self.belief_cross_entropies)
                if self.belief_cross_entropies
                else 0.0
            ),
            "total_tolls_paid": self.total_tolls_paid,
            "total_budget_spent": self.total_budget_spent,
            "budget_efficiency": self.total_budget_spent / max(self.initial_budget, 1),
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple episodes."""

    num_episodes: int
    mrx_wins: int
    police_wins: int

    # Core metrics
    win_rate: float  # MrX win rate
    win_rate_std: float

    mean_episode_length: float
    episode_length_std: float

    mean_belief_ce: float  # Mean cross-entropy at reveal times
    belief_ce_std: float

    # Mechanism metrics
    mean_tolls_paid: float
    mean_budget_spent: float
    mean_budget_efficiency: float

    # Time metrics
    mean_time_to_catch: float  # Average steps when Police wins
    mean_survival_time: float  # Average steps when MrX wins

    def to_dict(self) -> dict:
        return {
            "num_episodes": self.num_episodes,
            "mrx_wins": self.mrx_wins,
            "police_wins": self.police_wins,
            "win_rate": self.win_rate,
            "win_rate_std": self.win_rate_std,
            "mean_episode_length": self.mean_episode_length,
            "episode_length_std": self.episode_length_std,
            "mean_belief_ce": self.mean_belief_ce,
            "belief_ce_std": self.belief_ce_std,
            "mean_tolls_paid": self.mean_tolls_paid,
            "mean_budget_spent": self.mean_budget_spent,
            "mean_budget_efficiency": self.mean_budget_efficiency,
            "mean_time_to_catch": self.mean_time_to_catch,
            "mean_survival_time": self.mean_survival_time,
        }


class MetricsTracker:
    """Tracks and aggregates evaluation metrics across episodes."""

    def __init__(self):
        self.episodes: List[EpisodeMetrics] = []
        self._current_episode: Optional[EpisodeMetrics] = None

    def start_episode(self, initial_budget: float = 0.0):
        """Start tracking a new episode."""
        self._current_episode = EpisodeMetrics(
            winner="",
            episode_length=0,
            mrx_survived=False,
            initial_budget=initial_budget,
        )

    def record_step(
        self,
        step: int,
        belief: np.ndarray = None,
        true_mrx_pos: int = None,
        is_reveal: bool = False,
        toll_paid: float = 0.0,
        budget_spent: float = 0.0,
    ):
        """Record metrics for a single step."""
        if self._current_episode is None:
            return

        self._current_episode.episode_length = step
        self._current_episode.total_tolls_paid += toll_paid
        self._current_episode.total_budget_spent += budget_spent

        # Record belief quality at reveal times
        if is_reveal and belief is not None and true_mrx_pos is not None:
            ce = belief_cross_entropy(belief, true_mrx_pos)
            self._current_episode.belief_cross_entropies.append(ce)
            self._current_episode.reveal_steps.append(step)

    def end_episode(
        self,
        winner: str,
        mrx_final_pos: int = -1,
        police_final_positions: List[int] = None,
    ):
        """End the current episode and record final state."""
        if self._current_episode is None:
            return

        self._current_episode.winner = winner
        self._current_episode.mrx_survived = winner == "MrX"
        self._current_episode.mrx_final_position = mrx_final_pos
        self._current_episode.police_final_positions = police_final_positions or []

        self.episodes.append(self._current_episode)
        self._current_episode = None

    def get_aggregated_metrics(self) -> AggregatedMetrics:
        """Compute aggregated metrics across all episodes."""
        if not self.episodes:
            return AggregatedMetrics(
                num_episodes=0,
                mrx_wins=0,
                police_wins=0,
                win_rate=0.5,
                win_rate_std=0.0,
                mean_episode_length=0.0,
                episode_length_std=0.0,
                mean_belief_ce=0.0,
                belief_ce_std=0.0,
                mean_tolls_paid=0.0,
                mean_budget_spent=0.0,
                mean_budget_efficiency=0.0,
                mean_time_to_catch=0.0,
                mean_survival_time=0.0,
            )

        mrx_wins = sum(1 for e in self.episodes if e.winner == "MrX")
        police_wins = sum(1 for e in self.episodes if e.winner == "Police")

        win_rates = [1.0 if e.winner == "MrX" else 0.0 for e in self.episodes]
        lengths = [e.episode_length for e in self.episodes]

        # Belief cross-entropies
        all_ces = []
        for e in self.episodes:
            all_ces.extend(e.belief_cross_entropies)

        # Time to catch (Police wins only)
        police_win_lengths = [
            e.episode_length for e in self.episodes if e.winner == "Police"
        ]
        mrx_win_lengths = [e.episode_length for e in self.episodes if e.winner == "MrX"]

        return AggregatedMetrics(
            num_episodes=len(self.episodes),
            mrx_wins=mrx_wins,
            police_wins=police_wins,
            win_rate=float(np.mean(win_rates)),
            win_rate_std=float(np.std(win_rates)),
            mean_episode_length=float(np.mean(lengths)),
            episode_length_std=float(np.std(lengths)),
            mean_belief_ce=float(np.mean(all_ces)) if all_ces else 0.0,
            belief_ce_std=float(np.std(all_ces)) if all_ces else 0.0,
            mean_tolls_paid=float(np.mean([e.total_tolls_paid for e in self.episodes])),
            mean_budget_spent=float(
                np.mean([e.total_budget_spent for e in self.episodes])
            ),
            mean_budget_efficiency=float(
                np.mean(
                    [
                        e.total_budget_spent / max(e.initial_budget, 1)
                        for e in self.episodes
                    ]
                )
            ),
            mean_time_to_catch=(
                float(np.mean(police_win_lengths)) if police_win_lengths else 0.0
            ),
            mean_survival_time=(
                float(np.mean(mrx_win_lengths)) if mrx_win_lengths else 0.0
            ),
        )

    def reset(self):
        """Clear all recorded episodes."""
        self.episodes.clear()
        self._current_episode = None

    def save(self, filepath: str):
        """Save metrics to JSON file."""
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )
        data = {
            "aggregated": self.get_aggregated_metrics().to_dict(),
            "episodes": [e.to_dict() for e in self.episodes],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load metrics from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return data


# ============================================================================
# Metric 1: Balance (Win Rate)
# ============================================================================


def compute_win_rate(
    episodes: List[EpisodeMetrics], target: float = 0.5
) -> Dict[str, float]:
    """Compute win rate metrics.

    Args:
        episodes: List of episode metrics.
        target: Target win rate (default 0.5 for balance).

    Returns:
        Dictionary with win rate statistics.
    """
    if not episodes:
        return {"win_rate": 0.5, "variance": 0.0, "distance_to_target": 0.0}

    wins = [1.0 if e.winner == "MrX" else 0.0 for e in episodes]
    win_rate = np.mean(wins)
    variance = np.var(wins)

    return {
        "win_rate": float(win_rate),
        "variance": float(variance),
        "std": float(np.std(wins)),
        "distance_to_target": float(abs(win_rate - target)),
        "mrx_wins": sum(wins),
        "police_wins": len(wins) - sum(wins),
        "total_games": len(wins),
    }


# ============================================================================
# Metric 2: Belief Quality (Cross-Entropy)
# ============================================================================


def belief_cross_entropy(belief: np.ndarray, true_index: int) -> float:
    """Compute cross-entropy between belief distribution and true position.

    Args:
        belief: Probability distribution over nodes.
        true_index: True MrX position.

    Returns:
        Cross-entropy value (lower is better).
    """
    belief = np.clip(belief, 1e-8, 1.0)
    belief = belief / belief.sum()
    return float(-np.log(belief[true_index]))


def compute_belief_quality(episodes: List[EpisodeMetrics]) -> Dict[str, float]:
    """Compute belief quality metrics across episodes.

    Args:
        episodes: List of episode metrics.

    Returns:
        Dictionary with belief quality statistics.
    """
    all_ces = []
    for e in episodes:
        all_ces.extend(e.belief_cross_entropies)

    if not all_ces:
        return {
            "mean_cross_entropy": 0.0,
            "std_cross_entropy": 0.0,
            "min_cross_entropy": 0.0,
            "max_cross_entropy": 0.0,
            "num_reveals": 0,
        }

    return {
        "mean_cross_entropy": float(np.mean(all_ces)),
        "std_cross_entropy": float(np.std(all_ces)),
        "min_cross_entropy": float(np.min(all_ces)),
        "max_cross_entropy": float(np.max(all_ces)),
        "num_reveals": len(all_ces),
    }


# ============================================================================
# Metric 3: Time-to-Catch / Survival Time
# ============================================================================


def compute_time_metrics(episodes: List[EpisodeMetrics]) -> Dict[str, float]:
    """Compute time-related metrics.

    Args:
        episodes: List of episode metrics.

    Returns:
        Dictionary with time statistics.
    """
    if not episodes:
        return {
            "mean_episode_length": 0.0,
            "mean_time_to_catch": 0.0,
            "mean_survival_time": 0.0,
        }

    all_lengths = [e.episode_length for e in episodes]
    police_win_lengths = [e.episode_length for e in episodes if e.winner == "Police"]
    mrx_win_lengths = [e.episode_length for e in episodes if e.winner == "MrX"]

    return {
        "mean_episode_length": float(np.mean(all_lengths)),
        "std_episode_length": float(np.std(all_lengths)),
        "mean_time_to_catch": (
            float(np.mean(police_win_lengths)) if police_win_lengths else 0.0
        ),
        "std_time_to_catch": (
            float(np.std(police_win_lengths)) if police_win_lengths else 0.0
        ),
        "mean_survival_time": (
            float(np.mean(mrx_win_lengths)) if mrx_win_lengths else 0.0
        ),
        "std_survival_time": float(np.std(mrx_win_lengths)) if mrx_win_lengths else 0.0,
        "min_episode_length": int(np.min(all_lengths)),
        "max_episode_length": int(np.max(all_lengths)),
    }


# ============================================================================
# Additional Metrics
# ============================================================================


def compute_mechanism_cost(episodes: List[EpisodeMetrics]) -> Dict[str, float]:
    """Compute mechanism cost metrics.

    Args:
        episodes: List of episode metrics.

    Returns:
        Dictionary with cost statistics.
    """
    if not episodes:
        return {"mean_tolls": 0.0, "mean_budget_spent": 0.0, "budget_efficiency": 0.0}

    tolls = [e.total_tolls_paid for e in episodes]
    budgets = [e.total_budget_spent for e in episodes]
    efficiencies = [e.total_budget_spent / max(e.initial_budget, 1) for e in episodes]

    return {
        "mean_tolls": float(np.mean(tolls)),
        "std_tolls": float(np.std(tolls)),
        "mean_budget_spent": float(np.mean(budgets)),
        "std_budget_spent": float(np.std(budgets)),
        "mean_budget_efficiency": float(np.mean(efficiencies)),
    }


def generate_metrics_report(
    tracker: MetricsTracker,
    experiment_name: str = "Experiment",
) -> str:
    """Generate a formatted text report of all metrics.

    Args:
        tracker: MetricsTracker with recorded episodes.
        experiment_name: Name of the experiment.

    Returns:
        Formatted report string.
    """
    agg = tracker.get_aggregated_metrics()
    win_rate = compute_win_rate(tracker.episodes)
    belief = compute_belief_quality(tracker.episodes)
    time_metrics = compute_time_metrics(tracker.episodes)
    cost = compute_mechanism_cost(tracker.episodes)

    lines = [
        "=" * 70,
        f"METRICS REPORT: {experiment_name}",
        "=" * 70,
        "",
        "📊 METRIC 1: Balance (Win Rate)",
        "-" * 40,
        f"  MrX Win Rate:        {win_rate['win_rate']:.2%} (target: 50%)",
        f"  Variance:            {win_rate['variance']:.4f}",
        f"  Distance to Target:  {win_rate['distance_to_target']:.4f}",
        f"  Games Played:        {win_rate['total_games']}",
        f"    - MrX Wins:        {int(win_rate['mrx_wins'])}",
        f"    - Police Wins:     {int(win_rate['police_wins'])}",
        "",
        "📊 METRIC 2: Belief Quality (Cross-Entropy at Reveal Times)",
        "-" * 40,
        f"  Mean Cross-Entropy:  {belief['mean_cross_entropy']:.4f}",
        f"  Std Cross-Entropy:   {belief['std_cross_entropy']:.4f}",
        f"  Min Cross-Entropy:   {belief['min_cross_entropy']:.4f}",
        f"  Max Cross-Entropy:   {belief['max_cross_entropy']:.4f}",
        f"  Total Reveals:       {belief['num_reveals']}",
        "",
        "📊 METRIC 3: Time-to-Catch / Survival Time",
        "-" * 40,
        f"  Mean Episode Length: {time_metrics['mean_episode_length']:.1f} steps",
        f"  Mean Time-to-Catch:  {time_metrics['mean_time_to_catch']:.1f} steps (Police wins)",
        f"  Mean Survival Time:  {time_metrics['mean_survival_time']:.1f} steps (MrX wins)",
        f"  Episode Range:       [{time_metrics['min_episode_length']}, {time_metrics['max_episode_length']}]",
        "",
        "📊 Additional: Mechanism Cost/Efficiency",
        "-" * 40,
        f"  Mean Tolls Paid:     {cost['mean_tolls']:.2f}",
        f"  Mean Budget Spent:   {cost['mean_budget_spent']:.2f}",
        f"  Budget Efficiency:   {cost['mean_budget_efficiency']:.2%}",
        "",
        "=" * 70,
    ]

    return "\n".join(lines)


def save_metrics_json(
    tracker: MetricsTracker, filepath: str, experiment_name: str = ""
):
    """Save all metrics to a JSON file for plotting.

    Args:
        tracker: MetricsTracker with recorded episodes.
        filepath: Output file path.
        experiment_name: Name of the experiment.
    """
    data = {
        "experiment": experiment_name,
        "aggregated": tracker.get_aggregated_metrics().to_dict(),
        "win_rate": compute_win_rate(tracker.episodes),
        "belief_quality": compute_belief_quality(tracker.episodes),
        "time_metrics": compute_time_metrics(tracker.episodes),
        "mechanism_cost": compute_mechanism_cost(tracker.episodes),
        "episodes": [e.to_dict() for e in tracker.episodes],
    }

    os.makedirs(
        os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True
    )
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
