"""
Reward calculation module for Scotland Yard environment.

This module handles all reward computation logic for both MrX and Police agents,
including distance penalties, position rewards, grouping penalties, and termination checks.
"""

import numpy as np
from typing import Dict, Tuple, List


class RewardCalculator:
    """Calculates rewards for agents in the Scotland Yard game."""

    def __init__(self, reward_weights: Dict[str, float], logger):
        """
        Initialize the reward calculator.

        Args:
            reward_weights: Dictionary of reward component weights
            logger: Logger instance for debugging
        """
        self.reward_weights = reward_weights
        self.logger = logger

    def calculate_rewards_and_terminations(
        self,
        mrx_pos: int,
        police_positions: List[int],
        timestep: int,
        epoch: int,
        is_no_money: bool,
        agents: List[str],
        distance_matrix: np.ndarray,
        get_possible_moves_func,
        node_visit_counts: Dict[int, int],
    ) -> Tuple[Dict[str, float], Dict[str, bool], Dict[str, bool], str]:
        """
        Compute rewards and check termination/truncation conditions.

        Args:
            mrx_pos: MrX's current position
            police_positions: List of police positions
            timestep: Current timestep
            epoch: Current epoch
            is_no_money: Whether police ran out of money
            agents: List of agent names
            distance_matrix: Precomputed distance matrix
            get_possible_moves_func: Function to get possible moves for position
            node_visit_counts: Dictionary tracking node visit counts

        Returns:
            Tuple of (rewards, terminations, truncations, winner)
        """
        self.logger.log(
            "Calculating rewards and checking termination conditions., ", level="debug"
        )
        terminations = {a: False for a in agents}
        truncations = {a: False for a in agents}
        rewards = {a: 0 for a in agents}
        winner = None

        if mrx_pos in police_positions:
            self.logger.log("MrX has been caught by the police., ", level="info")
            rewards = {a: (-1 if a == "MrX" else 1) for a in agents}
            terminations = {a: True for a in agents}
            winner = "Police"
        elif timestep > 250:
            self.logger.log(
                "Maximum timestep exceeded. Truncating episode., ", level="info"
            )
            rewards = {a: (1 if a == "MrX" else 0) for a in agents}
            truncations = {a: True for a in agents}
            winner = "MrX"
        elif is_no_money:
            self.logger.log("Police out of money. Truncating episode., ", level="info")
            rewards = {a: (1 if a == "MrX" else 0) for a in agents}
            terminations = {a: True for a in agents}
            winner = "MrX"
        else:
            rewards = self.calculate_rewards(
                mrx_pos=mrx_pos,
                police_positions=police_positions,
                timestep=timestep,
                epoch=epoch,
                agents=agents,
                distance_matrix=distance_matrix,
                get_possible_moves_func=get_possible_moves_func,
                node_visit_counts=node_visit_counts,
            )

        return rewards, terminations, truncations, winner

    def calculate_rewards(
        self,
        mrx_pos: int,
        police_positions: List[int],
        timestep: int,
        epoch: int,
        agents: List[str],
        distance_matrix: np.ndarray,
        get_possible_moves_func,
        node_visit_counts: Dict[int, int],
    ) -> Dict[str, float]:
        """
        Compute rewards for all agents using vectorized matrix indexing.
        """
        self.logger.log(
            "Calculating individual rewards for agents (vectorized).", level="debug"
        )
        rewards = {}

        # Vectorized slice: MrX to all police
        # distance_matrix[row_indices, col_indices]
        police_distances = distance_matrix[mrx_pos, police_positions]

        num_police = len(police_positions)
        # Vectorized sub-matrix: all police to each other
        # Use np.ix_ to slice both rows and columns by the list of positions
        police_dist_mat = distance_matrix[np.ix_(police_positions, police_positions)]

        if num_police > 1:
            mask = ~np.eye(num_police, dtype=bool)
            exp_dists = np.exp(-police_dist_mat)

            group_penalties = (exp_dists * mask).sum(axis=1)
            overlap_penalties = ((police_dist_mat <= 1) & mask).sum(axis=1)

            prox_mask = (police_dist_mat > 1) & mask
            proximity_scores = (exp_dists * prox_mask).sum(axis=1)
        else:
            group_penalties = np.zeros(num_police)
            overlap_penalties = np.zeros(num_police)
            proximity_scores = np.zeros(num_police)

        # MrX reward
        closest_distance = np.min(police_distances)
        avg_distance = np.mean(police_distances)

        position_penalty = len(get_possible_moves_func(mrx_pos, 0)[0])
        mrX_reward = (
            self.reward_weights["Mrx_closest"] * (-1 / (closest_distance + 1))
            + self.reward_weights["Mrx_average"] * (-1 / (avg_distance + 1))
            + self.reward_weights["Mrx_position"] * (position_penalty)
            + (1 - self.reward_weights["Mrx_time"]) * (0.1 * timestep)
        )
        rewards["MrX"] = mrX_reward

        # Log MrX stats
        self.logger.log_scalar("episode_step", timestep)
        self.logger.log_scalar(f"episode/epoch_{epoch}/MrX_total_reward", mrX_reward)
        self.logger.log_scalar(
            f"episode/epoch_{epoch}/average_distance_to_MrX", avg_distance
        )

        # Police rewards (vectorized over agent list)
        for i, police in enumerate(agents[1:]):
            distance_to_mrX = police_distances[i]
            police_pos = police_positions[i]

            # Coverage reward using precomputed log lookup if needed, but dict access is okay for 15 agents
            coverage_reward = np.exp(-np.log1p(node_visit_counts[police_pos]))

            police_reward = (
                self.reward_weights["Police_distance"] * (np.exp(-distance_to_mrX))
                + self.reward_weights["Police_group"] * (group_penalties[i])
                + self.reward_weights["Police_position"]
                * len(get_possible_moves_func(police_pos, i + 1)[0])
                + (1 - self.reward_weights["Police_time"]) * (0.05 * timestep)
                + self.reward_weights["Police_proximity"] * proximity_scores[i]
                - self.reward_weights["Police_overlap_penalty"] * overlap_penalties[i]
                + self.reward_weights["Police_coverage"] * coverage_reward
            )
            rewards[police] = police_reward

        return rewards
