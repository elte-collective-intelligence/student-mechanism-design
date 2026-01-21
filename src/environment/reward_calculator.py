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
        get_distance_func,
        get_possible_moves_func,
        node_visit_counts: Dict[int, int]
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
            get_distance_func: Function to compute distance between nodes
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
                get_distance_func=get_distance_func,
                get_possible_moves_func=get_possible_moves_func,
                node_visit_counts=node_visit_counts
            )
        
        return rewards, terminations, truncations, winner
    
    def calculate_rewards(
        self,
        mrx_pos: int,
        police_positions: List[int],
        timestep: int,
        epoch: int,
        agents: List[str],
        get_distance_func,
        get_possible_moves_func,
        node_visit_counts: Dict[int, int]
    ) -> Dict[str, float]:
        """
        Compute rewards for all agents based on the specified components.
        Rewards are weighted by the reward_weights parameters.
        
        Args:
            mrx_pos: MrX's current position
            police_positions: List of police positions
            timestep: Current timestep
            epoch: Current epoch
            agents: List of agent names
            get_distance_func: Function to compute distance between nodes
            get_possible_moves_func: Function to get possible moves for position
            node_visit_counts: Dictionary tracking node visit counts
            
        Returns:
            Dictionary mapping agent names to their rewards
        """
        self.logger.log("Calculating individual rewards for agents., ", level="debug")
        rewards = {}
        
        # Compute rewards for MrX
        police_distances = [
            get_distance_func(mrx_pos, police_pos)
            for police_pos in police_positions
        ]
        self.logger.log(
            f"Police distances from MrX: {police_distances}, ", level="debug"
        )
        closest_distance = min(police_distances)
        avg_distance = np.mean(police_distances)
        self.logger.log(
            f"MrX closest distance: {closest_distance}, average distance: {avg_distance}, ",
            level="debug",
        )
        position_penalty = len(get_possible_moves_func(mrx_pos, 0)[0])
        mrX_reward = (
            self.reward_weights["Mrx_closest"]
            * (-1 / (closest_distance + 1))  # Distance penalty
            + self.reward_weights["Mrx_average"]
            * (-1 / (avg_distance + 1))  # Average distance penalty
            + self.reward_weights["Mrx_position"]
            * (position_penalty)  # Position reward
            + (1 - self.reward_weights["Mrx_time"])
            * (0.1 * timestep)  # Time reward
        )
        rewards["MrX"] = mrX_reward
        self.logger.log(f"MrX reward: {mrX_reward}, ", level="debug")
        
        # Log MrX reward components as scalars
        distance_penalty_mrX = -1 / (closest_distance + 1)
        avg_distance_penalty_mrX = -1 / (avg_distance + 1)
        time_reward_mrX = 0.1 * timestep
        
        self.logger.log_scalar("episode_step", timestep)
        self.logger.log_scalar(
            f"episode/epoch_{epoch}/MrX_distance_penalty", distance_penalty_mrX
        )
        self.logger.log_scalar(
            f"episode/epoch_{epoch}/MrX_avg_distance_penalty",
            avg_distance_penalty_mrX,
        )
        self.logger.log_scalar(
            f"episode/epoch_{epoch}/MrX_time_reward", time_reward_mrX
        )
        self.logger.log_scalar(
            f"episode/epoch_{epoch}/MrX_total_reward", mrX_reward
        )
        self.logger.log_scalar(
            f"episode/epoch_{epoch}/average_distance_to_MrX", avg_distance
        )
        
        police_dist_sum = 0.0
        for police_pos_1 in police_positions:
            for police_pos_2 in police_positions:
                police_dist_sum += get_distance_func(police_pos_1, police_pos_2)
        self.logger.log_scalar(
            f"episode/epoch_{epoch}/average_distance_between_officers",
            police_dist_sum / (len(police_positions) ** 2),
        )
        
        # Compute rewards for police
        for i, police in enumerate(agents[1:]):  # Skip MrX
            police_pos = police_positions[i]
            distance_to_mrX = get_distance_func(police_pos, mrx_pos)
            group_penalty = sum(
                np.exp(-get_distance_func(police_pos, other_police_pos))
                for j, other_police_pos in enumerate(police_positions)
                if i != j
            )
            position_penalty = len(get_possible_moves_func(police_pos, i)[0])
            
            overlap_penalty = sum(
                1.0
                for j, other_police_pos in enumerate(police_positions)
                if i != j and get_distance_func(police_pos, other_police_pos) <= 1
            )
            
            proximity_score = sum(
                np.exp(-get_distance_func(police_pos, other_police_pos))
                for j, other_police_pos in enumerate(police_positions)
                if i != j and get_distance_func(police_pos, other_police_pos) > 1
            )
            
            visit_count = node_visit_counts[police_pos]
            coverage_reward = np.exp(-np.log1p(visit_count))
            
            self.logger.log(
                f"{police} distance to MrX: {distance_to_mrX}, group penalty: {group_penalty}, position penalty: {position_penalty}, overlap penalty: {overlap_penalty}, proximity score: {proximity_score}, ",
                level="debug",
            )
            
            police_reward = (
                self.reward_weights["Police_distance"]
                * (np.exp(-distance_to_mrX))  # Distance reward
                + self.reward_weights["Police_group"]
                * (group_penalty)  # Grouping penalty
                + self.reward_weights["Police_position"]
                * (position_penalty)  # Position reward
                + (1 - self.reward_weights["Police_time"])
                * (0.05 * timestep)  # Time penalty
                + self.reward_weights["Police_proximity"]
                * proximity_score  # Proximity reward
                - self.reward_weights["Police_overlap_penalty"]
                * overlap_penalty  # Overlap penalty
                + self.reward_weights["Police_coverage"]
                * coverage_reward  # Coverage reward
            )
            rewards[police] = police_reward
            self.logger.log(f"{police} reward: {police_reward}, ", level="debug")
            
            # Log Police reward components as scalars
            distance_reward_police = np.exp(-distance_to_mrX)
            grouping_penalty_police = group_penalty
            position_reward_police = position_penalty
            time_penalty_police = 0.05 * timestep
            
            self.logger.log_scalar(
                f"episode/epoch_{epoch}/{police}_distance_reward",
                distance_reward_police,
            )
            self.logger.log_scalar(
                f"episode/epoch_{epoch}/{police}_grouping_penalty",
                grouping_penalty_police,
            )
            self.logger.log_scalar(
                f"episode/epoch_{epoch}/{police}_position_reward",
                position_reward_police,
            )
            self.logger.log_scalar(
                f"episode/epoch_{epoch}/{police}_time_penalty", time_penalty_police
            )
            self.logger.log_scalar(
                f"episode/epoch_{epoch}/{police}_total_reward", police_reward
            )
            self.logger.log_scalar(
                f"episode/epoch_{epoch}/{police}_coverage_reward", coverage_reward
            )
            self.logger.log_scalar(
                f"episode/epoch_{epoch}/{police}_proximity_score", proximity_score
            )
            self.logger.log_scalar(
                f"episode/epoch_{epoch}/{police}_overlap_penalty", overlap_penalty
            )
        
        self.logger.log(f"All rewards calculated: {rewards}, ", level="debug")
        return rewards
