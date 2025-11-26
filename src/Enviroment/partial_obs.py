"""Partial observability wrapper with belief tracking.

The wrapper hides MrX's position from the police except at configurable reveal
intervals.  It optionally injects noisy hints and a belief map into the
police observations.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from Enviroment.base_env import BaseEnvironment
from Enviroment.belief_module import ParticleBeliefTracker, LearnedBeliefEncoder


class PartialObservationWrapper(BaseEnvironment):
    def __init__(
        self,
        env: BaseEnvironment,
        reveal_interval: int = 5,
        reveal_probability: float = 0.0,
        noisy_hint_prob: float = 0.2,
        use_learned_belief: bool = False,
    ):
        super().__init__()
        self.env = env
        self.reveal_interval = reveal_interval
        self.reveal_probability = reveal_probability
        self.noisy_hint_prob = noisy_hint_prob
        self.use_learned_belief = use_learned_belief
        self.step_count = 0
        self.belief_tracker: ParticleBeliefTracker | None = None
        self.belief_encoder: LearnedBeliefEncoder | None = None

    @property
    def agents(self):  # passthrough for PettingZoo compatibility
        return self.env.agents

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        num_nodes = observations["MrX"]["adjacency_matrix"].shape[0]
        self.belief_tracker = ParticleBeliefTracker(num_nodes=num_nodes)
        self.belief_encoder = LearnedBeliefEncoder(observations["MrX"]["node_features"].shape[1])
        self.step_count = 0
        wrapped_obs = self._wrap_observations(observations)
        return wrapped_obs, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.step_count += 1
        wrapped_obs = self._wrap_observations(observations)
        return wrapped_obs, rewards, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def action_space(self, agent):
        return self.env.action_space(agent)

    def _wrap_observations(self, observations: Dict[str, dict]):
        reveal = self._should_reveal()
        mrx_pos = observations["MrX"].get("MrX_pos")
        adjacency = observations["MrX"].get("adjacency_matrix")
        hint = self._generate_hint(adjacency, mrx_pos) if adjacency is not None else None

        if self.belief_tracker is None and adjacency is not None:
            self.belief_tracker = ParticleBeliefTracker(num_nodes=adjacency.shape[0])
        belief_map = None
        if self.belief_tracker is not None and adjacency is not None:
            belief_map = self.belief_tracker.update(adjacency, hint, reveal if reveal else None)

        for agent, obs in observations.items():
            if agent.startswith("Police"):
                obs = dict(obs)
                if reveal:
                    obs["MrX_revealed"] = mrx_pos
                else:
                    obs["MrX_revealed"] = None
                obs["belief_map"] = belief_map
                obs["hint"] = hint
                # Hide true position when not revealed
                obs.pop("MrX_pos", None)
                observations[agent] = obs
            else:
                observations[agent] = obs
        return observations

    def _should_reveal(self) -> bool:
        if self.reveal_interval <= 0:
            return False
        deterministic = self.step_count % self.reveal_interval == 0 and self.step_count > 0
        stochastic = np.random.random() < self.reveal_probability
        return deterministic or stochastic

    def _generate_hint(self, adjacency: np.ndarray, mr_x_pos: int) -> List[int] | None:
        if mr_x_pos is None or self.noisy_hint_prob <= 0:
            return None
        neighbors = np.nonzero(adjacency[mr_x_pos])[0].tolist()
        if np.random.random() < self.noisy_hint_prob and neighbors:
            return neighbors[: min(len(neighbors), 3)]
        return None
