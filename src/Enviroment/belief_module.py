"""Belief tracking utilities for partial observability.

The module implements two simple belief trackers:
- :class:`ParticleBeliefTracker` keeps a discrete distribution using a
  lightweight particle filter over the graph.
- :class:`LearnedBeliefEncoder` is a placeholder neural encoder that can be
  swapped with a learned model during training but falls back to a deterministic
  feature-based prior for unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BeliefState:
    particles: np.ndarray
    weights: np.ndarray
    num_nodes: int

    def distribution(self) -> np.ndarray:
        """Aggregate particle weights into a node-level distribution."""
        if self.weights.sum() == 0:
            particle_weights = np.ones_like(self.weights) / len(self.weights)
        else:
            particle_weights = self.weights / self.weights.sum()
        node_probs = np.bincount(self.particles, weights=particle_weights, minlength=self.num_nodes)
        if node_probs.sum() == 0:
            return np.ones(self.num_nodes) / self.num_nodes
        return node_probs / node_probs.sum()


class ParticleBeliefTracker:
    """Minimal particle filter for tracking MrX over a graph."""

    def __init__(self, num_nodes: int, num_particles: int = 128, rng: np.random.Generator | None = None):
        self.num_nodes = num_nodes
        self.num_particles = num_particles
        self.rng = rng or np.random.default_rng()
        particles = self.rng.integers(0, num_nodes, size=num_particles)
        weights = np.ones(num_particles) / num_particles
        self.state = BeliefState(particles=particles, weights=weights, num_nodes=num_nodes)

    def reset(self, mr_x_position: int | None = None):
        particles = self.rng.integers(0, self.num_nodes, size=self.num_particles)
        if mr_x_position is not None:
            particles[:] = mr_x_position
        self.state = BeliefState(particles=particles, weights=np.ones(self.num_particles) / self.num_particles, num_nodes=self.num_nodes)

    def update(self, adjacency: np.ndarray, observation_hint: List[int] | None = None, reveal: int | None = None) -> np.ndarray:
        """Advance the belief one step.

        Args:
            adjacency: binary adjacency matrix of the graph.
            observation_hint: optional list of candidate nodes received from
                noisy sensors.
            reveal: when provided the belief collapses to a delta on that node.
        Returns:
            Probability distribution over nodes (np.ndarray with shape [num_nodes]).
        """

        if reveal is not None:
            self.reset(reveal)
            return self.state.distribution()

        # Propagate particles uniformly over neighbours including staying put.
        new_particles = []
        for particle in self.state.particles:
            neighbors = np.nonzero(adjacency[particle])[0].tolist()
            if neighbors:
                next_pos = self.rng.choice(neighbors)
            else:
                next_pos = particle
            new_particles.append(next_pos)
        particles = np.asarray(new_particles)
        weights = np.copy(self.state.weights)

        if observation_hint:
            hint_mask = np.zeros(self.num_nodes)
            hint_mask[observation_hint] = 1.0
            likelihoods = 0.1 + 0.9 * hint_mask[particles]
            weights *= likelihoods

        self.state = BeliefState(particles=particles, weights=weights, num_nodes=self.num_nodes)
        return self.state.distribution()


class LearnedBeliefEncoder(nn.Module):
    """Shallow encoder producing a belief map from node features.

    The model is intentionally tiny to keep unit tests fast while exposing a
    consistent interface for future learned models.
    """

    def __init__(self, in_features: int, hidden_size: int = 32):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.proj(node_features))
        logits = self.head(x).squeeze(-1)
        return torch.softmax(logits, dim=-1)

    def predict(self, node_features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(node_features, dtype=torch.float32)
            probs = self.forward(tensor)
        return probs.cpu().numpy()
