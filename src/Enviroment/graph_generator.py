"""Graph generator utilities for controllable Scotland Yard experiments.

This module provides a thin wrapper around the existing :class:`ConnectedGraph`
space to allow sampling graphs from configurable distributions.  The functions
are intentionally lightweight so they can be plugged into Hydra configs or
meta-learning loops without creating additional dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
import json
import os
import random

from Enviroment.graph_layout import ConnectedGraph


@dataclass
class GraphGenerationConfig:
    """Configuration controlling random graph sampling.

    Attributes:
        num_nodes: Number of nodes in the graph.
        num_edges: Total edges to sample.  When omitted, defaults to a tree.
        max_edges_per_node: Degree cap to avoid hubs dominating.
        max_edge_weight: Upper bound for sampled edge weights.
        seed: Random seed for reproducibility. If None, a random seed is generated.
    """

    num_nodes: int = 12
    num_edges: Optional[int] = None
    max_edges_per_node: int = 4
    max_edge_weight: int = 5
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "max_edges_per_node": self.max_edges_per_node,
            "max_edge_weight": self.max_edge_weight,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GraphGenerationConfig":
        return cls(**d)


@dataclass
class GraphSample:
    """A sampled graph with its generation config and seed for reproducibility."""

    graph: object
    config: GraphGenerationConfig
    seed: int
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "seed": self.seed,
            "stats": self.stats,
        }


class GraphGenerator:
    """Utility class that centralises graph sampling.

    It reuses :class:`ConnectedGraph` to stay compatible with the existing
    environment while allowing upstream modules to override sampling statistics
    (e.g. for OOD evaluation).
    """

    def __init__(self, config: GraphGenerationConfig | None = None):
        self.config = config or GraphGenerationConfig()
        self.space = ConnectedGraph()
        # Ensure ConnectedGraph respects the configured maximum weight.
        self.space.MAX_WEIGHT = self.config.max_edge_weight
        self._generated_seeds: List[int] = []

    def sample(
        self, overrides: GraphGenerationConfig | None = None, seed: int | None = None
    ):
        """Sample a graph with optional seed for reproducibility.

        Args:
            overrides: Optional config overrides.
            seed: Random seed. If None, uses config seed or generates one.

        Returns:
            The sampled graph object.
        """
        cfg = overrides or self.config

        # Determine seed
        if seed is not None:
            actual_seed = seed
        elif cfg.seed is not None:
            actual_seed = cfg.seed
        else:
            actual_seed = np.random.randint(0, 2**31)

        # Set seeds for reproducibility
        np.random.seed(actual_seed)
        random.seed(actual_seed)
        self._generated_seeds.append(actual_seed)

        return self.space.sample(
            num_nodes=cfg.num_nodes,
            num_edges=cfg.num_edges,
            max_edges_per_node=cfg.max_edges_per_node,
        )

    def sample_with_statistics(
        self, overrides: GraphGenerationConfig | None = None, seed: int | None = None
    ) -> Tuple[object, dict]:
        """Return a graph and lightweight statistics for logging.

        Statistics are designed for meta-learning and OOD evaluation scripts.

        Args:
            overrides: Optional config overrides.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (graph, statistics dict including seed).
        """
        cfg = overrides or self.config

        # Determine seed
        if seed is not None:
            actual_seed = seed
        elif cfg.seed is not None:
            actual_seed = cfg.seed
        else:
            actual_seed = np.random.randint(0, 2**31)

        graph = self.sample(overrides, seed=actual_seed)
        edge_weights = np.asarray(graph.edges)
        stats = {
            "num_nodes": graph.nodes.shape[0],
            "num_edges": graph.edge_links.shape[0],
            "edge_weight_mean": (
                float(edge_weights.mean()) if edge_weights.size else 0.0
            ),
            "edge_weight_std": float(edge_weights.std()) if edge_weights.size else 0.0,
            "seed": actual_seed,
            "config": cfg.to_dict(),
        }
        return graph, stats

    def sample_with_record(
        self, overrides: GraphGenerationConfig | None = None, seed: int | None = None
    ) -> GraphSample:
        """Sample a graph and return a full record for reproducibility.

        Args:
            overrides: Optional config overrides.
            seed: Random seed.

        Returns:
            GraphSample object containing graph, config, seed, and stats.
        """
        cfg = overrides or self.config
        graph, stats = self.sample_with_statistics(overrides, seed)
        return GraphSample(
            graph=graph,
            config=cfg,
            seed=stats["seed"],
            stats=stats,
        )

    def save_seeds(self, filepath: str):
        """Save all generated seeds to a JSON file for reproducibility.

        Args:
            filepath: Path to save the seeds file.
        """
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )
        with open(filepath, "w") as f:
            json.dump(
                {
                    "seeds": self._generated_seeds,
                    "config": self.config.to_dict(),
                },
                f,
                indent=2,
            )

    def load_seeds(self, filepath: str) -> List[int]:
        """Load seeds from a JSON file.

        Args:
            filepath: Path to the seeds file.

        Returns:
            List of seeds.
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return data.get("seeds", [])

    def clear_seed_history(self):
        """Clear the recorded seed history."""
        self._generated_seeds.clear()
