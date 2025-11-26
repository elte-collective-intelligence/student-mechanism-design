"""Graph generator utilities for controllable Scotland Yard experiments.

This module provides a thin wrapper around the existing :class:`ConnectedGraph`
space to allow sampling graphs from configurable distributions.  The functions
are intentionally lightweight so they can be plugged into Hydra configs or
meta-learning loops without creating additional dependencies.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from Enviroment.graph_layout import ConnectedGraph


@dataclass
class GraphGenerationConfig:
    """Configuration controlling random graph sampling.

    Attributes:
        num_nodes: Number of nodes in the graph.
        num_edges: Total edges to sample.  When omitted, defaults to a tree.
        max_edges_per_node: Degree cap to avoid hubs dominating.
        max_edge_weight: Upper bound for sampled edge weights.
    """

    num_nodes: int = 12
    num_edges: Optional[int] = None
    max_edges_per_node: int = 4
    max_edge_weight: int = 5


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

    def sample(self, overrides: GraphGenerationConfig | None = None):
        cfg = overrides or self.config
        return self.space.sample(
            num_nodes=cfg.num_nodes,
            num_edges=cfg.num_edges,
            max_edges_per_node=cfg.max_edges_per_node,
        )

    def sample_with_statistics(self, overrides: GraphGenerationConfig | None = None) -> Tuple[object, dict]:
        """Return a graph and lightweight statistics for logging.

        Statistics are designed for meta-learning and OOD evaluation scripts.
        """

        graph = self.sample(overrides)
        edge_weights = np.asarray(graph.edges)
        stats = {
            "num_nodes": graph.nodes.shape[0],
            "num_edges": graph.edge_links.shape[0],
            "edge_weight_mean": float(edge_weights.mean()) if edge_weights.size else 0.0,
            "edge_weight_std": float(edge_weights.std()) if edge_weights.size else 0.0,
        }
        return graph, stats
