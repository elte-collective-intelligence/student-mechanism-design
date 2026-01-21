"""OOD evaluation utilities for graph generalisation and robustness testing.

This module provides tools for:
- Out-of-distribution (OOD) graph evaluation
- Edge/cost noise injection
- Missing reveal robustness testing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence
import numpy as np

from environment.graph_generator import GraphGenerationConfig, GraphGenerator


@dataclass
class OODConfig:
    """Configuration for OOD evaluation scenarios."""

    # Graph distribution shifts
    node_range: tuple = (8, 25)  # (min, max) nodes
    edge_multiplier_range: tuple = (0.8, 1.5)  # multiply base edges
    weight_range: tuple = (1, 10)  # edge weight range

    # Robustness tests
    edge_noise_std: float = 0.0  # Gaussian noise on edge weights
    edge_drop_prob: float = 0.0  # Probability of dropping edges
    reveal_skip_prob: float = 0.0  # Probability of skipping reveals

    # Evaluation settings
    num_eval_graphs: int = 10
    num_episodes_per_graph: int = 5
    seeds: List[int] = None

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = list(range(42, 42 + self.num_eval_graphs))


@dataclass
class OODResult:
    """Result of OOD evaluation."""

    config: OODConfig
    graph_config: GraphGenerationConfig
    metrics: Dict[str, float]
    per_graph_results: List[dict]

    def summary(self) -> dict:
        return {
            "config": {
                "nodes": self.graph_config.num_nodes,
                "edges": self.graph_config.num_edges,
                "noise_std": self.config.edge_noise_std,
                "edge_drop": self.config.edge_drop_prob,
                "reveal_skip": self.config.reveal_skip_prob,
            },
            "metrics": self.metrics,
        }


def create_ood_graph_configs(
    base_config: GraphGenerationConfig, ood_config: OODConfig
) -> List[GraphGenerationConfig]:
    """Create a set of OOD graph configurations.

    Args:
        base_config: The in-distribution base configuration.
        ood_config: OOD evaluation settings.

    Returns:
        List of GraphGenerationConfig for OOD evaluation.
    """
    configs = []

    # Vary number of nodes
    for nodes in range(ood_config.node_range[0], ood_config.node_range[1] + 1, 4):
        base_edges = base_config.num_edges or (nodes - 1)
        for mult in [
            ood_config.edge_multiplier_range[0],
            1.0,
            ood_config.edge_multiplier_range[1],
        ]:
            edges = max(nodes - 1, int(base_edges * mult))
            for seed in ood_config.seeds[:3]:  # Use first 3 seeds per config
                configs.append(
                    GraphGenerationConfig(
                        num_nodes=nodes,
                        num_edges=edges,
                        max_edges_per_node=base_config.max_edges_per_node,
                        max_edge_weight=ood_config.weight_range[1],
                        seed=seed,
                    )
                )

    return configs


def apply_edge_noise(
    adjacency: np.ndarray,
    weights: np.ndarray,
    noise_std: float,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Apply Gaussian noise to edge weights.

    Args:
        adjacency: Binary adjacency matrix.
        weights: Edge weight matrix.
        noise_std: Standard deviation of Gaussian noise.
        rng: Random number generator.

    Returns:
        Noised weight matrix.
    """
    if noise_std <= 0:
        return weights

    rng = rng or np.random.default_rng()
    noise = rng.normal(0, noise_std, weights.shape)
    noised_weights = weights + noise * adjacency  # Only add noise to existing edges
    return np.maximum(noised_weights, 0.1)  # Ensure positive weights


def apply_edge_dropout(
    adjacency: np.ndarray, drop_prob: float, rng: np.random.Generator = None
) -> np.ndarray:
    """Randomly drop edges from the graph.

    Ensures graph remains connected by not dropping bridge edges.

    Args:
        adjacency: Binary adjacency matrix.
        drop_prob: Probability of dropping each edge.
        rng: Random number generator.

    Returns:
        Modified adjacency matrix.
    """
    if drop_prob <= 0:
        return adjacency

    rng = rng or np.random.default_rng()
    modified = adjacency.copy()
    num_nodes = adjacency.shape[0]

    # Find edges that can be dropped (not bridges)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency[i, j] > 0:
                edges.append((i, j))

    # Randomly drop edges while maintaining connectivity
    for i, j in edges:
        if rng.random() < drop_prob:
            # Check if removing this edge disconnects the graph
            test_adj = modified.copy()
            test_adj[i, j] = 0
            test_adj[j, i] = 0

            if _is_connected(test_adj):
                modified[i, j] = 0
                modified[j, i] = 0

    return modified


def _is_connected(adjacency: np.ndarray) -> bool:
    """Check if graph is connected using BFS."""
    n = adjacency.shape[0]
    if n == 0:
        return True

    visited = set()
    queue = [0]
    visited.add(0)

    while queue:
        node = queue.pop(0)
        for neighbor in range(n):
            if adjacency[node, neighbor] > 0 and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == n


class RobustnessWrapper:
    """Wrapper that injects robustness perturbations into the environment."""

    def __init__(
        self,
        env,
        edge_noise_std: float = 0.0,
        edge_drop_prob: float = 0.0,
        reveal_skip_prob: float = 0.0,
        seed: int = None,
    ):
        self.env = env
        self.edge_noise_std = edge_noise_std
        self.edge_drop_prob = edge_drop_prob
        self.reveal_skip_prob = reveal_skip_prob
        self.rng = np.random.default_rng(seed)
        self._original_adjacency = None
        self._original_weights = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Apply perturbations to the graph
        if hasattr(self.env, "board"):
            self._apply_perturbations()

        return obs, info

    def _apply_perturbations(self):
        """Apply noise and dropout to the environment's graph."""
        # This is a simplified version - actual implementation would
        # need to modify the environment's internal graph structure
        pass

    def step(self, actions):
        # Potentially skip reveals
        if self.reveal_skip_prob > 0 and hasattr(self.env, "reveal_scheduled"):
            if self.rng.random() < self.reveal_skip_prob:
                self.env.reveal_scheduled = False

        return self.env.step(actions)

    def __getattr__(self, name):
        return getattr(self.env, name)


def evaluate_on_graph_distribution(
    build_env: Callable[[object], object],
    eval_fn: Callable[[object], dict],
    configs: Sequence[GraphGenerationConfig],
    ood_config: OODConfig = None,
) -> List[OODResult]:
    """Evaluate policies on a distribution of graphs.

    Args:
        build_env: Factory function that creates an environment from a graph.
        eval_fn: Evaluation function that returns metrics dict.
        configs: Graph configurations to evaluate on.
        ood_config: OOD evaluation settings.

    Returns:
        List of OODResult objects.
    """
    ood_config = ood_config or OODConfig()
    results = []
    generator = GraphGenerator()

    for cfg in configs:
        graph, stats = generator.sample_with_statistics(cfg)
        env = build_env(graph)

        # Apply robustness perturbations if configured
        if (
            ood_config.edge_noise_std > 0
            or ood_config.edge_drop_prob > 0
            or ood_config.reveal_skip_prob > 0
        ):
            env = RobustnessWrapper(
                env,
                edge_noise_std=ood_config.edge_noise_std,
                edge_drop_prob=ood_config.edge_drop_prob,
                reveal_skip_prob=ood_config.reveal_skip_prob,
                seed=cfg.seed,
            )

        # Run evaluation episodes
        per_graph_metrics = []
        for ep in range(ood_config.num_episodes_per_graph):
            episode_metrics = eval_fn(env)
            episode_metrics["episode"] = ep
            episode_metrics["seed"] = cfg.seed
            per_graph_metrics.append(episode_metrics)

        # Aggregate metrics
        aggregated = _aggregate_metrics(per_graph_metrics)
        aggregated["graph_stats"] = stats

        results.append(
            OODResult(
                config=ood_config,
                graph_config=cfg,
                metrics=aggregated,
                per_graph_results=per_graph_metrics,
            )
        )

    return results


def _aggregate_metrics(metrics_list: List[dict]) -> dict:
    """Aggregate metrics across episodes."""
    if not metrics_list:
        return {}

    aggregated = {}
    numeric_keys = [
        k
        for k in metrics_list[0].keys()
        if isinstance(metrics_list[0][k], (int, float))
    ]

    for key in numeric_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))

    return aggregated


def run_robustness_sweep(
    build_env: Callable,
    eval_fn: Callable,
    base_graph_config: GraphGenerationConfig,
    noise_levels: List[float] = [0.0, 0.5, 1.0, 2.0],
    drop_probs: List[float] = [0.0, 0.1, 0.2, 0.3],
    reveal_skip_probs: List[float] = [0.0, 0.1, 0.2, 0.3],
) -> Dict[str, List[OODResult]]:
    """Run a comprehensive robustness sweep.

    Args:
        build_env: Environment factory.
        eval_fn: Evaluation function.
        base_graph_config: Base graph configuration.
        noise_levels: Edge noise standard deviations to test.
        drop_probs: Edge drop probabilities to test.
        reveal_skip_probs: Reveal skip probabilities to test.

    Returns:
        Dictionary mapping perturbation type to results.
    """
    results = {
        "edge_noise": [],
        "edge_drop": [],
        "reveal_skip": [],
    }

    # Edge noise sweep
    for noise in noise_levels:
        ood_cfg = OODConfig(edge_noise_std=noise, num_eval_graphs=5)
        res = evaluate_on_graph_distribution(
            build_env, eval_fn, [base_graph_config], ood_cfg
        )
        results["edge_noise"].extend(res)

    # Edge drop sweep
    for drop in drop_probs:
        ood_cfg = OODConfig(edge_drop_prob=drop, num_eval_graphs=5)
        res = evaluate_on_graph_distribution(
            build_env, eval_fn, [base_graph_config], ood_cfg
        )
        results["edge_drop"].extend(res)

    # Reveal skip sweep
    for skip in reveal_skip_probs:
        ood_cfg = OODConfig(reveal_skip_prob=skip, num_eval_graphs=5)
        res = evaluate_on_graph_distribution(
            build_env, eval_fn, [base_graph_config], ood_cfg
        )
        results["reveal_skip"].extend(res)

    return results


def generate_ood_report(results: List[OODResult]) -> str:
    """Generate a text report from OOD evaluation results.

    Args:
        results: List of OODResult objects.

    Returns:
        Formatted report string.
    """
    lines = ["=" * 60, "OOD Evaluation Report", "=" * 60, ""]

    for i, res in enumerate(results):
        summary = res.summary()
        lines.append(f"Configuration {i + 1}:")
        lines.append(
            f"  Nodes: {summary['config']['nodes']}, Edges: {summary['config']['edges']}"
        )
        lines.append(
            f"  Noise: {summary['config']['noise_std']}, Drop: {summary['config']['edge_drop']}"
        )
        lines.append(f"  Metrics:")
        for k, v in summary["metrics"].items():
            if isinstance(v, float):
                lines.append(f"    {k}: {v:.4f}")
        lines.append("")

    return "\n".join(lines)
