"""OOD evaluation utilities for graph generalisation."""

from __future__ import annotations

from typing import Callable, Dict, Sequence

from Enviroment.graph_generator import GraphGenerationConfig, GraphGenerator


def evaluate_on_graph_distribution(build_env: Callable[[object], object], configs: Sequence[GraphGenerationConfig]):
    results = []
    generator = GraphGenerator()
    for cfg in configs:
        graph, stats = generator.sample_with_statistics(cfg)
        env = build_env(graph)
        results.append({"config": cfg, "stats": stats, "env": env})
    return results
