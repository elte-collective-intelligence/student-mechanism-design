import math
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from eval.attention_viz import (  # noqa: E402
    aggregate_attention,
    compute_attention_summary,
)


def test_aggregate_attention_mean_over_heads():
    """Mean is taken over heads, both directions of an undirected edge merged."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
    )
    alpha = torch.tensor(
        [
            [0.4, 0.6],  # 0->1, mean 0.5
            [0.2, 0.4],  # 1->0, mean 0.3
            [0.8, 1.0],  # 1->2, mean 0.9
            [0.6, 0.4],  # 2->1, mean 0.5
            [0.1, 0.3],  # 2->0, mean 0.2
            [0.5, 0.7],  # 0->2, mean 0.6
        ]
    )

    weights = aggregate_attention(edge_index, alpha, num_nodes=3)

    assert set(weights.keys()) == {(0, 1), (1, 2), (0, 2)}
    assert abs(weights[(0, 1)] - 0.4) < 1e-6  # mean(0.5, 0.3)
    assert abs(weights[(1, 2)] - 0.7) < 1e-6  # mean(0.9, 0.5)
    assert abs(weights[(0, 2)] - 0.4) < 1e-6  # mean(0.6, 0.2)


def test_aggregate_attention_drops_self_loops():
    """Self-loops are dropped by default (GATConv adds them)."""
    edge_index = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]], dtype=torch.long)
    alpha = torch.tensor([[0.5], [0.3], [0.9], [0.7]])

    weights = aggregate_attention(edge_index, alpha, num_nodes=2)

    assert set(weights.keys()) == {(0, 1)}
    assert abs(weights[(0, 1)] - 0.4) < 1e-6


def test_compute_attention_summary_basic():
    """Summary contains expected keys, top_edges sorted desc, mass tallies correct."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
    )
    alpha = torch.tensor([[0.4], [0.4], [0.9], [0.5], [0.1], [0.7]])  # single head
    summary = compute_attention_summary(
        edge_index, alpha, mrx_pos=0, police_positions=[2], num_nodes=3, top_k=3
    )

    assert set(summary.keys()) == {
        "mean_entropy",
        "mass_on_mrx_edges",
        "mass_on_police_edges",
        "top_edges",
    }
    assert summary["mean_entropy"] > 0
    weights_desc = [e["weight"] for e in summary["top_edges"]]
    assert weights_desc == sorted(weights_desc, reverse=True)

    # Edges incident to MrX (node 0): (0,1) and (0,2) → 0.4 + 0.4 = 0.8
    assert abs(summary["mass_on_mrx_edges"] - 0.8) < 1e-6
    # Edges incident to police (node 2): (1,2) and (0,2) → 0.7 + 0.4 = 1.1
    assert abs(summary["mass_on_police_edges"] - 1.1) < 1e-6


def test_compute_attention_summary_entropy_uniform():
    """Uniform attention should give entropy = log(num_edges)."""
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)
    alpha = torch.tensor([[0.5], [0.5], [0.5], [0.5]])
    summary = compute_attention_summary(
        edge_index, alpha, mrx_pos=0, police_positions=[1, 2], num_nodes=3
    )
    # Two undirected edges, each with weight 0.5 → uniform → entropy = log(2)
    assert abs(summary["mean_entropy"] - math.log(2)) < 1e-6
