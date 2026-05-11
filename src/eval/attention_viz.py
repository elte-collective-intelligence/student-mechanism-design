"""Attention visualization helpers for GAT and Transformer agents.

Consumes the (edge_index, alpha) tuples returned by ``agent.get_attention(graph)``
and renders them as edge-weight overlays on the same graph layout used by
``GameVisualizer``. By default, only the last layer's attention is rendered,
averaged across heads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def aggregate_attention(
    edge_index: torch.Tensor,
    alpha: torch.Tensor,
    num_nodes: int,
    drop_self_loops: bool = True,
) -> Dict[Tuple[int, int], float]:
    """Reduce a single layer's attention to one weight per undirected edge."""
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if alpha.dim() != 2 or alpha.shape[0] != edge_index.shape[1]:
        raise ValueError("alpha must have shape [E, heads].")

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    weights = alpha.mean(dim=-1).cpu().numpy()

    agg: Dict[Tuple[int, int], List[float]] = {}
    for u, v, w in zip(src, dst, weights):
        u, v = int(u), int(v)
        if drop_self_loops and u == v:
            continue
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            continue
        key = (u, v) if u < v else (v, u)
        agg.setdefault(key, []).append(float(w))

    return {edge: float(np.mean(ws)) for edge, ws in agg.items()}


def compute_attention_summary(
    edge_index: torch.Tensor,
    alpha: torch.Tensor,
    mrx_pos: int,
    police_positions: List[int],
    num_nodes: int,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Compute per-step scalar summaries of a single attention layer."""
    weights = aggregate_attention(edge_index, alpha, num_nodes=num_nodes)
    if not weights:
        return {
            "mean_entropy": 0.0,
            "mass_on_mrx_edges": 0.0,
            "mass_on_police_edges": 0.0,
            "top_edges": [],
        }

    edges = list(weights.keys())
    vals = np.array([weights[e] for e in edges], dtype=np.float64)
    total = vals.sum()
    probs = vals / total if total > 1e-12 else np.full_like(vals, 1.0 / len(vals))
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    mrx = int(mrx_pos)
    police = {int(p) for p in police_positions}

    mass_mrx = float(sum(w for (u, v), w in weights.items() if u == mrx or v == mrx))
    mass_police = float(
        sum(w for (u, v), w in weights.items() if u in police or v in police)
    )

    order = np.argsort(-vals)[:top_k]
    top_edges = [
        {"u": int(edges[i][0]), "v": int(edges[i][1]), "weight": float(vals[i])}
        for i in order
    ]

    return {
        "mean_entropy": entropy,
        "mass_on_mrx_edges": mass_mrx,
        "mass_on_police_edges": mass_police,
        "top_edges": top_edges,
    }


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def render_attention_frame(
    board,
    mrx_pos: Optional[int],
    police_positions: Optional[List[int]],
    edge_index: torch.Tensor,
    alpha: torch.Tensor,
    title: str = "",
    layout: Optional[Dict[int, Tuple[float, float]]] = None,
    cmap_name: str = "viridis",
    min_width: float = 0.5,
    max_width: float = 6.0,
) -> np.ndarray:
    """Render one frame with attention as edge color and width."""
    num_nodes = board.nodes.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(tuple(e) for e in board.edge_links)

    if layout is None:
        layout = nx.kamada_kawai_layout(G)

    attn = aggregate_attention(edge_index, alpha, num_nodes=num_nodes)
    edge_list = list(G.edges())
    edge_weights = np.array(
        [attn.get((min(u, v), max(u, v)), 0.0) for u, v in edge_list]
    )
    norm = _normalize(edge_weights)
    widths = (min_width + (max_width - min_width) * norm).tolist()
    cmap = plt.get_cmap(cmap_name)
    edge_colors = [cmap(float(w)) for w in norm]

    node_colors = ["gray"] * num_nodes
    if mrx_pos is not None:
        node_colors[int(mrx_pos)] = "red"
    if police_positions is not None:
        for p in police_positions:
            node_colors[int(p)] = "blue"

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title or "Attention Overlay", fontsize=16)
    nx.draw_networkx_nodes(
        G, layout, ax=ax, node_size=700, node_color=node_colors, edgecolors="black"
    )
    nx.draw_networkx_edges(
        G, layout, ax=ax, edgelist=edge_list, width=widths, edge_color=edge_colors
    )
    nx.draw_networkx_labels(
        G, layout, ax=ax, font_size=10, font_family="sans-serif", font_color="white"
    )
    ax.legend(
        handles=[
            mpatches.Patch(color="red", label="MrX"),
            mpatches.Patch(color="blue", label="Police"),
        ],
        loc="upper right",
    )
    ax.axis("off")

    fig.canvas.draw()
    buffer = fig.canvas.tostring_argb()
    w, h = fig.canvas.get_width_height()
    image = np.frombuffer(buffer, dtype=np.uint8).reshape(h, w, 4)[:, :, 1:].copy()
    plt.close(fig)
    return image


def save_attention_gif(
    frames: List[np.ndarray], path: str, interval: int = 400
) -> None:
    """Save a list of frames as a GIF."""
    if not frames:
        return
    fig, ax = plt.subplots()
    ax.axis("off")
    img = ax.imshow(frames[0], animated=True)

    def update(i):
        img.set_array(frames[i])
        return (img,)

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=interval, blit=True, repeat_delay=10
    )
    anim.save(path)
    plt.close(fig)
