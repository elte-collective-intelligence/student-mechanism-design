"""Plotting helpers for correlating attention with strategic events.

Consumes the JSON sidecar written by ``run_attention_episode.py`` and produces
three plots tying attention summaries to gameplay events (reveal steps,
capture step).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(events_path: str) -> Dict[str, Any]:
    """Load the episode-events JSON."""
    with open(events_path, "r") as f:
        return json.load(f)


def _series(events: Dict[str, Any], key: str) -> np.ndarray:
    """Extract a per-step series from events['steps']."""
    return np.array([s["attention_summary"][key] for s in events["steps"]])


def _mark_events(ax, events: Dict[str, Any]) -> None:
    """Overlay reveal-step and capture-step markers on a time-series axis."""
    for r in events.get("reveal_steps", []):
        ax.axvline(r, color="orange", linestyle="--", alpha=0.5, label="_reveal")
    cap = events.get("capture_step")
    if cap is not None:
        ax.axvline(cap, color="red", linestyle="-", alpha=0.8, label="_capture")
    handles = [
        plt.Line2D([0], [0], color="orange", linestyle="--", label="reveal step"),
        plt.Line2D([0], [0], color="red", linestyle="-", label="capture step"),
    ]
    ax.legend(handles=handles, loc="best")


def plot_attention_entropy_vs_time(events_path: str, out_path: str) -> None:
    """Line plot of mean attention entropy per step with event markers."""
    events = _load(events_path)
    steps = np.array([s["step"] for s in events["steps"]])
    entropy = _series(events, "mean_entropy")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, entropy, marker="o", color="steelblue")
    ax.set_xlabel("step")
    ax.set_ylabel("attention entropy (last layer)")
    ax.set_title(f"Attention entropy vs time — winner: {events.get('winner')}")
    _mark_events(ax, events)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_attention_on_mrx_node(events_path: str, out_path: str) -> None:
    """Line plot of attention mass on edges incident to MrX/police nodes."""
    events = _load(events_path)
    steps = np.array([s["step"] for s in events["steps"]])
    mass_mrx = _series(events, "mass_on_mrx_edges")
    mass_police = _series(events, "mass_on_police_edges")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, mass_mrx, marker="o", label="edges incident to MrX node")
    ax.plot(steps, mass_police, marker="s", label="edges incident to police nodes")
    ax.set_xlabel("step")
    ax.set_ylabel("attention mass")
    ax.set_title("Attention concentration on agent nodes")
    _mark_events(ax, events)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_capture_step_heatmap(events_path: str, out_path: str) -> None:
    """Bar chart of the top attention edges at the capture (or final) step."""
    events = _load(events_path)
    steps: List[Dict[str, Any]] = events["steps"]
    if not steps:
        return
    target_idx = events.get("capture_step")
    if target_idx is None or target_idx >= len(steps):
        target_idx = len(steps) - 1
    top = steps[target_idx]["attention_summary"]["top_edges"]
    labels = [f"({e['u']}, {e['v']})" for e in top]
    weights = [e["weight"] for e in top]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(top)), weights, color="indianred")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("attention weight")
    ax.set_title(
        f"Top edges at step {target_idx} "
        f"({'capture' if events.get('capture_step') is not None else 'final'} step)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_all(events_path: str, out_dir: str) -> None:
    """Convenience: write all three plots into out_dir."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    plot_attention_entropy_vs_time(
        events_path, os.path.join(out_dir, "attention_entropy_vs_time.png")
    )
    plot_attention_on_mrx_node(
        events_path, os.path.join(out_dir, "attention_on_mrx_node.png")
    )
    plot_capture_step_heatmap(
        events_path, os.path.join(out_dir, "capture_step_top_edges.png")
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, help="Path to events JSON.")
    parser.add_argument("--out-dir", default="logs/attention/plots")
    args = parser.parse_args()

    plot_all(args.events, args.out_dir)
