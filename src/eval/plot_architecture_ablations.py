"""Generate publication-quality plots for the architecture ablation study.

Reads `ablation_results.csv` produced by `architecture_ablations.py` and
creates 5-6 PNG files in `src/artifacts/semester_contribution/plots/`.

Usage:
    python src/eval/plot_architecture_ablations.py
    python src/eval/plot_architecture_ablations.py --results path/to/ablation_results.csv
    python src/eval/plot_architecture_ablations.py --attention  # also generate heatmaps
    python src/eval/plot_architecture_ablations.py --tb_logs logs/gnn_experiment:gnn logs/gat_experiment:gat
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from environment.yard import CustomEnvironment
from training.utils import create_graph_data, device
from eval.architecture_ablations import build_agent, _NULL_REWARD_WEIGHTS

# seaborn is optional — gracefully degrade to plain matplotlib
try:
    import seaborn as sns

    _HAS_SNS = True
    sns.set_theme(style="whitegrid", font_scale=1.1)
except ImportError:
    _HAS_SNS = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_ARCH_COLORS = {"gnn": "#4C72B0", "gat": "#DD8452", "transformer": "#55A868"}
_ARCH_LABELS = {"gnn": "GNN", "gat": "GAT", "transformer": "Transformer"}
_DEFAULT_RESULTS = "src/artifacts/semester_contribution/ablation_results.csv"
_DEFAULT_PLOTS = "src/artifacts/semester_contribution/plots"


# ---------------------------------------------------------------------------
# Plot 1 — Win rate by architecture (bar chart)
# ---------------------------------------------------------------------------


def plot_win_rate_by_arch(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))

    archs = df["arch"].unique()
    x = np.arange(len(archs))
    width = 0.5

    for i, arch in enumerate(archs):
        subset = df[df["arch"] == arch]["win_rate"]
        mean, std = subset.mean(), subset.std()
        ax.bar(
            x[i],
            mean,
            width,
            yerr=std,
            capsize=5,
            color=_ARCH_COLORS.get(arch, "grey"),
            label=_ARCH_LABELS.get(arch, arch),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_ARCH_LABELS.get(a, a) for a in archs])
    ax.set_ylabel("Win rate (MrX)")
    ax.set_title("Win rate by architecture (mean ± std over all seeds & configs)")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, out_dir / "win_rate_by_arch.png")


# ---------------------------------------------------------------------------
# Plot 2 — Sample efficiency (win rate vs mean episode length)
# ---------------------------------------------------------------------------


def plot_sample_efficiency(df: pd.DataFrame, out_dir: Path):
    agg = (
        df.groupby(["arch", "n_layers", "hidden_dim", "n_heads"])[
            ["win_rate", "mean_episode_length"]
        ]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    for arch in agg["arch"].unique():
        subset = agg[agg["arch"] == arch]
        ax.scatter(
            subset["mean_episode_length"],
            subset["win_rate"],
            label=_ARCH_LABELS.get(arch, arch),
            color=_ARCH_COLORS.get(arch, "grey"),
            alpha=0.75,
            s=60,
        )

    ax.set_xlabel("Mean episode length (steps)")
    ax.set_ylabel("Win rate (MrX)")
    ax.set_title("Sample efficiency: win rate vs episode length")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "sample_efficiency.png")


# ---------------------------------------------------------------------------
# Plot 3 — Performance vs parameter count
# ---------------------------------------------------------------------------


def plot_perf_vs_params(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))

    for arch in df["arch"].unique():
        subset = df[df["arch"] == arch]
        ax.scatter(
            subset["n_params"],
            subset["win_rate"],
            label=_ARCH_LABELS.get(arch, arch),
            color=_ARCH_COLORS.get(arch, "grey"),
            alpha=0.65,
            s=50,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Win rate (MrX)")
    ax.set_title("Performance vs model size")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "perf_vs_params.png")


# ---------------------------------------------------------------------------
# Plot 4 / 5 — Attention heatmaps (requires live model + env)
# ---------------------------------------------------------------------------


def _edges_to_dense(edge_index: "torch.Tensor", alpha: "torch.Tensor", n_nodes: int):
    """Convert edge-based attention weights to a dense node×node matrix."""

    mat = np.zeros((n_nodes, n_nodes))
    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    weights = alpha.cpu().float().mean(dim=-1).numpy()  # average over heads
    for s, d, w in zip(src, dst, weights):
        mat[s, d] = w
    return mat


def plot_attention_heatmap(
    arch: str,
    checkpoint_dir: str,
    n_layers: int,
    hidden_dim: int,
    n_heads: int,
    seed: int,
    size_name: str,
    size_cfg: dict,
    out_dir: Path,
):
    """Run one episode, extract attention from last layer, save heatmap."""

    if arch not in ("gat", "transformer"):
        print(f"[skip] Attention heatmaps not supported for arch={arch}")
        return

    num_police = size_cfg["num_police_agents"]
    num_agents = num_police + 1
    node_feature_size = num_agents + 1

    ckpt_dir = (
        Path(checkpoint_dir)
        / f"{arch}_L{n_layers}_H{hidden_dim}_h{n_heads}_s{seed}_{size_name}"
    )
    mrx_path = ckpt_dir / "MrX.pt"
    if not mrx_path.exists():
        print(f"[skip] checkpoint not found for attention heatmap: {mrx_path}")
        return

    mrx_agent = build_agent(arch, n_layers, hidden_dim, n_heads, node_feature_size)
    state_dict = torch.load(str(mrx_path), map_location=device, weights_only=True)
    mrx_agent.load_state_dict(state_dict, strict=False)
    mrx_agent.model.eval()

    class _SilentLogger:
        def log(self, *a, **kw):
            pass

        def log_scalar(self, *a, **kw):
            pass

        def close(self):
            pass

    env_wrappable = CustomEnvironment(
        number_of_agents=num_agents,
        agent_money=size_cfg["agent_money"],
        reward_weights=_NULL_REWARD_WEIGHTS,
        logger=_SilentLogger(),
        epoch=0,
        graph_nodes=size_cfg["graph_nodes"],
        graph_edges=size_cfg["graph_edges"],
        vis_configs={
            "visualize_game": False,
            "visualize_heatmap": False,
            "save_visualization": False,
            "save_dir": "logs/vis",
        },
    )
    env = PettingZooWrapper(env=env_wrappable)
    state = env.reset(episode=0)

    # Collect one step's attention
    mrx_graph = create_graph_data(state, "MrX", env).to(device)
    with torch.no_grad():
        _, attentions = mrx_agent.model(mrx_graph, return_attention=True)

    # Use last hidden layer (not the output layer)
    layer_idx = min(len(attentions) - 2, len(attentions) - 1)
    edge_index, alpha = attentions[layer_idx]
    n_nodes = mrx_graph.num_nodes
    mat = _edges_to_dense(edge_index, alpha, n_nodes)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Attention weight")
    ax.set_xlabel("Target node")
    ax.set_ylabel("Source node")
    arch_label = _ARCH_LABELS.get(arch, arch)
    ax.set_title(
        f"{arch_label} attention heatmap\n"
        f"({size_name} graph, L{n_layers} H{hidden_dim} h{n_heads}, seed {seed})"
    )
    fig.tight_layout()
    fname = (
        f"attention_{arch}_{size_name}_L{n_layers}_H{hidden_dim}_h{n_heads}_s{seed}.png"
    )
    _save(fig, out_dir / fname)


# ---------------------------------------------------------------------------
# Plot 6 — Win rate vs n_layers
# ---------------------------------------------------------------------------


def plot_win_rate_vs_layers(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))

    for arch in df["arch"].unique():
        subset = df[df["arch"] == arch]
        layer_agg = subset.groupby("n_layers")["win_rate"].agg(["mean", "std"])
        ax.plot(
            layer_agg.index,
            layer_agg["mean"],
            marker="o",
            label=_ARCH_LABELS.get(arch, arch),
            color=_ARCH_COLORS.get(arch, "grey"),
        )
        ax.fill_between(
            layer_agg.index,
            layer_agg["mean"] - layer_agg["std"],
            layer_agg["mean"] + layer_agg["std"],
            alpha=0.2,
            color=_ARCH_COLORS.get(arch, "grey"),
        )

    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Win rate (MrX)")
    ax.set_title("Win rate vs depth (mean ± std over seeds & hidden dims)")
    ax.legend()
    ax.set_xticks(sorted(df["n_layers"].unique()))
    fig.tight_layout()
    _save(fig, out_dir / "win_rate_vs_layers.png")


# ---------------------------------------------------------------------------
# Learning curves from TensorBoard logs
# ---------------------------------------------------------------------------


def plot_learning_curves_from_tb(
    log_dirs: Dict[str, str],
    tag: str,
    out_dir: Path,
):
    """Read TensorBoard event files and plot win ratio over epochs.

    Args:
        log_dirs: mapping of arch name → path to TensorBoard log dir.
        tag: scalar tag to extract (e.g. "epoch/win_ratio").
        out_dir: directory to save the plot.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print(
            "[warn] tensorboard not installed. Skipping learning curves. "
            "Install with: pip install tensorboard"
        )
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    found_any = False

    for arch, log_path in log_dirs.items():
        if not Path(log_path).exists():
            print(f"[warn] TensorBoard log dir not found: {log_path}")
            continue

        ea = EventAccumulator(log_path)
        ea.Reload()

        if tag not in ea.Tags().get("scalars", []):
            available = ea.Tags().get("scalars", [])
            print(
                f"[warn] Tag '{tag}' not found in {log_path}. "
                f"Available: {available[:8]}"
            )
            continue

        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax.plot(
            steps,
            values,
            label=_ARCH_LABELS.get(arch, arch),
            color=_ARCH_COLORS.get(arch, "grey"),
        )
        found_any = True

    if not found_any:
        plt.close(fig)
        print("[warn] No learning curve data found. Skipping plot.")
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Win ratio (Police)")
    ax.set_title("Learning curves by architecture")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, out_dir / "learning_curves.png")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate architecture ablation plots."
    )
    parser.add_argument(
        "--results",
        default=_DEFAULT_RESULTS,
        help="Path to ablation_results.csv",
    )
    parser.add_argument(
        "--out_dir",
        default=_DEFAULT_PLOTS,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Also generate attention heatmaps (requires checkpoints + config)",
    )
    parser.add_argument(
        "--attention_config",
        default="src/configs/eval/ablation.yaml",
        help="Ablation config for attention heatmap runs",
    )
    parser.add_argument(
        "--tb_logs",
        nargs="+",
        metavar="PATH:ARCH",
        help=(
            "TensorBoard log dirs for learning curves. "
            "Format: 'path/to/logs:arch_name', e.g. "
            "logs/gnn_experiment:gnn logs/gat_experiment:gat"
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load results CSV ---
    results_path = Path(args.results)
    if not results_path.exists():
        print(
            f"Results file not found: {results_path}\n"
            "Run architecture_ablations.py first to generate it."
        )
        return

    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} rows from {results_path}")
    print(f"Architectures present: {sorted(df['arch'].unique())}")
    print(f"Saving plots to: {out_dir}\n")

    # --- CSV-based plots ---
    print("Plot 1: Win rate by architecture")
    plot_win_rate_by_arch(df, out_dir)

    print("Plot 2: Sample efficiency")
    plot_sample_efficiency(df, out_dir)

    print("Plot 3: Performance vs parameter count")
    plot_perf_vs_params(df, out_dir)

    print("Plot 6: Win rate vs n_layers")
    plot_win_rate_vs_layers(df, out_dir)

    # --- Attention heatmaps (require live model + env) ---
    if args.attention:

        with open(args.attention_config) as f:
            cfg = yaml.safe_load(f)

        print("\nPlot 4 / 5: Attention heatmaps")
        for arch in ("gat", "transformer"):
            # Use best config: pick highest mean win_rate combination
            arch_df = df[df["arch"] == arch]
            if arch_df.empty:
                print(f"  [skip] no results for {arch}")
                continue
            best = arch_df.loc[arch_df["win_rate"].idxmax()]
            size_name = best["graph_size"]
            size_cfg = cfg["graph_sizes"][size_name]
            plot_attention_heatmap(
                arch=arch,
                checkpoint_dir=cfg["checkpoint_dir"],
                n_layers=int(best["n_layers"]),
                hidden_dim=int(best["hidden_dim"]),
                n_heads=int(best["n_heads"]),
                seed=int(best["seed"]),
                size_name=size_name,
                size_cfg=size_cfg,
                out_dir=out_dir,
            )

    # --- Learning curves from TensorBoard ---
    if args.tb_logs:
        log_dirs = {}
        for entry in args.tb_logs:
            if ":" not in entry:
                print(
                    f"[warn] Skipping malformed --tb_logs entry: {entry!r} (expected path:arch)"
                )
                continue
            path, arch = entry.rsplit(":", 1)
            log_dirs[arch] = path

        if log_dirs:
            print("\nPlot 7: Learning curves from TensorBoard")
            plot_learning_curves_from_tb(log_dirs, "epoch/win_ratio", out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
