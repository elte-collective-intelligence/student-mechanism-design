"""Plotting utilities for ablation study results.

This script generates comparison plots for ablation experiments.
"""

import json
import os
import argparse
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_ablation_results(json_path: str) -> dict:
    """Load ablation results from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def plot_belief_ablation(results: dict, output_dir: str):
    """Generate plots for belief ablation study.

    Creates:
    1. Win rate comparison
    2. Belief quality (cross-entropy) comparison
    3. Episode length comparison
    """
    variants = results["variants"]
    variant_names = list(variants.keys())

    # Prepare data
    win_rates = [variants[v]["metrics"]["win_rate"] for v in variant_names]
    win_rate_stds = [variants[v]["metrics"]["win_rate_std"] for v in variant_names]
    belief_ces = [variants[v]["metrics"]["mean_belief_ce"] for v in variant_names]
    belief_ce_stds = [variants[v]["metrics"]["belief_ce_std"] for v in variant_names]
    episode_lengths = [
        variants[v]["metrics"]["mean_episode_length"] for v in variant_names
    ]
    episode_length_stds = [
        variants[v]["metrics"]["episode_length_std"] for v in variant_names
    ]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Win Rate
    ax1 = axes[0]
    x_pos = np.arange(len(variant_names))
    bars1 = ax1.bar(x_pos, win_rates, yerr=win_rate_stds, capsize=5, alpha=0.7)
    ax1.axhline(y=0.5, color="red", linestyle="--", linewidth=2, label="Target (50%)")
    ax1.set_xlabel("Belief Variant", fontsize=12, fontweight="bold")
    ax1.set_ylabel("MrX Win Rate", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Win Rate Comparison\n(Ablation 1: Belief Tracking)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([v.replace("_", "\n") for v in variant_names])
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars1, win_rates, win_rate_stds)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2%}\n±{std:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Plot 2: Belief Quality (Cross-Entropy)
    ax2 = axes[1]
    bars2 = ax2.bar(
        x_pos, belief_ces, yerr=belief_ce_stds, capsize=5, alpha=0.7, color="orange"
    )
    ax2.set_xlabel("Belief Variant", fontsize=12, fontweight="bold")
    ax2.set_ylabel(
        "Mean Cross-Entropy (lower is better)", fontsize=12, fontweight="bold"
    )
    ax2.set_title(
        "Belief Quality at Reveal Times\n(Ablation 1: Belief Tracking)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([v.replace("_", "\n") for v in variant_names])
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, (bar, val, std) in enumerate(zip(bars2, belief_ces, belief_ce_stds)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}\n±{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Plot 3: Episode Length
    ax3 = axes[2]
    bars3 = ax3.bar(
        x_pos,
        episode_lengths,
        yerr=episode_length_stds,
        capsize=5,
        alpha=0.7,
        color="green",
    )
    ax3.set_xlabel("Belief Variant", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Mean Episode Length (steps)", fontsize=12, fontweight="bold")
    ax3.set_title(
        "Average Episode Duration\n(Ablation 1: Belief Tracking)",
        fontsize=14,
        fontweight="bold",
    )
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([v.replace("_", "\n") for v in variant_names])
    ax3.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, (bar, val, std) in enumerate(
        zip(bars3, episode_lengths, episode_length_stds)
    ):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}\n±{std:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "belief_ablation_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved belief ablation plot to {output_path}")
    plt.close()


def plot_mechanism_ablation(results: dict, output_dir: str):
    """Generate plots for mechanism ablation study.

    Creates:
    1. Win rate comparison
    2. Mechanism cost comparison
    3. Time-to-catch/survive comparison
    """
    variants = results["variants"]
    variant_names = list(variants.keys())

    # Prepare data
    win_rates = [variants[v]["metrics"]["win_rate"] for v in variant_names]
    win_rate_stds = [variants[v]["metrics"]["win_rate_std"] for v in variant_names]
    budget_spent = [variants[v]["metrics"]["mean_budget_spent"] for v in variant_names]
    tolls_paid = [variants[v]["metrics"]["mean_tolls_paid"] for v in variant_names]
    time_to_catch = [
        variants[v]["metrics"]["mean_time_to_catch"] for v in variant_names
    ]
    survival_time = [
        variants[v]["metrics"]["mean_survival_time"] for v in variant_names
    ]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Win Rate
    ax1 = axes[0, 0]
    x_pos = np.arange(len(variant_names))
    bars1 = ax1.bar(x_pos, win_rates, yerr=win_rate_stds, capsize=5, alpha=0.7)
    ax1.axhline(y=0.5, color="red", linestyle="--", linewidth=2, label="Target (50%)")
    ax1.set_xlabel("Mechanism Variant", fontsize=12, fontweight="bold")
    ax1.set_ylabel("MrX Win Rate", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Win Rate Comparison\n(Ablation 2: Mechanism Design)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([v.replace("_", "\n") for v in variant_names], fontsize=10)
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    for i, (bar, val, std) in enumerate(zip(bars1, win_rates, win_rate_stds)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2%}\n±{std:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Plot 2: Budget Spent
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, budget_spent, alpha=0.7, color="orange")
    ax2.set_xlabel("Mechanism Variant", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Mean Budget Spent", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Budget Efficiency\n(Ablation 2: Mechanism Design)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([v.replace("_", "\n") for v in variant_names], fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars2, budget_spent):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Plot 3: Tolls Paid
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos, tolls_paid, alpha=0.7, color="green")
    ax3.set_xlabel("Mechanism Variant", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Mean Tolls Paid", fontsize=12, fontweight="bold")
    ax3.set_title(
        "Total Edge Costs\n(Ablation 2: Mechanism Design)",
        fontsize=14,
        fontweight="bold",
    )
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([v.replace("_", "\n") for v in variant_names], fontsize=10)
    ax3.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars3, tolls_paid):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Plot 4: Time Comparison
    ax4 = axes[1, 1]
    width = 0.35
    x_pos_offset = x_pos - width / 2
    bars4a = ax4.bar(
        x_pos_offset,
        time_to_catch,
        width,
        label="Time to Catch (Police wins)",
        alpha=0.7,
        color="blue",
    )
    bars4b = ax4.bar(
        x_pos_offset + width,
        survival_time,
        width,
        label="Survival Time (MrX wins)",
        alpha=0.7,
        color="red",
    )
    ax4.set_xlabel("Mechanism Variant", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Mean Steps", fontsize=12, fontweight="bold")
    ax4.set_title(
        "Episode Duration by Outcome\n(Ablation 2: Mechanism Design)",
        fontsize=14,
        fontweight="bold",
    )
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([v.replace("_", "\n") for v in variant_names], fontsize=10)
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "mechanism_ablation_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved mechanism ablation plot to {output_path}")
    plt.close()


def generate_summary_table(
    belief_results: dict, mechanism_results: dict, output_dir: str
):
    """Generate a summary table comparing all variants."""

    summary_lines = [
        "=" * 80,
        "ABLATION STUDIES SUMMARY",
        "=" * 80,
        "",
        "ABLATION 1: BELIEF TRACKING",
        "-" * 80,
        f"{'Variant':<20} {'Win Rate':<15} {'Belief CE':<15} {'Ep Length':<15}",
        "-" * 80,
    ]

    for name, data in belief_results["variants"].items():
        m = data["metrics"]
        summary_lines.append(
            f"{name:<20} {m['win_rate']:>6.2%} ±{m['win_rate_std']:5.2%}  "
            f"{m['mean_belief_ce']:>6.3f} ±{m['belief_ce_std']:5.3f}  "
            f"{m['mean_episode_length']:>6.1f} ±{m['episode_length_std']:5.1f}"
        )

    summary_lines.extend(
        [
            "",
            "ABLATION 2: MECHANISM DESIGN",
            "-" * 80,
            f"{'Variant':<20} {'Win Rate':<15} {'Budget Spent':<15} {'Tolls Paid':<15}",
            "-" * 80,
        ]
    )

    for name, data in mechanism_results["variants"].items():
        m = data["metrics"]
        summary_lines.append(
            f"{name:<20} {m['win_rate']:>6.2%} ±{m['win_rate_std']:5.2%}  "
            f"{m['mean_budget_spent']:>10.2f}     "
            f"{m['mean_tolls_paid']:>10.2f}"
        )

    summary_lines.extend(
        [
            "",
            "=" * 80,
        ]
    )

    summary_text = "\n".join(summary_lines)

    # Save to file
    output_path = os.path.join(output_dir, "ablation_summary.txt")
    with open(output_path, "w") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\nSummary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="logs/ablations",
        help="Directory containing ablation results JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/ablations",
        help="Directory to save plots",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    belief_path = os.path.join(args.input_dir, "belief_results.json")
    mechanism_path = os.path.join(args.input_dir, "mechanism_results.json")

    if not os.path.exists(belief_path):
        print(f"ERROR: Belief results not found at {belief_path}")
        print("Please run: python src/eval/run_ablations.py --ablation belief")
        return

    if not os.path.exists(mechanism_path):
        print(f"ERROR: Mechanism results not found at {mechanism_path}")
        print("Please run: python src/eval/run_ablations.py --ablation mechanism")
        return

    belief_results = load_ablation_results(belief_path)
    mechanism_results = load_ablation_results(mechanism_path)

    # Generate plots
    print("\nGenerating plots...")
    plot_belief_ablation(belief_results, args.output_dir)
    plot_mechanism_ablation(mechanism_results, args.output_dir)
    generate_summary_table(belief_results, mechanism_results, args.output_dir)

    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80)
    print(f"Plots saved to: {args.output_dir}/")
    print("  - belief_ablation_comparison.png")
    print("  - mechanism_ablation_comparison.png")
    print("  - ablation_summary.txt")


if __name__ == "__main__":
    main()
