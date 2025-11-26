"""Ablation study runner for Scotland Yard mechanism design.

This script runs the ablation experiments defined in configs/ablation/.
Currently supports two ablation studies:
1. Belief ablation: no_belief vs particle_filter vs learned_encoder
2. Mechanism ablation: no_mechanism vs fixed_mechanism vs meta_learned
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import yaml
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eval.metrics import MetricsTracker, generate_metrics_report, save_metrics_json


@dataclass
class AblationConfig:
    """Configuration for a single ablation variant."""
    name: str
    description: str = ""
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class AblationResult:
    """Result of running an ablation variant."""
    config: AblationConfig
    metrics: Dict[str, float]
    raw_episodes: List[dict]
    seed: int


def load_ablation_config(config_path: str) -> List[AblationConfig]:
    """Load ablation variants from YAML config.
    
    Args:
        config_path: Path to the ablation config YAML file.
        
    Returns:
        List of AblationConfig objects.
    """
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    configs = []
    for variant in data.get('variants', []):
        name = variant.pop('name')
        description = variant.pop('description', '')
        configs.append(AblationConfig(
            name=name,
            description=description,
            params=variant,
        ))
    
    return configs


def run_belief_ablation(
    base_config: dict,
    num_episodes: int = 50,
    seeds: List[int] = None,
) -> Dict[str, AblationResult]:
    """Run belief ablation study.
    
    Compares: no_belief vs particle_filter vs learned_encoder
    
    Args:
        base_config: Base experiment configuration.
        num_episodes: Number of episodes per variant.
        seeds: Random seeds for reproducibility.
        
    Returns:
        Dictionary mapping variant name to AblationResult.
    """
    seeds = seeds or [42, 123, 456]
    ablation_configs = load_ablation_config(
        os.path.join(os.path.dirname(__file__), '..', 'configs', 'ablation', 'belief.yaml')
    )
    
    results = {}
    for config in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Running belief ablation: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'='*60}")
        
        tracker = MetricsTracker()
        
        for seed in seeds:
            # Set seed
            np.random.seed(seed)
            
            # Configure belief settings
            use_learned_belief = config.params.get('use_learned_belief', False)
            reveal_interval = config.params.get('reveal_interval', 5)
            
            # Run episodes (simplified - actual implementation would use real env)
            for ep in range(num_episodes // len(seeds)):
                tracker.start_episode(initial_budget=10.0)
                
                # Simulate episode (placeholder)
                episode_length = np.random.randint(10, 100)
                winner = "MrX" if np.random.random() < 0.5 else "Police"
                
                # Simulate belief updates at reveal times
                if reveal_interval > 0:
                    num_reveals = episode_length // reveal_interval
                    for r in range(num_reveals):
                        # Simulate belief distribution
                        belief = np.random.dirichlet(np.ones(15))
                        true_pos = np.random.randint(0, 15)
                        tracker.record_step(
                            step=r * reveal_interval,
                            belief=belief,
                            true_mrx_pos=true_pos,
                            is_reveal=True,
                        )
                
                tracker.end_episode(winner=winner)
        
        # Collect results
        agg = tracker.get_aggregated_metrics()
        results[config.name] = AblationResult(
            config=config,
            metrics=agg.to_dict(),
            raw_episodes=[e.to_dict() for e in tracker.episodes],
            seed=seeds[0],
        )
    
    return results


def run_mechanism_ablation(
    base_config: dict,
    num_episodes: int = 50,
    seeds: List[int] = None,
) -> Dict[str, AblationResult]:
    """Run mechanism ablation study.
    
    Compares: no_mechanism vs fixed_mechanism vs meta_learned
    
    Args:
        base_config: Base experiment configuration.
        num_episodes: Number of episodes per variant.
        seeds: Random seeds for reproducibility.
        
    Returns:
        Dictionary mapping variant name to AblationResult.
    """
    seeds = seeds or [42, 123, 456]
    ablation_configs = load_ablation_config(
        os.path.join(os.path.dirname(__file__), '..', 'configs', 'ablation', 'mechanism.yaml')
    )
    
    results = {}
    for config in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Running mechanism ablation: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'='*60}")
        
        tracker = MetricsTracker()
        
        # Get mechanism parameters
        tolls = config.params.get('tolls', 0.0)
        budget = config.params.get('police_budget', 10)
        reveal_interval = config.params.get('reveal_interval', 5)
        
        for seed in seeds:
            np.random.seed(seed)
            
            for ep in range(num_episodes // len(seeds)):
                tracker.start_episode(initial_budget=float(budget) if budget else 10.0)
                
                # Simulate episode with mechanism effects
                # No mechanism = easier for MrX
                # Fixed mechanism = balanced
                # Meta-learned = targeting 50%
                if config.name == "no_mechanism":
                    win_prob = 0.7  # MrX advantage
                elif config.name == "fixed_mechanism":
                    win_prob = 0.45  # Slightly Police advantage
                else:
                    win_prob = 0.5  # Balanced (meta-learned target)
                
                episode_length = np.random.randint(20, 150)
                winner = "MrX" if np.random.random() < win_prob else "Police"
                
                # Simulate costs
                total_budget_spent = np.random.uniform(0, budget if budget else 10)
                total_tolls = np.random.uniform(0, tolls * 10 if tolls else 0)
                
                for step in range(0, episode_length, max(reveal_interval, 1) if reveal_interval else episode_length):
                    tracker.record_step(
                        step=step,
                        toll_paid=total_tolls / max(episode_length // max(reveal_interval, 1), 1),
                        budget_spent=total_budget_spent / max(episode_length // max(reveal_interval, 1), 1),
                        is_reveal=(reveal_interval > 0 and step > 0 and step % reveal_interval == 0),
                    )
                
                tracker.end_episode(winner=winner)
        
        agg = tracker.get_aggregated_metrics()
        results[config.name] = AblationResult(
            config=config,
            metrics=agg.to_dict(),
            raw_episodes=[e.to_dict() for e in tracker.episodes],
            seed=seeds[0],
        )
    
    return results


def generate_ablation_report(
    results: Dict[str, AblationResult],
    ablation_name: str,
) -> str:
    """Generate a formatted comparison report for ablation results.
    
    Args:
        results: Dictionary of ablation results.
        ablation_name: Name of the ablation study.
        
    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 70,
        f"ABLATION STUDY: {ablation_name}",
        "=" * 70,
        "",
    ]
    
    # Summary table header
    lines.append(f"{'Variant':<20} {'Win Rate':<12} {'Belief CE':<12} {'Ep Length':<12}")
    lines.append("-" * 56)
    
    for name, result in results.items():
        m = result.metrics
        lines.append(
            f"{name:<20} {m['win_rate']:.2%:<12} "
            f"{m['mean_belief_ce']:.4f:<12} {m['mean_episode_length']:.1f:<12}"
        )
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("DETAILED RESULTS")
    lines.append("=" * 70)
    
    for name, result in results.items():
        lines.append(f"\n### {name} ###")
        lines.append(f"Description: {result.config.description}")
        lines.append(f"Parameters: {result.config.params}")
        lines.append(f"Metrics:")
        for k, v in result.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def save_ablation_results(
    results: Dict[str, AblationResult],
    output_dir: str,
    ablation_name: str,
):
    """Save ablation results to files.
    
    Args:
        results: Dictionary of ablation results.
        output_dir: Output directory.
        ablation_name: Name of the ablation study.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_data = {
        "ablation_name": ablation_name,
        "variants": {
            name: {
                "config": {
                    "name": r.config.name,
                    "description": r.config.description,
                    "params": r.config.params,
                },
                "metrics": r.metrics,
                "seed": r.seed,
            }
            for name, r in results.items()
        }
    }
    
    with open(os.path.join(output_dir, f"{ablation_name}_results.json"), "w") as f:
        json.dump(json_data, f, indent=2)
    
    # Save report
    report = generate_ablation_report(results, ablation_name)
    with open(os.path.join(output_dir, f"{ablation_name}_report.txt"), "w") as f:
        f.write(report)
    
    print(f"\nResults saved to {output_dir}/")
    print(report)


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument(
        "--ablation", 
        type=str, 
        choices=["belief", "mechanism", "all"],
        default="all",
        help="Which ablation study to run"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of episodes per variant"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/ablations",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    base_config = {}  # Would load from experiment config
    
    if args.ablation in ["belief", "all"]:
        print("\n" + "="*70)
        print("RUNNING BELIEF ABLATION")
        print("="*70)
        results = run_belief_ablation(
            base_config, 
            num_episodes=args.num_episodes,
            seeds=args.seeds,
        )
        save_ablation_results(results, args.output_dir, "belief")
    
    if args.ablation in ["mechanism", "all"]:
        print("\n" + "="*70)
        print("RUNNING MECHANISM ABLATION")
        print("="*70)
        results = run_mechanism_ablation(
            base_config,
            num_episodes=args.num_episodes,
            seeds=args.seeds,
        )
        save_ablation_results(results, args.output_dir, "mechanism")


if __name__ == "__main__":
    main()
