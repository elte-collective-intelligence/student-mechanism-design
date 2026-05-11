import os
import pandas as pd
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys
from ..src.eval.architecture_ablations import build_agent, count_parameters

# Add src to path to import build_agent and count_parameters
sys.path.insert(0, os.path.abspath("src"))


def extract_from_tb(log_dir):
    """Extract final metrics from TensorBoard logs."""
    event_acc = EventAccumulator(str(log_dir))
    event_acc.Reload()
    
    tags = event_acc.Tags().get("scalars", [])
    
    results = {}
    
    if "epoch/win_ratio" in tags:
        # Get last 5 epochs average
        values = [e.value for e in event_acc.Scalars("epoch/win_ratio")]
        results["win_rate"] = float(np.mean(values[-5:])) if values else 0.0
    else:
        results["win_rate"] = 0.0
        
    if "episode/steps" in tags:
        # Get last 50 episodes average
        values = [e.value for e in event_acc.Scalars("episode/steps")]
        results["mean_episode_length"] = float(np.mean(values[-50:])) if values else 0.0
    else:
        results["mean_episode_length"] = 0.0
        
    results["mean_entropy"] = 0.0 # Entropy not easily available in trainer logs
    
    return results

def main():
    ablation_dir = Path("src/configs/experiments/ablation_runs")
    output_path = Path("src/artifacts/semester_contribution/ablation_results.csv")
    
    rows = []
    
    # Pre-calculate n_params for unique architectures to save time
    params_cache = {}
    
    runs = list(ablation_dir.iterdir())
    print(f"Processing {len(runs)} runs...")
    
    for i, run_folder in enumerate(runs):
        if not run_folder.is_dir():
            continue
            
        run_name = run_folder.name
        # gat_L2_H64_h2_s1_small
        parts = run_name.split("_")
        if len(parts) < 6:
            continue
            
        arch = parts[0]
        n_layers = int(parts[1][1:])
        hidden_dim = int(parts[2][1:])
        n_heads = int(parts[3][1:])
        seed = int(parts[4][1:])
        graph_size = parts[5]
        
        log_dir = run_folder / "logs"
        if not log_dir.exists():
            continue
            
        print(f"[{i+1}/{len(runs)}] {run_name}")
        
        try:
            metrics = extract_from_tb(log_dir)
            
            # Calculate parameters
            # num_police from size
            num_police = 5 if graph_size == "small" else 15
            num_agents = num_police + 1
            node_feature_size = num_agents + 1
            
            cache_key = (arch, n_layers, hidden_dim, n_heads, node_feature_size)
            if cache_key not in params_cache:
                agent = build_agent(arch, n_layers, hidden_dim, n_heads, node_feature_size)
                params_cache[cache_key] = count_parameters(agent)
            
            n_params = params_cache[cache_key]
            
            rows.append({
                "arch": arch,
                "n_layers": n_layers,
                "hidden_dim": hidden_dim,
                "n_heads": n_heads,
                "seed": seed,
                "graph_size": graph_size,
                "win_rate": metrics["win_rate"],
                "mean_episode_length": metrics["mean_episode_length"],
                "mean_entropy": metrics["mean_entropy"],
                "n_params": n_params
            })
        except Exception as e:
            print(f"  Error processing {run_name}: {e}")
            
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nDone. Saved to {output_path}")

if __name__ == "__main__":
    main()
