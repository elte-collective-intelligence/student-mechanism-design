import yaml
import os
import subprocess
import time
import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_single_training(args):
    """Worker function for parallel execution."""
    (
        run_name,
        arch,
        n_layers,
        hidden_dim,
        n_heads,
        seed,
        size_name,
        size_cfg,
        cfg,
        checkpoint_root,
        python_exe,
    ) = args

    ckpt_dir = checkpoint_root / run_name

    # Skip if exists
    if (ckpt_dir / "MrX.pt").exists():
        return f"Skipped {run_name}"

    # Isolated directory for each run
    run_dir = os.path.abspath(f"src/configs/experiments/ablation_runs/{run_name}")
    os.makedirs(run_dir, exist_ok=True)

    # 1. Create agent config
    agent_cfg = cfg["agent_common"].copy()
    agent_cfg.update(cfg["arch_defaults"].get(arch, {}))
    agent_cfg.update(
        {
            "agent_type": arch,
            "hidden_dim": hidden_dim,
            "num_layers": n_layers,
        }
    )
    if arch == "gat":
        agent_cfg["heads"] = n_heads
    elif arch == "transformer":
        agent_cfg["num_heads"] = n_heads

    tmp_agent_path = os.path.join(run_dir, "agent.yaml")
    with open(tmp_agent_path, "w") as f:
        yaml.dump(agent_cfg, f)

    # 2. Create experiment config
    exp_cfg = {
        "agent_configurations": [
            {
                "num_police_agents": size_cfg["num_police_agents"],
                "agent_money": size_cfg["agent_money"],
            }
        ],
        "graph_nodes": size_cfg["graph_nodes"],
        "graph_edges": size_cfg["graph_edges"],
        "num_episodes": cfg["training"]["num_episodes"],
        "num_eval_episodes": cfg["training"]["num_eval_episodes"],
        "epochs": cfg["training"]["epochs"],
        "random_seed": seed,
        "log_configs": "default",
        "vis_configs": "none",
        "wandb_run_name": f"ablation_{run_name}",
        "wandb_group": "ablation_study",
        "wandb_resume": True,
        "ablation_checkpoint_dir": str(ckpt_dir),
    }
    tmp_exp_path = os.path.join(run_dir, "config.yaml")
    with open(tmp_exp_path, "w") as f:
        yaml.dump(exp_cfg, f)

    # 3. Run training
    try:
        subprocess.run(
            [
                python_exe,
                "src/main.py",
                "--config",
                tmp_exp_path,
                "--agent_configs",
                tmp_agent_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return f"Finished {run_name}"
    except subprocess.CalledProcessError as e:
        return f"FAILED {run_name}: {e.stderr.decode()}"


def main():
    # Load ablation config
    with open("src/configs/eval/ablation.yaml") as f:
        cfg = yaml.safe_load(f)

    python_exe = "python"
    checkpoint_root = Path(cfg["checkpoint_dir"])

    # Configuration for parallelism
    # Limit parallel runs to avoid OOM. 2-4 is usually safe for GNNs.
    max_workers = 3

    grid_runs = []
    for arch in cfg["architectures"]:
        for setting in cfg["sweep"]:
            n_layers = setting["n_layers"]
            hidden_dim = setting["hidden_dim"]
            n_heads = 1 if arch == "gnn" else setting["n_heads"]

            for seed in cfg["seeds"]:
                for size_name, size_cfg in cfg["graph_sizes"].items():
                    run_name = f"{arch}_L{n_layers}_H{hidden_dim}_h{n_heads}_s{seed}_{size_name}"
                    grid_runs.append(
                        (
                            run_name,
                            arch,
                            n_layers,
                            hidden_dim,
                            n_heads,
                            seed,
                            size_name,
                            size_cfg,
                            cfg,
                            checkpoint_root,
                            python_exe,
                        )
                    )

    total_runs = len(grid_runs)
    print(
        f"Starting parallel ablation sweep ({max_workers} workers). Total runs: {total_runs}"
    )

    start_time = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_training, run): run[0] for run in grid_runs
        }

        for future in as_completed(futures):
            completed += 1
            result = future.result()

            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            runs_left = total_runs - completed
            eta = str(datetime.timedelta(seconds=int(avg_time * runs_left)))

            print(f"[{completed:03d}/{total_runs}] {result} | ETA: {eta}")


if __name__ == "__main__":
    main()
