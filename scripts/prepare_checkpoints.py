import shutil
from pathlib import Path

def prepare_checkpoints():
    ablation_runs_dir = Path("src/configs/experiments/ablation_runs")
    checkpoint_dir = Path("src/artifacts/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not ablation_runs_dir.exists():
        print(f"Error: {ablation_runs_dir} does not exist.")
        return

    for run_folder in ablation_runs_dir.iterdir():
        if not run_folder.is_dir():
            continue
        
        run_name = run_folder.name
        # Skip folders that don't match the pattern (arch_L..._H..._h..._s..._size)
        if "_" not in run_name:
            continue
            
        logs_dir = run_folder / "logs"
        if not logs_dir.exists():
            continue
            
        # Determine num_agents from the filenames
        # Pattern: MrX_{N}_agents.pt
        mrx_files = list(logs_dir.glob("MrX_*_agents.pt"))
        if not mrx_files:
            continue
            
        # Use the latest one or any one (usually there's only one per run in ablation)
        mrx_src = mrx_files[0]
        police_src = logs_dir / mrx_src.name.replace("MrX", "Police")
        
        if not police_src.exists():
            print(f"Warning: Police checkpoint not found in {logs_dir}")
            continue
            
        dst_folder = checkpoint_dir / run_name
        dst_folder.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(mrx_src, dst_folder / "MrX.pt")
        shutil.copy2(police_src, dst_folder / "Police.pt")
        print(f"Copied {run_name} checkpoints.")

if __name__ == "__main__":
    prepare_checkpoints()
