import os
import sys
import torch
import yaml
import numpy as np
from pathlib import Path
import time
import random

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

# Imports
from eval.architecture_ablations import build_agent
from environment.yard import CustomEnvironment
from training.utils import create_graph_data, is_episode_done, extract_step_info, device
from torchrl.envs import step_mdp
from torchrl.envs.libs.pettingzoo import PettingZooWrapper

_FULL_REWARD_WEIGHTS = {
    "Police_distance": 0.1,
    "Police_group": 0.1,
    "Police_position": 0.1,
    "Police_time": 0.0,
    "Mrx_closest": 0.3,
    "Mrx_average": 0.2,
    "Mrx_position": 0.1,
    "Mrx_time": 0.0,
    "Police_coverage": 0.05,
    "Police_proximity": 0.05,
    "Police_overlap_penalty": 0.0,
}

def generate_gif(arch, n_layers, hidden_dim, n_heads, size_name, seed, out_name):
    with open("src/configs/eval/ablation.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    if arch == "transformer":
        cfg["arch_defaults"]["transformer"]["use_positional_encoding"] = False
    
    size_cfg = cfg["graph_sizes"][size_name]
    num_police = size_cfg["num_police_agents"]
    num_agents_env = num_police + 1
    node_feature_size = num_agents_env + 1
    
    agent = build_agent(arch, n_layers, hidden_dim, n_heads, node_feature_size, cfg=cfg)
    
    ckpt_path = Path(cfg["checkpoint_dir"]) / f"{arch}_L{n_layers}_H{hidden_dim}_h{n_heads}_s{seed}_{size_name}" / "MrX.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
        
    state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    agent.load_state_dict(state_dict, strict=False)
    agent.model.eval()
    
    vis_config = {
        "visualize_game": False,
        "visualize_heatmap": False,
        "save_visualization": True,
        "save_dir": "src/artifacts/semester_contribution/vis_large"
    }
    
    class _SilentLogger:
        def log(self, *a, **kw): pass
        def log_scalar(self, *a, **kw): pass
        def close(self): pass

    env_wrappable = CustomEnvironment(
        number_of_agents=num_agents_env,
        agent_money=size_cfg["agent_money"],
        reward_weights=_FULL_REWARD_WEIGHTS,
        logger=_SilentLogger(),
        epoch=0,
        graph_nodes=size_cfg["graph_nodes"],
        graph_edges=size_cfg["graph_edges"],
        vis_configs=vis_config,
    )
    
    env = PettingZooWrapper(env=env_wrappable)
    
    print(f"Running episode for {arch} {size_name}...")
    state = env.reset(episode=0)
    done = False
    step_count = 0
    
    while not done and step_count < 30: 
        graph_data = create_graph_data(state, env).to(device)
        
        possible_moves_mrx = env_wrappable.get_possible_moves(0)
        action_mask_mrx = torch.zeros(graph_data.num_nodes, dtype=torch.int32, device=device)
        action_mask_mrx[possible_moves_mrx] = 1
        
        action_mrx = agent.select_action(graph_data, action_mask_mrx)
        if action_mrx is None: action_mrx = env_wrappable.MrX_pos[0]
        
        attention_data = getattr(agent, "last_attention", None)
        
        if attention_data is not None:
            env_wrappable.render(attention_data=attention_data[-1])
        else:
            env_wrappable.render()
            
        for i, agent_name in enumerate(env.possible_agents):
            if agent_name == "MrX":
                state[agent_name]["action"] = torch.tensor([action_mrx], dtype=torch.int64)
            else:
                moves = env_wrappable.get_possible_moves(i)
                p_act = random.choice(moves) if len(moves) > 0 else env_wrappable.DEFAULT_ACTION
                state[agent_name]["action"] = torch.tensor([p_act], dtype=torch.int64)
            
        state_stepped = env.step(state)
        next_state = step_mdp(state_stepped)
        
        rewards, terminations, truncations = extract_step_info(next_state, env.possible_agents)
        done = is_episode_done(terminations, truncations)
        state = next_state
        step_count += 1

    print(f"Episode finished in {step_count} steps. Saving GIF...")
    env_wrappable.save_visualizations()
    time.sleep(2)
    
    vis_dir = Path(vis_config["save_dir"])
    saved_gifs = list(vis_dir.glob("run_epoch_0-episode_1.gif"))
    if saved_gifs:
        dst = Path("src/artifacts/semester_contribution") / f"{out_name}.gif"
        if dst.exists(): os.remove(dst)
        os.rename(str(saved_gifs[0]), str(dst))
        print(f"Successfully saved: {dst}")

if __name__ == "__main__":
    vis_large = Path("src/artifacts/semester_contribution/vis_large")
    if vis_large.exists():
        for f in vis_large.glob("*.gif"): os.remove(f)
    vis_large.mkdir(parents=True, exist_ok=True)
    
    generate_gif(arch="gat", n_layers=3, hidden_dim=128, n_heads=4, size_name="large", seed=3, out_name="gat_large_attention")
    generate_gif(arch="transformer", n_layers=2, hidden_dim=64, n_heads=2, size_name="large", seed=1, out_name="transformer_large_attention")
