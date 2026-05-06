"""Unified agent training module for GNN, GAT, and Transformer agents."""

import torch
import random
import numpy as np
from collections import deque

from logger import Logger
from agent.gnn_agent import GNNAgent
from agent.gat_agent import GATAgent
from agent.transformer_agent import TransformerAgent
from agent.random_agent import RandomAgent
from reward_net import RewardWeightNet
from environment.yard import CustomEnvironment
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs import step_mdp

from training.utils import (
    device,
    create_curriculum,
    create_graph_data,
    extract_step_info,
    is_episode_done,
)


def train_gnn(args, agent_configs, logger_configs, visualization_configs):
    """
    Main training function for Graph-based RL agents.

    Supports GNN, GAT, and Transformer architectures with curriculum learning
    and meta-learned reward weighting.
    """
    logger = Logger(
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_resume=args.wandb_resume,
        configs=logger_configs,
    )

    logger.log("Logger initialized.", level="debug")

    # Initialize reward weight network for meta-learning
    reward_weight_net = RewardWeightNet().to(device)
    logger.log("RewardWeightNet initialized.")

    meta_optimizer = torch.optim.Adam(reward_weight_net.parameters(), lr=0.001)
    meta_criterion = torch.nn.MSELoss()

    # Validate agent configurations
    if not hasattr(args, "agent_configurations") or not args.agent_configurations:
        raise ValueError("args.agent_configurations must be a non-empty list.")

    # Compute curriculum values
    agent_money_values = np.asarray(
        [v["agent_money"] for v in args.agent_configurations]
    )
    average_agent_money = np.sum(agent_money_values) / agent_money_values.shape[0]

    node_curriculum, edge_curriculum, money_curriculum = create_curriculum(
        args.epochs, args.graph_nodes, args.graph_edges, average_agent_money, 0.5
    )

    # Main training loop
    for epoch in range(args.epochs):
        logger.log(f"Starting epoch {epoch + 1}/{args.epochs}.", level="info")

        # Select random agent configuration for this epoch
        selected_config = random.choice(args.agent_configurations)
        num_police = selected_config["num_police_agents"]
        num_agents_total = num_police + 1  # Police + MrX
        agent_money = selected_config["agent_money"]

        # Meta-learning: Predict reward weights
        inputs = torch.FloatTensor(
            [[num_agents_total, agent_money, args.graph_nodes, args.graph_edges]]
        ).to(device)
        predicted_weight = reward_weight_net(inputs)

        reward_weights = {
            "Police_distance": predicted_weight[0, 0],
            "Police_group": predicted_weight[0, 1],
            "Police_position": predicted_weight[0, 2],
            "Police_time": predicted_weight[0, 3],
            "Mrx_closest": predicted_weight[0, 4],
            "Mrx_average": predicted_weight[0, 5],
            "Mrx_position": predicted_weight[0, 6],
            "Mrx_time": predicted_weight[0, 7],
            "Police_coverage": predicted_weight[0, 8],
            "Police_proximity": predicted_weight[0, 9],
            "Police_overlap_penalty": predicted_weight[0, 10],
        }

        # Initialize environment
        env_wrappable = CustomEnvironment(
            number_of_agents=num_agents_total,
            agent_money=agent_money,
            reward_weights=reward_weights,
            logger=logger,
            epoch=epoch,
            graph_nodes=args.graph_nodes,
            graph_edges=args.graph_edges,
            vis_configs=visualization_configs,
        )
        env = PettingZooWrapper(env=env_wrappable)

        # Agent Factory
        node_feature_size = num_agents_total + 1
        agent_type = agent_configs.get("agent_type", "gnn").lower()

        common_params = {
            "node_feature_size": node_feature_size,
            "device": device,
            "gamma": agent_configs.get("gamma", 0.99),
            "lr": agent_configs.get("lr", 1e-3),
            "batch_size": agent_configs.get("batch_size", 64),
            "buffer_size": agent_configs.get("buffer_size", 10000),
            "epsilon": agent_configs.get("epsilon", 1.0),
            "epsilon_decay": agent_configs.get("epsilon_decay", 0.995),
            "epsilon_min": agent_configs.get("epsilon_min", 0.01),
        }

        if agent_type == "gnn":
            mrX_agent = GNNAgent(**common_params)
            police_agent = GNNAgent(**common_params)
        elif agent_type == "gat":
            mrX_agent = GATAgent(**common_params)
            police_agent = GATAgent(**common_params)
        elif agent_type == "transformer":
            common_params["pos_dim"] = agent_configs.get("pos_dim", 8)
            mrX_agent = TransformerAgent(**common_params)
            police_agent = TransformerAgent(**common_params)
        else:
            mrX_agent = RandomAgent()
            police_agent = RandomAgent()

        # Load existing weights
        MrX_model_name = f"MrX_{agent_type}_{node_feature_size}"
        Police_model_name = f"Police_{agent_type}_{node_feature_size}"

        if agent_type != "random":
            if logger.model_exists(MrX_model_name):
                mrX_agent.load_state_dict(
                    logger.load_model(MrX_model_name), strict=False
                )
            if logger.model_exists(Police_model_name):
                police_agent.load_state_dict(
                    logger.load_model(Police_model_name), strict=False
                )

        # Episode Loop
        mrx_wins, police_wins = 0, 0

        for episode in range(args.num_episodes):
            state = env.reset(episode=episode)
            done = False
            ep_step, ep_mrx_rew, ep_police_rew = 0, 0.0, 0.0

            while not done:
                actions = {}

                # MrX Turn
                mrx_graph = create_graph_data(state, "MrX", env).to(device)
                mrx_mask = torch.zeros(
                    mrx_graph.num_nodes, dtype=torch.int32, device=device
                )
                mrx_mask[env.get_possible_moves(0)] = 1
                mrx_act = mrX_agent.select_action(mrx_graph, mrx_mask)
                actions["MrX"] = mrx_act

                # Police Turns
                for i in range(num_police):
                    p_name = f"Police{i}"
                    p_graph = create_graph_data(state, p_name, env).to(device)
                    p_mask = torch.zeros(
                        p_graph.num_nodes, dtype=torch.int32, device=device
                    )
                    p_mask[env.get_possible_moves(i + 1)] = 1
                    p_act = police_agent.select_action(p_graph, p_mask)
                    actions[p_name] = (
                        p_act if p_act is not None else env_wrappable.DEFAULT_ACTION
                    )

                # Apply Actions
                for obj_id, act in actions.items():
                    state[obj_id]["action"] = torch.tensor(
                        [act if act is not None else 0], dtype=torch.int64
                    )

                # Step Environment
                state_stepped = env.step(state)
                next_state = step_mdp(state_stepped)
                rewards, terminations, truncations = extract_step_info(
                    next_state, env.possible_agents
                )
                done = is_episode_done(terminations, truncations)

                # Training Updates (Skip for RandomAgent)
                if agent_type != "random":
                    # MrX Update
                    mrx_next_graph = create_graph_data(next_state, "MrX", env).to(
                        device
                    )
                    mrX_agent.update(
                        mrx_graph,
                        mrx_act,
                        rewards.get("MrX", 0.0),
                        mrx_next_graph,
                        terminations.get("MrX", False) or truncations.get("MrX", False),
                    )

                    # Police Update
                    for i in range(num_police):
                        p_name = f"Police{i}"
                        p_graph = create_graph_data(state, p_name, env).to(device)
                        p_next_graph = create_graph_data(next_state, p_name, env).to(
                            device
                        )
                        police_agent.update(
                            p_graph,
                            actions[p_name],
                            rewards.get(p_name, 0.0),
                            p_next_graph,
                            terminations.get(p_name, False)
                            or truncations.get(p_name, False),
                        )

                # Metrics
                ep_mrx_rew += rewards.get("MrX", 0.0)
                ep_police_rew += sum(
                    rewards.get(f"Police{i}", 0.0) for i in range(num_police)
                )
                ep_step += 1
                state = next_state

            # Tally Winners
            winner = env_wrappable.current_winner
            if winner == "MrX":
                mrx_wins += 1
            elif winner == "Police":
                police_wins += 1

            env_wrappable.save_visualizations()

        # Epoch Meta-Update: Adjust reward weights toward a 50% win balance
        total_games = mrx_wins + police_wins
        win_ratio = mrx_wins / total_games if total_games > 0 else 0.5

        win_ratio_t = torch.FloatTensor([[win_ratio]]).to(device).requires_grad_()
        target_t = torch.FloatTensor([[0.5]]).to(device)

        meta_loss = meta_criterion(win_ratio_t, target_t)
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        # Logging & Saving
        logger.log_scalar("epoch/win_ratio", win_ratio, epoch)
        if agent_type != "random":
            logger.log_model(mrX_agent, MrX_model_name)
            logger.log_model(police_agent, Police_model_name)
            logger.log_model(reward_weight_net, "RewardWeightNet")

    logger.log("Training complete.", level="info")
    logger.close()
