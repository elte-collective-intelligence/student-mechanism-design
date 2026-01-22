"""MAPPO agent training module."""

import torch
import random
import numpy as np

from logger import Logger
from agent.mappo_agent import MappoAgent
from agent.random_agent import RandomAgent
from reward_net import RewardWeightNet
from environment.yard import CustomEnvironment
from torchrl.envs.libs.pettingzoo import PettingZooWrapper

from training.utils import (
    device,
    create_curriculum,
    extract_step_info,
    is_episode_done,
)

from torchrl.envs import step_mdp


def train_mappo(args, agent_configs, logger_configs, visualization_configs):
    """
    Main training function for MAPPO agents.

    Trains Multi-Agent Proximal Policy Optimization (MAPPO) agents
    for both MrX and Police using actor-critic architecture.

    Args:
        args: Training configuration (epochs, episodes, graph size, etc.)
        agent_configs: Agent hyperparameters (lr, gamma, clip_param, etc.)
        logger_configs: Logging configuration (WandB, TensorBoard)
        visualization_configs: Visualization settings
    """
    logger = Logger(
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_resume=args.wandb_resume,
        configs=logger_configs,
    )

    logger.log("MAPPO Training: Logger initialized.", level="debug")

    # Initialize reward weight network
    reward_weight_net = RewardWeightNet().to(device)
    logger.log("RewardWeightNet initialized.")

    optimizer = torch.optim.Adam(reward_weight_net.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Validate configurations
    if not hasattr(args, "agent_configurations") or not args.agent_configurations:
        raise ValueError("args.agent_configurations must be provided.")

    agent_money_values = np.asarray(
        [v["agent_money"] for v in args.agent_configurations]
    )
    average_agent_money = np.mean(agent_money_values)
    logger.log(f"Average agent money: {average_agent_money}")

    node_curriculum, edge_curriculum, money_curriculum = create_curriculum(
        args.epochs, args.graph_nodes, args.graph_edges, average_agent_money, 0.5
    )
    logger.log(f"MAPPO curriculum created", level="info")

    # Training loop
    for epoch in range(args.epochs):
        logger.log(f"MAPPO Epoch {epoch + 1}/{args.epochs} started.", level="info")

        selected_config = random.choice(args.agent_configurations)
        num_agents = selected_config["num_police_agents"] + 1
        agent_money = selected_config["agent_money"]

        logger.log(
            f"Configuration: {num_agents} agents, {agent_money} money.", level="info"
        )

        # Predict reward weights
        inputs = torch.FloatTensor(
            [[num_agents, agent_money, args.graph_nodes, args.graph_edges]]
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

        logger.log_weights(reward_weights)

        # Create environment
        env_wrappable = CustomEnvironment(
            number_of_agents=num_agents,
            agent_money=agent_money,
            reward_weights=reward_weights,
            logger=logger,
            epoch=epoch,
            graph_nodes=args.graph_nodes,
            graph_edges=args.graph_edges,
            vis_configs=visualization_configs,
        )

        env = PettingZooWrapper(env=env_wrappable)

        # Get observation dimensions
        sample_obs = env.reset(episode=0)
        mrx_features = sample_obs["MrX"]["observation"]["node_features"]
        feature_dim = mrx_features.shape[1]
        global_obs_dim = feature_dim * (num_agents + 1)
        max_action_dim = args.graph_nodes

        # Initialize MAPPO agents
        if agent_configs["agent_type"] == "mappo":
            mrX_agent = MappoAgent(
                n_agents=1,
                obs_size=feature_dim,
                global_obs_size=global_obs_dim,
                action_size=max_action_dim,
                hidden_size=agent_configs["hidden_size"],
                device=device,
                lr=agent_configs.get("lr", 3e-4),
                gamma=agent_configs.get("gamma", 0.99),
                epsilon=agent_configs.get("epsilon", 0.2),
            )

            police_agent = MappoAgent(
                n_agents=num_agents,
                obs_size=feature_dim,
                global_obs_size=global_obs_dim,
                action_size=max_action_dim,
                hidden_size=agent_configs["hidden_size"],
                device=device,
                lr=agent_configs.get("lr", 3e-4),
                gamma=agent_configs.get("gamma", 0.99),
                epsilon=agent_configs.get("epsilon", 0.2),
            )
        else:
            mrX_agent = RandomAgent()
            police_agent = RandomAgent()

        # Episode training
        mrx_wins = 0
        police_wins = 0

        for episode in range(args.num_episodes):
            logger.log(f"MAPPO Epoch {epoch + 1}, Episode {episode + 1}", level="info")

            state = env.reset(episode=episode)
            done = False
            episode_step = 0
            episode_mrx_reward = 0.0
            episode_police_reward = 0.0

            while not done:
                actions = {}
                obs_list = []
                log_probs_list = []

                # MrX action
                mrx_obs = state["MrX"]["observation"]["MrX_pos"]
                mrx_obs = (
                    mrx_obs.detach().clone().to(dtype=torch.float32, device=device)
                )

                # Get possible moves and create action mask
                possible_moves = env.get_possible_moves(0)
                mrX_action_mask = torch.zeros(
                    max_action_dim, dtype=torch.float32, device=device
                )
                for move in possible_moves:
                    if move < max_action_dim:
                        mrX_action_mask[move] = 1.0

                mrX_action, mrX_log_prob, _ = mrX_agent.select_action(
                    0, mrx_obs, mrX_action_mask
                )
                actions["MrX"] = mrX_action
                obs_list.append(mrx_obs)
                log_probs_list.append(mrX_log_prob)

                # Police actions
                for i in range(num_agents):
                    police_name = f"Police{i}"
                    police_obs = state[police_name]["observation"]["Polices_pos"].sum(
                        dim=1
                    )
                    police_obs = (
                        police_obs.detach()
                        .clone()
                        .to(dtype=torch.float32, device=device)
                    )

                    possible_moves = env.get_possible_moves(i + 1)
                    police_action_mask = torch.zeros(
                        max_action_dim, dtype=torch.float32, device=device
                    )
                    for move in possible_moves:
                        if move < max_action_dim:
                            police_action_mask[move] = 1.0

                    police_action, police_log_prob, _ = police_agent.select_action(
                        i, police_obs, police_action_mask
                    )
                    actions[police_name] = police_action
                    obs_list.append(police_obs)
                    log_probs_list.append(police_log_prob)

                # Set actions in state TensorDict
                for obj_id, act in actions.items():
                    if act is not None:
                        state[obj_id]["action"] = torch.tensor([act], dtype=torch.int64)
                    else:
                        state[obj_id]["action"] = torch.tensor(
                            [env_wrappable.DEFAULT_ACTION], dtype=torch.int64
                        )

                # Execute step
                state_stepped = env.step(state)
                next_state = step_mdp(state_stepped)

                # Extract rewards and done flags
                rewards, terminations, truncations = extract_step_info(
                    next_state, env.possible_agents
                )
                done = is_episode_done(terminations, truncations)

                # Prepare rewards and dones lists
                rewards_list = [rewards.get("MrX", 0.0)] + [
                    rewards.get(f"Police{i}", 0.0) for i in range(num_agents)
                ]
                dones_list = [
                    terminations.get("MrX", False) or truncations.get("MrX", False)
                ] + [
                    terminations.get(f"Police{i}", False)
                    or truncations.get(f"Police{i}", False)
                    for i in range(num_agents)
                ]

                # Track episode metrics
                episode_mrx_reward += rewards.get("MrX", 0.0)
                episode_police_reward += sum(
                    rewards.get(f"Police{i}", 0.0) for i in range(num_agents)
                )
                episode_step += 1

                # Create global observation from all agent observations
                processed_obs_list = []
                for obs in obs_list:
                    obs = obs.squeeze().unsqueeze(0)
                    processed_obs_list.append(obs)
                global_obs = torch.cat(processed_obs_list)

                # Store experiences
                mrX_agent.store(
                    [obs_list[0]],
                    global_obs,
                    [actions["MrX"]],
                    [rewards_list[0]],
                    [log_probs_list[0]],
                    [dones_list[0]],
                )

                for i in range(num_agents):
                    police_name = f"Police{i}"
                    police_agent.store(
                        [obs_list[i + 1]],
                        global_obs,
                        [actions[police_name]],
                        [rewards_list[i + 1]],
                        [log_probs_list[i + 1]],
                        [dones_list[i + 1]],
                    )

                state = next_state

            # Update policies after episode
            mrX_agent.ppo_update()
            police_agent.ppo_update()

            # Track winner and log metrics
            winner = env_wrappable.current_winner
            logger.log(
                f"Episode {episode + 1} finished. Winner: {winner}, "
                f"Steps: {episode_step}, MrX Reward: {episode_mrx_reward:.2f}, "
                f"Police Reward: {episode_police_reward:.2f}",
                level="info",
            )

            # Save visualizations if enabled
            env_wrappable.save_visualizations()

            # Log to tensorboard
            logger.log_scalar(f"mappo/episode_steps", episode_step, episode)
            logger.log_scalar(f"mappo/episode_mrx_reward", episode_mrx_reward, episode)
            logger.log_scalar(
                f"mappo/episode_police_reward", episode_police_reward, episode
            )

            if winner == "MrX":
                mrx_wins += 1
            elif winner == "Police":
                police_wins += 1

        # Compute metrics
        total_games = mrx_wins + police_wins
        win_ratio = mrx_wins / total_games if total_games > 0 else 0.5

        logger.log(
            f"\n{'='*60}\n"
            f"MAPPO Epoch {epoch + 1}/{args.epochs} Summary:\n"
            f"  MrX Wins: {mrx_wins}/{total_games} ({win_ratio*100:.1f}%)\n"
            f"  Police Wins: {police_wins}/{total_games} ({(1-win_ratio)*100:.1f}%)\n"
            f"  Target: 50% (balanced)\n"
            f"{'='*60}",
            level="info",
        )
        logger.log_scalar("mappo/mrx_wins", mrx_wins, epoch)
        logger.log_scalar("mappo/police_wins", police_wins, epoch)
        logger.log_scalar("mappo/win_ratio", win_ratio, epoch)

        # Update meta-learner
        target_difficulty = torch.FloatTensor([[0.5]]).to(device).requires_grad_()
        win_ratio_tensor = torch.FloatTensor([[win_ratio]]).to(device).requires_grad_()

        loss = criterion(win_ratio_tensor, target_difficulty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log(
            f"MAPPO Epoch {epoch + 1}: Loss: {loss.item()}, Win Ratio: {win_ratio}",
            level="info",
        )

        # Save models
        logger.log_model(mrX_agent, f"MAPPO_MrX_{epoch}")
        logger.log_model(police_agent, f"MAPPO_Police_{epoch}")
        logger.log_model(reward_weight_net, "MAPPO_RewardWeightNet")

        logger.log(
            f"MAPPO Epoch {epoch + 1} complete. Win ratio: {win_ratio:.3f}",
            level="info",
        )

    logger.log("MAPPO Training complete.", level="info")
    logger.close()
