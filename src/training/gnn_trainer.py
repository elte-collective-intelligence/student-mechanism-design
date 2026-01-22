"""GNN agent training module."""

import torch
import random
import numpy as np

from logger import Logger
from agent.gnn_agent import GNNAgent
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
    Main training function for GNN agents.

    Trains GNN-based agents (MrX and Police) using graph neural networks
    with curriculum learning over multiple epochs and agent configurations.

    Args:
        args: Training configuration (epochs, episodes, graph size, etc.)
        agent_configs: Agent hyperparameters (lr, gamma, epsilon, etc.)
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

    logger.log("Logger initialized.", level="debug")

    # Initialize reward weight network for meta-learning
    reward_weight_net = RewardWeightNet().to(device)
    logger.log("RewardWeightNet initialized and moved to device.")

    optimizer = torch.optim.Adam(reward_weight_net.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    logger.log("Loss function (MSELoss) initialized.", level="debug")

    # Validate agent configurations
    if not hasattr(args, "agent_configurations") or not args.agent_configurations:
        raise ValueError(
            "args.agent_configurations must be a non-empty list of agent configurations."
        )

    # Compute average agent money for curriculum
    agent_money_values = np.asarray(
        [v["agent_money"] for v in args.agent_configurations]
    )
    average_agent_money = np.sum(agent_money_values) / agent_money_values.shape[0]
    logger.log(f"Average agent money: {average_agent_money}")

    # Create curriculum for progressive difficulty
    node_curriculum, edge_curriculum, money_curriculum = create_curriculum(
        args.epochs, args.graph_nodes, args.graph_edges, average_agent_money, 0.5
    )
    logger.log(f"Node curriculum: {node_curriculum}", level="info")
    logger.log(f"Edge curriculum: {edge_curriculum}", level="info")

    # Main training loop
    for epoch in range(args.epochs):
        logger.log(f"Starting epoch {epoch + 1}/{args.epochs}.", level="info")

        # Select random agent configuration
        selected_config = random.choice(args.agent_configurations)
        num_agents = selected_config["num_police_agents"] + 1
        agent_money = selected_config["agent_money"]

        logger.log(
            f"Chosen configuration: {num_agents} agents, {agent_money} money.",
            level="info",
        )
        logger.log_scalar("epoch/num_agents", num_agents)
        logger.log_scalar("epoch/agent_money", agent_money)

        # Predict reward weights from agent configuration
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

        logger.log(
            f"Epoch {epoch + 1}: Predicted weights: {reward_weights}", level="debug"
        )
        logger.log_weights(reward_weights)

        # Create environment with predicted difficulty
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
        logger.log(f"Environment created with weights {reward_weights}.", level="debug")

        # Initialize agents
        node_feature_size = env.number_of_agents + 1
        mrX_action_size = env.action_space("MrX").n
        police_action_size = env.action_space("Police0").n

        logger.log(
            f"Node feature size: {node_feature_size}, MrX action size: {mrX_action_size}, "
            f"Police action size: {police_action_size}",
            level="debug",
        )

        MrX_model_name = f"MrX_{node_feature_size}_agents"
        Police_model_name = f"Police_{node_feature_size}_agents"

        # Initialize GNN agents
        if agent_configs["agent_type"] == "gnn":
            mrX_agent = GNNAgent(
                node_feature_size=node_feature_size,
                device=device,
                gamma=agent_configs["gamma"],
                lr=agent_configs["lr"],
                batch_size=agent_configs["batch_size"],
                buffer_size=agent_configs["buffer_size"],
                epsilon=agent_configs["epsilon"],
                epsilon_decay=agent_configs["epsilon_decay"],
                epsilon_min=agent_configs["epsilon_min"],
            )
            if logger.model_exists(MrX_model_name):
                mrX_agent.load_state_dict(
                    logger.load_model(MrX_model_name), strict=False
                )

            police_agent = GNNAgent(
                node_feature_size=node_feature_size,
                device=device,
                gamma=agent_configs["gamma"],
                lr=agent_configs["lr"],
                batch_size=agent_configs["batch_size"],
                buffer_size=agent_configs["buffer_size"],
                epsilon=agent_configs["epsilon"],
                epsilon_decay=agent_configs["epsilon_decay"],
                epsilon_min=agent_configs["epsilon_min"],
            )
            if logger.model_exists(Police_model_name):
                police_agent.load_state_dict(
                    logger.load_model(Police_model_name), strict=False
                )
        else:
            mrX_agent = RandomAgent()
            police_agent = RandomAgent()

        logger.log("Agents initialized.", level="debug")

        # Episode training loop
        mrx_wins = 0
        police_wins = 0

        for episode in range(args.num_episodes):
            logger.log(
                f"Epoch {epoch + 1}, Episode {episode + 1} started.", level="info"
            )

            state = env.reset(episode=episode)
            done = False
            episode_step = 0
            episode_reward = 0.0
            episode_mrx_reward = 0.0
            episode_police_reward = 0.0

            while not done:
                # Prepare actions
                actions = {}

                # MrX action
                mrX_graph_data = create_graph_data(state, "MrX", env).to(device)
                mrX_possible_moves = env.get_possible_moves(0)
                mrX_action_mask = torch.zeros(
                    mrX_graph_data.num_nodes, dtype=torch.int32, device=device
                )
                mrX_action_mask[mrX_possible_moves] = 1
                mrX_action = mrX_agent.select_action(mrX_graph_data, mrX_action_mask)
                actions["MrX"] = mrX_action

                # Police actions
                for i in range(num_agents):
                    police_name = f"Police{i}"
                    police_graph_data = create_graph_data(state, police_name, env).to(
                        device
                    )
                    police_possible_moves = env.get_possible_moves(i + 1)
                    police_action_mask = torch.zeros(
                        police_graph_data.num_nodes, dtype=torch.int32, device=device
                    )
                    police_action_mask[police_possible_moves] = 1
                    police_action = police_agent.select_action(
                        police_graph_data, police_action_mask
                    )
                    if police_action is None:
                        police_action = env_wrappable.DEFAULT_ACTION
                    actions[police_name] = police_action

                # Execute actions - set them in the state TensorDict
                for obj_id, act in actions.items():
                    if act is not None:
                        state[obj_id]["action"] = torch.tensor([act], dtype=torch.int64)
                    else:
                        # Use default action if agent returns None
                        state[obj_id]["action"] = torch.tensor(
                            [env_wrappable.DEFAULT_ACTION], dtype=torch.int64
                        )

                state_stepped = env.step(state)
                next_state = step_mdp(state_stepped)

                # Extract episode info
                rewards, terminations, truncations = extract_step_info(
                    next_state, env.possible_agents
                )
                done = is_episode_done(terminations, truncations)

                # Track episode metrics
                episode_mrx_reward += rewards.get("MrX", 0.0)
                episode_police_reward += sum(
                    rewards.get(f"Police{i}", 0.0) for i in range(num_agents)
                )
                episode_reward += rewards.get("MrX", 0.0)
                episode_step += 1

                # Update agents immediately
                if agent_configs["agent_type"] == "gnn":
                    # Update MrX agent
                    mrX_next_graph_data = create_graph_data(next_state, "MrX", env).to(
                        device
                    )
                    mrX_agent.update(
                        mrX_graph_data,
                        mrX_action,
                        rewards.get("MrX", 0.0),
                        mrX_next_graph_data,
                        not terminations.get("Police0", False),
                    )

                    # Update Police agents
                    for i in range(num_agents):
                        police_name = f"Police{i}"
                        police_graph_data_stored = create_graph_data(
                            state, police_name, env
                        ).to(device)
                        police_next_graph_data = create_graph_data(
                            next_state, police_name, env
                        ).to(device)
                        police_agent.update(
                            police_graph_data_stored,
                            actions[police_name],
                            rewards.get(police_name, 0.0),
                            police_next_graph_data,
                            terminations.get(police_name, False),
                        )

                state = next_state

            # Track episode winner and log metrics
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
            logger.log_scalar(f"episode/steps", episode_step, episode)
            logger.log_scalar(f"episode/mrx_reward", episode_mrx_reward, episode)
            logger.log_scalar(f"episode/police_reward", episode_police_reward, episode)
            logger.log_scalar(f"episode/total_reward", episode_reward, episode)

            if winner == "MrX":
                mrx_wins += 1
            elif winner == "Police":
                police_wins += 1

        # Compute win rate
        total_games = mrx_wins + police_wins
        win_ratio = mrx_wins / total_games if total_games > 0 else 0.5

        logger.log(
            f"\n{'='*60}\n"
            f"Epoch {epoch + 1}/{args.epochs} Summary:\n"
            f"  MrX Wins: {mrx_wins}/{total_games} ({win_ratio*100:.1f}%)\n"
            f"  Police Wins: {police_wins}/{total_games} ({(1-win_ratio)*100:.1f}%)\n"
            f"  Target: 50% (balanced)\n"
            f"{'='*60}",
            level="info",
        )
        logger.log_scalar("epoch/mrx_wins", mrx_wins, epoch)
        logger.log_scalar("epoch/police_wins", police_wins, epoch)
        logger.log_scalar("epoch/win_ratio", win_ratio, epoch)

        # Update reward weight network (match original behavior)
        target_difficulty = torch.FloatTensor([[0.5]]).to(device).requires_grad_()
        win_ratio_tensor = torch.FloatTensor([[win_ratio]]).to(device).requires_grad_()

        loss = criterion(win_ratio_tensor, target_difficulty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log(
            f"Epoch {epoch + 1}: Loss: {loss.item()}, Win Ratio: {win_ratio}",
            level="info",
        )

        # Save models
        if agent_configs["agent_type"] == "gnn":
            logger.log_model(mrX_agent, MrX_model_name)
            logger.log_model(police_agent, Police_model_name)
            logger.log_model(reward_weight_net, "RewardWeightNet")
            logger.log(f"Models saved for epoch {epoch + 1}.", level="info")

    logger.log("Training complete.", level="info")
    logger.close()
