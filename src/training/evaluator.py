"""Evaluation module for trained agents."""

import torch
import random

from logger import Logger
from agent.gnn_agent import GNNAgent
from agent.random_agent import RandomAgent
from environment.yard import CustomEnvironment
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs import step_mdp

from training.utils import (
    device,
    create_graph_data,
    extract_step_info,
    is_episode_done,
)


def evaluate_gnn(args, agent_configs, logger_configs, visualization_configs):
    """
    Evaluate trained GNN agents.

    Runs evaluation episodes with trained models to assess performance
    on held-out configurations or different graph distributions.

    Args:
        args: Evaluation configuration
        agent_configs: Agent hyperparameters
        logger_configs: Logging configuration
        visualization_configs: Visualization settings
    """
    logger = Logger(
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name + "_eval",
        wandb_resume=False,
        configs=logger_configs,
    )

    logger.log("Evaluation: Logger initialized.", level="info")

    # Validation
    if not hasattr(args, "agent_configurations") or not args.agent_configurations:
        raise ValueError("args.agent_configurations must be provided for evaluation.")

    # Evaluation loop over configurations
    for config_idx, config in enumerate(args.agent_configurations):
        num_agents = config["num_police_agents"] + 1
        agent_money = config["agent_money"]

        logger.log(
            f"Evaluating configuration {config_idx + 1}: {num_agents} agents, {agent_money} money",
            level="info",
        )

        # Use default reward weights for evaluation
        reward_weights = {
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

        # Create environment
        env_wrappable = CustomEnvironment(
            number_of_agents=num_agents,
            agent_money=agent_money,
            reward_weights=reward_weights,
            logger=logger,
            epoch=0,
            graph_nodes=args.graph_nodes,
            graph_edges=args.graph_edges,
            vis_configs=visualization_configs,
        )

        env = PettingZooWrapper(env=env_wrappable)

        # Initialize agents
        node_feature_size = env.number_of_agents + 1
        mrX_action_size = env.action_space("MrX").n
        police_action_size = env.action_space("Police0").n

        MrX_model_name = f"MrX_{node_feature_size}_agents"
        Police_model_name = f"Police_{node_feature_size}_agents"

        # Load trained models
        if agent_configs["agent_type"] == "gnn":
            mrX_agent = GNNAgent(
                node_feature_size=node_feature_size,
                device=device,
                gamma=agent_configs["gamma"],
                lr=agent_configs["lr"],
                batch_size=agent_configs["batch_size"],
                buffer_size=agent_configs["buffer_size"],
                epsilon=0.0,  # No exploration during evaluation
                epsilon_decay=1.0,
                epsilon_min=0.0,
            )

            if logger.model_exists(MrX_model_name):
                mrX_agent.load_state_dict(
                    logger.load_model(MrX_model_name), strict=False
                )
                logger.log(f"Loaded MrX model: {MrX_model_name}", level="info")
            else:
                logger.log(
                    f"Warning: MrX model {MrX_model_name} not found. Using untrained agent.",
                    level="warning",
                )

            police_agent = GNNAgent(
                node_feature_size=node_feature_size,
                device=device,
                gamma=agent_configs["gamma"],
                lr=agent_configs["lr"],
                batch_size=agent_configs["batch_size"],
                buffer_size=agent_configs["buffer_size"],
                epsilon=0.0,  # No exploration
                epsilon_decay=1.0,
                epsilon_min=0.0,
            )

            if logger.model_exists(Police_model_name):
                police_agent.load_state_dict(
                    logger.load_model(Police_model_name), strict=False
                )
                logger.log(f"Loaded Police model: {Police_model_name}", level="info")
            else:
                logger.log(
                    f"Warning: Police model {Police_model_name} not found. Using untrained agent.",
                    level="warning",
                )
        else:
            mrX_agent = RandomAgent()
            police_agent = RandomAgent()

        # Run evaluation episodes
        mrx_wins = 0
        police_wins = 0
        total_steps = []

        for episode in range(args.num_eval_episodes):
            logger.log(
                f"Evaluation episode {episode + 1}/{args.num_eval_episodes}",
                level="info",
            )

            state = env.reset(episode=episode)
            done = False
            episode_steps = 0

            while not done:
                actions = {}

                # MrX action (no exploration)
                mrX_graph_data = create_graph_data(state, "MrX", env).to(device)
                mrX_possible_moves = env.get_possible_moves(0)
                mrX_action_mask = torch.zeros(
                    mrX_graph_data.num_nodes, dtype=torch.int32, device=device
                )
                mrX_action_mask[mrX_possible_moves] = 1
                mrX_action = mrX_agent.select_action(mrX_graph_data, mrX_action_mask)
                actions["MrX"] = mrX_action

                # Police actions (no exploration)
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

                state = next_state
                episode_steps += 1

            # Track results
            winner = env_wrappable.current_winner
            if winner == "MrX":
                mrx_wins += 1
            elif winner == "Police":
                police_wins += 1

            # Save visualizations if enabled
            env_wrappable.save_visualizations()

            total_steps.append(episode_steps)
            logger.log(
                f"Episode {episode + 1} complete. Winner: {winner}, Steps: {episode_steps}",
                level="info",
            )

        # Compute and log metrics
        total_games = mrx_wins + police_wins
        win_rate = mrx_wins / total_games if total_games > 0 else 0.0
        avg_steps = sum(total_steps) / len(total_steps) if total_steps else 0

        logger.log(f"Configuration {config_idx + 1} Results:", level="info")
        logger.log(f"  MrX wins: {mrx_wins}/{args.num_eval_episodes}", level="info")
        logger.log(
            f"  Police wins: {police_wins}/{args.num_eval_episodes}", level="info"
        )
        logger.log(f"  Win rate: {win_rate:.3f}", level="info")
        logger.log(f"  Average steps: {avg_steps:.1f}", level="info")

        logger.log_scalar(f"eval/config_{config_idx}/mrx_wins", mrx_wins)
        logger.log_scalar(f"eval/config_{config_idx}/police_wins", police_wins)
        logger.log_scalar(f"eval/config_{config_idx}/win_rate", win_rate)
        logger.log_scalar(f"eval/config_{config_idx}/avg_steps", avg_steps)

    logger.log("Evaluation complete.", level="info")
    logger.close()


def evaluate_mappo(args, agent_configs, logger_configs, visualization_configs):
    """
    Evaluate trained MAPPO agents.

    Similar to evaluate_gnn but for MAPPO agents.
    """
    logger = Logger(
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name + "_mappo_eval",
        wandb_resume=False,
        configs=logger_configs,
    )

    logger.log("MAPPO Evaluation started.", level="info")

    # Implementation similar to evaluate_gnn but for MAPPO
    # (Placeholder for brevity - would follow same pattern)

    logger.log("MAPPO Evaluation complete.", level="info")
    logger.close()
