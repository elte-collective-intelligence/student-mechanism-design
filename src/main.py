import torch, re, os, random

import torch.nn as nn
import torch.optim as optim
import numpy as np
from logger import Logger  # Your custom Logger class
from RLAgent.gnn_agent import GNNAgent
from RLAgent.mappo_agent import MappoAgent
from RLAgent.random_agent import RandomAgent
from reward_net import RewardWeightNet
from Enviroment.yard import CustomEnvironment
from torch_geometric.data import Data

from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs import step_mdp

# Define the device at the beginning
print(f"CUDA is available: {torch.cuda.is_available()}")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")  # You may consider logging this instead
class RewardWeightNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, output_size=11):
        super(RewardWeightNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
def create_curriculum(num_epochs, base_graph_nodes,base_graph_edges, base_money,curriculum_range):
    if num_epochs <= 1:
        return np.asarray([base_graph_nodes]),np.asarray([base_graph_edges])
    node_curriculum = np.arange(base_graph_nodes - curriculum_range * base_graph_nodes,base_graph_nodes + curriculum_range * base_graph_nodes + 1,((base_graph_nodes + curriculum_range * base_graph_nodes) - (base_graph_nodes - curriculum_range * base_graph_nodes))/max(num_epochs-1,1))
    edge_curriculum = np.arange(base_graph_edges - curriculum_range * base_graph_edges,base_graph_edges + curriculum_range * base_graph_edges + 1,((base_graph_edges + curriculum_range * base_graph_edges) - (base_graph_edges - curriculum_range * base_graph_edges))/max(num_epochs-1,1))
    money_curriculum = np.arange(base_money + curriculum_range * base_money,base_money - curriculum_range * base_money - 1,-((base_money + curriculum_range * base_money) - (base_money - curriculum_range * base_money))/max(num_epochs-1,1))
    return node_curriculum,edge_curriculum,money_curriculum
def modify_curriculum(win_ratio,node_curriculum,edge_curriculum,money_curriculum,modification_rate):
    modification_percentage = 1.0 + (2.0 * modification_rate) * win_ratio - modification_rate
    return node_curriculum * modification_percentage,edge_curriculum * modification_percentage,money_curriculum * modification_percentage
def train(args,agent_configs,logger_configs,visualization_configs):
    """
    Main training function:
    - Initializes the logger, networks (including Meta RL net for reward weights), and optimizer.
    - Iterates over epochs:
        - Chooses an agent/environment configuration.
        - Predicts new reward weights via the RewardWeightNet (Meta RL).
        - Inside loop: runs episodes where agents (MrX & Police) select actions based on GNN.
        - Steps through the environment, gathers rewards, and updates agents.
        - After episodes, evaluates performance and updates RewardWeightNet towards target difficulty.
    """
    logger = Logger(
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_resume=args.wandb_resume,
        configs=logger_configs
    )

    logger.log("Logger initialized.", level="debug")

    # Initialize DifficultyNet and move it to the GPU
    reward_weight_net = RewardWeightNet().to(device)
    logger.log("DifficultyNet initialized and moved to device.")

    optimizer = optim.Adam(reward_weight_net.parameters(), lr=0.001)

    criterion = nn.MSELoss()
    logger.log("Loss function (MSELoss) initialized.", level="debug")

    logger.log(f"Starting training with variable agents and money settings.", level="debug")
    # Validate that the agent configurations list is provided and not empty
    if not hasattr(args, 'agent_configurations') or not args.agent_configurations:
        raise ValueError("args.agent_configurations must be a non-empty list of (num_agents, agent_money) tuples.")
    agent_money_values = np.asarray([v['agent_money'] for v in args.agent_configurations])
    average_agent_money = np.sum(agent_money_values)/agent_money_values.shape[0]
    logger.log(f"Average agent money: {average_agent_money}")
    node_curriculum,edge_curriculum,money_curriculum = create_curriculum(args.epochs,args.graph_nodes,args.graph_edges,average_agent_money,0.5)
    logger.log(f"Node curriculum: {node_curriculum}",level="info")
    logger.log(f"Edge curriculum: {edge_curriculum}",level="info")
    for epoch in range(args.epochs):
        logger.log(f"Starting epoch {epoch + 1}/{args.epochs}.", level="info")
        # Randomly select a (num_agents, agent_money) tuple from the predefined list
        selected_config = random.choice(args.agent_configurations)
        num_agents, agent_money = selected_config["num_police_agents"] + 1, selected_config["agent_money"]  # Unpack the tuple
        logger.log(f"Choosen configuration: {num_agents} agents, {agent_money} money.", level="info")
        logger.log_scalar('epoch/num_agents', num_agents)
        logger.log_scalar('epoch/agent_money', agent_money)
        # Predict the difficulty from the number of agents and money
        inputs = torch.FloatTensor([[num_agents, agent_money, args.graph_nodes, args.graph_edges]]).to(
            device)  # Move inputs to GPU
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
            "Police_overlap_penalty": predicted_weight[0, 10]
        }

        logger.log(f"Epoch {epoch + 1}: Predicted weights: {reward_weights}", level="debug")
        logger.log_weights(reward_weights)
        # Create environment with predicted difficulty
        env_wrappable = CustomEnvironment(
            number_of_agents=num_agents,
            agent_money=agent_money,
            reward_weights=reward_weights,
            logger=logger,
            epoch=epoch,
            graph_nodes=int(node_curriculum[epoch]),
            graph_edges=int(edge_curriculum[epoch]),
            vis_configs=visualization_configs
        )

        env = PettingZooWrapper(env=env_wrappable)

        logger.log(f"Environment created with weights {reward_weights}.", level="debug")

        # Determine node feature size from the environment
        node_feature_size = env.number_of_agents + 1  # Assuming node features exist
        mrX_action_size = env.action_space('MrX').n
        police_action_size = env.action_space('Police0').n  # Assuming all police have the same action space
        logger.log(
            f"Node feature size: {node_feature_size}, MrX action size: {mrX_action_size}, Police action size: {police_action_size}",
            level="debug")

        MrX_model_name = f'MrX_{node_feature_size}_agents'
        Police_model_name = f'Police_{node_feature_size}_agents'

        # Initialize GNN agents with graph-specific parameters and move them to GPU
        if agent_configs["agent_type"] == "gnn":
            mrX_agent = GNNAgent(node_feature_size=node_feature_size, device=device, gamma=agent_configs["gamma"],
                                 lr=agent_configs["lr"], batch_size=agent_configs["batch_size"],
                                 buffer_size=agent_configs["buffer_size"], epsilon=agent_configs["epsilon"],
                                 epsilon_decay=agent_configs["epsilon_decay"], epsilon_min=agent_configs["epsilon_min"])
            if logger.model_exists(MrX_model_name):
                mrX_agent.load_state_dict(logger.load_model(MrX_model_name), strict=False)
        elif agent_configs["agent_type"] == "mappo":
            pass
        else:
            mrX_agent = RandomAgent()
        if agent_configs["agent_type"] == "gnn":
            police_agent = GNNAgent(node_feature_size=node_feature_size, device=device, gamma=agent_configs["gamma"],
                                    lr=agent_configs["lr"], batch_size=agent_configs["batch_size"],
                                    buffer_size=agent_configs["buffer_size"], epsilon=agent_configs["epsilon"],
                                    epsilon_decay=agent_configs["epsilon_decay"],
                                    epsilon_min=agent_configs["epsilon_min"])
            if logger.model_exists(Police_model_name):
                police_agent.load_state_dict(logger.load_model(Police_model_name), strict=False)
        else:
            police_agent = RandomAgent()
        logger.log("GNN agents for MrX and Police initialized.", level="debug")

        # Train the MrX and Police agents in the environment
        for episode in range(args.num_episodes):
            logger.log(f"Epoch {epoch + 1}, Episode {episode + 1} started.", level="info")
            state = env.reset(episode=episode)
            done = False
            total_reward = 0

            while not done:
                # Create graph data for GNN and move to GPU
                mrX_graph = create_graph_data(state, 'MrX', env).to(device)
                police_graphs = [
                    create_graph_data(state, f'Police{i}', env).to(device)
                    for i in range(num_agents)
                ]
                logger.log(f"Created graph data for MrX and Police agents.", level="debug")

                # MrX selects an action
                # mrX_action = mrX_agent.select_action(mrX_graph, torch.ones(mrX_action_size, device=device))

                mrX_action_size = env.action_space('MrX').n
                mrX_possible_moves = env.get_possible_moves(0)
                action_mask = torch.zeros(mrX_graph.num_nodes, dtype=torch.int32, device=device)
                action_mask[mrX_possible_moves] = 1
                mrX_action = mrX_agent.select_action(mrX_graph, action_mask)
                logger.log(f"MrX selected action: {mrX_action}", level="debug")

                # Police agents select actions
                agent_actions = {'MrX': mrX_action}
                for i in range(num_agents):
                    police_action_size = env.action_space(f'Police{i}').n
                    police_possible_moves = env.get_possible_moves(i + 1)
                    action_mask = torch.zeros(police_graphs[i].num_nodes, dtype=torch.int32, device=device)
                    action_mask[police_possible_moves] = 1
                    police_action = police_agent.select_action(
                        police_graphs[i],
                        action_mask
                    )
                    if police_action is None:
                        police_action = env.DEFAULT_ACTION
                    agent_actions[f'Police{i}'] = police_action
                    logger.log(f"Police{i} selected action: {police_action}", level="debug")
                # Execute actions for MrX and Police
                for obj_id, act in agent_actions.items():
                    if act is not None:
                        state[obj_id]["action"] = torch.tensor([act], dtype=torch.int64)

                state_stepped = env.step(state)
                next_state = step_mdp(state_stepped)

                rewards = {agent_id: next_state[agent_id]['reward'].squeeze() for agent_id in env.possible_agents}
                terminations = {agent_id: next_state[agent_id]['terminated'].squeeze() for agent_id in
                                env.possible_agents}
                truncation = {agent_id: next_state[agent_id]['truncated'].squeeze() for agent_id in env.possible_agents}
                logger.log(
                    f"Executed actions. Rewards: {rewards}, Terminations: {terminations}, Truncations: {truncation}",
                    level="debug")

                done = terminations.get('Police0', False) or all(truncation.values())
                logger.log(f"Episode done: {done}", level="debug")

                # Update MrX agent
                mrX_next_graph = create_graph_data(next_state, 'MrX', env).to(device)
                mrX_agent.update(
                    mrX_graph,
                    mrX_action,
                    rewards.get('MrX', 0.0),
                    mrX_next_graph,
                    not terminations.get('Police0', False)
                )
                logger.log(f"MrX agent updated with reward: {rewards.get('MrX', 0.0)}", level="debug")

                # Update shared police agent
                for i in range(num_agents):
                    police_next_graph = create_graph_data(next_state, f'Police{i}', env).to(device)
                    police_agent.update(
                        police_graphs[i],
                        agent_actions.get(f'Police{i}'),
                        rewards.get(f'Police{i}', 0.0),
                        police_next_graph,
                        terminations.get(f'Police{i}', False)
                    )
                    logger.log(f"Police{i} agent updated with reward: {rewards.get(f'Police{i}', 0.0)}", level="debug")

                total_reward += rewards.get('MrX', 0.0)
                state = next_state
                logger.log(f"Total reward updated to: {total_reward}", level="debug")

            logger.log(f"Epoch {epoch + 1}, Episode {episode + 1}, Total Reward: {total_reward}", level="debug")
            # logger.log_scalar(f'Episode_total_reward{epoch}', total_reward, episode)

        # Evaluate performance and calculate the target difficulty
        logger.log(f"Evaluating agent balance after epoch {epoch + 1}.", level="debug")
        logger.log_model(mrX_agent, f'MrX_{node_feature_size}_agents')
        logger.log_model(police_agent, f'Police_{node_feature_size}_agents')
        logger.log_model(reward_weight_net, 'RewardWeightNet')

        wins = 0

        for episode in range(args.num_eval_episodes):
            logger.log(f"Epoch {epoch + 1}, Evaluation Episode {episode + 1} started.", level="info")
            state = env.reset(episode=episode)
            done = False
            total_reward = 0

            while not done:
                # Create graph data for GNN and move to GPU
                mrX_graph = create_graph_data(state, 'MrX', env).to(device)
                police_graphs = [
                    create_graph_data(state, f'Police{i}', env).to(device)
                    for i in range(num_agents)
                ]
                logger.log(f"Created graph data for MrX and Police agents.", level="debug")

                # MrX selects an action
                # mrX_action = mrX_agent.select_action(mrX_graph, torch.ones(mrX_action_size, device=device))

                mrX_action_size = env.action_space('MrX').n
                mrX_possible_moves = env.get_possible_moves(0)
                action_mask = torch.zeros(mrX_graph.num_nodes, dtype=torch.int32, device=device)
                action_mask[mrX_possible_moves] = 1
                mrX_action = mrX_agent.select_action(mrX_graph, action_mask)
                logger.log(f"MrX selected action: {mrX_action}", level="debug")

                # Police agents select actions
                agent_actions = {'MrX': mrX_action}
                for i in range(num_agents):
                    police_action_size = env.action_space(f'Police{i}').n
                    police_possible_moves = env.get_possible_moves(i + 1)
                    action_mask = torch.zeros(police_graphs[i].num_nodes, dtype=torch.int32, device=device)
                    action_mask[police_possible_moves] = 1
                    police_action = police_agent.select_action(
                        police_graphs[i],
                        action_mask
                    )
                    if police_action is None:
                        police_action = env.DEFAULT_ACTION
                    agent_actions[f'Police{i}'] = police_action
                    logger.log(f"Police{i} selected action: {police_action}", level="debug")

                for obj_id, act in agent_actions.items():
                    state[obj_id]["action"] = torch.tensor([act], dtype=torch.int64)

                # Execute actions for MrX and Police
                state_stepped = env.step(state)
                next_state = step_mdp(state_stepped)

                rewards = {agent_id: next_state[agent_id]['reward'].squeeze() for agent_id in env.possible_agents}
                terminations = {agent_id: next_state[agent_id]['terminated'].squeeze() for agent_id in
                                env.possible_agents}
                truncation = {agent_id: next_state[agent_id]['truncated'].squeeze() for agent_id in env.possible_agents}
                winner = env.current_winner
                logger.log(
                    f"Executed actions. Rewards: {rewards}, Terminations: {terminations}, Truncations: {truncation}",
                    level="debug")

                done = terminations.get('Police0', False) or all(truncation.values())
                logger.log(f"Episode done: {done}", level="debug")

                total_reward += rewards.get('MrX', 0.0)
                state = next_state
                logger.log(f"Total reward updated to: {total_reward}", level="debug")
                if done:
                    if winner == 'MrX':
                        wins += 1
                        logger.log(f"MrX won the evaluation episode.", level="info")
                    else:
                        logger.log(f"MrX lost the evaluation episode.",level="info")
            env.save_visualizations()
        win_ratio = wins / args.num_eval_episodes

        logger.log(f"Evaluation completed. Win Ratio: {win_ratio}")

        logger.log(f"Epoch {epoch + 1}, Episode {episode + 1}, Total Reward: {total_reward}", level="debug")

        # win_ratio = evaluate_agent_balance(mrX_agent, police_agent, env, args.num_eval_episodes, device)
        logger.log(f"Epoch {epoch + 1}: Win Ratio: {win_ratio}",level="info")
        node_curriculum,edge_curriculum,money_curriculum = modify_curriculum(win_ratio,node_curriculum,edge_curriculum,money_curriculum,0.1)
        logger.log(f"Modified node curriculum: {node_curriculum}",level="info")
        logger.log(f"Modified edge curriculum: {edge_curriculum}",level="info")
        target_difficulty = compute_target_difficulty(win_ratio)
        logger.log(f"Epoch {epoch + 1}: Computed target difficulty: {target_difficulty}", level="info")

        # Train the DifficultyNet based on the difference between predicted and target difficulty
        target_tensor = torch.FloatTensor([target_difficulty]).to(device).requires_grad_()  # Move target to GPU
        win_ratio_tensor = torch.FloatTensor([win_ratio]).to(device).requires_grad_()
        loss = criterion(win_ratio_tensor, target_tensor)
        logger.log(
            f"Epoch {epoch + 1}: Loss: {loss.item()}, Win Ratio: {win_ratio}, "
            f"Real Difficulty: {win_ratio}, Target Difficulty: {target_difficulty}"
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.log(f"Epoch {epoch + 1}: Optimizer step completed.", level="debug")

        logger.log_scalar('epoch/loss', loss.item())
        logger.log_scalar('epoch/win_ratio', win_ratio)

    logger.log("Training completed.")
    logger.close()


def train_mappo(args, agent_configs, logger_configs, visualization_configs):
    """
    Train a MAPPO (Multi-Agent Proximal Policy Optimization) agent.
    """
    # Initialize logger
    logger = Logger(
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_resume=args.wandb_resume,
        configs=logger_configs
    )
    logger.log("Logger initialized.", level="debug")
    # Set up meta-learning components
    reward_weight_net = RewardWeightNet().to(device)
    logger.log("DifficultyNet initialized and moved to device.")
    optimizer = optim.Adam(reward_weight_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    agent_money_values = np.asarray([v['agent_money'] for v in args.agent_configurations])
    average_agent_money = np.sum(agent_money_values)/agent_money_values.shape[0]
    logger.log(f"Average agent money: {average_agent_money}")
    node_curriculum,edge_curriculum,money_curriculum = create_curriculum(args.epochs,args.graph_nodes,args.graph_edges,average_agent_money,0.5)
    logger.log(f"Money curriculum: {money_curriculum}",level="info")
    logger.log("Loss function (MSELoss) initialized.", level="debug")
    # Training loop over epochs
    for epoch in range(args.epochs):
        logger.log_scalar('epoch_step', epoch)
        selected_config = random.choice(args.agent_configurations)
        num_agents = selected_config["num_police_agents"]
        agent_money = selected_config["agent_money"]
        logger.log(f"Number of police agents: {num_agents}, Agent money: {agent_money}, ")

        inputs = torch.FloatTensor([[num_agents, agent_money, args.graph_nodes, args.graph_edges]]).to(device)
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
            "Police_overlap_penalty": predicted_weight[0, 10]
        }
        logger.log(f"Epoch {epoch + 1}: Predicted weights: {reward_weights}", level="debug")
        logger.log_weights(reward_weights)

        # Create environment with predicted difficulty
        env_wrappable = CustomEnvironment(
            number_of_agents=num_agents,
            agent_money=int(money_curriculum[epoch]),
            reward_weights=reward_weights,
            logger=logger,
            epoch=epoch,
            graph_nodes=args.graph_nodes,
            graph_edges=args.graph_edges,
            vis_configs=visualization_configs
        )

        env = PettingZooWrapper(env=env_wrappable)
        logger.log(f"Environment created with weights {reward_weights}.", level="debug")

        # Reset environment
        mrx_key = 'MrX'
        initial_state = env.reset(episode=0)
        logger.log(f"Environment reset with weights {reward_weights}.", level="debug")
        mrx_features = initial_state[mrx_key]['observation']['node_features']

        # Extract dimensions
        feature_dim = mrx_features.shape[1]
        num_nodes = mrx_features.shape[0]

        # Global observations
        global_obs_dim = feature_dim * (num_agents + 1)

        # Set action space size to number of nodes
        max_action_dim = args.graph_nodes

        # Create MrX agent with a single policy
        mrX_agent = MappoAgent(
            n_agents=1,
            obs_size=feature_dim,
            global_obs_size=global_obs_dim,
            action_size=max_action_dim,
            hidden_size=agent_configs["hidden_size"],
            device=device,
            gamma=agent_configs["gamma"],
            lr=agent_configs["lr"],
            batch_size=agent_configs["batch_size"],
            buffer_size=agent_configs["buffer_size"],
            epsilon=agent_configs["epsilon"]
        )

        # Create Police agent with multiple policies
        police_agent = MappoAgent(
            n_agents=num_agents,
            obs_size=feature_dim,
            global_obs_size=global_obs_dim,
            action_size=max_action_dim,
            hidden_size=agent_configs["hidden_size"],
            device=device,
            gamma=agent_configs["gamma"],
            lr=agent_configs["lr"],
            batch_size=agent_configs["batch_size"],
            buffer_size=agent_configs["buffer_size"],
            epsilon=agent_configs["epsilon"]
        )
        # Training episodes
        for episode in range(args.num_episodes):
            logger.log(f"Epoch {epoch + 1}, Episode {episode + 1} started.", level="info")
            state = env.reset(episode=episode)
            done = False
            total_reward = 0

            # Episode loop
            while not done:
                obs_list = []
                actions = []
                log_probs = []

                # Get node features for MrX's current node
                mrx_node_features = state[mrx_key]['observation']['MrX_pos']
                #mrx_obs = torch.tensor(mrx_node_features, dtype=torch.float32, device=device)
                mrx_obs = mrx_node_features.detach().clone().to(dtype=torch.float32, device=device)
                # Get valid moves for MrX
                possible_moves = env.get_possible_moves(0)

                # Create action mask for MrX
                action_mask = torch.zeros(max_action_dim, dtype=torch.float32, device=device)
                for move in possible_moves:
                    if move < max_action_dim:
                        action_mask[move] = 1.0

                # Select action for MrX
                try:
                    mrx_action, mrx_log_prob, _ = mrX_agent.select_action(0, mrx_obs, action_mask)
                    obs_list.append(mrx_obs)
                    actions.append(mrx_action)
                    log_probs.append(mrx_log_prob)
                except Exception as e:
                    print(f"Error in MrX action selection: {e}")
                    print(f"MrX obs shape: {mrx_obs.shape}, expected obs_size: {feature_dim}")
                    print(f"Action mask shape: {action_mask.shape}")
                    raise

                # Process Police agents
                for police_idx in range(num_agents):
                    agent_key = f'Police{police_idx}'

                    # Get node features for police's current node
                    police_node_features = state[agent_key]['observation']['Polices_pos'].sum(dim=1)

                    #police_obs = torch.tensor(police_node_features, dtype=torch.float32, device=device)
                    police_obs = police_node_features.detach().clone().to(dtype=torch.float32, device=device)
                    # Get valid moves for this police agent
                    possible_moves = env.get_possible_moves(police_idx + 1)

                    # Create action mask for police
                    action_mask = torch.zeros(max_action_dim, dtype=torch.float32, device=device)
                    for move in possible_moves:
                        if move < max_action_dim:
                            action_mask[move] = 1.0

                    # Select action for police
                    try:
                        police_action, police_log_prob, _ = police_agent.select_action(police_idx, police_obs,
                                                                                       action_mask)
                        obs_list.append(police_obs)
                        actions.append(police_action)
                        log_probs.append(police_log_prob)
                    except Exception as e:
                        print(f"Error in Police {police_idx} action selection: {e}")
                        print(f"Police obs shape: {police_obs.shape}, expected obs_size: {feature_dim}")
                        print(f"Action mask shape: {action_mask.shape}")
                        raise

                # Prepare actions for environment step
                agent_actions = {
                    'MrX': actions[0],
                    **{f'Police{i}': actions[i + 1] for i in range(num_agents)}
                }
                for obj_id, act in agent_actions.items():
                    if act is not None:
                        state[obj_id]["action"] = torch.tensor([act], dtype=torch.int64)

                # Take environment step
                next_state = env.step(state)['next']

                rewards = {agent_id: next_state[agent_id]['reward'].squeeze() for agent_id in env.possible_agents}
                terminations = {agent_id: next_state[agent_id]['terminated'].squeeze() for agent_id in
                                env.possible_agents}
                truncation = {agent_id: next_state[agent_id]['truncated'].squeeze() for agent_id in env.possible_agents}
                logger.log(
                    f"Executed actions. Rewards: {rewards}, Terminations: {terminations}, Truncations: {truncation}",
                    level="debug")

                done = terminations.get('Police0', False) or all(truncation.values())
                logger.log(f"Episode done: {done}", level="debug")

                # Extract rewards and dones for all agents
                rewards = [rewards.get('MrX', 0.0)] + [rewards.get(f'Police{i}', 0.0) for i in
                                                       range(num_agents)]
                dones = [terminations.get('MrX', False) or truncation.get('MrX', False)] + \
                        [terminations.get(f'Police{i}', False) or truncation.get(f'Police{i}', False) for i in
                         range(num_agents)]

                # Create global observation
                processed_obs_list = []
                for obs in obs_list:
                    obs = obs.squeeze()
                    obs = obs.unsqueeze(0)
                    processed_obs_list.append(obs)

                global_obs = torch.cat(processed_obs_list)

                # Store experiences for MrX
                mrX_agent.store([obs_list[0]], global_obs, [actions[0]], [rewards[0]], [log_probs[0]], [dones[0]])

                # Store experiences for Police agents
                for i in range(num_agents):
                    police_agent.store([obs_list[i + 1]], global_obs, [actions[i + 1]],
                                       [rewards[i + 1]], [log_probs[i + 1]], [dones[i + 1]])

                # Update state and track reward
                total_reward += rewards[0]
                state = next_state

            # Update policies after episode
            mrX_agent.ppo_update()
            police_agent.ppo_update()

            logger.log_scalar(f'episode/mrx_reward', total_reward)

        logger.log(f"Evaluating agent balance after epoch {epoch + 1}.", level="debug")
        logger.log_model(mrX_agent, f"mappo_MrX_{num_agents}_agents")
        logger.log_model(police_agent, f"mappo_Police_{num_agents}_agents")
        logger.log_model(reward_weight_net, 'mappo_RewardWeightNet')
        # Evaluation loop
        wins = 0
        for eval_ep in range(args.num_eval_episodes):
            state = env.reset(episode=eval_ep)
            done = False

            while not done:
                actions = []

                mrx_node_features = state[mrx_key]['observation']['MrX_pos']
                #mrx_obs = torch.tensor(mrx_node_features, dtype=torch.float32, device=device)
                mrx_obs = mrx_node_features.detach().clone().to(dtype=torch.float32, device=device)
                possible_moves = env.get_possible_moves(0)
                action_mask = torch.zeros(max_action_dim, dtype=torch.float32, device=device)
                for move in possible_moves:
                    if move < max_action_dim:
                        action_mask[move] = 1.0

                mrx_action, _, _ = mrX_agent.select_action(0, mrx_obs, action_mask)
                actions.append(mrx_action)

                # Process Police agents in evaluation mode
                for police_idx in range(num_agents):
                    agent_key = f'Police{police_idx}'
                    police_node_features = state[agent_key]['observation']['Polices_pos'].sum(dim=1)
                    #police_obs = torch.tensor(police_node_features, dtype=torch.float32, device=device)
                    police_obs = police_node_features.detach().clone().to(dtype=torch.float32, device=device)
                    possible_moves = env.get_possible_moves(police_idx + 1)
                    action_mask = torch.zeros(max_action_dim, dtype=torch.float32, device=device)
                    for move in possible_moves:
                        if move < max_action_dim:
                            action_mask[move] = 1.0

                    police_action, _, _ = police_agent.select_action(police_idx, police_obs, action_mask)
                    actions.append(police_action)

                agent_actions = {
                    'MrX': actions[0],
                    **{f'Police{i}': actions[i + 1] for i in range(num_agents)}
                }

                for obj_id, act in agent_actions.items():
                    if act is not None:
                        state[obj_id]["action"] = torch.tensor([act], dtype=torch.int64)

                # Take environment step
                next_state = env.step(state)['next']

                rewards = {agent_id: next_state[agent_id]['reward'].squeeze() for agent_id in env.possible_agents}
                terminations = {agent_id: next_state[agent_id]['terminated'].squeeze() for agent_id in
                                env.possible_agents}
                truncation = {agent_id: next_state[agent_id]['truncated'].squeeze() for agent_id in env.possible_agents}
                winner = env.current_winner
                logger.log(
                    f"Executed actions. Rewards: {rewards}, Terminations: {terminations}, Truncations: {truncation}",
                    level="debug")

                done = terminations.get('Police0', False) or all(truncation.values())
                logger.log(f"Episode done: {done}", level="debug")
                state = next_state

                if done and winner == 'MrX':
                    wins += 1
            env.save_visualizations()
        # Calculate win ratio and update difficulty through meta-learning
        win_ratio = wins / args.num_eval_episodes
        node_curriculum,edge_curriculum,money_curriculum= modify_curriculum(win_ratio,node_curriculum,edge_curriculum,money_curriculum,0.1)
        target_difficulty = compute_target_difficulty(win_ratio)

        win_tensor = torch.tensor(win_ratio, dtype=torch.float32, device=device)
        target_tensor = torch.tensor(target_difficulty, dtype=torch.float32, device=device)
        target_tensor.requires_grad = True
        loss = criterion(win_tensor, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log(
            f"Epoch {epoch + 1}: Loss: {loss.item()}, Win Ratio: {win_ratio}, "
            f"Real Difficulty: {win_ratio}, Target Difficulty: {target_difficulty}"
        )

        logger.log_scalar('epoch/loss', loss.item())
        logger.log_scalar('epoch/win_ratio', win_ratio)

    logger.log("Training completed.")
    logger.close()


def create_graph_data(state, agent_id, env):
    """
    Create a PyTorch Geometric Data object from the environment state for the specified agent.
    """
    logger = env.logger  # Access the logger from the environment
    logger.log(f"Creating graph data for agent {agent_id}.", level="debug")

    edge_index = torch.tensor(env.board.edge_links.T, dtype=torch.long)
    edge_features = torch.tensor(env.board.edges, dtype=torch.float)

    num_nodes = env.board.nodes.shape[0]
    num_features = env.number_of_agents + 1  # Adjust based on your feature design

    # Ensure num_features is an integer greater than 0
    node_features = np.zeros((num_nodes, num_features), dtype=np.float32)

    # Highlight MrX position
    mrX_pos = state.get('MrX', {}).get('observation', None).get('MrX_pos', None)
    if mrX_pos is not None:
        node_features[mrX_pos, 0] = 1  # MrX is at index 0
        logger.log(f"Agent {agent_id}: MrX position encoded at node {mrX_pos}.", level="debug")

    # Highlight Police positions
    for i in range(env.number_of_agents - 1):
        police_pos = state.get(f'Police{i}', {}).get('observation', None).get('Polices_pos', None)
        if police_pos is not None and len(police_pos) > 0:
            node_features[police_pos[0], i + 1] = 1  # Police indices start from 1
            logger.log(f"Agent {agent_id}: Police{i} position encoded at node {police_pos[0]}.", level="debug")

    node_features = torch.tensor(node_features, dtype=torch.float32)

    # Move tensors to the appropriate device
    edge_index = edge_index.to(device)
    edge_features = edge_features.to(device)
    node_features = node_features.to(device)

    # Create PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    logger.log(f"Graph data for agent {agent_id} created.", level="debug")
    return data


def evaluate(args, agent_configs, logger_configs, visualization_configs):
    """
    Evaluation function:
    - Loads pre-trained RewardWeightNet and agent models.
    - Applies the predicted reward weights in the environment.
    - Runs a fixed number of episodes:
        - MrX and Police agents act based on GNN policies (no training).
        - Logs performance metrics (e.g., win ratio).
    """
    logger = Logger(
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_resume=args.wandb_resume,
        configs=logger_configs
    )
    reward_weight_net = RewardWeightNet().to(device)
    reward_weight_net.load_state_dict(logger.load_model('RewardWeightNet'), strict=False)
    reward_weight_net.eval()
    if agent_configs["agent_type"] == "gnn":
        police_agent = GNNAgent(node_feature_size=3, device=device, gamma=agent_configs["gamma"],
                                lr=agent_configs["lr"], batch_size=agent_configs["batch_size"],
                                buffer_size=agent_configs["buffer_size"], epsilon=agent_configs["epsilon"],
                                epsilon_decay=agent_configs["epsilon_decay"], epsilon_min=agent_configs["epsilon_min"])
    else:
        police_agent = RandomAgent()
    for config in args.agent_configurations:
        num_agents, agent_money = config["num_police_agents"] + 1, config["agent_money"]  # Unpack the tuple
        logger.log(f"Choosen configuration: {num_agents} agents, {agent_money} money.", level="info")
        # print(selected_config)
        logger.log_scalar('epoch/num_agents', num_agents)
        logger.log_scalar('epoch/agent_money', agent_money)
        # Predict the difficulty from the number of agents and money
        inputs = torch.FloatTensor([[num_agents, agent_money, args.graph_nodes, args.graph_edges]]).to(
            device)  # Move inputs to GPU
        predicted_weight = reward_weight_net(inputs)
        # print(predicted_weight)
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
            "Police_overlap_penalty": predicted_weight[0, 10]
        }
        logger.log(f"Predicted weights: {reward_weights}", level="debug")
        logger.log_weights(reward_weights)
        # Create environment with predicted difficulty
        env_wrappable = CustomEnvironment(
            number_of_agents=num_agents,
            agent_money=agent_money,
            reward_weights=reward_weights,
            logger=logger,
            epoch=1,
            graph_nodes=args.graph_nodes,
            graph_edges=args.graph_edges,
            vis_configs=visualization_configs
        )
        env = PettingZooWrapper(env=env_wrappable)

        node_feature_size = env.number_of_agents + 1  # Assuming node features exist
        mrX_action_size = env.action_space('MrX').n
        police_action_size = env.action_space('Police0').n  # Assuming all police have the same action space
        logger.log(
            f"Node feature size: {node_feature_size}, MrX action size: {mrX_action_size}, Police action size: {police_action_size}",
            level="debug")

        # Initialize GNN agents with graph-specific parameters and move them to GPU

        MrX_model_name = f'MrX_{node_feature_size}_agents'
        Police_model_name = f'Police_{node_feature_size}_agents'
        for name in [MrX_model_name, Police_model_name]:
            if not logger.model_exists(name):
                logger.log(f"WARNING: the weights for the {name} do not exist!", level="info")
        if agent_configs["agent_type"] == "gnn":
            mrX_agent = GNNAgent(node_feature_size=node_feature_size, device=device, gamma=agent_configs["gamma"],
                                 lr=agent_configs["lr"], batch_size=agent_configs["batch_size"],
                                 buffer_size=agent_configs["buffer_size"], epsilon=agent_configs["epsilon"],
                                 epsilon_decay=agent_configs["epsilon_decay"], epsilon_min=agent_configs["epsilon_min"])
            mrX_agent.load_state_dict(logger.load_model(MrX_model_name), strict=False)
        else:
            mrX_agent = RandomAgent()
        if agent_configs["agent_type"] == "gnn":
            police_agent = GNNAgent(node_feature_size=node_feature_size, device=device, gamma=agent_configs["gamma"],
                                    lr=agent_configs["lr"], batch_size=agent_configs["batch_size"],
                                    buffer_size=agent_configs["buffer_size"], epsilon=agent_configs["epsilon"],
                                    epsilon_decay=agent_configs["epsilon_decay"],
                                    epsilon_min=agent_configs["epsilon_min"])
            police_agent.load_state_dict(logger.load_model(Police_model_name), strict=False)
        else:
            police_agent = RandomAgent()

        wins = 0
        for episode in range(args.num_eval_episodes):
            logger.log(f"Evaluation Episode {episode + 1} started.", level="info")
            state = env.reset(episode=episode)
            done = False
            total_reward = 0
            while not done:
                # Create graph data for GNN and move to GPU
                mrX_graph = create_graph_data(state, 'MrX', env).to(device)
                police_graphs = [
                    create_graph_data(state, f'Police{i}', env).to(device)
                    for i in range(num_agents)
                ]
                logger.log(f"Created graph data for MrX and Police agents.", level="debug")

                # MrX selects an action
                # mrX_action = mrX_agent.select_action(mrX_graph, torch.ones(mrX_action_size, device=device))

                mrX_action_size = env.action_space('MrX').n
                mrX_possible_moves = env.get_possible_moves(0)
                action_mask = torch.zeros(mrX_graph.num_nodes, dtype=torch.int32, device=device)
                action_mask[mrX_possible_moves] = 1
                mrX_action = mrX_agent.select_action(mrX_graph, action_mask)
                logger.log(f"MrX selected action: {mrX_action}", level="debug")

                # Police agents select actions
                agent_actions = {'MrX': mrX_action}
                for i in range(num_agents):
                    police_action_size = env.action_space(f'Police{i}').n
                    police_possible_moves = env.get_possible_moves(i + 1)
                    action_mask = torch.zeros(police_graphs[i].num_nodes, dtype=torch.int32, device=device)
                    action_mask[police_possible_moves] = 1
                    police_action = police_agent.select_action(
                        police_graphs[i],
                        action_mask
                    )

                    if police_action is None:
                        police_action = env.DEFAULT_ACTION
                    agent_actions[f'Police{i}'] = police_action
                    logger.log(f"Police{i} selected action: {police_action}", level="debug")

                for obj_id, act in agent_actions.items():
                    if act is not None:
                        state[obj_id]["action"] = torch.tensor([act], dtype=torch.int64)

                # Execute actions for MrX and Police
                state_stepped = env.step(state)
                next_state = step_mdp(state_stepped)
                rewards = {agent_id: next_state[agent_id]['reward'].squeeze() for agent_id in env.possible_agents}
                terminations = {agent_id: next_state[agent_id]['terminated'].squeeze() for agent_id in
                                env.possible_agents}
                truncation = {agent_id: next_state[agent_id]['truncated'].squeeze() for agent_id in env.possible_agents}
                winner = env.current_winner
                logger.log(
                    f"Executed actions. Rewards: {rewards}, Terminations: {terminations}, Truncations: {truncation}",
                    level="debug")

                done = terminations.get('Police0', False) or all(truncation.values())
                logger.log(f"Episode done: {done}", level="debug")

                total_reward += rewards.get('MrX', 0.0)
                state = next_state
                logger.log(f"Total reward updated to: {total_reward}", level="debug")
                if done:
                    if winner == 'MrX':
                        wins += 1
                        logger.log(f"MrX won the evaluation episode.", level="info")
                    else:
                        logger.log(f"MrX lost the evaluation episode.", level="info")
            env.save_visualizations()
        win_ratio = wins / args.num_eval_episodes
        logger.log(f"Evaluation completed. Win Ratio: {win_ratio}")
        return


def evaluate_mappo(args, agent_configs, logger_configs, visualization_configs):
    """
    MAPPO-specific evaluation function.
    Evaluates trained MAPPO agents using RewardWeightNet predicted difficulty.
    """
    logger = Logger(
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_resume=args.wandb_resume,
        configs=logger_configs
    )

    reward_weight_net = RewardWeightNet().to(device)
    reward_weight_net.load_state_dict(logger.load_model('mappo_RewardWeightNet'), strict=False)
    reward_weight_net.eval()

    for config in args.agent_configurations:
        num_agents = config["num_police_agents"]
        agent_money = config["agent_money"]
        logger.log(f"Evaluating MAPPO with {num_agents} agents and {agent_money} money.", level="info")

        # Predict weights
        inputs = torch.FloatTensor([[num_agents, agent_money, args.graph_nodes, args.graph_edges]]).to(device)
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
            "Police_overlap_penalty": predicted_weight[0, 10]
        }
        node_curriculum, edge_curriculum = create_curriculum(args.epochs, args.graph_nodes, args.graph_edges, 0.5)
        # Create environment
        env_wrappable = CustomEnvironment(
            number_of_agents=num_agents,
            agent_money=agent_money,
            reward_weights=reward_weights,
            logger=logger,
            epoch=1,
            graph_nodes=args.graph_nodes,
            graph_edges=args.graph_edges,
            vis_configs=visualization_configs
        )
        env = PettingZooWrapper(env=env_wrappable)

        # Get obs/action dimensions
        state = env.reset(episode=0)
        mrx_obs = torch.tensor(state['MrX']['observation']['MrX_pos'], dtype=torch.float32, device=device)
        police_obs = torch.tensor(state['Police0']['observation']['Polices_pos'], dtype=torch.float32, device=device)
        mrx_feat_dim = mrx_obs.shape[-1]
        police_feat_dim = police_obs.shape[-1]
        global_obs_dim = mrx_feat_dim * (num_agents + 1)
        action_dim = args.graph_nodes

        # Load agents
        mrX_agent = MappoAgent(1, mrx_feat_dim, global_obs_dim, action_dim, agent_configs["hidden_size"],
                               device, agent_configs["gamma"], agent_configs["lr"],
                               agent_configs["batch_size"], agent_configs["buffer_size"],
                               agent_configs["epsilon"])
        police_agent = MappoAgent(num_agents, police_feat_dim, global_obs_dim, action_dim, agent_configs["hidden_size"],
                                  device, agent_configs["gamma"], agent_configs["lr"],
                                  agent_configs["batch_size"], agent_configs["buffer_size"],
                                  agent_configs["epsilon"])

        MrX_model_name = f'mappo_MrX_{num_agents}_agents'
        Police_model_name = f'mappo_Police_{num_agents}_agents'
        for name in [MrX_model_name, Police_model_name]:
            if not logger.model_exists(name):
                logger.log(f"WARNING: the weights for the {name} do not exist! ABORTING EVALUATION.", level="info")
                quit()

        mrX_agent.load_state_dict(logger.load_model(MrX_model_name), strict=False)
        police_agent.load_state_dict(logger.load_model(Police_model_name), strict=False)

        # Run evaluation episodes
        wins = 0
        for episode in range(args.num_eval_episodes):
            state = env.reset(episode=episode)
            done = False

            while not done:
                actions = []

                mrx_obs = torch.tensor(state['MrX']['observation']['MrX_pos'], dtype=torch.float32, device=device)
                mrx_moves = env.get_possible_moves(0)
                mrx_mask = torch.zeros(action_dim, dtype=torch.float32, device=device)
                mrx_mask[mrx_moves] = 1.0
                mrx_action, _, _ = mrX_agent.select_action(0, mrx_obs, mrx_mask)
                actions.append(mrx_action)

                for i in range(num_agents):
                    police_obs = torch.tensor(state[f'Police{i}']['observation']['Polices_pos'],dtype=torch.float32, device=device).sum(dim=1)
                    police_moves = env.get_possible_moves(i + 1)
                    police_mask = torch.zeros(action_dim, dtype=torch.float32, device=device)
                    police_mask[police_moves] = 1.0
                    police_action, _, _ = police_agent.select_action(i, police_obs, police_mask)
                    actions.append(police_action)

                agent_actions = {'MrX': actions[0], **{f'Police{i}': actions[i + 1] for i in range(num_agents)}}
                for agent_id, action in agent_actions.items():
                    if action is not None:
                        state[agent_id]["action"] = torch.tensor([action], dtype=torch.int64)

                next_state = env.step(state)['next']
                state = next_state
                terminations = {aid: next_state[aid]['terminated'].squeeze() for aid in env.possible_agents}
                truncations = {aid: next_state[aid]['truncated'].squeeze() for aid in env.possible_agents}
                done = terminations.get('Police0', False) or all(truncations.values())

                if done and env.current_winner == 'MrX':
                    wins += 1

        win_ratio = wins / args.num_eval_episodes
        logger.log(f"MAPPO Evaluation completed. Win ratio: {win_ratio}", level="info")
        return


def compute_target_difficulty(win_ratio, target_balance=0.5):
    """Adjust the target difficulty based on the win/loss ratio."""
    # You can add logging here if needed
    return target_balance


if __name__ == "__main__":
    import argparse
    import yaml
    import sys

    parser = argparse.ArgumentParser(description="Train MrX and Police agents with dynamic difficulty prediction.")
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file.')

    # Add all the other arguments with default=argparse.SUPPRESS
    parser.add_argument('--graph_nodes', type=int, default=argparse.SUPPRESS, help='Number of nodes in the graph')
    parser.add_argument('--graph_edges', type=int, default=argparse.SUPPRESS, help='Number of edges in the graph')
    parser.add_argument('--state_size', type=int, default=argparse.SUPPRESS, help='State size for the agent')
    parser.add_argument('--action_size', type=int, default=argparse.SUPPRESS, help='Action size for the agent')
    parser.add_argument('--num_episodes', type=int, default=argparse.SUPPRESS, help='Number of episodes per epoch')
    parser.add_argument('--num_eval_episodes', type=int, default=argparse.SUPPRESS,
                        help='Number of evaluation episodes')
    parser.add_argument('--epochs', type=int, default=argparse.SUPPRESS, help='Number of training epochs')
    parser.add_argument('--exp_dir', type=str, default=argparse.SUPPRESS, help='Directory where logs will be saved')
    parser.add_argument('--wandb_api_key', type=str, default=argparse.SUPPRESS, help='Weights & Biases API key')
    parser.add_argument('--wandb_project', type=str, default=argparse.SUPPRESS, help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=argparse.SUPPRESS,
                        help='Weights & Biases entity (user or team)')
    parser.add_argument('--wandb_run_name', type=str, default=argparse.SUPPRESS,
                        help='Custom name for the Weights & Biases run')
    parser.add_argument('--wandb_resume', action='store_true', help='Resume Weights & Biases run if it exists')
    parser.add_argument('--random_seed', type=int, default=argparse.SUPPRESS, help='Random seed for reproducibility')
    parser.add_argument('--evaluate', type=bool, default=argparse.SUPPRESS, help='Set to True to evaluate the agents')
    # Add agent_configurations argument
    parser.add_argument('--agent_configurations', type=str, default=argparse.SUPPRESS,
                        help='List of (num_police_agents, agent_money) tuples separated by semicolons. E.g., "2,30;3,40;4,50"')
    parser.add_argument('--log_configs', type=str, default="default", help='Select a logger configuration!')
    parser.add_argument('--agent_configs', type=str, default="default", help='Select an agent configuration!')
    parser.add_argument('--vis_configs', type=str, default="default", help='Select an agent configuration!')
    # Parse command-line arguments
    args = parser.parse_args()
    args_dict = vars(args)
    if args_dict["wandb_api_key"] == "null":
        args_dict["wandb_api_key"] = ""
    if args_dict["wandb_project"] == "null":
        args_dict["wandb_project"] = ""
    if args_dict["wandb_entity"] == "null":
        args_dict["wandb_entity"] = ""
    # Default values for all parameters
    default_values = {
        'graph_nodes': 50,
        'graph_edges': 110,
        'state_size': 1,
        'action_size': 5,
        'num_episodes': 100,
        'num_eval_episodes': 10,
        'epochs': 50,
        'exp_dir': '/',
        'wandb_api_key': None,
        'wandb_project': None,
        'wandb_entity': None,
        'wandb_run_name': None,
        'wandb_resume': False,
        'agent_configurations': [(2, 30), (3, 40), (4, 50)],  # Default configurations
        'random_seed': 42,
        'evaluate': False,
        'log_configs': 'default',
        'agent_configs': 'default'
    }

    # If a config file is provided, load its parameters
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_args = yaml.safe_load(f)
            if config_args is None:
                config_args = {}
        except FileNotFoundError:
            print(f"Configuration file {args.config} not found.", file=sys.stderr)
            sys.exit(1)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}", file=sys.stderr)
            sys.exit(1)
        # Remove 'config' key if present to prevent overwriting
        config_args.pop('config', None)
    else:
        config_args = {}
    yaml_loader = yaml.SafeLoader
    yaml_loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open("./src/configs/agent/" + args_dict["agent_configs"] + ".yaml", 'r') as f:
        agent_configs = yaml.load(f, Loader=yaml_loader)
    with open("./src/configs/logger/" + args_dict["log_configs"] + ".yaml", 'r') as f:
        logger_configs = yaml.load(f, Loader=yaml_loader)
    with open("./src/configs/visualization/" + args_dict["vis_configs"] + ".yaml", 'r') as f:
        visualization_configs = yaml.load(f, Loader=yaml_loader)
    logger_configs["log_dir"] = os.path.join(args_dict["exp_dir"], logger_configs["log_dir"])
    os.makedirs(logger_configs["log_dir"], exist_ok=True)
    visualization_configs["save_dir"] = os.path.join(args_dict["exp_dir"], visualization_configs["save_dir"])
    os.makedirs(visualization_configs["save_dir"], exist_ok=True)
    # Handle agent_configurations from command-line if provided
    if 'agent_configurations' in args_dict:
        # Parse the string into a list of tuples or dictionaries
        # Depending on your config file format, you might need to adjust this
        try:
            config_str = args_dict['agent_configurations']
            agent_configurations = []
            for item in config_str.split(';'):
                if ',' in item:
                    num_police_agents, agent_money = item.split(',')
                    agent_configurations.append((int(num_police_agents.strip()), float(agent_money.strip())))
                elif ':' in item:
                    # Handle dictionary-like input e.g., "num_police_agents:2,agent_money:30"
                    parts = item.split(',')
                    config = {}
                    for part in parts:
                        key, value = part.split(':')
                        config[key.strip()] = float(value.strip()) if 'money' in key else int(value.strip())
                    agent_configurations.append((config['num_police_agents'], config['agent_money']))
                else:
                    raise ValueError(f"Invalid format for agent_configurations item: '{item}'")
            config_args['agent_configurations'] = agent_configurations
        except Exception as e:
            raise ValueError(f"Error parsing agent_configurations: {e}")

    # Combine default values, config file, and command-line arguments
    # Priority: command-line args > config file > default values
    combined_args = {**default_values, **config_args, **args_dict}

    # Convert combined_args to Namespace
    args = argparse.Namespace(**combined_args)

    if args.evaluate:
        if agent_configs["agent_type"] == "mappo":
            evaluate_mappo(args, agent_configs, logger_configs, visualization_configs)
        else:
            evaluate(args, agent_configs, logger_configs, visualization_configs)
    else:
        if agent_configs["agent_type"] == "mappo":
            train_mappo(args, agent_configs, logger_configs, visualization_configs)
        else:
            train(args, agent_configs, logger_configs, visualization_configs)