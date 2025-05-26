# %%
%load_ext autoreload
%autoreload 2

# %%
from argparse import Namespace

# %%
ns = Namespace(graph_nodes=10, graph_edges=10, state_size=1, action_size=5, num_episodes=5, num_eval_episodes=3, epochs=2, log_dir='./src/experiments/smoke_train/logs', wandb_api_key=None, wandb_project=None, wandb_entity=None, wandb_run_name='smoke_train', wandb_resume=False, agent_configurations=[{'num_police_agents': 2, 'agent_money': 16}, {'num_police_agents': 2, 'agent_money': 18}, {'num_police_agents': 2, 'agent_money': 20}, {'num_police_agents': 2, 'agent_money': 22}, {'num_police_agents': 2, 'agent_money': 24}, {'num_police_agents': 2, 'agent_money': 20}, {'num_police_agents': 3, 'agent_money': 10}, {'num_police_agents': 3, 'agent_money': 12}, {'num_police_agents': 3, 'agent_money': 14}, {'num_police_agents': 3, 'agent_money': 16}, {'num_police_agents': 3, 'agent_money': 18}, {'num_police_agents': 4, 'agent_money': 10}, {'num_police_agents': 4, 'agent_money': 12}, {'num_police_agents': 4, 'agent_money': 14}, {'num_police_agents': 4, 'agent_money': 16}, {'num_police_agents': 5, 'agent_money': 8}, {'num_police_agents': 5, 'agent_money': 10}, {'num_police_agents': 5, 'agent_money': 12}, {'num_police_agents': 6, 'agent_money': 10}, {'num_police_agents': 6, 'agent_money': 6}, {'num_police_agents': 6, 'agent_money': 8}], random_seed=42, evaluate=False, config='./src/experiments/smoke_train/config.yml')


from main import *

# %%
args = ns

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from logger import Logger  # Your custom Logger class
from RLAgent.gnn_agent import GNNAgent
from Enviroment.yard import CustomEnvironment
from torch_geometric.data import Data
import random
from torchrl.envs.libs.pettingzoo import PettingZooWrapper, PettingZooEnv
# Define the device at the beginning
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")  # You may consider logging this instead

# %%
# from torchrl.envs.libs.pettingzoo import PettingZooWrapper
# from pettingzoo.butterfly import pistonball_v6
# kwargs = {"n_pistons": 21, "continuous": True}
# env = PettingZooWrapper(env=pistonball_v6.parallel_env(**kwargs),return_state=True,
#                         group_map=None, # Use default for parallel (all pistons grouped together)
#                         )
# # print(env.group_map)
# env.reset()
# # # env.rollout(10)

# # # from pettingzoo.classic import tictactoe_v3
# # # from torchrl.envs.libs.pettingzoo import PettingZooWrapper
# # # from torchrl.envs.utils import MarlGroupMapType
# # # env = PettingZooWrapper(
# # #  env=tictactoe_v3.env(),
# # #  use_mask=True, # Must use it since one player plays at a time
# # #  group_map=None # # Use default for AEC (one group per player)
# # #  )
# # # print(env.group_map)
# # # env.rollout(10)
# print()

# %%
logger = Logger(
        log_dir=args.log_dir,
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_resume=args.wandb_resume
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


# %%
for epoch in range(args.epochs):
    logger.log_scalar('epoch_step', epoch)

    logger.log(f"Starting epoch {epoch + 1}/{args.epochs}.", level="info")
    
    # Randomly select a (num_agents, agent_money) tuple from the predefined list
    # print(args.agent_configurations)
    selected_config = random.choice(args.agent_configurations)  # Ensure args.agent_configurations is defined
    num_agents, agent_money = selected_config["num_police_agents"], selected_config["agent_money"]  # Unpack the tuple
    logger.log(f"Choosen configuration: {num_agents} agents, {agent_money} money.", level="info")
    # print(selected_config)
    logger.log_scalar('epoch/num_agents', num_agents)
    logger.log_scalar('epoch/agent_money', agent_money)
    # Predict the difficulty from the number of agents and money
    inputs = torch.FloatTensor([[num_agents, agent_money, args.graph_nodes, args.graph_edges]]).to(device)  # Move inputs to GPU
    predicted_weight = reward_weight_net(inputs)
    reward_weights = {
        "Police_distance" : predicted_weight[0,0],
        "Police_group": predicted_weight[0,1],
        "Police_position": predicted_weight[0,2],
        "Police_time": predicted_weight[0,3],
        "Mrx_closest": predicted_weight[0,4],
        "Mrx_average": predicted_weight[0,5],
        "Mrx_position": predicted_weight[0,6],
        "Mrx_time": predicted_weight[0,7]
    }

    logger.log(f"Epoch {epoch + 1}: Predicted weights: {reward_weights}", level="debug")
    logger.log_weights(reward_weights)
    # Create environment with predicted difficulty
    env_base = CustomEnvironment(
        number_of_agents=num_agents,
        agent_money=agent_money,
        reward_weights=reward_weights,
        logger=logger,
        epoch=epoch,
        graph_nodes=args.graph_nodes,
        graph_edges=args.graph_edges
    )
    # print(f"num agents: {num_agents},\n agent money: {agent_money},\n reward weights: {reward_weights}")
    # print(f"graph nodes: {args.graph_nodes},\n graph edges: {args.graph_edges}")
    env = PettingZooWrapper(env=env_base)
    # env = PettingZooEnv(env=env_base, parallel=True)
    logger.log(f"Environment created with weights {reward_weights}.",level="debug")

    # Determine node feature size from the environment
    node_feature_size = env.number_of_agents + 1  # Assuming node features exist
    mrX_action_size = env.action_space('MrX').n
    police_action_size = env.action_space('Police0').n  # Assuming all police have the same action space
    logger.log(f"Node feature size: {node_feature_size}, MrX action size: {mrX_action_size}, Police action size: {police_action_size}",level="debug")

    # Initialize GNN agents with graph-specific parameters and move them to GPU
    mrX_agent = GNNAgent(node_feature_size=node_feature_size, device=device)
    police_agent = GNNAgent(node_feature_size=node_feature_size, device=device)
    logger.log("GNN agents for MrX and Police initialized.",level="debug")

    # Train the MrX and Police agents in the environment
    for episode in range(args.num_episodes):
        logger.log(f"Epoch {epoch + 1}, Episode {episode + 1} started.",level="info")
        print(episode)
        print(type(env))
        state, _ = env.reset(episode=episode)

# %%
# from pettingzoo import ParallelEnv

# par_env = ParallelEnv()
# par_env.possible_agents

# %%
# logger = Logger(
#         log_dir=args.log_dir,
#         wandb_api_key=args.wandb_api_key,
#         wandb_project=args.wandb_project,
#         wandb_entity=args.wandb_entity,
#         wandb_run_name=args.wandb_run_name,
#         wandb_resume=args.wandb_resume
#     )

# logger.log("Logger initialized.", level="debug")

# # Initialize DifficultyNet and move it to the GPU
# reward_weight_net = RewardWeightNet().to(device)
# logger.log("DifficultyNet initialized and moved to device.")

# optimizer = optim.Adam(reward_weight_net.parameters(), lr=0.001)

# criterion = nn.MSELoss()
# logger.log("Loss function (MSELoss) initialized.", level="debug")

# logger.log(f"Starting training with variable agents and money settings.", level="debug")

# # Validate that the agent configurations list is provided and not empty
# if not hasattr(args, 'agent_configurations') or not args.agent_configurations:
#     raise ValueError("args.agent_configurations must be a non-empty list of (num_agents, agent_money) tuples.")

# for epoch in range(args.epochs):
#     logger.log_scalar('epoch_step', epoch)

#     logger.log(f"Starting epoch {epoch + 1}/{args.epochs}.", level="info")
    
#     # Randomly select a (num_agents, agent_money) tuple from the predefined list
#     # print(args.agent_configurations)
#     selected_config = random.choice(args.agent_configurations)  # Ensure args.agent_configurations is defined
#     num_agents, agent_money = selected_config["num_police_agents"], selected_config["agent_money"]  # Unpack the tuple
#     logger.log(f"Choosen configuration: {num_agents} agents, {agent_money} money.", level="info")
#     # print(selected_config)
#     logger.log_scalar('epoch/num_agents', num_agents)
#     logger.log_scalar('epoch/agent_money', agent_money)
#     # Predict the difficulty from the number of agents and money
#     inputs = torch.FloatTensor([[num_agents, agent_money, args.graph_nodes, args.graph_edges]]).to(device)  # Move inputs to GPU
#     predicted_weight = reward_weight_net(inputs)
#     reward_weights = {
#         "Police_distance" : predicted_weight[0,0],
#         "Police_group": predicted_weight[0,1],
#         "Police_position": predicted_weight[0,2],
#         "Police_time": predicted_weight[0,3],
#         "Mrx_closest": predicted_weight[0,4],
#         "Mrx_average": predicted_weight[0,5],
#         "Mrx_position": predicted_weight[0,6],
#         "Mrx_time": predicted_weight[0,7]
#     }

#     logger.log(f"Epoch {epoch + 1}: Predicted weights: {reward_weights}", level="debug")
#     logger.log_weights(reward_weights)
#     # Create environment with predicted difficulty
#     env_base = CustomEnvironment(
#         number_of_agents=num_agents,
#         agent_money=agent_money,
#         reward_weights=reward_weights,
#         logger=logger,
#         epoch=epoch,
#         graph_nodes=args.graph_nodes,
#         graph_edges=args.graph_edges
#     )
#     env = PettingZooWrapper(env=env_base)
#     logger.log(f"Environment created with weights {reward_weights}.",level="debug")

#     # Determine node feature size from the environment
#     node_feature_size = env.number_of_agents + 1  # Assuming node features exist
#     mrX_action_size = env.action_space('MrX').n
#     police_action_size = env.action_space('Police0').n  # Assuming all police have the same action space
#     logger.log(f"Node feature size: {node_feature_size}, MrX action size: {mrX_action_size}, Police action size: {police_action_size}",level="debug")

#     # Initialize GNN agents with graph-specific parameters and move them to GPU
#     mrX_agent = GNNAgent(node_feature_size=node_feature_size, device=device)
#     police_agent = GNNAgent(node_feature_size=node_feature_size, device=device)
#     logger.log("GNN agents for MrX and Police initialized.",level="debug")

#     # Train the MrX and Police agents in the environment
#     for episode in range(args.num_episodes):
#         logger.log(f"Epoch {epoch + 1}, Episode {episode + 1} started.",level="info")
#         state, _ = env.reset(episode=episode)
#         done = False
#         total_reward = 0

#         while not done:
#             # Create graph data for GNN and move to GPU
#             mrX_graph = create_graph_data(state, 'MrX', env).to(device)
#             police_graphs = [
#                 create_graph_data(state, f'Police{i}', env).to(device)
#                 for i in range(num_agents)
#             ]
#             logger.log(f"Created graph data for MrX and Police agents.",level="debug")

#             # MrX selects an action
#             # mrX_action = mrX_agent.select_action(mrX_graph, torch.ones(mrX_action_size, device=device))

#             mrX_action_size = env.action_space('MrX').n
#             mrX_possible_moves = env.get_possible_moves(0)
#             action_mask = torch.zeros(mrX_graph.num_nodes, dtype=torch.int32, device=device)
#             action_mask[ mrX_possible_moves] = 1
#             mrX_action = mrX_agent.select_action(mrX_graph,action_mask)
#             logger.log(f"MrX selected action: {mrX_action}",level="debug")

#             # Police agents select actions
#             agent_actions = {'MrX': mrX_action}
#             for i in range(num_agents):
#                 police_action_size = env.action_space(f'Police{i}').n
#                 police_possible_moves = env.get_possible_moves(i+1)
#                 action_mask = torch.zeros(police_graphs[i].num_nodes, dtype=torch.int32, device=device)
#                 action_mask[ police_possible_moves] = 1
#                 police_action = police_agent.select_action(
#                     police_graphs[i],
#                     action_mask
#                 )
#                 agent_actions[f'Police{i}'] = police_action
#                 logger.log(f"Police{i} selected action: {police_action}",level="debug")

#             # Execute actions for MrX and Police
#             next_state, rewards, terminations, truncation, _, _ = env.step(agent_actions)
#             logger.log(f"Executed actions. Rewards: {rewards}, Terminations: {terminations}, Truncations: {truncation}",level="debug")

#             done = terminations.get('Police0', False) or all(truncation.values())
#             logger.log(f"Episode done: {done}",level="debug")

#             # Update MrX agent
#             mrX_next_graph = create_graph_data(next_state, 'MrX', env).to(device)
#             mrX_agent.update(
#                 mrX_graph,
#                 mrX_action,
#                 rewards.get('MrX', 0.0),
#                 mrX_next_graph,
#                 not terminations.get('Police0', False)
#             )
#             logger.log(f"MrX agent updated with reward: {rewards.get('MrX', 0.0)}",level="debug")

#             # Update shared police agent
#             for i in range(num_agents):
#                 police_next_graph = create_graph_data(next_state, f'Police{i}', env).to(device)
#                 police_agent.update(
#                     police_graphs[i],
#                     agent_actions.get(f'Police{i}'),
#                     rewards.get(f'Police{i}', 0.0),
#                     police_next_graph,
#                     terminations.get(f'Police{i}', False)
#                 )
#                 logger.log(f"Police{i} agent updated with reward: {rewards.get(f'Police{i}', 0.0)}",level="debug")

#             total_reward += rewards.get('MrX', 0.0)
#             state = next_state
#             logger.log(f"Total reward updated to: {total_reward}",level="debug")

#         logger.log(f"Epoch {epoch + 1}, Episode {episode + 1}, Total Reward: {total_reward}",level="debug")
#         # logger.log_scalar(f'Episode_total_reward{epoch}', total_reward, episode)

#     # Evaluate performance and calculate the target difficulty
#     logger.log(f"Evaluating agent balance after epoch {epoch + 1}.",level="debug")
#     # logger.log_model(mrX_agent, 'MrX')
#     # logger.log_model(police_agent, 'Police')
#     # logger.log_model(reward_weight_net, 'RewardWeightNet')

#     wins = 0

#     for episode in range(args.num_eval_episodes):
#         logger.log(f"Epoch {epoch + 1}, Evaluation Episode {episode + 1} started.",level="info")
#         state, _ = env.reset(episode=episode)
#         done = False
#         total_reward = 0

#         while not done:
#             # Create graph data for GNN and move to GPU
#             mrX_graph = create_graph_data(state, 'MrX', env).to(device)
#             police_graphs = [
#                 create_graph_data(state, f'Police{i}', env).to(device)
#                 for i in range(num_agents)
#             ]
#             logger.log(f"Created graph data for MrX and Police agents.",level="debug")

#             # MrX selects an action
#             # mrX_action = mrX_agent.select_action(mrX_graph, torch.ones(mrX_action_size, device=device))

#             mrX_action_size = env.action_space('MrX').n
#             mrX_possible_moves = env.get_possible_moves(0)
#             action_mask = torch.zeros(mrX_graph.num_nodes, dtype=torch.int32, device=device)
#             action_mask[ mrX_possible_moves] = 1
#             mrX_action = mrX_agent.select_action(mrX_graph,action_mask)
#             logger.log(f"MrX selected action: {mrX_action}",level="debug")

#             # Police agents select actions
#             agent_actions = {'MrX': mrX_action}
#             for i in range(num_agents):
#                 police_action_size = env.action_space(f'Police{i}').n
#                 police_possible_moves = env.get_possible_moves(i+1)
#                 action_mask = torch.zeros(police_graphs[i].num_nodes, dtype=torch.int32, device=device)
#                 action_mask[ police_possible_moves] = 1
#                 police_action = police_agent.select_action(
#                     police_graphs[i],
#                     action_mask
#                 )
#                 agent_actions[f'Police{i}'] = police_action
#                 logger.log(f"Police{i} selected action: {police_action}",level="debug")

#             # Execute actions for MrX and Police
#             next_state, rewards, terminations, truncation, winner, _ = env.step(agent_actions)
#             logger.log(f"Executed actions. Rewards: {rewards}, Terminations: {terminations}, Truncations: {truncation}",level="debug")

#             done = terminations.get('Police0', False) or all(truncation.values())
#             logger.log(f"Episode done: {done}",level="debug")

#             total_reward += rewards.get('MrX', 0.0)
#             state = next_state
#             logger.log(f"Total reward updated to: {total_reward}",level="debug")
#             if done:
#                 if winner == 'MrX':
#                     wins += 1
#                     logger.log(f"MrX won the evaluation episode.",level="info")
#                 else:
#                     logger.log(f"MrX lost the evaluation episode.",level="info")

#     win_ratio = wins / args.num_eval_episodes
#     logger.log(f"Evaluation completed. Win Ratio: {win_ratio}")

#     logger.log(f"Epoch {epoch + 1}, Episode {episode + 1}, Total Reward: {total_reward}",level="debug")

#     # win_ratio = evaluate_agent_balance(mrX_agent, police_agent, env, args.num_eval_episodes, device)
#     logger.log(f"Epoch {epoch + 1}: Win Ratio: {win_ratio}",level="info")

#     target_difficulty = compute_target_difficulty(win_ratio)
#     logger.log(f"Epoch {epoch + 1}: Computed target difficulty: {target_difficulty}",level="info")

#     # Train the DifficultyNet based on the difference between predicted and target difficulty
#     target_tensor = torch.FloatTensor([target_difficulty]).to(device).requires_grad_()  # Move target to GPU
#     win_ratio_tensor = torch.FloatTensor([win_ratio]).to(device).requires_grad_()
#     loss = criterion(win_ratio_tensor , target_tensor)
#     logger.log(
#         f"Epoch {epoch + 1}: Loss: {loss.item()}, Win Ratio: {win_ratio}, "
#         f"Real Difficulty: {win_ratio}, Target Difficulty: {target_difficulty}"
#     )
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     logger.log(f"Epoch {epoch + 1}: Optimizer step completed.",level="debug")

#     logger.log_scalar('epoch/loss', loss.item())
#     logger.log_scalar('epoch/win_ratio', win_ratio)

# logger.log("Training completed.")
# logger.close()


