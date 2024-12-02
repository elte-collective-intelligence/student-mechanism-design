import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from logger import Logger  # Your custom Logger class
from RLAgent.gnn_agent import GNNAgent
from Enviroment.yard import CustomEnvironment
from torch_geometric.data import Data

# Define the device at the beginning
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")  # You may consider logging this instead

class RewardWeightNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=8):
        super(RewardWeightNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x) 

def train(args):
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

    optimizer_difficulty = optim.Adam(reward_weight_net.parameters(), lr=0.001)
    logger.log("Optimizer for DifficultyNet initialized.",level="debug")

    criterion = nn.MSELoss()
    logger.log("Loss function (MSELoss) initialized.",level="debug")

    logger.log(f"Starting training with {args.num_police_agents} agents and {args.agent_money} money per agent.",level="debug")

    for epoch in range(args.epochs):
        logger.log_scalar('epoch_step', epoch)

        logger.log(f"Starting epoch {epoch + 1}/{args.epochs}.",level="info")
        num_agents = args.num_police_agents - 1
        agent_money = args.agent_money

        # Predict the difficulty from the number of agents and money
        inputs = torch.FloatTensor([[num_agents, agent_money]]).to(device)  # Move inputs to GPU
        predicted_weight = reward_weight_net(inputs)
        # print(predicted_weight)
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

        logger.log(f"Epoch {epoch + 1}: Predicted weights: {reward_weights}")
        logger.log_weights(reward_weights)
        # Create environment with predicted difficulty
        env = CustomEnvironment(
            number_of_agents=num_agents + 1,
            agent_money=agent_money,
            reward_weights=reward_weights,
            logger=logger,
            epoch=epoch
        )
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
            state, _ = env.reset(episode=episode)
            done = False
            total_reward = 0

            while not done:
                # Create graph data for GNN and move to GPU
                mrX_graph = create_graph_data(state, 'MrX', env).to(device)
                police_graphs = [
                    create_graph_data(state, f'Police{i}', env).to(device)
                    for i in range(args.num_police_agents)
                ]
                logger.log(f"Created graph data for MrX and Police agents.",level="debug")

                # MrX selects an action
                mrX_action = mrX_agent.select_action(mrX_graph, torch.ones(mrX_action_size, device=device))
                logger.log(f"MrX selected action: {mrX_action}",level="debug")

                # Police agents select actions
                agent_actions = {'MrX': mrX_action}
                for i in range(args.num_police_agents):
                    police_action = police_agent.select_action(
                        police_graphs[i],
                        torch.ones(police_action_size, device=device)
                    )
                    agent_actions[f'Police{i}'] = police_action
                    logger.log(f"Police{i} selected action: {police_action}",level="debug")

                # Execute actions for MrX and Police
                next_state, rewards, terminations, truncation, _ = env.step(agent_actions)
                logger.log(f"Executed actions. Rewards: {rewards}, Terminations: {terminations}, Truncations: {truncation}",level="debug")

                done = terminations.get('Police0', False) or all(truncation.values())
                logger.log(f"Episode done: {done}",level="debug")

                # Update MrX agent
                mrX_next_graph = create_graph_data(next_state, 'MrX', env).to(device)
                mrX_agent.update(
                    mrX_graph,
                    mrX_action,
                    rewards.get('MrX', 0.0),
                    mrX_next_graph,
                    not terminations.get('Police0', False)
                )
                logger.log(f"MrX agent updated with reward: {rewards.get('MrX', 0.0)}",level="debug")

                # Update shared police agent
                for i in range(args.num_police_agents):
                    police_next_graph = create_graph_data(next_state, f'Police{i}', env).to(device)
                    police_agent.update(
                        police_graphs[i],
                        agent_actions.get(f'Police{i}'),
                        rewards.get(f'Police{i}', 0.0),
                        police_next_graph,
                        terminations.get(f'Police{i}', False)
                    )
                    logger.log(f"Police{i} agent updated with reward: {rewards.get(f'Police{i}', 0.0)}",level="debug")

                total_reward += rewards.get('MrX', 0.0)
                state = next_state
                logger.log(f"Total reward updated to: {total_reward}",level="debug")

            logger.log(f"Epoch {epoch + 1}, Episode {episode + 1}, Total Reward: {total_reward}",level="debug")
            # logger.log_scalar(f'Episode_total_reward{epoch}', total_reward, episode)

        # Evaluate performance and calculate the target difficulty
        logger.log(f"Evaluating agent balance after epoch {epoch + 1}.",level="debug")
        logger.log_model(mrX_agent, 'MrX')
        logger.log_model(police_agent, 'Police')
        logger.log_model(reward_weight_net, 'RewardWeightNet')

        win_ratio = evaluate_agent_balance(mrX_agent, police_agent, env, args.num_eval_episodes, device)
        logger.log(f"Epoch {epoch + 1}: Win Ratio: {win_ratio}",level="info")

        target_difficulty = compute_target_difficulty(win_ratio)
        logger.log(f"Epoch {epoch + 1}: Computed target difficulty: {target_difficulty}",level="info")

        # Train the DifficultyNet based on the difference between predicted and target difficulty
        target_tensor = torch.FloatTensor([target_difficulty]).to(device)  # Move target to GPU
        loss = criterion(predicted_weight, target_tensor)
        logger.log(
            f"Epoch {epoch + 1}: Loss: {loss.item()}, Win Ratio: {win_ratio}, "
            f"Predicted Difficulty: {predicted_weight}, Target Difficulty: {target_difficulty}"
        )
        optimizer_difficulty.zero_grad()
        loss.backward()
        optimizer_difficulty.step()
        logger.log(f"Epoch {epoch + 1}: Optimizer step completed.",level="debug")

        logger.log_scalar('epoch/loss', loss.item())
        logger.log_scalar('epoch/win_ratio', win_ratio)

    logger.log("Training completed.")
    logger.close()

def create_graph_data(state, agent_id, env):
    """
    Create a PyTorch Geometric Data object from the environment state for the specified agent.
    """
    logger = env.logger  # Access the logger from the environment
    logger.log(f"Creating graph data for agent {agent_id}.",level="debug")

    edge_index = torch.tensor(env.board.edge_links.T, dtype=torch.long)
    edge_features = torch.tensor(env.board.edges, dtype=torch.float)

    num_nodes = env.board.nodes.shape[0]
    num_features = env.number_of_agents + 1  # Adjust based on your feature design

    # Ensure num_features is an integer greater than 0
    node_features = np.zeros((num_nodes, num_features), dtype=np.float32)

    # Highlight MrX position
    mrX_pos = state.get('MrX', {}).get('MrX_pos', None)
    if mrX_pos is not None:
        node_features[mrX_pos, 0] = 1  # MrX is at index 0
        logger.log(f"Agent {agent_id}: MrX position encoded at node {mrX_pos}.",level="debug")

    # Highlight Police positions
    for i in range(env.number_of_agents - 1):
        police_pos = state.get(f'Police{i}', {}).get('Polices_pos', None)
        if police_pos is not None and len(police_pos) > 0:
            node_features[police_pos[0], i + 1] = 1  # Police indices start from 1
            logger.log(f"Agent {agent_id}: Police{i} position encoded at node {police_pos[0]}.",level="debug")

    node_features = torch.tensor(node_features, dtype=torch.float32)

    # Move tensors to the appropriate device
    edge_index = edge_index.to(device)
    edge_features = edge_features.to(device)
    node_features = node_features.to(device)

    # Create PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    logger.log(f"Graph data for agent {agent_id} created.",level="debug")
    return data

def evaluate_agent_balance(mrX_agent, police_agent, env, num_eval_episodes, device):
    """Evaluate the agents' win ratio."""
    logger = env.logger  # Access the logger from the environment
    logger.log(f"Starting evaluation of agent balance over {num_eval_episodes} episodes.")
    wins = 0
    mrX_action_size = env.action_space('MrX').n
    police_action_size = env.action_space('Police0').n 

    for episode in range(num_eval_episodes):
        logger.log(f"Evaluation Episode {episode + 1} started.")
        state, _ = env.reset(episode=episode)
        done = False
        while not done:

            mrX_graph = create_graph_data(state, 'MrX', env).to(device)
            police_graphs = [
                create_graph_data(state, f'Police{i}', env).to(device)
                for i in range(args.num_police_agents)
            ]

            mrX_action = mrX_agent.select_action(mrX_graph, torch.ones(mrX_action_size, device=device))
            logger.log(f"MrX selected action: {mrX_action}",level="debug")

            # Police agents select actions
            agent_actions = {'MrX': mrX_action}
            for i in range(args.num_police_agents):
                police_action = police_agent.select_action(
                    police_graphs[i],
                    torch.ones(police_action_size, device=device)
                )
                agent_actions[f'Police{i}'] = police_action
                logger.log(f"Police{i} selected action: {police_action}",level="debug")
 
            next_state, rewards, terminations, truncation, _ = env.step(agent_actions)
            logger.log(f"Evaluation Episode {episode + 1}: Executed actions. Rewards: {rewards}, Terminations: {terminations}, Truncations: {truncation}",level="debug")

            done = terminations.get(f'Police0', False) or all(truncation.values()) 
            if rewards.get('MrX', 0.0) > 0 or all(truncation.values()):
                wins += 1
                logger.log(f"Evaluation Episode {episode + 1}: MrX won.")
            else:
                logger.log(f"Evaluation Episode {episode + 1}: MrX lost.")

            state = next_state

    win_ratio = wins / num_eval_episodes
    logger.log(f"Evaluation completed. Win Ratio: {win_ratio}")
    return win_ratio

def compute_target_difficulty(win_ratio, target_balance=0.5):
    """Adjust the target difficulty based on the win/loss ratio."""
    # You can add logging here if needed
    return target_balance

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Train MrX and Police agents with dynamic difficulty prediction.")
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file.')

    # Add all the other arguments with default=argparse.SUPPRESS
    parser.add_argument('--num_agents', type=int, default=argparse.SUPPRESS, help='Initial number of agents in the environment')
    parser.add_argument('--agent_money', type=float, default=argparse.SUPPRESS, help='Initial money for each agent')
    parser.add_argument('--state_size', type=int, default=argparse.SUPPRESS, help='State size for the agent')
    parser.add_argument('--action_size', type=int, default=argparse.SUPPRESS, help='Action size for the agent')
    parser.add_argument('--num_episodes', type=int, default=argparse.SUPPRESS, help='Number of episodes per epoch')
    parser.add_argument('--num_eval_episodes', type=int, default=argparse.SUPPRESS, help='Number of evaluation episodes')
    parser.add_argument('--epochs', type=int, default=argparse.SUPPRESS, help='Number of training epochs')
    parser.add_argument('--num_police_agents', type=int, default=argparse.SUPPRESS, help='Number of police agents')
    parser.add_argument('--log_dir', type=str, default=argparse.SUPPRESS, help='Directory where logs will be saved')
    parser.add_argument('--wandb_api_key', type=str, default=argparse.SUPPRESS, help='Weights & Biases API key')
    parser.add_argument('--wandb_project', type=str, default=argparse.SUPPRESS, help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=argparse.SUPPRESS, help='Weights & Biases entity (user or team)')
    parser.add_argument('--wandb_run_name', type=str, default=argparse.SUPPRESS, help='Custom name for the Weights & Biases run')
    parser.add_argument('--wandb_resume', action='store_true', help='Resume Weights & Biases run if it exists')

    # Parse command-line arguments
    args = parser.parse_args()
    args_dict = vars(args)

    # Default values for all parameters
    default_values = {
        'agent_money': 10.0,
        'state_size': 1,
        'action_size': 5,
        'num_episodes': 1,
        'num_eval_episodes': 10,
        'epochs': 10,
        'num_police_agents': 3,
        'log_dir': 'logs',
        'wandb_api_key': None,
        'wandb_project': None,
        'wandb_entity': None,
        'wandb_run_name': None,
        'wandb_resume': False,
    }

    # If a config file is provided, load its parameters
    if args.config:
        with open(args.config, 'r') as f:
            config_args = yaml.safe_load(f)
        # Remove 'config' key if present to prevent overwriting
        config_args.pop('config', None)
    else:
        config_args = {}

    # Combine default values, config file, and command-line arguments
    # Priority: command-line args > config file > default values
    combined_args = {**default_values, **config_args, **args_dict}

    # Convert combined_args to Namespace
    args = argparse.Namespace(**combined_args)

    train(args)
