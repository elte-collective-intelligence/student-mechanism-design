import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from logger import Logger  # Your custom Logger class
from RLAgent.dqn_agent import DQNAgent
from Enviroment.yard import CustomEnvironment

class DifficultyNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=1):
        super(DifficultyNet, self).__init__()
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

    # Initialize DifficultyNet and its optimizer
    difficulty_net = DifficultyNet()
    optimizer_difficulty = optim.Adam(difficulty_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    logger.log(f"Starting training with {args.num_agents} agents and {args.agent_money} money per agent.")

    for epoch in range(args.epochs):
        num_agents = args.num_agents
        agent_money = args.agent_money

        # Predict the difficulty from number of agents and money
        inputs = torch.FloatTensor([[num_agents, agent_money]])
        predicted_difficulty = difficulty_net(inputs)

        # Log predicted difficulty
        logger.log(f"Epoch {epoch + 1}: Predicted difficulty: {predicted_difficulty.item()}")

        # Create environment with predicted difficulty
        env = CustomEnvironment(number_of_agents=num_agents+1, agent_money=agent_money, difficulty=predicted_difficulty.item())

        # Determine action sizes for MrX and Police agents
        mrX_action_size = env.action_space('MrX').n
        police_action_size = env.action_space('Police0').n  # Assuming all police have the same action space

        # Initialize DQN agents with correct action sizes
        mrX_agent = DQNAgent(state_size=args.state_size, action_size=mrX_action_size)  # Replace ... with other hyperparameters as needed
        police_agent = DQNAgent(state_size=args.state_size, action_size=police_action_size)

        # Train the MrX and Police agents in the environment
        for episode in range(args.num_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                # MrX selects an action
                mrX_action = mrX_agent.select_action(state['MrX']['MrX_pos'], np.ones(mrX_action_size))

                # All police share the same policy and select the same action network
                agent_actions = {}
                agent_actions['MrX'] = mrX_action
                for i in range(args.num_police_agents):  # Assuming multiple police agents
                    police_action = police_agent.select_action(state[f'Police{i}']['Polices_pos'][0], np.ones(police_action_size))
                    agent_actions[f'Police{i}'] = police_action

                # Execute actions for MrX and Police
                next_state, rewards, terminations, truncation, _ = env.step(agent_actions)
                done = terminations.get(f'Police0', False) or all(truncation.values())  # Use .get() to handle missing keys gracefully

                # Update MrX agent
                mrX_agent.update(
                    state['MrX']['MrX_pos'],
                    mrX_action,
                    rewards.get('MrX', 0.0),
                    next_state['MrX']['MrX_pos'],
                    not terminations.get('Police0', False)
                )

                # Update shared police agent
                for i in range(args.num_police_agents):
                    police_action = agent_actions.get(f'Police{i}')
                    if police_action is not None:
                        police_agent.update(
                            state[f'Police{i}']['Polices_pos'][0],
                            police_action,
                            rewards.get(f'Police{i}', 0.0),
                            next_state[f'Police{i}']['Polices_pos'][0],
                            terminations.get(f'Police{i}', False)
                        )

                total_reward += rewards.get('MrX', 0.0)
                state = next_state

            logger.log(f"Epoch {epoch+1}, Episode {episode+1}, Total Reward: {total_reward}")
            logger.log_scalar('total_reward', total_reward, epoch * args.num_episodes + episode)

        # Evaluate performance and calculate the target difficulty
        win_ratio = evaluate_agent_balance(mrX_agent, police_agent, env, args.num_eval_episodes)
        target_difficulty = compute_target_difficulty(win_ratio)

        # Train the DifficultyNet based on the difference between predicted and target difficulty
        loss = criterion(predicted_difficulty, torch.FloatTensor([target_difficulty]))
        logger.log(f"Epoch {epoch+1}: Loss: {loss.item()}, Win Ratio: {win_ratio}, Predicted Difficulty: {predicted_difficulty.item()}, Target Difficulty: {target_difficulty}")
        optimizer_difficulty.zero_grad()
        loss.backward()
        optimizer_difficulty.step()

        logger.log_scalar('loss', loss.item(), epoch)
        logger.log_scalar('win_ratio', win_ratio, epoch)

    logger.close()


def evaluate_agent_balance(mrX_agent, police_agent, env, num_eval_episodes):
    """Evaluate the agents' win ratio."""
    wins = 0
    mrX_action_size = env.action_space('MrX').n
    police_action_size = env.action_space('Police0').n 
    
    for episode in range(num_eval_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # MrX selects an action
            agent_actions = {}
            mrX_action =  mrX_agent.select_action(state['MrX']['MrX_pos'], np.ones(mrX_action_size))
            agent_actions['MrX'] = mrX_action
            # All police share the same policy and select the same action
            for i in range(args.num_police_agents):  # Assuming multiple police agents
                police_action = police_agent.select_action(state[f'Police{i}']['Polices_pos'][0], np.ones(police_action_size))
                agent_actions[f'Police{i}'] = police_action

            next_state, rewards, terminations, truncation, _ = env.step(agent_actions)
            done = terminations.get(f'Police0', False) or all(truncation.values()) 
            if rewards['MrX'] > 0 or all(truncation.values()):
                wins += 1
            state = next_state

    win_ratio = wins / num_eval_episodes
    return win_ratio

def compute_target_difficulty(win_ratio, target_balance=0.5):
    """Adjust the target difficulty based on the win/loss ratio."""
    return win_ratio

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MrX and Police agents with dynamic difficulty prediction.")
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file.')

    parser.add_argument('--num_agents', type=int, default=2, help='Initial number of agents in the environment')
    parser.add_argument('--agent_money', type=float, default=10.0, help='Initial money for each agent')
    parser.add_argument('--state_size', type=int, default=1, help='State size for the agent')
    parser.add_argument('--action_size', type=int, default=5, help='Action size for the agent')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes per epoch')
    parser.add_argument('--num_eval_episodes', type=int, default=20, help='Number of evaluation episodes')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--num_police_agents', type=int, default=3, help='Number of police agents')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory where logs will be saved')
    parser.add_argument('--wandb_api_key', type=str, help='Weights & Biases API key')
    parser.add_argument('--wandb_project', type=str, help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, help='Weights & Biases entity (user or team)')
    parser.add_argument('--wandb_run_name', type=str, help='Custom name for the Weights & Biases run')
    parser.add_argument('--wandb_resume', action='store_true', help='Resume Weights & Biases run if it exists')

    args = parser.parse_args()
    train(args)
