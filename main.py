import argparse

import yaml
from Enviroment.yard import CustomEnvironment
from MetaLearning.maml import MAMLMetaLearningSystem
from RLAgent.random_agent import RandomAgent
from logger import Logger  

def main(args):
    logger = Logger(
        log_dir=args.log_dir,
        wandb_api_key=args.wandb_api_key,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_resume=args.wandb_resume
    )

    policy = RandomAgent()
    maml_system = MAMLMetaLearningSystem(policy, logger=logger)

    logger.log("Creating meta-training and meta-testing tasks...")
    meta_train_tasks = [
        CustomEnvironment(args.num_agents, args.agent_money) for _ in range(args.num_meta_train_tasks)
    ]
    meta_test_tasks = [
        CustomEnvironment(args.num_agents, args.agent_money) for _ in range(args.num_meta_test_tasks)
    ]

    logger.log("Starting meta-training...")
    maml_system.meta_train(meta_train_tasks)

    logger.log("Starting meta-evaluation...")
    maml_system.meta_evaluate(meta_test_tasks)

    maml_system.save_meta_model(args.save_path)
    logger.log(f"Meta-trained model saved to {args.save_path}")

    logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a MAML meta-learning system.")
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file.')

    parser.add_argument('--num_agents', type=int, default=2, help='Number of agents in the environment')
    parser.add_argument('--agent_money', type=float, default=10.0, help='Initial money for each agent')
    parser.add_argument('--num_meta_train_tasks', type=int, default=10, help='Number of meta-training tasks')
    parser.add_argument('--num_meta_test_tasks', type=int, default=5, help='Number of meta-testing tasks')
    parser.add_argument('--save_path', type=str, default='maml_policy.pth', help='Path to save the trained model')

    parser.add_argument('--log_dir', type=str, default='logs', help='Directory where logs will be saved')
    parser.add_argument('--wandb_api_key', type=str, help='Weights & Biases api key')
    parser.add_argument('--wandb_project', type=str, help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, help='Weights & Biases entity (user or team)')
    parser.add_argument('--wandb_run_name', type=str, help='Custom name for the Weights & Biases run')
    parser.add_argument('--wandb_resume', action='store_true', help='Resume Weights & Biases run if it exists')

    args = parser.parse_args()

    # If config file is provided, override the defaults
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(args, key, value)

    main(args)
