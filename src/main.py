"""Main entry point for training and evaluating Scotland Yard RL agents.

This module provides a clean interface for training and evaluation:
- GNN-based agents (train_gnn/evaluate_gnn)
- MAPPO-based agents (train_mappo/evaluate_mappo)

Usage:
    # Training with GNN agents
    python main.py --config experiments/smoke_train/config.yml --agent_configs gnn

    # Training with MAPPO agents
    python main.py --config experiments/smoke_train/config.yml --agent_configs mappo

    # Evaluation
    python main.py --config experiments/smoke_train/config.yml --evaluate True
"""

import argparse
import sys
import os
import re
import yaml
import warnings

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Suppress known warnings from TorchRL library
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")
warnings.filterwarnings(
    "ignore", message=".*PettingZoo in TorchRL is tested using version.*"
)

# Import training and evaluation functions
from training.gnn_trainer import train_gnn
from training.mappo_trainer import train_mappo
from training.evaluator import evaluate_gnn, evaluate_mappo


def load_wandb_config():
    """Load WandB configuration from wandb_data.json."""
    try:
        with open(os.path.join(SCRIPT_DIR, "wandb_data.json"), "r") as f:
            import json

            wandb_data = json.load(f)
            return wandb_data
    except FileNotFoundError:
        print("Warning: wandb_data.json not found. WandB logging disabled.")
        return {
            "wandb_api_key": "null",
            "wandb_project": "null",
            "wandb_entity": "null",
        }


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Scotland Yard Multi-Agent RL Training and Evaluation"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file",
    )
    parser.add_argument(
        "--agent_configs",
        type=str,
        default="gnn",
        help="Agent configuration name (gnn, mappo, random)",
    )
    parser.add_argument(
        "--log_configs",
        type=str,
        default="default",
        help="Logger configuration (default, verbose)",
    )
    parser.add_argument(
        "--vis_configs",
        type=str,
        default="none",
        help="Visualization configuration (none, default, full)",
    )
    parser.add_argument(
        "--evaluate",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="Run in evaluation mode",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="WandB API key (overrides config)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name (overrides config)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity/username (overrides config)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name",
    )
    parser.add_argument(
        "--wandb_resume",
        action="store_true",
        help="Resume WandB run if exists",
    )

    return parser.parse_args()


def load_configs(args):
    """Load all configuration files."""
    # Load experiment config
    try:
        with open(args.config, "r") as f:
            experiment_config = yaml.safe_load(f)
        if experiment_config is None:
            experiment_config = {}
    except FileNotFoundError:
        print(f"Error: Configuration file {args.config} not found.", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}", file=sys.stderr)
        sys.exit(1)

    # Setup YAML loader for floats
    yaml_loader = yaml.SafeLoader
    yaml_loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            r"""^(?:
            [-+]?(?:[0-9][0-9_]*)\.[ 0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
            |[-+]?\.(?:inf|Inf|INF)
            |\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    # Load agent configuration
    agent_config_name = experiment_config.get("agent_configs", args.agent_configs)
    with open(
        os.path.join(SCRIPT_DIR, f"configs/agent/{agent_config_name}.yaml"), "r"
    ) as f:
        agent_configs = yaml.load(f, Loader=yaml_loader)

    # Load logger configuration
    logger_config_name = experiment_config.get("log_configs", args.log_configs)
    with open(
        os.path.join(SCRIPT_DIR, f"configs/logger/{logger_config_name}.yaml"), "r"
    ) as f:
        logger_configs = yaml.load(f, Loader=yaml_loader)

    # Load visualization configuration
    vis_config_name = experiment_config.get("vis_configs", args.vis_configs)
    vis_config_path = os.path.join(
        SCRIPT_DIR, f"configs/visualization/{vis_config_name}.yaml"
    )
    with open(vis_config_path, "r") as f:
        visualization_configs = yaml.load(f, Loader=yaml_loader)

    return experiment_config, agent_configs, logger_configs, visualization_configs


def merge_configs(args, experiment_config):
    """Merge command-line args with config files."""
    # Default values
    defaults = {
        "graph_nodes": 50,
        "graph_edges": 110,
        "state_size": 1,
        "action_size": 5,
        "num_episodes": 100,
        "num_eval_episodes": 10,
        "epochs": 50,
        "exp_dir": os.path.dirname(args.config),
        "agent_configurations": [(2, 30), (3, 40), (4, 50)],
        "random_seed": 42,
        "evaluate": False,
    }

    # Load WandB config
    wandb_config = load_wandb_config()
    defaults.update(
        {
            "wandb_api_key": wandb_config.get("wandb_api_key"),
            "wandb_project": wandb_config.get("wandb_project"),
            "wandb_entity": wandb_config.get("wandb_entity"),
            "wandb_run_name": experiment_config.get("wandb_run_name", "default_run"),
            "wandb_resume": False,
        }
    )

    # Merge: defaults < experiment_config < command-line args
    combined = {**defaults, **experiment_config}

    # Override with command-line arguments if provided
    if args.wandb_api_key:
        combined["wandb_api_key"] = args.wandb_api_key
    if args.wandb_project:
        combined["wandb_project"] = args.wandb_project
    if args.wandb_entity:
        combined["wandb_entity"] = args.wandb_entity
    if args.wandb_run_name:
        combined["wandb_run_name"] = args.wandb_run_name
    if args.wandb_resume:
        combined["wandb_resume"] = True
    if args.evaluate:
        combined["evaluate"] = True

    # Handle null values
    for key in ["wandb_api_key", "wandb_project", "wandb_entity"]:
        if combined.get(key) == "null":
            combined[key] = ""

    return argparse.Namespace(**combined)


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()

    # Load configurations
    experiment_config, agent_configs, logger_configs, visualization_configs = (
        load_configs(args)
    )

    # Merge all configs
    combined_args = merge_configs(args, experiment_config)

    # Update logger and vis config paths
    logger_configs["log_dir"] = os.path.join(
        combined_args.exp_dir, logger_configs["log_dir"]
    )
    os.makedirs(logger_configs["log_dir"], exist_ok=True)

    visualization_configs["save_dir"] = os.path.join(
        combined_args.exp_dir, visualization_configs["save_dir"]
    )
    os.makedirs(visualization_configs["save_dir"], exist_ok=True)

    # Route to appropriate function
    if combined_args.evaluate:
        print(f"Starting evaluation with {agent_configs['agent_type']} agents...")
        if agent_configs["agent_type"] == "mappo":
            evaluate_mappo(
                combined_args, agent_configs, logger_configs, visualization_configs
            )
        else:
            evaluate_gnn(
                combined_args, agent_configs, logger_configs, visualization_configs
            )
    else:
        print(f"Starting training with {agent_configs['agent_type']} agents...")
        if agent_configs["agent_type"] == "mappo":
            train_mappo(
                combined_args, agent_configs, logger_configs, visualization_configs
            )
        else:
            train_gnn(
                combined_args, agent_configs, logger_configs, visualization_configs
            )

    print("Done!")


if __name__ == "__main__":
    main()
