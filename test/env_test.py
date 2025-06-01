
import torch, sys, os

from dataclasses import dataclass

from torchrl.envs.libs.pettingzoo import PettingZooWrapper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from Enviroment.yard import CustomEnvironment
from logger import Logger
from reward_net import RewardWeightNet


@dataclass
class DummyArgs:
    num_agents: int = 5
    agent_money: int = 10
    graph_nodes: int = 15
    graph_edges: int = 20

vis_conf = {"visualize_game": False, "visualize_heatmap": False, "save_visualization": False, "save_dir": 'logs/vis'}
logger = Logger(wandb_api_key="", configs={"verbose": False, "log_dir": 'logs', "log_file": 'run.log'})

def test_env_reset_does_not_throw():
    args = DummyArgs()
    env_wrappable = CustomEnvironment(
        number_of_agents=args.num_agents,
        agent_money=args.agent_money,
        reward_weights=reward_weights(args),
        logger=logger,
        epoch=0,
        graph_nodes=args.graph_nodes,
        graph_edges=args.graph_edges,
        vis_configs=vis_conf
    )
    env = PettingZooWrapper(env=env_wrappable)
    try:
        env.reset()
    except Exception as e:
        assert False, f"env.reset() raised an exception: {e}"

# def test_env_step_does_not_throw():
#     args = DummyArgs()
#     env_wrappable = CustomEnvironment(
#         number_of_agents=args.num_agents,
#         agent_money=args.agent_money,
#         reward_weights=reward_weights(args),
#         logger=logger,
#         epoch=0,
#         graph_nodes=args.graph_nodes,
#         graph_edges=args.graph_edges,
#         vis_configs=vis_conf
#     )
#     env = PettingZooWrapper(env=env_wrappable)
#     env.reset()
#     agent = next(iter(env.agent_iter()))
#     action = env.action_space(agent).sample()
#     try:
#         env.step(action)
#     except Exception as e:
#         assert False, f"env.step() raised an exception: {e}"

def reward_weights(args):
    reward_weight_net = RewardWeightNet()

    inputs = torch.FloatTensor([[args.num_agents, args.agent_money, args.graph_nodes, args.graph_edges]])
    predicted_weight = reward_weight_net(inputs)
    reward_weights = {
        "Police_distance": predicted_weight[0, 0],
        "Police_group": predicted_weight[0, 1],
        "Police_position": predicted_weight[0, 2],
        "Police_time": predicted_weight[0, 3],
        "Mrx_closest": predicted_weight[0, 4],
        "Mrx_average": predicted_weight[0, 5],
        "Mrx_position": predicted_weight[0, 6],
        "Mrx_time": predicted_weight[0, 7]
    }
    return reward_weights