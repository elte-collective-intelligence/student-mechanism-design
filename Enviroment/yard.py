import functools
import random
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from gymnasium.spaces import Discrete, MultiDiscrete, Graph

from Enviroment.base_env import BaseEnvironment
from Enviroment.graph_layout import ConnectedGraph

class CustomEnvironment(BaseEnvironment):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "scotland_yard_env",
    }

    def __init__(self, number_of_agents):
        """
        The init method takes in environment arguments.
        """
        self.number_of_agents = number_of_agents
        self.observation_graph = ConnectedGraph(node_space=Discrete(0), edge_space=Discrete(4, start= 1))
        self.reset()


    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - prisoner x and y coordinates
        - guard x and y coordinates
        - escape x and y coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.board = self.observation_graph.sample(num_nodes=20,num_edges=30)

        self.agents = ["MrX"] + [f"Police{n}" for n in range(self.number_of_agents-1)]
        agent_starting_postions = list(np.random.choice(self.board.nodes.shape[0], size=self.number_of_agents, replace=False, seed=seed))
        
        self.MrX_pos = [agent_starting_postions[0]]
        self.police_positions = agent_starting_postions[1:]

        observations = {
            a : {"MrX_pos": agent_starting_postions[0],
                 "Polices_pos" : agent_starting_postions[1:],
                 "Currency": 1000000000000 if a == "MrX" else 20}
            for a in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - prisoner x and y coordinates
        - guard x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        # Execute actions
        mrX = actions["MrX"]
        mrx_pos = (self.MrX_pos + self.police_positions)[self.agents.index("MrX")]
        pos_to_go = np.concatenate([self.board.edge_links[self.board.edge_links[:,0] == mrx_pos][:,1],
                                        self.board.edge_links[self.board.edge_links[:,1] == mrx_pos][:,0]])[mrX]
        if pos_to_go not in self.police_positions:
            self.MrX_pos = [pos_to_go]
        
        for police in actions.keys():
            if police != "MrX":
                police_action = actions[police]
                police_pos = (self.MrX_pos + self.police_positions)[self.agents.index(police)]
                pos_to_go = np.concatenate([self.board.edge_links[self.board.edge_links[:,0] == police_pos][:,1],
                                                self.board.edge_links[self.board.edge_links[:,1] == police_pos][:,0]])[police_action]
                if pos_to_go not in self.police_positions:
                    self.police_positions[self.agents.index(police)] = pos_to_go

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.MrX_pos in self.police_positions:
            rewards = { a : (-1 if a == "MrX" else 1) for a in self.agents}
            terminations = {a: True for a in self.agents}

        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            rewards = { a : (1 if a == "MrX" else 0) for a in self.agents}
            truncations = {a: True for a in self.agents}
        self.timestep += 1

        # Get observations
        observations = {
            a : {"MrX_pos": self.MrX_pos[0],
                 "Polices_pos" : self.police_positions[1:],
                 "Currency": 1000000000000 if a == "MrX" else 20}
            for a in self.agents
        }

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        G = nx.DiGraph()
        # Add edges and their attributes from the GraphInstance
        for edge,edge_w in zip(self.board.edge_links, self.board.edges):
            G.add_edge(edge[0], edge[1], weight=edge_w)
            G.add_edge(edge[1], edge[0], weight=edge_w)
        pos = nx.shell_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='green')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self.board

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        agent_pos = (self.MrX_pos + self.police_positions)[self.agents.index(agent)]
        posible_nodes = np.concatenate([self.board.edge_links[self.board.edge_links[:,0] == agent_pos][:,1],
                                        self.board.edge_links[self.board.edge_links[:,1] == agent_pos][:,0]])
        return Discrete(posible_nodes.shape[0])