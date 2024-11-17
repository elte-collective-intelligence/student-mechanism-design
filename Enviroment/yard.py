import functools
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from gymnasium.spaces import Discrete, MultiDiscrete, Graph
from Enviroment.base_env import BaseEnvironment
from Enviroment.graph_layout import ConnectedGraph

class CustomEnvironment(BaseEnvironment):
    metadata = {"name": "scotland_yard_env"}

    def __init__(self, number_of_agents, agent_money, difficulty, logger):
        """
        Initialize the environment with given parameters.
        """
        self.difficulty = difficulty  # Difficulty parameter (0 to 1)
        self.number_of_agents = number_of_agents  # Number of police agents
        self.observation_graph = ConnectedGraph(node_space=Discrete(1), edge_space=Discrete(4, start=1))
        self.logger = logger
        self.logger.log(f"Initializing CustomEnvironment with number_of_agents={number_of_agents}, agent_money={agent_money}, difficulty={difficulty}", level="debug")
        self.agent_money = agent_money
        self.reset()

        hyperparams = {
            "number_of_agents": self.number_of_agents,
            "agent_money": self.agent_money,
        }
        self.logger.log_hyperparameters(hyperparams)
        self.logger.log("Environment initialized with hyperparameters: " + str(hyperparams))

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        self.logger.log("Resetting the environment.", level="debug")
        self.board = self.observation_graph.sample(num_nodes=20, num_edges=30)
        self.logger.log(f"Generated board with {self.board.nodes.shape[0]} nodes and {self.board.edge_links.shape[0]} edges.", level="debug")

        self.agents = ["MrX"] + [f"Police{n}" for n in range(self.number_of_agents)]
        agent_starting_positions = list(
            np.random.choice(self.board.nodes.shape[0], size=self.number_of_agents + 1, replace=False)
        )
        self.logger.log(f"Agent starting positions: {agent_starting_positions}", level="debug")

        self.MrX_pos = [agent_starting_positions[0]]
        self.police_positions = agent_starting_positions[1:]
        self.timestep = 0
        self.logger.log(f"MrX initial position: {self.MrX_pos[0]}", level="debug")
        self.logger.log(f"Police initial positions: {self.police_positions}", level="debug")

        observations = self._get_graph_observations()
        infos = {a: {} for a in self.agents}  # Dummy infos
        self.logger.log("Environment reset complete.", level="debug")
        return observations, infos

    def step(self, actions):
        """
        Execute actions for all agents and update the environment.
        """
        self.logger.log(f"Step {self.timestep}: Received actions: {actions}", level="debug")
        mrX_action = actions["MrX"]
        mrx_pos = self.MrX_pos[0]
        self.logger.log(f"MrX current position: {mrx_pos}, action taken: {mrX_action}", level="debug")

        # Process MrX's action
        possible_positions = self._get_possible_moves(mrx_pos)
        self.logger.log(f"MrX possible moves from position {mrx_pos}: {possible_positions}", level="debug")
        if mrX_action < len(possible_positions):
            pos_to_go = possible_positions[mrX_action]
            self.logger.log(f"MrX moves to position {pos_to_go}", level="debug")
        else:
            pos_to_go = mrx_pos  # Stay in the same position if the action is out of bounds
            self.logger.log(f"MrX action out of bounds. Staying at position {pos_to_go}", level="debug")

        if pos_to_go not in self.police_positions:
            self.MrX_pos = [pos_to_go]
            self.logger.log(f"MrX position updated to {self.MrX_pos[0]}", level="debug")
        else:
            self.logger.log(f"MrX move blocked by police at position {pos_to_go}", level="debug")

        # Process police actions
        for police in actions.keys():
            if police != "MrX":
                police_index = self.agents.index(police) - 1
                police_pos = self.police_positions[police_index]
                self.logger.log(f"{police} current position: {police_pos}, action taken: {actions[police]}", level="debug")
                possible_positions = self._get_possible_moves(police_pos)
                self.logger.log(f"{police} possible moves from position {police_pos}: {possible_positions}", level="debug")

                police_action = actions[police]
                if police_action < len(possible_positions):
                    pos_to_go = possible_positions[police_action]
                    self.logger.log(f"{police} moves to position {pos_to_go}", level="debug")
                else:
                    pos_to_go = police_pos  # Stay in the same position if the action is out of bounds
                    self.logger.log(f"{police} action out of bounds. Staying at position {pos_to_go}, ",level="debug")

                if pos_to_go not in self.police_positions:
                    self.police_positions[police_index] = pos_to_go
                    self.logger.log(f"{police} position updated to {self.police_positions[police_index]}, ",level="debug")
                else:
                    self.logger.log(f"{police} move blocked by another police at position {pos_to_go}, ",level="debug")

        # Compute rewards and check termination/truncation
        rewards, terminations, truncations = self._calculate_rewards_terminations()
        self.logger.log(f"Rewards: {rewards}, ",level="debug")
        self.logger.log(f"Terminations: {terminations}, ",level="debug")
        self.logger.log(f"Truncations: {truncations}, ",level="debug")

        # Get new observations
        observations = self._get_graph_observations()
        infos = {a: {} for a in self.agents}
        self.logger.log("Generated new observations., ",level="debug")

        if any(terminations.values()) or all(truncations.values()):
            self.logger.log("Termination or truncation condition met. Ending episode., ",level="debug")
            self.agents = []

        self.logger.log(f"Step {self.timestep} completed., ",level="debug")
        return observations, rewards, terminations, truncations, infos

    def _get_graph_observations(self):
        """
        Create graph-based observations for all agents.
        Includes adjacency matrix, node features, and edge features.
        """
        self.logger.log("Generating graph observations., ",level="debug")
        adjacency_matrix = self._get_adjacency_matrix()
        node_features = np.zeros((self.board.nodes.shape[0], self.number_of_agents + 1))

        # Encode agent positions as node features
        node_features[self.MrX_pos[0], 0] = 1  # MrX position
        self.logger.log(f"MrX position encoded in node features: {self.MrX_pos[0]}, ",level="debug")
        for i, pos in enumerate(self.police_positions):
            node_features[pos, i + 1] = 1  # Police positions
            self.logger.log(f"Police{i} position encoded in node features: {pos}, ",level="debug")

        edge_index = self.board.edge_links.T  # Edge index for GNN (source, target pairs)
        edge_features = self.board.edges  # Edge weights

        observations = {
            agent: {
                "adjacency_matrix": adjacency_matrix,
                "node_features": node_features,
                "edge_index": edge_index,
                "edge_features": edge_features,
                "MrX_pos": self.MrX_pos[0],
                "Polices_pos" : self.police_positions[1:],
                "Currency": 1000000000000 if agent == "MrX" else 20
            }
            for agent in self.agents
        }
        self.logger.log("Graph observations generated., ",level="debug")
        return observations

    def _calculate_rewards_terminations(self):
        """
        Compute rewards and check termination/truncation conditions.
        """
        self.logger.log("Calculating rewards and checking termination conditions., ",level="debug")
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}

        if self.MrX_pos[0] in self.police_positions:
            self.logger.log("MrX has been caught by the police., ",level="debug")
            rewards = {a: self.difficulty * (-1 if a == "MrX" else 1) for a in self.agents}
            terminations = {a: True for a in self.agents}
        elif self.timestep > 100:
            self.logger.log("Maximum timestep exceeded. Truncating episode., ",level="debug")
            rewards = {a: self.difficulty * (1 if a == "MrX" else 0) for a in self.agents}
            truncations = {a: True for a in self.agents}
        else:
            rewards = self.calculate_rewards()

        self.timestep += 1
        self.logger.log(f"Updated timestep to {self.timestep}, ",level="debug")
        return rewards, terminations, truncations
    def calculate_rewards(self):
        """
        Compute rewards for all agents based on the specified components.
        Rewards are weighted by the `difficulty` parameter, which ranges from 0 to 1.
        """
        self.logger.log("Calculating individual rewards for agents., ", level="debug")
        rewards = {}
        mrX_pos = self.MrX_pos[0]

        # Compute rewards for MrX
        police_distances = [self.get_distance(mrX_pos, police_pos) for police_pos in self.police_positions]
        self.logger.log(f"Police distances from MrX: {police_distances}, ", level="debug")
        closest_distance = min(police_distances)
        avg_distance = np.mean(police_distances)
        self.logger.log(f"MrX closest distance: {closest_distance}, average distance: {avg_distance}, ", level="debug")

        mrX_reward = (
            self.difficulty * (-1 / (closest_distance + 1))  # Distance penalty
            + self.difficulty * (-1 / (avg_distance + 1))  # Average distance penalty
            + (1 - self.difficulty) * (0.1 * self.timestep)  # Time reward
        )
        rewards["MrX"] = mrX_reward
        self.logger.log(f"MrX reward: {mrX_reward}, ", level="debug")

        # Log MrX reward components as scalars
        distance_penalty_mrX = -1 / (closest_distance + 1)
        avg_distance_penalty_mrX = -1 / (avg_distance + 1)
        time_reward_mrX = 0.1 * self.timestep

        self.logger.log_scalar('MrX_distance_penalty', distance_penalty_mrX, self.timestep)
        self.logger.log_scalar('MrX_avg_distance_penalty', avg_distance_penalty_mrX, self.timestep)
        self.logger.log_scalar('MrX_time_reward', time_reward_mrX, self.timestep)
        self.logger.log_scalar('MrX_total_reward', mrX_reward, self.timestep)

        # Compute rewards for police
        for i, police in enumerate(self.agents[1:]):  # Skip MrX
            police_pos = self.police_positions[i]
            distance_to_mrX = self.get_distance(police_pos, mrX_pos)
            group_penalty = sum(
                np.exp(-self.get_distance(police_pos, other_police_pos))
                for j, other_police_pos in enumerate(self.police_positions)
                if i != j
            )
            position_penalty = len(self._get_possible_moves(police_pos))
            self.logger.log(f"{police} distance to MrX: {distance_to_mrX}, group penalty: {group_penalty}, position penalty: {position_penalty}, ", level="debug")

            police_reward = (
                self.difficulty * (np.exp(-distance_to_mrX))  # Distance reward
                - self.difficulty * (group_penalty)  # Grouping penalty
                + self.difficulty * (position_penalty)  # Position reward
                - (1 - self.difficulty) * (0.05 * self.timestep)  # Time penalty
            )
            rewards[police] = police_reward
            self.logger.log(f"{police} reward: {police_reward}, ", level="debug")

            # Log Police reward components as scalars
            distance_reward_police = np.exp(-distance_to_mrX)
            grouping_penalty_police = group_penalty
            position_reward_police = position_penalty
            time_penalty_police = 0.05 * self.timestep

            self.logger.log_scalar(f'{police}_distance_reward', distance_reward_police, self.timestep)
            self.logger.log_scalar(f'{police}_grouping_penalty', grouping_penalty_police, self.timestep)
            self.logger.log_scalar(f'{police}_position_reward', position_reward_police, self.timestep)
            self.logger.log_scalar(f'{police}_time_penalty', time_penalty_police, self.timestep)
            self.logger.log_scalar(f'{police}_total_reward', police_reward, self.timestep)

        self.logger.log(f"All rewards calculated: {rewards}, ", level="debug")
        return rewards


    def get_distance(self, node1, node2):
        """
        Compute the shortest path distance between two nodes using BFS.
        """
        self.logger.log(f"Calculating distance between node {node1} and node {node2}., ",level="debug")
        if node1 == node2:
            self.logger.log("Both nodes are the same. Distance is 0., ",level="debug")
            return 0
        queue = [(node1, 0)]  # (current node, distance)
        visited = set()
        visited.add(node1)

        while queue:
            current, dist = queue.pop(0)
            neighbors = np.concatenate([
                self.board.edge_links[self.board.edge_links[:, 0] == current][:, 1],
                self.board.edge_links[self.board.edge_links[:, 1] == current][:, 0]
            ])
            self.logger.log(f"Visiting node {current}, distance {dist}. Neighbors: {neighbors}, ",level="debug")
            for neighbor in neighbors:
                if neighbor == node2:
                    self.logger.log(f"Found node {node2} from node {current}. Distance: {dist + 1}, ",level="debug")
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
                    self.logger.log(f"Added node {neighbor} to queue with distance {dist + 1}., ",level="debug")
        self.logger.log(f"No path found between node {node1} and node {node2}. Returning infinity., ",level="debug")
        return float('inf')  # Return infinity if no path exists

    def _get_adjacency_matrix(self):
        """
        Generate the adjacency matrix of the graph.
        """
        self.logger.log("Generating adjacency matrix., ",level="debug")
        adjacency_matrix = np.zeros((self.board.nodes.shape[0], self.board.nodes.shape[0]))
        for edge in self.board.edge_links:
            adjacency_matrix[edge[0], edge[1]] = 1
            adjacency_matrix[edge[1], edge[0]] = 1  # Undirected graph
        self.logger.log("Adjacency matrix generated., ",level="debug")
        return adjacency_matrix

    def _get_possible_moves(self, pos):
        """
        Get possible moves from a node position.
        """
        possible_moves = np.concatenate(
            [
                self.board.edge_links[self.board.edge_links[:, 0] == pos][:, 1],
                self.board.edge_links[self.board.edge_links[:, 1] == pos][:, 0],
            ]
        )
        self.logger.log(f"Possible moves from position {pos}: {possible_moves}, ",level="debug")
        return possible_moves

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Define the observation space for GNN input.
        """
        self.logger.log(f"Defining observation space for agent {agent}., ",level="debug")
        node_features_dim = self.number_of_agents + 1  # MrX + police agents
        adjacency_matrix_shape = (self.board.nodes.shape[0], self.board.nodes.shape[0])

        space = {
            "adjacency_matrix": Graph(adjacency_matrix_shape),
            "node_features": MultiDiscrete([2] * node_features_dim),
            "edge_index": Graph((2, self.board.edge_links.shape[0])),
            "edge_features": Discrete(1),  # Assuming edge weights are single discrete values
        }
        self.logger.log(f"Observation space for agent {agent}: {space}, ",level="debug")
        return space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Define the action space for agents as node indices.
        """
        self.logger.log(f"Defining action space for agent {agent}., ",level="debug")
        space = Discrete(self.board.nodes.shape[0])
        self.logger.log(f"Action space for agent {agent}: {space}, ",level="debug")
        return space

    def render(self, mode="human"):
        """
        Render the current state of the environment.
        Args:
            mode (str): The mode of rendering. Defaults to "human".
        """
        self.logger.log("Rendering environment state.")
        print("Timestep:", self.timestep)
        print("MrX Position:", self.MrX_pos[0])
        print("Police Positions:", self.police_positions)
        print("Graph Edges:", self.board.edge_links)
        self.logger.log(f"Rendered state at timestep {self.timestep}.")
