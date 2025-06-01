from collections import deque
import functools
import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from gymnasium.spaces import Discrete, MultiDiscrete, Graph, Box, Dict, MultiBinary
from Enviroment.base_env import BaseEnvironment
from Enviroment.graph_layout import ConnectedGraph
from tensordict import TensorDict
import time
import cv2 as cv

MAX_MONEY_LIMIT = 1000

class CustomEnvironment(BaseEnvironment):
    metadata = {"name": "scotland_yard_env"}
    DEFAULT_ACTION = -1

    def __init__(self, number_of_agents, agent_money, reward_weights, logger, epoch, graph_nodes, graph_edges, vis_configs):
        """
        Initialize the environment with given parameters.
        """
        self.reward_weights = reward_weights  # weight of each reward component
        self.number_of_agents = number_of_agents  # Number of police agents
        self.observation_graph = ConnectedGraph(node_space=Discrete(1), edge_space=Discrete(4, start=1))
        self.logger = logger
        self.logger.log(f"Initializing CustomEnvironment with number_of_agents={number_of_agents}, agent_money={agent_money}, reward_weights={reward_weights}", level="debug")
        self.agent_money = agent_money
        self.fig = None
        self.ax = None
        self.pos = None
        self.node_colors = None
        self.node_collection = None
        self.edge_collection = None
        self.label_collection = None  # To store node labels
        #### Heatmap
        self.heatmap = None
        self.heatmap_colors = None
        self.node_visits = None
        self.hm_node_collection = None
        self.hm_edge_collection = None
        self.hm_label_collection = None
        ####
        self.vis_config = vis_configs
        self.visualize = vis_configs["visualize_game"]
        self.visualize_heatmap = vis_configs["visualize_heatmap"]
        self.graph_nodes = graph_nodes
        self.graph_edges = graph_edges
        self.possible_agents = ["MrX"] + [f"Police{n}" for n in range(self.number_of_agents)]
        self.current_winner = None #TODO: write it into the info or sth, that's clunky
        self.G = None
        self.avg_distance = 0
        self.run_images = []
        self.heatmap_images = []
        self.reset()
        self.epoch = epoch
        self.episode = 0


    def reset(self, episode=0, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        self.episode = episode
        self.board = self.observation_graph.sample(num_nodes=self.graph_nodes, num_edges=self.graph_edges)
        self.heatmap = self.board
        self.logger.log("Resetting the environment.", level="debug")
        self.logger.log(f"Generated board with {self.board.nodes.shape[0]} nodes and {self.board.edge_links.shape[0]} edges.", level="debug")

        self.agents = self.possible_agents
        self.node_visits = np.zeros((self.board.nodes.shape[0]),dtype=np.int32)
        agent_starting_positions = list(
            np.random.choice(self.board.nodes.shape[0], size=self.number_of_agents + 1, replace=False)
        )
        self.agents_money = [MAX_MONEY_LIMIT] + [self.agent_money for _ in range(self.number_of_agents)] 

        self.logger.log(f"Agent starting positions: {agent_starting_positions}", level="debug")

        self.MrX_pos = [agent_starting_positions[0]]
        self.police_positions = agent_starting_positions[1:]
        self.timestep = 0
        self.logger.log(f"MrX initial position: {self.MrX_pos[0]}", level="debug")
        self.logger.log(f"Police initial positions: {self.police_positions}", level="debug")
        self.avg_distance = 0
        infos = {a: {} for a in self.agents}  # Dummy infos
        self.logger.log("Environment reset complete.", level="debug")
        self.close_render()
        self.initialize_render()

        observations = self._get_graph_observations()
        return observations, infos

    def step(self, actions):
        """
        Execute actions for all agents and update the environment.
        """
        for pos in self.police_positions:
            self.node_visits[pos] += 1
        self.node_visits[self.MrX_pos[0]] += 1
        self.render()
        self.logger.log(f"Step {self.timestep}: Received actions: {actions}", level="debug")
        mrX_action = actions["MrX"]
        mrx_pos = self.MrX_pos[0]
        self.logger.log(f"MrX current position: {mrx_pos}, action taken: {mrX_action}", level="debug")
        if mrX_action is not None:
            # Process MrX's action
            possible_positions, _ = self._get_possible_moves(mrx_pos,0)
            self.logger.log(f"MrX possible moves from position {mrx_pos}: {possible_positions}", level="debug")
            if mrX_action in possible_positions:
                pos_to_go = mrX_action
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
        is_no_money = True
        for police in actions.keys():
            if police != "MrX":
                police_index = self.agents.index(police) - 1
                
                police_pos = self.police_positions[police_index]
                self.logger.log(f"{police} current position: {police_pos}, action taken: {actions[police]}", level="debug")
                possible_positions, positions_costs = self._get_possible_moves(police_pos, police_index+1)
                self.logger.log(f"{police} possible moves from position {police_pos}: {possible_positions}", level="debug")
                police_action = actions[police]
                #if police_action == self.DEFAULT_ACTION:
                if police_action is None or int(self.agents_money[police_index+1]) == 0 or police_action == self.DEFAULT_ACTION:
                    continue
                is_no_money = False
                if police_action in possible_positions:
                    pos_to_go = police_action
                    self.logger.log(f"{police} moves to position {pos_to_go}", level="debug")
                else:
                    pos_to_go = police_pos  # Stay in the same position if the action is out of bounds
                    #print("POS to go: ",pos_to_go)
                    self.logger.log(f"{police} action out of bounds. Staying at position {pos_to_go}, ",level="debug")

                if pos_to_go not in self.police_positions and pos_to_go != police_pos:
                    self.police_positions[police_index] = pos_to_go
                    log_msg = f"{police} position updated to {self.police_positions[police_index]}, Money: {self.agents_money[police_index+1]} -> "
                    self.agents_money[police_index+1] -= min(positions_costs[np.where(possible_positions == pos_to_go)])
                    log_msg += str(self.agents_money[police_index+1])
                    self.logger.log(log_msg, level="debug")
                else:
                    self.logger.log(f"{police} move blocked by another police at position {pos_to_go}, ",level="debug")

        # Compute rewards and check termination/truncation
        rewards, terminations, truncations, winner = self._calculate_rewards_terminations(is_no_money)
        self.current_winner = winner
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
            self.render()

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
                "Polices_pos" : self.police_positions[1:], # this includes ALL polics pos BOT ONESELF
                "Currency": self.agents_money[1:] # TODO: this includes ALL police money
            }
            for agent in self.agents
        }
        self.logger.log("Graph observations generated., ",level="debug")
        return observations

    def _calculate_rewards_terminations(self, is_no_money):
        """
        Compute rewards and check termination/truncation conditions.
        """
        self.logger.log("Calculating rewards and checking termination conditions., ",level="debug")
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        winner = None
        if self.MrX_pos[0] in self.police_positions:
            self.logger.log("MrX has been caught by the police., ",level="info")
            rewards = {a: (-1 if a == "MrX" else 1) for a in self.agents}
            terminations = {a: True for a in self.agents}
            winner = "Police"
        elif self.timestep > 250:
            self.logger.log("Maximum timestep exceeded. Truncating episode., ",level="info")
            rewards = {a: (1 if a == "MrX" else 0) for a in self.agents}
            truncations = {a: True for a in self.agents}
            winner = "MrX"
        elif is_no_money:
            self.logger.log("Police out of money. Truncating episode., ",level="info")
            rewards = {a: (1 if a == "MrX" else 0) for a in self.agents}
            terminations = {a: True for a in self.agents}
            winner = "MrX"
        else:
            rewards = self.calculate_rewards()

        self.timestep += 1
        self.logger.log(f"Updated timestep to {self.timestep}, ",level="debug")
        return rewards, terminations, truncations, winner
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
        self.avg_distance = np.mean(police_distances)
        self.logger.log(f"MrX closest distance: {closest_distance}, average distance: {self.avg_distance}, ", level="debug")
        position_penalty = len(self._get_possible_moves(mrX_pos,0)[0])
        mrX_reward = (
            self.reward_weights["Mrx_closest"] * (-1 / (closest_distance + 1))  # Distance penalty
            + self.reward_weights["Mrx_average"] * (-1 / (self.avg_distance + 1))  # Average distance penalty
            + self.reward_weights["Mrx_position"] * (position_penalty)  # Position reward
            + (1 - self.reward_weights["Mrx_time"]) * (0.1 * self.timestep)  # Time reward
        )
        rewards["MrX"] = mrX_reward
        self.logger.log(f"MrX reward: {mrX_reward}, ", level="debug")

        # Log MrX reward components as scalars
        distance_penalty_mrX = -1 / (closest_distance + 1)
        avg_distance_penalty_mrX = -1 / (self.avg_distance + 1)
        time_reward_mrX = 0.1 * self.timestep

        self.logger.log_scalar("episode_step", self.timestep)

        self.logger.log_scalar(f'episode/epoch_{self.epoch}/MrX_distance_penalty', distance_penalty_mrX)
        self.logger.log_scalar(f'episode/epoch_{self.epoch}/MrX_avg_distance_penalty', avg_distance_penalty_mrX)
        self.logger.log_scalar(f'episode/epoch_{self.epoch}/MrX_time_reward', time_reward_mrX)
        self.logger.log_scalar(f'episode/epoch_{self.epoch}/MrX_total_reward', mrX_reward)
        self.logger.log_scalar(f'episode/epoch_{self.epoch}/average_distance_to_MrX', self.avg_distance)
        police_dist_sum = 0.0
        for police_pos_1 in self.police_positions:
            for police_pos_2 in self.police_positions:
                police_dist_sum += self.get_distance(police_pos_1, police_pos_2)
        self.logger.log_scalar(f'episode/epoch_{self.epoch}/average_distance_between_officers', police_dist_sum / (len(self.police_positions)**2))
        # Compute rewards for police
        for i, police in enumerate(self.agents[1:]):  # Skip MrX
            police_pos = self.police_positions[i]
            distance_to_mrX = self.get_distance(police_pos, mrX_pos)
            group_penalty = sum(
                np.exp(-self.get_distance(police_pos, other_police_pos))
                for j, other_police_pos in enumerate(self.police_positions)
                if i != j
            )
            position_penalty = len(self._get_possible_moves(police_pos,i)[0])
            self.logger.log(f"{police} distance to MrX: {distance_to_mrX}, group penalty: {group_penalty}, position penalty: {position_penalty}, ", level="debug")

            police_reward = (
                self.reward_weights["Police_distance"] * (np.exp(-distance_to_mrX))  # Distance reward
                + self.reward_weights["Police_group"] * (group_penalty)  # Grouping penalty
                + self.reward_weights["Police_position"] * (position_penalty)  # Position reward
                + (1 - self.reward_weights["Police_time"]) * (0.05 * self.timestep)  # Time penalty
            )
            rewards[police] = police_reward
            self.logger.log(f"{police} reward: {police_reward}, ", level="debug")

            # Log Police reward components as scalars
            distance_reward_police = np.exp(-distance_to_mrX)
            grouping_penalty_police = group_penalty
            position_reward_police = position_penalty
            time_penalty_police = 0.05 * self.timestep

            self.logger.log_scalar(f'episode/epoch_{self.epoch}/{police}_distance_reward', distance_reward_police)
            self.logger.log_scalar(f'episode/epoch_{self.epoch}/{police}_grouping_penalty', grouping_penalty_police)
            self.logger.log_scalar(f'episode/epoch_{self.epoch}/{police}_position_reward', position_reward_police)
            self.logger.log_scalar(f'episode/epoch_{self.epoch}/{police}_time_penalty', time_penalty_police)
            self.logger.log_scalar(f'episode/epoch_{self.epoch}/{police}_total_reward', police_reward)

        self.logger.log(f"All rewards calculated: {rewards}, ", level="debug")
        return rewards


    def get_distance(self, node1: int, node2: int) -> float:
        """
        Compute the shortest path distance between two nodes using Dijkstra's algorithm,
        considering the weights of the edges.

        Args:
            node1 (int): The starting node.
            node2 (int): The target node.

        Returns:
            float: The shortest distance (sum of edge weights) between node1 and node2
                if a path exists. Returns float('inf') if no path exists.
        """
        self.logger.log(f"Calculating weighted distance between node {node1} and node {node2}.", level="debug")

        if node1 == node2:
            self.logger.log("Both nodes are the same. Distance is 0.", level="debug")
            return 0.0

        # Initialize the priority queue with (cumulative_distance, node)
        priority_queue = [(0.0, node1)]
        # Dictionary to keep track of the minimum distance to each node
        distances = {node1: 0.0}
        # Set to keep track of visited nodes
        visited = set()

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            self.logger.log(
                f"Popped node {current_node} with current distance {current_distance} from the priority queue.",
                level="debug"
            )

            if current_node in visited:
                self.logger.log(
                    f"Node {current_node} has already been visited. Skipping.",
                    level="debug"
                )
                continue

            # Mark the current node as visited
            visited.add(current_node)
            self.logger.log(f"Visiting node {current_node}.", level="debug")

            # If we've reached the target node, return the distance
            if current_node == node2:
                self.logger.log(
                    f"Reached target node {node2}. Total distance: {current_distance}.",
                    level="debug"
                )
                return current_distance

            # Find all neighbors of the current node
            # Assuming edge_links[:, 0] is the source and edge_links[:, 1] is the destination
            mask_from = self.board.edge_links[:, 0] == current_node
            mask_to = self.board.edge_links[:, 1] == current_node

            # Extract neighbors and their corresponding edge weights
            neighbors_from = self.board.edge_links[mask_from][:, 1]
            weights_from = self.board.edges[mask_from]
            neighbors_to = self.board.edge_links[mask_to][:, 0]
            weights_to = self.board.edges[mask_to]

            # Combine neighbors and weights
            neighbors = np.concatenate((neighbors_from, neighbors_to))
            weights = np.concatenate((weights_from, weights_to))

            self.logger.log(
                f"Neighbors of node {current_node}: {neighbors} with weights {weights}.",
                level="debug"
            )

            # Iterate through neighbors and update distances
            for neighbor, weight in zip(neighbors, weights):
                if neighbor in visited:
                    self.logger.log(
                        f"Neighbor node {neighbor} has already been visited. Skipping.",
                        level="debug"
                    )
                    continue

                new_distance = current_distance + weight
                self.logger.log(
                    f"Evaluating neighbor {neighbor}: current distance {current_distance} + weight {weight} = {new_distance}.",
                    level="debug"
                )

                # If this path to neighbor is shorter, update the distance and add to the queue
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(priority_queue, (new_distance, neighbor))
                    self.logger.log(
                        f"Updating distance for node {neighbor} to {new_distance} and adding to priority queue.",
                        level="debug"
                    )

        self.logger.log(
            f"No path found between node {node1} and node {node2}. Returning infinity.",
            level="debug"
        )
        return float('inf')  # Return infinity if no path exists within budget

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

    def _get_possible_moves(self, pos, agent_idx):
        """
        Get possible moves from a node position.
        Returns unique neighboring nodes that the agent can afford to move to,
        along with the corresponding edge weights (minimum weight if multiple edges exist).
        
        Args:
            pos (int): Current position of the agent.
            agent_idx (int): Index of the agent in the agents list.
            
        Returns:
            tuple:
                - np.ndarray: Array of unique neighboring node indices that the agent can move to.
                - np.ndarray: Array of corresponding edge weights.
        """
        # Retrieve the agent's available money
        agent_money = self.agents_money[agent_idx]
        
        # Find all edges where the current position is the source
        mask_source = self.board.edge_links[:, 0] == pos
        # Find all edges where the current position is the target
        mask_target = self.board.edge_links[:, 1] == pos
        # Filter edges based on the agent's available money
        affordable_edges_source = mask_source & (self.board.edges <= agent_money)
        affordable_edges_target = mask_target & (self.board.edges <= agent_money)
        # Extract neighboring nodes and corresponding edge weights from affordable edges
        neighbors_from = self.board.edge_links[affordable_edges_source][:, 1]
        weights_from = self.board.edges[affordable_edges_source]
        neighbors_to = self.board.edge_links[affordable_edges_target][:, 0]
        weights_to = self.board.edges[affordable_edges_target]
        # Combine neighbors and weights
        combined_neighbors = np.concatenate([neighbors_from, neighbors_to])
        combined_weights = np.concatenate([weights_from, weights_to])

        # Create a structured array to facilitate finding the minimum weight for each neighbor
        dtype = [('node', combined_neighbors.dtype), ('weight', combined_weights.dtype)]
        structured_array = np.array(list(zip(combined_neighbors, combined_weights)), dtype=dtype)
        
        # Sort by node and then by weight to bring the minimum weight first for each node
        sorted_array = np.sort(structured_array, order=['node', 'weight'])
        
        # Use np.unique to find unique nodes, keeping the first occurrence (minimum weight)
        unique_nodes, indices = np.unique(sorted_array['node'], return_index=True)
        unique_weights = sorted_array['weight'][indices]
        
        self.logger.log(
            f"Agent {self.agents[agent_idx]} (money: {self.agents_money[agent_idx]}) at position {pos} can move to: {unique_nodes} with weights {unique_weights}", 
            level="debug"
        )
        
        return unique_nodes, unique_weights

    
    def get_possible_moves(self, agent_idx):
        if agent_idx == 0:
            pos = self.MrX_pos[0]
        else:
            pos = self.police_positions[agent_idx - 1]
        moves, _ =  self._get_possible_moves(pos, agent_idx)
        return moves
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Define the action space for a given agent based on the number of neighboring nodes
        that the agent can afford to move to.
        
        Args:
            agent (str): Name of the agent.
            
        Returns:
            gymnasium.spaces.Discrete: The action space representing the number of possible moves.
        """
        agent_idx = self.agents.index(agent)
        possible_moves = self.get_possible_moves(agent_idx)
        num_actions = len(possible_moves)
        
        # If there are no possible moves, define a single action (e.g., stay in place)
        if num_actions == 0:
            return Discrete(1)
        else:
            return Discrete(num_actions)



    @functools.lru_cache(maxsize=None) #TODO: this is broken from the beginning??? IT IS
    def observation_space(self, agent):
        """
        Define the observation space for GNN input.
        """
        self.logger.log(f"Defining observation space for agent {agent}., ",level="debug")
        node_features_dim = self.number_of_agents + 1  # MrX + police agents
        adjacency_matrix_shape = (self.board.nodes.shape[0], self.board.nodes.shape[0])
        space = Dict({
            "adjacency_matrix": Box(
                low=0.0, high=1.0, shape= adjacency_matrix_shape, dtype=np.int64
            ),
            "node_features": Box(
                low=0.0, high=1.0, shape=(self.board.nodes.shape[0], node_features_dim), dtype=np.int64
            ),
            "edge_index": Box(
                low=0, high=self.board.nodes.shape[0], shape=(2, self.board.edge_links.shape[0]), dtype=np.int32  
            ),
            "edge_features": Box(
                low=0, high=ConnectedGraph.MAX_WEIGHT, shape=(self.board.edges.shape[0],), dtype=np.int32  
            ),
            "MrX_pos": Discrete(self.board.nodes.shape[0]), 
            "Polices_pos": MultiDiscrete([self.board.nodes.shape[0]] * (self.number_of_agents-1)), 
            "Currency": MultiDiscrete([self.agent_money+1] * (self.number_of_agents-0))  
        })
        self.logger.log(f"Observation space for agent {agent}: {space}, ",level="debug")
        return space

    
    def initialize_render(self, reset=False):
        """
        Initialize the matplotlib plot for rendering the graph.
        """

        self.logger.log("Initializing render plot.", level="debug")
        self.run_images = []
        self.heatmap_images = []
        # Create a NetworkX graph
        graph = self.board
        self.G = nx.Graph()
        num_nodes = graph.nodes.shape[0]
        self.G.add_nodes_from(range(num_nodes))
        edges = [tuple(edge) for edge in graph.edge_links]
        self.G.add_edges_from(edges)
        # Add edge weights if available
        if hasattr(graph, 'edges') and graph.edges is not None:
            for idx, edge in enumerate(graph.edge_links):
                self.G.edges[tuple(edge)]['weight'] = graph.edges[idx]

        # Choose a layout
        self.pos = nx.kamada_kawai_layout(self.G)  # For reproducibility

        # Initialize matplotlib figure and axis
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title(f"Connected Graph Visualization at Timestep {self.timestep}", fontsize=16)

        # Draw nodes
        self.node_colors = ['lightblue'] * self.G.number_of_nodes()  # Default color
        self.heatmap_colors = ['lightblue'] * self.G.number_of_nodes()
        # Highlight MrX and police positions
        self.update_node_colors()

        self.node_collection = nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, node_size=700, node_color=self.node_colors, edgecolors='black')

        # Draw edges
        self.edge_collection = nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, width=2, edge_color='darkgray')

        self.label_collection = nx.draw_networkx_labels(self.G, self.pos, ax=self.ax, font_size=10, font_family="sans-serif",font_color="white")
        # Add legend
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='MrX')
        blue_patch = mpatches.Patch(color='blue', label='Police')
        self.ax.legend(handles=[red_patch, blue_patch], loc='upper right')

        self.ax.axis('off')  # Hide the axes

        # Display the plot
        plt.show()

        # Display the plot
        plt.show()
        self.logger.log("Render plot initialized.", level="debug")

    def update_node_colors(self):
        """
        Update the colors of the nodes based on the positions of MrX and police agents.
        """
        # Reset all colors to default
        self.node_colors = ['gray'] * self.board.nodes.shape[0]
        self.heatmap_colors = []
        # Color MrX's position red
        self.node_colors[self.MrX_pos[0]] = 'red'

        # Color police positions blue
        for pos in self.police_positions:
            self.node_colors[pos] = 'blue'
        visit_max = np.amax([np.amax(self.node_visits),1.0])
        for pos in range(self.node_visits.shape[0]):
            self.heatmap_colors.append(((self.node_visits[pos]/visit_max) * 0.9,0,0))

    def close_render(self):
        """Closes the matplotlib plot."""
        if self.fig is not None:
            plt.ioff()  # Turn off interactive mode
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.node_collection = None
            self.edge_collection = None
            self.hm_node_collection = None
            self.hm_edge_collection = None
            self.logger.log("Render plot closed.", level="debug")
    def save_visualizations(self):
        if self.vis_config["save_visualization"] == True:
            if len(self.run_images) > 0:
                self.logger.log(f"Saving GIF as run_epoch_{self.epoch}-episode_{self.episode+1}.gif")
                f,a = plt.subplots()
                img = a.imshow(self.run_images[0],animated=True)
                def update_gif(i):
                    img.set_array(self.run_images[i])
                    return img,
                animation_fig = animation.FuncAnimation(f,update_gif,frames=len(self.run_images),interval=400,blit=True,repeat_delay=10,)
                plt.show()
                animation_fig.save(self.vis_config["save_dir"]+"/"+f"run_epoch_{self.epoch}-episode_{self.episode+1}.gif")
            if len(self.heatmap_images) > 0:
                self.logger.log(f"Saving GIF as heatmap_epoch_{self.epoch}-episode_{self.episode+1}.gif")
                f,a = plt.subplots()
                img = a.imshow(self.heatmap_images[0],animated=True)
                def update_gif(i):
                    img.set_array(self.heatmap_images[i])
                    return img,
                animation_fig = animation.FuncAnimation(f,update_gif,frames=len(self.heatmap_images),interval=400,blit=True,repeat_delay=10,)
                plt.show()
                animation_fig.save(self.vis_config["save_dir"]+"/"+f"heatmap_epoch_{self.epoch}-episode_{self.episode+1}.gif")
    def render(self):
        """Renders the environment."""
        if self.fig is None or self.ax is None:
            # If render has not been initialized, initialize it
            self.initialize_render()
            return
        if self.visualize or self.visualize_heatmap:
            # Update node colors based on current positions
            self.update_node_colors()
        if self.visualize:
            # Update the node colors in the plot
            self.node_collection.set_color(self.node_colors)

            # Update the title
            self.ax.set_title(f"Episode: {self.episode}, Timestep: {self.timestep}", fontsize=16)
            # Redraw the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            buffer = self.fig.canvas.tostring_argb()
            w,h = self.fig.canvas.get_width_height()
            image = np.frombuffer(buffer,dtype=np.uint8).reshape(h,w,4)[:,:,1:]
            self.run_images.append(image)
            self.logger.log_plt("chart",plt)
        if self.visualize_heatmap:
            self.node_collection.set_color(self.heatmap_colors)
            # Update the title
            self.ax.set_title(f"Heatmap Episode: {self.episode}, Timestep: {self.timestep}", fontsize=16)

            # Redraw the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            buffer = self.fig.canvas.tostring_argb()
            w,h = self.fig.canvas.get_width_height()
            image = np.frombuffer(buffer,dtype=np.uint8).reshape(h,w,4)[:,:,1:]
            self.heatmap_images.append(image)
            self.logger.log_plt("heatmap",plt)

    def get_mrx_position(self):
        """Return the current node index where MrX is located."""
        return self.MrX_pos  # Adapt to your environment's internal representation

    def get_police_position(self, police_idx):
        """Return the current node index where a specific police agent is located."""
        return self.police_positions[police_idx]  # Adapt to your environment's internal representation