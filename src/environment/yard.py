import functools
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict, MultiBinary
from environment.base_env import BaseEnvironment
from environment.graph_layout import ConnectedGraph
from environment.reward_calculator import RewardCalculator
from environment.pathfinding import Pathfinder
from environment.visualization import GameVisualizer
from collections import defaultdict

MAX_MONEY_LIMIT = 1000


class CustomEnvironment(BaseEnvironment):
    metadata = {"name": "scotland_yard_env"}
    DEFAULT_ACTION = -1

    def __init__(
        self,
        number_of_agents,
        agent_money,
        reward_weights,
        logger,
        epoch,
        graph_nodes,
        graph_edges,
        vis_configs,
    ):
        """
        Initialize the environment with given parameters.
        """
        self.reward_weights = reward_weights  # weight of each reward component
        self.number_of_agents = number_of_agents  # Number of police agents
        self.observation_graph = ConnectedGraph(
            node_space=Discrete(1), edge_space=Discrete(4, start=1)
        )
        self.logger = logger
        self.logger.log(
            f"Initializing CustomEnvironment with number_of_agents={number_of_agents}, agent_money={agent_money}, reward_weights={reward_weights}",
            level="debug",
        )
        self.agent_money = agent_money

        # Initialize helper modules
        self.reward_calculator = RewardCalculator(reward_weights, logger)
        self.pathfinder = Pathfinder(logger)
        self.visualizer = GameVisualizer(vis_configs, logger)

        self.vis_config = vis_configs
        self.visualize = vis_configs["visualize_game"]
        self.visualize_heatmap = vis_configs["visualize_heatmap"]
        self.graph_nodes = graph_nodes
        self.graph_edges = graph_edges
        self.possible_agents = ["MrX"] + [
            f"Police{n}" for n in range(self.number_of_agents)
        ]
        self.current_winner = None
        self.avg_distance = 0
        self.node_visit_counts = defaultdict(int)

        # Initialize state tracking
        self.epoch = epoch
        self.episode = 0

        # Generate a reference graph to determine actual achievable edge count
        # This ensures consistent tensor shapes for TorchRL
        reference_board = self.observation_graph.sample(
            num_nodes=self.graph_nodes, num_edges=self.graph_edges
        )
        self.actual_num_edges = reference_board.edge_links.shape[0]
        if self.actual_num_edges != self.graph_edges:
            self.logger.log(
                f"Graph constraint: Using {self.actual_num_edges} edges instead of "
                f"requested {self.graph_edges} due to max_edges_per_node constraint.",
                level="info",
            )

        self.reset()

    def reset(self, episode=0, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        self.episode = episode
        self.node_visit_counts.clear()

        # Generate graphs until we get one with the expected edge count
        # This ensures consistent tensor shapes for TorchRL
        max_attempts = 100
        for attempt in range(max_attempts):
            self.board = self.observation_graph.sample(
                num_nodes=self.graph_nodes, num_edges=self.graph_edges
            )
            if self.board.edge_links.shape[0] == self.actual_num_edges:
                break
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"Failed to generate graph with {self.actual_num_edges} edges after {max_attempts} attempts. "
                    f"Last attempt had {self.board.edge_links.shape[0]} edges. "
                    f"Consider adjusting graph_edges parameter or max_edges_per_node constraint."
                )

        self.heatmap = self.board
        self.logger.log("Resetting the environment.", level="debug")
        self.logger.log(
            f"Generated board with {self.board.nodes.shape[0]} nodes and {self.board.edge_links.shape[0]} edges.",
            level="debug",
        )

        self.agents = self.possible_agents
        self.node_visits = np.zeros((self.board.nodes.shape[0]), dtype=np.int32)
        agent_starting_positions = list(
            np.random.choice(
                self.board.nodes.shape[0], size=self.number_of_agents + 1, replace=False
            )
        )
        self.agents_money = [MAX_MONEY_LIMIT] + [
            self.agent_money for _ in range(self.number_of_agents)
        ]

        self.logger.log(
            f"Agent starting positions: {agent_starting_positions}", level="debug"
        )

        self.MrX_pos = [agent_starting_positions[0]]
        self.police_positions = agent_starting_positions[1:]
        self.timestep = 0
        self.logger.log(f"MrX initial position: {self.MrX_pos[0]}", level="debug")
        self.logger.log(
            f"Police initial positions: {self.police_positions}", level="debug"
        )
        self.avg_distance = 0
        infos = {a: {} for a in self.agents}  # Dummy infos
        self.logger.log("Environment reset complete.", level="debug")

        # Update pathfinder and visualizer with new board state
        self.pathfinder.set_board(self.board)
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
        self.logger.log(
            f"Step {self.timestep}: Received actions: {actions}", level="debug"
        )
        mrX_action = actions["MrX"]
        mrx_pos = self.MrX_pos[0]
        self.logger.log(
            f"MrX current position: {mrx_pos}, action taken: {mrX_action}",
            level="debug",
        )
        if mrX_action is not None:
            # Process MrX's action
            possible_positions, _ = self._get_possible_moves(mrx_pos, 0)
            self.logger.log(
                f"MrX possible moves from position {mrx_pos}: {possible_positions}",
                level="debug",
            )
            if mrX_action in possible_positions:
                pos_to_go = mrX_action
                self.logger.log(f"MrX moves to position {pos_to_go}", level="debug")
            else:
                pos_to_go = (
                    mrx_pos  # Stay in the same position if the action is out of bounds
                )
                self.logger.log(
                    f"MrX action out of bounds. Staying at position {pos_to_go}",
                    level="debug",
                )

            if pos_to_go not in self.police_positions:
                self.MrX_pos = [pos_to_go]
                self.logger.log(
                    f"MrX position updated to {self.MrX_pos[0]}", level="debug"
                )
            else:
                self.logger.log(
                    f"MrX move blocked by police at position {pos_to_go}", level="debug"
                )

        # Process police actions
        is_no_money = True
        for police in actions.keys():
            if police != "MrX":
                police_index = self.agents.index(police) - 1

                police_pos = self.police_positions[police_index]
                self.logger.log(
                    f"{police} current position: {police_pos}, action taken: {actions[police]}",
                    level="debug",
                )
                possible_positions, positions_costs = self._get_possible_moves(
                    police_pos, police_index + 1
                )
                self.logger.log(
                    f"{police} possible moves from position {police_pos}: {possible_positions}",
                    level="debug",
                )
                police_action = actions[police]
                # if police_action == self.DEFAULT_ACTION:
                if (
                    police_action is None
                    or int(self.agents_money[police_index + 1]) == 0
                    or police_action == self.DEFAULT_ACTION
                ):
                    continue
                is_no_money = False
                # TODO: all police blocked
                if police_action in possible_positions:
                    pos_to_go = police_action
                    self.logger.log(
                        f"{police} moves to position {pos_to_go}", level="debug"
                    )
                else:
                    pos_to_go = police_pos  # Stay in the same position if the action is out of bounds
                    # print("POS to go: ",pos_to_go)
                    self.logger.log(
                        f"{police} action out of bounds. Staying at position {pos_to_go}, ",
                        level="debug",
                    )

                if pos_to_go not in self.police_positions and pos_to_go != police_pos:
                    self.police_positions[police_index] = pos_to_go
                    log_msg = f"{police} position updated to {self.police_positions[police_index]}, Money: {self.agents_money[police_index+1]} -> "
                    self.agents_money[police_index + 1] -= min(
                        positions_costs[np.where(possible_positions == pos_to_go)]
                    )
                    log_msg += str(self.agents_money[police_index + 1])
                    self.logger.log(log_msg, level="debug")
                else:
                    self.logger.log(
                        f"{police} move blocked by another police at position {pos_to_go}, ",
                        level="debug",
                    )
        for pos in self.police_positions:
            self.node_visit_counts[pos] += 1
        # Compute rewards and check termination/truncation
        rewards, terminations, truncations, winner = (
            self._calculate_rewards_terminations(is_no_money)
        )
        self.current_winner = winner
        self.logger.log(f"Rewards: {rewards}, ", level="debug")
        self.logger.log(f"Terminations: {terminations}, ", level="debug")
        self.logger.log(f"Truncations: {truncations}, ", level="debug")

        # Get new observations
        observations = self._get_graph_observations()
        infos = {a: {} for a in self.agents}
        self.logger.log("Generated new observations., ", level="debug")

        if any(terminations.values()) or all(truncations.values()):
            self.logger.log(
                "Termination or truncation condition met. Ending episode., ",
                level="debug",
            )
            self.agents = []
            self.render()

        self.logger.log(f"Step {self.timestep} completed., ", level="debug")
        return observations, rewards, terminations, truncations, infos

    def _get_graph_observations(self):
        """
        Create graph-based observations for all agents.
        Includes adjacency matrix, node features, edge features, and action masks.
        """
        self.logger.log("Generating graph observations., ", level="debug")
        adjacency_matrix = self._get_adjacency_matrix()
        edge_weight_matrix = self._get_edge_weight_matrix()
        node_features = np.zeros((self.board.nodes.shape[0], self.number_of_agents + 1))

        # Encode agent positions as node features
        node_features[self.MrX_pos[0], 0] = 1  # MrX position
        self.logger.log(
            f"MrX position encoded in node features: {self.MrX_pos[0]}, ", level="debug"
        )
        for i, pos in enumerate(self.police_positions):
            node_features[pos, i + 1] = 1  # Police positions
            self.logger.log(
                f"Police{i} position encoded in node features: {pos}, ", level="debug"
            )

        edge_index = (
            self.board.edge_links.T
        )  # Edge index for GNN (source, target pairs)
        edge_features = self.board.edges  # Edge weights

        # Compute action masks for all agents
        from environment.action_mask import compute_action_mask

        observations = {}
        for agent in self.agents:
            agent_idx = self.agents.index(agent)
            if agent == "MrX":
                agent_pos = self.MrX_pos[0]
                agent_budget = self.agents_money[0]
            else:
                police_idx = agent_idx - 1
                agent_pos = self.police_positions[police_idx]
                agent_budget = self.agents_money[agent_idx]

            # Compute action mask with fixed index→node mapping
            mask_result = compute_action_mask(
                adjacency=adjacency_matrix,
                current_node=agent_pos,
                budget=agent_budget,
                edge_weights=edge_weight_matrix,
            )

            observations[agent] = {
                "adjacency_matrix": adjacency_matrix,
                "node_features": node_features,
                "edge_index": edge_index,
                "edge_features": edge_features,
                "MrX_pos": self.MrX_pos[0],
                "Polices_pos": self.police_positions[:],  # All police positions
                "Currency": self.agents_money[1:],  # All police money
                "action_mask": mask_result.mask,  # Boolean mask over nodes
                "agent_position": agent_pos,
                "agent_budget": np.array(
                    [agent_budget], dtype=np.float32
                ),  # As array for Box space
            }

        self.logger.log("Graph observations generated., ", level="debug")
        return observations

    def _calculate_rewards_terminations(self, is_no_money):
        """
        Compute rewards and check termination/truncation conditions.
        """
        rewards, terminations, truncations, winner = (
            self.reward_calculator.calculate_rewards_and_terminations(
                mrx_pos=self.MrX_pos[0],
                police_positions=self.police_positions,
                timestep=self.timestep,
                epoch=self.epoch,
                is_no_money=is_no_money,
                agents=self.agents,
                get_distance_func=self.get_distance,
                get_possible_moves_func=self._get_possible_moves,
                node_visit_counts=self.node_visit_counts,
            )
        )

        self.timestep += 1
        self.logger.log(f"Updated timestep to {self.timestep}, ", level="debug")
        return rewards, terminations, truncations, winner

    def calculate_rewards(self):
        """
        Compute rewards for all agents based on the specified components.
        Rewards are weighted by the reward_weights parameters.
        """
        return self.reward_calculator.calculate_rewards(
            mrx_pos=self.MrX_pos[0],
            police_positions=self.police_positions,
            timestep=self.timestep,
            epoch=self.epoch,
            agents=self.agents,
            get_distance_func=self.get_distance,
            get_possible_moves_func=self._get_possible_moves,
            node_visit_counts=self.node_visit_counts,
        )

    def get_distance(self, node1: int, node2: int) -> float:
        """
        Compute the shortest path distance between two nodes using Dijkstra's algorithm,
        considering the weights of the edges.

        Args:
            node1: The starting node
            node2: The target node

        Returns:
            The shortest distance (sum of edge weights) between node1 and node2
            if a path exists. Returns float('inf') if no path exists.
        """
        return self.pathfinder.get_distance(node1, node2)

    def _get_adjacency_matrix(self):
        """
        Generate the adjacency matrix of the graph.
        """
        self.logger.log("Generating adjacency matrix., ", level="debug")
        adjacency_matrix = np.zeros(
            (self.board.nodes.shape[0], self.board.nodes.shape[0])
        )
        for edge in self.board.edge_links:
            adjacency_matrix[edge[0], edge[1]] = 1
            adjacency_matrix[edge[1], edge[0]] = 1  # Undirected graph
        self.logger.log("Adjacency matrix generated., ", level="debug")
        return adjacency_matrix

    def _get_edge_weight_matrix(self):
        """
        Generate the edge weight matrix of the graph.
        Returns a matrix where weight_matrix[i,j] is the cost to traverse from node i to j.
        """
        num_nodes = self.board.nodes.shape[0]
        weight_matrix = np.full((num_nodes, num_nodes), np.inf)
        np.fill_diagonal(weight_matrix, 0)

        for idx, edge in enumerate(self.board.edge_links):
            weight = self.board.edges[idx]
            weight_matrix[edge[0], edge[1]] = weight
            weight_matrix[edge[1], edge[0]] = weight  # Undirected graph

        return weight_matrix

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
        dtype = [("node", combined_neighbors.dtype), ("weight", combined_weights.dtype)]
        structured_array = np.array(
            list(zip(combined_neighbors, combined_weights)), dtype=dtype
        )

        # Sort by node and then by weight to bring the minimum weight first for each node
        sorted_array = np.sort(structured_array, order=["node", "weight"])

        # Use np.unique to find unique nodes, keeping the first occurrence (minimum weight)
        unique_nodes, indices = np.unique(sorted_array["node"], return_index=True)
        unique_weights = sorted_array["weight"][indices]

        self.logger.log(
            f"Agent {self.agents[agent_idx]} (money: {self.agents_money[agent_idx]}) at position {pos} can move to: {unique_nodes} with weights {unique_weights}",
            level="debug",
        )

        return unique_nodes, unique_weights

    def get_possible_moves(self, agent_idx):
        if agent_idx == 0:
            pos = self.MrX_pos[0]
        else:
            pos = self.police_positions[agent_idx - 1]
        moves, _ = self._get_possible_moves(pos, agent_idx)
        return moves

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Define the action space for a given agent.

        The action space is Discrete(num_nodes) where action i corresponds to moving
        to node i. Invalid actions are masked via the action_mask in observations.
        This ensures a FIXED index→node mapping as required by the assignment.

        Args:
            agent (str): Name of the agent.

        Returns:
            gymnasium.spaces.Discrete: The action space representing all nodes.
        """
        num_nodes = self.board.nodes.shape[0]
        return Discrete(num_nodes)

    @functools.lru_cache(
        maxsize=None
    )  # TODO: this is broken from the beginning??? IT IS
    def observation_space(self, agent):
        """
        Define the observation space for GNN input.
        """
        self.logger.log(
            f"Defining observation space for agent {agent}., ", level="debug"
        )
        node_features_dim = self.number_of_agents + 1  # MrX + police agents
        num_nodes = self.board.nodes.shape[0]
        adjacency_matrix_shape = (num_nodes, num_nodes)
        space = Dict(
            {
                "adjacency_matrix": Box(
                    low=0.0, high=1.0, shape=adjacency_matrix_shape, dtype=np.int64
                ),
                "node_features": Box(
                    low=0.0,
                    high=1.0,
                    shape=(num_nodes, node_features_dim),
                    dtype=np.int64,
                ),
                "edge_index": Box(
                    low=0,
                    high=num_nodes,
                    shape=(2, self.actual_num_edges),
                    dtype=np.int32,
                ),
                "edge_features": Box(
                    low=0,
                    high=ConnectedGraph.MAX_WEIGHT,
                    shape=(self.actual_num_edges,),
                    dtype=np.int32,
                ),
                "MrX_pos": Discrete(num_nodes),
                "Polices_pos": MultiDiscrete(
                    [num_nodes] * self.number_of_agents
                ),  # number of police agents
                "Currency": MultiDiscrete(
                    [self.agent_money + 1] * self.number_of_agents
                ),  # number of police agents
                # Action mask fields - fixed index→node mapping
                "action_mask": MultiBinary(num_nodes),  # Boolean mask over nodes
                "agent_position": Discrete(num_nodes),
                "agent_budget": Box(
                    low=0.0, high=MAX_MONEY_LIMIT, shape=(1,), dtype=np.float32
                ),
            }
        )
        self.logger.log(
            f"Observation space for agent {agent}: {space}, ", level="debug"
        )
        return space

    def initialize_render(self, reset=False):
        """
        Initialize the matplotlib plot for rendering the graph.
        """
        self.visualizer.set_game_state(
            board=self.board,
            mrx_pos=self.MrX_pos[0],
            police_positions=self.police_positions,
            node_visits=self.node_visits,
            timestep=self.timestep,
            epoch=self.epoch,
            episode=self.episode,
        )
        self.visualizer.initialize_render(reset=reset)

    def update_node_colors(self):
        """
        Update the colors of the nodes based on the positions of MrX and police agents.
        """
        self.visualizer.update_node_colors()

    def close_render(self):
        """Closes the matplotlib plot."""
        self.visualizer.close_render()

    def save_visualizations(self):
        """Save visualization GIFs if enabled."""
        self.visualizer.save_visualizations()

    def render(self):
        """Renders the environment."""
        # Update visualizer with current state before rendering
        self.visualizer.set_game_state(
            board=self.board,
            mrx_pos=self.MrX_pos[0],
            police_positions=self.police_positions,
            node_visits=self.node_visits,
            timestep=self.timestep,
            epoch=self.epoch,
            episode=self.episode,
        )
        self.visualizer.render()

    def get_mrx_position(self):
        """Return the current node index where MrX is located."""
        return self.MrX_pos  # Adapt to your environment's internal representation

    def get_police_position(self, police_idx):
        """Return the current node index where a specific police agent is located."""
        return self.police_positions[
            police_idx
        ]  # Adapt to your environment's internal representation
