"""
Pathfinding module for Scotland Yard environment.

This module precomputes all-pairs shortest paths using scipy for efficient
distance lookups during reward computation and agent decision-making.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class Pathfinder:
    """Handles pathfinding operations for the Scotland Yard game graph.

    Uses precomputed all-pairs shortest paths for O(1) distance lookups
    instead of per-call Dijkstra, in case the graph is static.
    """

    def __init__(self, logger):
        """
        Initialize the pathfinder.

        Args:
            logger: Logger instance for debugging
        """
        self.logger = logger
        self.board = None
        self._distance_matrix = None

    def set_board(self, board):
        """
        Set the current game board (graph) and precompute all-pairs shortest paths.

        Args:
            board: Graph object with nodes, edges, and edge_links
        """
        self.board = board
        n = board.nodes.shape[0]

        # Build sparse weighted adjacency matrix
        row = board.edge_links[:, 0]
        col = board.edge_links[:, 1]
        weights = board.edges.astype(np.float64)

        # Construct undirected graph (symmetric matrix)
        A = csr_matrix((weights, (row, col)), shape=(n, n))
        A = A + A.T
        # For duplicate edges, keep the minimum weight
        A.data = np.minimum(A.data, A.data)

        # Compute all-pairs shortest paths (Dijkstra, C implementation)
        self._distance_matrix = shortest_path(A, directed=False)

        self.logger.log(
            f"Precomputed all-pairs shortest paths for {n}-node graph.",
            level="debug",
        )

    def get_distance(self, node1: int, node2: int, dynamic: bool = False) -> float:
        """
        Compute the shortest path distance between two nodes.

        Dispatches to the precomputed matrix (fast, O(1)) by default,
        or to the on-the-fly Dijkstra implementation when ``dynamic=True``.

        Use ``dynamic=True`` for graphs whose edge weights change between
        steps (e.g. toll roads, dynamic environments). For static graphs
        the precomputed mode is always preferred.

        Args:
            node1: The starting node.
            node2: The target node.
            dynamic: If True, compute on-the-fly with Dijkstra instead of
                using the precomputed distance matrix. Defaults to False.

        Returns:
            The shortest distance (sum of edge weights) between node1 and
            node2. Returns float('inf') if no path exists.
        """
        if dynamic:
            return self.get_distance_dijkstra(node1, node2)
        return self.get_distance_precomputed(node1, node2)

    def get_distance_precomputed(self, node1: int, node2: int) -> float:
        """
        Look up the shortest path distance from the precomputed all-pairs matrix.

        Requires ``set_board()`` to have been called first.

        Args:
            node1: The starting node.
            node2: The target node.

        Returns:
            The shortest distance (sum of edge weights) between node1 and node2.
            Returns float('inf') if no path exists.
        """
        if self._distance_matrix is None:
            raise RuntimeError(
                "Pathfinder.set_board() must be called before get_distance()"
            )
        return float(self._distance_matrix[node1, node2])

    def get_distance_dijkstra(self, node1: int, node2: int) -> float:
        """
        Compute the shortest path distance on-the-fly using Dijkstra's algorithm.

        Suitable for dynamic graphs where edge weights change between steps.
        Slower than the precomputed variant (O(E log V) per call) but always
        reflects the current board state.

        Args:
            node1: The starting node.
            node2: The target node.

        Returns:
            The shortest distance (sum of edge weights) between node1 and node2
            if a path exists. Returns float('inf') if no path exists.
        """
        import heapq

        self.logger.log(
            f"Calculating weighted distance between node {node1} and node {node2}.",
            level="debug",
        )

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

            if current_node in visited:
                continue

            visited.add(current_node)

            if current_node == node2:
                self.logger.log(
                    f"Reached target node {node2}. "
                    f"Total distance: {current_distance}.",
                    level="debug",
                )
                return current_distance

            # Find all neighbours of the current node (undirected)
            mask_from = self.board.edge_links[:, 0] == current_node
            mask_to = self.board.edge_links[:, 1] == current_node

            neighbors = np.concatenate(
                (
                    self.board.edge_links[mask_from][:, 1],
                    self.board.edge_links[mask_to][:, 0],
                )
            )
            weights = np.concatenate(
                (
                    self.board.edges[mask_from],
                    self.board.edges[mask_to],
                )
            )

            for neighbor, weight in zip(neighbors, weights):
                if neighbor in visited:
                    continue
                new_distance = current_distance + weight
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(priority_queue, (new_distance, neighbor))

        self.logger.log(
            f"No path found between node {node1} and node {node2}. "
            "Returning infinity.",
            level="debug",
        )
        return float("inf")
