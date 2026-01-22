"""
Pathfinding module for Scotland Yard environment.

This module implements Dijkstra's algorithm for computing weighted shortest paths
between nodes in the game graph.
"""

import heapq
import numpy as np
from typing import Tuple


class Pathfinder:
    """Handles pathfinding operations for the Scotland Yard game graph."""

    def __init__(self, logger):
        """
        Initialize the pathfinder.

        Args:
            logger: Logger instance for debugging
        """
        self.logger = logger
        self.board = None

    def set_board(self, board):
        """
        Set the current game board (graph).

        Args:
            board: Graph object with nodes, edges, and edge_links
        """
        self.board = board

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
            self.logger.log(
                f"Popped node {current_node} with current distance {current_distance} from the priority queue.",
                level="debug",
            )

            if current_node in visited:
                self.logger.log(
                    f"Node {current_node} has already been visited. Skipping.",
                    level="debug",
                )
                continue

            # Mark the current node as visited
            visited.add(current_node)
            self.logger.log(f"Visiting node {current_node}.", level="debug")

            # If we've reached the target node, return the distance
            if current_node == node2:
                self.logger.log(
                    f"Reached target node {node2}. Total distance: {current_distance}.",
                    level="debug",
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
                level="debug",
            )

            # Iterate through neighbors and update distances
            for neighbor, weight in zip(neighbors, weights):
                if neighbor in visited:
                    self.logger.log(
                        f"Neighbor node {neighbor} has already been visited. Skipping.",
                        level="debug",
                    )
                    continue

                new_distance = current_distance + weight
                self.logger.log(
                    f"Evaluating neighbor {neighbor}: current distance {current_distance} + weight {weight} = {new_distance}.",
                    level="debug",
                )

                # If this path to neighbor is shorter, update the distance and add to the queue
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(priority_queue, (new_distance, neighbor))
                    self.logger.log(
                        f"Updating distance for node {neighbor} to {new_distance} and adding to priority queue.",
                        level="debug",
                    )

        self.logger.log(
            f"No path found between node {node1} and node {node2}. Returning infinity.",
            level="debug",
        )
        return float("inf")  # Return infinity if no path exists
