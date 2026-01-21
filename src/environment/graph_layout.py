import random
import warnings
from gymnasium.spaces import Graph
import numpy as np


class ConnectedGraph(Graph):
    MAX_WEIGHT = 5

    def sample(self, mask=None, num_nodes=10, num_edges=None, max_edges_per_node=4):
        # Remove all initial edges
        graph_json = {"nodes": list(range(num_nodes)), "edge_links": [], "edges": []}
        # graph = self.from_jsonable([graph_json])[0]

        # Create a connected tree using Kruskal's or Prim's algorithm
        edges = self._create_tree(num_nodes)
        graph_json["edge_links"] = edges
        graph_json["edges"] = [np.random.randint(1, self.MAX_WEIGHT) for _ in edges]

        # Add additional edges randomly
        if num_edges is None:
            num_edges = num_nodes - 1  # Minimal case for a connected graph

        extra_edges = num_edges - len(edges)
        if extra_edges > 0:
            possible_edges = [
                (i, j)
                for i in range(num_nodes)
                for j in range(i + 1, num_nodes)
                if (i, j) not in edges and (j, i) not in edges
            ]
            random.shuffle(possible_edges)

            cnt = 1
            for edge in possible_edges:
                cnt += 1
                if extra_edges <= 0:
                    break

                # Check the max_edges_per_node constraint
                if (
                    sum([1 for e in graph_json["edge_links"] if edge[0] in e])
                    < max_edges_per_node
                    and sum([1 for e in graph_json["edge_links"] if edge[1] in e])
                    < max_edges_per_node
                ):
                    graph_json["edge_links"].append(edge)
                    graph_json["edges"].append(np.random.randint(1, self.MAX_WEIGHT))
                    extra_edges -= 1

        # Note: Edge count may differ from requested due to max_edges_per_node constraint.
        # This is handled by the environment layer, so we don't warn here to avoid spam.

        return self.from_jsonable([graph_json])[0]

    def _create_tree(self, num_nodes):
        """
        Creates a connected tree using a randomized Prim's algorithm.
        """
        edges = []
        visited = set()
        unvisited = set(range(num_nodes))

        # Start from a random node
        current_node = random.choice(list(unvisited))
        visited.add(current_node)
        unvisited.remove(current_node)

        while unvisited:
            # Find all potential edges from visited to unvisited nodes
            possible_edges = [
                (node, neighbor) for node in visited for neighbor in unvisited
            ]
            # Randomly select an edge
            edge = random.choice(possible_edges)
            edges.append(edge)
            # Mark the neighbor as visited
            visited.add(edge[1])
            unvisited.remove(edge[1])

        return edges
