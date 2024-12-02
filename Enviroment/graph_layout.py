from gymnasium.spaces import Graph, Box, Discrete
import numpy as np

class ConnectedGraph(Graph):

    def sample(self, mask = None, num_nodes = 10, num_edges = None):
        graph = super().sample( mask, num_nodes, num_edges )
        is_conncected, correction = self._is_connected(graph)
        while not is_conncected:
            graph_json = self.to_jsonable([graph])[0]
            graph_json["edge_links"].append(correction)
            graph_json["edges"].append(np.random.randint(1, 4))
            graph = self.from_jsonable([graph_json])[0]
            is_conncected, correction = self._is_connected(graph)
        return graph

    def _is_connected(self, graph):
        if not graph or graph.edge_links is None:
            return False

        visited = np.array([False] * graph.nodes.shape[0])
        queue = [0]  # Start BFS from node 0
        visited[0] = True

        while queue:
            node = queue.pop(0)
            neighbors = graph.edge_links[graph.edge_links[:,0] == node][:,1]
            visited[neighbors] = True
            queue.extend(list(neighbors[~visited[neighbors]]))
            neighbors = graph.edge_links[graph.edge_links[:,1] == node][:,0]
            visited[neighbors] = True
            queue.extend(list(neighbors[~visited[neighbors]]))

        if np.all(visited):
            return True, None
        else:
            # return False, [np.random.choice(np.where(visited == True)[0]), np.random.choice(np.where(visited == False)[0])]



            return False, [np.argmax(visited), np.argmin(visited)]