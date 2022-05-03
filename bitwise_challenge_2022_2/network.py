"""Network implementation"""

from collections import defaultdict, deque
import json
import locale
import math
import os
import sys
from typing import Any, Iterable

import networkx as nx


class BaseNetwork:

    """Base network representation

    Attributes:
        nodes: dictionary of node id to (x, y) coordinates
        edges: dictionary of edge id to tuple(point id, another point id)
        weights: list of tuples (node_from, node_to, weight)
    """

    def __init__(
        self,
        nodes: dict[int, tuple[int, int]],
        edges: dict[int, tuple[int, int]],
        weights: Any | None = None,
    ):
        self.nodes = nodes
        self.edges = edges
        self._weights = weights

        self._base_score: float | None = None

    @classmethod
    def from_json(
        cls, fname: os.PathLike, encoding=locale.getpreferredencoding()
    ) -> "BaseNetwork":
        """Read initial structure of problem

        Args:
            fname (os.PathLike): location of structure json

        Returns:
            OriginalNetwork: _description_
        """
        with open(fname, "r", encoding=encoding) as file:
            j_struct = json.load(file)["points"]

        nodes: dict[int, tuple(int, int)] = {}
        edges: dict[int, tuple[int, int]] = {}

        for point_s in j_struct:
            point_id = int(point_s)

            nodes[point_id] = tuple(j_struct[point_s][i] for i in ["x", "y"])

            for edge_s, to_id in j_struct[point_s]["edges"].items():
                edge_id = int(edge_s)
                if edge_id in edges:
                    continue
                edges[edge_id] = (point_id, to_id)

        return BaseNetwork(nodes, edges)

    def as_graph(self, remove_edges: Iterable[int] | None = None) -> "NetworkGraph":
        """Create copy of the network

        Args:
            remove_edges (Iterable[int] | None): edge ids to be removed

        Returns:
            NetworkGraph: Graph representation of the network, possibly
              without specified edges.
        """
        edge_list = [
            (id_a, id_b, self.weights[edge_id])
            for edge_id, (id_a, id_b) in self.edges.items()
            if remove_edges is None or edge_id not in remove_edges
        ]

        return NetworkGraph(edge_list)

    @property
    def weights(self):
        """Edge weights, edge_id: weight"""
        if self._weights is not None:
            return self._weights

        weigths = {}
        for edge_id, (id_a, id_b) in self.edges.items():
            x_a, y_a = self.nodes[id_a]
            x_b, y_b = self.nodes[id_b]

            weight = math.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)

            weigths[edge_id] = weight

        self._weights = weigths
        return self._weights

    @property
    def base_score(self):
        """Base score for the network"""
        if self._base_score is None:
            net = self.as_graph()
            self._base_score = net.evaluate()
        return self._base_score

    def comparison_score(self, other: "NetworkGraph") -> float:
        """Compare other network to base network, return score"""
        p_other = other.evaluate()
        return (self.base_score / p_other - 1) * 1000


class NetworkGraph:

    """Modified network graph

    Attributes:
        edges (list[tuple[int, int, float]]): list of (id_from, id_to, weight)
    """

    def __init__(self, edges: list[tuple[int, int, float]]):
        self.edges = edges
        self._total_weight: float | None = None
        self._score: float | None = None
        self._is_connected: bool | None = None

    def __repr__(self):
        return f"NetworkGraph(edges={self.edges})"

    def _edge_index(self, id_a: int, id_b: int) -> int:
        """Get index of edge in self.edges by start and end

        Args:
            id_a (int): start node id
            id_b (int): end node id

        Raises:
            ValueError: if edge is not found

        Returns:
            int: index in self.edge
        """
        for i, item in enumerate(self.edges):
            if item[:2] == (id_a, id_b):
                return i
        raise ValueError(f"Edge {id_a}, {id_b} not found in edges")

    def evaluate(self, A=0.1, B=2.1) -> float:  # pylint: disable=invalid-name
        """Evaluate solution fitness"""
        if self._score is None:
            if not self.is_connected:
                self._score = sys.float_info.max
            else:
                self._score = A * self.total_weight + B * self._avg_distance()
        return self._score

    @property
    def is_connected(self):
        """Underlying graph is connected"""
        if self._is_connected is None:
            self._is_connected = self._is_connected_bfs()
        return self._is_connected

    def _build_adjacency_list(self) -> dict[list]:
        ad_list = defaultdict(list)
        for (id_a, id_b, _) in self.edges:
            ad_list[id_a].append(id_b)
            ad_list[id_b].append(id_a)
        return ad_list

    def _is_connected_bfs(self) -> bool:
        """Check connectiviness by performing BFS

        Returns:
            bool: True if all nodes are reached
        """
        graph = self._build_adjacency_list()
        root = graph[next(iter(graph))][0]

        visited = {root}
        queue = deque([root])

        while queue:
            source = queue.popleft()

            for i in graph[source]:
                if i not in visited:
                    queue.append(i)
                    visited.add(i)

        return len(visited) == len(graph)

    @property
    def total_weight(self) -> float:
        """Total weight of all edges in the graph"""
        if self._total_weight is None:
            self._total_weight = sum(k for _, _, k in self.edges)
        return self._total_weight

    def _avg_distance(self) -> float:
        """Average distance of each point to every other point"""
        graph = nx.Graph()
        graph.add_weighted_edges_from(self.edges)
        return nx.average_shortest_path_length(
            graph, weight="weight", method="floyd-warshall-numpy"
        )
