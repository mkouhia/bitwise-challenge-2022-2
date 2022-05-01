"""Network implementation"""

import json
import locale
import math
import os
import sys
from typing import Any

import networkx as nx


class BaseNetwork:

    """Base network representation

    Attributes:
        nodes: dictionary of node id to (x, y) coordinates
        edges: dictionary of edge id to tuple(point id, another point id)
    """

    def __init__(
        self, nodes: dict[int, tuple[int, int]], edges: dict[int, tuple[int, int]]
    ):
        self.nodes = nodes
        self.edges = edges

        self._network: "NetworkGraph" | None = None

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
                edges[edge_id] = (point_id, to_id)

        return BaseNetwork(nodes, edges)

    def as_graph(self, remove_edges: list[int] | None = None) -> "NetworkGraph":
        """Create copy of the network

        Args:
            remove_edges (list[int] | None): list of edge ids to be removed

        Returns:
            NetworkGraph: Graph representation of the network, possibly
              without specified edges.
        """
        if self._network is None:
            self._network = NetworkGraph(self._init_graph())

        if remove_edges is None or len(remove_edges) == 0:
            return self._network

        new_network = self._network.copy()

        for edge_id in remove_edges:
            id_a, id_b = self.edges[edge_id]
            new_network.remove_edge(id_a, id_b)

        return new_network

    def _init_graph(self) -> nx.Graph:
        """Initialize internal graph representation"""
        graph = nx.Graph()

        edge_list = []
        for (id_a, id_b) in self.edges.values():
            x_a, y_a = self.nodes[id_a]
            x_b, y_b = self.nodes[id_b]

            weight = math.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)

            edge_list.append((id_a, id_b, weight))

        graph.add_weighted_edges_from(edge_list)

        return graph

    def comparison_score(self, other: "NetworkGraph") -> float:
        """Compare other network to base network, return score"""
        p_this = self.as_graph().evaluate()
        p_other = other.evaluate()
        return (p_this / p_other - 1) * 1000


class NetworkGraph:

    """Modified network graph"""

    def __init__(self, graph: nx.Graph):
        self._graph = graph
        self._total_weight: float | None = None
        self._score: float | None = None

    def __repr__(self):
        return f"NetworkGraph(graph={repr(self._graph)})"

    def copy(self) -> "NetworkGraph":
        """Create a copy of underlying graph, set total weight"""
        new_network = NetworkGraph(self._graph.copy())

        # pylint: disable=protected-access
        new_network._total_weight = self.total_weight

        return new_network

    def remove_edge(self, id_a: Any, id_b: Any):
        """Remove edge from underlying graph

        Args:
            id_a (Any): start node id
            id_b (Any): end node id
        """
        edge_weight = self._graph.edges[id_a, id_b]["weight"]

        self._graph.remove_edge(id_a, id_b)
        self._total_weight -= edge_weight
        self._score = None

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
        return nx.is_connected(self._graph)

    @property
    def total_weight(self) -> float:
        """Total weight of all edges in the graph"""
        if self._total_weight is None:
            self._total_weight = sum(k for _, _, k in self._graph.edges.data("weight"))
        return self._total_weight

    def _avg_distance(self) -> float:
        """Average distance of each point to every other point"""
        return nx.average_shortest_path_length(
            self._graph, weight="weight", method="floyd-warshall-numpy"
        )
