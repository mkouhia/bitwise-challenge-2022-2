"""Network implementation"""

import json
import locale
import math
import os
from typing import Any

import numpy as np
import numba
from numba import optional
from numba.experimental import jitclass
from numba.types import int8, float64, boolean


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

    def to_adjacency_matrix(self) -> np.ndarray:
        """Create copy of the network

        Rename nodes, so that in resulting network graph, all nodes are
        indexed from 0...(len(g)-1).

        Returns:
            np.ndarray: Two-dimensional matrix of floats, where
              non-edges are represented with np.inf, and diagonal items
              are zeros.
        """
        nodes = list(self.nodes.keys())
        nodes.sort()
        n_len = len(nodes)

        if n_len == 0:
            adj = np.empty((0, 0), float)

        else:
            adj = np.full((n_len, n_len), np.inf, float)

            for edge_id, (id_a, id_b) in self.edges.items():
                adj[id_a, id_b] = self.weights[edge_id]
                adj[id_b, id_a] = self.weights[edge_id]
            np.fill_diagonal(adj, 0)

        return adj

    def get_edge_matrix(self) -> np.ndarray:
        """Get edges as numpy array, Nx2, rows are to-from."""
        return np.array([self.edges[i] for i in range(len(self.edges))])

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
            mat = self.to_adjacency_matrix()
            net = NetworkGraph(mat)
            self._base_score = net.evaluate()
        return self._base_score

    def comparison_score(self, other: "NetworkGraph") -> float:
        """Compare other network to base network, return score"""
        p_other = other.evaluate()
        return (self.base_score / p_other - 1) * 1000


spec = [
    ("adjacency_matrix", float64[:, :]),
    ("_total_weight", optional(float64)),
    ("_score", optional(float64)),
    ("_is_connected", optional(boolean)),
]


@jitclass(spec)
class NetworkGraph:

    """Modified network graph

    Node IDs _must_ be from 0 to N_nodes-1. This is not checked, but
    is up to the user to enforce.

    Attributes:
        edges (list[tuple[int, int, float]]): list of (id_from, id_to, weight)
        n_nodes (int): number of nodes
    """

    def __init__(self, adjacency_matrix: np.ndarray):
        self.adjacency_matrix = adjacency_matrix.copy()
        self._total_weight: float | None = None
        self._score: float | None = None
        self._is_connected: bool | None = None

    def __repr__(self):
        return f"NetworkGraph(adjacency_matrix={self.adjacency_matrix})"

    def remove_edges(self, edges: np.ndarray):
        """Remove edges from graph

        Args:
            edges (np.ndarray): Mx2 array of integers, where M is number
              of removed edges. Each row is (id_from, id_to) pair.
        """
        for row in range(len(edges)):
            self.adjacency_matrix[edges[row, 0], edges[row, 1]] = np.inf
            self.adjacency_matrix[edges[row, 1], edges[row, 0]] = np.inf

    def evaluate(self, A=0.1, B=2.1) -> float:  # pylint: disable=invalid-name
        """Evaluate solution fitness"""
        if self._score is None:
            if not self.is_connected:
                self._score = np.inf
            else:
                self._score = A * self.total_weight + B * self._avg_distance()
        return self._score

    @property
    def is_connected(self):
        """Underlying graph is connected"""
        if self._is_connected is None:
            self._is_connected = self._is_connected_bfs()
        return self._is_connected

    def _is_connected_bfs(self) -> bool:
        """Check connectiviness by performing breadth-first-search

        Returns:
            bool: True if all nodes are reached
        """
        root = 0

        visited = {root}
        queue = [root]

        while queue:
            source = queue.pop(0)

            for i in range(len(self.adjacency_matrix)):
                value = self.adjacency_matrix[source, i]
                if value == 0 or value == np.inf:  # pylint: disable=consider-using-in
                    continue
                if i not in visited:
                    queue.append(i)
                    visited.add(i)

        return len(visited) == len(self.adjacency_matrix)

    @property
    def total_weight(self) -> float:
        """Total weight of all edges in the graph"""
        if self._total_weight is None:
            self._total_weight = 0.0
            for i in range(len(self.adjacency_matrix)):
                for j in range(len(self.adjacency_matrix)):
                    if self.adjacency_matrix[i, j] != np.inf:
                        self._total_weight = (
                            self._total_weight + self.adjacency_matrix[i, j]
                        )
            self._total_weight = self._total_weight / 2
        return self._total_weight

    def _avg_distance(self) -> float:
        """Average distance of each point to every other point"""
        node_count = len(self.adjacency_matrix)
        mat = floyd_distance(self.adjacency_matrix, node_count)

        return mat.sum() / (node_count * (node_count - 1))


@numba.njit(fastmath=True)
def floyd_distance(matrix: np.ndarray, num_nodes: int) -> np.ndarray:
    """Calculate Floyd-Warshall distance matrix"""
    # pylint: disable=not-an-iterable
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if matrix[i, j] > matrix[i, k] + matrix[k, j]:
                    matrix[i, j] = matrix[i, k] + matrix[k, j]
    return matrix


@numba.njit(parallel=True)
def evaluate_many(
    base_matrix: np.ndarray,
    edge_options: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Evaluate multiple network options in parallel

    Args:
        base_matrix (np.ndarray): adjacency matrix for the base network
        edge_options (np.ndarray): Matrix of NxM, where N is number
          of parallel options and M is the number of edges in the base
          matrix. Values are True/False, depending on whether the edge
          is removed or not.
        edges (np.ndarray): Matrix of Nx2, where rows represent edges
          as (start node, end node).

    Returns:
        np.ndarray: Nx2 matrix, containing objective function values
          in column 0 and constraint values in column 1.
    """

    result = np.empty((len(edge_options), 2))

    for i in numba.prange(len(edge_options)):  # pylint: disable=not-an-iterable

        row_x_binary = edge_options[i]
        remove_edges = edges[~row_x_binary]

        new_net = NetworkGraph(base_matrix)
        new_net.remove_edges(remove_edges)

        result[i, 0] = new_net.evaluate()
        result[i, 1] = -1 if new_net.is_connected else 1

    return result


@numba.njit(parallel=True)
def get_comparison_mat(edge_options: np.ndarray):
    result = np.empty(len(edge_options))
    for i in numba.prange(len(edge_options)):  # pylint: disable=not-an-iterable
        result[i] = base_2_to_10_array(edge_options[i].astype("int"))
    return result


@numba.njit
def split_chunks_fill(arr, size=8):
    i0 = 0
    i1 = size
    while i1 <= len(arr):
        yield arr[i0:i1]
        i0 += size
        i1 += size
    yield arr[i0:]


@numba.njit
def base_2_to_10_array(arr):
    """Convert base2 array to base-10

    This might overflow, but does not matter. It is used for hashing.
    """
    res = 0
    for bit in arr:
        res = (res << 1) ^ bit
    return res
