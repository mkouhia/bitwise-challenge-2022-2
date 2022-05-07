"""Test network graphs"""

from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_array_equal
import networkx as nx

from bitwise_challenge_2022_2.network import BaseNetwork, NetworkGraph


@pytest.fixture(name="base_network")
def base_network_fx() -> BaseNetwork:
    """Create base network for testing"""
    nodes = {0: (1, 2), 1: (2, 3), 2: (3, 4), 3: (3, 5)}
    edges = {0: (0, 1), 1: (1, 2), 2: (1, 3), 3: (0, 3)}
    weights = {0: 1, 1: 2, 2: 4, 3: 8}
    return BaseNetwork(nodes, edges, weights)


@pytest.fixture(name="challenge_base")
def challenge_base_fx() -> BaseNetwork:
    """Actual challenge network"""
    network_json = (
        Path(__file__).parents[1]
        / "bitwise_challenge_2022_2"
        / "koodipahkina-data.json"
    )
    return BaseNetwork.from_json(network_json)


def edges_to_adjacency_matrix(
    edges: list[tuple[int, int, float]], n_len: int
) -> np.ndarray:
    """Convert edge list to adjacency matrix"""
    adj = np.full((n_len, n_len), np.inf, float)

    for (id_a, id_b, weight) in edges:
        adj[id_a, id_b] = weight
        adj[id_b, id_a] = weight
    np.fill_diagonal(adj, 0)

    return adj


@pytest.fixture(name="simple_graph")
def simple_graph_fx() -> NetworkGraph:
    """Create simple graph for testing"""
    edges = [
        (0, 1, 1),
        (1, 2, 2),
        (1, 3, 4),
        (0, 3, 8),
    ]
    adj = edges_to_adjacency_matrix(edges, 4)
    return NetworkGraph(adj)


def test_base_from_json(mocker, base_network):
    """Test json reading"""
    test_json = """{
    "points": {
        "0": {"x": 1, "y": 2, "edges": {"0": 1, "3": 3}},
        "1": {"x": 2, "y": 3, "edges": {"0": 0, "1": 2, "2": 3}},
        "2": {"x": 3, "y": 4, "edges": {"1": 1}},
        "3": {"x": 3, "y": 5, "edges": {"2": 3, "3": 0}}
    }}"""
    mocker.patch("builtins.open", mocker.mock_open(read_data=test_json))

    net = BaseNetwork.from_json(test_json)
    assert net.nodes == base_network.nodes
    assert net.edges == base_network.edges


def test_base_to_adjacency_matrix(
    base_network: BaseNetwork, simple_graph: NetworkGraph
):
    """Base network is converted to NetworkGraph"""
    assert_array_equal(
        base_network.to_adjacency_matrix(), simple_graph.adjacency_matrix
    )


def test_get_edge_matrix(base_network: BaseNetwork):
    """Edges are ordered in matrix correctly"""
    expected = np.array([(0, 1), (1, 2), (1, 3), (0, 3)])
    assert_array_equal(base_network.get_edge_matrix(), expected)


def test_get_weight_vector(base_network: BaseNetwork):
    """Proper weights are returned"""
    expected = np.array([1, 2, 4, 8])
    assert_array_equal(base_network.get_weight_vector(), expected)


def test_challenge_edge_ids(challenge_base: BaseNetwork):
    """Challenge edge ids are from 0 to N-1"""
    assert challenge_base.edges.keys() == set(range(len(challenge_base.edges)))


def test_challenge_node_ids(challenge_base: BaseNetwork):
    """Challenge node ids are from 0 to N-1"""
    assert challenge_base.nodes.keys() == set(range(len(challenge_base.nodes)))


def test_remove_edges(simple_graph: NetworkGraph):
    """Edges are removed"""
    arr = np.array([[0, 1], [0, 3]])
    simple_graph.remove_edges(arr)

    expected = np.array(
        [
            [0.0, np.inf, np.inf, np.inf],
            [np.inf, 0.0, 2.0, 4.0],
            [np.inf, 2.0, 0.0, np.inf],
            [np.inf, 4.0, np.inf, 0.0],
        ]
    )
    assert_array_equal(simple_graph.adjacency_matrix, expected)


def test_evaluate(simple_graph: NetworkGraph):
    """Evaluation score is as per guidelines"""
    assert simple_graph.evaluate(0.1, 3.0) == 12.0


def test_is_connected(simple_graph: NetworkGraph):
    """All nodes are connected to each other"""
    assert simple_graph.is_connected


def test_is_not_connected():
    """Unconnected graph"""
    edges = [
        (0, 1, 1),
        (1, 2, 2),
        (3, 4, 4),
        (4, 5, 8),
    ]
    adj = edges_to_adjacency_matrix(edges, 6)
    graph = NetworkGraph(adj)
    assert not graph.is_connected


def test_repair_with_edges():
    """When repaired, graph is connected"""
    edges = np.array([[0, 1], [0, 3]])
    weights = np.array([1.0, 8.0])

    adj = np.array(
        [
            [0.0, np.inf, np.inf, np.inf],
            [np.inf, 0.0, 2.0, 4.0],
            [np.inf, 2.0, 0.0, np.inf],
            [np.inf, 4.0, np.inf, 0.0],
        ]
    )
    expected = adj.copy()
    expected[0, 1] = 1.0
    expected[1, 0] = 1.0

    net = NetworkGraph(adj)
    net.repair_with_edges(edges, weights)
    assert_array_equal(net.adjacency_matrix, expected)
    assert net._is_connected_bfs(True, 0)  # pylint: disable=protected-access


def test_total_weight(simple_graph: NetworkGraph):
    """Total weight is the sum of all weights"""
    assert simple_graph.total_weight == 15


def test_avg_distance(simple_graph: NetworkGraph):
    """Average distance is calculated correctly"""
    distances = [
        1,  # 0, 1
        3,  # 0, 2
        5,  # 0, 3
        2,  # 1, 2
        4,  # 1, 3
        6,  # 2, 3
    ]
    # pylint: disable=protected-access
    assert simple_graph._avg_distance() == sum(distances) / len(distances)


@pytest.fixture(name="networkx_base")
def networkx_base_fx(challenge_base: BaseNetwork) -> nx.Graph:
    """Create networkx graph corresponding to base network"""
    graph = nx.Graph()
    edges = [
        (u, v, challenge_base.weights[id])
        for id, (u, v) in challenge_base.edges.items()
    ]
    graph.add_weighted_edges_from(edges)
    graph.add_nodes_from(challenge_base.nodes.keys())
    return graph


def test_compare_networkx_base(challenge_base: BaseNetwork, networkx_base: nx.Graph):
    """Compare base network avg shortest path length to networkx"""
    mat = challenge_base.to_adjacency_matrix()
    net = NetworkGraph(mat)

    expected = nx.average_shortest_path_length(
        networkx_base, weight="weight", method="floyd-warshall-numpy"
    )
    received = net._avg_distance()  # pylint: disable=protected-access

    assert received == pytest.approx(expected)


def test_compare_networkx_modified(
    challenge_base: BaseNetwork, networkx_base: nx.Graph
):
    """Compare modified network to netorkx"""
    # fmt: off
    edge_ids = [0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 15, 16, 17, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 38, 39, 40, 43, 44, 45, 47, 49, 52, 53, 55, 57, 60, 62, 63, 65, 66, 67, 68, 70, 71, 72, 74, 76, 78, 79, 80, 81, 82, 85, 86, 88, 89, 90, 91, 92, 94, 95, 96, 98, 100, 101, 102, 103, 104, 105, 111, 112, 113, 114, 115, 116, 117, 120, 122, 123, 124, 125, 126, 128, 130, 132, 133, 135, 137, 138, 139, 140, 141, 142, 144, 146, 148, 149, 151, 153, 154, 155, 157, 159, 160, 161, 163, 164, 167, 168, 169, 170, 172, 174, 177, 178, 179, 180, 181, 183, 184, 185, 186, 187, 189, 191, 192, 193, 194, 195, 196, 198, 201, 203, 204, 205, 207, 209, 212, 214, 218, 219, 220, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 233, 234, 235, 237, 238, 241, 242, 243, 244, 247, 248, 249, 250, 252, 255, 257, 258, 259, 260, 262, 263, 264, 266, 267, 268, 269, 270, 271, 274, 275, 276, 277, 279, 284, 286, 287, 289, 290, 291, 295, 296, 297, 298, 299, 301, 302, 303, 304, 306, 307, 309, 311, 312, 313, 314, 317, 318, 319, 320, 322, 325, 326, 329, 334, 335, 337, 339, 340, 341, 345, 347, 350, 354, 356, 359, 360, 361, 363, 365, 366, 369, 373, 374, 375, 378, 380, 382, 388, 389, 390, 391, 395, 396, 397, 399, 401, 402, 403, 404, 406, 407, 410, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431]
    # fmt: on

    mat = challenge_base.to_adjacency_matrix()
    net = NetworkGraph(mat)
    all_edges = challenge_base.get_edge_matrix()
    remove_edges = all_edges[edge_ids]

    net.remove_edges(remove_edges)
    networkx_base.remove_edges_from(remove_edges)

    assert net.is_connected
    assert nx.is_connected(networkx_base)

    expected = nx.average_shortest_path_length(
        networkx_base, weight="weight", method="floyd-warshall-numpy"
    )
    received = net._avg_distance()  # pylint: disable=protected-access

    assert received == pytest.approx(expected)
