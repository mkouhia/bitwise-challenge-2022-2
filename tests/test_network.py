"""Test network graphs"""

from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_array_equal

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
