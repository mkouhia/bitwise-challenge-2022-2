"""Test network graphs"""

import pytest

from bitwise_challenge_2022_2.network import BaseNetwork, NetworkGraph


@pytest.fixture(name="simple_graph")
def simple_graph_fx() -> NetworkGraph:
    """Create simple graph for testing"""
    edges = [
        (0, 1, 1),
        (1, 2, 2),
        (1, 3, 4),
        (0, 3, 8),
    ]
    return NetworkGraph(edges)


def test_base_from_json(mocker):
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
    assert net.nodes == {0: (1, 2), 1: (2, 3), 2: (3, 4), 3: (3, 5)}
    assert net.edges == {0: (0, 1), 1: (1, 2), 2: (1, 3), 3: (0, 3)}


def test_base_as_graph(simple_graph: NetworkGraph):
    """Base network is converted to NetworkGraph"""
    nodes = {0: (1, 2), 1: (2, 3), 2: (3, 4), 3: (3, 5)}
    edges = {0: (0, 1), 1: (1, 2), 2: (1, 3), 3: (0, 3)}
    weights = {0: 1, 1: 2, 2: 4, 3: 8}
    net = BaseNetwork(nodes, edges, weights)

    assert net.as_graph().edges == simple_graph.edges


def test_evaluate(simple_graph: NetworkGraph):
    """Evaluation score is as per guidelines"""
    assert simple_graph.evaluate(A=0.1, B=3.0) == 12.0


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
    graph = NetworkGraph(edges)
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
