"""Test network graphs"""

from pathlib import Path
import pytest

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


@pytest.fixture(name="simple_graph")
def simple_graph_fx() -> NetworkGraph:
    """Create simple graph for testing"""
    edges = [
        (0, 1, 1),
        (1, 2, 2),
        (1, 3, 4),
        (0, 3, 8),
    ]
    return NetworkGraph(edges, 4)


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


def test_base_as_graph(base_network: BaseNetwork, simple_graph: NetworkGraph):
    """Base network is converted to NetworkGraph"""
    assert base_network.as_graph().edges == simple_graph.edges


def test_base_as_graph_del(base_network: BaseNetwork):
    """Base network is converted to NetworkGraph"""
    net = base_network.as_graph(remove_edges=[0, 3])
    expected = [[], [(2, 2), (3, 4)], [(1, 2)], [(1, 4)]]
    assert net.adjacency_list == expected
    assert not net.is_connected


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
    graph = NetworkGraph(edges, 6)
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
