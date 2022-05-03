"""Test network graphs"""

import networkx as nx
import pytest

from bitwise_challenge_2022_2.network import NetworkGraph


@pytest.fixture(name="simple_graph")
def simple_graph_fx() -> NetworkGraph:
    """Create simple graph for testing"""
    edges = [
        (0, 1, 1),
        (1, 2, 2),
        (1, 3, 4),
        (3, 0, 8),
    ]
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    return NetworkGraph(graph)


def test_copy(simple_graph: NetworkGraph):
    """A new graph is returned, with same graph and total weight"""
    another = simple_graph.copy()

    # pylint: disable=protected-access
    assert list(another._graph.edges.data()) == list(simple_graph._graph.edges.data())
    assert another._graph is not simple_graph._graph
    assert another._total_weight == simple_graph._total_weight


def test_remove_edge(simple_graph: NetworkGraph):
    """Edge is removed from underlying graph"""
    simple_graph.remove_edge(0, 1)

    # pylint: disable=protected-access
    assert not simple_graph._graph.has_edge(0, 1)


def test_evaluate(simple_graph: NetworkGraph):
    """Evaluation score is as per guidelines"""
    assert simple_graph.evaluate(A=0.1, B=3.0) == 12.0


def test_is_connected(simple_graph: NetworkGraph):
    """All nodes are connected to each other"""
    assert simple_graph.is_connected


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
