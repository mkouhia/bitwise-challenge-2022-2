"""Test optimization"""

from pathlib import Path

import pytest
import numpy as np

from bitwise_challenge_2022_2.network import BaseNetwork, NetworkGraph
from bitwise_challenge_2022_2.optimize import _create_feasible_solutions


@pytest.fixture(name="network_json")
def network_json_fx() -> Path:
    """Return path to challenge json"""
    return (
        Path(__file__).parents[1]
        / "bitwise_challenge_2022_2"
        / "koodipahkina-data.json"
    )


def test_create_feasible_solutions(network_json: Path):
    """Solution vectors represent feasible solutions"""

    rng = np.random.default_rng()
    count = 7
    x_created = _create_feasible_solutions(network_json, rng, count)
    x_bool = x_created.astype(bool)

    assert len(x_created) == count

    base_network = BaseNetwork.from_json(network_json)
    base_matrix = base_network.to_adjacency_matrix()
    edges = base_network.get_edge_matrix()

    for i in range(count):
        new_net = NetworkGraph(base_matrix)
        new_net.remove_edges(edges[~x_bool[i]])
        assert new_net.is_connected
