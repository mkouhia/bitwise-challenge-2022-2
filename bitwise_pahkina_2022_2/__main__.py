"""Command line interface"""

from pathlib import Path

from .network import BaseNetwork
from .optimize import optimize


def main():
    """Generate base network, create variations, find best solution"""
    base_net = BaseNetwork.from_json(Path(__file__).parent / "koodipahkina-data.json")

    optimize(base_net)

    # i = 0
    # for del_edges in [
    #     (0, 2, 6),
    #     (7, 3, 5, 1, 10, 20),
    # ]:
    #     net_x = base_net.as_graph(remove_edges=del_edges)
    #     print(f"{i:>6} {base_net.comparison_score(net_x):9.3f} {del_edges}")

    #     i += 1


if __name__ == "__main__":
    main()
