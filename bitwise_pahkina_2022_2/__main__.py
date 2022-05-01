"""Command line interface"""

from pathlib import Path

from .network import BaseNetwork
from .optimize import optimize


def main():
    """Generate base network, create variations, find best solution"""
    network_json = str((Path(__file__).parent / "koodipahkina-data.json").absolute())
    res = optimize(network_json, termination={"n_max_gen": 20}, seed=1, verbose=True)
    base_net = BaseNetwork.from_json(network_json)

    print(f"Best solution found: \nF = {res.F}")

    best_x_binary = res.opt.get("pheno")[0]
    removed_edges = (best_x_binary == 0).nonzero()[0]
    new_net = base_net.as_graph(remove_edges=removed_edges.tolist())

    score = base_net.comparison_score(new_net)
    print(f"Best score: {score:.3f}")

    print(
        "Solution:",
    )
    print(", ".join(removed_edges.astype(str)))

    print(res.exec_time)


if __name__ == "__main__":
    main()
