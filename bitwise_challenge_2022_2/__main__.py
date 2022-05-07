"""Command line interface"""

import argparse
from pathlib import Path
import sys
import numpy as np

from pymoo.core.result import Result

from .network import BaseNetwork, NetworkGraph
from .optimize import optimize


def main(argv: list[str] = None):
    """Generate base network, create variations, find best solution

    Args:
        argv (list[str]): List of command line arguments. Defaults to None.
    """
    parsed_args = _parse_args(argv)

    termination = {}
    if parsed_args.termination is not None:
        for item in parsed_args.termination.split(","):
            parts = item.split(":")
            val = int(parts[1]) if parts[0] in ["nth_gen", "n_last", "n_max_gen", "n_max_evals"] else float(parts[1])
            termination[parts[0]] = val

    try:
        network_json = Path(__file__).parent / "koodipahkina-data.json"
        res = optimize(
            network_json,
            termination=termination,
            seed=1,
            verbose=not parsed_args.quiet,
            x_path=parsed_args.xpath,
            metric_log=parsed_args.metric_log,
            resume=parsed_args.resume,
            plot=parsed_args.plot,
        )

        if not parsed_args.quiet:
            base_net = BaseNetwork.from_json(network_json)
            _print_report(res, base_net)

    except KeyboardInterrupt:
        print("\nInterrupted, exiting")
        sys.exit(1)


def _print_report(res: Result, base_net: BaseNetwork):
    x_binary = res.opt.get("pheno")[0]
    del_edges = (x_binary == 0).nonzero()[0]
    remove_edges = np.array([base_net.edges[i] for i in del_edges.tolist()])

    base_mat = base_net.to_adjacency_matrix()
    new_net = NetworkGraph(base_mat)
    new_net.remove_edges(remove_edges)

    score = base_net.comparison_score(new_net)

    print("\nBinary random key genetic algorithm")
    print(f"- Arguments: {' '.join(sys.argv)}")
    print("Random initialization")
    print(f"{res.algorithm.n_gen} generations")
    print(f"Best score: {score:.3f}")
    print(f"Execution time: {res.exec_time:.2f} s")
    print("Solution:")
    print(", ".join(del_edges.astype(str)))


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser(description="""Optimize liana network.""")
    parser.add_argument(
        "--termination",
        default="n_max_evals:1000000",
        help="""Termination specification in format key1:value1,key2:value2.
            Available values: see keyword arguments at
            https://pymoo.org/interface/termination.html .
            Default: n_max_evals:1000000""",
    )
    parser.add_argument(
        "--metric-log",
        default="opt_log.txt",
        help="Location for logging optimization running metrics. Default: opt_log.txt",
    )
    parser.add_argument(
        "--xpath",
        default="opt_X_latest.npy",
        help="Location for saving intermediate x values for population. Default: opt_X_latest.npy",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from xfile")
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="Use multiprocessing for problem evaluation",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    parser.add_argument(
        "--plot", action="store_true", help="Display progress plot during calculation"
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    main()
