"""Command line interface"""

import argparse
import multiprocessing
from pathlib import Path
import signal
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

    if parsed_args.multiprocessing:
        n_threads = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(n_threads, _init_worker)
    else:
        pool = None

    termination = {}
    if parsed_args.max_gen is not None:
        termination["n_max_gen"] = parsed_args.max_gen

    try:
        network_json = Path(__file__).parent / "koodipahkina-data.json"
        res = optimize(
            network_json,
            pool=pool,
            termination=termination,
            seed=1,
            verbose=not parsed_args.quiet,
            x_path=parsed_args.xpath,
            metric_log=parsed_args.metric_log,
            resume=parsed_args.resume,
        )

        if not parsed_args.quiet:
            base_net = BaseNetwork.from_json(network_json)
            _print_report(res, base_net)

    except KeyboardInterrupt:
        print("\nInterrupted, exiting")
        if pool is not None:
            pool.terminate()
            pool.join()
        sys.exit(1)
    else:
        if pool is not None:
            pool.close()
            pool.join()


def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _print_report(res: Result, base_net: BaseNetwork):
    x_binary = res.opt.get("pheno")[0]
    del_edges = (x_binary == 0).nonzero()[0]
    remove_edges = np.array([base_net.edges[i] for i in del_edges.tolist()])

    base_mat = base_net.to_adjacency_matrix()
    new_net = NetworkGraph(base_mat)
    new_net.remove_edges(remove_edges)

    score = base_net.comparison_score(new_net)

    print(f"\nBest score: {score:.3f}")
    print(f"Execution time: {res.exec_time:.2f} s")
    print("Solution:")
    print(", ".join(del_edges.astype(str)))


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser(description="""Optimize liana network.""")
    parser.add_argument(
        "--max-gen",
        default=None,
        type=int,
        help="Maximum number of generations for genetic algorithm. Default: None",
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

    return parser.parse_args(args)


if __name__ == "__main__":
    main()
