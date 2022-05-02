"""Command line interface"""

import argparse
from pathlib import Path
import sys

from pymoo.core.result import Result

from .network import BaseNetwork
from .optimize import optimize


def main(argv: list[str] = None):
    """Generate base network, create variations, find best solution

    Args:
        argv (list[str]): List of command line arguments. Defaults to None.
    """
    try:
        parsed_args = _parse_args(argv)

        network_json = Path(__file__).parent / "koodipahkina-data.json"
        res = optimize(
            network_json,
            termination={"n_max_gen": parsed_args.max_gen},
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
        sys.exit(1)


def _print_report(res: Result, base_net: BaseNetwork):
    best_x_binary = res.opt.get("pheno")[0]
    removed_edges = (best_x_binary == 0).nonzero()[0]
    new_net = base_net.as_graph(remove_edges=removed_edges.tolist())
    score = base_net.comparison_score(new_net)

    print(f"\nBest score: {score:.3f}")
    print(f"Execution time: {res.exec_time:.2f} s")
    print("Solution:")
    print(", ".join(removed_edges.astype(str)))


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser(description="""Optimize liana network.""")
    parser.add_argument(
        "--max-gen",
        default=75,
        type=int,
        help="Maximum number of generations for genetic algorithm. Default: 75",
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
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")

    return parser.parse_args(args)


if __name__ == "__main__":
    main()
