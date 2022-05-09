"""Command line interface"""

import argparse
from pathlib import Path
import sys

from .optimize import BRKGAOptimization


def main(argv: list[str] = None):
    """Generate base network, create variations, find best solution

    Args:
        argv (list[str]): List of command line arguments. Defaults to None.
    """
    parsed_args = _parse_args(argv)

    restart = {}
    if parsed_args.restart is not None:
        for item in parsed_args.restart.split(","):
            parts = item.split(":")
            val = (
                int(parts[1])
                if parts[0] in ["nth_gen", "n_last", "n_max_gen", "n_max_evals"]
                else float(parts[1])
            )
            restart[parts[0]] = val

    try:
        network_json = Path(__file__).parent / "koodipahkina-data.json"
        solver = BRKGAOptimization(network_json)
        solver.optimize(
            termination=restart,
            n_trials=parsed_args.n_trials,
            optuna_prune=parsed_args.optuna_prune,
            seed=parsed_args.seed,
            verbose=not parsed_args.quiet,
            x_path=parsed_args.xpath,
            metric_log=parsed_args.metric_log,
            resume=parsed_args.resume,
            plot=parsed_args.plot,
            elite_frac=parsed_args.elite_frac,
            mutant_frac=parsed_args.mutant_frac,
            elite_bias=parsed_args.elite_bias,
        )

    except KeyboardInterrupt:
        print("\nInterrupted, exiting")
        sys.exit(1)


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser(description="""Optimize liana network.""")
    parser.add_argument(
        "--n-trials",
        default=None,
        type=int,
        help="Number of trials for hyperparameter optimization. Default: None",
    )
    parser.add_argument(
        "--optuna-prune",
        action="store_true",
        help="Use pruning in hyperparameter optimization. Default: no pruning",
    )
    parser.add_argument(
        "--restart",
        default="n_max_evals:1000000,n_last:500,n_max_gen:1000",
        help="""Restart specification in format key1:value1,key2:value2.
            Available values: see keyword arguments at
            https://pymoo.org/interface/termination.html .
            Default: n_max_evals:1000000,n_last:500,n_max_gen:1000""",
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
    parser.add_argument("--seed", default=1, type=int, help="Random seed. Default: 1")
    parser.add_argument(
        "--elite-frac",
        default=0.2,
        type=float,
        help="Elite fraction of population. Default: 0,2",
    )
    parser.add_argument(
        "--mutant-frac",
        default=0.1,
        type=float,
        help="Mutant fraction of population. Default: 0.1",
    )
    parser.add_argument(
        "--elite-bias",
        default=0.7,
        type=float,
        help="Probability of elite gene transfer. Default: 0.7",
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    main()
