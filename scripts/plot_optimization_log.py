"""Create plot from optimization logs"""

import argparse
import sys
import time

try:
    import plotext as plt
except ImportError:
    print("Plotext not installed, exiting")
    sys.exit(1)


def main(argv: list[str] = None):
    """Plot optimization logs

    Args:
        argv (list[str]): List of command line arguments. Defaults to None.
    """
    parsed_args = _parse_args(argv)

    with open(parsed_args.metric_log, "r", encoding="utf-8") as logf:
        headers = logf.readline().strip().split(",")

    if parsed_args.list:
        print("Available variables:")
        print("\n".join(headers))
        sys.exit(0)

    plt.title("Optimization logs")

    while True:

        logs = plt.read_data(parsed_args.metric_log, header=False, delimiter=",")

        plt.clear_terminal()
        plt.clear_data()
        plt.clear_figure()

        for y_raw, side in [(parsed_args.y1, "left"), (parsed_args.y2, "right")]:
            if y_raw is None:
                continue

            for y_i in y_raw.split(","):
                i = headers.index(y_i)
                plt.plot(logs[i], yside=side, label=headers[i])
        plt.show()

        if parsed_args.watch:
            time.sleep(1.0)
        else:
            break


def _parse_args(args: list[str]) -> argparse.Namespace:
    """Parse arguments from command line"""
    parser = argparse.ArgumentParser(description="""Plot optimization logs""")
    parser.add_argument(
        "--metric-log",
        default="opt_log.txt",
        help="Location for logging optimization running metrics. Default: opt_log.txt",
    )
    parser.add_argument("--list", action="store_true", help="List available variables")
    parser.add_argument(
        "--watch", action="store_true", help="Watch for file, update plot continously"
    )
    parser.add_argument(
        "--y1",
        default="fopt",
        help="Variables to be plotted on the left y-axis. Separated by comma. Default: fopt",
    )
    parser.add_argument(
        "--y2",
        help="Variables to be plotted on the right y-axis. Separated by comma.",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
