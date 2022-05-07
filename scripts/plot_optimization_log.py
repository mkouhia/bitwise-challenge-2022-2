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

        y_scales = (
            parsed_args.yscale.split(":")
            if ":" in parsed_args.yscale
            else [parsed_args.yscale, parsed_args.yscale]
        )
        plt.yscale(y_scales[0], yside="left")
        plt.yscale(y_scales[1], yside="right")
        plt.xscale(parsed_args.xscale)

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
        help="""Location for logging optimization running metrics.
Default: opt_log.txt""",
    )
    parser.add_argument("--list", action="store_true", help="List available variables")
    parser.add_argument(
        "--watch", action="store_true", help="Watch for file, update plot continously"
    )
    parser.add_argument(
        "--y1",
        default="f_opt,f_avg",
        help="""Variables to be plotted on the left y-axis.
Separated by comma. Default: f_opt,f_avg""",
    )
    parser.add_argument(
        "--y2",
        default=None,
        help="""Variables to be plotted on the right y-axis.
Separated by comma. Default: None""",
    )
    parser.add_argument(
        "--yscale",
        default="linear",
        help="""Y axis scaling, linear|log, or [linear|log]:[linear|log]
            for different left/right values. Default: linear""",
    )
    parser.add_argument(
        "--xscale",
        default="linear",
        help="X axis scaling, linear|log. Default: linear",
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
