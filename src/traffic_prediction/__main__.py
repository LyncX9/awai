from __future__ import annotations

import argparse

from traffic_prediction.pipelines.offline import main as offline_main


def main() -> None:
    parser = argparse.ArgumentParser(prog="traffic_prediction")
    parser.add_argument(
        "command",
        choices=["offline-pipeline"],
        help="Command to execute.",
    )
    args = parser.parse_args()

    if args.command == "offline-pipeline":
        offline_main()


if __name__ == "__main__":
    main()

