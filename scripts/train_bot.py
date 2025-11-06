from __future__ import annotations

import argparse
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from crbot.training import TrainingStatus, train_agent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Clash Royale bot.")
    parser.add_argument("--episodes", type=int, default=10_000, help="Number of training episodes to run.")
    parser.add_argument("--batch-size", type=int, default=32, help="Replay batch size.")
    parser.add_argument("--no-web", action="store_true", help="Disable the status web server.")
    parser.add_argument("--web-host", default="127.0.0.1", help="Status web server host.")
    parser.add_argument("--web-port", type=int, default=5000, help="Status web server port.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    status = TrainingStatus()
    train_agent(
        episodes=args.episodes,
        batch_size=args.batch_size,
        status=status,
        start_web=not args.no_web,
        web_host=args.web_host,
        web_port=args.web_port,
    )


if __name__ == "__main__":
    main()
