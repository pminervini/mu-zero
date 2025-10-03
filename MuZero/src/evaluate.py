from __future__ import annotations

import argparse
import json
from pathlib import Path

from muzero_ttt import MuZeroAgent, MuZeroConfig


def default_checkpoint() -> Path:
    return Path(__file__).resolve().parent.parent / "muzero_checkpoint.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MuZero TicTacToe agent")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint(), help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=80, help="Number of evaluation games")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MuZeroConfig()
    agent = MuZeroAgent(config)
    if args.checkpoint.is_file():
        agent.load(str(args.checkpoint))
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    results = agent.evaluate_vs_rule_based(args.games)
    print(json.dumps({"games": args.games, "results": results}, indent=2))


if __name__ == "__main__":
    main()
