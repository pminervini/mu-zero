from __future__ import annotations

import argparse
import json
import time

from muzero_ttt import MuZeroAgent, MuZeroConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MuZero on TicTacToe")
    parser.add_argument("--loops", type=int, default=None, help="Override number of training loops")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to save the trained model")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to an existing checkpoint to resume from")
    parser.add_argument("--evaluation-games", type=int, default=40, help="Number of evaluation games after each loop")
    parser.add_argument("--stop-on-flawless", action="store_true", help="Stop when MuZero no longer loses to the rule-based agent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MuZeroConfig()
    if args.loops is not None:
        config.training_loops = args.loops

    agent = MuZeroAgent(config)

    if args.load_checkpoint:
        agent.load(args.load_checkpoint)

    print(json.dumps({"device": str(agent.device)}, indent=2))

    for loop_index in range(config.training_loops):
        start_time = time.time()
        agent.self_play(config.self_play_games_per_loop)
        metrics = agent.train(config.training_steps_per_loop)
        duration = time.time() - start_time

        evaluation = agent.evaluate_vs_rule_based(args.evaluation_games)

        loop_report = {
            "loop": loop_index + 1,
            "self_play_games": config.self_play_games_per_loop,
            "training_steps": config.training_steps_per_loop,
            "metrics": metrics,
            "evaluation": evaluation,
            "duration_sec": round(duration, 2),
        }
        print(json.dumps(loop_report, indent=2))

        if args.stop_on_flawless and evaluation["losses"] == 0:
            print("Stopping early: MuZero has no losses against the rule-based agent.")
            break

    if args.checkpoint:
        agent.save(args.checkpoint)

    final_eval = agent.evaluate_vs_rule_based(max(args.evaluation_games, 80))
    print(json.dumps({"final_evaluation": final_eval}, indent=2))


if __name__ == "__main__":
    main()
