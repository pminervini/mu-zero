#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server, WSGIServer

# Ensure we can import the local package when running from the repo root.
REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np

from muzero_ttt import MuZeroAgent, MuZeroConfig, TicTacToe

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Tic-Tac-Toe against a MuZero checkpoint via a small web UI.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a MuZero checkpoint (produced by train.py)")
    parser.add_argument("--host", default="127.0.0.1", help="Web server host (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: %(default)s)")
    parser.add_argument("--agent-first", action="store_true", help="Let MuZero make the opening move (default: human starts).")
    return parser.parse_args(argv)


class GameSession:
    """In-memory single-player session that marshals moves between the human and MuZero."""

    def __init__(self, agent: MuZeroAgent, human_starts: bool) -> None:
        self._agent = agent
        self._lock = threading.Lock()
        self._last_policy: np.ndarray | None = None
        self._human_symbol = "X"
        self._agent_symbol = "O"
        self._reset(human_starts=human_starts)

    @property
    def last_policy(self) -> np.ndarray | None:
        return self._last_policy

    @property
    def message(self) -> str:
        return self._status_message

    @property
    def game(self) -> TicTacToe:
        return self._game

    def reset(self, human_starts: bool = True) -> str:
        with self._lock:
            return self._reset(human_starts)

    def human_move(self, action: int) -> str:
        with self._lock:
            if self._game.is_terminal():
                return self._set_status("Game over. Reset to play again.")
            if self._game.to_play != 1:
                return self._set_status("Hang on, MuZero is thinking.")
            if action not in self._game.legal_actions():
                return self._set_status("Illegal move. Pick an empty square.")
            self._game.apply(action)
            if self._game.is_terminal():
                return self._final_status()
            return self._muzero_reply()

    def muzero_move(self) -> str:
        with self._lock:
            if self._game.is_terminal():
                return self._set_status("Game over. Reset to play again.")
            if self._game.to_play != -1:
                return self._set_status("It's your turn!")
            return self._muzero_reply()

    # Internal helpers -------------------------------------------------

    def _reset(self, human_starts: bool) -> str:
        self._game = TicTacToe()
        self._last_policy = None
        if human_starts:
            return self._set_status("Your move (you are X).")
        self._game.to_play = -1  # Let MuZero start.
        return self._muzero_reply()

    def _set_status(self, message: str) -> str:
        self._status_message = message
        return message

    def _muzero_reply(self) -> str:
        if self._game.is_terminal():
            return self._final_status()
        root, policy = self._agent.mcts.run(self._game, add_exploration_noise=False)
        policy_np = policy.detach().cpu().numpy()
        self._last_policy = policy_np
        legal = self._game.legal_actions()
        if not legal:
            return self._final_status()
        best_action = max(legal, key=lambda idx: policy_np[idx])
        self._game.apply(best_action)
        if self._game.is_terminal():
            return self._final_status()
        human_chance = f"MuZero plays {self._agent_symbol}. Your move!"
        return self._set_status(human_chance)

    def _final_status(self) -> str:
        outcome = self._game.outcome()
        if outcome > 0:
            return self._set_status("You win! 🎉")
        if outcome < 0:
            return self._set_status("MuZero wins. 🤖")
        return self._set_status("It's a draw.")


def render_page(session: GameSession, device: str) -> str:
    board_html = ["<table class=\"board\">"]
    symbols = {1: "X", -1: "O", 0: "&middot;"}
    game = session.game
    game_over = game.is_terminal()
    for row in range(3):
        board_html.append("<tr>")
        for col in range(3):
            idx = row * 3 + col
            value = int(game.board[row, col])
            token = symbols[value]
            if value == 0 and not game_over and game.to_play == 1:
                cell = f'<a class="cell" href="/?action=move&pos={idx}">{token}</a>'
            else:
                cell = f'<span class="cell">{token}</span>'
            board_html.append(f"<td>{cell}</td>")
        board_html.append("</tr>")
    board_html.append("</table>")

    policy_html = ""
    policy = session.last_policy
    if policy is not None:
        probs = ", ".join(f"{p:.2f}" for p in policy)
        policy_html = f"<p class=\"policy\"><strong>MuZero policy (visit fractions):</strong> [{probs}]</p>"

    style = """
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; text-align: center; margin: 0 auto; padding: 2rem; max-width: 640px; }
    h1 { margin-bottom: 0.5rem; }
    p.status { font-size: 1.2rem; margin-top: 0.5rem; }
    table.board { border-collapse: collapse; margin: 1.5rem auto; }
    table.board td { border: 2px solid #333; width: 80px; height: 80px; font-size: 2.5rem; }
    .cell { display: inline-block; width: 100%; height: 100%; line-height: 80px; text-decoration: none; color: #111; }
    a.cell:hover { background: #e8f0fe; }
    .controls { margin-top: 1rem; display: flex; justify-content: center; gap: 1rem; }
    .controls a { text-decoration: none; color: #0053ba; font-weight: 600; }
    .footnote { margin-top: 2rem; color: #555; font-size: 0.9rem; }
    .policy { font-family: monospace; color: #444; }
    """

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>MuZero Tic-Tac-Toe</title>
<style>{style}</style>
</head>
<body>
  <h1>MuZero Tic-Tac-Toe</h1>
  <p class=\"status\">{session.message}</p>
  {''.join(board_html)}
  <div class=\"controls\">
    <a href="/?action=reset">Reset</a>
    <a href="/?action=agent-first">MuZero first</a>
    <a href="/?action=agent-move">MuZero move</a>
  </div>
  {policy_html}
  <p class=\"footnote\">Serving from device: <strong>{device}</strong></p>
</body>
</html>"""


def build_app(session: GameSession, device: str) -> Callable:
    def application(environ, start_response):
        try:
            if environ.get("PATH_INFO", "/") != "/":
                start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
                return [b"Not Found"]

            query_params = parse_qs(environ.get("QUERY_STRING", ""))
            action = query_params.get("action", [None])[0]
            message = None

            if action == "move":
                pos_str = query_params.get("pos", [None])[0]
                if pos_str is None:
                    message = session.reset()
                else:
                    try:
                        move = int(pos_str)
                    except ValueError:
                        message = session.reset()
                    else:
                        message = session.human_move(move)
            elif action == "reset":
                message = session.reset(human_starts=True)
            elif action == "agent-first":
                message = session.reset(human_starts=False)
            elif action == "agent-move":
                message = session.muzero_move()

            if message:
                logging.info("Status: %s", message)

            body = render_page(session, device)
            payload = body.encode("utf-8")
            headers = [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(payload)))]
            start_response("200 OK", headers)
            return [payload]
        except Exception as exc:  # pragma: no cover - defensive
            logging.exception("Failed to render page")
            payload = f"Internal Server Error: {exc}".encode("utf-8")
            headers = [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(payload)))]
            start_response("500 Internal Server Error", headers)
            return [payload]

    return application


def main(argv: list[str] | None = None) -> None:
    args = parse_cli(argv)

    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = MuZeroConfig()
    agent = MuZeroAgent(config)
    agent.load(str(checkpoint_path))
    agent.network.eval()
    device_str = str(agent.device)

    session = GameSession(agent=agent, human_starts=not args.agent_first)
    app = build_app(session, device=device_str)

    with make_server(args.host, args.port, app) as httpd:  # type: WSGIServer
        url = f"http://{args.host}:{args.port}" if args.host not in {"0.0.0.0", "::"} else f"http://127.0.0.1:{args.port}"
        logging.info("Serving MuZero Tic-Tac-Toe on %s (device: %s)", url, device_str)
        logging.info("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:  # pragma: no cover - manual interrupt
            logging.info("Server stopped.")


if __name__ == "__main__":
    main()
