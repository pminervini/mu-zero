from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TicTacToe:
    """Simple TicTacToe environment suitable for MuZero self-play."""

    board: np.ndarray
    to_play: int

    def __init__(self) -> None:
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.to_play = 1  # 1 for "X", -1 for "O"

    def clone(self) -> "TicTacToe":
        clone = TicTacToe()
        clone.board = self.board.copy()
        clone.to_play = self.to_play
        return clone

    def reset(self) -> None:
        self.board.fill(0)
        self.to_play = 1

    @staticmethod
    def action_to_coords(action: int) -> tuple[int, int]:
        if action < 0 or action >= 9:
            raise ValueError(f"Action index out of range: {action}")
        return divmod(action, 3)

    def observation(self) -> np.ndarray:
        current = (self.board == self.to_play).astype(np.float32)
        opponent = (self.board == -self.to_play).astype(np.float32)
        return np.stack([current, opponent], axis=0)

    def legal_actions(self) -> List[int]:
        return [i for i in range(9) if self.board.flat[i] == 0]

    def apply(self, action: int) -> float:
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action}")
        row, col = self.action_to_coords(action)
        player = self.to_play
        self.board[row, col] = player

        winner = self._check_winner()
        reward = 0.0
        if winner == player:
            reward = 1.0
        elif winner == -player:
            reward = -1.0
        elif self.is_full():
            reward = 0.0

        self.to_play = -player
        return reward

    def undo(self, action: int) -> None:
        row, col = self.action_to_coords(action)
        if self.board[row, col] == 0:
            raise ValueError("Cannot undo an empty square")
        self.board[row, col] = 0
        self.to_play = -self.to_play

    def is_full(self) -> bool:
        return bool(np.all(self.board != 0))

    def is_terminal(self) -> bool:
        return self._check_winner() != 0 or self.is_full()

    def _check_winner(self) -> int:
        lines = []
        lines.extend(list(self.board))  # rows
        lines.extend(list(self.board.T))  # columns
        lines.append(np.diag(self.board))
        lines.append(np.diag(np.fliplr(self.board)))
        for line in lines:
            total = np.sum(line)
            if total == 3:
                return 1
            if total == -3:
                return -1
        return 0

    def outcome(self) -> float:
        winner = self._check_winner()
        if winner == 0:
            return 0.0
        return 1.0 if winner == 1 else -1.0

    def render(self) -> str:
        symbols = {1: "X", -1: "O", 0: " "}
        rows = [" | ".join(symbols[self.board[r, c]] for c in range(3)) for r in range(3)]
        return "\n---------\n".join(rows)
