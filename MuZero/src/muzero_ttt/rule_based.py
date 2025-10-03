from __future__ import annotations

import random
from typing import List

import numpy as np

from .game import TicTacToe


class RuleBasedAgent:
    """Heuristic TicTacToe agent that plays close to optimally."""

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)

    def select_action(self, game: TicTacToe) -> int:
        legal = game.legal_actions()
        if not legal:
            raise ValueError("No legal actions available")
        player = game.to_play

        win_move = self._find_winning_move(game, legal, player)
        if win_move is not None:
            return win_move

        block_move = self._find_winning_move(game, legal, -player)
        if block_move is not None:
            return block_move

        if 4 in legal:
            return 4

        corners = [c for c in (0, 2, 6, 8) if c in legal]
        if corners:
            return self.random.choice(corners)

        return self.random.choice(legal)

    def _find_winning_move(self, game: TicTacToe, legal: List[int], player: int) -> int | None:
        for action in legal:
            row, col = TicTacToe.action_to_coords(action)
            if game.board[row, col] != 0:
                continue
            if self._would_win(game, action, player):
                return action
        return None

    def _would_win(self, game: TicTacToe, action: int, player: int) -> bool:
        row, col = TicTacToe.action_to_coords(action)
        temp_board = game.board.copy()
        temp_board[row, col] = player
        # Check all winning lines that include the new move
        if np.sum(temp_board[row, :]) == 3 * player:
            return True
        if np.sum(temp_board[:, col]) == 3 * player:
            return True
        if row == col and np.sum(np.diag(temp_board)) == 3 * player:
            return True
        if row + col == 2 and np.sum(np.diag(np.fliplr(temp_board))) == 3 * player:
            return True
        return False
