from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MuZeroConfig:
    """Configuration bundle for the MuZero TicTacToe agent."""

    action_space: List[int] = field(default_factory=lambda: list(range(9)))
    observation_shape: tuple[int, int, int] = (2, 3, 3)
    discount: float = 1.0
    num_simulations: int = 80
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.1
    pb_c_base: float = 19652
    pb_c_init: float = 1.25

    replay_buffer_size: int = 160
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    num_unroll_steps: int = 5
    td_steps: int = 9
    training_steps_per_loop: int = 240
    self_play_games_per_loop: int = 10
    training_loops: int = 30

    temperature_initial_moves: int = 4
    temperature: float = 0.6
    temperature_endgame: float = 0.05

    device_preference: tuple[str, ...] = ("mps", "cuda", "cpu")

    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_unroll_steps < 0:
            raise ValueError("num_unroll_steps must be non-negative")
        if self.td_steps < 1:
            raise ValueError("td_steps must be at least one")
