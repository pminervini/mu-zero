from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from torch import nn

from .config import MuZeroConfig
from .game import TicTacToe
from .mcts import MCTS, Node
from .network import MuZeroNet
from .rule_based import RuleBasedAgent
from .utils import select_device, set_seed


@dataclass
class GameHistory:
    action_space_size: int
    observations: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    to_play: List[int]
    root_values: List[float]
    child_visits: List[np.ndarray]

    def __init__(self, action_space_size: int) -> None:
        self.action_space_size = action_space_size
        self.observations = []
        self.actions = []
        self.rewards = []
        self.to_play = []
        self.root_values = []
        self.child_visits = []

    def store_step(
        self,
        observation: np.ndarray,
        to_play: int,
        action: int,
        reward: float,
        root_value: float,
        policy: torch.Tensor,
    ) -> None:
        self.observations.append(observation.astype(np.float32))
        self.to_play.append(to_play)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.root_values.append(float(root_value))
        self.child_visits.append(policy.cpu().numpy().astype(np.float32, copy=True))

    def make_target(
        self,
        state_index: int,
        num_unroll_steps: int,
        td_steps: int,
        discount: float,
    ) -> List[tuple[float, float, np.ndarray]]:
        targets: List[tuple[float, float, np.ndarray]] = []
        for step in range(num_unroll_steps + 1):
            current_index = state_index + step
            value = self._compute_value(current_index, state_index, td_steps, discount)
            reward_index = state_index + step - 1
            reward = self.rewards[reward_index] if step > 0 and reward_index < len(self.rewards) else 0.0
            policy = self._policy_target(current_index)
            targets.append((value, reward, policy))
        return targets

    def _policy_target(self, index: int) -> np.ndarray:
        if index < len(self.child_visits):
            visits = self.child_visits[index]
            total = np.sum(visits)
            if total > 0:
                return visits / total
            mask = visits > 0
            if np.any(mask):
                distribution = np.zeros_like(visits)
                distribution[mask] = 1.0 / np.count_nonzero(mask)
                return distribution
        return np.full(self.action_space_size, 1.0 / self.action_space_size, dtype=np.float32)

    def _compute_value(
        self,
        current_index: int,
        reference_index: int,
        td_steps: int,
        discount: float,
    ) -> float:
        if current_index >= len(self.observations):
            return 0.0
        value = 0.0
        current_player = self.to_play[reference_index]
        for i in range(td_steps):
            idx = current_index + i
            if idx >= len(self.rewards):
                break
            reward = self.rewards[idx]
            player = self.to_play[idx]
            if player != current_player:
                reward = -reward
            value += (discount**i) * reward
        bootstrap_index = current_index + td_steps
        if bootstrap_index < len(self.root_values):
            bootstrap_value = self.root_values[bootstrap_index]
            if self.to_play[bootstrap_index] != current_player:
                bootstrap_value = -bootstrap_value
            value += (discount**td_steps) * bootstrap_value
        return value


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: List[GameHistory] = []

    def __len__(self) -> int:  # pragma: no cover - simple container helper
        return len(self.buffer)

    def add_game(self, history: GameHistory) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(history)

    def sample_batch(self, batch_size: int) -> List[tuple[GameHistory, int]]:
        games = [random.choice(self.buffer) for _ in range(batch_size)]
        batch = []
        for game in games:
            position = random.randrange(len(game.observations))
            batch.append((game, position))
        return batch


class MuZeroAgent:
    def __init__(self, config: MuZeroConfig) -> None:
        self.config = config
        set_seed(config.seed)
        self.device = select_device(config.device_preference)
        self.network = MuZeroNet(action_space_size=len(config.action_space))
        self.network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.mcts = MCTS(config, self.network, self.device)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self.mse_loss = nn.MSELoss()

    def self_play(self, num_games: int) -> List[GameHistory]:
        self.network.eval()
        histories: List[GameHistory] = []
        for _ in range(num_games):
            game = TicTacToe()
            history = GameHistory(len(self.config.action_space))
            while not game.is_terminal():
                observation = game.observation()
                to_play = game.to_play
                root, policy = self.mcts.run(game, add_exploration_noise=True)
                temperature = self._temperature(len(history.actions))
                action = self._sample_action(policy.cpu().numpy(), game.legal_actions(), temperature)
                reward = game.apply(action)
                history.store_step(
                    observation=observation,
                    to_play=to_play,
                    action=action,
                    reward=reward,
                    root_value=root.value(),
                    policy=policy,
                )
            histories.append(history)
            self.replay_buffer.add_game(history)
        return histories

    def train(self, num_steps: int) -> dict[str, float]:
        if len(self.replay_buffer) == 0:
            return {}
        self.network.train()
        metrics = {"value_loss": 0.0, "reward_loss": 0.0, "policy_loss": 0.0}
        for step in range(num_steps):
            batch = self.replay_buffer.sample_batch(self.config.batch_size)
            loss, batch_metrics = self._update_network(batch)
            for key in metrics:
                metrics[key] += batch_metrics[key]
        for key in metrics:
            metrics[key] /= max(1, num_steps)
        return metrics

    def evaluate_vs_rule_based(self, num_games: int, rule_agent: RuleBasedAgent | None = None) -> dict[str, int]:
        if rule_agent is None:
            rule_agent = RuleBasedAgent(seed=self.config.seed)
        results = {"wins": 0, "losses": 0, "draws": 0}
        self.network.eval()
        for game_index in range(num_games):
            game = TicTacToe()
            muzero_player = 1 if game_index % 2 == 0 else -1
            while not game.is_terminal():
                if game.to_play == muzero_player:
                    root, policy = self.mcts.run(game, add_exploration_noise=False)
                    action = self._sample_action(policy.cpu().numpy(), game.legal_actions(), temperature=0.0)
                else:
                    action = rule_agent.select_action(game)
                game.apply(action)
            result = game.outcome() * muzero_player
            if result > 0:
                results["wins"] += 1
            elif result < 0:
                results["losses"] += 1
            else:
                results["draws"] += 1
        return results

    def save(self, path: str) -> None:
        checkpoint = {
            "model_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config.__dict__,
        }
        with open(path, "wb") as handle:
            torch.save(checkpoint, handle)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

    def _update_network(self, batch: List[tuple[GameHistory, int]]) -> tuple[torch.Tensor, dict[str, float]]:
        value_loss = torch.tensor(0.0, device=self.device)
        reward_loss = torch.tensor(0.0, device=self.device)
        policy_loss = torch.tensor(0.0, device=self.device)

        for game, start_index in batch:
            observation_tensor = torch.from_numpy(game.observations[start_index]).to(self.device)
            targets = game.make_target(start_index, self.config.num_unroll_steps, self.config.td_steps, self.config.discount)
            hidden_state_output = self.network.initial_inference(observation_tensor)

            target_value0 = torch.tensor(targets[0][0], dtype=torch.float32, device=self.device)
            target_policy0 = torch.tensor(targets[0][2], dtype=torch.float32, device=self.device)

            value_loss = value_loss + self.mse_loss(hidden_state_output.value.squeeze(0), target_value0)
            policy_loss = policy_loss + self._policy_cross_entropy(hidden_state_output.policy_logits.squeeze(0), target_policy0)

            hidden_state = hidden_state_output.hidden_state

            for step, target in enumerate(targets[1:]):
                if start_index + step >= len(game.actions):
                    break
                action_id = game.actions[start_index + step]
                action_tensor = torch.tensor([action_id], dtype=torch.long, device=self.device)
                recurrent_output = self.network.recurrent_inference(hidden_state, action_tensor)

                target_value = torch.tensor(target[0], dtype=torch.float32, device=self.device)
                target_reward = torch.tensor(target[1], dtype=torch.float32, device=self.device)
                target_policy = torch.tensor(target[2], dtype=torch.float32, device=self.device)

                value_loss = value_loss + self.mse_loss(recurrent_output.value.squeeze(0), target_value)
                reward_loss = reward_loss + self.mse_loss(recurrent_output.reward.squeeze(0), target_reward)
                policy_loss = policy_loss + self._policy_cross_entropy(recurrent_output.policy_logits.squeeze(0), target_policy)

                hidden_state = recurrent_output.hidden_state

        batch_size = len(batch)
        total_loss = (value_loss + reward_loss + policy_loss) / batch_size
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.optimizer.step()
        metrics = {
            "value_loss": float((value_loss / batch_size).item()),
            "reward_loss": float((reward_loss / batch_size).item()),
            "policy_loss": float((policy_loss / batch_size).item()),
        }
        return total_loss.detach(), metrics

    def _policy_cross_entropy(self, logits: torch.Tensor, target_policy: torch.Tensor) -> torch.Tensor:
        log_prob = torch.log_softmax(logits, dim=-1)
        return -(target_policy * log_prob).sum()

    def _temperature(self, move_index: int) -> float:
        if move_index < self.config.temperature_initial_moves:
            return self.config.temperature
        return self.config.temperature_endgame

    def _sample_action(self, policy: np.ndarray, legal_actions: Sequence[int], temperature: float) -> int:
        mask = np.zeros_like(policy)
        mask[legal_actions] = 1
        probabilities = policy * mask
        if probabilities.sum() <= 0:
            probabilities = mask
        if temperature <= 1e-5:
            return int(np.argmax(probabilities))
        logits = np.log(probabilities + 1e-8) / max(temperature, 1e-5)
        logits = logits - logits.max()
        exp_logits = np.exp(logits) * mask
        if exp_logits.sum() <= 0:
            exp_logits = mask
        distribution = exp_logits / exp_logits.sum()
        return int(np.random.choice(len(distribution), p=distribution))
