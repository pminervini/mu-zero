from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import torch

from .config import MuZeroConfig
from .game import TicTacToe
from .network import MuZeroNet


@dataclass
class MinMaxStats:
    minimum: float = field(default=float("inf"))
    maximum: float = field(default=float("-inf"))

    def update(self, value: float) -> None:
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    def __init__(self, prior: float, to_play: int, reward: float = 0.0, value: float = 0.0) -> None:
        self.prior = prior
        self.to_play = to_play
        self.reward = reward
        self.value_estimate = value
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, "Node"] = {}
        self.hidden_state: torch.Tensor | None = None
        self.policy_logits: torch.Tensor | None = None
        self.state: TicTacToe | None = None

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, config: MuZeroConfig, network: MuZeroNet, device: torch.device) -> None:
        if self.state is None or self.policy_logits is None or self.hidden_state is None:
            return
        if self.children:
            return

        policy = torch.softmax(self.policy_logits, dim=0).detach().cpu().numpy()
        legal_actions = self.state.legal_actions()
        masked_policy = np.zeros_like(policy)
        if legal_actions:
            exp_sum = np.sum(policy[legal_actions])
            if exp_sum <= 0:
                masked_policy[legal_actions] = 1.0 / len(legal_actions)
            else:
                masked_policy[legal_actions] = policy[legal_actions] / exp_sum
        else:
            masked_policy.fill(0.0)

        parent_hidden = self.hidden_state.detach()
        for action in legal_actions:
            next_state = self.state.clone()
            env_reward = next_state.apply(action)
            with torch.no_grad():
                network_output = network.recurrent_inference(
                    parent_hidden, torch.tensor([action], dtype=torch.long, device=device)
                )
            child = Node(
                prior=float(masked_policy[action]),
                to_play=next_state.to_play,
                reward=float(env_reward),
                value=float(network_output.value.squeeze(0).item()),
            )
            child.hidden_state = network_output.hidden_state.squeeze(0).detach()
            child.policy_logits = network_output.policy_logits.squeeze(0).detach()
            child.state = next_state
            self.children[action] = child

    def is_expanded(self) -> bool:
        return bool(self.children)


class MCTS:
    def __init__(self, config: MuZeroConfig, network: MuZeroNet, device: torch.device) -> None:
        self.config = config
        self.network = network
        self.device = device

    def run(self, game: TicTacToe, add_exploration_noise: bool = True) -> Tuple[Node, torch.Tensor]:
        return self._run_search(game, add_exploration_noise)

    def _run_search(self, game: TicTacToe, add_exploration_noise: bool) -> Tuple[Node, torch.Tensor]:
        min_max_stats = MinMaxStats()
        root = Node(prior=1.0, to_play=game.to_play)
        observation = torch.from_numpy(game.observation()).to(self.device)
        with torch.no_grad():
            network_output = self.network.initial_inference(observation)
        root.hidden_state = network_output.hidden_state.squeeze(0).detach()
        root.policy_logits = network_output.policy_logits.squeeze(0).detach()
        root.value_estimate = float(network_output.value.squeeze(0).item())
        root.state = game.clone()
        root.expand(self.config, self.network, self.device)
        if add_exploration_noise:
            self._add_exploration_noise(root)

        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            simulation_state = game.clone()

            while node.is_expanded() and not simulation_state.is_terminal():
                action, node = self._select_child(node, min_max_stats)
                simulation_state.apply(action)
                search_path.append(node)

            if not simulation_state.is_terminal():
                node.expand(self.config, self.network, self.device)

            value = self._value_for_state(simulation_state, node)
            self._backpropagate(search_path, value, min_max_stats)

        policy = self._visit_count_distribution(root)
        return root, policy

    def _value_for_state(self, state: TicTacToe, node: Node) -> float:
        if state.is_terminal():
            outcome = state.outcome()
            return outcome * node.to_play
        return node.value_estimate

    def _select_child(self, node: Node, min_max_stats: MinMaxStats) -> Tuple[int, Node]:
        best_score = float("-inf")
        best_action = -1
        best_child = None
        for action, child in node.children.items():
            score = self._ucb_score(node, child, min_max_stats)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        if best_child is None:
            raise RuntimeError("Failed to select a child node")
        return best_action, best_child

    def _ucb_score(self, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
        prior_score = self._puct_prior(parent, child)
        if child.visit_count > 0:
            q_value = child.reward + self.config.discount * (-child.value())
        else:
            q_value = child.reward + self.config.discount * (-child.value_estimate)
        min_max_stats.update(q_value)
        value_score = min_max_stats.normalize(q_value)
        return prior_score + value_score

    def _puct_prior(self, parent: Node, child: Node) -> float:
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
        return pb_c * child.prior

    def _backpropagate(self, search_path: list[Node], value: float, min_max_stats: MinMaxStats) -> None:
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.config.discount * (-value)
            min_max_stats.update(value)

    def _visit_count_distribution(self, root: Node) -> torch.Tensor:
        visit_counts = np.array([root.children[a].visit_count if a in root.children else 0 for a in self.config.action_space], dtype=np.float32)
        if visit_counts.sum() == 0:
            visit_counts += 1.0
        return torch.tensor(visit_counts / visit_counts.sum(), dtype=torch.float32)

    def _add_exploration_noise(self, node: Node) -> None:
        if not node.children:
            return
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(node.children))
        for (action, child), eta in zip(node.children.items(), noise):
            child.prior = child.prior * (1 - self.config.dirichlet_fraction) + eta * self.config.dirichlet_fraction
