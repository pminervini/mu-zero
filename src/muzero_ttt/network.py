from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class NetworkOutput:
    value: torch.Tensor
    reward: torch.Tensor
    policy_logits: torch.Tensor
    hidden_state: torch.Tensor


class MuZeroNet(nn.Module):
    """Small neural network used for MuZero on TicTacToe."""

    def __init__(self, action_space_size: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.action_space_size = action_space_size
        self.hidden_dim = hidden_dim

        self.representation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(18, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.prediction_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_space_size),
        )
        self.prediction_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.dynamics_state = nn.Sequential(
            nn.Linear(hidden_dim + action_space_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.dynamics_reward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        obs = self._prepare_observation(observation)
        hidden_state = torch.tanh(self.representation(obs))
        policy_logits = self.prediction_policy(hidden_state)
        value = self.prediction_value(hidden_state).squeeze(-1)
        reward = torch.zeros_like(value)
        return NetworkOutput(value=value, reward=reward, policy_logits=policy_logits, hidden_state=hidden_state)

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> NetworkOutput:
        hidden = self._prepare_hidden(hidden_state)
        action_one_hot = self._prepare_action(action)
        dynamics_input = torch.cat([hidden, action_one_hot], dim=-1)
        next_hidden = torch.tanh(self.dynamics_state(dynamics_input))
        policy_logits = self.prediction_policy(next_hidden)
        value = self.prediction_value(next_hidden).squeeze(-1)
        reward = self.dynamics_reward(next_hidden).squeeze(-1)
        return NetworkOutput(value=value, reward=reward, policy_logits=policy_logits, hidden_state=next_hidden)

    def _prepare_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.dim() == 3:
            observation = observation.unsqueeze(0)
        return observation.float()

    def _prepare_hidden(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        return hidden_state.float()

    def _prepare_action(self, action: torch.Tensor | int) -> torch.Tensor:
        if isinstance(action, int):
            action_tensor = torch.tensor([action], dtype=torch.long, device=self._device_of_parameters())
        else:
            action_tensor = action.long()
        one_hot = torch.zeros((action_tensor.shape[0], self.action_space_size), device=action_tensor.device)
        one_hot.scatter_(1, action_tensor.view(-1, 1), 1.0)
        return one_hot

    def _device_of_parameters(self) -> torch.device:
        return next(self.parameters()).device
