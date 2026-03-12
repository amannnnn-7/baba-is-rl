from __future__ import annotations

from typing import Iterator

import torch
from torch import Tensor


class RolloutStorage:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        observation_shape: tuple[int, ...],
        device: torch.device,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.observation_shape = observation_shape
        self.device = device

        self.observations = torch.zeros(
            (rollout_steps, num_envs, *observation_shape),
            dtype=torch.float32,
            device=device,
        )
        self.actions = torch.zeros(
            (rollout_steps, num_envs), dtype=torch.long, device=device
        )
        self.log_probs = torch.zeros(
            (rollout_steps, num_envs), dtype=torch.float32, device=device
        )
        self.rewards = torch.zeros(
            (rollout_steps, num_envs), dtype=torch.float32, device=device
        )
        self.dones = torch.zeros(
            (rollout_steps, num_envs), dtype=torch.float32, device=device
        )
        self.values = torch.zeros(
            (rollout_steps, num_envs), dtype=torch.float32, device=device
        )
        self.advantages = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)

    def add(
        self,
        step: int,
        observations: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        rewards: Tensor,
        dones: Tensor,
        values: Tensor,
    ) -> None:
        self.observations[step].copy_(observations)
        self.actions[step].copy_(actions)
        self.log_probs[step].copy_(log_probs)
        self.rewards[step].copy_(rewards)
        self.dones[step].copy_(dones)
        self.values[step].copy_(values)

    def compute_returns_and_advantages(
        self,
        last_values: Tensor,
        last_dones: Tensor,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        last_advantage = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones
            else:
                next_values = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]

            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            self.advantages[step] = last_advantage

        self.returns = self.advantages + self.values

    def iter_minibatches(self, minibatch_size: int) -> Iterator[dict[str, Tensor]]:
        batch_size = self.rollout_steps * self.num_envs
        if batch_size % minibatch_size != 0:
            raise ValueError(
                f"Minibatch size {minibatch_size} must divide the rollout batch size {batch_size}."
            )

        flat_observations = self.observations.reshape(batch_size, *self.observation_shape)
        flat_actions = self.actions.reshape(batch_size)
        flat_log_probs = self.log_probs.reshape(batch_size)
        flat_advantages = self.advantages.reshape(batch_size)
        flat_returns = self.returns.reshape(batch_size)
        flat_values = self.values.reshape(batch_size)

        permutation = torch.randperm(batch_size, device=self.device)
        for start in range(0, batch_size, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            yield {
                "observations": flat_observations[indices],
                "actions": flat_actions[indices],
                "log_probs": flat_log_probs[indices],
                "advantages": flat_advantages[indices],
                "returns": flat_returns[indices],
                "values": flat_values[indices],
            }
