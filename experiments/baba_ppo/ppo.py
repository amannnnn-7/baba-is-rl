from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from experiments.baba_ppo.config import PPOConfig
from experiments.baba_ppo.storage import RolloutStorage


@dataclass(slots=True)
class PPOUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    learning_rate: float


def ppo_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    storage: RolloutStorage,
    config: PPOConfig,
    *,
    autocast_factory: Callable[[], object] | None = None,
) -> PPOUpdateStats:
    autocast_factory = autocast_factory or nullcontext
    policy.train()

    flat_advantages = storage.advantages.reshape(-1)
    normalized_advantages = (flat_advantages - flat_advantages.mean()) / (
        flat_advantages.std(unbiased=False) + 1e-8
    )
    storage.advantages.copy_(normalized_advantages.reshape_as(storage.advantages))

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    clip_fractions: list[float] = []

    stop_early = False
    for _ in range(config.update_epochs):
        for batch in storage.iter_minibatches(config.minibatch_size):
            with autocast_factory():
                _, new_log_probs, entropy, new_values = policy.get_action_and_value(
                    batch["observations"], batch["actions"]
                )
                log_ratio = new_log_probs - batch["log_probs"]
                ratio = log_ratio.exp()

                unclipped = -batch["advantages"] * ratio
                clipped = -batch["advantages"] * torch.clamp(
                    ratio,
                    1.0 - config.clip_coef,
                    1.0 + config.clip_coef,
                )
                policy_loss = torch.max(unclipped, clipped).mean()

                if config.clip_vloss:
                    value_delta = new_values - batch["values"]
                    value_clipped = batch["values"] + value_delta.clamp(
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    value_loss_unclipped = (new_values - batch["returns"]).pow(2)
                    value_loss_clipped = (value_clipped - batch["returns"]).pow(2)
                    value_loss = 0.5 * torch.max(
                        value_loss_unclipped, value_loss_clipped
                    ).mean()
                else:
                    value_loss = 0.5 * (new_values - batch["returns"]).pow(2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    + config.vf_coef * value_loss
                    - config.ent_coef * entropy_loss
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > config.clip_coef
                ).float().mean()

            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy_loss.detach().cpu()))
            approx_kls.append(float(approx_kl.detach().cpu()))
            clip_fractions.append(float(clip_fraction.detach().cpu()))

            if config.target_kl is not None and float(approx_kl) > config.target_kl:
                stop_early = True
                break

        if stop_early:
            break

    learning_rate = optimizer.param_groups[0]["lr"]
    return PPOUpdateStats(
        policy_loss=float(np.mean(policy_losses)),
        value_loss=float(np.mean(value_losses)),
        entropy=float(np.mean(entropies)),
        approx_kl=float(np.mean(approx_kls)),
        clip_fraction=float(np.mean(clip_fractions)),
        learning_rate=float(learning_rate),
    )
