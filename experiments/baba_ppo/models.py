from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from experiments.baba_ppo.config import ModelConfig


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.block(inputs))


class SpatialTransformerActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        num_actions: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        channels, height, width = observation_shape
        d_model = config.transformer_dim

        self.height = height
        self.width = width
        self.num_actions = num_actions

        self.stem = nn.Sequential(
            nn.Conv2d(channels, config.stem_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, config.stem_channels),
            nn.GELU(),
            ResidualConvBlock(config.stem_channels),
            ResidualConvBlock(config.stem_channels),
            ResidualConvBlock(config.stem_channels),
            nn.Conv2d(config.stem_channels, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dim_feedforward=int(d_model * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.final_norm = nn.LayerNorm(d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.row_embedding = nn.Parameter(torch.zeros(1, height, d_model))
        self.col_embedding = nn.Parameter(torch.zeros(1, width, d_model))

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model, 1),
        )

        self.apply(_init_weights)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.row_embedding, std=0.02)
        nn.init.normal_(self.col_embedding, std=0.02)

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.stem(observations.float())
        batch_size = hidden.shape[0]

        tokens = hidden.permute(0, 2, 3, 1).reshape(batch_size, self.height * self.width, -1)
        position = (
            self.row_embedding[:, :, None, :] + self.col_embedding[:, None, :, :]
        ).reshape(1, self.height * self.width, -1)
        cls_token = self.cls_token.expand(batch_size, -1, -1)

        sequence = torch.cat((cls_token, tokens + position), dim=1)
        encoded = self.encoder(sequence)
        pooled = self.final_norm(encoded[:, 0])
        logits = self.policy_head(pooled)
        values = self.value_head(pooled).squeeze(-1)
        return logits, values

    def get_value(self, observations: Tensor) -> Tensor:
        _, values = self(observations)
        return values

    def get_action_and_value(
        self,
        observations: Tensor,
        actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits, values = self(observations)
        distribution = Categorical(logits=logits)
        if actions is None:
            actions = distribution.sample()
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return actions, log_prob, entropy, values


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
