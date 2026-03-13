from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parent


@dataclass(slots=True)
class EnvConfig:
    train_level_dir: Path = field(
        default_factory=lambda: EXPERIMENT_ROOT / "levels" / "train"
    )
    eval_level_dir: Path = field(
        default_factory=lambda: EXPERIMENT_ROOT / "levels" / "test"
    )
    level_sampling: str = "random"
    max_steps: int = 160
    step_penalty: float = -0.01
    win_reward: float = 5.0
    loss_reward: float = -1.0
    stuck_visit_limit: int = 3
    num_envs: int = 16


@dataclass(slots=True)
class ModelConfig:
    stem_channels: int = 128
    transformer_dim: int = 512
    num_heads: int = 8
    num_layers: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass(slots=True)
class PPOConfig:
    total_timesteps: int = 2_000_000
    rollout_steps: int = 128
    update_epochs: int = 6
    minibatch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    anneal_lr: bool = True

    @property
    def batch_size(self) -> int:
        return self.rollout_steps


@dataclass(slots=True)
class RuntimeConfig:
    seed: int = 7
    device: str = "auto"
    mixed_precision: bool = True
    compile_model: bool = False
    log_every_updates: int = 1
    eval_every_updates: int = 10
    checkpoint_every_updates: int = 10
    num_eval_episodes: int = 32
    eval_batch_size: int = 8


@dataclass(slots=True)
class CurriculumStageConfig:
    name: str
    level_filenames: list[str]


@dataclass(slots=True)
class CurriculumConfig:
    enabled: bool = False
    promotion_threshold: float = 0.8
    consecutive_evals: int = 2
    min_updates_per_stage: int = 10
    stages: list[CurriculumStageConfig] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    def to_dict(self) -> dict[str, Any]:
        return _convert_paths(asdict(self))


def _convert_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _convert_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_convert_paths(item) for item in value]
    return value
