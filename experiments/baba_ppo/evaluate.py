from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.baba_ppo.config import ExperimentConfig
from experiments.baba_ppo.envs import discover_level_paths, resolve_pad_shape
from experiments.baba_ppo.models import SpatialTransformerActorCritic
from experiments.baba_ppo.trainer import evaluate_policy, resolve_device


def apply_checkpoint_config(
    config: ExperimentConfig,
    checkpoint_config: dict[str, object] | None,
) -> None:
    if not checkpoint_config:
        return

    for section_name in ("env", "model", "ppo", "runtime", "curriculum"):
        section_values = checkpoint_config.get(section_name)
        section = getattr(config, section_name)
        if not isinstance(section_values, dict):
            continue
        for key, value in section_values.items():
            if hasattr(section, key):
                current_value = getattr(section, key)
                if isinstance(current_value, Path):
                    setattr(section, key, Path(value))
                else:
                    setattr(section, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint on Baba Is Auto.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-level-dir", type=Path, default=None)
    parser.add_argument("--eval-level-dir", type=Path, default=None)
    parser.add_argument("--num-eval-episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    apply_checkpoint_config(config, checkpoint.get("config"))

    if args.train_level_dir is not None:
        config.env.train_level_dir = args.train_level_dir
    if args.eval_level_dir is not None:
        config.env.eval_level_dir = args.eval_level_dir
    if args.num_eval_episodes is not None:
        config.runtime.num_eval_episodes = args.num_eval_episodes
    if args.device is not None:
        config.runtime.device = args.device

    train_levels = discover_level_paths(config.env.train_level_dir)
    try:
        eval_levels = discover_level_paths(config.env.eval_level_dir)
    except FileNotFoundError:
        eval_levels = train_levels

    device = resolve_device(config.runtime.device)

    pad_shape = resolve_pad_shape(train_levels + eval_levels)

    from Extensions.BabaRL import BabaIsAutoEnv

    sample_env = BabaIsAutoEnv(
        level_paths=train_levels,
        pad_to_shape=pad_shape,
        max_steps=config.env.max_steps,
    )
    policy = SpatialTransformerActorCritic(
        sample_env.observation_space.shape,
        sample_env.action_space.n,
        config.model,
    ).to(device)
    sample_env.close()

    policy.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_policy(policy, config, checkpoint_step=int(checkpoint.get("global_step", 0)))
    print(metrics)


if __name__ == "__main__":
    main()
