from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.baba_ppo.config import ExperimentConfig
from experiments.baba_ppo.envs import discover_level_paths
from experiments.baba_ppo.trainer import PPOTrainer


def build_first_run_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.env.train_level_dir = Path("experiments/baba_ppo/levels/train")
    config.env.eval_level_dir = Path("experiments/baba_ppo/levels/test")
    config.env.level_sampling = "random"
    config.env.max_steps = 192
    config.env.stuck_visit_limit = 4
    config.env.num_envs = 32

    config.ppo.total_timesteps = 5_000_000
    config.ppo.rollout_steps = 128
    config.ppo.minibatch_size = 1024
    config.ppo.update_epochs = 6
    config.ppo.learning_rate = 3e-4

    config.runtime.device = "auto"
    config.runtime.mixed_precision = True
    config.runtime.compile_model = False
    config.runtime.eval_every_updates = 25
    config.runtime.checkpoint_every_updates = 25
    config.runtime.num_eval_episodes = 40
    config.runtime.eval_batch_size = 8
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the first PPO training run for the Baba curriculum."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override the total PPO environment steps.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the per-episode horizon.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override the number of parallel environments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned budget without starting training.",
    )
    return parser.parse_args()


def print_training_budget(config: ExperimentConfig) -> None:
    train_levels = discover_level_paths(config.env.train_level_dir)
    num_levels = len(train_levels)
    total_timesteps = config.ppo.total_timesteps
    max_steps = config.env.max_steps
    num_envs = config.env.num_envs

    min_total_episodes = math.ceil(total_timesteps / max_steps)
    min_episodes_per_level = min_total_episodes / num_levels
    avg_96_total = math.ceil(total_timesteps / 96)
    avg_64_total = math.ceil(total_timesteps / 64)

    print("First PPO run preset")
    print(f"- training levels: {num_levels}")
    print(f"- parallel environments: {num_envs}")
    print(f"- total timesteps: {total_timesteps:,}")
    print(f"- max steps per episode: {max_steps}")
    print(f"- level sampling: {config.env.level_sampling}")
    print(f"- rollout batch size: {config.env.num_envs * config.ppo.rollout_steps:,}")
    print()
    print("Episode budget")
    print(
        "- worst case if every episode uses the full horizon: "
        f"{min_total_episodes:,} total episodes, about {min_episodes_per_level:,.0f} per level"
    )
    print(
        "- if the average episode length settles near 96 steps: "
        f"{avg_96_total:,} total episodes, about {avg_96_total / num_levels:,.0f} per level"
    )
    print(
        "- if the average episode length settles near 64 steps: "
        f"{avg_64_total:,} total episodes, about {avg_64_total / num_levels:,.0f} per level"
    )
    print()
    print("Interpretation")
    print("- episodes are not capped per level; the trainer uses a global timestep budget")
    print("- with random level sampling, each level is expected to receive roughly the same number of attempts")
    print("- the 192-step horizon gives extra room for rule-building and exploration compared with the base 160-step default")


def main() -> None:
    args = parse_args()
    config = build_first_run_config()

    if args.total_timesteps is not None:
        config.ppo.total_timesteps = args.total_timesteps
    if args.max_steps is not None:
        config.env.max_steps = args.max_steps
    if args.num_envs is not None:
        config.env.num_envs = args.num_envs

    print_training_budget(config)
    if args.dry_run:
        return

    trainer = PPOTrainer(config)
    best_checkpoint = trainer.train()
    print(f"Best checkpoint saved to {best_checkpoint}")


if __name__ == "__main__":
    main()