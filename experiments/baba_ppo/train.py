from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.baba_ppo.config import ExperimentConfig
from experiments.baba_ppo.trainer import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent on Baba Is Auto.")
    parser.add_argument("--train-level-dir", type=Path, default=None)
    parser.add_argument("--eval-level-dir", type=Path, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--disable-mixed-precision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig()

    if args.train_level_dir is not None:
        config.env.train_level_dir = args.train_level_dir
    if args.eval_level_dir is not None:
        config.env.eval_level_dir = args.eval_level_dir
    if args.total_timesteps is not None:
        config.ppo.total_timesteps = args.total_timesteps
    if args.num_envs is not None:
        config.env.num_envs = args.num_envs
    if args.rollout_steps is not None:
        config.ppo.rollout_steps = args.rollout_steps
    if args.minibatch_size is not None:
        config.ppo.minibatch_size = args.minibatch_size
    if args.learning_rate is not None:
        config.ppo.learning_rate = args.learning_rate
    if args.seed is not None:
        config.runtime.seed = args.seed
    if args.device is not None:
        config.runtime.device = args.device
    if args.compile_model:
        config.runtime.compile_model = True
    if args.disable_mixed_precision:
        config.runtime.mixed_precision = False

    trainer = PPOTrainer(config)
    best_checkpoint = trainer.train()
    print(f"Best checkpoint saved to {best_checkpoint}")


if __name__ == "__main__":
    main()
