from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.baba_ppo.config import ExperimentConfig
from experiments.baba_ppo.config import CurriculumStageConfig
from experiments.baba_ppo.envs import discover_level_paths
from experiments.baba_ppo.trainer import PPOTrainer


def build_first_run_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.env.train_level_dir = Path("experiments/baba_ppo/levels/train")
    config.env.eval_level_dir = Path("experiments/baba_ppo/levels/test")
    config.env.level_sampling = "random"
    config.env.max_steps = 192
    config.env.stuck_visit_limit = 6
    config.env.num_envs = 32

    config.ppo.total_timesteps = 75_000_000
    config.ppo.rollout_steps = 128
    config.ppo.minibatch_size = 1024
    config.ppo.update_epochs = 4
    config.ppo.learning_rate = 1e-4
    config.ppo.ent_coef = 0.03
    config.ppo.target_kl = 0.01

    config.runtime.device = "auto"
    config.runtime.mixed_precision = True
    config.runtime.compile_model = False
    config.runtime.eval_every_updates = 10
    config.runtime.checkpoint_every_updates = 25
    config.runtime.num_eval_episodes = 60
    config.runtime.eval_batch_size = 8

    config.curriculum.enabled = True
    config.curriculum.promotion_threshold = 0.8
    config.curriculum.consecutive_evals = 5
    config.curriculum.min_updates_per_stage = 10
    config.curriculum.stages = [
        CurriculumStageConfig(
            name="stage_1_navigation_basics",
            level_filenames=[
                "01_reach_flag.txt",
                "02_reach_rock.txt",
                "03_reach_rocket.txt",
                "04_diagonal_rock.txt",
            ],
        ),
        CurriculumStageConfig(
            name="stage_2_alternate_you",
            level_filenames=[
                "05_rock_is_you_maze.txt",
                "06_flag_is_you_maze.txt",
                "07_rocket_is_you_maze.txt",
            ],
        ),
        CurriculumStageConfig(
            name="stage_3_rule_building_intro",
            level_filenames=[
                "08_form_flag_win.txt",
                "09_form_rock_win.txt",
                "10_form_rocket_win.txt",
                "11_wall_maze_form_flag.txt",
                "12_defeat_form_flag.txt",
                "13_wall_maze_form_rock.txt",
            ],
        ),
        CurriculumStageConfig(
            name="stage_4_push_and_compose",
            level_filenames=[
                "14_rock_push_and_form_flag.txt",
                "15_wall_push_and_form_flag.txt",
                "16_rock_push_skull_form_flag.txt",
                "17_rocket_you_form_flag.txt",
                "18_flag_you_form_rock.txt",
            ],
        ),
        CurriculumStageConfig(
            name="stage_5_full_solvable_mix",
            level_filenames=[
                "01_reach_flag.txt",
                "02_reach_rock.txt",
                "03_reach_rocket.txt",
                "04_diagonal_rock.txt",
                "05_rock_is_you_maze.txt",
                "06_flag_is_you_maze.txt",
                "07_rocket_is_you_maze.txt",
                "08_form_flag_win.txt",
                "09_form_rock_win.txt",
                "10_form_rocket_win.txt",
                "11_wall_maze_form_flag.txt",
                "12_defeat_form_flag.txt",
                "13_wall_maze_form_rock.txt",
                "14_rock_push_and_form_flag.txt",
                "15_wall_push_and_form_flag.txt",
                "16_rock_push_skull_form_flag.txt",
                "17_rocket_you_form_flag.txt",
                "18_flag_you_form_rock.txt",
            ],
        ),
    ]
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
    print("- sampling happens within the currently active curriculum stage, not across every level at once")
    print("- the 192-step horizon gives extra room for rule-building and exploration compared with the base 160-step default")
    if config.curriculum.enabled and config.curriculum.stages:
        print()
        print("Curriculum plan")
        print(
            "- stage promotion requires every level in the active stage to reach win_rate >= "
            f"{config.curriculum.promotion_threshold:.2f} for "
            f"{config.curriculum.consecutive_evals} consecutive evaluations"
        )
        print(f"- minimum updates before promotion: {config.curriculum.min_updates_per_stage}")
        for index, stage in enumerate(config.curriculum.stages, start=1):
            print(f"  {index}. {stage.name}: {len(stage.level_filenames)} level(s)")
        print("- unwinnable levels 19 and 20 are excluded from the promotion curriculum")


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