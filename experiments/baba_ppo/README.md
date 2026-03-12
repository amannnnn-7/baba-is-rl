# Baba PPO experiments

This folder contains a modular PPO training and evaluation pipeline built on top of `Extensions.BabaRL.BabaIsAutoEnv`.

## Layout

- `level_specs/train` and `level_specs/test`: natural-language level grids.
- `levels/train` and `levels/test`: compiled numeric maps consumed by `pyBaba`.
- `build_levels.py`: converts natural-language grids into numeric map files.
- `train.py`: PPO training entrypoint.
- `evaluate.py`: checkpoint evaluation entrypoint.

## Level workflow

1. Add natural-language level specs under `level_specs/train` or `level_specs/test`.
2. Run `python experiments/baba_ppo/build_levels.py` from the repository root.
3. Start training with `python experiments/baba_ppo/train.py`.
4. Evaluate a checkpoint with `python experiments/baba_ppo/evaluate.py --checkpoint experiments/baba_ppo/checkpoints/best.pt`.

## Natural-language grid format

Each non-empty line is a row. Tokens are whitespace-separated.

- Use object tokens such as `baba`, `wall`, `flag`, `rock`, `lava`, `skull`.
- Use text tokens such as `text_baba`, `text_wall`, `text_flag`.
- Use rule tokens such as `is`, `you`, `win`, `stop`, `push`, `sink`, `defeat`.
- Use `.` for `EMPTY`.

Example:

```text
wall wall wall wall wall
wall baba . flag wall
wall text_baba is you wall
wall text_flag is win wall
wall wall wall wall wall
```

## PPO defaults

- Reward shaping: `+5` win, `-1` loss/stuck, `-0.01` per step.
- Episode horizon: `160` steps.
- Policy: convolutional stem + deep spatial transformer actor-critic.
- CUDA: enabled automatically when available, with optional BF16 autocast.

## First training run

- Use `python experiments/baba_ppo/run_first_training.py --dry-run` to inspect the initial training budget.
- Use `python experiments/baba_ppo/run_first_training.py` to start the first PPO run.
- The starter preset uses a larger horizon and timestep budget than the base defaults so each curriculum level gets many attempts.
- Training and evaluation both show live `tqdm` progress bars.
- Logs are written as JSONL files in `experiments/baba_ppo/logs`:
	- `train_metrics.jsonl`: PPO update summaries
	- `episode_metrics.jsonl`: per-episode training records
	- `eval_metrics.jsonl`: evaluation summaries
	- `eval_episode_metrics.jsonl`: per-episode evaluation records
