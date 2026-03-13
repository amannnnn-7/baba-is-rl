from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import gymnasium as gym
import numpy as np
import pyBaba

from Extensions.BabaRL import BabaIsAutoEnv
from experiments.baba_ppo.config import EnvConfig


def discover_level_paths(level_dir: str | Path) -> list[Path]:
    level_dir = Path(level_dir)
    if not level_dir.exists():
        raise FileNotFoundError(f"Level directory does not exist: {level_dir}")

    level_paths = [
        path
        for path in sorted(level_dir.glob("*.txt"))
        if path.name.lower() != "readme.md"
    ]
    if not level_paths:
        raise FileNotFoundError(f"No compiled level files were found in {level_dir}")
    return level_paths


def read_level_shape(level_path: str | Path) -> tuple[int, int]:
    level_path = Path(level_path)
    with level_path.open("r", encoding="utf-8") as handle:
        width, height = map(int, handle.readline().split())
    return height, width


def resolve_pad_shape(level_paths: Sequence[str | Path]) -> tuple[int, int]:
    shapes = [read_level_shape(path) for path in level_paths]
    return max(height for height, _ in shapes), max(width for _, width in shapes)


def state_signature(game: pyBaba.Game) -> tuple[Any, ...]:
    game_map = game.GetMap()
    width = game_map.GetWidth()
    height = game_map.GetHeight()

    cells: list[tuple[int, ...]] = []
    for y_pos in range(height):
        for x_pos in range(width):
            cell_types = tuple(sorted(int(obj_type) for obj_type in game_map.At(x_pos, y_pos).GetTypes()))
            cells.append(cell_types)

    return (
        width,
        height,
        int(game.GetPlayerIcon()),
        int(game.GetPlayState()),
        tuple(cells),
    )


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env: BabaIsAutoEnv, config: EnvConfig) -> None:
        super().__init__(env)
        self.config = config
        self._state_signature: tuple[Any, ...] | None = None
        self._visit_counts: defaultdict[tuple[Any, ...], int] = defaultdict(int)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._state_signature = state_signature(self.unwrapped.game)
        self._visit_counts.clear()
        self._visit_counts[self._state_signature] = 1
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        previous_signature = self._state_signature
        observation, _, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        current_signature = state_signature(self.unwrapped.game)
        self._state_signature = current_signature
        self._visit_counts[current_signature] += 1

        state_changed = current_signature != previous_signature
        repeated_loop = self._visit_counts[current_signature] >= self.config.stuck_visit_limit
        stuck = repeated_loop

        reward = self.config.step_penalty
        if info.get("play_state") == pyBaba.PlayState.WON.name:
            reward = self.config.win_reward
        elif info.get("play_state") == pyBaba.PlayState.LOST.name:
            reward = self.config.loss_reward
        elif not terminated and not truncated and repeated_loop:
            reward = self.config.loss_reward
            terminated = True
            truncated = False
            info["is_stuck"] = True

        info["state_changed"] = state_changed
        info["state_visit_count"] = self._visit_counts[current_signature]
        return observation, reward, terminated, truncated, info


class BabaEnvBatch:
    def __init__(self, envs: Sequence[gym.Env[np.ndarray, int]]) -> None:
        self.envs = list(envs)
        if not self.envs:
            raise ValueError("At least one environment is required.")
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
        observations: list[np.ndarray] = []
        infos: list[dict[str, Any]] = []

        self._episode_returns.fill(0.0)
        self._episode_lengths.fill(0)

        for index, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + index
            observation, info = env.reset(seed=env_seed)
            observations.append(observation)
            infos.append(dict(info))

        return np.stack(observations), infos

    def step(
        self, actions: np.ndarray | Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observations: list[np.ndarray] = []
        rewards: list[float] = []
        terminateds: list[bool] = []
        truncateds: list[bool] = []
        infos: list[dict[str, Any]] = []

        for index, (env, action) in enumerate(zip(self.envs, actions)):
            observation, reward, terminated, truncated, info = env.step(int(action))
            info = dict(info)

            self._episode_returns[index] += float(reward)
            self._episode_lengths[index] += 1

            if terminated or truncated:
                info["episode"] = {
                    "r": float(self._episode_returns[index]),
                    "l": int(self._episode_lengths[index]),
                }
                final_observation = observation
                final_info = dict(info)
                observation, reset_info = env.reset()
                info["final_observation"] = final_observation
                info["final_info"] = final_info
                info["reset_info"] = dict(reset_info)
                self._episode_returns[index] = 0.0
                self._episode_lengths[index] = 0

            observations.append(observation)
            rewards.append(float(reward))
            terminateds.append(bool(terminated))
            truncateds.append(bool(truncated))
            infos.append(info)

        return (
            np.stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminateds, dtype=np.bool_),
            np.asarray(truncateds, dtype=np.bool_),
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


def make_env(
    level_paths: Sequence[str | Path],
    *,
    config: EnvConfig,
    pad_to_shape: tuple[int, int],
    level_sampling: str,
) -> gym.Env[np.ndarray, int]:
    base_env = BabaIsAutoEnv(
        level_paths=level_paths,
        level_sampling=level_sampling,
        max_steps=config.max_steps,
        step_reward=config.step_penalty,
        win_reward=config.win_reward,
        loss_reward=config.loss_reward,
        pad_to_shape=pad_to_shape,
    )
    return RewardShapingWrapper(base_env, config)


def make_batched_env(
    level_paths: Sequence[str | Path],
    *,
    config: EnvConfig,
    num_envs: int,
    pad_to_shape: tuple[int, int],
    level_sampling: str,
) -> BabaEnvBatch:
    envs = [
        make_env(
            level_paths,
            config=config,
            pad_to_shape=pad_to_shape,
            level_sampling=level_sampling,
        )
        for _ in range(num_envs)
    ]
    return BabaEnvBatch(envs)
