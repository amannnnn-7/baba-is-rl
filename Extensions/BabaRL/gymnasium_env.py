from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pyBaba

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LEVEL = _REPO_ROOT / "Resources" / "Maps" / "baba_is_you.txt"
_SENTINEL_OBJECT_NAMES = {"NOUN_TYPE", "OP_TYPE", "PROPERTY_TYPE", "ICON_TYPE"}
_OBJECT_TYPE_MEMBERS = pyBaba.ObjectType.__members__
_OBSERVATION_OBJECTS = tuple(
    member
    for name, member in _OBJECT_TYPE_MEMBERS.items()
    if name not in _SENTINEL_OBJECT_NAMES
)
_OBJECT_TO_CHANNEL = {int(member): idx for idx, member in enumerate(_OBSERVATION_OBJECTS)}
_OBJECT_CHANNEL_NAMES = tuple(
    name for name in _OBJECT_TYPE_MEMBERS if name not in _SENTINEL_OBJECT_NAMES
)
_ACTIONS = (
    pyBaba.Direction.UP,
    pyBaba.Direction.DOWN,
    pyBaba.Direction.LEFT,
    pyBaba.Direction.RIGHT,
)
_ACTION_MEANINGS = ("up", "down", "left", "right")


class BabaIsAutoEnv(gym.Env[np.ndarray, int]):
    """Thin Gymnasium wrapper around `pyBaba.Game`.

    The wrapper keeps the simulator on CPU and exposes a modern Gymnasium API.
    Observations are multi-hot tensors with one channel per `pyBaba.ObjectType`
    value (excluding enum sentinels), padded to a fixed `(height, width)` when
    multiple training levels are used.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        level_paths: Sequence[str | Path] | None = None,
        *,
        level_sampling: str = "cycle",
        max_steps: int = 200,
        step_reward: float = -0.5,
        win_reward: float = 200.0,
        loss_reward: float = -100.0,
        pad_to_shape: tuple[int, int] | None = None,
        render_mode: str | None = None,
        dtype: np.dtype | type[np.floating[Any]] = np.float32,
    ) -> None:
        if render_mode is not None:
            raise ValueError("This wrapper is headless. Use render_mode=None.")
        if level_sampling not in {"cycle", "random"}:
            raise ValueError("level_sampling must be 'cycle' or 'random'.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")

        self.level_paths = self._normalize_level_paths(level_paths)
        self.level_sampling = level_sampling
        self.max_steps = max_steps
        self.step_reward = float(step_reward)
        self.win_reward = float(win_reward)
        self.loss_reward = float(loss_reward)
        self.dtype = np.dtype(dtype)

        self._level_shapes = {path: self._read_level_shape(path) for path in self.level_paths}
        self._target_height, self._target_width = self._resolve_target_shape(pad_to_shape)
        self._cycle_index = 0
        self._elapsed_steps = 0
        self._game: pyBaba.Game | None = None
        self._current_level_path: Path | None = None

        self.action_space = spaces.Discrete(len(_ACTIONS))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(_OBSERVATION_OBJECTS), self._target_height, self._target_width),
            dtype=self.dtype,
        )

    @property
    def current_level_path(self) -> Path | None:
        return self._current_level_path

    @property
    def game(self) -> pyBaba.Game | None:
        return self._game

    @staticmethod
    def action_meanings() -> tuple[str, ...]:
        return _ACTION_MEANINGS

    @staticmethod
    def observation_channel_names() -> tuple[str, ...]:
        return _OBJECT_CHANNEL_NAMES

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        level_path = self._resolve_reset_level_path(options)
        self._game = pyBaba.Game(str(level_path))
        self._current_level_path = level_path
        self._elapsed_steps = 0

        observation = self._get_observation()
        info = self._build_info()
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._game is None:
            raise RuntimeError("Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Expected [0, {self.action_space.n - 1}].")

        self._game.MovePlayer(_ACTIONS[action])
        self._elapsed_steps += 1

        play_state = self._game.GetPlayState()
        terminated = play_state in {pyBaba.PlayState.WON, pyBaba.PlayState.LOST}
        truncated = not terminated and self._elapsed_steps >= self.max_steps

        if play_state == pyBaba.PlayState.WON:
            reward = self.win_reward
        elif play_state == pyBaba.PlayState.LOST:
            reward = self.loss_reward
        else:
            reward = self.step_reward

        observation = self._get_observation()
        info = self._build_info()
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _get_observation(self) -> np.ndarray:
        if self._game is None:
            raise RuntimeError("The environment has not been reset yet.")

        game_map = self._game.GetMap()
        width = game_map.GetWidth()
        height = game_map.GetHeight()

        if height > self._target_height or width > self._target_width:
            raise ValueError(
                f"Level shape {(height, width)} exceeds pad_to_shape {(self._target_height, self._target_width)}."
            )

        observation = np.zeros(self.observation_space.shape, dtype=self.dtype)

        for y_pos in range(height):
            for x_pos in range(width):
                for obj_type in game_map.At(x_pos, y_pos).GetTypes():
                    channel = _OBJECT_TO_CHANNEL.get(int(obj_type))
                    if channel is not None:
                        observation[channel, y_pos, x_pos] = 1.0

        return observation

    def _build_info(self) -> dict[str, Any]:
        if self._game is None or self._current_level_path is None:
            return {}

        game_map = self._game.GetMap()
        return {
            "level_path": str(self._current_level_path),
            "width": game_map.GetWidth(),
            "height": game_map.GetHeight(),
            "elapsed_steps": self._elapsed_steps,
            "player_icon": int(self._game.GetPlayerIcon()),
            "player_icon_name": pyBaba.ConvertIconToText(self._game.GetPlayerIcon()).name,
            "play_state": self._game.GetPlayState().name,
        }

    def _resolve_reset_level_path(self, options: Mapping[str, Any] | None) -> Path:
        if options and "level_path" in options:
            requested = Path(options["level_path"]).expanduser()
            if not requested.is_absolute():
                requested = (_REPO_ROOT / requested).resolve()
            if not requested.exists():
                raise FileNotFoundError(f"Level file not found: {requested}")
            return requested

        if self.level_sampling == "random":
            index = int(self.np_random.integers(len(self.level_paths)))
            return self.level_paths[index]

        level_path = self.level_paths[self._cycle_index % len(self.level_paths)]
        self._cycle_index += 1
        return level_path

    def _resolve_target_shape(self, pad_to_shape: tuple[int, int] | None) -> tuple[int, int]:
        if pad_to_shape is not None:
            if len(pad_to_shape) != 2:
                raise ValueError("pad_to_shape must be a (height, width) tuple.")
            target_height, target_width = pad_to_shape
        else:
            target_width = max(width for width, _ in self._level_shapes.values())
            target_height = max(height for _, height in self._level_shapes.values())

        if target_height <= 0 or target_width <= 0:
            raise ValueError("Target observation shape must be positive.")

        for width, height in self._level_shapes.values():
            if height > target_height or width > target_width:
                raise ValueError("pad_to_shape is smaller than one of the provided levels.")

        return int(target_height), int(target_width)

    @staticmethod
    def _normalize_level_paths(level_paths: Sequence[str | Path] | None) -> list[Path]:
        raw_paths: Iterable[str | Path] = level_paths if level_paths else (_DEFAULT_LEVEL,)
        normalized: list[Path] = []

        for raw_path in raw_paths:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = (_REPO_ROOT / path).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Level file not found: {path}")
            normalized.append(path)

        return normalized

    @staticmethod
    def _read_level_shape(path: Path) -> tuple[int, int]:
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip().split()
        if len(first_line) != 2:
            raise ValueError(f"Invalid level header in {path}.")
        width, height = map(int, first_line)
        return width, height
