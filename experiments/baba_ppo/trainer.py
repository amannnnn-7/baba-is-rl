from __future__ import annotations

import json
import random
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim
from tqdm.auto import tqdm

from experiments.baba_ppo.config import EXPERIMENT_ROOT, ExperimentConfig
from experiments.baba_ppo.envs import (
    discover_level_paths,
    make_batched_env,
    resolve_pad_shape,
)
from experiments.baba_ppo.models import SpatialTransformerActorCritic
from experiments.baba_ppo.ppo import PPOUpdateStats, ppo_update
from experiments.baba_ppo.storage import RolloutStorage


ACTION_NAMES = ("up", "down", "left", "right")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def evaluate_policy(
    policy: SpatialTransformerActorCritic,
    config: ExperimentConfig,
    *,
    checkpoint_step: int,
    metrics_path: Path | None = None,
    show_progress: bool = True,
) -> dict[str, float]:
    eval_level_dir = config.env.eval_level_dir
    try:
        level_paths = discover_level_paths(eval_level_dir)
    except FileNotFoundError:
        level_paths = discover_level_paths(config.env.train_level_dir)

    train_levels = discover_level_paths(config.env.train_level_dir)
    pad_shape = resolve_pad_shape(train_levels + level_paths)
    batch_size = min(config.runtime.eval_batch_size, config.runtime.num_eval_episodes)
    eval_env = make_batched_env(
        level_paths,
        config=config.env,
        num_envs=batch_size,
        pad_to_shape=pad_shape,
        level_sampling="cycle",
    )

    observations, _ = eval_env.reset(seed=config.runtime.seed + checkpoint_step)
    returns: list[float] = []
    lengths: list[int] = []
    wins = 0
    stucks = 0
    progress = tqdm(
        total=config.runtime.num_eval_episodes,
        desc="Eval episodes",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )

    policy.eval()
    while len(returns) < config.runtime.num_eval_episodes:
        obs_tensor = torch.as_tensor(observations, device=next(policy.parameters()).device)
        with torch.no_grad():
            logits, _ = policy(obs_tensor)
            actions = logits.argmax(dim=-1)

        observations, _, _, _, infos = eval_env.step(actions.cpu().numpy())
        for info in infos:
            if "episode" not in info:
                continue
            final_info = info.get("final_info", info)
            episode_return = float(info["episode"]["r"])
            episode_length = int(info["episode"]["l"])
            returns.append(float(info["episode"]["r"]))
            lengths.append(int(info["episode"]["l"]))
            wins += int(final_info.get("play_state") == "WON")
            stucks += int(final_info.get("is_stuck", False))
            if metrics_path is not None:
                append_jsonl(
                    metrics_path,
                    {
                        "global_step": checkpoint_step,
                        "level_path": final_info.get("level_path"),
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "num_actions": episode_length,
                        "play_state": final_info.get("play_state"),
                        "is_stuck": bool(final_info.get("is_stuck", False)),
                        "state_visit_count": int(final_info.get("state_visit_count", 0)),
                        "width": final_info.get("width"),
                        "height": final_info.get("height"),
                    },
                )
            progress.update(1)
            progress.set_postfix(
                mean_return=f"{np.mean(returns):.2f}",
                win_rate=f"{wins / len(returns):.2f}",
            )
            if len(returns) >= config.runtime.num_eval_episodes:
                break

    eval_env.close()
    progress.close()
    return {
        "step": float(checkpoint_step),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "win_rate": wins / len(returns),
        "stuck_rate": stucks / len(returns),
    }


class PPOTrainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = resolve_device(config.runtime.device)
        self.output_root = EXPERIMENT_ROOT
        self.checkpoint_dir = self.output_root / "checkpoints"
        self.log_dir = self.output_root / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.train_metrics_path = self.log_dir / "train_metrics.jsonl"
        self.episode_metrics_path = self.log_dir / "episode_metrics.jsonl"
        self.eval_metrics_path = self.log_dir / "eval_metrics.jsonl"
        self.eval_episode_metrics_path = self.log_dir / "eval_episode_metrics.jsonl"

        set_global_seed(config.runtime.seed)
        torch.backends.cudnn.benchmark = self.device.type == "cuda"

        self.train_levels = discover_level_paths(config.env.train_level_dir)
        try:
            self.eval_levels = discover_level_paths(config.env.eval_level_dir)
        except FileNotFoundError:
            self.eval_levels = self.train_levels

        self.pad_shape = resolve_pad_shape(self.train_levels + self.eval_levels)
        self.train_env = make_batched_env(
            self.train_levels,
            config=config.env,
            num_envs=config.env.num_envs,
            pad_to_shape=self.pad_shape,
            level_sampling=config.env.level_sampling,
        )

        observation_shape = self.train_env.single_observation_space.shape
        num_actions = self.train_env.single_action_space.n
        self.policy = SpatialTransformerActorCritic(
            observation_shape,
            num_actions,
            config.model,
        ).to(self.device)
        if config.runtime.compile_model and hasattr(torch, "compile"):
            self.policy = torch.compile(self.policy)

        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config.ppo.learning_rate,
            weight_decay=config.ppo.weight_decay,
            eps=1e-5,
        )

    def autocast_context(self):
        if self.device.type != "cuda" or not self.config.runtime.mixed_precision:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def train(self) -> Path:
        rollout_batch_size = self.config.env.num_envs * self.config.ppo.rollout_steps
        total_updates = self.config.ppo.total_timesteps // rollout_batch_size
        if total_updates <= 0:
            raise ValueError("total_timesteps must be at least one rollout batch.")

        observations, _ = self.train_env.reset(seed=self.config.runtime.seed)
        next_dones = torch.zeros(self.config.env.num_envs, dtype=torch.float32, device=self.device)
        recent_episodes: deque[dict[str, Any]] = deque(maxlen=100)
        best_eval_return = float("-inf")
        best_checkpoint = self.checkpoint_dir / "best.pt"

        config_path = self.log_dir / "config.json"
        config_path.write_text(json.dumps(self.config.to_dict(), indent=2), encoding="utf-8")

        global_step = 0
        progress = tqdm(
            range(1, total_updates + 1),
            total=total_updates,
            desc="PPO updates",
            dynamic_ncols=True,
        )
        for update in progress:
            if self.config.ppo.anneal_lr:
                frac = 1.0 - (update - 1) / total_updates
                self.optimizer.param_groups[0]["lr"] = frac * self.config.ppo.learning_rate

            storage = RolloutStorage(
                self.config.ppo.rollout_steps,
                self.config.env.num_envs,
                self.train_env.single_observation_space.shape,
                self.device,
            )

            for step in range(self.config.ppo.rollout_steps):
                obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    with self.autocast_context():
                        actions, log_probs, _, values = self.policy.get_action_and_value(obs_tensor)

                next_observations, rewards, terminateds, truncateds, infos = self.train_env.step(
                    actions.cpu().numpy()
                )
                done_flags = np.logical_or(terminateds, truncateds)
                reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
                done_tensor = torch.as_tensor(done_flags, dtype=torch.float32, device=self.device)

                storage.add(
                    step,
                    obs_tensor,
                    actions,
                    log_probs,
                    reward_tensor,
                    done_tensor,
                    values,
                )

                next_global_step = global_step + self.config.env.num_envs
                for env_index, info in enumerate(infos):
                    episode = info.get("episode")
                    if episode is not None:
                        episode_record = self._build_episode_metrics(
                            global_step=next_global_step,
                            update=update,
                            rollout_step=step,
                            env_index=env_index,
                            info=info,
                            terminated=bool(terminateds[env_index]),
                            truncated=bool(truncateds[env_index]),
                        )
                        recent_episodes.append(episode_record)
                        append_jsonl(self.episode_metrics_path, episode_record)

                observations = next_observations
                next_dones = done_tensor
                global_step = next_global_step

            with torch.no_grad():
                last_values = self.policy.get_value(
                    torch.as_tensor(observations, dtype=torch.float32, device=self.device)
                )

            storage.compute_returns_and_advantages(
                last_values,
                next_dones,
                gamma=self.config.ppo.gamma,
                gae_lambda=self.config.ppo.gae_lambda,
            )

            update_stats = ppo_update(
                self.policy,
                self.optimizer,
                storage,
                self.config.ppo,
                autocast_factory=self.autocast_context,
            )

            metrics = self._build_metrics(
                global_step,
                update,
                recent_episodes,
                update_stats,
                storage,
            )
            append_jsonl(self.train_metrics_path, metrics)
            progress.set_postfix(
                reward=f"{metrics['mean_episode_return']:.2f}",
                win=f"{metrics['recent_win_rate']:.2f}",
                stuck=f"{metrics['recent_stuck_rate']:.2f}",
                kl=f"{metrics['approx_kl']:.4f}",
            )

            if update % self.config.runtime.log_every_updates == 0:
                self._print_metrics(metrics)

            if update % self.config.runtime.checkpoint_every_updates == 0:
                self.save_checkpoint(self.checkpoint_dir / f"update_{update:05d}.pt", global_step)

            if update % self.config.runtime.eval_every_updates == 0 or update == total_updates:
                eval_metrics = evaluate_policy(
                    self.policy,
                    self.config,
                    checkpoint_step=global_step,
                    metrics_path=self.eval_episode_metrics_path,
                )
                append_jsonl(self.eval_metrics_path, eval_metrics)
                print(
                    "[eval] "
                    f"step={global_step} "
                    f"mean_return={eval_metrics['mean_return']:.3f} "
                    f"win_rate={eval_metrics['win_rate']:.3f} "
                    f"stuck_rate={eval_metrics['stuck_rate']:.3f}"
                )
                if eval_metrics["mean_return"] >= best_eval_return:
                    best_eval_return = eval_metrics["mean_return"]
                    self.save_checkpoint(best_checkpoint, global_step)

        progress.close()
        self.train_env.close()
        return best_checkpoint

    def save_checkpoint(self, checkpoint_path: Path, global_step: int) -> None:
        payload = {
            "global_step": global_step,
            "config": self.config.to_dict(),
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, checkpoint_path)

    def _build_metrics(
        self,
        global_step: int,
        update: int,
        recent_episodes: deque[dict[str, Any]],
        update_stats: PPOUpdateStats,
        storage: RolloutStorage,
    ) -> dict[str, Any]:
        if recent_episodes:
            mean_return = float(np.mean([episode["episode_return"] for episode in recent_episodes]))
            mean_length = float(np.mean([episode["episode_length"] for episode in recent_episodes]))
            recent_win_rate = float(np.mean([episode["play_state"] == "WON" for episode in recent_episodes]))
            recent_stuck_rate = float(np.mean([episode["is_stuck"] for episode in recent_episodes]))
            recent_loss_rate = float(np.mean([episode["play_state"] == "LOST" for episode in recent_episodes]))
        else:
            mean_return = 0.0
            mean_length = 0.0
            recent_win_rate = 0.0
            recent_stuck_rate = 0.0
            recent_loss_rate = 0.0

        rewards = storage.rewards.detach().cpu().numpy()
        values = storage.values.detach().cpu().numpy()
        advantages = storage.advantages.detach().cpu().numpy()
        returns = storage.returns.detach().cpu().numpy()
        dones = storage.dones.detach().cpu().numpy()
        actions = storage.actions.detach().cpu().numpy()
        action_counts = {
            name: int(np.sum(actions == index)) for index, name in enumerate(ACTION_NAMES)
        }
        action_fractions = {
            f"{name}_fraction": action_counts[name] / actions.size for name in ACTION_NAMES
        }

        return {
            "global_step": global_step,
            "update": update,
            "completed_episodes_window": len(recent_episodes),
            "mean_episode_return": mean_return,
            "mean_episode_length": mean_length,
            "recent_win_rate": recent_win_rate,
            "recent_stuck_rate": recent_stuck_rate,
            "recent_loss_rate": recent_loss_rate,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_min": float(rewards.min()),
            "reward_max": float(rewards.max()),
            "value_mean": float(values.mean()),
            "value_std": float(values.std()),
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std()),
            "return_target_mean": float(returns.mean()),
            "return_target_std": float(returns.std()),
            "done_fraction": float(dones.mean()),
            "policy_loss": update_stats.policy_loss,
            "value_loss": update_stats.value_loss,
            "entropy": update_stats.entropy,
            "approx_kl": update_stats.approx_kl,
            "clip_fraction": update_stats.clip_fraction,
            "learning_rate": update_stats.learning_rate,
            "action_counts": action_counts,
            **action_fractions,
        }

    @staticmethod
    def _build_episode_metrics(
        *,
        global_step: int,
        update: int,
        rollout_step: int,
        env_index: int,
        info: dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        episode = info["episode"]
        final_info = info.get("final_info", info)
        play_state = final_info.get("play_state")
        is_stuck = bool(final_info.get("is_stuck", False))

        if is_stuck:
            end_reason = "stuck"
        elif play_state == "WON":
            end_reason = "won"
        elif play_state == "LOST":
            end_reason = "lost"
        elif truncated:
            end_reason = "truncated"
        else:
            end_reason = "terminated"

        episode_length = int(episode["l"])
        episode_return = float(episode["r"])
        return {
            "global_step": global_step,
            "update": update,
            "rollout_step": rollout_step,
            "env_index": env_index,
            "level_path": final_info.get("level_path"),
            "width": final_info.get("width"),
            "height": final_info.get("height"),
            "episode_return": episode_return,
            "episode_length": episode_length,
            "num_actions": episode_length,
            "avg_reward_per_action": episode_return / max(episode_length, 1),
            "play_state": play_state,
            "player_icon": final_info.get("player_icon"),
            "player_icon_name": final_info.get("player_icon_name"),
            "is_stuck": is_stuck,
            "state_changed": final_info.get("state_changed"),
            "state_visit_count": final_info.get("state_visit_count"),
            "terminated": terminated,
            "truncated": truncated,
            "end_reason": end_reason,
        }

    @staticmethod
    def _print_metrics(metrics: dict[str, Any]) -> None:
        print(
            "[train] "
            f"step={metrics['global_step']} "
            f"update={metrics['update']} "
            f"return={metrics['mean_episode_return']:.3f} "
            f"length={metrics['mean_episode_length']:.2f} "
            f"win={metrics['recent_win_rate']:.3f} "
            f"stuck={metrics['recent_stuck_rate']:.3f} "
            f"policy_loss={metrics['policy_loss']:.4f} "
            f"value_loss={metrics['value_loss']:.4f} "
            f"entropy={metrics['entropy']:.4f} "
            f"kl={metrics['approx_kl']:.5f}"
        )
