#!/usr/bin/env python3
"""Train a PyTorch PPO high-level planner for the Go2 oval-track task.

The low-level Go2 locomotion policy is kept fixed.  PPO only trains the
high-level planner that maps the official 5D track observation to
``[vx, vy, yaw_rate]`` commands.

The simulator still runs through JAX/MJX because the starter low-level policy
and environment are JAX-based.  PyTorch is used for the high-level actor-critic
and PPO update.  The exported deployment planner is a NumPy ``.npz`` file, so
``run_track_bonus.py`` does not need PyTorch at evaluation time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from course_common import (
    DEFAULT_CONFIG_PATH,
    apply_stage_config,
    build_env_overrides,
    ensure_environment_available,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    set_runtime_env,
)
from test_policy import load_policy_with_workaround


ROOT = Path(__file__).resolve().parent
TRACK_LENGTH_M = 200.0
TURN_RADIUS_M = 18.25
HALF_WIDTH_M = 2.0
STRAIGHT_LENGTH_M = (TRACK_LENGTH_M - 2.0 * math.pi * TURN_RADIUS_M) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Fixed low-level Brax PPO checkpoint.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "highlevel_ppo_torch")
    parser.add_argument("--stage-name", choices=["stage_1", "stage_2"], default="stage_2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force-cpu", action="store_true")

    parser.add_argument("--total-updates", type=int, default=80)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.20)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--start-max-vx", type=float, default=1.25)
    parser.add_argument("--max-vx", type=float, default=2.50)
    parser.add_argument("--max-vy", type=float, default=0.30)
    parser.add_argument("--max-yaw-rate", type=float, default=0.80)
    parser.add_argument("--command-filter-alpha", type=float, default=0.38)
    parser.add_argument("--max-command-delta", type=float, default=0.18)
    parser.add_argument("--edge-slowdown-margin-norm", type=float, default=0.35)
    parser.add_argument("--stand-seconds", type=float, default=1.0)
    parser.add_argument("--max-episode-seconds", type=float, default=240.0)
    parser.add_argument("--start-target-straight-speed", type=float, default=1.10)
    parser.add_argument("--target-straight-speed", type=float, default=2.50)
    parser.add_argument("--start-target-curve-speed", type=float, default=0.70)
    parser.add_argument("--target-curve-speed", type=float, default=2.20)
    parser.add_argument("--speed-curriculum-updates", type=int, default=80)
    parser.add_argument("--speed-curriculum-warmup-updates", type=int, default=2)
    parser.add_argument("--progress-reward-scale", type=float, default=22.0)
    parser.add_argument("--speed-reward-scale", type=float, default=0.20)
    parser.add_argument("--target-speed-reward-scale", type=float, default=0.30)
    parser.add_argument("--curve-speed-reward-scale", type=float, default=0.20)
    parser.add_argument("--slow-penalty-scale", type=float, default=0.18)
    parser.add_argument("--backward-penalty-scale", type=float, default=0.40)
    parser.add_argument("--heading-speed-penalty", type=float, default=0.20)
    parser.add_argument("--lateral-speed-penalty", type=float, default=0.25)
    parser.add_argument("--edge-speed-penalty", type=float, default=0.35)
    parser.add_argument("--turn-speed-penalty", type=float, default=0.10)
    parser.add_argument("--min-speed-cap-scale", type=float, default=0.45)
    parser.add_argument("--use-racing-line", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-line-bias-norm", type=float, default=0.50)
    parser.add_argument("--line-vy-gain", type=float, default=0.22)
    parser.add_argument("--line-yaw-gain", type=float, default=0.28)
    parser.add_argument("--max-line-vy", type=float, default=0.18)
    parser.add_argument("--max-line-yaw", type=float, default=0.24)
    parser.add_argument("--line-lateral-weight", type=float, default=0.18)
    parser.add_argument("--center-lateral-weight", type=float, default=0.01)
    parser.add_argument("--line-bias-penalty", type=float, default=0.015)

    parser.add_argument("--start-randomization", choices=["fixed", "full_track", "curriculum"], default="curriculum")
    parser.add_argument("--start-s-m", type=float, default=0.0)
    parser.add_argument("--lateral-reset-std", type=float, default=0.15)
    parser.add_argument("--heading-reset-std", type=float, default=0.10)
    parser.add_argument("--eval-interval", type=int, default=10, help="Run no-render full evaluator every N updates. Use 0 to disable.")
    parser.add_argument("--eval-seconds", type=float, default=120.0)
    return parser.parse_args()


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, num_hidden_layers: int, action_dim: int = 3) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.actor_body = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(in_dim, self.action_dim)

        value_layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_hidden_layers):
            value_layers.append(nn.Linear(in_dim, hidden_dim))
            value_layers.append(nn.Tanh())
            in_dim = hidden_dim
        value_layers.append(nn.Linear(in_dim, 1))
        self.value_net = nn.Sequential(*value_layers)

        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.45))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        with torch.no_grad():
            self.actor_mean.bias[0].fill_(0.5)
            if self.action_dim > 3:
                self.actor_mean.bias[3:].zero_()

    def actor_forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(self.actor_body(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_net(obs).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return raw_action, log_prob, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return log_prob, entropy, value


def raw_action_to_command(raw_action: np.ndarray, *, max_vx: float, max_vy: float, max_yaw_rate: float) -> np.ndarray:
    raw_action = np.asarray(raw_action, dtype=np.float32)
    return np.stack(
        [
            0.5 * float(max_vx) * (np.tanh(raw_action[..., 0]) + 1.0),
            float(max_vy) * np.tanh(raw_action[..., 1]),
            float(max_yaw_rate) * np.tanh(raw_action[..., 2]),
        ],
        axis=-1,
    ).astype(np.float32)


def raw_action_to_line_bias(raw_action: np.ndarray, obs: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    raw_action = np.asarray(raw_action, dtype=np.float32)
    if not bool(args.use_racing_line) or raw_action.shape[-1] < 4:
        return np.zeros(raw_action.shape[0], dtype=np.float32)
    turn_gate = np.clip(np.abs(np.asarray(obs, dtype=np.float32)[:, 4]), 0.0, 1.0)
    bias = float(args.max_line_bias_norm) * np.tanh(raw_action[:, 3]) * turn_gate
    return bias.astype(np.float32)


def apply_racing_line_command_bias(
    command: np.ndarray,
    obs: np.ndarray,
    line_bias_norm: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if not bool(args.use_racing_line):
        return command
    command = np.asarray(command, dtype=np.float32).copy()
    line_error_norm = np.asarray(obs, dtype=np.float32)[:, 1] - np.asarray(line_bias_norm, dtype=np.float32)
    vy_bias = np.clip(
        -float(args.line_vy_gain) * line_error_norm,
        -float(args.max_line_vy),
        float(args.max_line_vy),
    )
    yaw_bias = np.clip(
        -float(args.line_yaw_gain) * line_error_norm,
        -float(args.max_line_yaw),
        float(args.max_line_yaw),
    )
    command[:, 1] += vy_bias
    command[:, 2] += yaw_bias
    return command.astype(np.float32)


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def speed_schedule(args: argparse.Namespace, update_idx: int | None) -> dict[str, float]:
    if update_idx is None:
        frac = 1.0
    else:
        warmup = int(max(args.speed_curriculum_warmup_updates, 0))
        duration = int(max(args.speed_curriculum_updates, 1))
        frac = smoothstep((int(update_idx) - warmup) / duration)

    def blend(start: float, final: float) -> float:
        return float(start + frac * (final - start))

    max_vx = blend(float(args.start_max_vx), float(args.max_vx))
    target_straight = blend(float(args.start_target_straight_speed), float(args.target_straight_speed))
    target_curve = blend(float(args.start_target_curve_speed), float(args.target_curve_speed))
    target_straight = min(target_straight, max_vx)
    target_curve = min(target_curve, max_vx)
    return {
        "fraction": float(frac),
        "max_vx": float(max_vx),
        "target_straight_speed": float(target_straight),
        "target_curve_speed": float(target_curve),
    }


def apply_stability_envelope(
    command: np.ndarray,
    obs: np.ndarray,
    args: argparse.Namespace,
    *,
    max_vx: float,
) -> np.ndarray:
    command = np.asarray(command, dtype=np.float32).copy()
    turn_intensity = np.clip(np.abs(obs[:, 4]), 0.0, 1.0)
    heading_risk = np.minimum(np.abs(obs[:, 3]) / 1.0, 1.0)
    lateral_risk = np.clip((np.abs(obs[:, 1]) - 0.35) / 0.65, 0.0, 1.0)
    edge_limit = max(float(args.edge_slowdown_margin_norm), 1e-6)
    edge_risk = np.clip((edge_limit - obs[:, 2]) / edge_limit, 0.0, 1.0)
    risk_scale = (
        1.0
        - float(args.heading_speed_penalty) * heading_risk
        - float(args.lateral_speed_penalty) * lateral_risk
        - float(args.edge_speed_penalty) * edge_risk
        - float(args.turn_speed_penalty) * turn_intensity
    )
    speed_cap = float(max_vx) * np.clip(risk_scale, float(args.min_speed_cap_scale), 1.0)
    command[:, 0] = np.clip(command[:, 0], 0.0, speed_cap)
    command[:, 1] = np.clip(command[:, 1], -float(args.max_vy), float(args.max_vy))
    command[:, 2] = np.clip(command[:, 2], -float(args.max_yaw_rate), float(args.max_yaw_rate))
    return command.astype(np.float32)


def jax_wrap_angle(angle: Any, jnp: Any) -> Any:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def make_jax_track_fns(jnp: Any) -> tuple[Any, Any]:
    def centerline_pose(s_value):
        s = s_value % TRACK_LENGTH_M
        straight = jnp.asarray(STRAIGHT_LENGTH_M, dtype=jnp.float32)
        radius = jnp.asarray(TURN_RADIUS_M, dtype=jnp.float32)
        half_straight = straight / 2.0
        right_turn_start = straight
        top_straight_start = straight + jnp.pi * radius
        left_turn_start = 2.0 * straight + jnp.pi * radius

        theta_right = -jnp.pi / 2.0 + (s - right_turn_start) / radius
        xy_right = jnp.array([half_straight, 0.0]) + radius * jnp.array([jnp.cos(theta_right), jnp.sin(theta_right)])
        heading_right = jax_wrap_angle(theta_right + jnp.pi / 2.0, jnp)

        theta_left = jnp.pi / 2.0 + (s - left_turn_start) / radius
        xy_left = jnp.array([-half_straight, 0.0]) + radius * jnp.array([jnp.cos(theta_left), jnp.sin(theta_left)])
        heading_left = jax_wrap_angle(theta_left + jnp.pi / 2.0, jnp)

        xy_bottom = jnp.array([-half_straight + s, -radius])
        xy_top = jnp.array([half_straight - (s - top_straight_start), radius])

        is_bottom = s < right_turn_start
        is_right = (s >= right_turn_start) & (s < top_straight_start)
        is_top = (s >= top_straight_start) & (s < left_turn_start)

        xy = jnp.where(is_bottom, xy_bottom, jnp.where(is_right, xy_right, jnp.where(is_top, xy_top, xy_left)))
        heading = jnp.where(is_bottom, 0.0, jnp.where(is_right, heading_right, jnp.where(is_top, jnp.pi, heading_left)))
        curvature = jnp.where(is_bottom | is_top, 0.0, 1.0 / radius)
        return xy.astype(jnp.float32), heading.astype(jnp.float32), curvature.astype(jnp.float32)

    def track_observation(qpos):
        base_x = qpos[0]
        base_y = qpos[1]
        w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
        base_yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        straight = jnp.asarray(STRAIGHT_LENGTH_M, dtype=jnp.float32)
        radius = jnp.asarray(TURN_RADIUS_M, dtype=jnp.float32)
        half_width = jnp.asarray(HALF_WIDTH_M, dtype=jnp.float32)
        half_straight = straight / 2.0
        xy = jnp.array([base_x, base_y])
        x_clamped = jnp.clip(base_x, -half_straight, half_straight)

        center_bottom = jnp.array([x_clamped, -radius])
        dist_bottom = jnp.sum(jnp.square(xy - center_bottom))
        s_bottom = x_clamped + half_straight
        head_bottom = 0.0
        curv_bottom = 0.0

        center_top = jnp.array([x_clamped, radius])
        dist_top = jnp.sum(jnp.square(xy - center_top))
        s_top = straight + jnp.pi * radius + (half_straight - x_clamped)
        head_top = jnp.pi
        curv_top = 0.0

        right_center = jnp.array([half_straight, 0.0])
        rel_right = xy - right_center
        theta_right = jnp.clip(jnp.arctan2(rel_right[1], rel_right[0]), -jnp.pi / 2.0, jnp.pi / 2.0)
        center_right = right_center + radius * jnp.array([jnp.cos(theta_right), jnp.sin(theta_right)])
        dist_right = jnp.sum(jnp.square(xy - center_right))
        s_right = straight + (theta_right + jnp.pi / 2.0) * radius
        head_right = theta_right + jnp.pi / 2.0
        curv_right = 1.0 / radius

        left_center = jnp.array([-half_straight, 0.0])
        rel_left = xy - left_center
        theta_left = jnp.arctan2(rel_left[1], rel_left[0])
        theta_left = jnp.where(theta_left < jnp.pi / 2.0, theta_left + 2.0 * jnp.pi, theta_left)
        theta_left = jnp.clip(theta_left, jnp.pi / 2.0, 3.0 * jnp.pi / 2.0)
        center_left = left_center + radius * jnp.array([jnp.cos(theta_left), jnp.sin(theta_left)])
        dist_left = jnp.sum(jnp.square(xy - center_left))
        s_left = 2.0 * straight + jnp.pi * radius + (theta_left - jnp.pi / 2.0) * radius
        head_left = theta_left + jnp.pi / 2.0
        curv_left = 1.0 / radius

        distances = jnp.array([dist_bottom, dist_right, dist_top, dist_left])
        best_idx = jnp.argmin(distances)
        s_values = jnp.array([s_bottom, s_right, s_top, s_left])
        heading_values = jnp.array([head_bottom, head_right, head_top, head_left])
        curvature_values = jnp.array([curv_bottom, curv_right, curv_top, curv_left])
        center_values = jnp.stack([center_bottom, center_right, center_top, center_left])

        s = s_values[best_idx]
        track_heading = heading_values[best_idx]
        curvature = curvature_values[best_idx]
        center = center_values[best_idx]
        normal = jnp.array([-jnp.sin(track_heading), jnp.cos(track_heading)])
        lateral_error = jnp.dot(xy - center, normal)
        heading_error = jax_wrap_angle(track_heading - base_yaw, jnp)

        return jnp.array(
            [
                (s % TRACK_LENGTH_M) / TRACK_LENGTH_M,
                lateral_error / half_width,
                (half_width - jnp.abs(lateral_error)) / half_width,
                heading_error,
                curvature * radius,
            ],
            dtype=jnp.float32,
        )

    return centerline_pose, track_observation


class JaxTrackBatchEnv:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.np_rng = np.random.default_rng(int(args.seed))
        self.course_cfg = load_json(args.config)
        self.course_cfg.setdefault("runtime_overrides", {})
        if args.force_cpu:
            self.course_cfg["runtime_overrides"]["force_cpu"] = True
            os.environ["JAX_PLATFORMS"] = "cpu"
        set_runtime_env(force_cpu=bool(args.force_cpu))

        self.stack = lazy_import_stack()
        self.jax = self.stack["jax"]
        self.jnp = self.jax.numpy
        self.registry = self.stack["registry"]
        self.locomotion_params = self.stack["locomotion_params"]
        self.env_name = self.course_cfg["environment_name"]
        ensure_environment_available(self.registry, self.env_name)

        env_cfg = self.registry.get_default_config(self.env_name)
        ppo_cfg = get_ppo_config(self.locomotion_params, self.env_name, self.course_cfg["backend_impl"])
        apply_stage_config(env_cfg, ppo_cfg, self.course_cfg, args.stage_name)
        self.dt = float(self.course_cfg["control"]["ctrl_dt"])
        self.max_episode_steps = int(round(float(args.max_episode_seconds) / self.dt))
        env_cfg.episode_length = self.max_episode_steps + 10
        env_cfg.noise_config.level = 0.0
        env_cfg.pert_config.enable = False
        self.env = self.registry.load(self.env_name, config=env_cfg, config_overrides=build_env_overrides(self.course_cfg))

        self.lowlevel_policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
        self.num_envs = int(args.num_envs)
        self.stand_steps = int(round(float(args.stand_seconds) / self.dt))
        self.centerline_pose_fn, self.track_obs_fn = make_jax_track_fns(self.jnp)
        self.track_obs_batch = self.jax.jit(self.jax.vmap(self.track_obs_fn))
        self.reset_batch_fn = self._make_reset_batch_fn()
        self.step_batch_fn = self._make_step_batch_fn()
        self.state = None
        self.obs = None
        self.prev_s = np.zeros(self.num_envs, dtype=np.float32)
        self.prev_speed = np.zeros(self.num_envs, dtype=np.float32)
        self.cum_progress = np.zeros(self.num_envs, dtype=np.float32)
        self.last_cmd = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.episode_step = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_count = 0
        self.global_reset_count = 0
        self.rng_key = self.jax.random.PRNGKey(int(args.seed))

    def _sample_starts(self, count: int, update_idx: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.args.start_randomization == "fixed":
            s = np.full(count, float(self.args.start_s_m) % TRACK_LENGTH_M, dtype=np.float32)
        elif self.args.start_randomization == "full_track":
            s = self.np_rng.uniform(0.0, TRACK_LENGTH_M, size=count).astype(np.float32)
        else:
            mix = min(1.0, 0.20 + 0.015 * float(update_idx))
            random_s = self.np_rng.uniform(0.0, TRACK_LENGTH_M, size=count).astype(np.float32)
            fixed_s = np.full(count, float(self.args.start_s_m) % TRACK_LENGTH_M, dtype=np.float32)
            use_random = self.np_rng.random(count) < mix
            s = np.where(use_random, random_s, fixed_s).astype(np.float32)
        lateral = self.np_rng.normal(0.0, float(self.args.lateral_reset_std), size=count)
        lateral = np.clip(lateral, -0.45 * HALF_WIDTH_M, 0.45 * HALF_WIDTH_M).astype(np.float32)
        heading = self.np_rng.normal(0.0, float(self.args.heading_reset_std), size=count)
        heading = np.clip(heading, -0.35, 0.35).astype(np.float32)
        return s, lateral, heading

    def _make_reset_batch_fn(self) -> Any:
        jax = self.jax
        jnp = self.jnp
        env = self.env
        centerline_pose = self.centerline_pose_fn
        from mujoco import mjx
        from mujoco.mjx._src import math as mjmath
        from mujoco_playground._src import mjx_env

        def reset_one(rng_key, start_s, lateral_offset, heading_offset):
            state = env.reset(rng_key)
            qpos = env._init_q
            qvel = jnp.zeros(env.mjx_model.nv)
            xy, heading, _ = centerline_pose(start_s)
            heading = heading + heading_offset
            normal = jnp.array([-jnp.sin(heading), jnp.cos(heading)], dtype=jnp.float32)
            xy = xy + lateral_offset * normal
            quat = mjmath.axis_angle_to_quat(jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32), heading)
            qpos = qpos.at[0:2].set(xy)
            qpos = qpos.at[3:7].set(quat)
            data = mjx_env.make_data(
                env.mj_model,
                qpos=qpos,
                qvel=qvel,
                ctrl=qpos[7:],
                impl=env.mjx_model.impl.value,
                naconmax=env._config.naconmax,
                njmax=env._config.njmax,
            )
            data = mjx.forward(env.mjx_model, data)
            state.info["command"] = jnp.zeros(3, dtype=jnp.float32)
            state.info["steps_until_next_cmd"] = jnp.asarray(10**9, dtype=jnp.int32)
            obs = env._get_obs(data, state.info)
            return state.replace(data=data, obs=obs, reward=jnp.zeros(()), done=jnp.zeros(()))

        return jax.jit(jax.vmap(reset_one))

    def _make_step_batch_fn(self) -> Any:
        jax = self.jax
        jnp = self.jnp
        env = self.env
        policy = self.lowlevel_policy
        step_fn = env.step

        def step_one(env_state, command, rng_key):
            env_state.info["command"] = command
            env_state.info["steps_until_next_cmd"] = jnp.asarray(10**9, dtype=jnp.int32)
            action, _ = policy(env_state.obs, rng_key)
            next_state = step_fn(env_state, action)
            next_state.info["command"] = command
            next_state.info["steps_until_next_cmd"] = jnp.asarray(10**9, dtype=jnp.int32)
            return next_state

        return jax.jit(jax.vmap(step_one))

    def reset(self, update_idx: int = 0) -> np.ndarray:
        starts, lateral, heading = self._sample_starts(self.num_envs, update_idx=update_idx)
        self.rng_key, reset_key = self.jax.random.split(self.rng_key)
        reset_keys = self.jax.random.split(reset_key, self.num_envs)
        self.state = self.reset_batch_fn(
            reset_keys,
            self.jnp.asarray(starts, dtype=self.jnp.float32),
            self.jnp.asarray(lateral, dtype=self.jnp.float32),
            self.jnp.asarray(heading, dtype=self.jnp.float32),
        )
        self.obs = np.array(self.track_obs_batch(self.state.data.qpos), dtype=np.float32, copy=True)
        self.prev_s = (self.obs[:, 0] * TRACK_LENGTH_M).astype(np.float32)
        self.prev_speed.fill(0.0)
        self.cum_progress.fill(0.0)
        self.last_cmd.fill(0.0)
        self.episode_step.fill(0)
        self.episode_return.fill(0.0)
        return self.obs.copy()

    def _reset_done_envs(self, done: np.ndarray, update_idx: int) -> None:
        if self.state is None or not np.any(done):
            return
        starts, lateral, heading = self._sample_starts(self.num_envs, update_idx=update_idx)
        self.rng_key, reset_key = self.jax.random.split(self.rng_key)
        reset_keys = self.jax.random.split(reset_key, self.num_envs)
        reset_state = self.reset_batch_fn(
            reset_keys,
            self.jnp.asarray(starts, dtype=self.jnp.float32),
            self.jnp.asarray(lateral, dtype=self.jnp.float32),
            self.jnp.asarray(heading, dtype=self.jnp.float32),
        )
        done_mask = self.jnp.asarray(done)

        def choose(next_leaf, reset_leaf):
            if not hasattr(next_leaf, "shape") or len(next_leaf.shape) == 0:
                return next_leaf
            mask = done_mask
            while len(mask.shape) < len(next_leaf.shape):
                mask = mask[..., None]
            return self.jnp.where(mask, reset_leaf, next_leaf)

        self.state = self.jax.tree_util.tree_map(choose, self.state, reset_state)
        reset_obs = np.array(self.track_obs_batch(reset_state.data.qpos), dtype=np.float32, copy=True)
        self.obs = np.array(self.obs, dtype=np.float32, copy=True)
        self.obs[done] = reset_obs[done]
        self.prev_s[done] = self.obs[done, 0] * TRACK_LENGTH_M
        self.prev_speed[done] = 0.0
        self.cum_progress[done] = 0.0
        self.last_cmd[done] = 0.0
        self.episode_step[done] = 0
        self.episode_return[done] = 0.0
        self.global_reset_count += int(np.sum(done))

    def step(self, raw_action: np.ndarray, update_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        if self.state is None or self.obs is None:
            raise RuntimeError("Call reset() before step().")

        speed_cfg = speed_schedule(self.args, update_idx)
        command = raw_action_to_command(
            raw_action,
            max_vx=float(speed_cfg["max_vx"]),
            max_vy=float(self.args.max_vy),
            max_yaw_rate=float(self.args.max_yaw_rate),
        )
        line_bias_norm = raw_action_to_line_bias(raw_action, self.obs, self.args)
        command = apply_racing_line_command_bias(command, self.obs, line_bias_norm, self.args)
        command = apply_stability_envelope(command, self.obs, self.args, max_vx=float(speed_cfg["max_vx"]))
        warmup = self.episode_step < self.stand_steps
        command[warmup] = 0.0

        alpha = float(np.clip(float(self.args.command_filter_alpha), 0.05, 1.0))
        command = alpha * command + (1.0 - alpha) * self.last_cmd
        delta_cmd = command - self.last_cmd
        command = self.last_cmd + np.clip(delta_cmd, -float(self.args.max_command_delta), float(self.args.max_command_delta))
        self.rng_key, step_key = self.jax.random.split(self.rng_key)
        step_keys = self.jax.random.split(step_key, self.num_envs)
        self.state = self.step_batch_fn(self.state, self.jnp.asarray(command, dtype=self.jnp.float32), step_keys)

        next_obs = np.array(self.track_obs_batch(self.state.data.qpos), dtype=np.float32, copy=True)
        s_next = next_obs[:, 0] * TRACK_LENGTH_M
        delta_s = s_next - self.prev_s
        delta_s = np.where(delta_s < -TRACK_LENGTH_M / 2.0, delta_s + TRACK_LENGTH_M, delta_s)
        delta_s = np.where(delta_s > TRACK_LENGTH_M / 2.0, delta_s - TRACK_LENGTH_M, delta_s)
        delta_s = np.where(warmup, 0.0, delta_s)
        progress_speed = delta_s / max(self.dt, 1e-6)
        accel_proxy = (progress_speed - self.prev_speed) / max(self.dt, 1e-6)

        qpos = np.asarray(self.state.data.qpos, dtype=np.float32)
        qvel = np.asarray(self.state.data.qvel, dtype=np.float32)
        torques = np.asarray(self.state.data.actuator_force, dtype=np.float32)
        energy = np.mean(np.abs(torques * qvel[:, 6:18]), axis=1)
        try:
            feet_vel = np.asarray(self.state.data.sensordata[:, self.env._foot_linvel_sensor_adr], dtype=np.float32)
            slip = np.mean(np.linalg.norm(feet_vel[:, :, :2], axis=-1), axis=1)
        except Exception:
            slip = np.zeros(self.num_envs, dtype=np.float32)

        lateral_m = np.abs(next_obs[:, 1]) * HALF_WIDTH_M
        line_error_norm = next_obs[:, 1] - line_bias_norm
        line_error_m = line_error_norm * HALF_WIDTH_M
        heading_abs = np.abs(next_obs[:, 3])
        margin_m = next_obs[:, 2] * HALF_WIDTH_M
        turn_intensity = np.clip(np.abs(next_obs[:, 4]), 0.0, 1.0)
        target_speed = (
            (1.0 - turn_intensity) * float(speed_cfg["target_straight_speed"])
            + turn_intensity * float(speed_cfg["target_curve_speed"])
        )
        target_speed = np.minimum(target_speed, float(speed_cfg["max_vx"]))
        target_speed = np.maximum(target_speed, 1e-3)
        speed_fraction = np.clip(progress_speed / target_speed, 0.0, 1.0)
        slow_gap = np.clip((target_speed - progress_speed) / target_speed, 0.0, 2.0)
        backward_speed = np.clip(-progress_speed, 0.0, 2.0)
        new_cum_progress = self.cum_progress + np.clip(delta_s, -0.20, 0.20)
        fallen = np.asarray(self.state.done, dtype=bool) | (qpos[:, 2] < 0.23)
        boundary = margin_m < 0.0
        finished = new_cum_progress >= TRACK_LENGTH_M
        timeout = (self.episode_step + 1) >= self.max_episode_steps
        done = fallen | boundary | finished | timeout

        reward = float(self.args.progress_reward_scale) * delta_s
        reward += float(self.args.speed_reward_scale) * np.clip(progress_speed, -0.5, float(speed_cfg["max_vx"]))
        reward += float(self.args.target_speed_reward_scale) * speed_fraction
        reward += float(self.args.curve_speed_reward_scale) * turn_intensity * np.clip(
            progress_speed,
            0.0,
            float(speed_cfg["max_vx"]),
        )
        reward -= float(self.args.slow_penalty_scale) * slow_gap
        reward -= float(self.args.backward_penalty_scale) * backward_speed
        reward += 0.02 * np.clip(next_obs[:, 2], 0.0, 1.0)
        reward -= float(self.args.line_lateral_weight) * np.square(line_error_m)
        reward -= float(self.args.center_lateral_weight) * np.square(lateral_m)
        reward -= float(self.args.line_bias_penalty) * np.square(line_bias_norm)
        reward -= 0.08 * np.square(heading_abs)
        reward -= 0.025 * np.sum(np.square(command - self.last_cmd), axis=1)
        reward -= 0.00002 * np.square(accel_proxy)
        reward -= 0.0005 * energy
        reward -= 0.015 * slip
        reward -= 3.0 * np.maximum(0.0, -margin_m)
        reward = reward.astype(np.float32)
        reward[fallen] -= 10.0
        reward[boundary] -= 12.0
        reward[finished] += 30.0

        self.episode_return += reward
        self.episode_step += 1
        episode_distances = new_cum_progress.copy()
        episode_returns = self.episode_return.copy()

        done_count = int(np.sum(done))
        finished_count = int(np.sum(finished))
        mean_done_distance = float(np.mean(episode_distances[done])) if done_count else 0.0
        mean_done_return = float(np.mean(episode_returns[done])) if done_count else 0.0
        self.episode_count += done_count

        self.obs = next_obs
        self.prev_s = s_next.astype(np.float32)
        self.prev_speed = progress_speed.astype(np.float32)
        self.cum_progress = new_cum_progress.astype(np.float32)
        self.last_cmd = command.astype(np.float32)
        self._reset_done_envs(done, update_idx=update_idx)

        info = {
            "done_count": float(done_count),
            "finished_count": float(finished_count),
            "mean_done_distance": mean_done_distance,
            "mean_done_return": mean_done_return,
            "mean_lateral_m": float(np.mean(lateral_m)),
            "mean_line_error_m": float(np.mean(np.abs(line_error_m))),
            "mean_line_bias_norm": float(np.mean(line_bias_norm)),
            "mean_abs_line_bias_norm": float(np.mean(np.abs(line_bias_norm))),
            "mean_margin_m": float(np.mean(margin_m)),
            "mean_progress_speed": float(np.mean(progress_speed)),
            "mean_target_speed": float(np.mean(target_speed)),
            "curriculum_fraction": float(speed_cfg["fraction"]),
            "current_max_vx": float(speed_cfg["max_vx"]),
            "current_target_straight_speed": float(speed_cfg["target_straight_speed"]),
            "current_target_curve_speed": float(speed_cfg["target_curve_speed"]),
            "fall_count": float(np.sum(fallen)),
            "boundary_count": float(np.sum(boundary)),
            "mean_energy": float(np.mean(energy)),
            "mean_slip": float(np.mean(slip)),
        }
        return self.obs.copy(), reward, done.astype(np.float32), info


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    rollout_steps, num_envs = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(num_envs, dtype=np.float32)
    for t in reversed(range(rollout_steps)):
        if t == rollout_steps - 1:
            next_nonterminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t]
            next_values = values[t + 1]
        delta = rewards[t] + float(gamma) * next_values * next_nonterminal - values[t]
        last_gae = delta + float(gamma) * float(gae_lambda) * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def export_numpy_planner(
    model: ActorCritic,
    args: argparse.Namespace,
    output_dir: Path,
    update_idx: int | None = None,
) -> None:
    actor_layers = [module for module in model.actor_body if isinstance(module, nn.Linear)]
    arrays: dict[str, np.ndarray] = {}
    for idx, layer in enumerate(actor_layers, start=1):
        arrays[f"w{idx}"] = layer.weight.detach().cpu().numpy().T.astype(np.float32)
        arrays[f"b{idx}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    arrays["w_out"] = model.actor_mean.weight.detach().cpu().numpy().T.astype(np.float32)
    arrays["b_out"] = model.actor_mean.bias.detach().cpu().numpy().astype(np.float32)
    np.savez(output_dir / "planner_weights.npz", **arrays)

    deploy_speed = speed_schedule(args, update_idx)
    planner_config = {
        "planner_type": "ppo_mlp",
        "weights_path": "planner_weights.npz",
        "stand_seconds": float(args.stand_seconds),
        "hidden_dim": int(args.hidden_dim),
        "num_hidden_layers": int(args.num_hidden_layers),
        "command_filter_alpha": float(args.command_filter_alpha),
        "max_straight_speed_mps": float(deploy_speed["max_vx"]),
        "max_lateral_speed_mps": float(args.max_vy),
        "max_yaw_rate_radps": float(args.max_yaw_rate),
        "edge_slowdown_margin_norm": float(args.edge_slowdown_margin_norm),
        "max_command_delta": float(args.max_command_delta),
        "use_stability_envelope": True,
        "heading_speed_penalty": float(args.heading_speed_penalty),
        "lateral_speed_penalty": float(args.lateral_speed_penalty),
        "edge_speed_penalty": float(args.edge_speed_penalty),
        "turn_speed_penalty": float(args.turn_speed_penalty),
        "min_speed_cap_scale": float(args.min_speed_cap_scale),
        "use_racing_line": bool(args.use_racing_line),
        "max_line_bias_norm": float(args.max_line_bias_norm),
        "line_vy_gain": float(args.line_vy_gain),
        "line_yaw_gain": float(args.line_yaw_gain),
        "max_line_vy": float(args.max_line_vy),
        "max_line_yaw": float(args.max_line_yaw),
        "track_length_m": TRACK_LENGTH_M,
        "turn_radius_m": TURN_RADIUS_M,
        "half_width_m": HALF_WIDTH_M,
        "training_curriculum_fraction": float(deploy_speed["fraction"]),
        "training_target_straight_speed_mps": float(deploy_speed["target_straight_speed"]),
        "training_target_curve_speed_mps": float(deploy_speed["target_curve_speed"]),
    }
    (output_dir / "planner_config.json").write_text(json.dumps(planner_config, indent=2), encoding="utf-8")


def maybe_run_eval(args: argparse.Namespace, output_dir: Path, update_idx: int) -> dict[str, Any] | None:
    if int(args.eval_interval) <= 0 or update_idx % int(args.eval_interval) != 0:
        return None
    eval_dir = output_dir / "eval" / f"update_{update_idx:04d}"
    cmd = [
        sys.executable,
        "run_track_bonus.py",
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--planner-config",
        str(output_dir / "planner_config.json"),
        "--config",
        str(args.config),
        "--output-dir",
        str(eval_dir),
        "--duration-seconds",
        str(args.eval_seconds),
        "--entry-name",
        "ppo_mlp",
        "--no-render",
    ]
    if args.force_cpu:
        cmd.append("--force-cpu")
    try:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return json.loads((eval_dir / "results.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc), "eval_dir": str(eval_dir)}


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    action_dim = 4 if bool(args.use_racing_line) and float(args.max_line_bias_norm) > 0.0 else 3
    model = ActorCritic(
        obs_dim=5,
        hidden_dim=int(args.hidden_dim),
        num_hidden_layers=int(args.num_hidden_layers),
        action_dim=action_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), eps=1e-5)
    env = JaxTrackBatchEnv(args)
    obs = env.reset(update_idx=0)
    history: list[dict[str, Any]] = []

    print(
        json.dumps(
            {
                "device": str(device),
                "num_envs": int(args.num_envs),
                "rollout_steps": int(args.rollout_steps),
                "action_dim": int(action_dim),
                "dt": env.dt,
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )

    for update_idx in range(1, int(args.total_updates) + 1):
        t0 = time.time()
        obs_buf = np.zeros((args.rollout_steps, args.num_envs, 5), dtype=np.float32)
        action_buf = np.zeros((args.rollout_steps, args.num_envs, action_dim), dtype=np.float32)
        logprob_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
        reward_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
        done_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
        value_buf = np.zeros((args.rollout_steps, args.num_envs), dtype=np.float32)
        rollout_infos: list[dict[str, float]] = []

        model.eval()
        for step_idx in range(int(args.rollout_steps)):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                raw_action, log_prob, _, value = model.act(obs_tensor)
            raw_action_np = raw_action.cpu().numpy().astype(np.float32)
            next_obs, reward, done, info = env.step(raw_action_np, update_idx=update_idx)

            obs_buf[step_idx] = obs
            action_buf[step_idx] = raw_action_np
            logprob_buf[step_idx] = log_prob.cpu().numpy()
            value_buf[step_idx] = value.cpu().numpy()
            reward_buf[step_idx] = reward
            done_buf[step_idx] = done
            rollout_infos.append(info)
            obs = next_obs

        with torch.no_grad():
            next_value = model.value(torch.as_tensor(obs, dtype=torch.float32, device=device)).cpu().numpy()
        advantages, returns = compute_gae(
            reward_buf,
            done_buf,
            value_buf,
            next_value,
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
        )

        batch_obs = torch.as_tensor(obs_buf.reshape(-1, 5), dtype=torch.float32, device=device)
        batch_actions = torch.as_tensor(action_buf.reshape(-1, action_dim), dtype=torch.float32, device=device)
        batch_old_logprob = torch.as_tensor(logprob_buf.reshape(-1), dtype=torch.float32, device=device)
        batch_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)
        batch_adv = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
        batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std(unbiased=False) + 1e-8)
        batch_size = batch_obs.shape[0]
        minibatch_size = min(int(args.minibatch_size), batch_size)

        model.train()
        policy_losses = []
        value_losses = []
        entropy_values = []
        approx_kls = []
        clipfracs = []
        indices = np.arange(batch_size)
        for _ in range(int(args.ppo_epochs)):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = torch.as_tensor(indices[start : start + minibatch_size], dtype=torch.long, device=device)
                new_logprob, entropy, new_value = model.evaluate_actions(batch_obs[mb_idx], batch_actions[mb_idx])
                logratio = new_logprob - batch_old_logprob[mb_idx]
                ratio = logratio.exp()
                mb_adv = batch_adv[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - float(args.clip_eps), 1.0 + float(args.clip_eps))
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                value_loss = 0.5 * torch.square(new_value - batch_returns[mb_idx]).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + float(args.value_coef) * value_loss - float(args.entropy_coef) * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > float(args.clip_eps)).float()).item()
                policy_losses.append(float(policy_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                entropy_values.append(float(entropy_loss.detach().cpu()))
                approx_kls.append(float(approx_kl))
                clipfracs.append(float(clipfrac))

        export_numpy_planner(model, args, output_dir, update_idx=update_idx)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "update": update_idx,
            },
            output_dir / "ppo_actor_critic.pt",
        )

        info_sum = {
            "done_count": sum(item["done_count"] for item in rollout_infos),
            "finished_count": sum(item["finished_count"] for item in rollout_infos),
            "mean_lateral_m": float(np.mean([item["mean_lateral_m"] for item in rollout_infos])),
            "mean_line_error_m": float(np.mean([item["mean_line_error_m"] for item in rollout_infos])),
            "mean_line_bias_norm": float(np.mean([item["mean_line_bias_norm"] for item in rollout_infos])),
            "mean_abs_line_bias_norm": float(np.mean([item["mean_abs_line_bias_norm"] for item in rollout_infos])),
            "mean_margin_m": float(np.mean([item["mean_margin_m"] for item in rollout_infos])),
            "mean_progress_speed": float(np.mean([item["mean_progress_speed"] for item in rollout_infos])),
            "mean_target_speed": float(np.mean([item["mean_target_speed"] for item in rollout_infos])),
            "curriculum_fraction": float(np.mean([item["curriculum_fraction"] for item in rollout_infos])),
            "current_max_vx": float(np.mean([item["current_max_vx"] for item in rollout_infos])),
            "current_target_straight_speed": float(
                np.mean([item["current_target_straight_speed"] for item in rollout_infos])
            ),
            "current_target_curve_speed": float(
                np.mean([item["current_target_curve_speed"] for item in rollout_infos])
            ),
            "fall_count": sum(item["fall_count"] for item in rollout_infos),
            "boundary_count": sum(item["boundary_count"] for item in rollout_infos),
            "mean_energy": float(np.mean([item["mean_energy"] for item in rollout_infos])),
            "mean_slip": float(np.mean([item["mean_slip"] for item in rollout_infos])),
        }
        done_distances = [item["mean_done_distance"] for item in rollout_infos if item["done_count"] > 0]
        done_returns = [item["mean_done_return"] for item in rollout_infos if item["done_count"] > 0]

        eval_payload = maybe_run_eval(args, output_dir, update_idx)
        record: dict[str, Any] = {
            "update": update_idx,
            "elapsed_s": time.time() - t0,
            "mean_reward": float(np.mean(reward_buf)),
            "mean_return_target": float(np.mean(returns)),
            "mean_advantage": float(np.mean(advantages)),
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "done_count": int(info_sum["done_count"]),
            "finished_count": int(info_sum["finished_count"]),
            "mean_done_distance": float(np.mean(done_distances)) if done_distances else 0.0,
            "mean_done_return": float(np.mean(done_returns)) if done_returns else 0.0,
            "mean_lateral_m": info_sum["mean_lateral_m"],
            "mean_line_error_m": info_sum["mean_line_error_m"],
            "mean_line_bias_norm": info_sum["mean_line_bias_norm"],
            "mean_abs_line_bias_norm": info_sum["mean_abs_line_bias_norm"],
            "mean_margin_m": info_sum["mean_margin_m"],
            "mean_progress_speed": info_sum["mean_progress_speed"],
            "mean_target_speed": info_sum["mean_target_speed"],
            "curriculum_fraction": info_sum["curriculum_fraction"],
            "current_max_vx": info_sum["current_max_vx"],
            "current_target_straight_speed": info_sum["current_target_straight_speed"],
            "current_target_curve_speed": info_sum["current_target_curve_speed"],
            "fall_count": int(info_sum["fall_count"]),
            "boundary_count": int(info_sum["boundary_count"]),
            "mean_energy": info_sum["mean_energy"],
            "mean_slip": info_sum["mean_slip"],
            "eval": eval_payload,
        }
        history.append(record)
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        eval_msg = ""
        if eval_payload and "metrics" in eval_payload:
            metrics = eval_payload["metrics"]
            eval_msg = (
                f" | eval_lap={metrics['lap_completion']:.3f}"
                f" dist={metrics['valid_distance_m']:.1f}m"
                f" fall={metrics['fall']} boundary={metrics['boundary_violation']}"
            )
        elif eval_payload and "error" in eval_payload:
            eval_msg = f" | eval_error={eval_payload['error']}"

        print(
            f"[update {update_idx:04d}] reward={record['mean_reward']:.3f} "
            f"speed={record['mean_progress_speed']:.2f}/{record['mean_target_speed']:.2f}m/s "
            f"cap={record['current_max_vx']:.2f} curr={record['curriculum_fraction']:.2f} "
            f"lateral={record['mean_lateral_m']:.2f}m "
            f"line={record['mean_abs_line_bias_norm']:.2f}/{record['mean_line_error_m']:.2f} "
            f"done={record['done_count']} finish={record['finished_count']} "
            f"kl={record['approx_kl']:.4f} ent={record['entropy']:.3f} "
            f"time={record['elapsed_s']:.1f}s{eval_msg}",
            flush=True,
        )

    print(json.dumps({"planner_config": str(output_dir / "planner_config.json"), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
