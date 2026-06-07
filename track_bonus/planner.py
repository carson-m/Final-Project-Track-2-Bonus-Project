"""Starter high-level planner for the 200 m track bonus.

The evaluator builds the official compact 5D track observation defined in
`track_bonus/controller_interface.py`. The high-level planner maps it to the
local joystick command consumed by the HW1 Go2 locomotion policy:

    5D track observation -> [vx, vy, yaw_rate]

This file is intentionally small.  It is a weak baseline and an interface
example, not a solved full-lap controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle
from track_bonus.controller_interface import TrackControllerObservation
from track_bonus.official_track import official_track


@dataclass(frozen=True)
class StarterPlannerConfig:
    planner_type: str = "starter_pd"
    speed_mps: float = 0.45
    min_speed_mps: float = 0.12
    max_lateral_speed_mps: float = 0.08
    max_yaw_rate_radps: float = 0.25
    k_heading: float = 0.55
    k_lateral: float = 0.08
    heading_slowdown: float = 0.45
    stand_seconds: float = 1.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StarterPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        values = {key: payload[key] for key in valid if key in payload}
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "StarterPlannerConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_type": self.planner_type,
            "speed_mps": self.speed_mps,
            "min_speed_mps": self.min_speed_mps,
            "max_lateral_speed_mps": self.max_lateral_speed_mps,
            "max_yaw_rate_radps": self.max_yaw_rate_radps,
            "k_heading": self.k_heading,
            "k_lateral": self.k_lateral,
            "heading_slowdown": self.heading_slowdown,
            "stand_seconds": self.stand_seconds,
        }


@dataclass(frozen=True)
class LearnedPlannerConfig:
    planner_type: str = "learned_mlp"
    weights_path: str = "planner_weights.npz"
    stand_seconds: float = 1.0
    hidden_dim: int = 32
    command_filter_alpha: float = 0.15
    max_straight_speed_mps: float = 0.95
    max_curve_speed_mps: float = 0.55
    max_lateral_speed_mps: float = 0.25
    max_yaw_rate_radps: float = 0.55
    edge_slowdown_margin_norm: float = 0.35
    max_command_delta: float = 0.08

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LearnedPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        values = {key: payload[key] for key in payload if key in valid}
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "LearnedPlannerConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_type": self.planner_type,
            "weights_path": self.weights_path,
            "stand_seconds": self.stand_seconds,
            "hidden_dim": self.hidden_dim,
            "command_filter_alpha": self.command_filter_alpha,
            "max_straight_speed_mps": self.max_straight_speed_mps,
            "max_curve_speed_mps": self.max_curve_speed_mps,
            "max_lateral_speed_mps": self.max_lateral_speed_mps,
            "max_yaw_rate_radps": self.max_yaw_rate_radps,
            "edge_slowdown_margin_norm": self.edge_slowdown_margin_norm,
            "max_command_delta": self.max_command_delta,
        }


@dataclass(frozen=True)
class PPOPlannerConfig:
    planner_type: str = "ppo_mlp"
    weights_path: str = "planner_weights.npz"
    stand_seconds: float = 1.0
    hidden_dim: int = 64
    num_hidden_layers: int = 2
    command_filter_alpha: float = 0.30
    max_straight_speed_mps: float = 2.50
    max_lateral_speed_mps: float = 0.30
    max_yaw_rate_radps: float = 0.80
    edge_slowdown_margin_norm: float = 0.35
    max_command_delta: float = 0.18
    use_stability_envelope: bool = True
    heading_speed_penalty: float = 0.20
    lateral_speed_penalty: float = 0.25
    edge_speed_penalty: float = 0.35
    turn_speed_penalty: float = 0.10
    min_speed_cap_scale: float = 0.45
    use_racing_line: bool = True
    max_line_bias_norm: float = 0.50
    line_vy_gain: float = 0.22
    line_yaw_gain: float = 0.28
    max_line_vy: float = 0.18
    max_line_yaw: float = 0.24
    track_length_m: float = 200.0
    turn_radius_m: float = 18.25
    half_width_m: float = 2.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PPOPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        values = {key: payload[key] for key in payload if key in valid}
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "PPOPlannerConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_type": self.planner_type,
            "weights_path": self.weights_path,
            "stand_seconds": self.stand_seconds,
            "hidden_dim": self.hidden_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "command_filter_alpha": self.command_filter_alpha,
            "max_straight_speed_mps": self.max_straight_speed_mps,
            "max_lateral_speed_mps": self.max_lateral_speed_mps,
            "max_yaw_rate_radps": self.max_yaw_rate_radps,
            "edge_slowdown_margin_norm": self.edge_slowdown_margin_norm,
            "max_command_delta": self.max_command_delta,
            "use_stability_envelope": self.use_stability_envelope,
            "heading_speed_penalty": self.heading_speed_penalty,
            "lateral_speed_penalty": self.lateral_speed_penalty,
            "edge_speed_penalty": self.edge_speed_penalty,
            "turn_speed_penalty": self.turn_speed_penalty,
            "min_speed_cap_scale": self.min_speed_cap_scale,
            "use_racing_line": self.use_racing_line,
            "max_line_bias_norm": self.max_line_bias_norm,
            "line_vy_gain": self.line_vy_gain,
            "line_yaw_gain": self.line_yaw_gain,
            "max_line_vy": self.max_line_vy,
            "max_line_yaw": self.max_line_yaw,
            "track_length_m": self.track_length_m,
            "turn_radius_m": self.turn_radius_m,
            "half_width_m": self.half_width_m,
        }


class StarterTrackPlanner:
    """Conservative coordinate-to-command baseline.

    The policy is deliberately simple and conservative. Students should improve
    it by changing this controller, replacing it with an MLP, or training a
    higher-level policy that produces the same command vector.
    """

    def __init__(
        self,
        config: StarterPlannerConfig | LearnedPlannerConfig | PPOPlannerConfig,
        weights: dict[str, np.ndarray] | None = None,
    ) -> None:
        if config.planner_type not in {"starter_pd", "learned_mlp", "ppo_mlp"}:
            raise ValueError(f"Unsupported planner_type: {config.planner_type!r}")
        self.config = config
        self.track: StandardOvalTrack = official_track()
        self.weights = weights
        self.prev_s: float | None = None
        self.prev_t: float | None = None
        self.last_cmd = np.zeros(3, dtype=np.float32)

    @classmethod
    def load(cls, path: Path) -> "StarterTrackPlanner":
        payload = json.loads(path.read_text(encoding="utf-8"))
        planner_type = payload.get("planner_type", "starter_pd")
        if planner_type == "learned_mlp":
            config = LearnedPlannerConfig.from_dict(payload)
            weights_path = path.parent / config.weights_path
            weights = None
            if weights_path.exists():
                with np.load(weights_path) as data:
                    weights = {key: np.asarray(data[key]) for key in data.files}
            return cls(config, weights)
        if planner_type == "ppo_mlp":
            config = PPOPlannerConfig.from_dict(payload)
            weights_path = path.parent / config.weights_path
            weights = None
            if weights_path.exists():
                with np.load(weights_path) as data:
                    weights = {key: np.asarray(data[key]) for key in data.files}
            return cls(config, weights)
        return cls(StarterPlannerConfig.from_dict(payload))

    def command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        if isinstance(self.config, LearnedPlannerConfig):
            return self._learned_command(obs, t)
        if isinstance(self.config, PPOPlannerConfig):
            return self._ppo_command(obs, t)

        if t < self.config.stand_seconds:
            return np.zeros(3, dtype=np.float32)
        return self.command_from_observation(obs)

    def command_from_observation(self, obs: TrackControllerObservation) -> np.ndarray:
        if not isinstance(self.config, StarterPlannerConfig):
            raise TypeError("command_from_observation is only available for starter_pd planners.")

        lateral_error = float(obs.lateral_error_norm) * float(self.track.half_width_m)
        lateral_bias = math.atan2(
            float(self.config.k_lateral) * lateral_error,
            max(float(self.config.speed_mps), 1e-3),
        )
        heading_error = wrap_angle(float(obs.heading_error_rad) - lateral_bias)

        speed_scale = 1.0 - float(self.config.heading_slowdown) * min(abs(heading_error), math.pi) / math.pi
        vx = np.clip(
            float(self.config.speed_mps) * speed_scale,
            float(self.config.min_speed_mps),
            float(self.config.speed_mps),
        )
        vy = np.clip(
            -float(self.config.k_lateral) * lateral_error,
            -float(self.config.max_lateral_speed_mps),
            float(self.config.max_lateral_speed_mps),
        )
        curvature = float(obs.curvature_norm) / max(float(self.track.turn_radius_m), 1e-6)
        yaw_rate = np.clip(
            curvature * vx + float(self.config.k_heading) * heading_error,
            -float(self.config.max_yaw_rate_radps),
            float(self.config.max_yaw_rate_radps),
        )
        return np.asarray([vx, vy, yaw_rate], dtype=np.float32)

    def _learned_command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        if not isinstance(self.config, LearnedPlannerConfig):
            raise TypeError("_learned_command is only available for learned_mlp planners.")

        if t < self.config.stand_seconds:
            self.prev_s = float(obs.lap_fraction) * 200.0
            self.prev_t = t
            self.last_cmd = np.zeros(3, dtype=np.float32)
            return np.zeros(3, dtype=np.float32)

        s = float(obs.lap_fraction) * 200.0
        if self.prev_s is None or self.prev_t is None or t == self.prev_t:
            v_est = 0.0
        else:
            delta_s = s - self.prev_s
            if delta_s < -100.0:
                delta_s += 200.0
            elif delta_s > 100.0:
                delta_s -= 200.0
            v_est = delta_s / (t - self.prev_t)

        def get_curv_norm(s_val: float) -> float:
            _, _, curvature = self.track.centerline_pose(s_val)
            return abs(float(curvature)) * float(self.track.turn_radius_m)

        x = np.array(
            [
                float(obs.lap_fraction),
                float(obs.lateral_error_norm),
                float(obs.boundary_margin_norm),
                float(obs.heading_error_rad),
                float(obs.curvature_norm),
                float(v_est),
                float(get_curv_norm(s + 2.0)),
                float(get_curv_norm(s + 5.0)),
            ],
            dtype=np.float32,
        )

        self.prev_s = s
        self.prev_t = t

        if self.weights is None:
            self.last_cmd = np.array([0.3, 0.0, 0.0], dtype=np.float32)
            return self.last_cmd

        w1, b1 = self.weights["w1"], self.weights["b1"]
        w2, b2 = self.weights["w2"], self.weights["b2"]

        h1 = np.maximum(0.0, np.dot(x, w1) + b1)
        out = np.dot(h1, w2) + b2

        cmd_raw = np.array(
            [
                0.5 * float(self.config.max_straight_speed_mps) * (np.tanh(out[0]) + 1.0),
                float(self.config.max_lateral_speed_mps) * np.tanh(out[1]),
                float(self.config.max_yaw_rate_radps) * np.tanh(out[2]),
            ],
            dtype=np.float32,
        )

        cmd_raw = self._apply_learned_stability_envelope(obs, s, cmd_raw)
        alpha = float(np.clip(self.config.command_filter_alpha, 0.05, 1.0))
        cmd_filtered = alpha * cmd_raw + (1.0 - alpha) * self.last_cmd
        max_delta = float(max(self.config.max_command_delta, 0.0))
        if max_delta > 0.0:
            delta = np.clip(cmd_filtered - self.last_cmd, -max_delta, max_delta)
            cmd_filtered = self.last_cmd + delta
        self.last_cmd = cmd_filtered.astype(np.float32)
        return self.last_cmd

    def _ppo_command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        if not isinstance(self.config, PPOPlannerConfig):
            raise TypeError("_ppo_command is only available for ppo_mlp planners.")

        if t < self.config.stand_seconds:
            self.last_cmd = np.zeros(3, dtype=np.float32)
            return np.zeros(3, dtype=np.float32)

        if self.weights is None:
            self.last_cmd = np.array([0.25, 0.0, 0.0], dtype=np.float32)
            return self.last_cmd

        x = obs.as_array().astype(np.float32)
        h = x
        for layer_idx in range(1, int(self.config.num_hidden_layers) + 1):
            w_key = f"w{layer_idx}"
            b_key = f"b{layer_idx}"
            if w_key not in self.weights or b_key not in self.weights:
                raise KeyError(f"Missing PPO planner layer weights: {w_key}/{b_key}")
            h = np.tanh(np.dot(h, self.weights[w_key]) + self.weights[b_key])

        if "w_out" not in self.weights or "b_out" not in self.weights:
            raise KeyError("Missing PPO planner output weights: w_out/b_out")
        out = np.dot(h, self.weights["w_out"]) + self.weights["b_out"]

        line_bias_norm = 0.0
        if bool(self.config.use_racing_line) and np.asarray(out).shape[0] >= 4:
            turn_gate = float(np.clip(abs(float(obs.curvature_norm)), 0.0, 1.0))
            line_bias_norm = float(self.config.max_line_bias_norm) * float(np.tanh(out[3])) * turn_gate

        cmd_raw = np.array(
            [
                0.5 * float(self.config.max_straight_speed_mps) * (np.tanh(out[0]) + 1.0),
                float(self.config.max_lateral_speed_mps) * np.tanh(out[1]),
                float(self.config.max_yaw_rate_radps) * np.tanh(out[2]),
            ],
            dtype=np.float32,
        )

        if bool(self.config.use_racing_line):
            line_error_norm = float(obs.lateral_error_norm) - line_bias_norm
            cmd_raw[1] += float(
                np.clip(
                    -float(self.config.line_vy_gain) * line_error_norm,
                    -float(self.config.max_line_vy),
                    float(self.config.max_line_vy),
                )
            )
            cmd_raw[2] += float(
                np.clip(
                    -float(self.config.line_yaw_gain) * line_error_norm,
                    -float(self.config.max_line_yaw),
                    float(self.config.max_line_yaw),
                )
            )

        s = float(obs.lap_fraction) * float(self.track.length_m)
        if bool(self.config.use_stability_envelope):
            cmd_raw = self._apply_ppo_stability_envelope(obs, s, cmd_raw)

        alpha = float(np.clip(self.config.command_filter_alpha, 0.05, 1.0))
        cmd_filtered = alpha * cmd_raw + (1.0 - alpha) * self.last_cmd
        max_delta = float(max(self.config.max_command_delta, 0.0))
        if max_delta > 0.0:
            delta = np.clip(cmd_filtered - self.last_cmd, -max_delta, max_delta)
            cmd_filtered = self.last_cmd + delta
        self.last_cmd = cmd_filtered.astype(np.float32)
        return self.last_cmd

    def _apply_learned_stability_envelope(
        self,
        obs: TrackControllerObservation,
        s: float,
        cmd: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(self.config, LearnedPlannerConfig):
            raise TypeError("_apply_learned_stability_envelope requires a learned_mlp planner.")

        def get_curv_norm(s_val: float) -> float:
            _, _, curvature = self.track.centerline_pose(s_val)
            return abs(float(curvature)) * float(self.track.turn_radius_m)

        turn_intensity = max(
            abs(float(obs.curvature_norm)),
            get_curv_norm(s + 2.0),
            get_curv_norm(s + 5.0),
        )
        turn_intensity = float(np.clip(turn_intensity, 0.0, 1.0))

        speed_cap = (
            (1.0 - turn_intensity) * float(self.config.max_straight_speed_mps)
            + turn_intensity * float(self.config.max_curve_speed_mps)
        )

        heading_risk = min(abs(float(obs.heading_error_rad)) / 1.0, 1.0)
        lateral_risk = np.clip((abs(float(obs.lateral_error_norm)) - 0.35) / 0.65, 0.0, 1.0)
        margin = float(obs.boundary_margin_norm)
        edge_limit = max(float(self.config.edge_slowdown_margin_norm), 1e-6)
        edge_risk = np.clip((edge_limit - margin) / edge_limit, 0.0, 1.0)

        risk_scale = 1.0 - 0.30 * heading_risk - 0.25 * lateral_risk - 0.35 * edge_risk
        speed_cap *= float(np.clip(risk_scale, 0.35, 1.0))

        vx = np.clip(float(cmd[0]), 0.0, speed_cap)
        vy = np.clip(
            float(cmd[1]),
            -float(self.config.max_lateral_speed_mps),
            float(self.config.max_lateral_speed_mps),
        )
        yaw_rate = np.clip(
            float(cmd[2]),
            -float(self.config.max_yaw_rate_radps),
            float(self.config.max_yaw_rate_radps),
        )
        return np.array([vx, vy, yaw_rate], dtype=np.float32)

    def _apply_ppo_stability_envelope(
        self,
        obs: TrackControllerObservation,
        s: float,
        cmd: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(self.config, PPOPlannerConfig):
            raise TypeError("_apply_ppo_stability_envelope requires a ppo_mlp planner.")

        def get_curv_norm(s_val: float) -> float:
            _, _, curvature = self.track.centerline_pose(s_val)
            return abs(float(curvature)) * float(self.track.turn_radius_m)

        turn_intensity = max(
            abs(float(obs.curvature_norm)),
            get_curv_norm(s + 2.0),
            get_curv_norm(s + 5.0),
        )
        turn_intensity = float(np.clip(turn_intensity, 0.0, 1.0))

        speed_cap = float(self.config.max_straight_speed_mps)
        heading_risk = min(abs(float(obs.heading_error_rad)) / 1.0, 1.0)
        lateral_risk = np.clip((abs(float(obs.lateral_error_norm)) - 0.35) / 0.65, 0.0, 1.0)
        edge_limit = max(float(self.config.edge_slowdown_margin_norm), 1e-6)
        edge_risk = np.clip((edge_limit - float(obs.boundary_margin_norm)) / edge_limit, 0.0, 1.0)
        turn_risk = float(self.config.turn_speed_penalty) * turn_intensity

        risk_scale = (
            1.0
            - float(self.config.heading_speed_penalty) * heading_risk
            - float(self.config.lateral_speed_penalty) * lateral_risk
            - float(self.config.edge_speed_penalty) * edge_risk
            - turn_risk
        )
        speed_cap *= float(np.clip(risk_scale, float(self.config.min_speed_cap_scale), 1.0))

        vx = np.clip(float(cmd[0]), 0.0, speed_cap)
        vy = np.clip(
            float(cmd[1]),
            -float(self.config.max_lateral_speed_mps),
            float(self.config.max_lateral_speed_mps),
        )
        yaw_rate = np.clip(
            float(cmd[2]),
            -float(self.config.max_yaw_rate_radps),
            float(self.config.max_yaw_rate_radps),
        )
        return np.array([vx, vy, yaw_rate], dtype=np.float32)
