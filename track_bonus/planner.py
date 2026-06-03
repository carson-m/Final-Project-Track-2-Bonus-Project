"""Learned Neural Network (MLP) High-Level Planner for the 200m Track."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle
from track_bonus.controller_interface import TrackControllerObservation
from track_bonus.official_track import official_track


@dataclass(frozen=True)
class LearnedPlannerConfig:
    planner_type: str = "learned_mlp"
    weights_path: str = "planner_weights.npz"
    stand_seconds: float = 1.0
    hidden_dim: int = 32

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LearnedPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        values = {key: payload[key] for key in payload if key in valid}
        return cls(**values)


class StarterTrackPlanner:
    """Entrypoint class keeping original names so evaluator scripts match."""

    def __init__(self, config: LearnedPlannerConfig, weights: dict[str, np.ndarray] | None = None):
        self.config = config
        self.track = official_track()
        self.weights = weights
        
        # --- STRATEGY 3 & 5: STATE HISTORY TRACKING FOR PROPRIOCEPTION & SMOOTHING ---
        self.prev_s: float | None = None
        self.prev_t: float | None = None
        self.last_cmd: np.ndarray = np.zeros(3, dtype=np.float32)

    @classmethod
    def load(cls, path: Path | str) -> "StarterTrackPlanner":
        path = Path(path)
        with open(path, "r") as f:
            payload = json.load(f)
        config = LearnedPlannerConfig.from_dict(payload)
        
        # Look for the weights file relative to the config json path
        weights_file = path.parent / config.weights_path
        weights = None
        if weights_file.exists():
            weights = np.load(weights_file)
            
        return cls(config, weights)

    def command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        # Give the robot time to stand up safely
        if t < self.config.stand_seconds:
            self.prev_s = float(obs.lap_fraction) * 200.0
            self.prev_t = t
            self.last_cmd = np.zeros(3, dtype=np.float32)
            return np.zeros(3, dtype=np.float32)

        # Total track geometry distance mapping
        s = float(obs.lap_fraction) * 200.0

        # --- STRATEGY 3: CLOSED-LOOP ESTIMATED PROGRESS SPEED CONTROLLER ---
        if self.prev_s is None or self.prev_t is None or t == self.prev_t:
            v_est = 0.0
        else:
            delta_s = s - self.prev_s
            # Resolve lap boundary cross wrap-arounds
            if delta_s < -100.0:
                delta_s += 200.0
            elif delta_s > 100.0:
                delta_s -= 200.0
            v_est = delta_s / (t - self.prev_t)

        # --- STRATEGY 1: LOOK-AHEAD TRACK CURVATURE EXTRACTION ---
        def get_curv_norm(s_val: float) -> float:
            s_mod = s_val % 200.0
            # Track Segment Definitions: Straight [0-50], Turn [50-100], Straight [100-150], Turn [150-200]
            if (50.0 <= s_mod < 100.0) or (150.0 <= s_mod < 200.0):
                return 1.0
            return 0.0

        # Build expanded 8D track observation array
        x = np.array([
            float(obs.lap_fraction),
            float(obs.lateral_error_norm),
            float(obs.boundary_margin_norm),
            float(obs.heading_error_rad),
            float(obs.curvature_norm),
            float(v_est),                 # Proprioceptive close-loop feedback
            float(get_curv_norm(s + 2.0)), # Look-ahead 2 meters
            float(get_curv_norm(s + 5.0))  # Look-ahead 5 meters
        ], dtype=np.float32)

        # Update historical trackers
        self.prev_s = s
        self.prev_t = t

        # Fallback safe crawl if weights aren't loaded or found yet
        if self.weights is None:
            self.last_cmd = np.array([0.3, 0.0, 0.0], dtype=np.float32)
            return self.last_cmd

        # Forward Pass: 8 Inputs -> Hidden Layers -> 3 Outputs
        w1, b1 = self.weights['w1'], self.weights['b1']
        w2, b2 = self.weights['w2'], self.weights['b2']

        # Layer 1 Activation (ReLU)
        h1 = np.maximum(0, np.dot(x, w1) + b1)
        
        # Layer 2 Output (Linear)
        out = np.dot(h1, w2) + b2

        # SMOOTH TANH ACTIVATION (Must perfectly match training structural output)
        vx = 1.5 * (np.tanh(out[0]) + 1.0)
        vy = 0.5 * np.tanh(out[1])
        yaw_rate = 1.0 * np.tanh(out[2])

        cmd_raw = np.array([vx, vy, yaw_rate], dtype=np.float32)

        # --- STRATEGY 5: SMOOTH EXPO ACTION RATE FILTER TO PREVENT CHATTER ---
        alpha = 0.20
        cmd_filtered = alpha * cmd_raw + (1.0 - alpha) * self.last_cmd
        self.last_cmd = cmd_filtered

        return cmd_filtered
