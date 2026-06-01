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
            return np.zeros(3, dtype=np.float32)

        # Build standard 5D track observation array
        x = np.array([
            float(obs.lap_fraction),
            float(obs.lateral_error_norm),
            float(obs.boundary_margin_norm),
            float(obs.heading_error_rad),
            float(obs.curvature_norm)
        ], dtype=np.float32)

        # Fallback safe crawl if weights aren't loaded or found yet
        if self.weights is None:
            return np.array([0.3, 0.0, 0.0], dtype=np.float32)

        # Forward Pass: 5 Inputs -> Hidden Layers -> 3 Outputs
        w1, b1 = self.weights['w1'], self.weights['b1']
        w2, b2 = self.weights['w2'], self.weights['b2']

        # Layer 1 Activation (ReLU)
        h1 = np.maximum(0, np.dot(x, w1) + b1)
        
        # Layer 2 Output (Linear)
        out = np.dot(h1, w2) + b2

        # Bound predictions to keep your low-level controller inside its safe operating envelope
        vx = np.clip(out[0], 0.0, 1.5)        # Forward speed limit
        vy = np.clip(out[1], -0.3, 0.3)       # Lateral drift adjustment limit 
        yaw_rate = np.clip(out[2], -1.2, 1.2) # Max turning angular velocity

        return np.array([vx, vy, yaw_rate], dtype=np.float32)
