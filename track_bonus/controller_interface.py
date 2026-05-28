"""Shared high-level controller interface for the track tournament.

The official high-level command is always:

    [vx_mps, vy_mps, yaw_rate_radps]

The command is consumed by the HW1-style low-level Go2 policy.  This module
keeps the command contract explicit so student planners remain compatible with
single-policy evaluation and later multi-policy tournament rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle
from track_bonus.planner import yaw_from_quat_wxyz


MAX_TOURNAMENT_ENTRIES = 10

# Conservative official command envelope. The evaluator clips to this range so
# a high-level controller cannot destabilize the low-level policy with invalid
# values or NaNs.
COMMAND_LOW = np.asarray([0.0, -0.50, -1.50], dtype=np.float32)
COMMAND_HIGH = np.asarray([1.50, 0.50, 1.50], dtype=np.float32)


@dataclass(frozen=True)
class TrackControllerObservation:
    """Allowed high-level observation features for the tournament controller."""

    t: float
    qpos: np.ndarray
    base_xy: np.ndarray
    base_yaw: float
    centerline_s_m: float
    lap_fraction: float
    lateral_error_m: float
    boundary_margin_m: float
    track_heading_rad: float
    heading_error_rad: float
    curvature_1pm: float
    track_length_m: float
    track_half_width_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": float(self.t),
            "qpos": np.asarray(self.qpos, dtype=np.float32).copy(),
            "base_xy": np.asarray(self.base_xy, dtype=np.float32).copy(),
            "base_yaw": float(self.base_yaw),
            "centerline_s_m": float(self.centerline_s_m),
            "lap_fraction": float(self.lap_fraction),
            "lateral_error_m": float(self.lateral_error_m),
            "boundary_margin_m": float(self.boundary_margin_m),
            "track_heading_rad": float(self.track_heading_rad),
            "heading_error_rad": float(self.heading_error_rad),
            "curvature_1pm": float(self.curvature_1pm),
            "track_length_m": float(self.track_length_m),
            "track_half_width_m": float(self.track_half_width_m),
        }


def build_track_controller_observation(
    *,
    qpos: np.ndarray,
    t: float,
    track: StandardOvalTrack,
) -> TrackControllerObservation:
    """Build the standardized high-level observation from the robot's own pose."""
    qpos = np.asarray(qpos, dtype=np.float32)
    base_xy = np.asarray(qpos[:2], dtype=np.float64)
    base_yaw = yaw_from_quat_wxyz(np.asarray(qpos[3:7], dtype=np.float64))
    projection = track.project_xy_to_track(base_xy)
    _, track_heading, curvature = track.centerline_pose(projection.s)
    heading_error = wrap_angle(track_heading - base_yaw)
    return TrackControllerObservation(
        t=float(t),
        qpos=qpos.copy(),
        base_xy=base_xy.astype(np.float32),
        base_yaw=float(base_yaw),
        centerline_s_m=float(projection.s),
        lap_fraction=float((projection.s % track.length_m) / track.length_m),
        lateral_error_m=float(projection.signed_lateral_error),
        boundary_margin_m=float(projection.distance_to_boundary),
        track_heading_rad=float(track_heading),
        heading_error_rad=float(heading_error),
        curvature_1pm=float(curvature),
        track_length_m=float(track.length_m),
        track_half_width_m=float(track.half_width_m),
    )


def sanitize_high_level_command(command: np.ndarray) -> np.ndarray:
    """Validate and clip a high-level command to the official command envelope."""
    command = np.asarray(command, dtype=np.float32)
    if command.shape != (3,):
        raise ValueError(f"High-level command must have shape (3,), got {command.shape}.")
    if not np.all(np.isfinite(command)):
        raise ValueError(f"High-level command contains non-finite values: {command!r}.")
    return np.clip(command, COMMAND_LOW, COMMAND_HIGH).astype(np.float32)
