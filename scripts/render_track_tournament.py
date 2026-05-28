#!/usr/bin/env python3
"""Render up to 10 independent track rollouts as one synchronized tournament.

This script is intentionally rollout-based.  Each student's controller and
low-level checkpoint are evaluated independently with `run_track_bonus.py`.
The resulting `race_rollouts.npz` files are then combined here, avoiding Python
module conflicts between different high-level controller implementations while
still producing a single MuJoCo scene with all robots starting together.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from competition.track_scene import build_track_model, render_track_video  # noqa: E402
from go2_pg_env.track import StandardOvalTrack  # noqa: E402
from track_bonus.controller_interface import MAX_TOURNAMENT_ENTRIES  # noqa: E402


DEFAULT_COLORS = [
    "#2563EB",
    "#DC2626",
    "#16A34A",
    "#F59E0B",
    "#7C3AED",
    "#0891B2",
    "#DB2777",
    "#65A30D",
    "#EA580C",
    "#475569",
]

HOME_QPOS = np.asarray(
    [
        0.0,
        0.0,
        0.31,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.8,
        -1.5,
        0.0,
        0.8,
        -1.5,
        0.0,
        0.8,
        -1.5,
        0.0,
        0.8,
        -1.5,
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=Path, help="JSON manifest of evaluated rollout files.")
    parser.add_argument("--demo-synthetic", action="store_true", help="Generate a synthetic 10-dog demo instead of reading entries.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-dogs", type=int, default=10)
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--track-length-m", type=float, default=200.0)
    parser.add_argument("--turn-radius-m", type=float, default=18.25)
    parser.add_argument("--track-half-width-m", type=float, default=2.0)
    parser.add_argument("--visual-lane-offsets", action="store_true", help="Offset rendered trajectories laterally for visibility only.")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--render-every", type=int, default=3)
    parser.add_argument("--render-fps", type=int, default=10)
    parser.add_argument("--render-camera-profile", choices=["showcase", "close", "overview"], default="overview")
    parser.add_argument("--asset-model-dir", type=Path)
    return parser.parse_args()


def _quat_from_yaw(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=np.float32)


def _normal_at(track: StandardOvalTrack, xy: np.ndarray) -> np.ndarray:
    projection = track.project_xy_to_track(np.asarray(xy, dtype=np.float64))
    _, heading, _ = track.centerline_pose(projection.s)
    return np.asarray([-math.sin(heading), math.cos(heading)], dtype=np.float32)


def _apply_visual_offsets(qpos: np.ndarray, track: StandardOvalTrack) -> np.ndarray:
    """Spread robots within the track for readability without changing scoring."""
    qpos = np.asarray(qpos, dtype=np.float32).copy()
    num_dogs = qpos.shape[0]
    offsets = np.linspace(-0.8 * track.half_width_m, 0.8 * track.half_width_m, num_dogs)
    for dog_idx, offset in enumerate(offsets):
        for step_idx in range(qpos.shape[1]):
            qpos[dog_idx, step_idx, :2] += float(offset) * _normal_at(track, qpos[dog_idx, step_idx, :2])
    return qpos


def _synthetic_trajectories(track: StandardOvalTrack, *, num_dogs: int, steps: int, duration: float) -> np.ndarray:
    qpos = np.zeros((num_dogs, steps, 19), dtype=np.float32)
    speeds = np.linspace(0.55, 1.25, num_dogs)
    offsets = np.linspace(-0.8 * track.half_width_m, 0.8 * track.half_width_m, num_dogs)
    times = np.linspace(0.0, float(duration), steps, endpoint=False)
    for dog_idx in range(num_dogs):
        for step_idx, t in enumerate(times):
            s = speeds[dog_idx] * t
            center, heading, _ = track.centerline_pose(float(s))
            normal = np.asarray([-math.sin(heading), math.cos(heading)], dtype=np.float32)
            pose = HOME_QPOS.copy()
            pose[:2] = np.asarray(center, dtype=np.float32) + offsets[dog_idx] * normal
            pose[3:7] = _quat_from_yaw(float(heading))
            qpos[dog_idx, step_idx] = pose
    return qpos


def _load_entries(entries_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    payload = json.loads(entries_path.read_text(encoding="utf-8"))
    entries = payload.get("entries") or payload.get("policies") or []
    if not entries:
        raise ValueError(f"No entries found in {entries_path}. Expected key 'entries'.")
    if len(entries) > MAX_TOURNAMENT_ENTRIES:
        raise ValueError(f"At most {MAX_TOURNAMENT_ENTRIES} entries are supported.")

    names: list[str] = []
    colors: list[str] = []
    trajectories = []
    base_dir = entries_path.resolve().parent
    for idx, entry in enumerate(entries):
        name = str(entry.get("name", f"entry_{idx}"))
        color = str(entry.get("color", DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]))
        rollout_path = Path(entry["rollout_npz"])
        if not rollout_path.is_absolute():
            rollout_path = base_dir / rollout_path
        data = np.load(rollout_path, allow_pickle=False)
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        if qpos.ndim == 3:
            qpos = qpos[0]
        if qpos.ndim != 2 or qpos.shape[-1] != 19:
            raise ValueError(f"{rollout_path} qpos must have shape [steps, 19] or [1, steps, 19].")
        names.append(name)
        colors.append(color)
        trajectories.append(qpos)

    min_steps = min(traj.shape[0] for traj in trajectories)
    stacked = np.stack([traj[:min_steps] for traj in trajectories], axis=0)
    return names, colors, stacked


def main() -> None:
    args = parse_args()
    if bool(args.demo_synthetic) == bool(args.entries):
        raise SystemExit("Use exactly one of --demo-synthetic or --entries.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    track = StandardOvalTrack(
        length_m=float(args.track_length_m),
        turn_radius_m=float(args.turn_radius_m),
        half_width_m=float(args.track_half_width_m),
    )

    if args.demo_synthetic:
        num_dogs = int(args.num_dogs)
        if num_dogs > MAX_TOURNAMENT_ENTRIES:
            raise ValueError(f"At most {MAX_TOURNAMENT_ENTRIES} dogs are supported.")
        names = [f"demo_{idx}" for idx in range(num_dogs)]
        colors = DEFAULT_COLORS[:num_dogs]
        trajectories_qpos = _synthetic_trajectories(
            track,
            num_dogs=num_dogs,
            steps=int(args.steps),
            duration=float(args.duration_seconds),
        )
    else:
        names, colors, trajectories_qpos = _load_entries(args.entries.resolve())
        if args.visual_lane_offsets:
            trajectories_qpos = _apply_visual_offsets(trajectories_qpos, track)

    model = build_track_model(
        num_dogs=int(trajectories_qpos.shape[0]),
        colors=colors,
        asset_model_dir=args.asset_model_dir,
        track=track,
    )

    np.savez_compressed(
        output_dir / "tournament_rollouts.npz",
        policy_names=np.asarray(names),
        qpos=trajectories_qpos,
        colors=np.asarray(colors),
    )

    video_path = None
    if not args.no_render:
        video_path = render_track_video(
            trajectories_qpos=trajectories_qpos,
            output_path=output_dir / "track_tournament.mp4",
            colors=colors,
            fps=int(args.render_fps),
            render_every=int(args.render_every),
            width=int(args.render_width),
            height=int(args.render_height),
            camera_profile=str(args.render_camera_profile),
            asset_model_dir=args.asset_model_dir,
            track_config={
                "track_length_m": track.length_m,
                "turn_radius_m": track.turn_radius_m,
                "half_width_m": track.half_width_m,
            },
        )

    summary = {
        "num_dogs": int(trajectories_qpos.shape[0]),
        "model_nq": int(model.nq),
        "model_nu": int(model.nu),
        "expected_nq": int(19 * trajectories_qpos.shape[0]),
        "expected_nu": int(12 * trajectories_qpos.shape[0]),
        "names": names,
        "colors": colors,
        "rollouts_npz": str(output_dir / "tournament_rollouts.npz"),
        "video_path": None if video_path is None else str(video_path),
    }
    (output_dir / "tournament_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
