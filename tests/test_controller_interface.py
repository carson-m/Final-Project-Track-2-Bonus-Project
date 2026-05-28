import numpy as np
import pytest

from competition.race_scene import resolve_go2_asset_model_dir
from competition.track_scene import build_track_model
from go2_pg_env.track import StandardOvalTrack
from track_bonus.controller_interface import (
    MAX_TOURNAMENT_ENTRIES,
    build_track_controller_observation,
    sanitize_high_level_command,
)


def test_sanitize_high_level_command_clips_and_validates_shape() -> None:
    clipped = sanitize_high_level_command(np.asarray([2.0, -1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(clipped, np.asarray([1.5, -0.5, 1.5], dtype=np.float32))
    with pytest.raises(ValueError):
        sanitize_high_level_command(np.asarray([0.1, 0.2], dtype=np.float32))
    with pytest.raises(ValueError):
        sanitize_high_level_command(np.asarray([0.1, np.nan, 0.2], dtype=np.float32))


def test_track_controller_observation_has_expected_fields() -> None:
    track = StandardOvalTrack()
    xy, heading, _ = track.centerline_pose(0.0)
    qpos = np.zeros(19, dtype=np.float32)
    qpos[:2] = xy
    qpos[2] = 0.31
    qpos[3:7] = np.asarray([np.cos(0.5 * heading), 0.0, 0.0, np.sin(0.5 * heading)], dtype=np.float32)
    obs = build_track_controller_observation(qpos=qpos, t=1.25, track=track)
    assert obs.qpos.shape == (19,)
    assert abs(obs.lateral_error_m) < 1e-6
    assert obs.boundary_margin_m == pytest.approx(track.half_width_m)
    assert abs(obs.heading_error_rad) < 1e-6
    assert 0.0 <= obs.lap_fraction < 1.0


def test_track_scene_compiles_ten_dogs_when_assets_are_available() -> None:
    try:
        resolve_go2_asset_model_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    model = build_track_model(num_dogs=MAX_TOURNAMENT_ENTRIES, colors=["#2563EB"] * MAX_TOURNAMENT_ENTRIES)
    assert model.nq == 19 * MAX_TOURNAMENT_ENTRIES
    assert model.nu == 12 * MAX_TOURNAMENT_ENTRIES
