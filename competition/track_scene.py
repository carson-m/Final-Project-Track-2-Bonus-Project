"""Build and render multi-Go2 scenes on the 200 m oval track."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from competition.race_scene import GO2_BODY_XML, resolve_go2_asset_model_dir, tint_dogs
from go2_pg_env.track import StandardOvalTrack


def _rgba_string(rgba: tuple[float, float, float, float]) -> str:
    return " ".join(f"{value:.3f}" for value in rgba)


def _track_points(track: StandardOvalTrack, *, marker_count: int, lateral_offset: float) -> np.ndarray:
    points = []
    for idx in range(marker_count):
        s = track.length_m * idx / marker_count
        center, heading, _ = track.centerline_pose(s)
        normal = np.asarray([-np.sin(heading), np.cos(heading)], dtype=np.float64)
        points.append(center + lateral_offset * normal)
    return np.asarray(points, dtype=np.float64)


def _capsule_segments(
    *,
    name: str,
    points: np.ndarray,
    z: float,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> list[str]:
    geoms = []
    rgba_text = _rgba_string(rgba)
    for idx in range(points.shape[0]):
        a = points[idx]
        b = points[(idx + 1) % points.shape[0]]
        geoms.append(
            f'<geom name="{name}_{idx}" type="capsule" '
            f'fromto="{a[0]:.3f} {a[1]:.3f} {z:.3f} {b[0]:.3f} {b[1]:.3f} {z:.3f}" '
            f'size="{radius:.3f}" rgba="{rgba_text}" contype="0" conaffinity="0"/>'
        )
    return geoms


def _start_finish_geoms(track: StandardOvalTrack) -> list[str]:
    center, heading, _ = track.centerline_pose(0.0)
    normal = np.asarray([-np.sin(heading), np.cos(heading)], dtype=np.float64)
    tangent = np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float64)
    inner = center - track.half_width_m * normal
    outer = center + track.half_width_m * normal
    stripe_a_inner = inner - 0.10 * tangent
    stripe_a_outer = outer - 0.10 * tangent
    stripe_b_inner = inner + 0.10 * tangent
    stripe_b_outer = outer + 0.10 * tangent
    return [
        (
            '<geom name="start_finish_green" type="capsule" '
            f'fromto="{stripe_a_inner[0]:.3f} {stripe_a_inner[1]:.3f} 0.035 '
            f'{stripe_a_outer[0]:.3f} {stripe_a_outer[1]:.3f} 0.035" '
            'size="0.060" rgba="0.05 0.70 0.24 1" contype="0" conaffinity="0"/>'
        ),
        (
            '<geom name="start_finish_red" type="capsule" '
            f'fromto="{stripe_b_inner[0]:.3f} {stripe_b_inner[1]:.3f} 0.036 '
            f'{stripe_b_outer[0]:.3f} {stripe_b_outer[1]:.3f} 0.036" '
            'size="0.060" rgba="0.86 0.10 0.10 1" contype="0" conaffinity="0"/>'
        ),
    ]


def _parent_track_xml(track: StandardOvalTrack, *, marker_count: int) -> str:
    extent_x = track.straight_length_m / 2.0 + track.turn_radius_m + track.half_width_m + 5.0
    extent_y = track.turn_radius_m + track.half_width_m + 5.0
    center_points = _track_points(track, marker_count=marker_count, lateral_offset=0.0)
    left_boundary = _track_points(track, marker_count=marker_count, lateral_offset=track.half_width_m)
    right_boundary = _track_points(track, marker_count=marker_count, lateral_offset=-track.half_width_m)
    geoms = []
    geoms.extend(
        _capsule_segments(
            name="centerline",
            points=center_points,
            z=0.026,
            radius=0.018,
            rgba=(1.0, 1.0, 1.0, 0.72),
        )
    )
    geoms.extend(
        _capsule_segments(
            name="outer_boundary",
            points=left_boundary,
            z=0.032,
            radius=0.040,
            rgba=(0.07, 0.08, 0.08, 0.95),
        )
    )
    geoms.extend(
        _capsule_segments(
            name="inner_boundary",
            points=right_boundary,
            z=0.032,
            radius=0.040,
            rgba=(0.07, 0.08, 0.08, 0.95),
        )
    )
    geoms.extend(_start_finish_geoms(track))

    return f"""
<mujoco model="go2 oval track competition">
  <option timestep="0.004" integrator="Euler"/>
  <visual>
    <headlight diffuse=".48 .48 .45" ambient=".18 .18 .18" specular=".18 .18 .18"/>
    <global azimuth="90" elevation="-35" offwidth="1920" offheight="1080"/>
    <map znear=".02" zfar="140"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".90 .95 1" rgb2=".60 .70 .84" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1=".50 .54 .50" rgb2=".42 .46 .43" markrgb=".23 .25 .24" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="12 7" reflectance="0"/>
  </asset>
  <worldbody>
    <light name="key_light" pos="0 -35 18" diffuse=".52 .50 .44"/>
    <light name="rim_light" pos="25 28 10" diffuse=".25 .32 .42"/>
    <geom name="floor" size="{extent_x:.3f} {extent_y:.3f} 0.01" type="plane" material="groundplane" contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>
    {"".join(geoms)}
  </worldbody>
</mujoco>
""".strip()


def build_track_model(
    *,
    num_dogs: int,
    colors: list[str] | None = None,
    robot_tint_strength: float = 0.32,
    asset_model_dir: str | Path | None = None,
    track: StandardOvalTrack | None = None,
    marker_count: int = 96,
):
    """Compile a MuJoCo model containing `num_dogs` Go2s on one oval track."""
    if num_dogs < 1:
        raise ValueError("num_dogs must be positive.")
    if num_dogs > 10:
        raise ValueError("The track renderer supports at most 10 dogs.")

    import mujoco

    track = track or StandardOvalTrack()
    asset_root = resolve_go2_asset_model_dir(asset_model_dir)
    parent = mujoco.MjSpec.from_string(_parent_track_xml(track, marker_count=int(marker_count)))
    for dog_idx in range(num_dogs):
        child = mujoco.MjSpec.from_file(GO2_BODY_XML.as_posix())
        child.modelfiledir = asset_root.as_posix()
        child.meshdir = "assets"
        parent.attach(child, prefix=f"dog{dog_idx}_", frame=parent.worldbody.add_frame())

    model = parent.compile()
    if model.nq != 19 * num_dogs or model.nu != 12 * num_dogs:
        raise RuntimeError(f"Unexpected track model dimensions: nq={model.nq}, nu={model.nu}, dogs={num_dogs}")
    if colors:
        tint_dogs(model, colors, blend=robot_tint_strength)
    return model


def _configure_camera(camera: Any, *, profile: str, trajectories_qpos: np.ndarray, step_idx: int, track: StandardOvalTrack) -> None:
    if profile == "close":
        xy = np.median(trajectories_qpos[:, step_idx, 0:2], axis=0)
        camera.lookat[:] = np.asarray([xy[0], xy[1], 0.35])
        camera.distance = 7.0
        camera.azimuth = 90.0
        camera.elevation = -24.0
    elif profile == "showcase":
        xy = np.median(trajectories_qpos[:, step_idx, 0:2], axis=0)
        camera.lookat[:] = np.asarray([xy[0], xy[1], 0.45])
        camera.distance = 10.0
        camera.azimuth = 110.0
        camera.elevation = -30.0
    else:
        camera.lookat[:] = np.asarray([0.0, 0.0, 0.25])
        camera.distance = max(58.0, track.straight_length_m + track.turn_radius_m)
        camera.azimuth = 90.0
        camera.elevation = -68.0


def render_track_video(
    *,
    trajectories_qpos: np.ndarray,
    output_path: Path,
    colors: list[str],
    fps: int,
    render_every: int,
    width: int,
    height: int,
    camera_profile: str = "overview",
    robot_tint_strength: float = 0.32,
    asset_model_dir: str | Path | None = None,
    track_config: dict[str, Any] | None = None,
) -> Path:
    """Render synchronized global-coordinate policy trajectories on an oval."""
    import mujoco

    track_config = track_config or {}
    track = StandardOvalTrack(
        length_m=float(track_config.get("track_length_m", 200.0)),
        turn_radius_m=float(track_config.get("turn_radius_m", 18.25)),
        half_width_m=float(track_config.get("half_width_m", 2.0)),
    )
    marker_count = int(track_config.get("marker_count", 96))

    trajectories_qpos = np.asarray(trajectories_qpos, dtype=np.float64)
    if trajectories_qpos.ndim != 3 or trajectories_qpos.shape[-1] != 19:
        raise ValueError("trajectories_qpos must have shape [num_policies, steps, 19].")

    num_dogs, num_steps, _ = trajectories_qpos.shape
    model = build_track_model(
        num_dogs=num_dogs,
        colors=colors,
        robot_tint_strength=robot_tint_strength,
        asset_model_dir=asset_model_dir,
        track=track,
        marker_count=marker_count,
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    frames = []
    for step_idx in range(0, num_steps, max(1, int(render_every))):
        for dog_idx in range(num_dogs):
            start = dog_idx * 19
            data.qpos[start : start + 19] = trajectories_qpos[dog_idx, step_idx]
        mujoco.mj_forward(model, data)
        _configure_camera(camera, profile=camera_profile, trajectories_qpos=trajectories_qpos, step_idx=step_idx, track=track)
        renderer.update_scene(data, camera)
        frames.append(renderer.render())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import mediapy as media

        media.write_video(output_path, frames, fps=int(fps))
        written_path = output_path
    except Exception:
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg  # noqa: F401

            written_path = output_path if output_path.suffix.lower() == ".mp4" else output_path.with_suffix(".mp4")
            with imageio.get_writer(
                written_path,
                fps=int(fps),
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1,
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
        except Exception:
            from PIL import Image

            written_path = output_path if output_path.suffix.lower() == ".gif" else output_path.with_suffix(".gif")
            pil_frames = [Image.fromarray(frame) for frame in frames]
            duration_ms = max(1, int(round(1000.0 / float(fps))))
            pil_frames[0].save(
                written_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
            )
    renderer.close()
    return written_path
