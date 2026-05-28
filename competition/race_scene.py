"""Build and render multi-Go2 race scenes for competition visualization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GO2_XML_DIR = ROOT / "go2_pg_env" / "xmls"
GO2_BODY_XML = GO2_XML_DIR / "go2_mjx_feetonly.xml"


def _hex_to_rgba(color: str) -> np.ndarray | None:
    color = color.strip()
    if not color.startswith("#") or len(color) not in (7, 9):
        return None
    try:
        values = [int(color[idx : idx + 2], 16) / 255.0 for idx in range(1, 7, 2)]
        alpha = int(color[7:9], 16) / 255.0 if len(color) == 9 else 1.0
    except ValueError:
        return None
    return np.asarray([*values, alpha], dtype=np.float32)


def _asset_dir_is_valid(model_dir: Path) -> bool:
    return (model_dir / "assets" / "base_0.obj").is_file()


def resolve_go2_asset_model_dir(explicit: str | Path | None = None) -> Path:
    """Return a directory that contains the Go2 `assets/` mesh folder."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("UNITREE_GO2_MODEL_DIR"):
        candidates.append(Path(os.environ["UNITREE_GO2_MODEL_DIR"]).expanduser())
    candidates.extend(
        [
            GO2_XML_DIR,
            ROOT.parent / "Referrals" / "unitree_mujoco-main" / "unitree_robots" / "go2",
            ROOT.parent / "unitree_mujoco" / "unitree_robots" / "go2",
            ROOT.parent / "unitree_mujoco-original" / "unitree_robots" / "go2",
        ]
    )

    for candidate in candidates:
        if _asset_dir_is_valid(candidate):
            return candidate.resolve()

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find Go2 mesh assets. Run scripts/copy_go2_assets.py, set "
        "UNITREE_GO2_MODEL_DIR, or pass asset_model_dir in the competition config.\n"
        f"Searched:\n{searched}"
    )


def _rgba_string(rgba: np.ndarray) -> str:
    return " ".join(f"{float(x):.3f}" for x in rgba)


def _parent_scene_xml(num_dogs: int, lane_spacing: float, track_length_m: float) -> str:
    half_width = max(2.0, (num_dogs - 1) * lane_spacing * 0.5 + lane_spacing)
    finish_x = max(1.0, float(track_length_m))
    lane_geoms = []
    for dog_idx in range(num_dogs):
        y = (dog_idx - (num_dogs - 1) * 0.5) * lane_spacing
        lane_geoms.append(
            f'<geom name="lane_{dog_idx}" type="box" pos="{finish_x * 0.5:.3f} {y:.3f} 0.012" '
            f'size="{finish_x * 0.5:.3f} 0.018 0.006" rgba="0.1 0.1 0.1 0.35" '
            'contype="0" conaffinity="0"/>'
        )
        lane_geoms.append(
            f'<geom name="lane_pad_{dog_idx}" type="box" pos="0.28 {y:.3f} 0.016" '
            f'size="0.20 0.18 0.008" rgba="0.2 0.2 0.2 0.55" '
            'contype="0" conaffinity="0"/>'
        )

    return f"""
<mujoco model="go2 running competition">
  <option timestep="0.004" integrator="Euler"/>
  <visual>
    <headlight diffuse=".48 .48 .45" ambient=".18 .18 .18" specular=".18 .18 .18"/>
    <global azimuth="90" elevation="-25" offwidth="1920" offheight="1080"/>
    <map znear=".02" zfar="80"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".90 .95 1" rgb2=".62 .72 .86" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1=".58 .60 .58" rgb2=".47 .50 .48" markrgb=".24 .26 .25" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="18 5" reflectance="0"/>
    <material name="rail_dark" rgba=".05 .06 .06 1"/>
  </asset>
  <worldbody>
    <light name="key_light" pos="{finish_x * 0.35:.3f} {-half_width * 0.65:.3f} 5.0" diffuse=".48 .46 .40"/>
    <light name="rim_light" pos="{finish_x * 0.55:.3f} {half_width * 0.65:.3f} 3.0" diffuse=".28 .35 .45"/>
    <geom name="floor" size="{finish_x + 5.0:.3f} {half_width + 2.0:.3f} 0.01" type="plane" material="groundplane" contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>
    <geom name="start_line" type="box" pos="0 0 0.016" size="0.055 {half_width:.3f} 0.008" rgba="0.05 0.70 0.24 1" contype="0" conaffinity="0"/>
    <geom name="finish_line" type="box" pos="{finish_x:.3f} 0 0.018" size="0.070 {half_width:.3f} 0.010" rgba="0.86 0.10 0.10 1" contype="0" conaffinity="0"/>
    <geom name="near_rail" type="box" pos="{finish_x * 0.5:.3f} {-half_width:.3f} 0.035" size="{finish_x * 0.5:.3f} 0.018 0.025" material="rail_dark" contype="0" conaffinity="0"/>
    <geom name="far_rail" type="box" pos="{finish_x * 0.5:.3f} {half_width:.3f} 0.035" size="{finish_x * 0.5:.3f} 0.018 0.025" material="rail_dark" contype="0" conaffinity="0"/>
    {"".join(lane_geoms)}
  </worldbody>
</mujoco>
""".strip()


def build_race_model(
    *,
    num_dogs: int,
    lane_spacing: float,
    track_length_m: float,
    colors: list[str] | None = None,
    robot_tint_strength: float = 0.32,
    asset_model_dir: str | Path | None = None,
):
    """Compile a MuJoCo model containing `num_dogs` prefixed Go2 robots."""
    if num_dogs < 1:
        raise ValueError("num_dogs must be positive.")
    if num_dogs > 10:
        raise ValueError("The race renderer supports at most 10 dogs.")

    import mujoco

    asset_root = resolve_go2_asset_model_dir(asset_model_dir)
    parent = mujoco.MjSpec.from_string(_parent_scene_xml(num_dogs, lane_spacing, track_length_m))
    for dog_idx in range(num_dogs):
        y = (dog_idx - (num_dogs - 1) * 0.5) * lane_spacing
        child = mujoco.MjSpec.from_file(GO2_BODY_XML.as_posix())
        child.modelfiledir = asset_root.as_posix()
        child.meshdir = "assets"
        frame = parent.worldbody.add_frame(pos=[0.0, y, 0.0])
        parent.attach(child, prefix=f"dog{dog_idx}_", frame=frame)

    model = parent.compile()
    if model.nq != 19 * num_dogs or model.nu != 12 * num_dogs:
        raise RuntimeError(f"Unexpected race model dimensions: nq={model.nq}, nu={model.nu}, dogs={num_dogs}")
    if colors:
        tint_dogs(model, colors, blend=robot_tint_strength)
        tint_lanes(model, colors)
    return model


def tint_dogs(model: Any, colors: list[str], *, blend: float = 0.32) -> None:
    """Subtly tint the visual mesh materials while preserving the Go2 look."""
    import mujoco

    blend = float(np.clip(blend, 0.0, 1.0))
    for dog_idx, color in enumerate(colors):
        rgba = _hex_to_rgba(color)
        if rgba is None:
            continue
        prefix = f"dog{dog_idx}_"
        for mat_id in range(model.nmat):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id) or ""
            if name.startswith(prefix) and not name.endswith("black"):
                original = np.asarray(model.mat_rgba[mat_id], dtype=np.float32)
                tinted = original * (1.0 - blend) + rgba * blend
                tinted[3] = original[3]
                model.mat_rgba[mat_id] = tinted


def tint_lanes(model: Any, colors: list[str]) -> None:
    """Use policy colors as small lane accents without covering the track."""
    import mujoco

    for dog_idx, color in enumerate(colors):
        rgba = _hex_to_rgba(color)
        if rgba is None:
            continue
        accent = rgba.copy()
        accent[3] = 0.72
        for geom_name in (f"lane_{dog_idx}", f"lane_pad_{dog_idx}"):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                model.geom_rgba[geom_id] = accent


def _pack_center_x(trajectories_qpos: np.ndarray, step_idx: int) -> float:
    x_positions = trajectories_qpos[:, step_idx, 0]
    return float(np.clip(np.median(x_positions), 0.6, max(0.7, np.max(x_positions) + 0.5)))


def _configure_camera(
    camera: Any,
    *,
    profile: str,
    trajectories_qpos: np.ndarray,
    step_idx: int,
    lane_spacing: float,
    track_length_m: float,
) -> None:
    num_dogs = int(trajectories_qpos.shape[0])
    pack_x = _pack_center_x(trajectories_qpos, step_idx)
    lane_width = max(1.0, (num_dogs - 1) * lane_spacing)

    if profile == "overview":
        camera.lookat[:] = np.asarray([track_length_m * 0.45, 0.0, 0.35])
        camera.distance = max(5.5, lane_width * 0.95, track_length_m * 0.48)
        camera.azimuth = 90.0
        camera.elevation = -34.0
    elif profile == "close":
        camera.lookat[:] = np.asarray([pack_x, 0.0, 0.34])
        camera.distance = max(4.2, lane_width * 0.62)
        camera.azimuth = 88.0
        camera.elevation = -23.0
    else:
        camera.lookat[:] = np.asarray([pack_x + 0.15, 0.0, 0.34])
        camera.distance = max(5.0, lane_width * 0.72)
        camera.azimuth = 90.0
        camera.elevation = -28.0


def render_race_video(
    *,
    trajectories_qpos: np.ndarray,
    output_path: Path,
    colors: list[str],
    fps: int,
    render_every: int,
    width: int,
    height: int,
    lane_spacing: float,
    track_length_m: float,
    camera_profile: str = "showcase",
    robot_tint_strength: float = 0.32,
    asset_model_dir: str | Path | None = None,
) -> Path:
    """Render synchronized policy trajectories into one multi-dog video."""
    import mujoco

    trajectories_qpos = np.asarray(trajectories_qpos, dtype=np.float64)
    if trajectories_qpos.ndim != 3 or trajectories_qpos.shape[-1] != 19:
        raise ValueError("trajectories_qpos must have shape [num_policies, steps, 19].")

    num_dogs, num_steps, _ = trajectories_qpos.shape
    model = build_race_model(
        num_dogs=num_dogs,
        lane_spacing=lane_spacing,
        track_length_m=track_length_m,
        colors=colors,
        robot_tint_strength=robot_tint_strength,
        asset_model_dir=asset_model_dir,
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
        _configure_camera(
            camera,
            profile=camera_profile,
            trajectories_qpos=trajectories_qpos,
            step_idx=step_idx,
            lane_spacing=lane_spacing,
            track_length_m=track_length_m,
        )
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
            # Some lab machines have MuJoCo rendering but no ffmpeg. Keep the
            # visualization path useful by falling back to a GIF that PIL can write.
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
