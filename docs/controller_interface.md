# Track Controller Interface Contract

This contract keeps all Track 2 submissions compatible with the official
evaluator and the later 10-dog tournament renderer.

## 1. System Architecture

Each tournament entry has two layers:

```text
high-level track controller:
  own robot state + track geometry -> [vx, vy, yaw_rate]

low-level Go2 locomotion policy:
  proprioception + command -> 12 joint actions
```

The official tournament does not simulate dog-dog collisions for scoring. Each
entry is rolled out independently, then the saved `qpos` trajectories are
synchronized in one MuJoCo scene for visualization. This makes ranking fair and
also avoids Python dependency conflicts between teams.

## 2. High-Level Input

The high-level controller may use only the current robot's own state and track
geometry. The standardized observation fields are:

- `t`: rollout time in seconds.
- `qpos`: current 19-dimensional MuJoCo generalized position for this robot.
- `base_xy`: global base position `[x, y]`.
- `base_yaw`: global heading angle.
- `centerline_s_m`: projected progress along the 200 m centerline.
- `lap_fraction`: `centerline_s_m / 200`.
- `lateral_error_m`: signed distance from the centerline.
- `boundary_margin_m`: distance to the nearest lane boundary.
- `track_heading_rad`: tangent direction of the centerline.
- `heading_error_rad`: track heading minus robot yaw.
- `curvature_1pm`: local centerline curvature.
- `track_length_m`: default `200.0`.
- `track_half_width_m`: default `2.0`.

The helper implementation is in:

```text
track_bonus/controller_interface.py
```

Students may compute the same features themselves, but the controller should
not depend on other robots, future states, hidden simulator internals, or
manually edited evaluator outputs.

## 3. High-Level Output

The output must be exactly:

```text
[vx_mps, vy_mps, yaw_rate_radps]
```

with shape `(3,)`.

The official evaluator sanitizes commands with:

```text
vx_mps:          [0.00, 1.50]
vy_mps:          [-0.50, 0.50]
yaw_rate_radps:  [-1.50, 1.50]
```

Non-finite commands or wrong-shaped commands are invalid. Commands outside the
range are clipped before they reach the low-level policy.

## 4. Low-Level Policy Requirement

The low-level checkpoint must remain a Brax PPO checkpoint compatible with the
HW1-style Go2 joystick environment:

- checkpoint directory contains `ppo_network_config.json`
- actor uses `policy_obs_key = "state"`
- actor does not require `privileged_state`
- action is the standard 12-dimensional Go2 joint target offset

Students can retrain or improve this low-level policy, but the runtime
checkpoint format must stay compatible with `run_track_bonus.py`.

## 5. Submission Compatibility

For the starter repository, the default high-level artifact is:

```text
planner_config.json
```

loaded by `StarterTrackPlanner`. Students can replace the planner logic during
development, but the final submission must still be runnable by the command
listed in `submission.json`.

Recommended tournament manifest for instructors:

```json
{
  "entries": [
    {
      "name": "team_a",
      "rollout_npz": "team_a/track_eval/race_rollouts.npz",
      "color": "#2563EB"
    }
  ]
}
```

The manifest uses rollout files rather than importing 10 teams' Python
controllers into one process. This is the key design choice that prevents
controller conflicts.

## 6. 10-Dog Visualization

The renderer supports at most 10 entries. Internally it attaches prefixed Go2
models into one MuJoCo model:

```text
dog0_..., dog1_..., ..., dog9_...
```

For 10 robots the compiled model should satisfy:

```text
nq = 19 * 10 = 190
nu = 12 * 10 = 120
```

Demo command:

```bash
python scripts/render_track_tournament.py \
  --demo-synthetic \
  --num-dogs 10 \
  --track-half-width-m 2.0 \
  --output-dir artifacts/ten_dog_demo
```

To combine real evaluated submissions:

```bash
python scripts/render_track_tournament.py \
  --entries tournament_entries.json \
  --visual-lane-offsets \
  --output-dir artifacts/tournament_render
```

`--visual-lane-offsets` spreads trajectories only for readability in the video.
It does not change the saved per-team scoring results.
