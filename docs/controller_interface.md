# Controller Interface

High-level controllers must use this interface so submissions share the same
contract.

```text
5D track observation -> [vx, vy, yaw_rate] -> Go2 low-level policy
```

## Input

```text
[
  lap_fraction,
  lateral_error_norm,
  boundary_margin_norm,
  heading_error_rad,
  curvature_norm
]
```

Defined in `track_bonus/controller_interface.py`.
These features are computed from the official 200 m oval used by the evaluator.
The planner may use the features, but should not redefine the track geometry.
Leaderboard submissions should train a learned policy that consumes exactly
this observation vector.

## Output

```text
[vx_mps, vy_mps, yaw_rate_radps]
```

Shape must be `(3,)`. Values must be finite. The evaluator does not clip or
rescale commands.

The default evaluator loads `StarterTrackPlanner.load(planner_config)` and then
calls `planner.command(track_observation, t)`. If you replace the controller
implementation, keep that entry point.
Learned weights can be loaded inside `StarterTrackPlanner.load(...)`; keep any
weight path in `planner_config.json`.
The evaluator validates the official track fields in the planner config and
uses its own fixed track for reset, scoring, and rendering.

## Checkpoint

Use the HW1 Brax PPO format:

- `ppo_network_config.json` exists
- actor `policy_obs_key = "state"`
- `state` observation shape is 48
- no `privileged_state` actor
- 12-dimensional action
