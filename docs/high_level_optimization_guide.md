# High-Level Planner Training

Leaderboard submissions must train a learned high-level planner. Keep the
5D input and 3D output fixed; replace the internals with your learned policy.
The starter planner is weak on purpose and only shows the interface.

## Starter Search Command

```bash
python train_highlevel_starter.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --output-dir artifacts/highlevel_train \
  --iterations 8 \
  --population 12
```

This is a minimal black-box search over starter planner parameters. It is useful
for debugging the loop, but it is not a final learned planner by itself.

## What To Train

Train a policy with parameters `theta`:

```text
track_observation[5] -> planner_theta -> [vx, vy, yaw_rate]
```

The training signal is the evaluator output in `results.json`, especially
`scores.composite_score`, `lap_completion`, `valid_distance_m`, boundary
violations, falls, and lateral error.

## Minimal Training Loop

1. Put your learned planner in `track_bonus/planner.py`.
2. Keep `StarterTrackPlanner.load(planner_config)` and
   `planner.command(track_observation, t)`.
3. Run short no-render rollouts with `run_track_bonus.py`.
4. Update planner parameters using your method.
5. Run the full evaluation and include your training method in the report.

Your method can be PPO/RL, CEM, evolution strategies, black-box optimization of
an MLP, or supervised pretraining followed by rollout optimization. If your
planner uses weights, submit the weights file and make `planner_config.json`
point to it.

## Possible Directions

- Replace the internals of `track_bonus/planner.py` with an MLP, RL policy, or
  another trained policy while keeping the same load/command entry point.
- Train the low-level policy to track nonzero `vy` and `yaw_rate`.
- Use staged evaluation: straight, turn entry, turn middle, turn exit, full lap.
- Keep the official track geometry fixed. The evaluator rejects mismatched
  track fields.

## Useful Metrics

- `lap_completion`, `valid_distance_m`, `finish_time`
- `fall`, `boundary_violation`
- lateral error, slip, energy

## Loop

1. Run starter eval.
2. Inspect `results.json` and `race.mp4`.
3. Improve low-level tracking or high-level planner.
4. Re-evaluate.
