# Track 2 Requirements

Goal: run Go2 as far as possible around a 200 m oval track in MuJoCo.

## Options

- Proposal-based final project.
- Go2 oval-track leaderboard route.
- Both, for bonus.

## Leaderboard Route

```text
5D track observation -> [vx, vy, yaw_rate] -> Go2 low-level policy
```

Use `docs/controller_interface.md`. The low-level checkpoint should stay
compatible with the HW1 Brax PPO format.
This repo evaluates one submission at a time; ranking compares submitted
outputs.

Leaderboard submissions must train a learned high-level planner for this fixed
interface. The provided starter planner is only a weak baseline for debugging.
Keep the 5D input and 3D output fixed; change the planner internals. The
official scene is fixed: 200 m centerline, 18.25 m turn radius, and 2.0 m half
width. Do not change the track geometry to improve score.

## Allowed

- Reuse a HW1 checkpoint.
- Retrain or modify the low-level Go2 policy.
- Train a learned high-level controller for the fixed track interface.

## Not Allowed

- Hard-code benchmark results.
- Delete or rename required output fields.
- Bypass the low-level policy with prewritten joint trajectories.
- Use privileged actor observations beyond normal `state`.
- Change the official track geometry.
- Submit only the hand-written starter planner as the final method.

## Ranking

- Completed laps rank before incomplete runs.
- Completed laps rank by lower `finish_time`.
- Incomplete runs rank by higher `valid_distance_m`.
- Ties use failures, boundary margin, lateral error, slip, and energy.

## Outputs And Metrics

Outputs: `results.json`, `leaderboard.csv`, `race_rollouts.npz`, optional
`race.mp4`. In this repo these files describe one evaluated submission.

Main metrics: `lap_completion`, `valid_distance_m`, `finish_time`, `fall`,
`boundary_violation`.

## Submission

```text
best_checkpoint/
planner_config.json
planner weights, if used
changed planner code, if any
submission.json
track_eval/results.json
optional track_eval/race.mp4
short_report.pdf
```

Report briefly: low-level changes, learned high-level planner design, where its
weights are stored, training method, final metrics, and one failed idea.

## Grading

Proposal projects use the final-project rubric. Leaderboard entries use the
same distribution as the final-project route. Ranking uses track distance,
completion time, method quality, analysis, presentation, and reproducibility.
Doing both routes is eligible for bonus.
