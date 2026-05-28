# High-Level Planner Optimization Guide

## Current Baseline

The provided high-level planner is `StarterTrackPlanner` in
`track_bonus/planner.py`.

It is not learned. It is a small coordinate-feedback controller:

```text
qpos + track geometry -> [vx, vy, yaw_rate]
```

It computes:

- projection onto the 200 m oval centerline
- lateral error from the centerline
- current yaw
- lookahead track heading
- heading error
- local curvature feedforward

Then it outputs a conservative joystick command for the low-level policy.

The JSON parameters in `configs/starter_planner.json` control speed, lookahead,
heading correction, lateral correction, and command limits.

All high-level methods must keep the same runtime interface:

```text
own robot state + track geometry -> [vx_mps, vy_mps, yaw_rate_radps]
```

The official evaluator clips commands to the envelope defined in
`track_bonus/controller_interface.py`.

## Is The High-Level Baseline Trained?

Only the optional script `train_highlevel_starter.py` performs optimization.
It does **not** use gradient descent through MuJoCo. It uses black-box search:

1. sample planner parameters near the current best config
2. write a candidate `planner_config.json`
3. run `run_track_bonus.py --no-render`
4. read `results.json`
5. use `scores.composite_score` as the objective
6. keep the best candidate and sample around it again

This is deliberately simple. It gives students a working optimization loop
without giving away a solved planner.

## Where Is The Optimization Signal?

The optimization signal is the evaluation score:

```text
scores.composite_score
```

from `track_eval/results.json`.

That score is built from:

- lap completion
- valid distance before fall or boundary violation
- progress speed
- line keeping
- fall / boundary violation
- energy proxy
- foot slip proxy

So the "gradient" in the starter repo is not an analytic gradient. It is a
black-box reward signal: try parameters, run a rollout, compare scores, update
the parameter distribution.

## Why No Differentiable Gradient?

The full system includes:

- a restored Brax PPO low-level policy
- MJX/MuJoCo physics
- discontinuous events such as falls and boundary exits
- video/rendering outputs
- checkpoint restore code

Backpropagating through this whole evaluation loop is not the intended starter
baseline. For this project, black-box search and RL-style reward optimization
are acceptable and much easier to reason about.

## What Can Students Optimize?

### 1. Starter JSON Parameters

Easiest path:

- increase `speed_mps`
- increase `max_yaw_rate_radps`
- tune `lookahead_m`
- tune `k_heading`
- tune `k_lateral`
- keep `max_lateral_speed_mps` within what the low-level policy can track

Use:

```bash
python train_highlevel_starter.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --output-dir artifacts/highlevel_train \
  --iterations 8 \
  --population 12
```

This writes:

```text
artifacts/highlevel_train/best_planner_config.json
artifacts/highlevel_train/search_summary.json
```

### 2. Replace The Planner With An MLP

Students may create a learned high-level policy:

```text
features -> MLP -> [vx, vy, yaw_rate]
```

Useful features:

- normalized global x/y
- sine/cosine lap phase
- lateral error / track half-width
- boundary margin / track half-width
- sine/cosine heading error
- curvature
- lookahead heading error

Possible training methods:

- supervised learning from a hand-designed controller
- dataset aggregation from rollouts
- black-box optimization over MLP weights for a small network
- policy gradient / RL on the high-level controller

### 3. Train The Low-Level Policy To Track Turns

The high-level planner can only output commands that the low-level policy can
track. If the low-level policy was trained only on forward `vx`, then increasing
high-level yaw commands will not solve the lap.

Students should improve `go2_pg_env/joystick.py` and
`configs/course_config.json` so stage 2 includes nonzero `vy` and `yaw_rate`.

The intended low-level improvement is:

```text
command distribution:
  vx forward range
  vy small lateral corrections
  yaw_rate left/right turns
```

Then the high-level planner has a real actuator-like interface for turning.

## What Metrics Should Students Watch?

Use `results.json`:

- If `lap_completion` is low: speed is too low, high-level steering is poor, or
  low-level command tracking is weak.
- If `valid_distance_m` is low: the tournament run ended early or made little
  progress before timeout.
- If `fall` is true: commands are too aggressive or low-level stability is not
  good enough.
- If `boundary_violation` is true: high-level line keeping is poor, or yaw
  correction arrives too late.
- If `rms_lateral_error` is high but no boundary violation occurs: planner is
  safe but sloppy.
- If `energy_proxy` or `foot_slip_proxy` is high: gait is inefficient or
  slipping during turns.

## Recommended Student Optimization Path

1. Run the starter planner with an existing low-level checkpoint.
2. Inspect whether failure is fall, boundary exit, or simply low progress.
3. Expand low-level stage 2 command sampling to include yaw-rate commands.
4. Retrain low-level PPO.
5. Run starter high-level search for short evaluations.
6. Evaluate the best config for a longer duration.
7. Replace the high-level planner with an MLP or RL policy if parameter search
   plateaus.
8. Submit the best checkpoint, planner, metrics, and report.

The core idea is that high-level optimization and low-level training must meet
in the middle: the planner should ask for useful commands, and the locomotion
policy must actually be able to execute them.
