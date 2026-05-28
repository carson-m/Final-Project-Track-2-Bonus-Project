# Go2 Track 2 Bonus Project Starter

Starter code for the Track 2 final-project option: make a Unitree Go2 run as
far as possible around a 200 m oval track in MuJoCo.

This repo is intentionally only a starter. It starts from a weak baseline and
leaves policy and planner design to you.

## Choices

- Proposal-based final project.
- Go2 oval-track leaderboard route.
- Both, for bonus.

Read `docs/assignment_requirements.md`.

## Controller

```text
5D track observation -> [vx, vy, yaw_rate] -> Go2 low-level policy
```

Interface: `docs/controller_interface.md`

Leaderboard submissions must train a learned high-level planner. Keep the 5D
input and 3D output fixed; change the planner internals. The starter planner is
only an interface example. The official track geometry is fixed by the
evaluator.

## Colab

Open `notebooks/track_bonus_colab_template.ipynb`.

## Commands

Train or reuse a low-level checkpoint:

```bash
python train.py \
  --config configs/course_config.json \
  --stage both \
  --output-dir artifacts/low_level_train
```

Evaluate:

```bash
python run_track_bonus.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --planner-config configs/starter_planner.json \
  --output-dir artifacts/track_eval
```

This evaluates one submission at a time. Course staff will compare submitted
results for ranking.

## Explore

- Improve low-level turning and command tracking in `go2_pg_env/joystick.py`.
- Train a learned high-level planner inside `track_bonus/planner.py`.
- Use `results.json`, `leaderboard.csv`, and `race.mp4` to understand failures.

More: `docs/high_level_optimization_guide.md`

## Submission

Typical leaderboard submission:

```text
best_checkpoint/
planner_config.json
planner weights, if used
changed planner code, if any
submission.json
track_eval/results.json
optional track_eval/race.mp4
short report
```
