# Go2 Track 2 Bonus Project Starter

This repository is the starter baseline for the final-project track bonus. It
extends the HW1 Go2 MuJoCo Playground assignment with a 200 m oval track
evaluation, while keeping the low-level locomotion pipeline aligned with the
original homework.

The intended controller is hierarchical:

```text
high-level planner:
  track coordinates -> [vx, vy, yaw_rate]

low-level policy:
  proprioception + command -> 12 Go2 joint actions
```

The starter high-level planner is intentionally weak. It is only an interface
example so students can run the benchmark and then improve it.

## Track 2 Work Modes

Students may choose one of three routes:

- Proposal-based final project: execute the project from the earlier proposal,
  present it in class, and submit a final report.
- Go2 oval-track tournament: submit a low-level checkpoint plus high-level
  controller and compete on the shared 200 m track benchmark.
- Both: complete a proposal-based final project and also enter the tournament
  for bonus points.

Tournament ranking is based first on completing a full lap without leaving the
track. Completed laps are ranked by `finish_time`; incomplete runs are ranked
by `valid_distance_m`, the distance traveled before the first fall or boundary
violation.

## Important

This repo does not include a trained solution checkpoint, a successful planner,
teacher rollouts, rendered solution videos, or tuned full-lap artifacts.
Students should reuse their HW1 checkpoint or train a new low-level policy.

## Colab Workflow

In Colab, set:

```python
COURSE_REPO_URL = "https://github.com/WeijieLai1024/Final-Project-Track-2-Bonus-Project.git"
COURSE_REPO_BRANCH = "main"
COURSE_REPO_DIR = Path("/content/go2_track_bonus_repo")
```

Then follow the notebook:

```text
notebooks/track_bonus_colab_template.ipynb
```

The notebook clones this repo, installs MuJoCo Playground dependencies, copies
Go2 assets, optionally trains a low-level policy, runs the starter track
evaluation, and renders a video.

Read the full assignment requirements before changing code:

```text
docs/assignment_requirements.md
docs/track2_assignment_handout.md
docs/controller_interface.md
```

For how the high-level planner can be optimized, read:

```text
docs/high_level_optimization_guide.md
```

For instructor-side release checks, see:

```text
docs/instructor_release_checklist.md
```

## Low-Level Training

The low-level policy uses the same Brax PPO checkpoint format as HW1:

```bash
python train.py \
  --config configs/course_config.json \
  --stage both \
  --output-dir artifacts/low_level_train
```

The default baseline remains simple. A good project should improve command
tracking for curved running, especially yaw-rate commands.

## Starter Track Evaluation

Run the track bonus evaluation with a low-level checkpoint:

```bash
python run_track_bonus.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --planner-config configs/starter_planner.json \
  --output-dir artifacts/track_eval
```

For a quick non-rendered check:

```bash
python run_track_bonus.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --planner-config configs/starter_planner.json \
  --output-dir artifacts/track_eval_smoke \
  --duration-seconds 5 \
  --no-render
```

Outputs:

- `results.json`
- `leaderboard.csv`
- `race_rollouts.npz`
- `race.mp4` unless `--no-render` is used

## High-Level Controller Contract

To keep all submissions compatible with the official tournament runner, the
high-level controller interface is fixed:

```text
input:
  own robot qpos, base xy/yaw, track progress, lateral error,
  boundary margin, heading error, curvature, and time

output:
  [vx_mps, vy_mps, yaw_rate_radps]
```

Official command limits:

```text
vx_mps:          [0.00, 1.50]
vy_mps:          [-0.50, 0.50]
yaw_rate_radps:  [-1.50, 1.50]
```

Commands are clipped by `track_bonus/controller_interface.py`. The high-level
controller must not depend on other robots, future states, hidden simulator
internals, or manually edited evaluator outputs. The low-level checkpoint must
remain a Brax PPO checkpoint with actor `policy_obs_key = "state"`.

For the full contract, read:

```text
docs/controller_interface.md
```

## 10-Dog Tournament Visualization

The release infrastructure supports synchronized visualization for up to 10
Go2 entries. Scoring remains independent per policy; the renderer combines the
saved `race_rollouts.npz` files into one MuJoCo scene so different teams'
Python controllers never need to be imported into the same process.

Synthetic infrastructure demo:

```bash
python scripts/render_track_tournament.py \
  --demo-synthetic \
  --num-dogs 10 \
  --output-dir artifacts/ten_dog_demo
```

Real tournament rendering after each team has produced `track_eval/race_rollouts.npz`:

```bash
python scripts/render_track_tournament.py \
  --entries tournament_entries.json \
  --visual-lane-offsets \
  --output-dir artifacts/tournament_render
```

For 10 dogs, the compiled MuJoCo model reports `nq = 190` and `nu = 120`.

## Optional High-Level Search

The included high-level trainer is a small black-box parameter search:

```bash
python train_highlevel_starter.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --output-dir artifacts/highlevel_train \
  --iterations 8 \
  --population 12
```

This is a scaffold, not the expected final method. You may replace it with an
MLP, RL controller, residual controller, or another learned high-level policy,
as long as the runtime outputs `[vx, vy, yaw_rate]`.

The starter search uses `scores.composite_score` from `run_track_bonus.py` as
its reward signal. It does not backpropagate through MuJoCo.

## Track Metrics

The benchmark reports:

- `lap_completion`
- `valid_distance_m`
- `finish_time`
- `mean_progress_speed`
- `rms_lateral_error`
- `max_lateral_error`
- `min_boundary_margin_m`
- `fall`
- `boundary_violation`
- `energy_proxy`
- `foot_slip_proxy`

## Recommended Files To Read

```text
go2_pg_env/joystick.py
go2_pg_env/track.py
track_bonus/planner.py
run_track_bonus.py
train_highlevel_starter.py
docs/assignment_requirements.md
docs/track2_assignment_handout.md
docs/controller_interface.md
docs/high_level_optimization_guide.md
```

## Submission

Submit:

- `best_checkpoint/`
- `planner_config.json` or your planner file
- `submission.json`
- `track_eval/results.json`
- optional `track_eval/race.mp4`
- a short report explaining the low-level policy, high-level planner, training
  method, final metrics, and at least one failed idea.

## Student Modification Boundary

Students should mostly modify:

- `go2_pg_env/joystick.py`
- `configs/course_config.json`
- `track_bonus/planner.py`
- `train_highlevel_starter.py` or a new high-level training script
- planner config files

Students should usually not modify:

- metric names
- checkpoint restore logic
- rollout bundle field names
- renderer-only code
