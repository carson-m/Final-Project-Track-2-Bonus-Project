# Track 2 Release Handout: Go2 Oval Tournament Or Final Project

## 1. Big Picture

Track 2 is a final-project-style extension of the Go2 locomotion homework. You
may use this repository in one of three ways:

1. Proposal-based final project.
2. Go2 oval-track tournament.
3. Both, for bonus points.

The tournament route is designed for students who want a concrete robotics
benchmark with a public leaderboard. The proposal route is designed for
students who want to pursue their own research idea, based on the proposal they
submitted earlier in the course.

## 2. Option A: Proposal-Based Final Project

Choose this option if your team wants to execute the project described in your
final project proposal. Your project should still be robotics-focused and
should include a reproducible implementation, experiments, and analysis.

Expected deliverables:

- Source code and configuration files.
- A class presentation or demo.
- A final project report.
- Clear instructions for reproducing the main experiments.
- Any trained checkpoints, videos, plots, or logs needed to support the report.

Suggested score distribution, matching the final-project style:

- 20% proposal alignment and problem framing.
- 30% technical implementation.
- 25% experiments, evaluation, and evidence.
- 10% class presentation or live demo.
- 15% final report quality and reproducibility.

## 3. Option B: Go2 Oval-Track Tournament

Choose this option if your team wants to compete on the shared 200 m Go2 track
benchmark. Your controller must keep the robot inside the lane while making as
much forward progress as possible.

The default architecture is hierarchical:

```text
track-aware high-level controller:
  robot pose and track geometry -> [vx, vy, yaw_rate]

HW1-style low-level locomotion policy:
  proprioception + command -> 12 joint actions
```

The starter repository provides a weak high-level planner and HW1-compatible
low-level training code. It does not provide a solved checkpoint, optimized
planner, teacher data, or successful full-lap artifact.

Tournament ranking:

- Completed full laps rank ahead of incomplete runs.
- Among completed laps, lower `finish_time` ranks higher.
- Among incomplete runs, larger `valid_distance_m` ranks higher.
- `valid_distance_m = 200 * lap_completion` and is measured before the first
  fall or boundary violation.
- Ties are broken by fewer terminal failures, larger boundary margin, lower
  lateral error, lower foot slip, and lower energy proxy.

Suggested tournament score distribution:

- 45% official benchmark performance and leaderboard rank.
- 20% technical method quality: low-level training, high-level controller,
  curriculum, and design justification.
- 15% evaluation analysis: videos, failure cases, ablations, and metric
  interpretation.
- 10% class presentation or demo.
- 10% reproducibility: checkpoint, planner, configs, report, and exact command
  used to produce `results.json`.

## 4. Option C: Do Both

Teams may complete a proposal-based final project and also submit a tournament
entry. This is the most ambitious route and is eligible for bonus points. The
default release assumption is up to 10 bonus points, with the exact cap and
application left to the instructor.

For this route, submit both deliverable sets:

- The proposal-based final project report and presentation.
- The tournament checkpoint, planner, evaluation artifacts, and short method
  description.

## 5. Tournament Objective

The robot starts on a standard 200 m oval track. The lane has a centerline and
finite width. The controller receives track-coordinate information at the
high-level layer and must output local velocity commands for the low-level Go2
policy.

Your goal is:

1. Stay upright.
2. Stay inside the track boundary.
3. Make forward progress along the oval.
4. Complete a full lap as quickly as possible.
5. Avoid excessive lateral error, slipping, and energy use.

The benchmark is not a pure top-speed contest. A controller that runs slowly but
stays safely inside the lane can beat a controller that briefly runs fast and
then exits the track.

## 6. What Students Should Improve

Strong tournament submissions usually improve both layers:

- Low-level policy: better tracking for `vx`, `vy`, and `yaw_rate`, especially
  during curves.
- High-level controller: better conversion from track coordinates into local
  commands.
- Curriculum: staged training or evaluation that covers straight running,
  entering a curve, staying in a curve, and leaving a curve.

Acceptable high-level approaches include:

- Tuned JSON planner parameters.
- Black-box search over planner parameters.
- Learned MLP controller.
- RL high-level policy.
- Residual controller on top of the starter planner.
- MPC-style or optimization-based controller, as long as it produces the
  required `[vx, vy, yaw_rate]` command at runtime.

All approaches must obey the same controller interface:

- input is the robot's own pose/state plus track geometry features
- output is exactly `[vx_mps, vy_mps, yaw_rate_radps]`
- official command limits are `[0.00, 1.50]`, `[-0.50, 0.50]`, and
  `[-1.50, 1.50]`
- the controller cannot depend on other robots, future states, or hidden
  simulator internals

Read:

```text
docs/controller_interface.md
```

## 7. Required Tournament Artifacts

Submit a folder or zip with:

- `best_checkpoint/`
- `planner_config.json` or your planner file
- `submission.json`
- `track_eval/results.json`
- optional `track_eval/race.mp4`
- `short_report.pdf`

Your `submission.json` should identify which route you chose:

```json
{
  "team_name": "your_team_name",
  "track2_option": "tournament",
  "checkpoint_dir": "best_checkpoint",
  "planner": "planner_config.json",
  "training_steps": 15000000,
  "evaluation_command": "python run_track_bonus.py --checkpoint-dir best_checkpoint --planner-config planner_config.json --output-dir track_eval",
  "notes": "Brief method summary"
}
```

Use `"track2_option": "proposal_final_project"` for the proposal route and
`"track2_option": "both"` if you complete both routes.

## 8. Colab Path

The public notebook is:

```text
notebooks/track_bonus_colab_template.ipynb
```

It walks through:

1. Cloning this repository.
2. Installing pinned MuJoCo Playground dependencies.
3. Copying Go2 assets.
4. Inspecting the HW1-style low-level environment.
5. Training or reusing a low-level `best_checkpoint/`.
6. Running the starter track evaluation.
7. Optionally running the high-level search scaffold.
8. Packaging the expected submission files.

The notebook intentionally skips full evaluation if no checkpoint exists. This
allows everyone to run the setup cells first, then plug in a HW1 checkpoint or
train a new low-level policy.

The repository also includes a 10-dog visualization utility:

```bash
python scripts/render_track_tournament.py \
  --demo-synthetic \
  --num-dogs 10 \
  --output-dir artifacts/ten_dog_demo
```

Real tournament videos should combine the independently evaluated
`race_rollouts.npz` files from each team. This avoids conflicts between
different teams' Python high-level controllers.

## 9. Academic Integrity And Benchmark Boundaries

Do not hard-code benchmark outputs or edit required metric names. Do not bypass
the low-level policy by directly writing joint trajectories into the evaluator.
Do not use privileged simulator signals in the low-level actor beyond the normal
HW1 `state` observation.

It is fine for the high-level controller to use global track coordinates. That
is the point of Track 2. If you discuss real-robot deployment in your report,
explain what localization system would be needed to provide those coordinates.

## 10. Final Verification

Before submission, run:

```bash
python run_track_bonus.py \
  --checkpoint-dir best_checkpoint \
  --planner-config planner_config.json \
  --output-dir track_eval
```

Then check:

- `track_eval/results.json` exists.
- `track_eval/leaderboard.csv` exists.
- `track_eval/race_rollouts.npz` exists.
- `track_eval/race.mp4` exists unless you intentionally used `--no-render`.
- `results.json` reports `lap_completion`, `valid_distance_m`,
  `finish_time`, `fall`, and `boundary_violation`.
