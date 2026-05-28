# Instructor Release Checklist

Use this checklist before publishing the Track 2 starter repository.

## 1. Student-Facing Documents

Confirm these files exist and are readable:

- `README.md`
- `docs/track2_assignment_handout.md`
- `docs/assignment_requirements.md`
- `docs/controller_interface.md`
- `docs/high_level_optimization_guide.md`
- `notebooks/track_bonus_colab_template.ipynb`

The handout should explain:

- proposal-based final project route
- Go2 oval-track tournament route
- combined route for bonus points
- tournament ranking by full-lap `finish_time` or incomplete-run
  `valid_distance_m`
- fixed high-level controller input/output contract
- 10-dog rollout-based tournament rendering path
- deliverables and reproducibility expectations

## 2. Starter Repository Boundaries

The public starter should not include:

- trained solution checkpoints
- optimized planners
- teacher rollouts
- rendered solution videos
- previous complete-lap artifacts
- private outputs or grading logs

Suggested release check:

```bash
find . -maxdepth 3 \( \
  -name outputs -o \
  -name baselines -o \
  -name best_checkpoint -o \
  -name '*.mp4' -o \
  -name '*.gif' -o \
  -name planner.npz -o \
  -name race_rollouts.npz \
\) -print
```

This command should not print any release artifact that gives away a solved
controller.

## 3. Code Smoke Tests

Run:

```bash
python -m pytest -q tests
python -m py_compile \
  run_track_bonus.py \
  train_highlevel_starter.py \
  scripts/render_track_tournament.py \
  track_bonus/controller_interface.py \
  track_bonus/planner.py \
  track_bonus/scoring.py \
  go2_pg_env/track.py
```

Notebook sanity check:

```bash
python - <<'PY'
from pathlib import Path
import nbformat

path = Path("notebooks/track_bonus_colab_template.ipynb")
nb = nbformat.read(path, as_version=4)
print("cells", len(nb.cells))
assert any("track2_assignment_handout" in cell.source for cell in nb.cells)
assert any("controller_interface" in cell.source for cell in nb.cells)
assert any("CHECKPOINT_DIR does not exist" in cell.source for cell in nb.cells)
PY
```

10-dog model compile check after Go2 assets are available:

```bash
python scripts/render_track_tournament.py \
  --demo-synthetic \
  --num-dogs 10 \
  --output-dir artifacts/ten_dog_compile_check \
  --no-render
```

The summary should report `model_nq = 190` and `model_nu = 120`.

## 4. Expected Student Commands

Low-level training:

```bash
python train.py \
  --config configs/course_config.json \
  --stage both \
  --output-dir artifacts/low_level_train
```

Starter evaluation:

```bash
python run_track_bonus.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --planner-config configs/starter_planner.json \
  --output-dir artifacts/track_eval
```

Short smoke evaluation:

```bash
python run_track_bonus.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --planner-config configs/starter_planner.json \
  --output-dir artifacts/track_eval_smoke \
  --duration-seconds 5 \
  --no-render
```

Optional high-level search:

```bash
python train_highlevel_starter.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --output-dir artifacts/highlevel_train \
  --iterations 8 \
  --population 12
```

## 5. Publication Check

After pushing to GitHub, make a fresh clone and repeat the code smoke tests.
The notebook should be able to run setup and dry-run cells without a prepared
checkpoint. Full training and rendered evaluation require MuJoCo/Brax
dependencies and a real low-level checkpoint.
