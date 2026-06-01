"""Cross-Entropy Method (CEM) optimizer for MLP High-Level Track Planner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

# Import directly from the course modules to bypass subprocess recompilation!
import run_track_bonus
from course_common import load_json, lazy_import_stack, set_runtime_env
from test_policy import load_policy_with_workaround
from track_bonus.official_track import official_track
from track_bonus.planner import StarterTrackPlanner
from track_bonus.scoring import compute_track_bonus_metrics, score_track_bonus

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Path to low-level locomotion checkpoint.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "course_config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "highlevel_mlp_cem")
    parser.add_argument("--iterations", type=int, default=12, help="Number of evolutionary generations.")
    parser.add_argument("--population", type=int, default=16, help="Candidates evaluated per generation.")
    parser.add_argument("--elite-frac", type=float, default=0.25, help="Fraction of population kept as elites.")
    parser.add_argument("--eval-seconds", type=float, default=45.0, help="Simulation time window per evaluation.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension layout size for the MLP.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def vector_to_weights(vec: np.ndarray, input_dim: int = 5, hidden_dim: int = 32, output_dim: int = 3) -> dict[str, np.ndarray]:
    w1_size = input_dim * hidden_dim
    b1_size = hidden_dim
    w2_size = hidden_dim * output_dim
    b2_size = output_dim

    idx = 0
    w1 = vec[idx : idx + w1_size].reshape(input_dim, hidden_dim)
    idx += w1_size
    b1 = vec[idx : idx + b1_size]
    idx += b1_size
    w2 = vec[idx : idx + w2_size].reshape(hidden_dim, output_dim)
    idx += w2_size
    b2 = vec[idx : idx + b2_size]
    
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_dim = 5
    hidden_dim = args.hidden_dim
    output_dim = 3
    num_params = (input_dim * hidden_dim) + hidden_dim + (hidden_dim * output_dim) + output_dim

    # --- 1. INITIALIZE JAX/BRAX ONCE ---
    print("Initializing JAX and loading MuJoCo/Brax environment (This may take 1-2 minutes)...")
    force_cpu = bool(args.force_cpu)
    set_runtime_env(force_cpu=force_cpu)
    stack = lazy_import_stack()
    course_cfg = load_json(args.config)
    course_cfg["runtime_overrides"] = {}
    
    num_steps = int(round(float(args.eval_seconds) / float(course_cfg["control"]["ctrl_dt"])))
    track = official_track()
    
    env = run_track_bonus._make_env(stack, course_cfg, "stage_2", episode_steps=num_steps)
    policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
    if not force_cpu:
        policy = stack["jax"].jit(policy)
    print("Environment loaded and compiled successfully!\n")
    # ------------------------------------

    mu = np.zeros(num_params, dtype=np.float32)
    sigma = np.ones(num_params, dtype=np.float32) * 0.15
    best_score = -1.0
    history = []
    num_elites = max(1, int(args.population * args.elite_frac))

    print(f"Starting MLP CEM Optimization: Total Parameters={num_params}, Elites={num_elites}/{args.population}\n")

    for iteration in range(args.iterations):
        t0 = time.time()
        
        candidates = []
        for i in range(args.population):
            if i == 0 and iteration > 0:
                candidates.append(mu.copy())
            else:
                candidates.append(mu + rng.normal(0.0, sigma))
        
        scores = [0.0] * args.population
        
        # IN-PROCESS EVALUATION (Extremely Fast)
        for cand_idx, vec in enumerate(candidates):
            cand_dir = args.output_dir / "candidates" / f"iter_{iteration:02d}_cand_{cand_idx:02d}"
            cand_dir.mkdir(parents=True, exist_ok=True)
            
            weights_dict = vector_to_weights(vec, input_dim, hidden_dim, output_dim)
            weights_path = cand_dir / "planner_weights.npz"
            np.savez(weights_path, **weights_dict)
            
            config_payload = {"planner_type": "learned_mlp", "weights_path": str(weights_path.resolve()), "stand_seconds": 1.0}
            planner_config_path = cand_dir / "planner_config.json"
            planner_config_path.write_text(json.dumps(config_payload, indent=2))
            
            # Load the planner for this candidate
            planner = StarterTrackPlanner.load(planner_config_path)
            
            # Run the rollout directly in memory
            try:
                result = run_track_bonus.rollout(
                    stack=stack,
                    env=env,
                    policy=policy,
                    planner=planner,
                    track=track,
                    num_steps=num_steps,
                    seed=int(args.seed),
                    start_s=0.0,
                    force_cpu=force_cpu,
                )
                metrics = compute_track_bonus_metrics(result, track)
                cand_score = float(score_track_bonus(metrics)["composite_score"])
            except Exception as e:
                print(f"Candidate {cand_idx} failed: {e}")
                cand_score = -1.0

            scores[cand_idx] = cand_score
            
            # Update overall best global archive tracker
            if cand_score > best_score:
                best_score = cand_score
                np.savez(args.output_dir / "planner_weights.npz", **weights_dict)
                (args.output_dir / "planner_config.json").write_text(json.dumps(config_payload, indent=2))
                (args.output_dir / "best_score.json").write_text(json.dumps({
                    "score": best_score, "iteration": iteration, "candidate": cand_idx
                }, indent=2))

        # Sort indices by fitness performance desc
        sorted_indices = np.argsort(scores)[::-1]
        elites = [candidates[idx] for idx in sorted_indices[:num_elites]]
        
        # Refit mean and variance distributions based on the elite parameter sets
        elites_arr = np.array(elites)
        mu = np.mean(elites_arr, axis=0)
        sigma = np.std(elites_arr, axis=0) + max(0.01, 0.1 * (0.85 ** iteration))
        
        dt = time.time() - t0
        print(f"[Gen {iteration:02d}] Best Gen Score: {max(scores):.3f} | Best Global Score: {best_score:.3f} | Step Time: {dt:.1f}s")
        
        history.append({
            "iteration": iteration,
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "best_global": float(best_score)
        })
        
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nOptimization Finished! Best Score Found: {best_score:.4f}")
    print(f"Deployment config and weights located at: {args.output_dir}")


if __name__ == "__main__":
    main()