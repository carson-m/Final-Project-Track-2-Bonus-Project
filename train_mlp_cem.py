"""Pure-JAX Vectorized (vmap) CEM optimizer for MLP High-Level Track Planner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

# Course imports
from course_common import (
    load_json, 
    lazy_import_stack, 
    set_runtime_env,
    ensure_environment_available,
    build_env_overrides,
    get_ppo_config,
    apply_stage_config,
)
from test_policy import load_policy_with_workaround

ROOT = Path(__file__).resolve().parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Path to low-level locomotion checkpoint.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "course_config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "highlevel_mlp_cem_vmap")
    parser.add_argument("--iterations", type=int, default=50, help="Number of evolutionary generations.")
    # --- STRATEGY 2: SCALE UP POPULATION SIZES TO EXPLOIT VMAP PARALLELISM ---
    parser.add_argument("--population", type=int, default=128, help="Candidates evaluated per generation.")
    parser.add_argument("--elite-frac", type=float, default=0.20, help="Fraction of population kept as elites.")
    parser.add_argument("--eval-seconds", type=float, default=45.0, help="Simulation time window per evaluation.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension layout size for the MLP.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()

def vector_to_weights(vec: np.ndarray, input_dim: int = 8, hidden_dim: int = 32, output_dim: int = 3) -> dict[str, np.ndarray]:
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

# ==========================================
# 1. PURE JAX HIGH-LEVEL PLANNER (Smooth Tanh Activation)
# ==========================================
def jax_mlp_forward(weights: dict, obs: jnp.ndarray) -> jnp.ndarray:
    """Takes a dictionary of JAX arrays and outputs [vx, vy, yaw_rate]"""
    w1, b1 = weights['w1'], weights['b1']
    w2, b2 = weights['w2'], weights['b2']
    
    # Layer 1: ReLU
    h1 = jnp.maximum(0.0, jnp.dot(obs, w1) + b1)
    # Layer 2: Linear Output
    out = jnp.dot(h1, w2) + b2
    
    # Smooth continuous mapping instead of sharp hard clipping boundaries
    vx = 1.5 * (jnp.tanh(out[0]) + 1.0) # Maps smoothly to [0.0, 3.0]
    vy = 0.5 * jnp.tanh(out[1])          # Maps smoothly to [-0.5, 0.5]
    yaw = 1.0 * jnp.tanh(out[2])         # Maps smoothly to [-1.0, 1.0]
    
    return jnp.array([vx, vy, yaw])

# ==========================================
# 2. PURE JAX TRACK MATH
# ==========================================
def jax_get_track_observation(qpos: jnp.ndarray) -> jnp.ndarray:
    """Calculates the 5D track observation purely on the GPU."""
    base_x = qpos[0]
    base_y = qpos[1]
    
    # Correctly convert the robot's quaternion to yaw angle
    w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
    base_yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    # Standard Track Specs
    L = 50.0
    R = 15.9154943  # 50 / pi
    half_width = 2.0 
    track_length = 2.0 * L + 2.0 * jnp.pi * R
    
    # 1. Determine which of the 4 track segments the robot is in
    is_bottom_straight = (base_x >= 0) & (base_x <= L) & (base_y < R)
    is_top_straight = (base_x >= 0) & (base_x <= L) & (base_y >= R)
    is_right_turn = (base_x > L)
    is_left_turn = (base_x < 0)
    
    # 2. Bottom Straight Math
    s_bottom = base_x
    lat_bottom = base_y
    head_bottom = 0.0
    curv_bottom = 0.0
    
    # 3. Top Straight Math
    s_top = L + jnp.pi * R + (L - base_x)
    lat_top = 2.0 * R - base_y
    head_top = jnp.pi
    curv_top = 0.0
    
    # 4. Right Turn Math
    dx_r = base_x - L
    dy_r = base_y - R
    angle_r = jnp.arctan2(dy_r, dx_r)
    s_right = L + (angle_r + jnp.pi/2.0) * R
    dist_r = jnp.sqrt(dx_r**2 + dy_r**2)
    lat_right = R - dist_r
    head_right = angle_r + jnp.pi/2.0
    curv_right = 1.0 / R
    
    # 5. Left Turn Math
    dx_l = base_x - 0.0
    dy_l = base_y - R
    angle_l = jnp.arctan2(dy_l, dx_l)
    angle_l_adj = jnp.where(angle_l < 0, angle_l + 2.0*jnp.pi, angle_l) # Wrap angle
    s_left = 2.0*L + jnp.pi*R + (angle_l_adj - jnp.pi/2.0) * R
    dist_l = jnp.sqrt(dx_l**2 + dy_l**2)
    lat_left = R - dist_l
    head_left = angle_l_adj + jnp.pi/2.0
    curv_left = 1.0 / R
    
    # 6. Combine active segment using jnp.where
    s = jnp.where(is_bottom_straight, s_bottom,
          jnp.where(is_top_straight, s_top,
            jnp.where(is_right_turn, s_right, s_left)))
            
    lateral_error = jnp.where(is_bottom_straight, lat_bottom,
                      jnp.where(is_top_straight, lat_top,
                        jnp.where(is_right_turn, lat_right, lat_left)))
                        
    track_heading = jnp.where(is_bottom_straight, head_bottom,
                      jnp.where(is_top_straight, head_top,
                        jnp.where(is_right_turn, head_right, head_left)))
                        
    curvature = jnp.where(is_bottom_straight, curv_bottom,
                  jnp.where(is_top_straight, curv_top,
                    jnp.where(is_right_turn, curv_right, curv_left)))
                    
    # 7. Calculate Final 5D Outputs
    lap_fraction = (s % track_length) / track_length
    lateral_error_norm = lateral_error / half_width
    boundary_margin_norm = (half_width - jnp.abs(lateral_error)) / half_width
    
    # Wrap heading error between -pi and pi
    heading_error = track_heading - base_yaw
    heading_error_rad = (heading_error + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    
    curvature_norm = curvature * R
    
    return jnp.array([
        lap_fraction, lateral_error_norm, boundary_margin_norm, 
        heading_error_rad, curvature_norm
    ], dtype=jnp.float32)


def jax_get_curv_norm(s_val: jnp.ndarray) -> jnp.ndarray:
    s_mod = s_val % 200.0
    is_turn = ((s_mod >= 50.0) & (s_mod < 100.0)) | ((s_mod >= 150.0) & (s_mod < 200.0))
    return jnp.where(is_turn, 1.0, 0.0)


# ==========================================
# MAIN SCRIPT
# ==========================================
def main() -> None:
    args = parse_args()
    rng_np = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- STRATEGY 1 & 3: EXPAND PARAMETERS TO ACCOMMODATE 8D OBSERVATION VECTOR ---
    input_dim = 8
    hidden_dim = args.hidden_dim
    output_dim = 3
    num_params = (input_dim * hidden_dim) + hidden_dim + (hidden_dim * output_dim) + output_dim

    print("Initializing JAX and loading MuJoCo/Brax environment (This may take 1-2 minutes)...")
    force_cpu = bool(args.force_cpu)
    set_runtime_env(force_cpu=force_cpu)
    stack = lazy_import_stack()
    course_cfg = load_json(args.config)
    course_cfg["runtime_overrides"] = {}
    
    ctrl_dt = float(course_cfg["control"]["ctrl_dt"])
    num_steps = int(round(float(args.eval_seconds) / ctrl_dt))
    
    registry = stack["registry"]
    locomotion_params = stack["locomotion_params"]
    env_name = course_cfg["environment_name"]
    ensure_environment_available(registry, env_name)

    env_cfg = registry.get_default_config(env_name)
    ppo_cfg = get_ppo_config(locomotion_params, env_name, course_cfg["backend_impl"])
    apply_stage_config(env_cfg, ppo_cfg, course_cfg, "stage_2")
    env_cfg.episode_length = int(num_steps)
    env_cfg.noise_config.level = 0.0
    env_cfg.pert_config.enable = False
    
    env = registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(course_cfg))
    
    ll_policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
    if not force_cpu:
        ll_policy = stack["jax"].jit(ll_policy)

    # ==========================================
    # 3. BUILD VMAP ROLLOUT ENGINE (WITH EXPANDED INPUTS & SMOOTHING)
    # ==========================================
    def step_single_robot(env_state, hl_weights, rng_key, is_standing, prev_real_s, last_cmd):
        # Extract base 5D features
        track_obs_5d = jax_get_track_observation(env_state.data.qpos)
        s_current = track_obs_5d[0] * 200.0
        
        # --- STRATEGY 3: VECTORIZED VELOCITY ESTIMATION ---
        delta_s = s_current - prev_real_s
        delta_s = jnp.where(delta_s < -100.0, delta_s + 200.0, delta_s)
        delta_s = jnp.where(delta_s > 100.0, delta_s - 200.0, delta_s)
        v_est = delta_s / ctrl_dt
        
        # --- STRATEGY 1: LOOK-AHEAD EXTENSIONS ---
        c2 = jax_get_curv_norm(s_current + 2.0)
        c5 = jax_get_curv_norm(s_current + 5.0)
        
        # Construct full 8D Input Space
        track_obs_8d = jnp.array([
            track_obs_5d[0], track_obs_5d[1], track_obs_5d[2],
            track_obs_5d[3], track_obs_5d[4], v_est, c2, c5
        ])
        
        cmd_raw = jax_mlp_forward(hl_weights, track_obs_8d)
        
        # --- STRATEGY 5: MATCHING EXPO GAIN RATE FILTER ---
        alpha = 0.20
        cmd = alpha * cmd_raw + (1.0 - alpha) * last_cmd
        
        # Override during stand-up phase
        cmd = jnp.where(is_standing, jnp.zeros_like(cmd), cmd)
        
        # Merge command into observation for low-level policy
        env_state.info["command"] = cmd
        env_state.info["steps_until_next_cmd"] = jnp.array(10**9, dtype=jnp.int32)
        
        # Step the low-level environment physics
        action, _ = ll_policy(env_state.obs, rng_key)
        next_state = env.step(env_state, action)
        next_state.info["command"] = cmd
        
        next_track_obs_5d = jax_get_track_observation(next_state.data.qpos)
        
        return next_state, next_track_obs_5d, s_current, cmd
    
    # Pass structural arrays properly across batch processing paths
    batch_robot_step = jax.vmap(step_single_robot, in_axes=(0, 0, 0, None, 0, 0))

    @jax.jit
    def batch_rollout(initial_states_batch, batch_weights, batch_keys):
        track_length = 200.0

        def scan_step(carry, step_idx):
            current_states, prev_s, cum_dist, prev_real_s_batch, last_cmd_batch = carry
            
            is_standing = (step_idx * ctrl_dt) < 1.0
            
            next_states, next_track_obs_batch, current_real_s, next_cmd_batch = batch_robot_step(
                current_states, batch_weights, batch_keys, is_standing, prev_real_s_batch, last_cmd_batch
            )
            
            s_next = next_track_obs_batch[:, 0] * track_length
            delta_s = s_next - prev_s
            delta_s = jnp.where(delta_s < -track_length / 2.0, delta_s + track_length, delta_s)
            delta_s = jnp.where(delta_s > track_length / 2.0, delta_s - track_length, delta_s)
            delta_s = jnp.where(is_standing, 0.0, delta_s)
            
            next_cum_dist = cum_dist + delta_s
            step_lateral_errors = jnp.abs(next_track_obs_batch[:, 1])
            
            z_height = next_states.data.qpos[:, 2]
            has_fallen = z_height < 0.23  
            
            return (next_states, s_next, next_cum_dist, current_real_s, next_cmd_batch), (next_cum_dist, step_lateral_errors, has_fallen)

        # Initialize trackers
        init_obs = jax.vmap(jax_get_track_observation)(initial_states_batch.data.qpos)
        init_s = init_obs[:, 0] * track_length
        init_cum_dist = jnp.zeros(args.population)
        init_cmd = jnp.zeros((args.population, 3))
        
        init_carry = (initial_states_batch, init_s, init_cum_dist, init_s, init_cmd)
        
        _, (all_cum_dists, all_lateral_errors, all_has_fallen) = jax.lax.scan(
            scan_step, init_carry, jnp.arange(num_steps)
        )
        return all_cum_dists, all_lateral_errors, all_has_fallen
    
    main_rng = jax.random.PRNGKey(args.seed)

    print("Environment loaded and compiled! Starting VMAP CEM Optimization...\n")

    mu = np.zeros(num_params, dtype=np.float32)
    mu[-3] = 1.0  # Encourage initial forward movement velocity maps
    sigma = np.ones(num_params, dtype=np.float32) * 0.5
    best_score = -100.0
    history = []
    num_elites = max(1, int(args.population * args.elite_frac))

    for iteration in range(args.iterations):
        t0 = time.time()
        
        candidates = []
        for i in range(args.population):
            if i == 0 and iteration > 0:
                candidates.append(mu.copy())
            else:
                candidates.append(mu + rng_np.normal(0.0, sigma))
        
        batch_weights = {
            'w1': jnp.stack([vector_to_weights(c, hidden_dim=hidden_dim)['w1'] for c in candidates]),
            'b1': jnp.stack([vector_to_weights(c, hidden_dim=hidden_dim)['b1'] for c in candidates]),
            'w2': jnp.stack([vector_to_weights(c, hidden_dim=hidden_dim)['w2'] for c in candidates]),
            'b2': jnp.stack([vector_to_weights(c, hidden_dim=hidden_dim)['b2'] for c in candidates]),
        }

        main_rng, reset_rng = jax.random.split(main_rng)
        batch_keys = jax.random.split(reset_rng, args.population)
        initial_states_batch = jax.vmap(env.reset)(batch_keys)

        all_cum_dists, all_lateral_errors, all_has_fallen = batch_rollout(initial_states_batch, batch_weights, batch_keys)
        
        # --- STRATEGY 4: CONTINUOUS, SMOOTH OPTIMIZATION LANDSCAPE SCORING ---
        track_length = 200.0
        max_unwrapped_lap = jnp.max(all_cum_dists, axis=0) / track_length
        mean_lateral_error = jnp.mean(all_lateral_errors, axis=0)
        
        progress_score = max_unwrapped_lap * 20.0
        lateral_penalty = 1.5 * mean_lateral_error
        survival_rate = jnp.mean(1.0 - all_has_fallen.astype(jnp.float32), axis=0)
        survival_bonus = jnp.where(max_unwrapped_lap > 0.01, 1.0 * survival_rate, 0.0)
        
        # Smooth boundaries instead of hard threshold cuts
        max_lateral = jnp.max(jnp.abs(all_lateral_errors), axis=0)
        boundary_penalty = jnp.where(max_lateral > 1.0, 8.0 * (max_lateral - 1.0) + 2.0, 0.0)
        fall_penalty = jnp.where(survival_rate < 1.0, 15.0 * (1.0 - survival_rate), 0.0)
        
        scores = progress_score - lateral_penalty + survival_bonus - boundary_penalty - fall_penalty
        scores = np.array(scores)

        # Save Best Candidate and Metadata
        best_idx = int(np.argmax(scores))
        gen_best_score = scores[best_idx]
        
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_weights = vector_to_weights(candidates[best_idx], hidden_dim=hidden_dim)
            np.savez(args.output_dir / "planner_weights.npz", **best_weights)
            
            config_payload = {
                "planner_type": "learned_mlp", 
                "weights_path": "planner_weights.npz", 
                "stand_seconds": 1.0,
                "hidden_dim": hidden_dim
            }
            (args.output_dir / "planner_config.json").write_text(json.dumps(config_payload, indent=2))
            
            (args.output_dir / "best_score.json").write_text(json.dumps({
                "score": float(best_score), "iteration": iteration, "candidate": best_idx
            }, indent=2))

        # Fit next generation distributions
        sorted_indices = np.argsort(scores)[::-1]
        elites = [candidates[idx] for idx in sorted_indices[:num_elites]]
        elites_arr = np.array(elites)
        mu = np.mean(elites_arr, axis=0)
        sigma = np.std(elites_arr, axis=0) + max(0.01, 0.1 * (0.85 ** iteration))
        
        dt = time.time() - t0
        print(f"[Gen {iteration:02d}] Best Gen Score: {gen_best_score:.3f} | Best Global: {best_score:.3f} | Step Time: {dt:.3f}s")
        
        history.append({
            "iteration": iteration,
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "best_global": float(best_score)
        })
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\nOptimization Finished! Deployment config located at: {args.output_dir}")

if __name__ == "__main__":
    main()
