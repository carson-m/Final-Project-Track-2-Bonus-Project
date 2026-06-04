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

COMMAND_FILTER_ALPHA = 0.15
TRACK_LENGTH_M = 200.0
TURN_RADIUS_M = 18.25
HALF_WIDTH_M = 2.0
STRAIGHT_LENGTH_M = (TRACK_LENGTH_M - 2.0 * np.pi * TURN_RADIUS_M) / 2.0
MAX_STRAIGHT_SPEED_MPS = 0.95
MAX_CURVE_SPEED_MPS = 0.55
MAX_LATERAL_SPEED_MPS = 0.25
MAX_YAW_RATE_RADPS = 0.55
EDGE_SLOWDOWN_MARGIN_NORM = 0.35
MAX_COMMAND_DELTA = 0.08

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
    parser.add_argument("--start-s-m", type=float, default=0.0, help="Official track progress used for training resets.")
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
    
    # Keep commands inside the low-level policy's training distribution.
    vx = 0.5 * MAX_STRAIGHT_SPEED_MPS * (jnp.tanh(out[0]) + 1.0)
    vy = MAX_LATERAL_SPEED_MPS * jnp.tanh(out[1])
    yaw = MAX_YAW_RATE_RADPS * jnp.tanh(out[2])
    
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
    
    straight = jnp.asarray(STRAIGHT_LENGTH_M, dtype=jnp.float32)
    radius = jnp.asarray(TURN_RADIUS_M, dtype=jnp.float32)
    half_width = jnp.asarray(HALF_WIDTH_M, dtype=jnp.float32)
    track_length = jnp.asarray(TRACK_LENGTH_M, dtype=jnp.float32)
    half_straight = straight / 2.0
    xy = jnp.array([base_x, base_y])

    x_clamped = jnp.clip(base_x, -half_straight, half_straight)

    # Bottom straight projection.
    center_bottom = jnp.array([x_clamped, -radius])
    dist_bottom = jnp.sum(jnp.square(xy - center_bottom))
    s_bottom = x_clamped + half_straight
    head_bottom = 0.0
    curv_bottom = 0.0

    # Top straight projection.
    center_top = jnp.array([x_clamped, radius])
    dist_top = jnp.sum(jnp.square(xy - center_top))
    s_top = straight + jnp.pi * radius + (half_straight - x_clamped)
    head_top = jnp.pi
    curv_top = 0.0

    # Right turn projection.
    right_center = jnp.array([half_straight, 0.0])
    rel_right = xy - right_center
    theta_right = jnp.clip(jnp.arctan2(rel_right[1], rel_right[0]), -jnp.pi / 2.0, jnp.pi / 2.0)
    center_right = right_center + radius * jnp.array([jnp.cos(theta_right), jnp.sin(theta_right)])
    dist_right = jnp.sum(jnp.square(xy - center_right))
    s_right = straight + (theta_right + jnp.pi / 2.0) * radius
    head_right = theta_right + jnp.pi / 2.0
    curv_right = 1.0 / radius

    # Left turn projection.
    left_center = jnp.array([-half_straight, 0.0])
    rel_left = xy - left_center
    theta_left = jnp.arctan2(rel_left[1], rel_left[0])
    theta_left = jnp.where(theta_left < jnp.pi / 2.0, theta_left + 2.0 * jnp.pi, theta_left)
    theta_left = jnp.clip(theta_left, jnp.pi / 2.0, 3.0 * jnp.pi / 2.0)
    center_left = left_center + radius * jnp.array([jnp.cos(theta_left), jnp.sin(theta_left)])
    dist_left = jnp.sum(jnp.square(xy - center_left))
    s_left = 2.0 * straight + jnp.pi * radius + (theta_left - jnp.pi / 2.0) * radius
    head_left = theta_left + jnp.pi / 2.0
    curv_left = 1.0 / radius

    distances = jnp.array([dist_bottom, dist_right, dist_top, dist_left])
    best_idx = jnp.argmin(distances)
    s_values = jnp.array([s_bottom, s_right, s_top, s_left])
    heading_values = jnp.array([head_bottom, head_right, head_top, head_left])
    curvature_values = jnp.array([curv_bottom, curv_right, curv_top, curv_left])
    center_values = jnp.stack([center_bottom, center_right, center_top, center_left])

    s = s_values[best_idx]
    track_heading = heading_values[best_idx]
    curvature = curvature_values[best_idx]
    center = center_values[best_idx]

    normal = jnp.array([-jnp.sin(track_heading), jnp.cos(track_heading)])
    lateral_error = jnp.dot(xy - center, normal)

    lap_fraction = (s % track_length) / track_length
    lateral_error_norm = lateral_error / half_width
    boundary_margin_norm = (half_width - jnp.abs(lateral_error)) / half_width
    
    # Wrap heading error between -pi and pi
    heading_error = track_heading - base_yaw
    heading_error_rad = (heading_error + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    
    curvature_norm = curvature * radius
    
    return jnp.array([
        lap_fraction, lateral_error_norm, boundary_margin_norm, 
        heading_error_rad, curvature_norm
    ], dtype=jnp.float32)


def jax_get_curv_norm(s_val: jnp.ndarray) -> jnp.ndarray:
    s_mod = s_val % TRACK_LENGTH_M
    straight = jnp.asarray(STRAIGHT_LENGTH_M, dtype=jnp.float32)
    turn_len = jnp.pi * TURN_RADIUS_M
    is_turn = ((s_mod >= straight) & (s_mod < straight + turn_len)) | (
        (s_mod >= 2.0 * straight + turn_len) & (s_mod < TRACK_LENGTH_M)
    )
    return jnp.where(is_turn, 1.0, 0.0)


def jax_apply_stability_envelope(
    track_obs_5d: jnp.ndarray,
    s_current: jnp.ndarray,
    lookahead_c2: jnp.ndarray,
    lookahead_c5: jnp.ndarray,
    cmd: jnp.ndarray,
) -> jnp.ndarray:
    del s_current
    turn_intensity = jnp.clip(
        jnp.maximum(jnp.maximum(jnp.abs(track_obs_5d[4]), lookahead_c2), lookahead_c5),
        0.0,
        1.0,
    )
    speed_cap = (1.0 - turn_intensity) * MAX_STRAIGHT_SPEED_MPS + turn_intensity * MAX_CURVE_SPEED_MPS

    heading_risk = jnp.minimum(jnp.abs(track_obs_5d[3]) / 1.0, 1.0)
    lateral_risk = jnp.clip((jnp.abs(track_obs_5d[1]) - 0.35) / 0.65, 0.0, 1.0)
    edge_risk = jnp.clip(
        (EDGE_SLOWDOWN_MARGIN_NORM - track_obs_5d[2]) / EDGE_SLOWDOWN_MARGIN_NORM,
        0.0,
        1.0,
    )
    risk_scale = jnp.clip(1.0 - 0.30 * heading_risk - 0.25 * lateral_risk - 0.35 * edge_risk, 0.35, 1.0)
    speed_cap = speed_cap * risk_scale

    return jnp.array(
        [
            jnp.clip(cmd[0], 0.0, speed_cap),
            jnp.clip(cmd[1], -MAX_LATERAL_SPEED_MPS, MAX_LATERAL_SPEED_MPS),
            jnp.clip(cmd[2], -MAX_YAW_RATE_RADPS, MAX_YAW_RATE_RADPS),
        ],
        dtype=jnp.float32,
    )


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

    from mujoco import mjx
    from mujoco.mjx._src import math as mjmath
    from mujoco_playground._src import mjx_env

    start_s = float(args.start_s_m % TRACK_LENGTH_M)
    right_turn_start = STRAIGHT_LENGTH_M
    top_straight_start = STRAIGHT_LENGTH_M + np.pi * TURN_RADIUS_M
    left_turn_start = 2.0 * STRAIGHT_LENGTH_M + np.pi * TURN_RADIUS_M
    half_straight = STRAIGHT_LENGTH_M / 2.0
    if start_s < right_turn_start:
        start_xy = np.array([-half_straight + start_s, -TURN_RADIUS_M], dtype=np.float32)
        start_heading = 0.0
    elif start_s < top_straight_start:
        theta = -np.pi / 2.0 + (start_s - right_turn_start) / TURN_RADIUS_M
        start_xy = np.array([half_straight, 0.0], dtype=np.float32) + TURN_RADIUS_M * np.array(
            [np.cos(theta), np.sin(theta)], dtype=np.float32
        )
        start_heading = theta + np.pi / 2.0
    elif start_s < left_turn_start:
        u = start_s - top_straight_start
        start_xy = np.array([half_straight - u, TURN_RADIUS_M], dtype=np.float32)
        start_heading = np.pi
    else:
        theta = np.pi / 2.0 + (start_s - left_turn_start) / TURN_RADIUS_M
        start_xy = np.array([-half_straight, 0.0], dtype=np.float32) + TURN_RADIUS_M * np.array(
            [np.cos(theta), np.sin(theta)], dtype=np.float32
        )
        start_heading = theta + np.pi / 2.0

    def reset_lowlevel_on_track(rng_key):
        state = env.reset(rng_key)
        qpos = env._init_q
        qvel = jnp.zeros(env.mjx_model.nv)
        quat = mjmath.axis_angle_to_quat(
            jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
            jnp.asarray(start_heading, dtype=jnp.float32),
        )
        qpos = qpos.at[0:2].set(jnp.asarray(start_xy, dtype=jnp.float32))
        qpos = qpos.at[3:7].set(quat)

        data = mjx_env.make_data(
            env.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=qpos[7:],
            impl=env.mjx_model.impl.value,
            naconmax=env._config.naconmax,
            njmax=env._config.njmax,
        )
        data = mjx.forward(env.mjx_model, data)
        state.info["command"] = jnp.zeros(3, dtype=jnp.float32)
        state.info["steps_until_next_cmd"] = jnp.array(10**9, dtype=jnp.int32)
        obs = env._get_obs(data, state.info)
        return state.replace(data=data, obs=obs, reward=jnp.zeros(()), done=jnp.zeros(()))

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
        cmd_raw = jax_apply_stability_envelope(track_obs_5d, s_current, c2, c5, cmd_raw)
        
        # --- STRATEGY 5: MATCHING EXPO GAIN RATE FILTER ---
        cmd = COMMAND_FILTER_ALPHA * cmd_raw + (1.0 - COMMAND_FILTER_ALPHA) * last_cmd
        cmd_delta = jnp.clip(cmd - last_cmd, -MAX_COMMAND_DELTA, MAX_COMMAND_DELTA)
        cmd = last_cmd + cmd_delta
        
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
            current_states, prev_s, cum_dist, prev_real_s_batch, last_cmd_batch, alive_batch = carry
            
            is_standing = (step_idx * ctrl_dt) < 1.0
            
            next_states, next_track_obs_batch, current_real_s, next_cmd_batch = batch_robot_step(
                current_states, batch_weights, batch_keys, is_standing, prev_real_s_batch, last_cmd_batch
            )
            
            s_next = next_track_obs_batch[:, 0] * track_length
            delta_s = s_next - prev_s
            delta_s = jnp.where(delta_s < -track_length / 2.0, delta_s + track_length, delta_s)
            delta_s = jnp.where(delta_s > track_length / 2.0, delta_s - track_length, delta_s)
            delta_s = jnp.where(is_standing | ~alive_batch, 0.0, delta_s)
            
            next_cum_dist = cum_dist + delta_s
            step_lateral_errors = jnp.abs(next_track_obs_batch[:, 1])
            
            z_height = next_states.data.qpos[:, 2]
            has_fallen = z_height < 0.23  
            boundary_violation = next_track_obs_batch[:, 2] < 0.0
            terminal = has_fallen | boundary_violation
            next_alive_batch = alive_batch & ~terminal
            
            return (
                next_states,
                s_next,
                next_cum_dist,
                current_real_s,
                next_cmd_batch,
                next_alive_batch,
            ), (next_cum_dist, step_lateral_errors, has_fallen, boundary_violation, next_alive_batch)

        # Initialize trackers
        init_obs = jax.vmap(jax_get_track_observation)(initial_states_batch.data.qpos)
        init_s = init_obs[:, 0] * track_length
        init_cum_dist = jnp.zeros(args.population)
        init_cmd = jnp.zeros((args.population, 3))
        init_alive = jnp.ones(args.population, dtype=bool)
        
        init_carry = (initial_states_batch, init_s, init_cum_dist, init_s, init_cmd, init_alive)
        
        _, (all_cum_dists, all_lateral_errors, all_has_fallen, all_boundary_violations, all_alive) = jax.lax.scan(
            scan_step, init_carry, jnp.arange(num_steps)
        )
        return all_cum_dists, all_lateral_errors, all_has_fallen, all_boundary_violations, all_alive
    
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
        initial_states_batch = jax.vmap(reset_lowlevel_on_track)(batch_keys)

        (
            all_cum_dists,
            all_lateral_errors,
            all_has_fallen,
            all_boundary_violations,
            all_alive,
        ) = batch_rollout(initial_states_batch, batch_weights, batch_keys)
        
        # --- STRATEGY 4: CONTINUOUS, SMOOTH OPTIMIZATION LANDSCAPE SCORING ---
        track_length = 200.0
        final_unwrapped_lap = all_cum_dists[-1] / track_length
        mean_lateral_error = jnp.mean(all_lateral_errors, axis=0)
        
        progress_score = final_unwrapped_lap * 20.0
        lateral_penalty = 1.5 * mean_lateral_error
        alive_rate = jnp.mean(all_alive.astype(jnp.float32), axis=0)
        survival_bonus = jnp.where(final_unwrapped_lap > 0.01, 1.0 * alive_rate, 0.0)
        
        # Smooth boundaries instead of hard threshold cuts
        max_lateral = jnp.max(jnp.abs(all_lateral_errors), axis=0)
        boundary_penalty = jnp.where(max_lateral > 1.0, 8.0 * (max_lateral - 1.0) + 2.0, 0.0)
        fall_rate = jnp.mean(all_has_fallen.astype(jnp.float32), axis=0)
        boundary_rate = jnp.mean(all_boundary_violations.astype(jnp.float32), axis=0)
        fall_penalty = jnp.where(fall_rate > 0.0, 15.0 * fall_rate, 0.0)
        terminal_penalty = 12.0 * boundary_rate
        
        scores = progress_score - lateral_penalty + survival_bonus - boundary_penalty - fall_penalty - terminal_penalty
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
                "hidden_dim": hidden_dim,
                "command_filter_alpha": COMMAND_FILTER_ALPHA,
                "max_straight_speed_mps": MAX_STRAIGHT_SPEED_MPS,
                "max_curve_speed_mps": MAX_CURVE_SPEED_MPS,
                "max_lateral_speed_mps": MAX_LATERAL_SPEED_MPS,
                "max_yaw_rate_radps": MAX_YAW_RATE_RADPS,
                "edge_slowdown_margin_norm": EDGE_SLOWDOWN_MARGIN_NORM,
                "max_command_delta": MAX_COMMAND_DELTA
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
