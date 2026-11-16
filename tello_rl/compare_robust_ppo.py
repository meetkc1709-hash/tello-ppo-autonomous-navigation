"""
Compare baseline PPO and localization-robust PPO.

Uses:
- Identical environment configuration
- Matching episode seeds
- Fixed localization error scales
- True-position collision checks
- Estimated-position arrival decisions

No real drone connection is required.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from tello_env import TelloNavEnv
from robust_obs import RobustObservationWrapper


BASELINE = "models/stage2_cam4/final_model.zip"
ROBUST = "models/stage2_cam4_robust/final_model.zip"
CONFIG = "models/stage2_cam4/env_config.json"

SCALES = [1.00, 0.95, 0.85, 0.75, 0.65]


def physical_collision(env):
    """Check collisions independently of the simulator's success flag."""

    pos = env.pos

    wall = bool(
        np.any(pos < env.drone_r)
        or np.any(pos > env.room - env.drone_r)
    )

    obstacle = False

    if len(env.obs_pos):
        obstacle = bool(
            np.any(
                np.linalg.norm(env.obs_pos - pos, axis=1)
                < env.obs_r + env.drone_r
            )
        )

    return wall or obstacle


def run_episode(model, config, scale, seed, drift):
    base_env = TelloNavEnv(**config)

    env = RobustObservationWrapper(
        base_env,
        min_scale=scale,
        max_scale=scale,
        drift_std=drift,
    )

    try:
        obs, _ = env.reset(seed=seed)

        # Record the initial geometry for consistency checks.
        initial_pos = base_env.pos.copy()
        initial_goal = base_env.goal.copy()
        initial_obstacles = base_env.obs_pos.copy()
        initial_radii = base_env.obs_r.copy()

        outcome = "timeout"
        false_arrival_event = False

        while True:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            true_dist = float(
                np.linalg.norm(base_env.goal - base_env.pos)
            )

            estimated_dist = float(
                np.linalg.norm(
                    base_env.goal - env.estimated_position
                )
            )

            crashed = physical_collision(base_env)
            true_arrival = true_dist < base_env.goal_radius
            estimated_arrival = estimated_dist < base_env.goal_radius

            if estimated_arrival and not true_arrival:
                false_arrival_event = True

            # Physical collisions take precedence over arrival.
            if crashed:
                outcome = "collision"
                break

            if true_arrival:
                outcome = "success"
                break

            # Simulate the real controller stopping based on
            # its estimated rather than actual position.
            if estimated_arrival:
                outcome = "false_arrival"
                break

            if terminated or truncated:
                outcome = "timeout"
                break

        result = {
            "seed": seed,
            "scale": scale,
            "drift": drift,
            "outcome": outcome,
            "success": int(outcome == "success"),
            "collision": int(outcome == "collision"),
            "false_arrival": int(outcome == "false_arrival"),
            "false_arrival_event": int(false_arrival_event),
            "timeout": int(outcome == "timeout"),
            "steps": base_env.steps,
            "flight_time_s": round(base_env.steps * base_env.dt, 3),
            "true_final_distance_m": round(true_dist, 4),
            "estimated_final_distance_m": round(estimated_dist, 4),
            "initial_position": initial_pos.tolist(),
            "goal_position": initial_goal.tolist(),
            "obstacles": initial_obstacles.tolist(),
            "obstacle_radii": initial_radii.tolist(),
        }

        return result

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift", type=float, default=0.0)
    parser.add_argument("--output", default="robust_comparison")

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    if args.drift < 0:
        parser.error("--drift must be nonnegative")

    with open(CONFIG) as f:
        config = json.load(f)

    models = {
        "Original PPO": PPO.load(BASELINE, device="cpu"),
        "Robust PPO 8k": PPO.load(ROBUST, device="cpu"),
        "Robust PPO 100k": PPO.load(
            "models/stage2_cam4_robust_100k/final_model.zip",
            device="cpu",
        ),
        "Robust PPO 100k Best": PPO.load(
            "models/stage2_cam4_robust_100k/best_model.zip",
            device="cpu",
        ),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = []

    for scale in SCALES:
        print(f"\nLOCALIZATION SCALE: {scale:.2f}")
        print("-" * 75)

        for model_name, model in models.items():
            results = []

            for episode in range(args.episodes):
                seed = args.seed + episode

                result = run_episode(
                    model=model,
                    config=config,
                    scale=scale,
                    seed=seed,
                    drift=args.drift,
                )

                result["model"] = model_name
                results.append(result)
                rows.append(result)

            successes = sum(r["success"] for r in results)
            collisions = sum(r["collision"] for r in results)
            false_arrivals = sum(r["false_arrival"] for r in results)
            timeouts = sum(r["timeout"] for r in results)

            avg_distance = np.mean([
                r["true_final_distance_m"] for r in results
            ])

            avg_time = np.mean([
                r["flight_time_s"] for r in results
            ])

            print(
                f"{model_name:15s} | "
                f"Success: {successes:3d}/{args.episodes} | "
                f"Collision: {collisions:3d} | "
                f"False arrival: {false_arrivals:3d} | "
                f"Timeout: {timeouts:3d} | "
                f"Final dist: {avg_distance:.2f} m | "
                f"Time: {avg_time:.1f} s"
            )

    # Save detailed episode results.
    detail_path = output / "episodes.csv"

    fields = [
        "model", "seed", "scale", "drift",
        "outcome", "success", "collision",
        "false_arrival", "false_arrival_event", "timeout",
        "steps", "flight_time_s",
        "true_final_distance_m",
        "estimated_final_distance_m",
        "initial_position", "goal_position",
        "obstacles", "obstacle_radii",
    ]

    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Save compact summary for plotting and reports.
    summary_path = output / "summary.csv"

    with open(summary_path, "w", newline="") as f:
        fields = [
            "model", "scale", "episodes",
            "success_rate", "collision_rate",
            "false_arrival_rate", "timeout_rate",
            "avg_final_distance_m", "avg_flight_time_s",
        ]

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for scale in SCALES:
            for model_name in models:
                group = [
                    r for r in rows
                    if r["model"] == model_name
                    and r["scale"] == scale
                ]

                writer.writerow({
                    "model": model_name,
                    "scale": scale,
                    "episodes": len(group),
                    "success_rate": np.mean(
                        [r["success"] for r in group]
                    ),
                    "collision_rate": np.mean(
                        [r["collision"] for r in group]
                    ),
                    "false_arrival_rate": np.mean(
                        [r["false_arrival"] for r in group]
                    ),
                    "timeout_rate": np.mean(
                        [r["timeout"] for r in group]
                    ),
                    "avg_final_distance_m": np.mean(
                        [r["true_final_distance_m"] for r in group]
                    ),
                    "avg_flight_time_s": np.mean(
                        [r["flight_time_s"] for r in group]
                    ),
                })

    print("\nEvaluation complete.")
    print("Detailed results:", detail_path)
    print("Summary results:", summary_path)


if __name__ == "__main__":
    main()
