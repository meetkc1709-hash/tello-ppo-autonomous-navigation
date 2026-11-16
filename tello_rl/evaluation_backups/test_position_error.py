import numpy as np
from stable_baselines3 import PPO

from tello_env import TelloNavEnv
from train import STAGES
from model_config import load_config

MODEL = "models/stage2_cam4/final_model.zip"

EPISODES = 30
SCALES = [1.00, 0.95, 0.90, 0.85, 0.80]

GOAL_RADIUS = 0.25
APPROACH_RADIUS = 0.65
CLEARANCE = 1.30

# Match our earlier evaluation configuration.
cfg = dict(STAGES[2])
cfg.pop("steps")

cfg["room_size"] = (3.0, 3.0)
cfg["sensor"] = "camera"
cfg["n_static_obstacles"] = 1
cfg["n_moving_obstacles"] = 0

saved = load_config(MODEL)

for key in (
    "max_speed",
    "drone_radius",
    "sensor",
    "cam_lag",
    "obstacle_radius",
    "line_offset",
    "safe_distance",
):
    if key in saved:
        value = saved[key]
        cfg[key] = tuple(value) if isinstance(value, list) else value

cfg["cam_lag"] = True

model = PPO.load(MODEL, device="cpu")
n_obs = model.observation_space.shape[0]


def estimated_observation(obs, env, est_pos, est_vel):
    """
    Replace position-dependent observation fields with
    estimated values. Keep simulated camera rays unchanged.
    """
    obs = obs.copy()

    scale = float(env.room.max())
    error = env.goal - est_pos

    obs[0:2] = error / scale
    obs[2:4] = est_vel / env.max_speed
    obs[4] = np.linalg.norm(error) / scale

    obs[5:9] = np.array([
        est_pos[0],
        env.room[0] - est_pos[0],
        est_pos[1],
        env.room[1] - est_pos[1]
    ]) / scale

    return obs


def run_episode(seed, position_scale, use_approach):
    env = TelloNavEnv(**cfg)

    obs, _ = env.reset(seed=seed)

    # Keep the simulator's physical collision checks.
    # Arrival is instead decided from estimated position.
    env.goal_radius = 0.01

    true_previous = env.pos.copy()

    # Assume the initial position is known.
    est_pos = env.pos.copy()
    est_vel = np.zeros(2, dtype=np.float32)

    approach_steps = 0
    outward_steps = 0
    physically_entered_goal = False

    previous_true_dist = float(
        np.linalg.norm(env.goal - env.pos)
    )

    while True:

        true_dist = float(
            np.linalg.norm(env.goal - env.pos)
        )

        estimated_dist = float(
            np.linalg.norm(env.goal - est_pos)
        )

        if true_dist < GOAL_RADIUS:
            physically_entered_goal = True

        # Equivalent to the real script deciding to land
        # because its estimated destination was reached.
        if estimated_dist < GOAL_RADIUS:
            return {
                "result": (
                    "SUCCESS"
                    if true_dist < GOAL_RADIUS
                    else "FALSE_LANDING"
                ),
                "true_error": true_dist,
                "estimated_error": estimated_dist,
                "time": env.steps * env.dt,
                "approach_steps": approach_steps,
                "outward_steps": outward_steps,
                "entered": physically_entered_goal,
            }

        controller_obs = estimated_observation(
            obs, env, est_pos, est_vel
        )

        action, _ = model.predict(
            controller_obs[:n_obs],
            deterministic=True
        )

        if use_approach and estimated_dist <= APPROACH_RADIUS:
            rays = env._cam_cur

            if (
                rays is not None
                and float(np.min(rays)) >= CLEARANCE
            ):
                error = np.asarray(
                    env.goal - est_pos,
                    dtype=np.float32
                )

                command = 0.65 * error - 1.10 * est_vel
                command = np.clip(command, -0.12, 0.12)

                action = np.clip(
                    command / env.max_speed,
                    -1.0,
                    1.0
                ).astype(np.float32)

                approach_steps += 1

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        # Actual physical displacement.
        true_displacement = env.pos - true_previous

        # Simulate velocity/dead-reckoning underestimation.
        estimated_displacement = (
            position_scale * true_displacement
        )

        est_pos = est_pos + estimated_displacement

        est_vel = (
            estimated_displacement / env.dt
        ).astype(np.float32)

        true_previous = env.pos.copy()

        new_true_dist = float(
            np.linalg.norm(env.goal - env.pos)
        )

        if new_true_dist < GOAL_RADIUS:
            physically_entered_goal = True

        if (
            true_dist <= APPROACH_RADIUS
            and new_true_dist > previous_true_dist + 0.001
        ):
            outward_steps += 1

        previous_true_dist = new_true_dist

        if terminated or truncated:

            if "crash" in info:
                result = "CRASH"
            elif truncated:
                result = "TIMEOUT"
            elif new_true_dist < GOAL_RADIUS:
                result = "SUCCESS"
            else:
                result = "OTHER"

            return {
                "result": result,
                "true_error": new_true_dist,
                "estimated_error": float(
                    np.linalg.norm(env.goal - est_pos)
                ),
                "time": env.steps * env.dt,
                "approach_steps": approach_steps,
                "outward_steps": outward_steps,
                "entered": physically_entered_goal,
            }


print("\n=== POSITION ESTIMATION SENSITIVITY TEST ===")
print(f"Episodes per configuration: {EPISODES}")

for use_approach in [False, True]:

    name = (
        "PPO + Goal Approach"
        if use_approach
        else "Original PPO"
    )

    print(f"\n{name}")
    print("-" * 65)

    for position_scale in SCALES:

        results = []

        for ep in range(EPISODES):
            seed = 10000 + ep

            results.append(
                run_episode(
                    seed=seed,
                    position_scale=position_scale,
                    use_approach=use_approach
                )
            )

        counts = {
            label: sum(
                r["result"] == label for r in results
            )
            for label in [
                "SUCCESS",
                "FALSE_LANDING",
                "CRASH",
                "TIMEOUT",
                "OTHER",
            ]
        }

        mean_true_error = np.mean([
            r["true_error"] for r in results
        ])

        total_approach = sum(
            r["approach_steps"] for r in results
        )

        total_outward = sum(
            r["outward_steps"] for r in results
        )

        print(
            f"Position scale: {position_scale:.2f} | "
            f"Success: {counts['SUCCESS']:2d}/30 | "
            f"False landing: {counts['FALSE_LANDING']:2d} | "
            f"Crashes: {counts['CRASH']:2d} | "
            f"Timeouts: {counts['TIMEOUT']:2d}"
        )

        print(
            f"  Mean final true error: "
            f"{mean_true_error:.2f} m | "
            f"Approach steps: {total_approach} | "
            f"Outward steps: {total_outward}"
        )

print("\nTest complete.")
