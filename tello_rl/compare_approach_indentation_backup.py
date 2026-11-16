import numpy as np
from stable_baselines3 import PPO

from tello_env import TelloNavEnv
from train import STAGES
from model_config import load_config

MODEL = "models/stage2_cam4/final_model.zip"
EPISODES = 30
APPROACH_RADIUS = 0.65
CLEARANCE = 1.3

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


def evaluate_episode(seed, use_approach):
    env = TelloNavEnv(**cfg)
    obs, _ = env.reset(seed=seed)

    previous_position = env.pos.copy()
    previous_distance = float(np.linalg.norm(env.goal - env.pos))

    path_length = 0.0
    near_goal_path = 0.0
    outward_steps = 0
    approach_steps = 0
    switches = 0

    entered_goal_region = False
    previous_mode = None

    done = False

    while not done:
        action, _ = model.predict(
            obs[:n_obs],
            deterministic=True
        )

        distance = float(np.linalg.norm(env.goal - env.pos))
        mode = "PPO"

        if use_approach and distance <= APPROACH_RADIUS:
            rays = env._cam_cur

            if rays is not None and float(np.min(rays)) >= CLEARANCE:
                error = np.asarray(
                    env.goal - env.pos,
                    dtype=np.float32
                )

                command = 0.65 * error - 1.10 * env.vel
                command = np.clip(command, -0.12, 0.12)

                action = np.clip(
                    command / env.max_speed,
                    -1.0,
                    1.0
                ).astype(np.float32)

                mode = "GOAL_APPROACH"
                approach_steps += 1

        if previous_mode is not None and mode != previous_mode:
            switches += 1

        previous_mode = mode

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        new_position = env.pos.copy()
        new_distance = float(np.linalg.norm(env.goal - env.pos))

        movement = float(np.linalg.norm(new_position - previous_position))
        path_length += movement

        if distance <= APPROACH_RADIUS:
            entered_goal_region = True
            near_goal_path += movement

            if new_distance > previous_distance + 0.001:
                outward_steps += 1

        previous_position = new_position
        previous_distance = new_distance

    return {
        "success": bool(info.get("is_success", False)),
        "crash": "crash" in info,
        "time": float(info["flight_time"]),
        "path": path_length,
        "near_path": near_goal_path,
        "outward": outward_steps,
        "approach": approach_steps,
        "switches": switches,
        "entered": entered_goal_region,
    }


results = {"PPO": [], "PPO + Approach": []}

for episode in range(EPISODES):
    seed = 10000 + episode

    results["PPO"].append(
        evaluate_episode(seed, False)
    )

    results["PPO + Approach"].append(
        evaluate_episode(seed, True)
    )

print("\n=== CONTROLLER COMPARISON ===")
print(f"Episodes per controller: {EPISODES}")

for name, episodes in results.items():
    success = np.mean([e["success"] for e in episodes])
    crash = np.mean([e["crash"] for e in episodes])

    successful = [e for e in episodes if e["success"]]
    near = [e for e in episodes if e["entered"]]

    print(f"\n{name}")
    print(f"  Success rate: {success:.1%}")
    print(f"  Crash rate: {crash:.1%}")

    if successful:
        print(
            "  Mean successful flight time: "
            f"{np.mean([e['time'] for e in successful]):.2f} s"
        )
        print(
            "  Mean successful path length: "
            f"{np.mean([e['path'] for e in successful]):.2f} m"
        )

    if near:
        print(
            "  Mean path traveled near goal: "
            f"{np.mean([e['near_path'] for e in near]):.2f} m"
        )
        print(
            "  Mean outward steps near goal: "
            f"{np.mean([e['outward'] for e in near]):.1f}"
        )

    print(
        "  Total approach steps: "
        f"{sum(e['approach'] for e in episodes)}"
    )
    print(
        "  Total mode switches: "
        f"{sum(e['switches'] for e in episodes)}"
    )

	print("\n=== GOAL-APPROACH EPISODE DETAILS ===")

	for i, episode in enumerate(results["PPO + Approach"]):
   	 if episode["switches"] >= 3 or episode["outward"] >= 3:
        	print(
            	f"Episode {i:02d} | "
            	f"Success: {episode['success']} | "
            	f"Switches: {episode['switches']} | "
            	f"Outward steps: {episode['outward']} | "
            	f"Near-goal path: {episode['near_path']:.2f} m | "
            	f"Flight time: {episode['time']:.2f} s"
        	)
