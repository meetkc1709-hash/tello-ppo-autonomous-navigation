"""
Evaluate a trained model: success rate, crash rate, and how close flight time is
to the ideal straight-line time. These are the numbers for your report.

    python evaluate.py --model models/stage1/best_model.zip --stage 1 --room 10 8
    python evaluate.py --baseline --stage 2 --room 10 8      # straight-line baseline
"""
import argparse
import numpy as np
from stable_baselines3 import PPO
from tello_env import TelloNavEnv
from train import STAGES

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None)
ap.add_argument("--baseline", action="store_true",
                help="ignore the model; fly straight at the goal (for comparison)")
ap.add_argument("--stage", type=int, default=1)
ap.add_argument("--episodes", type=int, default=200)
ap.add_argument("--room", type=float, nargs=2, default=[4.0, 4.0])
ap.add_argument("--sensor", choices=["perfect", "camera"], default="perfect")
ap.add_argument("--max_speed", type=float, default=None, help="for --baseline: match the model's speed")
ap.add_argument("--margin", type=float, default=None, help="for --baseline: match the model's margin")
ap.add_argument("--static", type=int, default=None)
ap.add_argument("--moving", type=int, default=None)
ap.add_argument("--sides", action="store_true",
                help="fixed left/right test: goal straight, to the left and to the right, with a box "
                     "on the line or slightly left/right of it - shows if the agent only dodges one way")
ap.add_argument("--cam_lag", action="store_true",
                help="test with the real camera's slow, delayed readings (camera sensor)")
args = ap.parse_args()

cfg = dict(STAGES[args.stage]); cfg.pop("steps"); cfg["room_size"] = tuple(args.room)
cfg["sensor"] = args.sensor
if args.max_speed is not None: cfg["max_speed"] = args.max_speed
if args.margin is not None: cfg["drone_radius"] = args.margin
if args.static is not None: cfg["n_static_obstacles"] = args.static
if args.moving is not None: cfg["n_moving_obstacles"] = args.moving
if args.model:
    from model_config import load_config
    saved = load_config(args.model)
    for k in ("max_speed", "drone_radius", "sensor", "cam_lag", "obstacle_radius", "line_offset",
              "safe_distance"):
        if k in saved:
            cfg[k] = tuple(saved[k]) if isinstance(saved[k], list) else saved[k]
if args.cam_lag:
    cfg["cam_lag"] = True
    if saved:
        print("Using the model's saved settings:", {k: saved[k] for k in ("max_speed", "drone_radius", "sensor") if k in saved})
env = TelloNavEnv(**cfg)
model = None if args.baseline else PPO.load(args.model)
n_obs = model.observation_space.shape[0] if model else None   # older models see fewer inputs

def run_episode(seed, options=None):
    obs, _ = env.reset(seed=seed, options=options)
    done, side = False, 0.0
    while not done:
        if args.baseline:
            action = obs[:2] / max(np.linalg.norm(obs[:2]), 1e-6)
        else:
            action, _ = model.predict(obs[:n_obs], deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        if options and "obstacles" in options:        # which side of the box did it pass?
            o = np.array(options["obstacles"][0][:2])
            d = np.array(options["goal"]) - np.array(options["start"])
            if abs(env.pos[0] - o[0]) < 0.15:            # beside the box: left (+) or right (-) of it
                rel = env.pos - o
                side = np.sign(d[0] * rel[1] - d[1] * rel[0])
        done = term or trunc
    return info, side

if args.sides:
    R = env.room
    start = (0.5, R[1] / 2)
    goals = {"straight": (R[0] - 0.5, R[1] / 2), "to the LEFT": (R[0] - 0.5, R[1] / 2 + 0.75),
             "to the RIGHT": (R[0] - 0.5, R[1] / 2 - 0.75)}
    n = max(10, args.episodes // 9)
    print(f"{'goal':14s}{'box':>16s}   success  crash   passed box on L / R")
    for gname, g in goals.items():
        d = np.array(g) - np.array(start); perp = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        for bname, off in (("box on line", 0.0), ("box 15cm left", 0.15), ("box 15cm right", -0.15)):
            c = np.array(start) + 0.5 * d + off * perp
            opts = {"start": start, "goal": g, "obstacles": [(c[0], c[1], 0.25)]}
            res = [run_episode(20_000 + k, opts) for k in range(n)]
            ok = sum(bool(i.get("is_success")) for i, _ in res)
            cr = sum("crash" in i for i, _ in res)
            L = sum(sd > 0 for _, sd in res); Rr = sum(sd < 0 for _, sd in res)
            print(f"{gname:14s}{bname:>16s}   {ok:3d}/{n:<3d}  {cr:3d}     {L:3d} / {Rr:<3d}")
    raise SystemExit

succ, crash, ratios = 0, 0, []
for ep in range(args.episodes):
    obs, _ = env.reset(seed=10_000 + ep)
    done = False
    while not done:
        if args.baseline:   # obs[0:2] = direction to goal -> full speed toward it
            action = obs[:2] / max(np.linalg.norm(obs[:2]), 1e-6)
        else:
            action, _ = model.predict(obs[:n_obs], deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
    if info.get("is_success"):
        succ += 1
        ratios.append(info["flight_time"] / info["ideal_time"])
    elif "crash" in info:
        crash += 1

print(f"Success rate : {succ/args.episodes:.1%}")
print(f"Crash rate   : {crash/args.episodes:.1%}")
if ratios:
    print(f"Time / ideal : {np.mean(ratios):.2f}x  (1.00 = perfect straight line at max speed)")
