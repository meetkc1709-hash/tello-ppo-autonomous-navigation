"""
Train PPO on TelloNavEnv with a simple curriculum.

    pip install "stable-baselines3[extra]" gymnasium numpy
    python train.py --stage 1 --room 10 8   # empty 10 m x 8 m room, A -> B
    (use the SAME --room for every stage, evaluate.py and run_on_tello.py)
    python train.py --stage 2 --room 10 8   # static obstacles (starts from stage-1 model)
    python train.py --stage 3 --room 10 8   # static + moving obstacles
    tensorboard --logdir logs        # watch reward / success rate

Each stage fine-tunes the previous stage's model, so the agent keeps what it learned.
"""
import argparse, os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from tello_env import TelloNavEnv

STAGES = {
    1: dict(n_static_obstacles=0, n_moving_obstacles=0, steps=300_000),
    2: dict(n_static_obstacles=3, n_moving_obstacles=0, steps=2_000_000),
    3: dict(n_static_obstacles=2, n_moving_obstacles=2, steps=2_000_000),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--room", type=float, nargs=2, default=[4.0, 4.0],
                    help="flight area width length in metres, e.g. --room 10 8")
    ap.add_argument("--sensor", choices=["perfect", "camera"], default="perfect",
                    help="camera = agent only sees what the front camera sees (for real self-detection)")
    ap.add_argument("--tag", default=None, help="model folder suffix (default: '_cam' for camera sensor)")
    ap.add_argument("--max_speed", type=float, default=0.8,
                    help="top speed (m/s) the agent may use - fly with the SAME value (saved with the model)")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="drone 'size' radius (m) used for collisions in training. Bigger = keeps a wider "
                         "berth around obstacles (real-world errors). 0.10 = real size, 0.25 = safe.")
    ap.add_argument("--cam_lag", action="store_true",
                    help="camera sensor: simulate the real depth camera's slow, delayed readings")
    ap.add_argument("--obs_size", type=float, nargs=2, default=None, metavar=("MIN_R", "MAX_R"),
                    help="obstacle radius range (m), default 0.15 0.30")
    ap.add_argument("--safe_distance", type=float, default=None,
                    help="start the 'too close' penalty at this distance (m), default 0.4")
    ap.add_argument("--line_offset", type=float, default=None,
                    help="how far beside the A->B line the blocking obstacle may sit (m), default 0.1")
    ap.add_argument("--line_offset_min", type=float, default=None,
                    help="blocking obstacle at least this far beside the line (m), so one side is clearly free")
    ap.add_argument("--ent_coef", type=float, default=None,
                    help="exploration bonus (e.g. 0.005) - helps the agent try new moves when fine-tuning")
    ap.add_argument("--reset_std", type=float, default=None,
                    help="when starting from an old model: reset its action noise to this (e.g. 0.6) so it "
                         "explores again instead of repeating its old habit")
    ap.add_argument("--init", default=None,
                    help="start from this model instead of the previous stage (e.g. models/stage2_cam/best_model.zip)")
    ap.add_argument("--steps", type=int, default=None, help="override training steps")
    ap.add_argument("--static", type=int, default=None, help="override number of fixed obstacles")
    ap.add_argument("--moving", type=int, default=None, help="override number of moving obstacles")
    args = ap.parse_args()

    cfg = dict(STAGES[args.stage]); total_steps = cfg.pop("steps")
    if args.static is not None: cfg["n_static_obstacles"] = args.static
    if args.moving is not None: cfg["n_moving_obstacles"] = args.moving
    cfg["room_size"] = tuple(args.room)
    cfg["sensor"] = args.sensor
    cfg["max_speed"] = args.max_speed
    cfg["drone_radius"] = args.margin
    if args.cam_lag: cfg["cam_lag"] = True
    if args.obs_size: cfg["obstacle_radius"] = tuple(args.obs_size)
    if args.safe_distance is not None: cfg["safe_distance"] = args.safe_distance
    if args.line_offset is not None: cfg["line_offset"] = args.line_offset
    if args.line_offset_min is not None: cfg["line_offset_min"] = args.line_offset_min
    if args.steps: total_steps = args.steps
    tag = args.tag if args.tag is not None else ("_cam" if args.sensor == "camera" else "")
    env = make_vec_env(TelloNavEnv, n_envs=args.n_envs, env_kwargs=cfg)
    eval_env = make_vec_env(TelloNavEnv, n_envs=1, env_kwargs=cfg)

    # remember the settings next to the model, so run_on_tello.py / evaluate.py can't mismatch them
    import json
    os.makedirs(f"models/stage{args.stage}{tag}", exist_ok=True)
    with open(f"models/stage{args.stage}{tag}/env_config.json", "w") as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, f, indent=2)
    prev = args.init or f"models/stage{args.stage-1}{tag}/best_model.zip"
    if (args.init or args.stage > 1) and os.path.exists(prev):
        print(f"Fine-tuning from {prev}")
        model = PPO.load(prev, env=env, tensorboard_log="logs")
    else:
        model = PPO(
            "MlpPolicy", env,
            n_steps=1024,          # per env -> 8192 samples per update
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log="logs",
            verbose=1,
        )

    if args.ent_coef is not None:
        model.ent_coef = args.ent_coef
    if args.reset_std is not None and hasattr(model.policy, "log_std"):
        import torch
        with torch.no_grad():
            model.policy.log_std.fill_(float(np.log(args.reset_std)))
        print(f"Action noise reset to std {args.reset_std}")

    eval_cb = EvalCallback(eval_env, best_model_save_path=f"models/stage{args.stage}{tag}",
                           log_path=f"logs/eval_stage{args.stage}{tag}",
                           eval_freq=10_000, n_eval_episodes=50, deterministic=True)
    model.learn(total_timesteps=total_steps, callback=eval_cb,
                tb_log_name=f"stage{args.stage}{tag}", reset_num_timesteps=False)
    model.save(f"models/stage{args.stage}{tag}/final_model")

if __name__ == "__main__":
    main()
