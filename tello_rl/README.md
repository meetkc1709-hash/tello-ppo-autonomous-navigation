# Tello PPO navigation – starter

Set `--room W L` (metres) to your real flight area and use the same value everywhere.

```
pip install "stable-baselines3[extra]" gymnasium numpy djitellopy
python train.py --stage 1 --room 10 8                 # A -> B in empty room
python evaluate.py --model models/stage1/best_model.zip --stage 1 --room 10 8
python train.py --stage 2 --room 10 8                 # + static obstacles
python train.py --stage 3 --room 10 8                 # + moving obstacles
tensorboard --logdir logs
python run_on_tello.py --model models/stage1/best_model.zip --goal 2.5 0.0 --room 10 8
```

Files
- `tello_env.py` – Gymnasium env (velocity-command drone, lag, latency, noise, obstacles, reward)
- `train.py` – PPO training with a 3-stage curriculum
- `evaluate.py` – success rate, crash rate, flight time vs. ideal straight-line time
- `run_on_tello.py` – runs the policy on a Tello EDU using mission pads for position

Baseline (greedy "fly straight at goal"), 200 episodes:
- empty room: 100% success, 1.21x ideal time
- 3 static + 2 moving obstacles: 67% success  ← PPO should beat this
