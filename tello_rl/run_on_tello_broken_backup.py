"""
Fly a trained PPO policy on the real Tello.

    # position from mission pads (+ speed readings between pads) - needs lab/5_pad_layout.py:
    python run_on_tello.py --model models/stage1/best_model.zip --room 2 2 --goal 1.5 1.0 --position pads

    # position from ArUco markers on the wall (needs lab steps 5-7 for markers):
    python run_on_tello.py --model models/stage1/best_model.zip --room 4.3 4 --goal 3.0 2.0 --position markers

    # position from the drone's own speed readings (no markers; drifts - short flights only):
    python run_on_tello.py --model models/stage1/best_model.zip --room 4.3 4 --goal 2.0 2.0 --position speed --start 0.5 2.0

Coordinates (same as the simulator): origin = BACK-RIGHT floor corner of the flight area,
x = toward the marker wall, y = to the left. The drone must take off FACING the marker wall.
--room must be the same as in training.

SAFETY: empty area, someone ready to step in, Ctrl+C in the terminal lands immediately.
The drone also lands if it leaves the safe area, loses tracking for 4 s, or times out.
Every flight is logged to flight_log_<time>.csv (great for the report).
"""
import argparse, csv, time
import cv2
import numpy as np
from djitellopy import Tello

def close_tello(t):
    """Close the drone connection once. djitellopy closes it again when Python exits, which
    fails with 'WinError 6 handle is invalid' - so disable that second close."""
    try:
        if getattr(t, "stream_on", False):
            t.streamoff()
    except Exception:
        pass
    try:
        t.end()
    except Exception:
        pass
    t.end = lambda *a, **k: None
from stable_baselines3 import PPO
from tello_env import TelloNavEnv

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--room", type=float, nargs=2, required=True, help="same as in training: depth width (m)")
ap.add_argument("--goal", type=float, nargs=2, default=None, help="goal x y (m) in room coordinates")
ap.add_argument("--goal_ahead", type=float, nargs=2, default=None, metavar=("FORWARD", "LEFT"),
                help="goal RELATIVE to the takeoff spot: metres forward and metres left (negative = right). "
                     "The pad can then be put anywhere - no fixed layout needed.")
ap.add_argument("--position", choices=["pads", "markers", "speed"], default="pads")
ap.add_argument("--max_blind", type=float, default=8.0, help="pads mode: land if no pad seen for this long (s)")
ap.add_argument("--height_cm", type=int, default=None,
                help="flying height. Default: 35 in pads mode (pads only detected below ~50 cm), else takeoff height")
ap.add_argument("--start", type=float, nargs=2, default=None, help="takeoff spot x y (needed for --position speed)")
ap.add_argument("--max_rc", type=int, default=None,
                help="safety cap on rc commands. Default: what the model's trained top speed needs. "
                     "A LOWER cap than that makes the drone slower than the agent expects (weaker dodges).")
ap.add_argument("--vy_sign", type=float, default=-1.0,
                help="speed mode only: from '3_calibrate.py --side' (+1 or -1)")
ap.add_argument("--obstacle", type=float, nargs=3, action="append", default=[], metavar=("X", "Y", "R"),
                help="known obstacle in room coordinates: centre x y and radius (m). Repeat for more.")
ap.add_argument("--obstacle_ahead", type=float, nargs=3, action="append", default=[],
                metavar=("FORWARD", "LEFT", "R"),
                help="known obstacle RELATIVE to the takeoff spot (use with --goal_ahead): metres forward, "
                     "metres left (negative = right), radius. Radius = half the object's width + ~5 cm.")
ap.add_argument("--detect", choices=["none", "camera"], default="none",
                help="camera = the drone finds obstacles itself with its front camera + depth model "
                     "(needs a model trained with --sensor camera)")
ap.add_argument("--preview", action="store_true",
                help="show a live camera window with obstacle and flight status during flight; press q in the preview to request landing")
ap.add_argument("--depth_height", type=int, default=196,
                help="image height for the depth model (smaller = faster; 140-280)")
ap.add_argument("--no_shield", action="store_true",
                help="turn off the safety brake (agent only)")
ap.add_argument("--shield_dist", type=float, default=1.0,
                help="safety brake starts when something is this close straight ahead (m)")
ap.add_argument("--stop_dist", type=float, default=0.45,
                help="safety brake: no forward motion closer than this (m)")
ap.add_argument("--no_memory", action="store_true",
                help="don't remember obstacles that drop out of the camera's view (not recommended)")
ap.add_argument("--pitch", type=float, default=0.0, help="camera tilt down (deg), from lab/8_check_depth.py")
ap.add_argument("--max_depth_age", type=float, default=2.5,
                help="stop if the newest depth reading is older than this (s)")
args = ap.parse_args()
if args.preview and args.detect != "camera":
    ap.error("--preview requires --detect camera (shares the same Tello video stream)")

from model_config import load_config
saved = load_config(args.model)
if saved.get("room_size") and [float(v) for v in saved["room_size"]] != [float(v) for v in args.room]:
    print(f"WARNING: model was trained with --room {saved['room_size']}, you gave {args.room}")
env = TelloNavEnv(room_size=tuple(args.room), pos_noise=0, vel_noise=0, randomize=False,
                  sensor="camera" if args.detect == "camera" else "perfect",
                  max_speed=saved.get("max_speed", 0.8), drone_radius=saved.get("drone_radius", 0.10))
print(f"Model top speed {env.max_speed:.2f} m/s, safety margin {env.drone_r:.2f} m")
model = PPO.load(args.model)
n_obs = model.observation_space.shape[0]     # older models (before the delay fix) use fewer inputs
if args.detect == "camera" and n_obs != env.observation_space.shape[0]:
    raise SystemExit("--detect camera needs a model trained with --sensor camera "
                     "(e.g. models/stage2_cam/best_model.zip)")
env.reset()
if (args.goal is None) == (args.goal_ahead is None):
    raise SystemExit("Give either --goal x y (room coordinates) or --goal_ahead forward left")
if args.goal_ahead is not None:
    # Place the trip in the middle of the trained room: the agent only cares about the goal
    # RELATIVE to the drone, so where the room "is" doesn't matter - only that both ends fit.
    rel = np.array(args.goal_ahead, dtype=np.float64)
    start_pt = np.array(args.room) / 2 - rel / 2
    goal_pt = start_pt + rel
    lo, hi = 0.3, np.array(args.room) - 0.3
    if np.any(start_pt < lo) or np.any(start_pt > hi) or np.any(goal_pt < lo) or np.any(goal_pt > hi):
        raise SystemExit(f"--goal_ahead {args.goal_ahead} is too far for a {args.room[0]} x {args.room[1]} m room "
                         f"(max about {args.room[0]-0.6:.1f} m forward / {args.room[1]-0.6:.1f} m sideways). "
                         "Use a shorter goal or train with a bigger --room.")
    args.start = [float(v) for v in start_pt]
    args.goal = [float(v) for v in goal_pt]
    print(f"Relative goal: {rel[0]:.2f} m forward, {rel[1]:.2f} m left of the takeoff spot")
if args.position == "speed" and args.start is None:
    raise SystemExit("--position speed needs --start x y, or use --goal_ahead (start is then automatic)")
env.goal = np.array(args.goal, dtype=np.float32)
# known obstacles (the agent plans around them exactly like in simulation)
obstacles = [list(o) for o in args.obstacle]
if args.obstacle_ahead:
    if args.goal_ahead is None:
        raise SystemExit("--obstacle_ahead needs --goal_ahead (both relative to the takeoff spot)")
    obstacles += [[args.start[0] + f, args.start[1] + l, r] for f, l, r in args.obstacle_ahead]
for ox, oy, r in obstacles:
    if not (0 < ox < args.room[0] and 0 < oy < args.room[1]):
        raise SystemExit(f"Obstacle at ({ox:.2f}, {oy:.2f}) is outside the trained room {args.room}")
    print(f"Obstacle: centre ({ox:.2f}, {oy:.2f}) m, radius {r:.2f} m")
env.obs_pos = np.array([[o[0], o[1]] for o in obstacles], np.float32).reshape(-1, 2)
env.obs_vel = np.zeros((len(obstacles), 2), np.float32)
env.obs_r = np.array([o[2] for o in obstacles], np.float32)
if not (np.all(env.goal > 0.2) and np.all(env.goal < env.room - 0.2)):
    raise SystemExit(f"Goal {args.goal} is outside the room {args.room}")

# ------------------------------------------------- speed -> rc (measured in the lab)
# rc 40 -> 0.31 m/s, rc 60 -> 0.51 m/s, rc 80 -> 0.82 m/s  =>  speed = 0.01275*rc - 0.22
RC_DEADZONE = 17
RC_PER_MPS = 78

needed_rc = int(np.ceil(RC_DEADZONE + RC_PER_MPS * env.max_speed))
if args.max_rc is None:
    args.max_rc = min(100, needed_rc)
elif args.max_rc < needed_rc:
    print(f"WARNING: --max_rc {args.max_rc} is below the {needed_rc} this model needs for its top speed "
          f"({env.max_speed:.2f} m/s) - the drone will be SLOWER than the agent expects. "
          "Better: train a slower model (train.py --max_speed ...).")

def speed_to_rc(v):
    if abs(v) < 0.03:
        return 0
    return int(np.sign(v) * min(RC_DEADZONE + RC_PER_MPS * abs(v), args.max_rc))

SHIELD = {"side": None}

# Limit rapid lateral command reversals
previous_lr = 0
MAX_LR_CHANGE = 10

def shield(action, rays, vel, goal_rel):
    """Safety brake on top of the agent (real drone only). If the camera sees something close
    straight ahead, limit the forward speed so the drone can still stop in time (it reacts ~0.5 s
    late), and make sure it is moving sideways toward the freer side. Returns (action, note)."""
    n = len(rays); mid = n // 2
    front = float(np.min(rays[mid - 1: mid + 2]))           # the 3 middle rays
    if front >= args.shield_dist:
        if front >= args.shield_dist + 0.3:
            SHIELD["side"] = None                            # encounter over - free to choose again
        return action, ""
    a = action.copy()
    # forward speed allowed: full at shield_dist, zero at stop_dist, back off a little if closer
    frac = (front - args.stop_dist) / max(args.shield_dist - args.stop_dist, 1e-3)
    a[0] = min(a[0], float(np.clip(frac, -0.4, 1.0)))
    # sideways: pick a side ONCE per encounter and stick to it (no zig-zag). Prefer the agent's
    # side, unless the camera shows clearly more room on the other side.
    left, right = float(np.mean(rays[:mid])), float(np.mean(rays[mid + 1:]))
    side = SHIELD.get("side")
    if side is None:
        if abs(a[1]) >= 0.2:
            side = np.sign(a[1])
            if (side > 0 and right > left + 0.3) or (side < 0 and left > right + 0.3):
                side = -side                                 # agent's side is blocked - switch
        else:
            side = np.sign(left - right) if abs(left - right) > 0.15 else (np.sign(goal_rel[1]) or 1.0)
        SHIELD["side"] = side
    need = float(np.clip(1.2 * (1 - frac), 0.5, 1.0))
    a[1] = side * max(abs(a[1]), need)
    return np.clip(a, -1, 1), f"brake fwd<={a[0]:.2f} side={'L' if side > 0 else 'R'} front={front:.2f}"

def to_rc(action):
    # env: x = forward, y = left ; Tello rc: left_right (+ = right), forward_backward (+ = forward)
    vx, vy = np.clip(action, -1, 1) * env.max_speed
    return speed_to_rc(-vy), speed_to_rc(vx)

# ------------------------------------------------- connect
tello = Tello()
tello.connect()
bat = tello.get_battery(); print("Battery:", bat, "%")
assert bat > 30, "Charge the battery first"

depth = None
memory = None
if args.detect == "camera":
    from depth_sensor import DepthRays, DepthThread
    tello.streamon()
    cam = tello.get_frame_read()
    time.sleep(2)
    depth = DepthThread(DepthRays(n_rays=env.n_rays, max_range=env.ray_range,
                                  input_height=args.depth_height, pitch_deg=args.pitch),
                        lambda: cam.frame)
    t_wait = time.time()
    while depth.latest()[0] is None and time.time() - t_wait < 20:
        time.sleep(0.2)
    if depth.latest()[0] is None:
        close_tello(tello); raise SystemExit("Depth model produced nothing - check with lab/8_check_depth.py")
    time.sleep(3)                                    # let the speed estimate settle
    print(f"Depth camera running at ~{depth.fps:.1f} rays/s")
    if not args.no_memory:
        from depth_sensor import ObstacleMemory
        memory = ObstacleMemory(env.ray_angles, env.ray_range)
    if depth.fps < 1.0 / args.max_depth_age:
        depth.stop(); close_tello(tello)
        raise SystemExit(f"Depth model too slow ({depth.fps:.2f}/s) for --max_depth_age {args.max_depth_age}. "
                         "Try --depth_height 140, or close other programs.")

if args.position == "pads":
    from pad_tracker import PadTracker
    tello.enable_mission_pads()
    tello.set_mission_pad_detection_direction(0)       # downward camera only (faster updates)
    start = args.start
    if args.goal_ahead is not None:
        # pad can be anywhere: it is simply 'the takeoff spot' = start
        pad_tracker = PadTracker(tello, start=start, yaw0=None, takeoff_pad_at=start)
    elif start is None:
        import json as _json
        from pad_tracker import LAYOUT_FILE
        with open(LAYOUT_FILE) as _f:
            _pads = _json.load(_f)["pads"]
        if len(_pads) == 1:                      # one pad = the takeoff spot -> start there
            start = list(_pads.values())[0]
            print(f"Starting position = takeoff pad at {start} (pad sightings will correct it)")
    if args.goal_ahead is None:
        pad_tracker = PadTracker(tello, start=start, yaw0=None)
elif args.position == "markers":
    from marker_tracker import MarkerTracker
    tracker = MarkerTracker()
    tello.streamon()
    reader = tello.get_frame_read()
    time.sleep(2)

def read_position(state):
    """(x, y) in metres, or None."""
    if args.position == "pads":
        p = pad_tracker.update()
        state["n_markers"] = pad_tracker.last_pad          # logs which pad (-1 = none)
        state["blind"] = pad_tracker.seconds_since_fix()
        return p
    if args.position == "markers":
        frame = reader.frame
        if frame is None:
            return None
        corners, ids = tracker.detect(frame)
        state["n_markers"] = len(ids)
        p = tracker.locate_from_corners(corners, ids) if ids else None
        return None if p is None else p[:2].astype(np.float32)
    # speed mode: add up the drone's own speed readings (dm/s -> m/s)
    now = time.time()
    dt = now - state.get("t_int", now)
    state["t_int"] = now
    v = speed_to_flight_frame(tello.get_speed_x(), tello.get_speed_y(), YAW0).astype(np.float32)
    state["dr"] = state.get("dr", np.array(args.start, np.float32)) + v * dt
    return state["dr"].copy()

# ------------------------------------------------- fly
log, state = [], {}
vel = np.zeros(2, np.float32)
last_pos, last_seen = None, time.time()
t_limit = env.max_steps * env.dt
HEIGHT = args.height_cm if args.height_cm is not None else (35 if args.position == "pads" else None)

def height_hold():
    """rc up/down value that keeps the drone at HEIGHT (uses the downward distance sensor)."""
    if HEIGHT is None:
        return 0
    try:
        h = tello.get_distance_tof()            # cm above the floor
    except Exception:
        return 0
    if h <= 0 or h > 300:                       # bad reading -> don't react
        return 0
    return int(np.clip(0.8 * (HEIGHT - h), -25, 25))

# The Tello's speed readings point along the direction it faced when SWITCHED ON. We read its
# heading now (before takeoff) and convert the readings with it - see pad_tracker.py.
from pad_tracker import speed_to_flight_frame
YAW0 = tello.get_yaw()
print(f"Heading since switch-on: {YAW0} deg (speed readings are corrected for this)")

result = "timeout"
try:
    tello.takeoff()
    if HEIGHT is not None:                          # go down to the low flying height
        try:
            drop = tello.get_distance_tof() - HEIGHT
            if drop >= 20:
                tello.move_down(int(min(drop, 200)))
        except Exception as e:
            print("Could not descend with move_down:", e)
        t_settle = time.time()
        while time.time() - t_settle < 1.5:         # settle at the new height
            tello.send_rc_control(0, 0, height_hold(), 0)
            time.sleep(0.1)
        try:
            print(f"Flying height: {tello.get_distance_tof()} cm (target {HEIGHT})")
        except Exception:
            pass
    if args.preview:
        cv2.namedWindow("Tello PPO Live Preview", cv2.WINDOW_NORMAL)
        print("Live preview enabled. Press q in the video window to request landing.")
    t0 = time.time()
    last_seen = t0
    result = "timeout"
    try:
        while time.time() - t0 < t_limit:
            loop_start = time.time()
            pos = read_position(state)
            now = time.time()
            if pos is None:
                tello.send_rc_control(0, 0, height_hold(), 0)   # no position yet / lost -> hover, don't guess
                if now - last_seen > 4.0:
                    result = "no position (no pad seen) - land" if args.position == "pads" else "lost tracking"
                    break
                time.sleep(0.05); continue
            if args.position == "pads" and state.get("blind", 0) > args.max_blind:
                result = f"no pad seen for {args.max_blind:.0f}s - position no longer trusted"; break
            if last_pos is not None:                        # smoothed velocity from position
                vel = 0.5 * vel + 0.5 * (pos - last_pos) / max(now - last_seen, 1e-3)
            last_pos, last_seen = pos, now
            env.pos, env.vel = pos, vel

            dist = float(np.linalg.norm(env.goal - pos))
            if dist < env.goal_radius:
                result = f"reached goal in {now - t0:.1f}s"; break
            if len(env.obs_pos) and np.any(np.linalg.norm(env.obs_pos - pos, axis=1) < env.obs_r + 0.05):
                result = f"too close to an obstacle at {pos.round(2)} - stopped"; break
            if np.any(pos < 0.1) or np.any(pos > env.room - 0.1):
                result = f"left safe area at {pos.round(2)}"; break

            if depth is not None:
                rays, age, _ = depth.latest()
                reading_time = time.time() - age
                if age > args.max_depth_age:                   # camera/depth stalled -> don't fly blind
                    tello.send_rc_control(0, 0, 0, 0)
                    result = f"depth camera stalled ({age:.1f}s old, {depth.fps:.2f} readings/s) - stopped"; break
                try:                                           # camera height for the floor filter
                    tof = tello.get_distance_tof()
                    if 10 < tof < 300:
                        depth.sensor.cam_height = tof / 100.0
                except Exception:
                    pass
                if memory is not None:                         # objects that went out of view stay known
                    memory.update(np.array(pos, float), rays, round(reading_time, 3))
                    rays = np.minimum(rays, memory.rays(np.array(pos, float)))
                env.external_new = (reading_time != state.get("last_reading"))
                state["last_reading"] = reading_time
                env.external_rays = rays
                state["min_ray"] = float(np.min(rays))
            action, _ = model.predict(env._get_obs()[:n_obs], deterministic=True)
            action = np.array(action, dtype=np.float32)
            state["shield"] = ""
            if depth is not None and not args.no_shield:
                action, state["shield"] = shield(action, rays, vel, goal_rel=np.asarray(args.goal) - np.asarray(pos))
            env.record_action(action)                       # agent knows what it already sent (0.5 s delay)
                        lr, fb = to_rc(action)

            # Smooth lateral movement only when the safety shield is inactive
            if not state["shield"]:
                lr = int(np.clip(
                    lr,
                    previous_lr - MAX_LR_CHANGE,
                    previous_lr + MAX_LR_CHANGE
                ))

            previous_lr = lr

            tello.send_rc_control(lr, fb, height_hold(), 0)   
            log.append([round(now - t0, 2), round(float(pos[0]), 3), round(float(pos[1]), 3),
                        round(float(vel[0]), 3), round(float(vel[1]), 3), round(dist, 3),
                        round(float(action[0]), 3), round(float(action[1]), 3), lr, fb,
                        state.get("n_markers", ""), state.get("min_ray", ""), state.get("shield", "")])
            if len(log) % 5 == 0:
                print(f"t={now - t0:4.1f}s  pos=({pos[0]:.2f},{pos[1]:.2f})  dist={dist:.2f}  rc lr={lr} fb={fb}"
                      + (f"  nearest={state['min_ray']:.2f}" if state.get("min_ray", "") != "" else "")
                      + (f"  [{state['shield']}]" if state.get("shield") else ""))
            if args.preview:
                # Reuse the camera frame reader already owned by this Tello connection.
                # A preview never launches another script or opens another UDP socket.
                frame = cam.frame
                if frame is not None:
                    display = frame.copy()
                    cv2.putText(display, f"Goal remaining (estimated): {dist:.2f} m", (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    depth_label = (f"Nearest depth estimate: {state['min_ray']:.2f} m"
                                   if isinstance(state.get('min_ray'), (float, int)) else "Depth: unavailable")
                    cv2.putText(display, depth_label, (12, 57),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    cv2.putText(display, f"RC right={lr} forward={fb}", (12, 86),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    if state["shield"]:
                        cv2.putText(display, "SAFETY BRAKE ACTIVE", (12, 115),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display, "Press Q to request landing", (12, display.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow("Tello PPO Live Preview", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    result = "stopped by Q in live preview"
                    break
            time.sleep(max(0.0, env.dt - (time.time() - loop_start)))
    except KeyboardInterrupt:
        result = "stopped by Ctrl+C"
finally:
    # save the log FIRST, so it survives even if the drone stops answering
    name = f"flight_log_{time.strftime('%H%M%S')}.csv"
    try:
        with open(name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x", "y", "vx", "vy", "dist_to_goal", "action_x", "action_y", "rc_lr", "rc_fb", "pad_or_markers", "nearest_seen_m", "shield"])
            w.writerows(log)
    except Exception as e:
        print("Could not save the flight log:", e)
    try:
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
    except Exception as e:
        print("LAND COMMAND FAILED - the drone may still be flying! Use the phone app or switch it off.", e)
    if depth is not None:
        depth.stop()
    if args.preview:
        try:
            cv2.destroyWindow("Tello PPO Live Preview")
        except cv2.error:
            pass
    close_tello(tello)
    name_written = name
    print(f"\nResult: {result}")
    if log:
        start = np.array(log[0][1:3])
        ideal = (np.linalg.norm(env.goal - start) - env.goal_radius) / env.max_speed
        print(f"Ideal straight-line time from start: {ideal:.1f}s")
    print(f"Saved {name}")
