"""
STEP 2 - Can the drone see the mission pad and tell where it is? (Tello EDU only)

Put the mission pad flat on the floor. Then:
    python 2_check_position.py           # drone on the ground / held by hand over the pad
    python 2_check_position.py --fly     # hover over the pad and read position

Prints pad id and x/y/z (cm) 5 times per second for 20 s. pad = -1 means no pad seen.
FIND THE PAD AXES (needed once, for pad_layout.json):
  Hold the drone over a pad, camera pointing to the FRONT of the flight area.
  1. Move it ~20 cm FORWARD  -> which number changes, x or y, and up or down?
       x goes up  -> pad_x_is +x      x goes down -> pad_x_is -x
       y goes up  -> pad_y_is +x      y goes down -> pad_y_is -x
  2. Move it ~20 cm LEFT     -> same idea for +y / -y.
  Then re-run 5_pad_layout.py with --pad_x_is ... --pad_y_is ...
If pad_layout.json exists, the room position (world x, y) is printed too - check it with a tape.
Saved to position_log.csv.
"""
import argparse, csv, time
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
import os, _paths
layout_ok = os.path.exists(os.path.join(_paths.ROOT, "pad_layout.json"))
if layout_ok:
    from pad_tracker import PadTracker

ap = argparse.ArgumentParser()
ap.add_argument("--fly", action="store_true")
ap.add_argument("--seconds", type=float, default=20)
args = ap.parse_args()

tello = Tello()
tello.connect()
print("Battery:", tello.get_battery(), "%")
try:
    tello.enable_mission_pads()
    tello.set_mission_pad_detection_direction(0)  # 0 = downward camera
except Exception as e:
    raise SystemExit(f"Mission pads not supported ({e}) - this is probably not a Tello EDU.")

time.sleep(2)   # give the drone a moment after switching pad detection on
print("Hold the drone ~40 cm above the CENTRE of the pad. 'floor' = real distance to the floor.\n")

rows = []
pt = PadTracker(tello) if layout_ok else None
try:
    if args.fly:
        tello.takeoff()
        time.sleep(1)
        drop = tello.get_distance_tof() - 35          # pads are only seen below ~50 cm
        if drop >= 20:
            tello.move_down(int(min(drop, 200)))
        time.sleep(1)
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        pad = tello.get_mission_pad_id()
        x, y, z = (tello.get_mission_pad_distance_x(), tello.get_mission_pad_distance_y(),
                   tello.get_mission_pad_distance_z())
        t = time.time() - t0
        if args.fly:
            tello.send_rc_control(0, 0, 0, 0)
        world = ""
        if pt is not None:
            w, _ = pt.pad_reading()
            world = f"   ->  room position x={w[0]:.2f} y={w[1]:.2f} m" if w is not None else "   ->  (pad not in layout)"
        floor = tello.get_distance_tof()
        hint = ""
        if pad == -1:
            hint = "   <- too close to the floor? (needs 25-50 cm)" if floor < 25 else \
                   ("   <- too high (needs under ~50 cm)" if floor > 55 else "   <- pad not under the camera?")
        print(f"t={t:5.1f}s  pad={pad:3d}  x={x:5d}  y={y:5d}  z={z:5d}  floor={floor:4d} cm"
              + (world if pad != -1 else hint))
        rows.append([round(t, 2), pad, x, y, z])
        time.sleep(0.2)
finally:
    if args.fly:
        tello.land()
    close_tello(tello)
    with open("position_log.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "pad", "x_cm", "y_cm", "z_cm"]); w.writerows(rows)
    seen = sum(1 for r in rows if r[1] != -1)
    print(f"Saved position_log.csv - pad seen in {seen}/{len(rows)} readings")
