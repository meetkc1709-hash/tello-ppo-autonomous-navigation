"""
STEP 3 - Measure how the real drone responds, so the simulator can match it.
(No mission pads needed.)

Needs ~3 m of clear space IN FRONT of the drone. Mark the take-off spot with tape.
    python 3_calibrate.py --rc 40
    python 3_calibrate.py --rc 70

What it does: take off, hover 2 s, fly FORWARD with rc=<value> for 1.5 s, stop, hover 2.5 s, land.
While flying it PRINTS the drone's own speed reading live, and logs it 20 times per second
to calib_rc<value>_<time>.csv (a new file every run). After landing it prints a speed summary.

ALSO MEASURE BY HAND: put tape on the floor directly under the drone after takeoff
(it may drift a little), and again where it hovers after stopping. Measure the distance
between the two marks (cm) and write it down next to the rc value.
Send the CSVs + distances back.
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

ap = argparse.ArgumentParser()
ap.add_argument("--rc", type=int, default=40, help="forward rc command, 10..100")
ap.add_argument("--move_time", type=float, default=1.5)
ap.add_argument("--side", action="store_true",
                help="fly to the RIGHT instead of forward (to check the sign of speed_y)")
args = ap.parse_args()
assert 10 <= args.rc <= 100

tello = Tello()
tello.connect()
bat = tello.get_battery(); print("Battery:", bat, "%")
YAW0 = tello.get_yaw()
print("Heading since switch-on:", YAW0, "deg  (speed readings get corrected for this)")
import _paths
from pad_tracker import speed_to_flight_frame
assert bat > 30, "Battery below 30% - charge it first"

rows = []
last_print = [0.0]

def log(phase, cmd, t0):
    t = round(time.time() - t0, 3)
    vx, vy, h = tello.get_speed_x(), tello.get_speed_y(), tello.get_height()
    rows.append([t, phase, cmd, vx, vy, h])
    if t - last_print[0] >= 0.2:            # live readout 5x per second
        last_print[0] = t
        print(f"t={t:5.2f}s  {phase:8s} rc={cmd:3d}  speed_x={vx:4d}  speed_y={vy:4d}  height={h} cm")

def run_phase(name, cmd, seconds, t0):
    end = time.time() + seconds
    while time.time() < end:
        if args.side:
            tello.send_rc_control(cmd, 0, 0, 0)   # positive left_right = move RIGHT
        else:
            tello.send_rc_control(0, cmd, 0, 0)   # (left_right, forward_back, up_down, yaw)
        log(name, cmd, t0)
        time.sleep(0.05)

def summary():
    if len(rows) < 2:
        return
    moving = [r for r in rows if r[1] == "forward"]
    half = moving[len(moving) // 2:] or moving
    if half:
        fl = sum(speed_to_flight_frame(r[3], r[4], YAW0) for r in half) / len(half)
        print(f"\nCORRECTED speed while moving: forward {fl[0]:+.2f} m/s, left {fl[1]:+.2f} m/s")
        print("  (forward test: forward should be clearly +, left ~0;  --side test: left clearly -, forward ~0)")
    if args.side:
        ys = [r[4] for r in rows if r[1] == 'forward']
        avg = sum(ys[len(ys)//2:]) / max(1, len(ys[len(ys)//2:]))
        print(f"\nSIDEWAYS test (drone moved RIGHT): average speed_y = {avg:.1f}")
        return
    fwd = [r[3] for r in rows if r[1] == "forward"]
    # speed integrated over the whole run = how far the drone thinks it moved
    dist_units = sum((rows[i][0] - rows[i - 1][0]) * (rows[i][3] + rows[i - 1][3]) / 2
                     for i in range(1, len(rows)))
    second_half = fwd[len(fwd) // 2:] or fwd
    print("\n=== Drone speed report (speed_x) ===")
    print(f"Peak reading          : {max((r[3] for r in rows), key=abs)}")
    if second_half:
        print(f"Steady reading        : {sum(second_half) / len(second_half):.1f}  (avg of 2nd half of forward phase)")
    print(f"Integrated distance   : {dist_units:.1f} units*s")
    print(f"  -> if unit is dm/s  : {dist_units * 10:.0f} cm,  steady speed {sum(second_half)/len(second_half)/10:.2f} m/s"
          if second_half else "")
    print(f"  -> if unit is cm/s  : {dist_units:.0f} cm,  steady speed {sum(second_half)/len(second_half)/100:.2f} m/s"
          if second_half else "")
    print("Compare these with your TAPE distance to see which unit is right.")

try:
    tello.takeoff()
    t0 = time.time()
    run_phase("hover", 0, 2.0, t0)
    run_phase("forward", args.rc, args.move_time, t0)
    run_phase("stop", 0, 2.5, t0)
finally:
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    close_tello(tello)
    name = f"calib_{'side' if args.side else 'rc'}{args.rc}_{time.strftime('%H%M%S')}.csv"   # new file every run
    with open(name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "phase", "rc_cmd", "speed_x", "speed_y", "height_cm"])
        w.writerows(rows)
    summary()
    print(f"\nSaved {name}  ->  now measure the distance moved with a tape!")
