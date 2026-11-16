"""
STEP 3 - Measure how the real drone responds, so the simulator can match it.
(No mission pads needed.)

Needs ~3 m of clear space IN FRONT of the drone. Mark the take-off spot with tape.
    python 3_calibrate.py --rc 40
    python 3_calibrate.py --rc 70

What it does: take off, hover 2 s, fly FORWARD with rc=<value> for 1.5 s, stop, hover 2.5 s, land.
While flying it logs the drone's own speed reading 20 times per second to calib_rc<value>.csv.

ALSO MEASURE BY HAND: put tape on the floor directly under the drone after takeoff
(it may drift a little), and again where it hovers after stopping. Measure the distance
between the two marks (cm) and write it down next to the rc value.
Send the CSVs + distances back.
"""
import argparse, csv, time
from djitellopy import Tello

ap = argparse.ArgumentParser()
ap.add_argument("--rc", type=int, default=40, help="forward rc command, 10..100")
ap.add_argument("--move_time", type=float, default=1.5)
args = ap.parse_args()
assert 10 <= args.rc <= 100

tello = Tello()
tello.connect()
bat = tello.get_battery(); print("Battery:", bat, "%")
assert bat > 30, "Battery below 30% - charge it first"

rows = []
def log(phase, cmd, t0):
    rows.append([round(time.time() - t0, 3), phase, cmd,
                 tello.get_speed_x(), tello.get_speed_y(), tello.get_height()])

def run_phase(name, cmd, seconds, t0):
    end = time.time() + seconds
    while time.time() < end:
        tello.send_rc_control(0, cmd, 0, 0)   # (left_right, forward_back, up_down, yaw)
        log(name, cmd, t0)
        time.sleep(0.05)

try:
    tello.takeoff()
    t0 = time.time()
    run_phase("hover", 0, 2.0, t0)
    run_phase("forward", args.rc, args.move_time, t0)
    run_phase("stop", 0, 2.5, t0)
finally:
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.end()
    name = f"calib_rc{args.rc}.csv"
    with open(name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "phase", "rc_cmd", "speed_x", "speed_y", "height_cm"])
        w.writerows(rows)
    print(f"Saved {name}  ->  now measure the distance moved with a tape!")
