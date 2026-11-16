"""
STEP 4 - Turn the calibration CSVs into simulator numbers. (No drone needed.)

    python 4_analyze_calibration.py                       # analyzes every calib_rc*.csv here
    python 4_analyze_calibration.py --tape 40=54.6 70=95  # also use tape averages (cm) per rc

For each run it estimates:
  - latency   : time from sending the forward command until the speed reading starts to change
  - tau       : time from that moment until speed reaches 63% of its steady value
  - steady    : steady forward speed reading while the command is held
  - distance  : speed integrated over time (= how far the drone thinks it moved)

Units: the Tello's speed reading has no clearly documented unit (dm/s or cm/s depending
on the source). If you pass --tape, the script compares the integrated distance with your
tape measurement to work out the unit and reports speed in m/s from the tape.
Note: the reading is a coarse integer, so timing values are approximate (~0.05-0.1 s).
"""
import argparse, csv, glob, re
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--files", default="calib_rc*.csv")
ap.add_argument("--tape", nargs="*", default=[], help="rc=average_cm pairs, e.g. 40=54.6")
ap.add_argument("--move_time", type=float, default=1.5)
args = ap.parse_args()
tape = {int(k): float(v) for k, v in (t.split("=") for t in args.tape)}

def analyze(path):
    t, phase, vx = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            t.append(float(r["t"])); phase.append(r["phase"]); vx.append(float(r["speed_x"]))
    t, vx = np.array(t), np.array(vx)
    fwd = np.array([p == "forward" for p in phase])
    if not fwd.any():
        return None
    t_cmd = t[fwd][0]
    baseline = np.median(vx[t < t_cmd]) if (t < t_cmd).any() else 0.0
    v = vx - baseline
    # steady speed: last half of the forward phase
    fwd_t = t[fwd]
    steady_mask = fwd & (t >= fwd_t[0] + 0.5 * (fwd_t[-1] - fwd_t[0]))
    steady = float(np.mean(v[steady_mask]))
    sign = 1 if steady >= 0 else -1
    after = (t >= t_cmd)
    moving = np.where(after & (sign * v >= max(1.0, 0.15 * abs(steady))))[0]
    latency = float(t[moving[0]] - t_cmd) if len(moving) else float("nan")
    reach = np.where(after & (sign * v >= 0.63 * abs(steady)))[0]
    tau = float(t[reach[0]] - t_cmd - latency) if len(reach) and len(moving) else float("nan")
    dist_units = float(np.trapezoid(v, t)) if hasattr(np, "trapezoid") else float(np.trapz(v, t))
    return dict(latency=latency, tau=max(tau, 0.0), steady=steady, dist_units=dist_units)

files = sorted(glob.glob(args.files))
if not files:
    raise SystemExit("No calib_rc*.csv files found in this folder.")

by_rc = defaultdict(list)
print(f"{'file':32s} {'latency(s)':>10s} {'tau(s)':>7s} {'steady':>7s} {'dist(units*s)':>14s}")
for path in files:
    m = re.search(r"calib_rc(\d+)", path)
    res = analyze(path)
    if not m or res is None:
        print(f"{path:32s}  (skipped - no forward phase)"); continue
    rc = int(m.group(1)); by_rc[rc].append(res)
    print(f"{path:32s} {res['latency']:10.2f} {res['tau']:7.2f} {res['steady']:7.1f} {res['dist_units']:14.1f}")

print("\n=== Summary per rc ===")
lat_all, tau_all, rc_per_mps = [], [], []
for rc in sorted(by_rc):
    rs = by_rc[rc]
    lat = np.nanmean([r["latency"] for r in rs]); tau = np.nanmean([r["tau"] for r in rs])
    steady = np.mean([r["steady"] for r in rs]); dist_u = np.mean([r["dist_units"] for r in rs])
    lat_all.append(lat); tau_all.append(tau)
    line = f"rc {rc:3d}: runs={len(rs)}  latency={lat:.2f}s  tau={tau:.2f}s  steady reading={steady:.1f}"
    if rc in tape:
        v_tape = tape[rc] / 100.0 / args.move_time           # m/s from tape
        cm_per_unit = tape[rc] / dist_u if dist_u else float("nan")
        unit = "dm/s" if 5 < cm_per_unit < 20 else ("cm/s" if 0.5 < cm_per_unit < 2 else "unclear")
        rc_per_mps.append(rc / v_tape)
        line += f"\n         tape: {tape[rc]:.1f} cm -> speed {v_tape:.2f} m/s -> RC_PER_MPS {rc / v_tape:.0f}" \
                f"   (speed reading unit looks like: {unit}, 1 unit*s = {cm_per_unit:.1f} cm)"
    else:
        line += f"\n         speed if reading is dm/s: {steady/10:.2f} m/s,  if cm/s: {steady/100:.2f} m/s" \
                f"   (pass --tape {rc}=<cm> to decide)"
    print(line)

print("\n=== Put these into the code ===")
dt = 0.1
print(f"tello_env.py : tau = {np.nanmean(tau_all):.2f}")
print(f"tello_env.py : latency_steps = {max(0, int(round(np.nanmean(lat_all) / dt)))}   "
      f"(measured {np.nanmean(lat_all):.2f}s / {dt}s per step)")
if rc_per_mps:
    r = float(np.mean(rc_per_mps))
    print(f"run_on_tello.py : RC_PER_MPS = {r:.0f}")
    print(f"tello_env.py : max_speed  -> pick a speed whose rc stays <= ~80, i.e. max_speed <= {80 / r:.2f} m/s")
