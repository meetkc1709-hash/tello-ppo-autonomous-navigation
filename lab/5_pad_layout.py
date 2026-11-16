"""
STEP 5 (pads) - Plan where to put the mission pads and write pad_layout.json. (No drone needed.)

    python 5_pad_layout.py --pads 1 2 3 4 --area 2.0 2.0

  --pads    : the pad numbers you have (the number printed on each pad), in any order
  --area    : flight area depth (toward the front, x) and width (y), in metres
  --pad_x_is / --pad_y_is : pad axis directions (from 2_check_position.py); defaults are a guess
  --vy_sign : from '3_calibrate.py --side'

Lays the pads out in an even grid, prints where each one goes, and writes pad_layout.json
next to pad_tracker.py. Place EVERY pad the SAME WAY ROUND (rocket pointing the same direction,
toward the front of the flight area). Then measure where you actually put them and fix the
numbers in pad_layout.json if they differ.
"""
import argparse, json, math, os
import _paths

ap = argparse.ArgumentParser()
ap.add_argument("--pads", type=int, nargs="+", required=True)
ap.add_argument("--area", type=float, nargs=2, required=True, metavar=("DEPTH_X", "WIDTH_Y"))
ap.add_argument("--pad_x_is", default="+x", choices=["+x", "-x", "+y", "-y"])
ap.add_argument("--pad_y_is", default="+y", choices=["+x", "-x", "+y", "-y"])
ap.add_argument("--vy_sign", type=float, default=-1.0)
ap.add_argument("--at", type=float, nargs="*", default=None,
                help="put the pads at these spots instead of a grid: x1 y1 x2 y2 ... (one pair per pad, same order)")
args = ap.parse_args()

n = len(args.pads)
X, Y = args.area
# choose a grid (rows along x, cols along y) whose cells are as square as possible
best = None
for rows in range(1, n + 1):
    cols = math.ceil(n / rows)
    cell_x, cell_y = X / rows, Y / cols
    score = max(cell_x, cell_y)
    if best is None or score < best[0]:
        best = (score, rows, cols)
_, rows, cols = best
pads = {}
k = 0
if args.at:
    if len(args.at) != 2 * n:
        raise SystemExit(f"--at needs {2*n} numbers (x y for each of the {n} pads)")
    for j, pid in enumerate(args.pads):
        pads[str(pid)] = [args.at[2 * j], args.at[2 * j + 1]]
    rows = cols = 1
    k = n
for r in range(rows if k < n else 0):
    for c in range(cols):
        if k >= n:
            break
        x = round((r + 0.5) * X / rows, 2)
        y = round((c + 0.5) * Y / cols, 2)
        pads[str(args.pads[k])] = [x, y]
        k += 1

layout = {"pads": pads,
          "axes": {"pad_x_is": args.pad_x_is, "pad_y_is": args.pad_y_is},
          "vy_sign": args.vy_sign,
          "_help": "pads: pad number -> [x, y] centre in metres. Origin = back-right floor corner, "
                   "x = forward, y = left. Measure real positions and edit if needed."}
path = os.path.join(_paths.ROOT, "pad_layout.json")
with open(path, "w") as f:
    json.dump(layout, f, indent=2)

gap = max(X / rows, Y / cols)
if not args.at:
    print(f"Grid: {rows} row(s) x {cols} column(s), about {gap:.2f} m between pads")
print("\nPad positions (origin = back-right corner, x = forward, y = to the LEFT):")
for pid, (x, y) in pads.items():
    print(f"  pad {pid}:  x = {x:.2f} m forward,  y = {y:.2f} m left")
print(f"\nWrote {path}")
print(f"Train / fly with:  --room {X} {Y}")
if gap > 1.2 and not args.at and n > 1:
    print("\nNOTE: pads are more than ~1.2 m apart - the drone relies on speed readings longer "
          "between pads. Use a smaller area or more pads if positions drift.")
