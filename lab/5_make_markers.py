"""
STEP 5 - Make the wall markers and the marker layout file. (No drone needed.)

    python 5_make_markers.py --wall_width 4.0 --wall_x 5.5

  --wall_width : how wide the flight area is along the marker wall (m)
  --wall_x     : distance from the BACK edge of the flight area to the marker wall (m)
  --spacing    : gap between marker centres (m). 0.5 keeps 2+ markers in view (most accurate)
  --height     : height of marker centres above the floor (m) - about the drone's flying height

Creates (in a "markers" folder):
  marker_00.png, marker_01.png ...  -> print each on its own sheet, 100% scale ("actual size")
  checkerboard.png                  -> print once, for 6_calibrate_camera.py
And writes marker_layout.json next to marker_tracker.py.

AFTER PRINTING: measure the side of the BLACK square of one marker (in m) and put it in
marker_layout.json as "marker_size_m". Tape markers flat on the wall, centres at the listed
positions (y measured from the RIGHT edge of the flight area when facing the wall).
"""
import argparse, json, os
import numpy as np
import cv2
import _paths

ap = argparse.ArgumentParser()
ap.add_argument("--wall_width", type=float, required=True)
ap.add_argument("--wall_x", type=float, required=True)
ap.add_argument("--spacing", type=float, default=0.5)
ap.add_argument("--height", type=float, default=0.9)
ap.add_argument("--size", type=float, default=0.16, help="expected printed black-square side (m)")
ap.add_argument("--dict", default="DICT_4X4_50",
                help="ArUco family: DICT_4X4_50 (best range) or e.g. DICT_6X6_50 for 6x6 markers")
ap.add_argument("--ids", type=int, nargs="*", default=None,
                help="use these marker IDs (left-to-right from the RIGHT edge) instead of 0,1,2,...")
args = ap.parse_args()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "markers")
os.makedirs(out, exist_ok=True)
d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))

n = int(np.floor((args.wall_width - 0.2) / args.spacing)) + 1
n = min(n, 50)
ids = args.ids if args.ids else list(range(n))
if len(ids) < n:
    raise SystemExit(f"Need {n} marker IDs for this wall, got {len(ids)}. Use a bigger --spacing or more IDs.")
ids = ids[:n]
ys = [round(0.1 + i * args.spacing + (args.wall_width - 0.2 - (n - 1) * args.spacing) / 2, 2) for i in range(n)]

# Letter/A4 at ~150 dpi: 1200 px wide. Black square 1000 px (~17 cm), white border, ID label.
for k, i in enumerate(ids):
    m = cv2.aruco.generateImageMarker(d, i, 1000)
    page = np.full((1400, 1200), 255, np.uint8)
    page[100:1100, 100:1100] = m
    cv2.putText(page, f"ID {i}   y = {ys[k]:.2f} m   centre height = {args.height:.2f} m",
                (100, 1250), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    cv2.imwrite(os.path.join(out, f"marker_{i:02d}.png"), page)

# checkerboard: 10 x 7 squares (9 x 6 inner corners)
sq = 120
board = np.full((7 * sq + 2 * sq, 10 * sq + 2 * sq), 255, np.uint8)
for r in range(7):
    for c in range(10):
        if (r + c) % 2 == 0:
            board[sq + r * sq: sq + (r + 1) * sq, sq + c * sq: sq + (c + 1) * sq] = 0
cv2.imwrite(os.path.join(out, "checkerboard.png"), board)

layout = {
    "dictionary": args.dict,
    "marker_size_m": args.size,
    "wall_x_m": args.wall_x,
    "markers": {str(i): [ys[k], args.height] for k, i in enumerate(ids)},
    "_help": "markers: id -> [y (m, from RIGHT edge of flight area, facing wall), centre height (m)]. "
             "marker_size_m = measured side of the printed BLACK square.",
}
path = os.path.join(_paths.ROOT, "marker_layout.json")
with open(path, "w") as f:
    json.dump(layout, f, indent=2)

print(f"Made {n} markers + checkerboard in {out}")
print(f"Wrote {path}")
print("\nTape positions (facing the wall, y measured from the RIGHT edge of the flight area):")
for k, i in enumerate(ids):
    print(f"  marker ID {i:2d}:  y = {ys[k]:.2f} m,  centre {args.height:.2f} m above floor")
depth = round(args.wall_x - 1.2, 1)
print(f"\nKeep the drone at least ~1.2 m from the wall (markers must fit in view).")
print(f"Train / fly with:  --room {depth} {args.wall_width}")
