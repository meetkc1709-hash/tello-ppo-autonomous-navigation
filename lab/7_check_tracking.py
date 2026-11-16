"""
STEP 7 - Check that the wall markers give a correct position.

    python 7_check_tracking.py           # drone switched on, carried BY HAND (no flying)
    python 7_check_tracking.py --fly     # take off and hover 15 s, printing position

A window shows the video with detected markers and the position. Tests to do by hand:
  - hold the drone at a measured spot (tape measure!) facing the wall -> x, y should match
  - walk 1 m toward the wall  -> x should go UP by ~1.0
  - step 1 m to the LEFT      -> y should go UP by ~1.0
Write down the measured vs. printed positions. Press q to quit.
Every reading is saved to tracking_log.csv.
"""
import argparse, csv, time
import cv2
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
import _paths
from marker_tracker import MarkerTracker

ap = argparse.ArgumentParser()
ap.add_argument("--fly", action="store_true")
ap.add_argument("--seconds", type=float, default=15)
args = ap.parse_args()

trk = MarkerTracker()
tello = Tello()
tello.connect()
print("Battery:", tello.get_battery(), "%")
tello.streamon()
reader = tello.get_frame_read()
time.sleep(2)

rows, t0, last_print = [], time.time(), 0.0
try:
    if args.fly:
        tello.takeoff()
        t0 = time.time()
    while True:
        t = time.time() - t0
        if args.fly and t > args.seconds:
            break
        frame = reader.frame
        if frame is None:
            continue
        corners, ids = trk.detect(frame)
        pos = trk.locate_from_corners(corners, ids) if ids else None
        if args.fly:
            tello.send_rc_control(0, 0, 0, 0)
        rows.append([round(t, 2), len(ids)] + ([round(v, 3) for v in pos] if pos is not None else ["", "", ""]))
        if t - last_print > 0.25:
            last_print = t
            print(f"t={t:5.1f}s markers={ids}  " +
                  (f"x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f} m" if pos is not None else "NO MARKER"))
        bgr = trk.draw(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), corners, ids, pos)
        cv2.imshow("Tello tracking", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    if args.fly:
        tello.send_rc_control(0, 0, 0, 0)
        tello.land()
    cv2.destroyAllWindows()
    tello.streamoff()
    close_tello(tello)
    with open("tracking_log.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "n_markers", "x", "y", "z"]); w.writerows(rows)
    seen = sum(1 for r in rows if r[2] != "")
    print(f"Saved tracking_log.csv - position found in {seen}/{len(rows)} frames")
