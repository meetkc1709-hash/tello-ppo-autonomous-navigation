"""
STEP 6 - Calibrate the Tello's camera (one time). Drone stays on the table - no flying.

    python 6_calibrate_camera.py

Print markers/checkerboard.png (from 5_make_markers.py) and tape it to something flat (a book).
A window shows the drone's video. Hold the board in front of the camera and slowly move it:
near/far, left/right/corners, tilted. The script grabs a picture automatically every second the
board is fully visible (green lines). After 25 pictures it calibrates and saves camera_calib.npz
next to marker_tracker.py. Press q to stop early (needs at least 10).
"""
import os, time
import numpy as np
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

PATTERN = (9, 6)       # inner corners of the 10 x 7 checkerboard
NEEDED = 25

objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)   # square size doesn't matter

tello = Tello()
tello.connect()
print("Battery:", tello.get_battery(), "%")
tello.streamon()
reader = tello.get_frame_read()
time.sleep(2)

obj_pts, img_pts, size, last = [], [], None, 0.0
try:
    while len(img_pts) < NEEDED:
        frame = reader.frame
        if frame is None:
            continue
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, PATTERN, cv2.CALIB_CB_FAST_CHECK)
        if found:
            cv2.drawChessboardCorners(bgr, PATTERN, corners, found)
            if time.time() - last > 1.0:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                obj_pts.append(objp); img_pts.append(corners); last = time.time()
                print(f"captured {len(img_pts)}/{NEEDED}")
        cv2.putText(bgr, f"{len(img_pts)}/{NEEDED}  (move the board around, q = finish)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Tello calibration", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cv2.destroyAllWindows()
    tello.streamoff()
    close_tello(tello)

if len(img_pts) < 10:
    raise SystemExit("Not enough pictures (need 10+). Run again.")
err, K, D, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
path = os.path.join(_paths.ROOT, "camera_calib.npz")
np.savez(path, K=K, D=D, image_size=np.array(size))
print(f"\nReprojection error: {err:.2f} px  (under ~0.5 is great, under 1.0 is fine)")
print("K =\n", K.round(1), "\nD =", D.ravel().round(3))
print("Saved", path)
