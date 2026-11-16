"""
STEP 8 - See what the depth camera "sees": the 9 distance rays the agent will get.

    python 8_check_depth.py --source webcam    # laptop webcam, no drone needed (quick test)
    python 8_check_depth.py --source tello     # the drone's front camera (drone on, not flying)

First run needs internet: it downloads the depth model (~100 MB). Install once:
    python -m pip install transformers pillow

Window: left = camera with the 9 rays drawn at the bottom (red = close, green = free);
the parts tinted BLUE are ignored (floor, and anything over 1 m above the camera),
right = the depth map (bright = near). The terminal prints the ray distances in metres,
LEFT edge of the view first. Test: walk a box toward the camera - the middle rays should drop
to roughly the real distance (check with a tape: within ~20% is good). Press q to quit.
"""
import argparse, time
import numpy as np
import cv2
import _paths
from depth_sensor import DepthRays

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
ap.add_argument("--source", choices=["webcam", "tello"], default="webcam")
ap.add_argument("--depth_height", type=int, default=196, help="smaller = faster (140-280)")
ap.add_argument("--cam_height", type=float, default=0.8,
                help="camera height above the floor (m) - measure it when the drone sits on a chair")
ap.add_argument("--pitch", type=float, default=0.0, help="camera tilt down (deg)")
ap.add_argument("--mode", choices=["band", "strip"], default="band",
                help="band = any object, any height (new); strip = old thin strip")
ap.add_argument("--save", action="store_true",
                help="save a picture + depth + rays every second to lab/depth_check/<time>/ (for review)")
ap.add_argument("--seconds", type=float, default=0, help="stop automatically after this long (0 = until q)")
args = ap.parse_args()

if args.source == "tello":
    from djitellopy import Tello

    tello = Tello(); tello.connect(); print("Battery:", tello.get_battery(), "%")
    tello.streamon(); reader = tello.get_frame_read(); time.sleep(2)
    get = lambda: reader.frame                                   # RGB
else:
    cap = cv2.VideoCapture(0)
    def get():
        ok, f = cap.read()
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None

sensor = DepthRays(input_height=args.depth_height, cam_height=args.cam_height,
                   pitch_deg=args.pitch, mode=args.mode)
last = 0
last_save = 0
t_start = time.time()
save_dir = None
if args.save:
    import os, csv
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "depth_check", time.strftime("%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    ray_log = open(os.path.join(save_dir, "rays.csv"), "w", newline="")
    ray_csv = csv.writer(ray_log)
    ray_csv.writerow(["t", "fps"] + [f"ray{i}" for i in range(sensor.n_rays)] +
                     ["centre_depth_m", "strip_min_m", "strip_median_m", "image_min_m"])
    print("Saving snapshots to", save_dir)
try:
    while True:
        frame = get()
        if frame is None:
            continue
        t0 = time.time()
        depth = sensor.depth_map(frame)
        rays = sensor.rays_from_depth(depth)
        fps = 1 / max(time.time() - t0, 1e-3)

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]
        ign = cv2.resize((~sensor.last_mask).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        bgr[ign] = (0.55 * bgr[ign] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)  # blue = ignored (floor/high)
        for i, r in enumerate(rays):
            x = int((i + 0.5) * W / len(rays))
            frac = r / sensor.max_range
            col = (0, int(255 * frac), int(255 * (1 - frac)))
            cv2.rectangle(bgr, (x - 18, H - 10 - int(120 * (1 - frac))), (x + 18, H - 10), col, -1)
            cv2.putText(bgr, f"{r:.1f}", (x - 18, H - 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.putText(bgr, f"{fps:.1f} fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        dvis = cv2.resize(depth, (W, H))
        dvis = cv2.applyColorMap(np.uint8(255 * (1 - np.clip(dvis / 4.0, 0, 1))), cv2.COLORMAP_INFERNO)
        cv2.imshow("depth check (q = quit)", np.hstack([bgr, dvis]))
        if save_dir and time.time() - last_save > 1.0:
            last_save = time.time()
            Hd, Wd = depth.shape
            r0, r1 = int((sensor.cy_frac - 0.1) * Hd), int((sensor.cy_frac + 0.1) * Hd)
            ts = time.time() - t_start
            ray_csv.writerow([round(ts, 1), round(fps, 2)] + [round(float(r), 3) for r in rays] +
                             [round(float(depth[Hd // 2, Wd // 2]), 3), round(float(depth[r0:r1].min()), 3),
                              round(float(np.median(depth[r0:r1])), 3), round(float(depth.min()), 3)])
            ray_log.flush()
            cv2.imwrite(os.path.join(save_dir, f"view_{ts:05.1f}s.jpg"),
                        cv2.resize(np.hstack([bgr, dvis]), (960, 360)))
        if args.seconds and time.time() - t_start > args.seconds:
            break
        if time.time() - last > 0.5:
            last = time.time()
            print("rays (m, left -> right):", " ".join(f"{r:4.2f}" for r in rays), f"  [{fps:.1f} fps]")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cv2.destroyAllWindows()
    if args.source == "tello":
        close_tello(tello)
