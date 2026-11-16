"""
MarkerTracker - find the Tello's position from ArUco markers taped on the wall it faces.

World frame (same as the simulator):
    origin = floor corner at the BACK-RIGHT of the flight area (standing there, facing the marker wall)
    x = forward, toward the marker wall      y = to the left      z = up
The marker wall is the plane x = wall_x.

Files (kept next to this script):
    marker_layout.json  - marker size, wall distance, and where each marker is on the wall
                          (written by lab/5_make_markers.py, edit it after measuring)
    camera_calib.npz    - Tello camera intrinsics (written by lab/6_calibrate_camera.py)
"""
import json, os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUT_FILE = os.path.join(HERE, "marker_layout.json")
CALIB_FILE = os.path.join(HERE, "camera_calib.npz")

# Rough Tello camera intrinsics for 960x720 video (community-measured values).
# Only used if camera_calib.npz is missing - calibrate for real accuracy.
DEFAULT_K = np.array([[921.2, 0, 459.9], [0, 919.0, 351.2], [0, 0, 1]], dtype=np.float64)
DEFAULT_D = np.zeros(5)

# Expected marker->camera rotation when the drone faces the wall, level:
# marker x (= world -y) -> camera x ; marker y (= world z) -> camera -y ; marker z (= world -x) -> camera -z
R_EXPECTED = np.diag([1.0, -1.0, -1.0])
# camera axes expressed in world axes (camera x = -y, camera y = -z, camera z = +x)
R_WORLD_FROM_CAM_LEVEL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)


class MarkerTracker:
    def __init__(self, layout_file=LAYOUT_FILE, calib_file=CALIB_FILE, max_tilt_deg=40.0):
        with open(layout_file) as f:
            layout = json.load(f)
        self.size = float(layout["marker_size_m"])
        self.wall_x = float(layout["wall_x_m"])
        self.markers = {int(k): np.array([self.wall_x, v[0], v[1]], dtype=np.float64)
                        for k, v in layout["markers"].items()}      # id -> world centre (x, y, z)
        dict_name = layout.get("dictionary", "DICT_4X4_50")
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name)),
            cv2.aruco.DetectorParameters())
        if os.path.exists(calib_file):
            c = np.load(calib_file)
            self.K, self.D = c["K"], c["D"]
            self.calibrated = True
        else:
            print("[MarkerTracker] WARNING: camera_calib.npz not found - using rough default "
                  "Tello intrinsics. Run lab/6_calibrate_camera.py for accurate positions.")
            self.K, self.D = DEFAULT_K, DEFAULT_D
            self.calibrated = False
        s = self.size / 2
        # corner order returned by the ArUco detector: top-left, top-right, bottom-right, bottom-left
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)
        self.max_tilt = np.radians(max_tilt_deg)

    def _world_corners(self, i):
        """World coords of marker i's corners in detector order (TL, TR, BR, BL), seen facing the wall."""
        c, h = self.markers[i], self.size / 2
        # marker x = world -y, marker y = world +z
        return np.array([[c[0], c[1] + h, c[2] + h], [c[0], c[1] - h, c[2] + h],
                         [c[0], c[1] - h, c[2] - h], [c[0], c[1] + h, c[2] - h]], dtype=np.float64)

    def detect(self, frame):
        """Return (corners, ids) of all known markers in an RGB or gray frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return [], []
        keep = [(c, int(i)) for c, i in zip(corners, ids.ravel()) if int(i) in self.markers]
        return [c for c, _ in keep], [i for _, i in keep]

    def locate_from_corners(self, corners, ids):
        """World (x, y, z) of the camera from detected marker corners, or None.
        Uses ALL visible markers in one pose solve (much more accurate than one marker alone)."""
        if not ids:
            return None
        obj = np.vstack([self._world_corners(i) for i in ids])
        img = np.vstack([np.asarray(c, dtype=np.float64).reshape(4, 2) for c in corners])
        flag = cv2.SOLVEPNP_IPPE_SQUARE if len(ids) == 1 else cv2.SOLVEPNP_IPPE
        if len(ids) == 1:   # IPPE_SQUARE needs the marker-centred square model
            obj_used = self.obj.astype(np.float64)
        else:
            obj_used = obj
        try:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj_used, img, self.K, self.D, flags=flag)
        except cv2.error:
            return None
        R_exp = R_WORLD_FROM_CAM_LEVEL.T if len(ids) > 1 else R_EXPECTED
        best = None
        for rv, tv in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rv)
            angle = np.arccos(np.clip((np.trace(R_exp.T @ R) - 1) / 2, -1, 1))
            if best is None or angle < best[0]:
                best = (angle, rv, tv)
        if best is None or best[0] > self.max_tilt:
            return None
        _, rv, tv = best
        if len(ids) > 1:
            rv, tv = cv2.solvePnPRefineLM(obj_used, img, self.K, self.D, rv, tv)
            R, _ = cv2.Rodrigues(rv)
            return (-R.T @ tv).reshape(3)
        # Single marker: its estimated ROTATION is unreliable at a distance (a few degrees of error
        # becomes tens of cm sideways), but the direction/distance to it (tvec) is reliable.
        # The drone never turns, so assume the camera faces the wall and only use tvec.
        t = np.asarray(tv, dtype=np.float64).reshape(3)       # marker centre in camera frame
        offset_world = R_WORLD_FROM_CAM_LEVEL @ t              # camera -> marker, in world axes
        return self.markers[ids[0]] - offset_world

    def locate(self, frame):
        """World (x, y, z) in metres from one video frame, plus ids used. (None, []) if no marker."""
        corners, ids = self.detect(frame)
        if not ids:
            return None, []
        return self.locate_from_corners(corners, ids), ids

    def draw(self, frame_bgr, corners, ids, pos):
        if ids:
            cv2.aruco.drawDetectedMarkers(frame_bgr, [np.asarray(c, np.float32) for c in corners],
                                          np.array(ids).reshape(-1, 1))
        txt = "no marker" if pos is None else f"x={pos[0]:.2f}  y={pos[1]:.2f}  z={pos[2]:.2f} m"
        cv2.putText(frame_bgr, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame_bgr
