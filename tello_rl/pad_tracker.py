"""
PadTracker - drone position from Tello EDU mission pads + the drone's own speed readings.

How it works:
  * Over a pad: the drone reports its offset from that pad (cm). pad centre + offset = exact position.
  * Between pads: position = last known position + speed x time (the drone's speed reading, dm/s).
  * Every time a pad is seen again, the position snaps back to the exact value (drift is reset).

World frame (same as the simulator): origin = BACK-RIGHT floor corner of the flight area,
x = forward (direction the drone faces at takeoff), y = to the left.

pad_layout.json (written by lab/5_pad_layout.py, kept next to this file):
  "pads":   { "1": [x, y], ... }   pad centres in metres
  "axes":   how the pad's own x/y map to world x/y - CHECK with lab/2_check_position.py:
            "pad_x_is": "+x" | "-x" | "+y" | "-y"   (world direction the pad's +x points)
            "pad_y_is": "+x" | "-x" | "+y" | "-y"
  (speed readings are converted with the takeoff heading - see speed_to_flight_frame)
"""
import json, os, time
import numpy as np


def speed_to_flight_frame(vgx, vgy, yaw0_deg):
    """Convert the Tello's raw speed readings (dm/s) to (forward, left) in m/s, in the frame the
    drone faced at takeoff.

    Measured in the lab (3 tests): the readings point along the direction the drone faced when
    it was SWITCHED ON, with the sign flipped, and the drone's 'yaw' says how far it has turned
    since then (clockwise positive). So: undo the sign, rotate by the takeoff heading.
    """
    ux, uy = -vgx / 10.0, -vgy / 10.0          # motion in the switch-on frame (x fwd, y right)
    c, s = np.cos(np.radians(yaw0_deg)), np.sin(np.radians(yaw0_deg))
    fwd = ux * c + uy * s
    right = -ux * s + uy * c
    return np.array([fwd, -right])

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUT_FILE = os.path.join(HERE, "pad_layout.json")
_DIR = {"+x": np.array([1.0, 0.0]), "-x": np.array([-1.0, 0.0]),
        "+y": np.array([0.0, 1.0]), "-y": np.array([0.0, -1.0])}


class PadTracker:
    def __init__(self, tello, layout_file=LAYOUT_FILE, start=None, yaw0=None, takeoff_pad_at=None):
        """takeoff_pad_at: if given, ANY pad seen is treated as the takeoff pad at this position
        (for --goal_ahead: the pad can be anywhere, no fixed layout needed)."""
        L = {}
        if os.path.exists(layout_file):
            with open(layout_file) as f:
                L = json.load(f)
        self.pads = {int(k): np.array(v, dtype=np.float64) for k, v in L.get("pads", {}).items()}
        self.any_pad_at = None if takeoff_pad_at is None else np.array(takeoff_pad_at, dtype=np.float64)
        ax = L.get("axes", {"pad_x_is": "+x", "pad_y_is": "+y"})
        self.ex, self.ey = _DIR[ax["pad_x_is"]], _DIR[ax["pad_y_is"]]
        self.tello = tello
        self.yaw0 = tello.get_yaw() if yaw0 is None else yaw0   # heading at takeoff
        self.pos = None if start is None else np.array(start, dtype=np.float64)
        self.t_last = None
        self.last_fix = None          # time of the last pad reading
        self.last_pad = -1

    def pad_reading(self):
        """World (x, y) from the pad currently under the drone, or None."""
        pid = self.tello.get_mission_pad_id()
        if self.any_pad_at is not None and pid > 0:
            centre = self.any_pad_at
        elif pid in self.pads:
            centre = self.pads[pid]
        else:
            return None, pid
        dx = self.tello.get_mission_pad_distance_x() / 100.0
        dy = self.tello.get_mission_pad_distance_y() / 100.0
        return centre + dx * self.ex + dy * self.ey, pid

    def update(self):
        """Call every control step. Returns (x, y) or None if position is still unknown."""
        now = time.time()
        # 1) dead reckoning from the drone's own speed (dm/s -> m/s)
        if self.pos is not None and self.t_last is not None:
            v = speed_to_flight_frame(self.tello.get_speed_x(), self.tello.get_speed_y(), self.yaw0)
            self.pos = self.pos + v * (now - self.t_last)
        self.t_last = now
        if self.last_fix is None and self.pos is not None:
            self.last_fix = now       # a given --start position counts as a known position
        # 2) exact fix whenever a pad is visible
        fix, pid = self.pad_reading()
        self.last_pad = pid
        if fix is not None:
            self.pos = fix
            self.last_fix = now
        return None if self.pos is None else self.pos.astype(np.float32)

    def seconds_since_fix(self):
        return float("inf") if self.last_fix is None else time.time() - self.last_fix
