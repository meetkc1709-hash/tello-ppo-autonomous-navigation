"""
DepthRays - turn the Tello's front-camera image into distance "rays", like a simple laser scanner,
using a monocular depth model (Depth Anything V2, metric indoor version). Works for ANY object:
nothing has to be marked or coloured.

    pip install transformers pillow     (once, with internet; the model (~100 MB) downloads on first use)

The rays match the simulator's camera sensor (tello_env.py, sensor="camera"):
  n_rays rays spread across the camera's horizontal view, LEFT edge first,
  each = distance (m) to the nearest thing along that direction, capped at max_range.

Which pixels count (mode="band", default): every pixel whose 3-D point is between
floor_margin above the FLOOR and top_above m above the CAMERA. So any object counts no matter
how tall or short it is (a 20 cm box, a chair, a person), and the floor itself is ignored.
This needs the camera's height above the floor: set .cam_height (m) - run_on_tello.py updates it
from the drone's downward distance sensor. mode="strip" is the old thin strip at the centre row.
"""
import threading, time
import numpy as np

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


class DepthRays:
    def __init__(self, n_rays=9, hfov_deg=55.0, max_range=2.0, strip=0.10, model_id=DEFAULT_MODEL,
                 cy_frac=351.2 / 720.0, percentile=10, load_model=True, input_height=196,
                 mode="band", cam_height=0.8, floor_margin=0.15, top_above=1.0, pitch_deg=0.0,
                 min_frac=0.03):
        """input_height: image height fed to the depth model (multiple of 14). Smaller = faster.
        196 is ~5-7x faster than the model's default 518 and still fine for objects 0.3-2 m away."""
        self.n_rays, self.max_range, self.strip = n_rays, max_range, strip
        self.hfov = np.radians(hfov_deg)
        self.angles = np.radians(np.linspace(hfov_deg / 2, -hfov_deg / 2, n_rays))   # + = left
        self.cy_frac, self.pct = cy_frac, percentile
        self.mode, self.cam_height = mode, cam_height
        self.floor_margin, self.top_above = floor_margin, top_above
        self.pitch = np.radians(pitch_deg)      # + = camera tilted DOWN
        self.min_frac = min_frac                # an obstacle must fill this share of a column's pixels
        self.last_mask = None
        self.pipe = None
        self.model = None
        self.input_h = int(round(input_height / 14)) * 14
        if load_model:
            import torch
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading depth model {model_id} on {self.device.upper()} (input height {self.input_h}) ...")
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
                self.proc = AutoImageProcessor.from_pretrained(model_id)
                self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()
                if self.device == "cpu":
                    torch.set_num_threads(max(1, (__import__("os").cpu_count() or 2) - 1))
            except Exception as e:                       # fall back to the simple pipeline
                print("[depth] direct model load failed, using pipeline:", e)
                from transformers import pipeline
                self.pipe = pipeline("depth-estimation", model=model_id,
                                     device=0 if self.device == "cuda" else -1)

    def depth_map(self, frame_rgb):
        """Metric depth (m) for an RGB frame (numpy HxWx3)."""
        from PIL import Image
        if self.model is not None:
            h = self.input_h
            w = int(round(h * frame_rgb.shape[1] / frame_rgb.shape[0] / 14)) * 14
            img = Image.fromarray(frame_rgb).resize((w, h))
            inputs = self.proc(images=img, return_tensors="pt", do_resize=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                d = self.model(**inputs).predicted_depth
            return np.squeeze(d.cpu().numpy()).astype(np.float32)
        out = self.pipe(Image.fromarray(frame_rgb))
        d = out["predicted_depth"]
        d = d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d)
        return np.squeeze(d).astype(np.float32)

    def keep_mask(self, depth):
        """True for pixels that are a possible obstacle (not floor, not high up)."""
        H, W = depth.shape
        f = (W / 2) / np.tan(self.hfov / 2)               # focal length in pixels (square pixels)
        if self.mode == "strip":
            m = np.zeros_like(depth, dtype=bool)
            m[max(0, int((self.cy_frac - self.strip) * H)):min(H, int((self.cy_frac + self.strip) * H))] = True
            return m
        v = np.arange(H, dtype=np.float32)
        down = np.arctan((v - self.cy_frac * H) / f) + self.pitch      # angle below horizontal, per row
        below = depth * np.tan(down)[:, None]                          # m below the camera (approx.)
        return (below < self.cam_height - self.floor_margin) & (below > -self.top_above)

    def rays_from_depth(self, depth):
        """depth: HxW metres (distance along the camera's viewing axis). Returns n_rays distances."""
        H, W = depth.shape
        keep = self.keep_mask(depth)
        self.last_mask = keep
        f = (W / 2) / np.tan(self.hfov / 2)
        half = max(1, int(W / (2 * self.n_rays)))
        rays = np.full(self.n_rays, self.max_range, dtype=np.float32)
        for i, a in enumerate(self.angles):
            u = int(round(W / 2 - f * np.tan(a)))         # + angle (left) -> left side of the image
            c0, c1 = max(0, u - half), min(W, u + half + 1)
            if c1 <= c0:
                continue
            col, k = depth[:, c0:c1], keep[:, c0:c1]
            vals = col[k]
            if vals.size < self.min_frac * col.size:      # nothing (but floor/ceiling) in this direction
                continue
            # nearest-ish thing: a low percentile, but over the column's whole height so that a
            # small object (only a few rows tall) still shows up
            near = np.sort(vals)[: max(1, int(self.min_frac * col.size))]
            z = float(np.percentile(near, 50))
            rays[i] = z / np.cos(a)                        # along-axis depth -> distance along the ray
        return np.clip(rays, 0.0, self.max_range)

    def rays(self, frame_rgb):
        return self.rays_from_depth(self.depth_map(frame_rgb))


class ObstacleMemory:
    """Remembers what the camera has seen, in room coordinates (from the drone's own position).

    Why: the camera looks straight ahead, so a SHORT object disappears below the bottom of the
    picture as the drone gets close (at 80 cm flying height a 25 cm box is out of view from
    ~1.4 m in). Without memory the rays would say "free" right before the crash.
    Points are forgotten when the camera can clearly see past them from far enough away, or
    after max_age seconds."""

    def __init__(self, angles, max_range, point_radius=0.10, max_age=30.0, recheck_dist=1.5):
        self.angles, self.max_range = np.asarray(angles), max_range
        self.r, self.max_age, self.recheck = point_radius, max_age, recheck_dist
        self.spacing = abs(self.angles[0] - self.angles[1]) if len(self.angles) > 1 else 0.1
        self.pts = np.zeros((0, 2)); self.t = np.zeros(0)
        self._last_reading = None

    def update(self, pos, cam_rays, reading_time):
        """Add a NEW depth reading (call every control step; repeats of the same reading are skipped)."""
        now = time.time()
        if len(self.t):
            keep = now - self.t < self.max_age
            if reading_time != self._last_reading and len(self.pts):
                rel = self.pts[keep] - pos
                d = np.linalg.norm(rel, axis=1)
                a = np.arctan2(rel[:, 1], rel[:, 0])
                k2 = np.ones(len(d), bool)
                for i, (di, ai) in enumerate(zip(d, a)):
                    j = int(np.argmin(np.abs(self.angles - ai)))
                    in_view = abs(self.angles[j] - ai) < self.spacing / 2
                    if in_view and di > self.recheck and cam_rays[j] > di + 0.3:
                        k2[i] = False                       # camera sees through it now -> was a ghost
                idx = np.flatnonzero(keep)[~k2]
                keep[idx] = False
            self.pts, self.t = self.pts[keep], self.t[keep]
        if reading_time == self._last_reading:
            return
        self._last_reading = reading_time
        cam_rays = np.asarray(cam_rays)
        hit = cam_rays < 0.95 * self.max_range
        if hit.any():
            new = pos + cam_rays[hit, None] * np.stack([np.cos(self.angles[hit]), np.sin(self.angles[hit])], 1)
            self.pts = np.vstack([self.pts, new]); self.t = np.concatenate([self.t, np.full(len(new), now)])
            if len(self.pts) > 400:
                self.pts, self.t = self.pts[-400:], self.t[-400:]

    def rays(self, pos):
        """Distance along each camera ray to the nearest remembered point (max_range if none)."""
        out = np.full(len(self.angles), self.max_range, dtype=np.float32)
        if not len(self.pts):
            return out
        rel = self.pts - pos
        d = np.linalg.norm(rel, axis=1)
        a = np.arctan2(rel[:, 1], rel[:, 0])
        for i, ang in enumerate(self.angles):
            width = self.spacing / 2 + np.arctan2(self.r, np.maximum(d, 0.05))
            m = np.abs(np.angle(np.exp(1j * (a - ang)))) < width
            if m.any():
                out[i] = min(out[i], float(np.maximum(d[m] - self.r, 0.0).min()))
        return out


class DepthThread:
    """Runs DepthRays in the background on the newest camera frame, so the 10 Hz control loop
    never waits for the (slower) depth model. latest() returns (rays, age_seconds, depth_map)."""

    def __init__(self, sensor, get_frame):
        self.sensor, self.get_frame = sensor, get_frame
        self._rays, self._depth, self._t = None, None, 0.0
        self.fps = 0.0
        self._stop = False
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        while not self._stop:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.02); continue
            t0 = time.time()
            try:
                d = self.sensor.depth_map(frame)
                r = self.sensor.rays_from_depth(d)
            except Exception as e:                 # keep running; report once in a while
                print("[depth]", e); time.sleep(0.2); continue
            self._rays, self._depth, self._t = r, d, time.time()
            self.fps = 0.8 * self.fps + 0.2 / max(time.time() - t0, 1e-3)

    def latest(self):
        age = time.time() - self._t if self._rays is not None else float("inf")
        return self._rays, age, self._depth

    def stop(self):
        self._stop = True
