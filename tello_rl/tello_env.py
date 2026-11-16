"""
TelloNavEnv - a lightweight Gymnasium environment for training a PPO agent
to fly a Tello from point A to point B as fast as possible.

Design idea (important for sim-to-real):
  The Tello already has its own onboard stabilizer. From Python you send
  *velocity-style* commands (send_rc_control). So we do NOT simulate rotor
  physics. We simulate "I ask for velocity v_cmd -> the drone reaches it
  after a short lag, with some noise". The policy learns at the same level
  of abstraction it will use on the real drone, which keeps the gap small.

Coordinate frame: x = forward, y = left (metres), fixed altitude (2D).
Start with 2D; add z later only if your demo needs it.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TelloNavEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        room_size=(4.0, 4.0),     # metres; match your real flight area
        dt=0.1,                   # 10 Hz control, realistic for Tello over Wi-Fi
        max_speed=0.8,            # m/s commanded at action = 1.0 (calibrate on real drone)
        tau=0.27,                 # velocity response time constant (s) - MEASURED in lab
        goal_radius=0.25,         # "arrived" if within this distance (m)
        max_steps=None,           # None = auto: 3x the time to fly the room diagonal
        n_static_obstacles=0,     # curriculum: 0 -> a few
        n_moving_obstacles=0,
        n_obs_in_state=3,         # how many nearest obstacles the agent "sees"
        obstacle_radius=(0.15, 0.30),
        drone_radius=0.10,
        pos_noise=0.02,           # observation noise (m) - mimics mission pad / tracker noise
        vel_noise=0.05,
        latency_steps=5,          # action delay in steps - MEASURED in lab (0.51 s)
        randomize=True,           # vary tau / latency / speed a bit every episode (sim-to-real robustness)
        sensor="perfect",         # "perfect": agent is told where obstacles are
                                  # "camera": agent only sees distance rays inside the front camera's view
        n_rays=9,                 # camera mode: number of distance rays across the view
        cam_hfov_deg=55.0,        # camera mode: Tello front camera horizontal field of view
        ray_range=2.0,            # camera mode: max distance the depth model is trusted (m)
        crash_penalty=50.0,       # big enough that cutting through obstacles never pays off
        safe_distance=0.4,        # (m) start a small penalty when closer than this to an obstacle
        cam_lag=False,            # camera mode: simulate the REAL depth camera's timing - a new reading
                                  # only every 2-4 steps (3-5 per second) and each reading 0.1-0.4 s old
        line_offset=0.1,          # how far (m) the "in the way" obstacle may sit beside the A->B line
        line_offset_min=0.0,      # ...and at least this far, so one side is clearly better to pass on
        seed=None,
    ):
        super().__init__()
        self.room = np.array(room_size, dtype=np.float32)
        self.dt, self.max_speed, self.tau = dt, max_speed, tau
        self.goal_radius = goal_radius
        if max_steps is None:
            diag = float(np.linalg.norm(self.room))
            max_steps = int(3 * diag / max_speed / dt)
        self.max_steps = max_steps
        self.n_static, self.n_moving = n_static_obstacles, n_moving_obstacles
        self.k = n_obs_in_state
        self.obs_r_range = obstacle_radius
        self.drone_r = drone_radius
        self.pos_noise, self.vel_noise = pos_noise, vel_noise
        self.latency = latency_steps
        self.nominal_tau, self.nominal_latency, self.nominal_speed = tau, latency_steps, max_speed
        self.randomize = randomize
        self.crash_penalty, self.safe_distance = crash_penalty, safe_distance
        self.cam_lag, self.line_offset, self.line_offset_min = cam_lag, line_offset, line_offset_min
        self.external_new = None      # real drone: True when external_rays holds a NEW camera reading

        # Action: desired (vx, vy) in [-1, 1], scaled by max_speed
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        # Observation:
        #   goal - pos (2), velocity (2), dist to goal (1),
        #   distances to 4 walls (4),
        #   for k nearest obstacles: rel_x, rel_y, surface_dist, rel_vx, rel_vy (5k)
        #   last N_HIST commands sent (2 each) - lets the agent account for the ~0.5 s delay
        self.n_hist = 6
        self.sensor = sensor
        self.n_rays, self.ray_range = n_rays, ray_range
        # ray directions in the drone frame, LEFT edge of the image first (+angle = left)
        self.ray_angles = np.radians(np.linspace(cam_hfov_deg / 2, -cam_hfov_deg / 2, n_rays))
        self.external_rays = None     # real drone: rays from the depth camera are put here
        if sensor == "camera":
            obstacle_dim = 2 * n_rays     # current rays + previous rays (shows motion)
        else:
            obstacle_dim = 5 * self.k
        obs_dim = 2 + 2 + 1 + 4 + obstacle_dim + 2 * self.n_hist
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.np_random, _ = gym.utils.seeding.np_random(seed)

    # ------------------------------------------------------------------ helpers
    def _rand_point(self, margin=0.4):
        return self.np_random.uniform([margin, margin], self.room - margin).astype(np.float32)

    def _spawn_obstacles(self):
        obs_pos, obs_vel, obs_r = [], [], []
        n_total = self.n_static + self.n_moving
        tries = 0
        # clearance kept around start and goal (smaller in small rooms so obstacles still fit)
        clear = 0.3 if self.room.min() < 3 else 0.5
        while len(obs_pos) < n_total and tries < 500:
            tries += 1
            r = self.np_random.uniform(*self.obs_r_range)
            if len(obs_pos) == 0 and tries <= 50 and self.np_random.random() < 0.6:
                # 60% of episodes: first obstacle sits ON the straight line A->B, so the agent
                # must learn to go AROUND things (otherwise random obstacles rarely get in the way)
                d = self.goal - self.pos
                perp = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
                p = (self.pos + d * self.np_random.uniform(0.4, 0.6)
                     + perp * self.np_random.choice([-1.0, 1.0])            # left or right of the line, 50/50
                     * self.np_random.uniform(self.line_offset_min, self.line_offset)).astype(np.float32)
            else:
                p = self._rand_point(margin=0.3)
            # keep start and goal clear
            if np.linalg.norm(p - self.pos) < r + clear or np.linalg.norm(p - self.goal) < r + clear:
                continue
            if any(np.linalg.norm(p - q) < r + rq + 0.1 for q, rq in zip(obs_pos, obs_r)):
                continue
            moving = len(obs_pos) >= self.n_static
            v = self.np_random.uniform(-0.3, 0.3, size=2) if moving else np.zeros(2)
            obs_pos.append(p); obs_vel.append(v); obs_r.append(r)
        self.obs_pos = np.array(obs_pos, dtype=np.float32).reshape(-1, 2)
        self.obs_vel = np.array(obs_vel, dtype=np.float32).reshape(-1, 2)
        self.obs_r = np.array(obs_r, dtype=np.float32)

    def _get_obs(self):
        p = self.pos + self.np_random.normal(0, self.pos_noise, 2)
        v = self.vel + self.np_random.normal(0, self.vel_noise, 2)
        scale = float(self.room.max())
        rel_goal = (self.goal - p) / scale
        dist = np.array([np.linalg.norm(self.goal - p) / scale])
        walls = np.array([p[0], self.room[0] - p[0], p[1], self.room[1] - p[1]]) / scale

        if self.sensor == "camera":
            if self.external_rays is not None:                 # real drone
                fresh = np.asarray(self.external_rays, dtype=np.float32)
                if self._cam_cur is None or self.external_new is None or self.external_new:
                    self._cam_prev = self._cam_cur if self._cam_cur is not None else fresh
                self._cam_cur = fresh
            else:                                              # simulator
                self._ray_hist.append(self._sim_rays(self.pos))
                self._ray_hist = self._ray_hist[-8:]
                if self._cam_cur is None or self.steps >= self._next_cam:
                    reading = self._ray_hist[max(-len(self._ray_hist), -1 - self._cam_delay)]
                    self._cam_prev = self._cam_cur if self._cam_cur is not None else reading
                    self._cam_cur = reading
                    self._next_cam = self.steps + self._cam_period
            rays = np.clip(self._cam_cur / self.ray_range, 0.0, 1.0)
            prev = np.clip(self._cam_prev / self.ray_range, 0.0, 1.0)   # the reading before (motion)
            hist = np.array(self.action_hist, dtype=np.float32).ravel()
            return np.concatenate([rel_goal, v / self.max_speed, dist, walls,
                                   rays, prev, hist]).astype(np.float32)

        obs_feats = np.zeros((self.k, 5), dtype=np.float32)
        obs_feats[:, 2] = 1.0  # "far away" default for empty slots
        if len(self.obs_pos):
            rel = self.obs_pos - p
            surf = np.linalg.norm(rel, axis=1) - self.obs_r - self.drone_r
            order = np.argsort(surf)[: self.k]
            for i, j in enumerate(order):
                obs_feats[i] = [rel[j, 0] / scale, rel[j, 1] / scale,
                                min(surf[j] / scale, 1.0),
                                (self.obs_vel[j, 0] - v[0]) / self.max_speed,
                                (self.obs_vel[j, 1] - v[1]) / self.max_speed]
        hist = np.array(self.action_hist, dtype=np.float32).ravel()
        return np.concatenate([rel_goal, v / self.max_speed, dist, walls,
                               obs_feats.ravel(), hist]).astype(np.float32)

    def _sim_rays(self, p):
        """What the depth camera would report: distance (m) along each ray to the nearest obstacle,
        ray_range if nothing. Only obstacles inside the camera's view are seen (it faces +x and
        never turns). Noise, missed detections and occasional false 'ghost' objects are added,
        like a real monocular depth model."""
        d = np.full(self.n_rays, self.ray_range, dtype=np.float64)
        dirs = np.stack([np.cos(self.ray_angles), np.sin(self.ray_angles)], axis=1)
        for c, r in zip(self.obs_pos, self.obs_r):
            f = p - c
            b = dirs @ f
            disc = b * b - (f @ f - r * r)
            hit = disc >= 0
            t = -b - np.sqrt(np.where(hit, disc, 0.0))
            t = np.where(hit & (t > -r), np.maximum(t, 0.0), np.inf)
            d = np.minimum(d, t)
        if self.randomize or self.sensor == "camera":
            hit = d < self.ray_range                                           # noise only on real hits:
            d = np.where(hit, d * (1 + self.np_random.normal(0, 0.10, self.n_rays)), d)  # ~10% error
            # (free directions stay "free" - otherwise the agent learns that 1.6-1.9 m readings are
            #  just noise and ignores obstacles until they are very close)
            miss = self.np_random.random(self.n_rays) < 0.05                    # missed detection
            d[miss] = self.ray_range
            if self.np_random.random() < 0.02:                                 # ghost object
                d[self.np_random.integers(self.n_rays)] = self.np_random.uniform(0.3, self.ray_range)
        return np.clip(d, 0.0, self.ray_range)

    def record_action(self, action):
        """Remember a command that was sent (also call this on the real drone)."""
        self.action_hist = self.action_hist[1:] + [np.asarray(action, dtype=np.float32)]

    # ---------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize:
            # +-30% reaction time, +-1 step delay, +-10% actual top speed around the measured values
            self.tau = self.nominal_tau * self.np_random.uniform(0.7, 1.3)
            self.latency = int(self.np_random.integers(max(0, self.nominal_latency - 1),
                                                       self.nominal_latency + 2))
            self.speed_gain = self.np_random.uniform(0.9, 1.1)
        else:
            self.tau, self.latency, self.speed_gain = self.nominal_tau, self.nominal_latency, 1.0
        # start and goal: far enough apart to be a real trip. Minimum distance scales with the
        # room (1.5 m in big rooms, ~60% of the usable diagonal in small ones like 2 x 2 m).
        margin = 0.3 if self.room.min() < 3 else 0.4
        usable_diag = float(np.linalg.norm(self.room - 2 * margin))
        min_dist = max(min(1.5, 0.6 * usable_diag), 2 * self.goal_radius)
        for _ in range(1000):
            self.pos = self._rand_point(margin)
            self.goal = self._rand_point(margin)
            if np.linalg.norm(self.goal - self.pos) >= min_dist:
                break
        self.vel = np.zeros(2, dtype=np.float32)
        self.action_queue = [np.zeros(2, dtype=np.float32)] * self.latency
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.action_hist = [np.zeros(2, dtype=np.float32)] * self.n_hist
        self.prev_rays = None
        self._cam_cur = self._cam_prev = None
        self._ray_hist = []
        self._next_cam = 0
        if self.cam_lag:
            self._cam_period = int(self.np_random.integers(2, 5))   # new reading every 0.2-0.4 s
            self._cam_delay = int(self.np_random.integers(1, 5))    # reading is 0.1-0.4 s old
        else:
            self._cam_period, self._cam_delay = 1, 0
        self._spawn_obstacles()
        if options:                         # fixed test layout (evaluate.py --sides)
            if "start" in options: self.pos = np.array(options["start"], np.float32)
            if "goal" in options: self.goal = np.array(options["goal"], np.float32)
            if "obstacles" in options:
                ob = np.array(options["obstacles"], np.float32).reshape(-1, 3)
                self.obs_pos, self.obs_r = ob[:, :2].copy(), ob[:, 2].copy()
                self.obs_vel = np.zeros_like(self.obs_pos)
        self.steps = 0
        self.prev_dist = float(np.linalg.norm(self.goal - self.pos))
        self.start_dist = self.prev_dist
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        # simulate Wi-Fi latency: the drone executes an older command
        self.action_queue.append(action)
        applied = self.action_queue.pop(0)
        self.record_action(action)

        # first-order velocity response (approximates Tello's onboard controller)
        v_cmd = applied * self.max_speed * self.speed_gain
        alpha = self.dt / (self.tau + self.dt)
        self.vel = self.vel + alpha * (v_cmd - self.vel)
        self.vel += self.np_random.normal(0, 0.02, 2)  # small disturbance (air, drift)
        self.pos = self.pos + self.vel * self.dt

        # move dynamic obstacles, bounce off walls
        if len(self.obs_pos):
            self.obs_pos += self.obs_vel * self.dt
            for d in range(2):
                hit = (self.obs_pos[:, d] < self.obs_r) | (self.obs_pos[:, d] > self.room[d] - self.obs_r)
                self.obs_vel[hit, d] *= -1

        self.steps += 1
        dist = float(np.linalg.norm(self.goal - self.pos))

        # ------------------------------------------------------ reward
        # 1) progress: reward getting closer (dense signal so PPO learns fast)
        progress = (self.prev_dist - dist)
        reward = 10.0 * progress
        # 2) time penalty: every step costs something -> "shortest time"
        reward -= 0.05
        # 3) smoothness: penalize jerky commands (helps on the real drone)
        reward -= 0.02 * float(np.sum((action - self.prev_action) ** 2))
        # 4) proximity: small, growing penalty when flying close to an obstacle
        #    (gives the agent an early warning signal instead of only learning from crashes)
        if len(self.obs_pos):
            surf = np.linalg.norm(self.obs_pos - self.pos, axis=1) - self.obs_r - self.drone_r
            closest = float(surf.min())
            if closest < self.safe_distance:
                reward -= 0.3 * (self.safe_distance - closest) / self.safe_distance
        self.prev_dist, self.prev_action = dist, action

        terminated, truncated, info = False, False, {}
        out_of_bounds = np.any(self.pos < self.drone_r) or np.any(self.pos > self.room - self.drone_r)
        collided = len(self.obs_pos) and np.any(
            np.linalg.norm(self.obs_pos - self.pos, axis=1) < self.obs_r + self.drone_r)

        if dist < self.goal_radius:
            reward += 10.0
            terminated = True
            info["is_success"] = True
        elif out_of_bounds or collided:
            reward -= self.crash_penalty
            terminated = True
            info["is_success"] = False
            info["crash"] = "wall" if out_of_bounds else "obstacle"
        elif self.steps >= self.max_steps:
            truncated = True
            info["is_success"] = False

        if terminated or truncated:
            info["flight_time"] = self.steps * self.dt
            # lower bound: straight line at max speed
            info["ideal_time"] = (self.start_dist - self.goal_radius) / self.max_speed
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print(f"t={self.steps*self.dt:4.1f}s pos={self.pos.round(2)} goal={self.goal.round(2)} "
              f"vel={self.vel.round(2)}")
