import gymnasium as gym
import numpy as np


class RobustObservationWrapper(gym.Wrapper):
    """
    Simulate accumulated localization error.

    The environment keeps its true physical state.
    The PPO agent receives an estimated state.

    Camera observations and action history are preserved.
    """

    def __init__(
        self,
        env,
        min_scale=0.85,
        max_scale=1.05,
        drift_std=0.002,
    ):
        super().__init__(env)

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.drift_std = drift_std

        self.position_scale = 1.0
        self.estimated_position = None
        self.estimated_velocity = None
        self.previous_true_position = None

        self.rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(
            seed=seed,
            options=options
        )

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.position_scale = self.rng.uniform(
            self.min_scale,
            self.max_scale
        )

        self.previous_true_position = self.env.unwrapped.pos.copy()

        # Assume that the initial position is known.
        self.estimated_position = self.previous_true_position.copy()
        self.estimated_velocity = np.zeros(
            2, dtype=np.float32
        )

        return self._modify_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        actual_position = self.env.unwrapped.pos.copy()

        actual_displacement = (
            actual_position - self.previous_true_position
        )

        # Systematic displacement-estimation error.
        estimated_displacement = (
            self.position_scale * actual_displacement
        )

        # Small accumulated random error.
        estimated_displacement += self.rng.normal(
            0.0,
            self.drift_std,
            size=2
        )

        self.estimated_position = (
            self.estimated_position + estimated_displacement
        ).astype(np.float32)

        dt = self.env.unwrapped.dt

        self.estimated_velocity = (
            estimated_displacement / dt
        ).astype(np.float32)

        self.previous_true_position = actual_position

        return (
            self._modify_observation(obs),
            reward,
            terminated,
            truncated,
            info,
        )

    def _modify_observation(self, obs):
        obs = obs.copy()
        env = self.env.unwrapped

        p = self.estimated_position
        v = self.estimated_velocity

        scale = float(env.room.max())
        goal_error = env.goal - p

        # Relative goal position.
        obs[0:2] = goal_error / scale

        # Estimated velocity.
        obs[2:4] = v / env.max_speed

        # Estimated distance to goal.
        obs[4] = np.linalg.norm(goal_error) / scale

        # Estimated distances to room boundaries.
        obs[5:9] = np.array([
            p[0],
            env.room[0] - p[0],
            p[1],
            env.room[1] - p[1],
        ]) / scale

        return obs.astype(np.float32)
