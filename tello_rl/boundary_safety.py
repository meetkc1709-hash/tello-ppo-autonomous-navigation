"""
Experimental boundary safety filter for PPO navigation.

Uses only controller-accessible observations and configured
environment parameters. Does not access true simulator position.

Observation layout:
    obs[2:4] = estimated velocity / max_speed
    obs[5:9] = estimated wall distances / room scale
"""

import numpy as np


class BoundarySafetyFilter:
    def __init__(
        self,
        room_size=(3.0, 3.0),
        max_speed=0.6,
        drone_radius=0.2,
        latency_s=0.6,
        response_time_s=0.35,
        safety_margin=0.12,
        correction_mode="reverse",
    ):
        self.room_size = np.asarray(room_size, dtype=np.float32)
        self.room_scale = float(np.max(self.room_size))

        self.max_speed = float(max_speed)
        self.drone_radius = float(drone_radius)
        self.latency_s = float(latency_s)
        self.response_time_s = float(response_time_s)
        self.safety_margin = float(safety_margin)

        if correction_mode not in ("reverse", "clamp", "adaptive"):
            raise ValueError(
                "correction_mode must be 'reverse', 'clamp', or 'adaptive'"
            )

        self.correction_mode = correction_mode

        self.interventions = 0
        self.total_actions = 0

    def reset(self):
        self.interventions = 0
        self.total_actions = 0

    def filter_action(self, obs, action):
        obs = np.asarray(obs, dtype=np.float32)
        original = np.asarray(action, dtype=np.float32).copy()
        safe = np.clip(original.copy(), -1.0, 1.0)

        estimated_velocity = obs[2:4] * self.max_speed
        wall_distances = obs[5:9] * self.room_scale

        # Walls are ordered:
        # left, right, bottom, top
        wall_pairs = [
            (0, 1),
            (2, 3),
        ]

        for axis, (negative_wall, positive_wall) in enumerate(wall_pairs):

            velocity = float(estimated_velocity[axis])
            proposed_action = float(safe[axis])

            # Evaluate each wall separately.
            for wall_index, direction in [
                (negative_wall, -1.0),
                (positive_wall, 1.0),
            ]:

                # Positive means movement toward this wall.
                toward_velocity = max(0.0, velocity * direction)
                toward_command = max(0.0, proposed_action * direction)

                commanded_speed = toward_command * self.max_speed

                # Conservative estimate of movement before and
                # during the braking response.
                approach_speed = max(
                    toward_velocity,
                    commanded_speed,
                )

                stopping_distance = (
                    approach_speed * self.latency_s
                    + toward_velocity * self.response_time_s
                )

                required_clearance = (
                    self.drone_radius
                    + self.safety_margin
                    + stopping_distance
                )

                measured_clearance = float(wall_distances[wall_index])

                if (
                    measured_clearance < required_clearance
                    and (toward_velocity > 0.01 or toward_command > 0.01)
                ):
                    # Command movement away from the wall.
                    # Delayed actions may still execute first.
                    if self.correction_mode == "reverse":
                        safe[axis] = -direction

                    elif self.correction_mode == "clamp":
                        # Suppress movement toward the wall.
                        if safe[axis] * direction > 0:
                            safe[axis] = 0.0

                    elif self.correction_mode == "adaptive":
                        # More urgent threshold based on estimated
                        # clearance and approach velocity.
                        critical_clearance = (
                            self.drone_radius
                            + self.safety_margin
                            + toward_velocity * self.latency_s
                        )

                        if (
                            measured_clearance < critical_clearance
                            and toward_velocity > 0.08
                        ):
                            # Critical risk: actively counter motion.
                            safe[axis] = -direction
                        elif safe[axis] * direction > 0:
                            # Moderate risk: suppress unsafe command.
                            safe[axis] = 0.0

                    break

        safe = np.clip(safe, -1.0, 1.0).astype(np.float32)

        intervened = not np.allclose(safe, original, atol=1e-6)

        self.total_actions += 1

        if intervened:
            self.interventions += 1

        return safe, intervened
