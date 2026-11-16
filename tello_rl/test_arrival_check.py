import numpy as np


class ArrivalCheck:
    def __init__(
        self,
        goal_radius=0.25,
        exit_radius=0.35,
        max_speed=0.05,
        settle_time=1.0,
    ):
        self.goal_radius = goal_radius
        self.exit_radius = exit_radius
        self.max_speed = max_speed
        self.settle_time = settle_time

        self.state = "NAVIGATING"
        self.stable_since = None

    def update(self, distance, speed, now):
        if self.state == "NAVIGATING":
            if distance < self.goal_radius:
                self.state = "SETTLING"
                self.stable_since = None

        if self.state == "SETTLING":
            if distance > self.exit_radius:
                self.state = "NAVIGATING"
                self.stable_since = None

            elif speed <= self.max_speed:
                if self.stable_since is None:
                    self.stable_since = now

                elif now - self.stable_since >= self.settle_time:
                    self.state = "ARRIVED"

            else:
                self.stable_since = None

        return self.state


def simulate(name, readings):
    checker = ArrivalCheck()

    print(f"\n{name}")
    print("Time    Distance    Speed    State")

    for t, distance, speed in readings:
        state = checker.update(distance, speed, t)

        print(
            f"{t:4.1f}s   "
            f"{distance:6.2f}m     "
            f"{speed:5.2f}m/s   "
            f"{state}"
        )


simulate(
    "Test 1: Stable arrival",
    [
        (0.0, 0.40, 0.20),
        (0.5, 0.23, 0.10),
        (1.0, 0.22, 0.03),
        (1.5, 0.21, 0.02),
        (2.0, 0.21, 0.02),
    ]
)

simulate(
    "Test 2: Moving too fast",
    [
        (0.0, 0.40, 0.20),
        (0.5, 0.20, 0.25),
        (1.0, 0.18, 0.18),
        (1.5, 0.20, 0.12),
        (2.0, 0.21, 0.08),
    ]
)

simulate(
    "Test 3: Drifts away",
    [
        (0.0, 0.40, 0.20),
        (0.5, 0.23, 0.04),
        (1.0, 0.28, 0.04),
        (1.5, 0.38, 0.08),
        (2.0, 0.42, 0.05),
    ]
)
