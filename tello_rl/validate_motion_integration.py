import csv
import math

with open("simulated_motion_events.csv", newline="") as f:
    events = {
        row["event"]: float(row["time_s"])
        for row in csv.DictReader(f)
    }

hover_duration = (
    events["hover_end"] - events["hover_start"]
)

forward_duration = (
    events["forward_end"] - events["forward_start"]
)

settling_duration = (
    events["settling_end"] - events["settling_start"]
)

# Analytical displacement for the synthetic motion model.
hover_distance = 0.0

# Constant forward velocity: v = 20 cm/s.
forward_distance = 20.0 * forward_duration

# Exponential settling velocity: v(t) = 20 * exp(-2t).
settling_distance = 10.0 * (
    1.0 - math.exp(-2.0 * settling_duration)
)

total_distance = (
    hover_distance
    + forward_distance
    + settling_distance
)

print("\n=== SYNTHETIC MOTION VALIDATION ===")

print(f"Hover duration: {hover_duration:.3f} s")
print(f"Forward duration: {forward_duration:.3f} s")
print(f"Settling duration: {settling_duration:.3f} s")

print(f"\nHover displacement: {hover_distance:.2f} cm")
print(f"Forward displacement: {forward_distance:.2f} cm")
print(f"Settling displacement: {settling_distance:.2f} cm")

print(f"\nTotal displacement: {total_distance:.2f} cm")

assert abs(hover_distance) < 1e-9
assert abs(forward_distance - 30.0) < 0.1
assert abs(settling_distance - 10.0) < 0.1

print("\nValidation passed.")
