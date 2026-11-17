import csv
from collections import defaultdict

TELEMETRY_FILE = "simulated_motion_telemetry.csv"
EVENT_FILE = "simulated_motion_events.csv"

with open(TELEMETRY_FILE, newline="") as f:
    telemetry = list(csv.DictReader(f))

with open(EVENT_FILE, newline="") as f:
    events = list(csv.DictReader(f))

samples = sorted(
    telemetry,
    key=lambda row: float(row["time_s"])
)

event_times = {
    row["event"]: float(row["time_s"])
    for row in events
}

phase_displacement = defaultdict(float)
phase_samples = defaultdict(list)

for row in samples:
    phase_samples[row["phase"]].append(
        float(row["vx_cm_s"])
    )

# Integrate across the complete telemetry timeline.
for i in range(len(samples) - 1):
    current = samples[i]
    following = samples[i + 1]

    t0 = float(current["time_s"])
    t1 = float(following["time_s"])

    v0 = float(current["vx_cm_s"])
    v1 = float(following["vx_cm_s"])

    dt = t1 - t0

    if dt <= 0:
        raise ValueError("Invalid telemetry timestamps.")

    displacement = 0.5 * (v0 + v1) * dt

    # Attribute each interval to its starting phase.
    phase_displacement[current["phase"]] += displacement

print("\n=== MOTION ANALYSIS V2 ===")

total_displacement = 0.0

for phase in ["hover", "forward", "settling"]:
    velocities = phase_samples[phase]

    distance = phase_displacement[phase]
    total_displacement += distance

    average_velocity = sum(velocities) / len(velocities)

    print(f"\nPhase: {phase}")
    print(f"Samples: {len(velocities)}")
    print(f"Average velocity: {average_velocity:.2f} cm/s")
    print(f"Estimated displacement: {distance:.2f} cm")

print(f"\nTotal estimated displacement: {total_displacement:.2f} cm")

print("\n=== EVENT DURATIONS ===")

for phase in ["hover", "forward", "settling"]:
    start = event_times[f"{phase}_start"]
    end = event_times[f"{phase}_end"]

    print(f"{phase}: {end - start:.3f} seconds")

print("\nAnalysis completed.")
