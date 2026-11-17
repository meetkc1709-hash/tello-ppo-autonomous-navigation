import csv
from collections import defaultdict
from statistics import mean

TELEMETRY_FILE = "simulated_motion_telemetry.csv"
EVENT_FILE = "simulated_motion_events.csv"

with open(TELEMETRY_FILE, newline="") as f:
    telemetry = list(csv.DictReader(f))

with open(EVENT_FILE, newline="") as f:
    events = list(csv.DictReader(f))

phases = defaultdict(list)

for row in telemetry:
    phases[row["phase"]].append(row)

print("\n=== MOTION TELEMETRY ANALYSIS ===")

total_displacement = 0.0

for phase in ["hover", "forward", "settling"]:
    samples = phases[phase]

    velocities = [
        float(row["vx_cm_s"])
        for row in samples
    ]

    timestamps = [
        float(row["time_s"])
        for row in samples
    ]

    average_velocity = mean(velocities)

    displacement = 0.0

    for i in range(len(samples) - 1):
        dt = timestamps[i + 1] - timestamps[i]

        displacement += (
            velocities[i] + velocities[i + 1]
        ) / 2 * dt

    total_displacement += displacement

    print(f"\nPhase: {phase}")
    print(f"Samples: {len(samples)}")
    print(f"Average velocity: {average_velocity:.2f} cm/s")
    print(f"Estimated displacement: {displacement:.2f} cm")

print("\nTotal estimated displacement:",
      round(total_displacement, 2), "cm")

print("\n=== EVENT TIMESTAMPS ===")

for event in events:
    print(
        f"{event['time_s']} s -> {event['event']}"
    )

print("\nAnalysis completed.")
