import csv
import math
import time
from pathlib import Path

OUTPUT = Path("simulated_motion_telemetry.csv")
SAMPLE_INTERVAL = 0.1

rows = []
events = []

def log_event(event):
    events.append({
        "time_s": round(time.monotonic() - start, 3),
        "event": event
    })

def record_sample(phase, elapsed):
    if phase == "hover":
        velocity = 0.0
    elif phase == "forward":
        velocity = 20.0
    else:
        velocity = 20.0 * math.exp(-2 * elapsed)

    rows.append({
        "time_s": round(time.monotonic() - start, 3),
        "phase": phase,
        "vx_cm_s": round(velocity, 3),
        "vy_cm_s": 0.0,
        "vz_cm_s": 0.0,
        "height_cm": 80,
        "yaw_deg": 0,
        "pitch_deg": 0,
        "roll_deg": 0
    })

def run_phase(phase, duration):
    phase_start = time.monotonic()
    log_event(f"{phase}_start")

    while time.monotonic() - phase_start < duration:
        elapsed = time.monotonic() - phase_start
        record_sample(phase, elapsed)
        time.sleep(SAMPLE_INTERVAL)

    log_event(f"{phase}_end")

start = time.monotonic()

run_phase("hover", 2)
run_phase("forward", 1.5)
run_phase("settling", 3)

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

with Path("simulated_motion_events.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=events[0].keys())
    writer.writeheader()
    writer.writerows(events)

print("Simulation completed.")
print("Telemetry samples:", len(rows))
print("Events recorded:", len(events))
print("Files saved:")
print("-", OUTPUT)
print("- simulated_motion_events.csv")
