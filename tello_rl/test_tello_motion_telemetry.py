import csv
import time
from pathlib import Path
from djitellopy import Tello

OUTPUT = Path("tello_motion_ground_test.csv")
DURATION_SECONDS = 10
SAMPLE_INTERVAL = 0.1

tello = Tello()
rows = []

try:
    print("Connecting to Tello...")
    tello.connect()

    print("Battery:", tello.get_battery(), "%")
    print("Temperature:", tello.get_temperature(), "C")

    print("\nStarting ground telemetry logging...")

    start = time.monotonic()

    while time.monotonic() - start < DURATION_SECONDS:
        elapsed = time.monotonic() - start

        row = {
            "time_s": round(elapsed, 3),
            "height_cm": tello.get_height(),
            "tof_cm": tello.get_distance_tof(),
            "vx_cm_s": tello.get_speed_x(),
            "vy_cm_s": tello.get_speed_y(),
            "vz_cm_s": tello.get_speed_z(),
            "ax": tello.get_acceleration_x(),
            "ay": tello.get_acceleration_y(),
            "az": tello.get_acceleration_z(),
            "yaw_deg": tello.get_yaw(),
            "pitch_deg": tello.get_pitch(),
            "roll_deg": tello.get_roll(),
            "battery_pct": tello.get_battery(),
        }

        rows.append(row)
        print(row)

        time.sleep(SAMPLE_INTERVAL)

finally:
    tello.end()

    if rows:
        with OUTPUT.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

        print("\nTelemetry saved to:", OUTPUT)
        print("Total samples:", len(rows))
