import csv
import time
from pathlib import Path
from djitellopy import Tello

OUTPUT = Path("tello_hover_telemetry.csv")
HOVER_SECONDS = 5

tello = Tello()
takeoff_attempted = False
rows = []

try:
    print("Connecting...")
    tello.connect()

    battery = tello.get_battery()
    print(f"Battery: {battery}%")
    print(f"Temperature: {tello.get_temperature()} C")

    if battery < 50:
        raise RuntimeError("Battery too low.")

    confirmation = input(
        "Confirm clear area and no temperature warnings. "
        "Type TAKEOFF: "
    )

    if confirmation != "TAKEOFF":
        raise SystemExit("Cancelled.")

    takeoff_attempted = True
    tello.takeoff()

    start = time.monotonic()

    while time.monotonic() - start < HOVER_SECONDS:
        elapsed = time.monotonic() - start

        row = {
            "time_s": round(elapsed, 3),
            "height_cm": tello.get_height(),
            "velocity_x_cm_s": tello.get_speed_x(),
            "velocity_y_cm_s": tello.get_speed_y(),
            "velocity_z_cm_s": tello.get_speed_z(),
            "battery": tello.get_battery(),
        }

        rows.append(row)
        print(row)

        time.sleep(0.2)

    print("Landing...")
    tello.land()
    takeoff_attempted = False

except KeyboardInterrupt:
    print("Interrupted. Attempting landing.")

except Exception as e:
    print("Error:", e)

finally:
    if takeoff_attempted:
        try:
            tello.land()
        except Exception as e:
            print("Landing command failed:", e)
            print("Use independent manual landing control.")

    tello.end()

    if rows:
        with OUTPUT.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"Telemetry saved to {OUTPUT}")
