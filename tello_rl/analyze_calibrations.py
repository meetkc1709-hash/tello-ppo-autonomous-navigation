import csv
import numpy as np
from pathlib import Path

files = sorted(Path("../lab").glob("calib*.csv"))

print(
    f"{'File':28s} "
    f"{'RC':>4s} "
    f"{'Move(s)':>8s} "
    f"{'Mean(m/s)':>10s} "
    f"{'Peak(m/s)':>10s} "
    f"{'Path(cm)':>9s} "
    f"{'Last nonzero after stop(s)':>27s}"
)

print("-" * 110)

for path in files:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 2:
        continue

    t = np.array([float(r["t"]) for r in rows])
    rc = np.array([float(r["rc_cmd"]) for r in rows])

    sx = np.array([float(r["speed_x"]) for r in rows]) / 10.0
    sy = np.array([float(r["speed_y"]) for r in rows]) / 10.0

    speed = np.sqrt(sx**2 + sy**2)

    # Duration represented by each telemetry sample.
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0, None)

    moving = rc != 0

    rc_value = int(np.max(np.abs(rc)))
    move_duration = float(np.sum(dt[moving]))

    mean_speed = float(np.mean(speed[moving])) if moving.any() else 0.0
    peak_speed = float(np.max(speed))

    # Integrated speed magnitude: path length, not net displacement.
    path_cm = float(np.sum(speed * dt) * 100)

    # Find the first transition from movement to zero RC.
    stop_indices = np.where((rc[:-1] != 0) & (rc[1:] == 0))[0] + 1

    stop_delay = float("nan")

    if len(stop_indices):
        stop_idx = int(stop_indices[0])
        stop_time = t[stop_idx]

        # Only inspect telemetry after the first movement stop.
        subsequent_nonzero = np.where(
            (np.arange(len(rows)) >= stop_idx) & (speed > 0)
        )[0]

        if len(subsequent_nonzero):
            stop_delay = float(t[subsequent_nonzero[-1]] - stop_time)
        else:
            stop_delay = 0.0

    print(
        f"{path.name:28s} "
        f"{rc_value:4d} "
        f"{move_duration:8.2f} "
        f"{mean_speed:10.3f} "
        f"{peak_speed:10.3f} "
        f"{path_cm:9.1f} "
        f"{stop_delay:27.2f}"
    )
